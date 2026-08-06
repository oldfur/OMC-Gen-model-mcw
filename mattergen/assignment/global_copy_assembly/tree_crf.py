"""Differentiable exact sum/max-product inference on a molecular tree."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .tree_builder import MolecularTree


@dataclass
class MAPResult:
    state_indices: dict[int, int]
    score: torch.Tensor


class TreeCRF:
    """Tree CRF over role permutation states, rooted at fixed identity anchor."""
    def __init__(self, tree: MolecularTree, *, num_states: int, identity_state: int):
        if num_states < 1 or not 0 <= identity_state < num_states:
            raise ValueError("invalid state-space or identity state")
        self.tree,self.num_states,self.identity_state=tree,num_states,identity_state

    def _validate(self, factors: dict[tuple[int,int],torch.Tensor]) -> None:
        if set(factors) != set(self.tree.tree_edges):
            raise ValueError("factors must contain exactly the oriented spanning-tree edges")
        for edge,value in factors.items():
            if value.shape != (self.num_states,self.num_states):
                raise ValueError(f"factor {edge} must have shape [{self.num_states},{self.num_states}]")
            if not torch.isfinite(value).all(): raise FloatingPointError(f"factor {edge} contains NaN or Inf")

    def _child_total(self, node: int, messages: dict[int,torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        terms=[messages[child] for child in self.tree.children[node]]
        return sum(terms) if terms else reference.new_zeros(self.num_states)

    def log_partition(self, factors: dict[tuple[int,int],torch.Tensor]) -> torch.Tensor:
        self._validate(factors); messages: dict[int,torch.Tensor]={}; reference=next(iter(factors.values()))
        for child in self.tree.postorder:
            parent=self.tree.parent[child]
            if parent is None: continue
            child_total=self._child_total(child,messages,reference)
            messages[child]=torch.logsumexp(factors[(parent,child)]+child_total[None,:],dim=1)
        root_total=self._child_total(self.tree.root,messages,reference)
        return root_total[self.identity_state]

    def target_score(self, factors: dict[tuple[int,int],torch.Tensor], target_states: dict[int,int]) -> torch.Tensor:
        self._validate(factors)
        if set(target_states)!=set(self.tree.preorder) or target_states[self.tree.root]!=self.identity_state:
            raise ValueError("target states must include every role and fix root to identity")
        if any(not isinstance(state,int) or not 0 <= state < self.num_states for state in target_states.values()):
            raise ValueError("target state index is out of range")
        return sum((factors[edge][target_states[edge[0]],target_states[edge[1]]] for edge in self.tree.tree_edges), start=next(iter(factors.values())).new_zeros(()))

    def nll(self, factors: dict[tuple[int,int],torch.Tensor], target_states: dict[int,int]) -> torch.Tensor:
        return (self.log_partition(factors)-self.target_score(factors,target_states))/max(1,len(self.tree.tree_edges))

    def map_decode(self, factors: dict[tuple[int,int],torch.Tensor]) -> MAPResult:
        self._validate(factors); messages: dict[int,torch.Tensor]={}; pointers: dict[int,torch.Tensor]={}; reference=next(iter(factors.values()))
        for child in self.tree.postorder:
            parent=self.tree.parent[child]
            if parent is None: continue
            values=factors[(parent,child)]+self._child_total(child,messages,reference)[None,:]
            messages[child],pointers[child]=values.max(dim=1)
        states={self.tree.root:self.identity_state}
        for parent in self.tree.preorder:
            for child in self.tree.children[parent]: states[child]=int(pointers[child][states[parent]].item())
        return MAPResult(state_indices=states,score=self.target_score(factors,states))

    def marginals(self, factors: dict[tuple[int,int],torch.Tensor]) -> tuple[dict[int,torch.Tensor],dict[tuple[int,int],torch.Tensor]]:
        """Optional exact node/edge marginals; all operations retain gradients."""
        self._validate(factors); up: dict[int,torch.Tensor]={}; reference=next(iter(factors.values()))
        for child in self.tree.postorder:
            parent=self.tree.parent[child]
            if parent is not None: up[child]=torch.logsumexp(factors[(parent,child)]+self._child_total(child,up,reference)[None,:],dim=1)
        zero=next(iter(factors.values())).new_full((self.num_states,),float("-inf")); down={self.tree.root:zero.clone()};down[self.tree.root][self.identity_state]=0.
        edge_log={}
        for parent in self.tree.preorder:
            for child in self.tree.children[parent]:
                siblings=sum((up[s] for s in self.tree.children[parent] if s!=child),start=zero.new_zeros(self.num_states))
                context=down[parent]+siblings
                joint=context[:,None]+factors[(parent,child)]+self._child_total(child,up,reference)[None,:]
                edge_log[(parent,child)]=joint-self.log_partition(factors)
                down[child]=torch.logsumexp(context[:,None]+factors[(parent,child)],dim=0)
        node={node:torch.softmax(down[node]+self._child_total(node,up,reference)-self.log_partition(factors),dim=0) for node in self.tree.preorder}
        return node,{edge:torch.exp(value) for edge,value in edge_log.items()}
