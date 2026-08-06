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

    def _normalise_factors(self, factors: dict[tuple[int,int],torch.Tensor]) -> dict[tuple[int,int],torch.Tensor]:
        """Remove one constant per tree edge without changing CRF probabilities.

        A constant shift of an entire edge factor shifts every complete-tree
        score by the same amount.  Removing each factor's detached maximum
        therefore leaves MAP states, marginals, and NLL unchanged while
        avoiding a catastrophic ``logZ - target_score`` subtraction after
        long optimization runs.
        """
        self._validate(factors)
        return {edge:value-value.detach().amax() for edge,value in factors.items()}

    def _child_total(self, node: int, messages: dict[int,torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        terms=[messages[child] for child in self.tree.children[node]]
        return sum(terms) if terms else reference.new_zeros(self.num_states)

    def _log_partition(self, factors: dict[tuple[int,int],torch.Tensor]) -> torch.Tensor:
        messages: dict[int,torch.Tensor]={}; reference=next(iter(factors.values()))
        for child in self.tree.postorder:
            parent=self.tree.parent[child]
            if parent is None: continue
            child_total=self._child_total(child,messages,reference)
            messages[child]=torch.logsumexp(factors[(parent,child)]+child_total[None,:],dim=1)
        root_total=self._child_total(self.tree.root,messages,reference)
        return root_total[self.identity_state]

    def _validate_target_states(self, target_states: dict[int,int]) -> None:
        if set(target_states)!=set(self.tree.preorder) or target_states[self.tree.root]!=self.identity_state:
            raise ValueError("target states must include every role and fix root to identity")
        if any(not isinstance(state,int) or not 0 <= state < self.num_states for state in target_states.values()):
            raise ValueError("target state index is out of range")

    def _target_score(self, factors: dict[tuple[int,int],torch.Tensor], target_states: dict[int,int]) -> torch.Tensor:
        return sum((factors[edge][target_states[edge[0]],target_states[edge[1]]] for edge in self.tree.tree_edges), start=next(iter(factors.values())).new_zeros(()))

    def log_partition(self, factors: dict[tuple[int,int],torch.Tensor]) -> torch.Tensor:
        self._validate(factors)
        return self._log_partition(factors)

    def target_score(self, factors: dict[tuple[int,int],torch.Tensor], target_states: dict[int,int]) -> torch.Tensor:
        self._validate_target_states(target_states)
        self._validate(factors)
        return self._target_score(factors,target_states)

    def log_partition_and_target_score(self, factors: dict[tuple[int,int],torch.Tensor], target_states: dict[int,int]) -> tuple[torch.Tensor,torch.Tensor]:
        """Return numerically comparable log-partition and target score."""
        self._validate_target_states(target_states)
        normalised=self._normalise_factors(factors)
        return self._log_partition(normalised),self._target_score(normalised,target_states)

    def nll(self, factors: dict[tuple[int,int],torch.Tensor], target_states: dict[int,int]) -> torch.Tensor:
        log_partition,target_score=self.log_partition_and_target_score(factors,target_states)
        return (log_partition-target_score)/max(1,len(self.tree.tree_edges))

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
        return MAPResult(state_indices=states,score=self._target_score(factors,states))

    def marginals(self, factors: dict[tuple[int,int],torch.Tensor]) -> tuple[dict[int,torch.Tensor],dict[tuple[int,int],torch.Tensor]]:
        """Optional exact node/edge marginals; all operations retain gradients."""
        self._validate(factors); up: dict[int,torch.Tensor]={}; reference=next(iter(factors.values()))
        for child in self.tree.postorder:
            parent=self.tree.parent[child]
            if parent is not None: up[child]=torch.logsumexp(factors[(parent,child)]+self._child_total(child,up,reference)[None,:],dim=1)
        zero=next(iter(factors.values())).new_full((self.num_states,),float("-inf")); down={self.tree.root:zero.clone()};down[self.tree.root][self.identity_state]=0.
        log_partition=self._log_partition(factors); edge_log={}
        for parent in self.tree.preorder:
            for child in self.tree.children[parent]:
                siblings=sum((up[s] for s in self.tree.children[parent] if s!=child),start=zero.new_zeros(self.num_states))
                context=down[parent]+siblings
                joint=context[:,None]+factors[(parent,child)]+self._child_total(child,up,reference)[None,:]
                edge_log[(parent,child)]=joint-log_partition
                down[child]=torch.logsumexp(context[:,None]+factors[(parent,child)],dim=0)
        node={node:torch.softmax(down[node]+self._child_total(node,up,reference)-log_partition,dim=0) for node in self.tree.preorder}
        return node,{edge:torch.exp(value) for edge,value in edge_log.items()}
