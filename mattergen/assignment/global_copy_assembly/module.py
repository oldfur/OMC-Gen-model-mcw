"""Standalone clean-geometry / oracle-R global structured copy assembly."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from mattergen.common.role_partition_diffusion.molecule_encoder import MolecularGraphEncoder
from mattergen.common.role_partition_diffusion.oracle_partition import ContextCrystalEncoder
from .pair_potential import BondPairPotential, permutation_factor
from .permutations import enumerate_permutations, identity_index, inverse_permutations
from .targets import AssemblyTarget, target_state_indices
from .tree_builder import MolecularTree, build_bfs_tree, select_anchor_role
from .tree_crf import TreeCRF
from .decoder import decode_copy_assembly


@dataclass(frozen=True)
class GlobalCopyAssemblyConfig:
    enabled: bool=False
    mode: str="clean_geometry_oracle_r"
    parameterization: str="role_permutation_tree_crf"
    anchor_policy: str="highest_degree_singleton"
    anchor_role: int|None=None
    tree_policy: str="bfs_from_anchor"
    crystal_hidden_dim: int=256
    crystal_num_layers: int=4
    molecular_hidden_dim: int=256
    pair_hidden_dim: int=256
    distance_rbf_dim: int=32
    tree_nll_weight: float=1.0
    pair_ce_weight: float=0.1
    use_oracle_role_assignment: bool=True
    use_oracle_copy_relation: bool=False
    use_copy_id_as_input: bool=False
    strict_finite_checks: bool=True
    strict_shape_checks: bool=True


class GlobalStructuredCopyAssembly(nn.Module):
    """Global tree-CRF copy assembly. Predictor forward has no copy-ID input."""
    def __init__(self, config: GlobalCopyAssemblyConfig=GlobalCopyAssemblyConfig()):
        super().__init__()
        if config.mode!="clean_geometry_oracle_r" or config.parameterization!="role_permutation_tree_crf": raise ValueError("only clean_geometry_oracle_r / role_permutation_tree_crf is implemented")
        if not config.use_oracle_role_assignment or config.use_oracle_copy_relation or config.use_copy_id_as_input: raise ValueError("this MVP requires oracle R only and forbids copy supervision as model input")
        self.config=config
        self.crystal_encoder=ContextCrystalEncoder(hidden=config.crystal_hidden_dim,layers=config.crystal_num_layers,rbf_dim=64)
        self.molecule_encoder=MolecularGraphEncoder(hidden=config.molecular_hidden_dim,layers=config.crystal_num_layers)
        if config.crystal_hidden_dim!=config.molecular_hidden_dim: raise ValueError("MVP requires matching crystal and molecular hidden dimensions")
        self.pair_potential=BondPairPotential(hidden=config.crystal_hidden_dim,pair_hidden=config.pair_hidden_dim,rbf_dim=config.distance_rbf_dim)

    @staticmethod
    def _bond_type(edge_index: torch.Tensor,bond_type: torch.Tensor,left:int,right:int)->int:
        matches=((edge_index[0]==left)&(edge_index[1]==right))|((edge_index[0]==right)&(edge_index[1]==left))
        found=bond_type[matches]
        if len(found)!=1: raise ValueError(f"tree edge ({left},{right}) has no unique molecular bond type")
        return int(found.item())

    def select_tree(self, role_orbits:list[list[int]], edge_index:torch.Tensor, *, M:int)->MolecularTree:
        if self.config.anchor_policy!="highest_degree_singleton" or self.config.tree_policy!="bfs_from_anchor": raise ValueError("unsupported anchor/tree policy")
        anchor=select_anchor_role(role_orbits,edge_index,explicit_anchor=self.config.anchor_role)
        return build_bfs_tree(edge_index,M=M,root=anchor)

    def factors(self, *, target:AssemblyTarget, tree:MolecularTree, z:torch.Tensor, frac:torch.Tensor, cell:torch.Tensor, role_z:torch.Tensor, role_edge_index:torch.Tensor, role_bond_type:torch.Tensor) -> tuple[dict[tuple[int,int],torch.Tensor],torch.Tensor,torch.Tensor]:
        """Return tree factors, state table, and raw tree pair-score matrices."""
        if target.anchor_role!=tree.root: raise ValueError("target anchor and tree root disagree")
        hx=self.crystal_encoder(z,frac,cell,context_mode="geometry_only")
        hm=self.molecule_encoder(role_z,role_edge_index,role_bond_type)
        states=enumerate_permutations(target.K,device=z.device);inverse=inverse_permutations(states);factors={};scores={}
        for parent,child in (*tree.tree_edges,*tree.non_tree_edges):
            parent_nodes,child_nodes=target.role_sets[parent],target.role_sets[child]
            score=self.pair_potential(hx[parent_nodes],hx[child_nodes],hm[parent],hm[child],self._bond_type(role_edge_index,role_bond_type,parent,child),frac[parent_nodes],frac[child_nodes],cell)
            scores[(parent,child)]=score
            if (parent,child) in tree.tree_edges: factors[(parent,child)]=permutation_factor(score,states,inverse)
        if self.config.strict_finite_checks and not all(torch.isfinite(value).all() for value in factors.values()): raise FloatingPointError("non-finite global copy-assembly factor")
        return factors,states,scores

    def loss(self, *, target:AssemblyTarget, tree:MolecularTree, z:torch.Tensor, frac:torch.Tensor, cell:torch.Tensor, role_z:torch.Tensor, role_edge_index:torch.Tensor, role_bond_type:torch.Tensor) -> dict[str,torch.Tensor]:
        factors,states,scores=self.factors(target=target,tree=tree,z=z,frac=frac,cell=cell,role_z=role_z,role_edge_index=role_edge_index,role_bond_type=role_bond_type)
        crf=TreeCRF(tree,num_states=len(states),identity_state=identity_index(states));target_states=target_state_indices(target,states);tree_nll=crf.nll(factors,target_states)
        pair_losses=[]
        for parent,child in tree.tree_edges:
            p_parent,p_child=target.permutations[parent],target.permutations[child];inv_child=torch.empty_like(p_child);inv_child[p_child]=torch.arange(target.K,device=p_child.device);match=inv_child[p_parent]
            pair_losses.extend([nn.functional.cross_entropy(scores[(parent,child)],match),nn.functional.cross_entropy(scores[(parent,child)].T,torch.argsort(match))])
        pair_ce=torch.stack(pair_losses).mean() if pair_losses else tree_nll.new_zeros(())
        total=self.config.tree_nll_weight*tree_nll+self.config.pair_ce_weight*pair_ce
        log_z=crf.log_partition(factors);target_score=crf.target_score(factors,target_states)
        return {"loss":total,"tree_nll":tree_nll,"pair_ce":pair_ce,"target_score":target_score,"log_partition":log_z,"target_log_probability":target_score-log_z}

    @torch.no_grad()
    def map_decode(self, *, target:AssemblyTarget, tree:MolecularTree, z:torch.Tensor, frac:torch.Tensor, cell:torch.Tensor, role_z:torch.Tensor, role_edge_index:torch.Tensor, role_bond_type:torch.Tensor) -> dict[str,object]:
        factors,states,scores=self.factors(target=target,tree=tree,z=z,frac=frac,cell=cell,role_z=role_z,role_edge_index=role_edge_index,role_bond_type=role_bond_type)
        crf=TreeCRF(tree,num_states=len(states),identity_state=identity_index(states));result=crf.map_decode(factors);G,C=decode_copy_assembly(target,states,result.state_indices)
        inverse=inverse_permutations(states)
        full_energy=sum((permutation_factor(score,states,inverse)[result.state_indices[edge[0]],result.state_indices[edge[1]]] for edge,score in scores.items()),start=result.score.new_zeros(()))
        return {"state_indices":result.state_indices,"G":G,"C":C,"tree_energy":result.score,"full_molecular_edge_energy":full_energy,"status":"CLEAN_GEOMETRY_ORACLE_R_ONLY"}
