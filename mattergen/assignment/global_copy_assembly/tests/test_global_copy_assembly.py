"""Tests intentionally authored but not executed in the implementation-only turn."""
from __future__ import annotations

import itertools
import pytest
import torch

from mattergen.assignment.global_copy_assembly.decoder import decode_copy_assembly
from mattergen.assignment.global_copy_assembly.metrics import projected_molecular_bonds
from mattergen.assignment.global_copy_assembly.pair_potential import permutation_factor
from mattergen.assignment.global_copy_assembly.permutations import compose, enumerate_permutations, identity_index, inverse_permutations
from mattergen.assignment.global_copy_assembly.targets import build_assembly_target, permutations_to_group, target_state_indices, validate_uniform_batch_k
from mattergen.assignment.global_copy_assembly.tree_builder import build_bfs_tree, select_anchor_role
from mattergen.assignment.global_copy_assembly.tree_crf import TreeCRF


def synthetic_target(M=3,K=2):
    # Stable atom indices are deliberately interleaved across roles/copies.
    role=torch.tensor([0,1,2,0,1,2]);copy=torch.tensor([0,1,0,1,0,1])
    return build_assembly_target(role,copy,M=M,K=K,anchor_role=0),role,copy


def test_permutation_convention_round_trip_and_identity():
    states=enumerate_permutations(3);inverse=inverse_permutations(states)
    assert len(states)==6 and torch.equal(states[identity_index(states)],torch.arange(3))
    for p,inv in zip(states,inverse): assert torch.equal(compose(p,inv),torch.arange(3))


def test_target_constructs_permutations_and_exact_C():
    target,_,copy=synthetic_target();G=permutations_to_group(target);C=G@G.T;C0=copy[:,None].eq(copy[None,:]).float()
    assert torch.equal(C,C0) and torch.equal(target.permutations[target.anchor_role],torch.arange(target.K))


def brute_force_logz(tree,factors,states,identity):
    values=[]
    for choice in itertools.product(range(states),repeat=len(tree.preorder)-1):
        assignment={tree.root:identity};assignment.update({role:value for role,value in zip(tree.preorder[1:],choice)})
        values.append(sum(factors[e][assignment[e[0]],assignment[e[1]]] for e in tree.tree_edges))
    return torch.logsumexp(torch.stack(values),0),torch.stack(values).max()


def test_tree_crf_matches_bruteforce_and_has_finite_gradient():
    tree=build_bfs_tree(torch.tensor([[0,1,1],[1,2,3]]),M=4,root=0);states=enumerate_permutations(2);factors={edge:torch.randn(2,2,requires_grad=True) for edge in tree.tree_edges};crf=TreeCRF(tree,num_states=2,identity_state=identity_index(states))
    logz,best=brute_force_logz(tree,factors,2,crf.identity_state)
    assert torch.allclose(crf.log_partition(factors),logz)
    decoded=crf.map_decode(factors);assert torch.allclose(decoded.score,best)
    target={0:crf.identity_state,1:0,2:1,3:0};loss=crf.nll(factors,target);loss.backward();assert all(torch.isfinite(x.grad).all() for x in factors.values())


def test_oracle_pair_scores_recover_C_and_projected_graph():
    target,role,copy=synthetic_target();states=enumerate_permutations(2);inverse=inverse_permutations(states);tree=build_bfs_tree(torch.tensor([[0,1],[1,2]]),M=3,root=0)
    factors={}
    for parent,child in tree.tree_edges:
        score=torch.full((2,2),-10.);p,q=target.permutations[parent],target.permutations[child];invq=torch.empty_like(q);invq[q]=torch.arange(2);score[torch.arange(2),invq[p]]=10.;factors[(parent,child)]=permutation_factor(score,states,inverse)
    crf=TreeCRF(tree,num_states=len(states),identity_state=identity_index(states));G,C=decode_copy_assembly(target,states,crf.map_decode(factors).state_indices);assert torch.equal(C,copy[:,None].eq(copy[None,:]).float()) and torch.equal(G.sum(0),torch.full((2,),3.))
    R=torch.nn.functional.one_hot(role,3).float();B=torch.zeros(3,3,1);B[0,1]=B[1,0]=B[1,2]=B[2,1]=1.;assert projected_molecular_bonds(R,C,B).any()


def test_copy_label_gauge_leaves_C_invariant():
    target,_,_=synthetic_target();G=permutations_to_group(target);sigma=torch.tensor([1,0]);assert torch.equal(G@G.T,(G[:,sigma])@(G[:,sigma]).T)


def test_non_bijective_decoded_permutation_fails_loudly():
    target,_,_=synthetic_target();bad={**target.permutations,1:torch.tensor([0,0])}
    with pytest.raises(ValueError): permutations_to_group(target,bad)


@pytest.mark.parametrize("bad",[torch.tensor([0,0,0,1,1,2]),torch.tensor([0,1,2,0,1])])
def test_bad_role_capacity_fails_loudly(bad):
    with pytest.raises(ValueError): build_assembly_target(bad,torch.tensor([0,1,0,1,0,1]),M=3,K=2,anchor_role=0)


def test_failure_modes_fail_loudly():
    with pytest.raises(ValueError): select_anchor_role([[0,1]],torch.tensor([[0],[1]]))
    with pytest.raises(ValueError): build_bfs_tree(torch.tensor([[0],[1]]),M=3,root=0)
    with pytest.raises(ValueError): validate_uniform_batch_k([2,3])
    with pytest.raises(ValueError): build_assembly_target(torch.tensor([0,1,2,0,1,2]),torch.tensor([0,0,0,0,0,0]),M=3,K=2,anchor_role=0)
    states=enumerate_permutations(2);tree=build_bfs_tree(torch.tensor([[0],[1]]),M=2,root=0);crf=TreeCRF(tree,num_states=2,identity_state=0)
    with pytest.raises(FloatingPointError): crf.log_partition({(0,1):torch.full((2,2),float("nan"))})
