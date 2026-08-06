"""Deterministic molecular spanning-tree construction."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import torch


@dataclass(frozen=True)
class MolecularTree:
    root: int
    parent: dict[int, int | None]
    children: dict[int, tuple[int, ...]]
    preorder: tuple[int, ...]
    postorder: tuple[int, ...]
    tree_edges: tuple[tuple[int, int], ...]  # parent, child
    non_tree_edges: tuple[tuple[int, int], ...]


def select_anchor_role(role_orbits: list[list[int]], edge_index: torch.Tensor, *, explicit_anchor: int | None = None) -> int:
    singleton = {orbit[0] for orbit in role_orbits if len(orbit) == 1}
    if not singleton:
        raise ValueError("no singleton automorphism orbit exists; a stable anchor cannot be selected")
    if explicit_anchor is not None:
        if explicit_anchor not in singleton:
            raise ValueError("explicit anchor_role must belong to a singleton automorphism orbit")
        return explicit_anchor
    degree = {role: 0 for role in singleton}
    for source, target in edge_index.T.tolist():
        if source in degree: degree[source] += 1
        if target in degree: degree[target] += 1
    return min(singleton, key=lambda role: (-degree[role], role))


def build_bfs_tree(edge_index: torch.Tensor, *, M: int, root: int) -> MolecularTree:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("molecular edge_index must have shape [2,E]")
    if not 0 <= root < M:
        raise ValueError("tree root is out of range")
    adjacency = {role: set() for role in range(M)}
    undirected = set()
    for source, target in edge_index.T.tolist():
        if not (0 <= source < M and 0 <= target < M):
            raise ValueError("molecular edge contains an out-of-range role")
        if source == target: continue
        adjacency[source].add(target); adjacency[target].add(source)
        undirected.add(tuple(sorted((source, target))))
    parent: dict[int, int | None] = {root: None}; order=[]; queue=deque([root])
    while queue:
        node=queue.popleft(); order.append(node)
        for neighbour in sorted(adjacency[node]):
            if neighbour not in parent:
                parent[neighbour]=node; queue.append(neighbour)
    if len(parent) != M:
        missing=sorted(set(range(M))-set(parent)); raise ValueError(f"molecular graph is disconnected from anchor {root}; missing roles {missing}")
    children={role: [] for role in range(M)}
    for child, ancestor in parent.items():
        if ancestor is not None: children[ancestor].append(child)
    tree_edges=tuple((parent[role],role) for role in order if parent[role] is not None)
    tree_undirected={tuple(sorted(edge)) for edge in tree_edges}
    return MolecularTree(root=root,parent=parent,children={k:tuple(sorted(v)) for k,v in children.items()},preorder=tuple(order),postorder=tuple(reversed(order)),tree_edges=tree_edges,non_tree_edges=tuple(sorted(undirected-tree_undirected)))
