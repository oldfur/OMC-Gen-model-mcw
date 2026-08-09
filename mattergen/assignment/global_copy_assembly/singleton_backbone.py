"""Singleton-orbit copy backbone with real + virtual molecular edges.

Molecular graph without non-singleton roles may be disconnected.  We build a
deterministic singleton graph that:

1. keeps real molecular edges among singleton roles;
2. if components remain disconnected, adds virtual edges using shortest-path
   relations on the *full* molecular graph (path may pass non-singleton roles);
3. builds a BFS spanning tree from the singleton anchor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import torch

from .tree_builder import MolecularTree


@dataclass(frozen=True)
class SingletonEdge:
    parent: int  # molecular role id
    child: int
    real: bool
    path_length: int  # 1 for real edges
    path_bond_types: tuple[int, ...]  # bond types along molecular shortest path


@dataclass(frozen=True)
class SingletonBackbone:
    """Singleton role subset + spanning tree for Stage A assembly."""

    singleton_roles: tuple[int, ...]  # sorted role ids
    role_local_index: dict[int, int]  # role -> 0..S-1
    tree: MolecularTree  # uses *local* indices 0..S-1
    edges: tuple[SingletonEdge, ...]  # tree + non-tree undirected unique
    tree_edges_roles: tuple[tuple[int, int], ...]  # (parent_role, child_role)
    non_tree_edges_roles: tuple[tuple[int, int], ...]
    virtual_tree_edges: tuple[tuple[int, int], ...]


def _undirected_adjacency(edge_index: torch.Tensor, bond_type: torch.Tensor, M: int) -> dict[int, list[tuple[int, int]]]:
    adj: dict[int, list[tuple[int, int]]] = {r: [] for r in range(M)}
    for (src, dst), bt in zip(edge_index.T.tolist(), bond_type.tolist()):
        if src == dst:
            continue
        adj[src].append((dst, int(bt)))
        adj[dst].append((src, int(bt)))
    # unique neighbors prefer first bond type
    cleaned: dict[int, list[tuple[int, int]]] = {r: [] for r in range(M)}
    for r, neigh in adj.items():
        seen = {}
        for n, bt in sorted(neigh):
            if n not in seen:
                seen[n] = bt
        cleaned[r] = [(n, seen[n]) for n in sorted(seen)]
    return cleaned


def _shortest_path(
    adj: dict[int, list[tuple[int, int]]],
    source: int,
    target: int,
) -> tuple[int, tuple[int, ...]] | None:
    """BFS shortest path; returns (length in edges, bond-type sequence)."""
    if source == target:
        return 0, ()
    parent: dict[int, tuple[int, int] | None] = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neigh, bt in adj[node]:
            if neigh in parent:
                continue
            parent[neigh] = (node, bt)
            if neigh == target:
                # reconstruct
                bonds: list[int] = []
                cur = target
                while cur != source:
                    prev, edge_bt = parent[cur]  # type: ignore[misc]
                    bonds.append(edge_bt)
                    cur = prev
                bonds.reverse()
                return len(bonds), tuple(bonds)
            queue.append(neigh)
    return None


def build_singleton_backbone(
    role_edge_index: torch.Tensor,
    role_bond_type: torch.Tensor,
    *,
    M: int,
    singleton_roles: list[int] | tuple[int, ...],
    anchor_role: int,
) -> SingletonBackbone:
    """Construct deterministic singleton backbone graph and spanning tree."""
    roles = tuple(sorted(int(r) for r in singleton_roles))
    if anchor_role not in roles:
        raise ValueError(f"anchor_role {anchor_role} is not a singleton role")
    if len(roles) < 1:
        raise ValueError("singleton_roles must be non-empty")
    role_local = {role: i for i, role in enumerate(roles)}
    S = len(roles)
    full_adj = _undirected_adjacency(role_edge_index, role_bond_type, M)

    # Real edges among singletons.
    real_undirected: set[tuple[int, int]] = set()
    real_bond: dict[tuple[int, int], int] = {}
    for a in roles:
        for b, bt in full_adj[a]:
            if b in role_local and a < b:
                real_undirected.add((a, b))
                real_bond[(a, b)] = bt

    # Virtual edges: complete graph among components via shortest path on full mol graph.
    # First find connected components using only real singleton edges.
    parent_cc = {r: r for r in roles}

    def find(x: int) -> int:
        while parent_cc[x] != x:
            parent_cc[x] = parent_cc[parent_cc[x]]
            x = parent_cc[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_cc[rb] = ra

    for a, b in real_undirected:
        union(a, b)
    components: dict[int, list[int]] = {}
    for r in roles:
        components.setdefault(find(r), []).append(r)

    virtual_undirected: set[tuple[int, int]] = set()
    virtual_path: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}
    comp_ids = list(components.keys())
    # Connect components into a meta-tree with shortest inter-component paths.
    if len(comp_ids) > 1:
        # Greedy: repeatedly add shortest path between different components.
        meta_parent = {c: c for c in comp_ids}

        def mfind(x: int) -> int:
            while meta_parent[x] != x:
                meta_parent[x] = meta_parent[meta_parent[x]]
                x = meta_parent[x]
            return x

        candidates: list[tuple[int, int, int, tuple[int, ...]]] = []
        for i, ci in enumerate(comp_ids):
            for cj in comp_ids[i + 1 :]:
                best = None
                for u in components[ci]:
                    for v in components[cj]:
                        sp = _shortest_path(full_adj, u, v)
                        if sp is None:
                            continue
                        length, bonds = sp
                        if best is None or (length, u, v) < (best[0], best[1], best[2]):
                            best = (length, u, v, bonds)
                if best is not None:
                    length, u, v, bonds = best
                    a, b = (u, v) if u < v else (v, u)
                    candidates.append((length, a, b, bonds))
        candidates.sort()
        for length, a, b, bonds in candidates:
            ca, cb = mfind(find(a)), mfind(find(b))
            if ca == cb:
                continue
            meta_parent[cb] = ca
            key = (a, b)
            virtual_undirected.add(key)
            virtual_path[key] = (length, bonds)
            if all(mfind(c) == mfind(comp_ids[0]) for c in comp_ids):
                break

    # Singleton adjacency for BFS tree (real + virtual).
    adj: dict[int, set[int]] = {r: set() for r in roles}
    edge_meta: dict[tuple[int, int], SingletonEdge] = {}
    for a, b in real_undirected:
        adj[a].add(b)
        adj[b].add(a)
        edge_meta[(a, b)] = SingletonEdge(
            parent=a, child=b, real=True, path_length=1, path_bond_types=(real_bond[(a, b)],)
        )
    for a, b in virtual_undirected:
        adj[a].add(b)
        adj[b].add(a)
        length, bonds = virtual_path[(a, b)]
        edge_meta[(a, b)] = SingletonEdge(
            parent=a, child=b, real=False, path_length=length, path_bond_types=bonds
        )

    # BFS tree in *role id* space, then map to local indices for MolecularTree.
    root = anchor_role
    parent_role: dict[int, int | None] = {root: None}
    order: list[int] = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neigh in sorted(adj[node]):
            if neigh not in parent_role:
                parent_role[neigh] = node
                queue.append(neigh)
    if len(parent_role) != S:
        missing = sorted(set(roles) - set(parent_role))
        raise ValueError(f"singleton backbone remains disconnected from anchor; missing {missing}")

    # Local-index MolecularTree
    local_parent: dict[int, int | None] = {}
    for role, par in parent_role.items():
        local_parent[role_local[role]] = None if par is None else role_local[par]
    children: dict[int, list[int]] = {i: [] for i in range(S)}
    local_order = [role_local[r] for r in order]
    for child, ancestor in local_parent.items():
        if ancestor is not None:
            children[ancestor].append(child)
    tree_edges_local = tuple(
        (local_parent[node], node) for node in local_order if local_parent[node] is not None
    )
    # non-tree undirected among singleton edges
    all_undirected = real_undirected | virtual_undirected
    tree_role_edges = []
    for role in order:
        par = parent_role[role]
        if par is not None:
            tree_role_edges.append((par, role))
    tree_undirected = {tuple(sorted(e)) for e in tree_role_edges}
    non_tree = tuple(sorted(all_undirected - tree_undirected))
    tree_edges_roles = tuple(tree_role_edges)
    virtual_tree = tuple(
        (p, c) for p, c in tree_edges_roles if not edge_meta[tuple(sorted((p, c)))].real
    )

    tree = MolecularTree(
        root=role_local[root],
        parent=local_parent,
        children={k: tuple(sorted(v)) for k, v in children.items()},
        preorder=tuple(local_order),
        postorder=tuple(reversed(local_order)),
        tree_edges=tree_edges_local,
        non_tree_edges=tuple(
            (role_local[a], role_local[b]) if role_local[a] < role_local[b] else (role_local[b], role_local[a])
            for a, b in non_tree
        ),
    )
    edges = tuple(edge_meta[k] for k in sorted(edge_meta.keys()))
    return SingletonBackbone(
        singleton_roles=roles,
        role_local_index=role_local,
        tree=tree,
        edges=edges,
        tree_edges_roles=tree_edges_roles,
        non_tree_edges_roles=non_tree,
        virtual_tree_edges=virtual_tree,
    )


def edge_meta_lookup(backbone: SingletonBackbone, role_a: int, role_b: int) -> SingletonEdge:
    key = tuple(sorted((int(role_a), int(role_b))))
    for edge in backbone.edges:
        if tuple(sorted((edge.parent, edge.child))) == key:
            return edge
    raise KeyError(f"no singleton edge between roles {role_a} and {role_b}")
