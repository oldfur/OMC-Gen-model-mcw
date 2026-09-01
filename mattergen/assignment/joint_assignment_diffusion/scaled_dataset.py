"""Pack OMC25 cache + molmap into assignment crystals (clean G0/C0/copy_of)."""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pack/audit path is numpy-only
    torch = None  # type: ignore


def _as_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        v = v.item()
    return str(v)


def _read_jsonl(path: Path):
    if str(path).endswith(".gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
    with opener(path, mode) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_molmap(path: Path) -> dict[str, dict]:
    out = {}
    for rec in _read_jsonl(path):
        if not rec.get("success"):
            continue
        mid = _as_str(rec.get("material_id", ""))
        if mid:
            out[mid] = rec
    return out


def _per_copy_role_graphs(*, copy: np.ndarray, role: np.ndarray, crystal_bonds: list) -> dict[int, frozenset]:
    """Undirected (role_i, role_j, bond_type) multiset per copy. No coordinates."""
    n = int(copy.shape[0])
    graphs: dict[int, set[tuple[int, int, int]]] = defaultdict(set)
    for bond in crystal_bonds or []:
        i = int(bond["begin"])
        j = int(bond["end"])
        if i < 0 or j < 0 or i >= n or j >= n:
            continue
        ci, cj = int(copy[i]), int(copy[j])
        if ci != cj:
            continue
        ri, rj = int(role[i]), int(role[j])
        if ri == rj:
            continue
        a, b = (ri, rj) if ri < rj else (rj, ri)
        graphs[ci].add((a, b, int(bond.get("type", 1))))
    return {ck: frozenset(g) for ck, g in graphs.items()}


def role_template_from_mapping(mapping: dict, crystal_bonds: list) -> dict[str, Any]:
    """Build role/copy tensors and a molecular-template graph (no clean coords)."""
    mol_id = np.asarray(mapping["mol_id"], dtype=np.int64)
    mol_atom_idx = np.asarray(mapping["mol_atom_idx"], dtype=np.int64)
    k = int(mapping["num_molecules"])
    n = int(mol_id.shape[0])
    if mol_atom_idx.shape[0] != n:
        raise ValueError("mol_id/mol_atom_idx length mismatch")
    roles_set = sorted(set(int(x) for x in mol_atom_idx.tolist()))
    role_remap = {r: i for i, r in enumerate(roles_set)}
    role = np.asarray([role_remap[int(x)] for x in mol_atom_idx.tolist()], dtype=np.int64)
    m = len(roles_set)
    # cardinality: each (copy, role) at most once; each copy has m atoms
    counts = Counter(zip(mol_id.tolist(), role.tolist()))
    if any(v != 1 for v in counts.values()):
        raise ValueError("duplicate (copy, role) slot")
    copy_sizes = Counter(int(x) for x in mol_id.tolist())
    if sorted(copy_sizes.keys()) != list(range(k)):
        # allow non-contiguous copy ids by remapping
        uniq = sorted(copy_sizes.keys())
        cmap = {c: i for i, c in enumerate(uniq)}
        mol_id = np.asarray([cmap[int(x)] for x in mol_id.tolist()], dtype=np.int64)
        k = len(uniq)
        copy_sizes = Counter(int(x) for x in mol_id.tolist())
    if any(v != m for v in copy_sizes.values()):
        raise ValueError("unequal_copy_sizes")
    # Every copy must contain the full role set {0..M-1} exactly once.
    for ck, sz in copy_sizes.items():
        roles_k = {int(role[i]) for i in range(n) if int(mol_id[i]) == int(ck)}
        if roles_k != set(range(m)):
            raise ValueError("copy_missing_roles")
    edges: list[list[int]] = []
    types: list[int] = []
    seen: set[tuple[int, int]] = set()
    for bond in crystal_bonds or []:
        i = int(bond["begin"])
        j = int(bond["end"])
        if i < 0 or j < 0 or i >= n or j >= n:
            continue
        if int(mol_id[i]) != int(mol_id[j]):
            continue
        ri, rj = int(role[i]), int(role[j])
        if ri == rj:
            continue
        key = (min(ri, rj), max(ri, rj))
        if key in seen:
            continue
        seen.add(key)
        bt = int(bond.get("type", 1))
        edges.extend([[ri, rj], [rj, ri]])
        types.extend([bt, bt])
    if edges:
        role_edge_index = np.asarray(edges, dtype=np.int64).T
        role_bond_type = np.asarray(types, dtype=np.int64)
    else:
        role_edge_index = np.zeros((2, 0), dtype=np.int64)
        role_bond_type = np.zeros((0,), dtype=np.int64)
    return {
        "copy": mol_id,
        "role": role,
        "M": m,
        "K": k,
        "role_edge_index": role_edge_index,
        "role_bond_type": role_bond_type,
    }


def pack_crystal(
    *,
    material_id: str,
    pos: np.ndarray,
    cell: np.ndarray,
    atomic_numbers: np.ndarray,
    mapping_record: dict,
) -> dict[str, Any]:
    n = int(atomic_numbers.shape[0])
    mapping = mapping_record["mapping"]
    tmpl = role_template_from_mapping(mapping, mapping_record.get("crystal_bonds") or [])
    if int(tmpl["copy"].shape[0]) != n:
        raise ValueError("mapping length != N")
    # role_z: representative Z per role from any atom of that role
    role_z = np.zeros((tmpl["M"],), dtype=np.int64)
    filled = np.zeros((tmpl["M"],), dtype=bool)
    for i in range(n):
        r = int(tmpl["role"][i])
        if not filled[r]:
            role_z[r] = int(atomic_numbers[i])
            filled[r] = True
    if not bool(filled.all()):
        raise ValueError("unfilled_roles")
    # Same role across copies must share atomic number (else mol_atom_idx mixed species).
    z_by_role: dict[int, set[int]] = defaultdict(set)
    for i in range(n):
        z_by_role[int(tmpl["role"][i])].add(int(atomic_numbers[i]))
    if any(len(s) != 1 for s in z_by_role.values()):
        raise ValueError("role_z_mismatch")
    # Per-copy role graphs must be identical (symmetry-equivalent copies / Z'=1).
    graphs = _per_copy_role_graphs(
        copy=tmpl["copy"],
        role=tmpl["role"],
        crystal_bonds=mapping_record.get("crystal_bonds") or [],
    )
    for ck in range(int(tmpl["K"])):
        graphs.setdefault(int(ck), frozenset())
    uniq_g = set(graphs.values())
    if len(uniq_g) != 1:
        raise ValueError("copy_graph_mismatch")
    orbits = [[i] for i in range(int(tmpl["M"]))]
    cell = np.asarray(cell, dtype=np.float32)
    if cell.ndim == 3:
        cell = cell.reshape(3, 3)
    return {
        "id": material_id,
        "pos": np.asarray(pos, dtype=np.float32),
        "cell": cell.astype(np.float32),
        "z": np.asarray(atomic_numbers, dtype=np.int64),
        "copy": tmpl["copy"],
        "role": tmpl["role"],
        "role_z": role_z,
        "role_edge_index": tmpl["role_edge_index"],
        "role_bond_type": tmpl["role_bond_type"],
        "N": n,
        "M": int(tmpl["M"]),
        "K": int(tmpl["K"]),
        "orbits": orbits,
        "num_atoms_per_mol": int(tmpl["M"]),
    }


def crystal_to_tensors(sample: dict, device=None) -> dict:
    """CPU/GPU tensors for training (assignment from discrete labels only)."""
    if torch is None:
        raise ImportError("crystal_to_tensors requires torch")
    dev = device or "cpu"

    def t(x, dtype):
        return torch.as_tensor(x, dtype=dtype, device=dev)

    return {
        "id": sample["id"],
        "pos": t(sample["pos"], torch.float32),
        "cell": t(sample["cell"], torch.float32),
        "z": t(sample["z"], torch.long),
        "copy": t(sample["copy"], torch.long),
        "role": t(sample["role"], torch.long),
        "role_z": t(sample["role_z"], torch.long),
        "role_edge_index": t(sample["role_edge_index"], torch.long),
        "role_bond_type": t(sample["role_bond_type"], torch.long),
        "N": int(sample["N"]),
        "M": int(sample["M"]),
        "K": int(sample["K"]),
        "Z": int(sample["K"]),
        "orbits": sample["orbits"],
    }


def clean_state_from_sample(sample_t: dict):
    from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
    from mattergen.assignment.joint_assignment_diffusion.state import a_from_role_and_copy

    partition = build_orbit_partition(sample_t["orbits"])
    return a_from_role_and_copy(
        role=sample_t["role"],
        copy=sample_t["copy"],
        partition=partition,
        atomic_numbers=sample_t["z"],
        role_z=sample_t["role_z"],
        K=int(sample_t["K"]),
    ), partition


def density_of(pos: np.ndarray, cell: np.ndarray) -> float:
    c = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    vol = abs(float(np.linalg.det(c)))
    n = int(pos.shape[0])
    return n / vol if vol > 1e-12 else float("nan")


def stratified_take(items: list[dict], n: int, rng: np.random.RandomState) -> list[dict]:
    """Spread copy-count and atom-count bins."""
    if n >= len(items):
        rng.shuffle(items)
        return list(items)
    buckets: dict[tuple, list] = defaultdict(list)
    for rec in items:
        k = int(rec["K"])
        n_at = int(rec["N"])
        nbin = 0 if n_at < 20 else 1 if n_at < 35 else 2
        buckets[(k, nbin)].append(rec)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    out: list[dict] = []
    # round-robin
    ptr = {k: 0 for k in keys}
    for k in keys:
        rng.shuffle(buckets[k])
    while len(out) < n:
        progressed = False
        for k in keys:
            i = ptr[k]
            if i < len(buckets[k]):
                out.append(buckets[k][i])
                ptr[k] = i + 1
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    rng.shuffle(out)
    return out[:n]


def dataset_stats(samples: list[dict]) -> dict:
    if not samples:
        return {"n": 0}
    ns = [int(s["N"]) for s in samples]
    ks = [int(s["K"]) for s in samples]
    ms = [int(s["M"]) for s in samples]
    dens = [density_of(s["pos"], s["cell"]) for s in samples]
    vols = [abs(float(np.linalg.det(np.asarray(s["cell"]).reshape(3, 3)))) for s in samples]
    return {
        "n": len(samples),
        "N_min": min(ns),
        "N_median": float(np.median(ns)),
        "N_max": max(ns),
        "K_hist": dict(sorted(Counter(ks).items())),
        "M_min": min(ms),
        "M_median": float(np.median(ms)),
        "M_max": max(ms),
        "density_min": float(np.nanmin(dens)),
        "density_median": float(np.nanmedian(dens)),
        "density_max": float(np.nanmax(dens)),
        "volume_median": float(np.median(vols)),
    }
