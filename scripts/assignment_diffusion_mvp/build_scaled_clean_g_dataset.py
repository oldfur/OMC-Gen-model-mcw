#!/usr/bin/env python3
"""Build a fixed OMC25 assignment subset from MatterGen cache + hybrid molmap."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.joint_assignment_diffusion.scaled_dataset import (
    dataset_stats,
    load_molmap,
    pack_crystal,
    stratified_take,
)


def _load_split(cache: Path, split: str):
    d = cache / split
    return {
        "num_atoms": np.load(d / "num_atoms.npy"),
        "pos": np.load(d / "pos.npy"),
        "cell": np.load(d / "cell.npy"),
        "atomic_numbers": np.load(d / "atomic_numbers.npy"),
        "structure_id": np.load(d / "structure_id.npy", allow_pickle=True),
    }


def _iter_crystals(blob):
    counts = blob["num_atoms"]
    off = np.concatenate([[0], np.cumsum(counts)])
    for i, n in enumerate(counts.tolist()):
        n = int(n)
        sid = blob["structure_id"][i]
        if isinstance(sid, bytes):
            sid = sid.decode()
        sid = str(sid)
        yield i, sid, n, off[i], off[i] + n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, required=True, help=".../cache/omc25_le50_mattergen")
    p.add_argument("--molmap-train", type=Path, required=True)
    p.add_argument("--molmap-val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-n", type=int, default=3000)
    p.add_argument("--val-n", type=int, default=250)
    p.add_argument("--test-n", type=int, default=150)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-atoms", type=int, default=50)
    p.add_argument("--min-copies", type=int, default=2, help="K>=this (assignment-relevant)")
    p.add_argument("--audit-only", action="store_true", help="Count rejects/examples; do not write train.pt")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    rng = np.random.RandomState(int(args.seed))
    maps = {}
    maps.update(load_molmap(args.molmap_train))
    maps.update(load_molmap(args.molmap_val))
    print(json.dumps({"event": "molmap_loaded", "n": len(maps)}), flush=True)

    reject = Counter()
    examples = defaultdict(list)
    valid_examples = []
    n_candidates = Counter()
    n_valid_partition = Counter()
    n_valid_orbit = Counter()
    pool_k = Counter()
    pool_m = Counter()
    PARTITION_REASONS = {
        "duplicate (copy, role) slot",
        "unequal_copy_sizes",
        "copy_missing_roles",
        "mol_id/mol_atom_idx length mismatch",
        "mapping length != N",
        "unfilled_roles",
    }
    ORBIT_REASONS = {"role_z_mismatch", "copy_graph_mismatch"}
    _ZSYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}

    def _formula(role_z) -> str:
        c = Counter(int(x) for x in role_z)
        parts = []
        for z in sorted(c, key=lambda q: (q not in (6, 1), q)):
            parts.append(f"{_ZSYM.get(z, str(z))}{c[z] if c[z] > 1 else ''}")
        return "".join(parts)

    def _note(reason: str, sid: str):
        reject[reason] += 1
        if len(examples[reason]) < 4:
            examples[reason].append(sid)

    def collect(split: str) -> list[dict]:
        blob = _load_split(args.cache_root, split)
        packed = []
        n_cand = 0
        for i, sid, n, a, b in _iter_crystals(blob):
            n_cand += 1
            rec = maps.get(sid)
            if rec is None:
                _note("no_molmap", sid)
                continue
            if n > int(args.max_atoms):
                _note("too_many_atoms", sid)
                continue
            try:
                sample = pack_crystal(
                    material_id=sid,
                    pos=blob["pos"][a:b],
                    cell=blob["cell"][i],
                    atomic_numbers=blob["atomic_numbers"][a:b],
                    mapping_record=rec,
                )
            except ValueError as exc:
                reason = str(exc).split(":")[0].strip() or "pack_failed"
                if reason in ORBIT_REASONS:
                    n_valid_partition[split] += 1
                _note(reason, sid)
                continue
            except Exception:
                _note("pack_failed", sid)
                continue
            n_valid_partition[split] += 1
            n_valid_orbit[split] += 1
            if int(sample["K"]) < int(args.min_copies):
                _note("too_few_copies", sid)
                continue
            if int(sample["M"]) < 1:
                _note("empty_roles", sid)
                continue
            packed.append(sample)
            pool_k[int(sample["K"])] += 1
            pool_m[int(sample["M"])] += 1
            if len(valid_examples) < 10:
                valid_examples.append(
                    {
                        "id": sid,
                        "split": split,
                        "N": int(sample["N"]),
                        "K": int(sample["K"]),
                        "M": int(sample["M"]),
                        "formula_per_mol": _formula(sample["role_z"]),
                        "role_z": [int(x) for x in sample["role_z"].tolist()],
                        "copy_atom_counts": dict(Counter(int(x) for x in sample["copy"].tolist())),
                    }
                )
        n_candidates[split] = n_cand
        print(
            json.dumps({"event": "collect", "split": split, "candidates": n_cand, "kept": len(packed)}),
            flush=True,
        )
        return packed

    train_pool = collect("train")
    val_pool = collect("val")
    train = stratified_take(train_pool, int(args.train_n), rng)
    # hold out test from val pool
    rng2 = np.random.RandomState(int(args.seed) + 1)
    rng2.shuffle(val_pool)
    test = val_pool[: int(args.test_n)]
    remain = val_pool[int(args.test_n) :]
    val = stratified_take(remain, int(args.val_n), np.random.RandomState(int(args.seed) + 2))
    # drop id overlap
    train_ids = {s["id"] for s in train}
    val = [s for s in val if s["id"] not in train_ids]
    test = [s for s in test if s["id"] not in train_ids and s["id"] not in {x["id"] for x in val}]

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    stats = {k: dataset_stats(v) for k, v in splits.items()}
    # too_few_copies still had valid partition+orbit; subtract them from experiment pool only.
    partition_reject = {k: int(v) for k, v in reject.items() if k in PARTITION_REASONS}
    orbit_reject = {k: int(v) for k, v in reject.items() if k in ORBIT_REASONS}
    audit = {
        "filter": "Zprime=1 equivalent repeated copies (same role set, role_z, copy graphs)",
        "total_candidate_crystals": int(sum(n_candidates.values())),
        "candidates_by_split": dict(n_candidates),
        "valid_partition_crystals": int(sum(n_valid_partition.values())),
        "valid_role_orbit_crystals": int(sum(n_valid_orbit.values())),
        "valid_experiment_pool_k_ge_2": {"train": len(train_pool), "val": len(val_pool)},
        "rejected": dict(reject),
        "rejected_total": int(sum(reject.values())),
        "rejected_partition": partition_reject,
        "rejected_orbit": orbit_reject,
        "copy_count_distribution": dict(sorted(pool_k.items())),
        "atoms_per_molecule_distribution": dict(sorted(pool_m.items())),
        "valid_partition_and_orbit": {k: len(v) for k, v in splits.items()},
        "pool_before_subsample": {"train": len(train_pool), "val": len(val_pool)},
        "examples_valid": valid_examples,
        "examples_rejected": {k: v for k, v in examples.items()},
    }
    (args.out / "assignment_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"event": "assignment_audit", **audit}, indent=2), flush=True)
    if args.audit_only:
        print(json.dumps({"event": "audit_only_done", "out": str(args.out / "assignment_audit.json")}), flush=True)
        return
    manifest = {
        "cache_root": str(args.cache_root),
        "molmap_train": str(args.molmap_train),
        "molmap_val": str(args.molmap_val),
        "seed": int(args.seed),
        "max_atoms": int(args.max_atoms),
        "min_copies": int(args.min_copies),
        "counts": {k: len(v) for k, v in splits.items()},
        "stats": stats,
        "ids": {k: [s["id"] for s in v] for k, v in splits.items()},
        "note": "Only symmetry-equivalent repeated copies (Z'=1). Roles are not shared across different molecular species.",
        "assignment_audit": str(args.out / "assignment_audit.json"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    torch_ok = True
    try:
        import torch
    except Exception:
        torch_ok = False
    for name, samples in splits.items():
        if torch_ok:
            import torch

            torch.save(samples, args.out / f"{name}.pt")
        else:
            np.savez_compressed(args.out / f"{name}.npz", samples=np.array(samples, dtype=object))
        (args.out / f"{name}_ids.json").write_text(json.dumps([s["id"] for s in samples], indent=2))
    print(json.dumps({"event": "scaled_dataset_done", "out": str(args.out), **manifest["counts"], "stats": stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()
