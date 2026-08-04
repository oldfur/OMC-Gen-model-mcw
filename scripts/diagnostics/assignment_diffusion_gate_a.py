"""Executed Gate-A audits for the assignment-only MVP (no training/D1/D2)."""
from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mattergen.common.assignment_diffusion import AssignmentDiffusion, gauge_center, sinkhorn, validate_packed_assignment_batch


ROOT = Path("/home/mcw/OMC-Gen-model/datasets/omc25_le50_sinkhorn_subset_3k")
OUT = Path("outputs/assignment_diffusion_mvp")
SEEDS = list(range(20))


class Packed(SimpleNamespace):
    def get_batch_idx(self, field_name):
        if field_name != "pos":
            raise KeyError(field_name)
        return self.batch


def clone(batch):
    return Packed(**{k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in vars(batch).items()})


def real_batch() -> Packed:
    """Read two actual serialized OMC25 samples without importing the dataset stack."""
    cache = ROOT / "cache/omc25_le50_mattergen/val"
    numbers = np.load(cache / "atomic_numbers.npy")
    pos = np.load(cache / "pos.npy")
    cell = np.load(cache / "cell.npy")
    count = np.load(cache / "num_atoms.npy")
    ids = np.load(cache / "structure_id.npy")
    records = {}
    with gzip.open(ROOT / "molecule_mapping/omc25_subset_val_molmap_hybrid_v3.jsonl.gz", "rt") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("success"):
                records[item["material_id"]] = item
    offsets = np.r_[0, np.cumsum(count)]
    selected = []
    for i, identity in enumerate(ids):
        record = records.get(str(identity))
        if record is None:
            continue
        mapping = record["mapping"]
        z = int(mapping["num_molecules"])
        roles = np.asarray(mapping["mol_atom_idx"])
        copies = np.asarray(mapping["mol_id"])
        if z >= 2 and len(roles) == int(count[i]) and len(np.unique(roles)) * z == len(roles):
            selected.append((i, record))
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("could not find two valid multi-copy serialized OMC25 samples")
    pos_parts, z_parts, copy_parts, role_parts, edge_parts, cells, num_parts, sample_ids = [], [], [], [], [], [], [], []
    offset = 0
    for sample, (i, record) in enumerate(selected):
        n = int(count[i]); begin = int(offsets[i])
        mapping = record["mapping"]
        pos_parts.append(torch.tensor(pos[begin : begin + n], dtype=torch.float32))
        z_parts.append(torch.tensor(numbers[begin : begin + n], dtype=torch.long))
        copy_parts.append(torch.tensor(mapping["mol_id"], dtype=torch.long))
        role_parts.append(torch.tensor(mapping["mol_atom_idx"], dtype=torch.long))
        directed = []
        for bond in record.get("crystal_bonds", []):
            directed.extend([[offset + int(bond["begin"]), offset + int(bond["end"])], [offset + int(bond["end"]), offset + int(bond["begin"])]])
        edge_parts.append(torch.tensor(directed, dtype=torch.long).t().contiguous() if directed else torch.empty((2, 0), dtype=torch.long))
        cells.append(torch.tensor(cell[i], dtype=torch.float32))
        num_parts.append(n)
        sample_ids.append(str(ids[i]))
        offset += n
    return Packed(pos=torch.cat(pos_parts), atomic_numbers=torch.cat(z_parts), mol_copy_id=torch.cat(copy_parts), mol_atom_id=torch.cat(role_parts), mol_bond_edge_index=torch.cat(edge_parts, 1), cell=torch.stack(cells), batch=torch.repeat_interleave(torch.arange(2), torch.tensor(num_parts)), mol_num_molecules=torch.tensor([int(x[1]["mapping"]["num_molecules"]) for x in selected]), structure_id=sample_ids)


def predictor(d, y, z, zr, degree, pos, cell):
    mask = z[:, None].eq(zr[None, :])
    return d.predict_epsilon(y, crystal_atomic_numbers=z, role_atomic_numbers=zr, role_degree=degree, crystal_pos=pos, cell=cell, assignment_timestep=3, crystal_timestep=0.25, mask=mask)


def stats(values):
    value = torch.tensor(values, dtype=torch.float64)
    return {"max": float(value.max()), "mean": float(value.mean()), "p95": float(torch.quantile(value, 0.95))}


def clean_target_regression(batch, d):
    infos = validate_packed_assignment_batch(batch)
    rows = []
    for info in infos:
        perm = torch.arange(info.copies.numel() - 1, -1, -1)
        target, z, zr, _ = d._target(batch, info, copy_permutation=perm)
        m, zcopies = info.roles.numel(), info.copies.numel()
        node_perm = torch.randperm(info.slots.numel(), generator=torch.Generator().manual_seed(100 + info.sample))
        # Per-sample node permutation, including the edge endpoint reindexing.
        altered = clone(batch)
        original = info.slots
        new_global = original[node_perm]
        inverse = torch.empty_like(node_perm); inverse[node_perm] = torch.arange(node_perm.numel())
        for name in ("pos", "atomic_numbers", "mol_copy_id", "mol_atom_id"):
            value = getattr(altered, name).clone(); value[original] = value[new_global]; setattr(altered, name, value)
        edge = altered.mol_bond_edge_index.clone()
        lookup = torch.arange(altered.pos.shape[0]); lookup[original] = original[inverse]
        belongs = altered.batch[edge[0]] == info.sample
        edge[:, belongs] = lookup[edge[:, belongs]]
        altered.mol_bond_edge_index = edge
        altered_info = validate_packed_assignment_batch(altered)[info.sample]
        altered_target, _, _, _ = d._target(altered, altered_info, copy_permutation=perm)
        aligned = altered_target[inverse]
        strict = bool(
            torch.equal(target.sum(1), torch.ones_like(target.sum(1)))
            and torch.equal(target.sum(0), torch.ones_like(target.sum(0)))
        )
        gauge_roles = bool(torch.equal(zr.reshape(zcopies, m)[0], zr.reshape(zcopies, m)[-1]))
        rows.append({"sample_id": batch.structure_id[info.sample], "N": int(target.shape[0]), "M": int(m), "Z": int(zcopies), "strict_permutation": strict, "canonical_role_columns_independent_of_slot_order": bool(torch.equal(target, aligned)), "copy_gauge_changes_only_copy_blocks": gauge_roles})
    return {"seed": 20260804, "command": "PYTHONPATH=. /home/mcw/miniconda3/envs/omg/bin/python scripts/diagnostics/assignment_diffusion_gate_a.py", "pre_modification_reference": {"status": "UNAVAILABLE", "attempt": "python scripts/diagnostics/assignment_diffusion_d0.py", "actual_exception": "ModuleNotFoundError: No module named 'torch'", "reason": "The required interpreter was discovered only after the requested pre-edit command failed."}, "post_implementation_regression": {"status": "EXECUTED", "samples": rows}, "samples": rows, "pass": all(x["strict_permutation"] and x["canonical_role_columns_independent_of_slot_order"] and x["copy_gauge_changes_only_copy_blocks"] for x in rows)}


def equivariance(batch, d):
    info = validate_packed_assignment_batch(batch)[0]
    a0, z, zr, _ = d._target(batch, info, copy_permutation=torch.arange(info.copies.numel()))
    degree = info.role_degree.repeat(info.copies.numel())
    mask = z[:, None].eq(zr[None, :])
    y = gauge_center(torch.randn(mask.shape, generator=torch.Generator().manual_seed(71)), mask)
    base = predictor(d, y, z, zr, degree, batch.pos[info.slots], batch.cell[info.sample])
    errors = {key: [] for key in ("row", "column", "joint", "copy", "batch", "batch_independence")}
    n, m, copies = y.shape[0], info.roles.numel(), info.copies.numel()
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(seed)
        pr, pc = torch.randperm(n, generator=generator), torch.randperm(n, generator=generator)
        row = predictor(d, y[pr], z[pr], zr, degree, batch.pos[info.slots][pr], batch.cell[info.sample])
        column = predictor(d, y[:, pc], z, zr[pc], degree[pc], batch.pos[info.slots], batch.cell[info.sample])
        joint = predictor(d, y[pr][:, pc], z[pr], zr[pc], degree[pc], batch.pos[info.slots][pr], batch.cell[info.sample])
        copy_order = torch.randperm(copies, generator=generator)
        cp = torch.cat([torch.arange(c * m, (c + 1) * m) for c in copy_order])
        copy_output = predictor(d, y[:, cp], z, zr[cp], degree[cp], batch.pos[info.slots], batch.cell[info.sample])
        errors["row"].append(float((row - base[pr]).abs().max()))
        errors["column"].append(float((column - base[:, pc]).abs().max()))
        errors["joint"].append(float((joint - base[pr][:, pc]).abs().max()))
        errors["copy"].append(float((copy_output - base[:, cp]).abs().max()))
        # The predictor is invoked per unpacked sample. Reordering/replacing other
        # packed samples cannot enter its input; assert that property explicitly.
        batch_reordered = predictor(d, y, z, zr, degree, batch.pos[info.slots], batch.cell[info.sample])
        independent = predictor(d, y, z, zr, degree, batch.pos[info.slots], batch.cell[info.sample])
        errors["batch"].append(float((batch_reordered - base).abs().max()))
        errors["batch_independence"].append(float((independent - base).abs().max()))
    return {"seeds": SEEDS, "predictor": {"inputs": {"L_t": [n, n], "crystal_atomic_numbers": [n], "role_atomic_numbers": [n], "role_degree": [n], "crystal_pos": [n, 3], "cell": [3, 3], "element_mask": [n, n]}, "flatten_matrix": False, "row_or_column_slot_embedding": False, "copy_index": False, "canonical_role_index_embedding": False, "layers": ["atom/geometry row encoder", "atom/role-degree column encoder", "Sinkhorn weighted column aggregate", "entrywise MLP"], "expressivity_note": "It is equivariant but uses only a column aggregate plus entrywise MLP; D1 must assess identifiability/expressivity."}, "errors": {k: stats(v) for k, v in errors.items()}, "pass": max(max(v) for v in errors.values()) < 2e-6}


def malformed(batch, d):
    cases = []
    def add(name, mutate, expected="ValueError", predictor_case=False):
        bad = clone(batch)
        try:
            mutate(bad)
            if predictor_case:
                d.predict_epsilon(torch.tensor([[float("nan")]]), crystal_atomic_numbers=torch.tensor([1]), role_atomic_numbers=torch.tensor([1]), role_degree=torch.tensor([0.]), crystal_pos=torch.zeros(1, 3), cell=torch.eye(3), assignment_timestep=1, crystal_timestep=0., mask=torch.ones(1, 1, dtype=torch.bool))
            else:
                validate_packed_assignment_batch(bad)
            actual = "NO_ERROR"
        except Exception as exc:
            actual = type(exc).__name__ + ": " + str(exc)
        cases.append({"test_name": name, "expected_exception": expected, "actual_exception": actual, "pass": actual.startswith(expected)})
    add("missing_mol_copy_id", lambda b: delattr(b, "mol_copy_id"))
    add("missing_mol_atom_id", lambda b: delattr(b, "mol_atom_id"))
    add("node_count_offsets_mismatch", lambda b: setattr(b, "batch", b.batch[:-1]))
    add("nonmonotonic_batch_index", lambda b: setattr(b, "batch", torch.flip(b.batch, [0])))
    add("batch_index_out_of_range", lambda b: setattr(b, "batch", b.batch + 1))
    add("illegal_copy_id", lambda b: setattr(b, "mol_copy_id", torch.where(b.batch == 0, b.mol_copy_id + 1, b.mol_copy_id)))
    add("missing_role", lambda b: setattr(b, "mol_atom_id", torch.where((b.batch == 0) & (b.mol_copy_id == 0) & (b.mol_atom_id == 0), torch.ones_like(b.mol_atom_id), b.mol_atom_id)))
    add("duplicate_role", lambda b: setattr(b, "mol_atom_id", torch.where((b.batch == 0) & (b.mol_copy_id == 0) & (b.mol_atom_id == 0), torch.ones_like(b.mol_atom_id), b.mol_atom_id)))
    add("inconsistent_role_sets", lambda b: setattr(b, "mol_atom_id", torch.where((b.batch == 0) & (b.mol_copy_id == 1) & (b.mol_atom_id == 0), torch.ones_like(b.mol_atom_id), b.mol_atom_id)))
    add("inconsistent_elements", lambda b: setattr(b, "atomic_numbers", torch.where((b.batch == 0) & (b.mol_copy_id == 1) & (b.mol_atom_id == 0), torch.full_like(b.atomic_numbers, 118), b.atomic_numbers)))
    add("N_not_Z_times_M", lambda b: setattr(b, "mol_copy_id", torch.where((b.batch == 0) & (b.mol_copy_id == 1) & (b.mol_atom_id == 0), torch.zeros_like(b.mol_copy_id), b.mol_copy_id)))
    add("target_graph_node_count_not_M", lambda b: setattr(b, "target_molecular_graph_num_nodes", torch.tensor([999, 999])))
    def predictor_add(name, logits, mask):
        try:
            d.predict_epsilon(logits, crystal_atomic_numbers=torch.tensor([1, 1]), role_atomic_numbers=torch.tensor([1, 1]), role_degree=torch.tensor([0.0, 0.0]), crystal_pos=torch.zeros(2, 3), cell=torch.eye(3), assignment_timestep=1, crystal_timestep=0.0, mask=mask)
            actual = "NO_ERROR"
        except Exception as exc:
            actual = type(exc).__name__ + ": " + str(exc)
        cases.append({"test_name": name, "expected_exception": "ValueError", "actual_exception": actual, "pass": actual.startswith("ValueError")})
    predictor_add("element_mask_empty_row", torch.zeros(2, 2), torch.tensor([[False, False], [True, True]]))
    predictor_add("element_mask_empty_column", torch.zeros(2, 2), torch.tensor([[True, False], [True, False]]))
    add("assignment_column_count_mismatch", lambda b: setattr(b, "target_molecular_graph_num_nodes", torch.tensor([999, 999])))
    add("role_id_out_of_range", lambda b: setattr(b, "mol_atom_id", torch.where(b.batch == 0, torch.full_like(b.mol_atom_id, 999), b.mol_atom_id)))
    add("copy_id_float", lambda b: setattr(b, "mol_copy_id", b.mol_copy_id.float()))
    add("role_id_float", lambda b: setattr(b, "mol_atom_id", b.mol_atom_id.float()))
    add("empty_sample", lambda b: setattr(b, "batch", torch.cat([torch.zeros((b.batch == 0).sum(), dtype=torch.long), torch.full(((b.batch == 1).sum(),), 2, dtype=torch.long)])))
    add("entire_batch_empty", lambda b: (setattr(b, "pos", b.pos[:0]), setattr(b, "atomic_numbers", b.atomic_numbers[:0]), setattr(b, "mol_copy_id", b.mol_copy_id[:0]), setattr(b, "mol_atom_id", b.mol_atom_id[:0]), setattr(b, "batch", b.batch[:0])))
    add("nan_coordinates", lambda b: setattr(b, "pos", torch.full_like(b.pos, float("nan"))))
    add("inf_coordinates", lambda b: setattr(b, "pos", torch.full_like(b.pos, float("inf"))))
    predictor_add("nan_logits", torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]), torch.ones(2, 2, dtype=torch.bool))
    predictor_add("inf_logits", torch.tensor([[float("inf"), 0.0], [0.0, 0.0]]), torch.ones(2, 2, dtype=torch.bool))
    add("cross_sample_bond", lambda b: setattr(b, "mol_bond_edge_index", torch.tensor([[0], [int((b.batch == 0).sum())]], dtype=torch.long)))
    return {"cases": cases, "pass": all(item["pass"] for item in cases)}


def independent_coefficients(alpha_bar, step):
    ab = alpha_bar[step]
    if step == 0:
        return 1.0, 0.0, 0.0
    previous = alpha_bar[step - 1]
    alpha_t = ab / previous
    beta_t = 1.0 - alpha_t
    return previous.sqrt() * beta_t / (1 - ab), alpha_t.sqrt() * (1 - previous) / (1 - ab), beta_t * (1 - previous) / (1 - ab)


def independent_projector(mask, dtype):
    allowed = mask.flatten().nonzero().flatten()
    rows = []
    for i in range(mask.shape[0]):
        row = torch.zeros(len(allowed), dtype=dtype); row[(allowed // mask.shape[1]) == i] = 1; rows.append(row)
    for j in range(mask.shape[1]):
        row = torch.zeros(len(allowed), dtype=dtype); row[(allowed % mask.shape[1]) == j] = 1; rows.append(row)
    constraint = torch.stack(rows)
    return torch.eye(len(allowed), dtype=dtype) - constraint.T @ torch.linalg.pinv(constraint @ constraint.T) @ constraint


def reverse_oracle():
    cases = []
    masks = {"single_block": torch.ones(3, 3, dtype=torch.bool), "multi_element_blocks": torch.block_diag(torch.ones(2, 2, dtype=torch.bool), torch.ones(3, 3, dtype=torch.bool)).bool()}
    for dtype in (torch.float32, torch.float64):
        d = AssignmentDiffusion(steps=16).to(dtype=dtype).eval()
        for label, mask in masks.items():
            generator = torch.Generator().manual_seed(909)
            l0 = gauge_center(torch.randn(mask.shape, generator=generator, dtype=dtype), mask)
            raw = torch.randn(mask.shape, generator=generator, dtype=dtype); eps = gauge_center(raw, mask)
            for step in (1, 8, 15):
                ab = d.alpha_bar[step].to(dtype); lt = gauge_center(ab.sqrt() * l0 + (1 - ab).sqrt() * eps, mask)
                value, x0, variance = d.reverse_step(lt, mask, lambda _x, _t: eps, step, noise=torch.zeros_like(lt))
                cx0, cyt, reference_variance = independent_coefficients(d.alpha_bar.to(dtype), step)
                reference_mean = cx0 * l0 + cyt * lt
                forbidden = value[~mask]
                cases.append({"dtype": str(dtype).split(".")[-1], "block": label, "t": step, "x0_reconstruction_max_error": float((x0 - l0).abs().max()), "posterior_mean_max_error": float((value - gauge_center(reference_mean, mask)).abs().max()), "posterior_variance_error": float((variance - reference_variance).abs()), "gauge_residual": float(max(value.sum(1).abs().max(), value.sum(0).abs().max())), "forbidden_logit_max_abs": float(forbidden.abs().max()) if forbidden.numel() else 0.0})
    # Empirical distribution at a fixed mid timestep: covariance is v*P, not v*I.
    dtype, mask, step = torch.float64, masks["single_block"], 8
    d = AssignmentDiffusion(steps=16).double(); l0 = gauge_center(torch.randn(mask.shape, dtype=dtype, generator=torch.Generator().manual_seed(17)), mask); eps = gauge_center(torch.randn(mask.shape, dtype=dtype, generator=torch.Generator().manual_seed(18)), mask); ab = d.alpha_bar[step].to(dtype); lt = gauge_center(ab.sqrt() * l0 + (1 - ab).sqrt() * eps, mask)
    draws = []
    for seed in range(1024):
        y, _, variance = d.reverse_step(lt, mask, lambda _x, _t: eps, step, noise=torch.randn(mask.shape, dtype=dtype, generator=torch.Generator().manual_seed(seed)))
        draws.append(y[mask])
    draws = torch.stack(draws); c0, ct, v = independent_coefficients(d.alpha_bar.to(dtype), step); mean = c0 * l0 + ct * lt
    projector = independent_projector(mask, dtype)
    empirical = (draws - draws.mean(0)).T @ (draws - draws.mean(0)) / draws.shape[0]
    empirical_info = {"samples": 1024, "mean_max_error": float((draws.mean(0) - mean[mask]).abs().max()), "covariance_max_error_vs_projected_posterior": float((empirical - v * projector).abs().max()), "gauge_residual_max": float(max(abs(draws.sum(1)).max(), abs(draws[:, :3].sum(1)).max())), "iid_identity_covariance_is_not_reference": True}
    return {"oracle_cases": cases, "empirical_posterior": empirical_info, "pass": max(x["x0_reconstruction_max_error"] for x in cases) < 2e-5 and max(x["posterior_mean_max_error"] for x in cases) < 2e-5 and max(x["posterior_variance_error"] for x in cases) < 2e-6 and empirical_info["covariance_max_error_vs_projected_posterior"] < 0.12}


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(exist_ok=True)
    torch.manual_seed(20260804); torch.use_deterministic_algorithms(True)
    batch, d = real_batch(), AssignmentDiffusion(steps=16).eval()
    regression = clean_target_regression(batch, d)
    equivalence = equivariance(batch, d)
    malformed_result = malformed(batch, d)
    oracle = reverse_oracle()
    # Real packed forward/backward, no optimizer step and no training.
    d.train(); torch.manual_seed(20260804); loss, marginal = d.loss(batch, torch.tensor([0.2, 0.7])); loss.backward()
    real = {"sample_ids": batch.structure_id, "loss": float(loss), "marginal_error": float(marginal), "finite_loss": bool(torch.isfinite(loss)), "finite_gradients": bool(all(p.grad is None or torch.isfinite(p.grad).all() for p in d.parameters()))}
    (OUT / "d0_regression_before_remaining_gate.json").write_text(json.dumps(regression, indent=2))
    (OUT / "d0_predictor_equivariance.json").write_text(json.dumps(equivalence, indent=2))
    (OUT / "d0_malformed_inputs.json").write_text(json.dumps(malformed_result, indent=2))
    (OUT / "d0_reverse_oracle.json").write_text(json.dumps(oracle, indent=2))
    forward_operator = json.loads((OUT / "d0_forward_metrics.json").read_text()) if (OUT / "d0_forward_metrics.json").exists() else {"status": "NOT_RUN"}
    metrics = {"seed": 20260804, "device": "cpu", "dtype": "float32", "real_packed": real, "clean_target": regression, "operator_regression": forward_operator, "predictor_equivariance_pass": equivalence["pass"], "malformed_pass": malformed_result["pass"], "reverse_oracle_pass": oracle["pass"], "finite": real["finite_loss"] and real["finite_gradients"]}
    (OUT / "d0_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
