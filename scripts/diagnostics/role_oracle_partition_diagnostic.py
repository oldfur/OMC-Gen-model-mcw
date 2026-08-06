"""Readily reproducible, fixed-RHODIN01 oracle copy-context diagnostic.

This is intentionally a one-step structured R experiment, not a sampler and
not a Q->C experiment.  Ground-truth copy information is represented only as
an N-by-N equality relation for the two oracle modes.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mattergen.common.role_partition_diffusion import OraclePartitionRoleDiagnostic, capacity_sinkhorn
from mattergen.common.role_partition_diffusion.swap_gibbs import SwapGibbsRoleDiffusion, assert_legal

OUT = Path("outputs/assignment_diffusion_mvp/role_oracle_partition_diagnostic")
SAMPLE = Path("outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt")
AUT = Path("outputs/assignment_diffusion_mvp/role_automorphism_audit/molecular_automorphisms.json")
SEED = 17
STEPS = int(os.environ.get("ORACLE_PARTITION_MAX_STEPS", "5000"))
MODES = ("geometry_only", "oracle_same_copy", "oracle_copy_local")
HIDDEN = 256
LAYERS = 4
LR = 2e-4
WEIGHT_DECAY = 1e-2


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load() -> dict:
    raw = torch.load(SAMPLE, map_location="cpu", weights_only=False)
    return {key: (value.cuda() if isinstance(value, torch.Tensor) else value) for key, value in raw.items()}


def terminal_states(s: dict) -> list[torch.Tensor]:
    diffusion = SwapGibbsRoleDiffusion(steps=64, terminal_randomization_steps=128)
    return [diffusion.terminal_prior(s["role"], s["z"], torch.Generator(device="cuda").manual_seed(1000 + n)) for n in range(32)]


def partition_relation(s: dict, kind: str) -> torch.Tensor | None:
    n = int(s["N"])
    if kind == "correct":
        return s["copy"][:, None].eq(s["copy"][None, :])
    if kind == "zero":
        return torch.zeros((n, n), dtype=torch.bool, device="cuda")
    if kind == "shuffled":
        # A valid equal-size partition built from a fixed permutation.  It is a
        # relation only: the labels below are discarded immediately.
        generator = torch.Generator(device="cuda").manual_seed(SEED + 991)
        order = torch.randperm(n, device="cuda", generator=generator)
        group = torch.empty(n, dtype=torch.long, device="cuda")
        group[order] = torch.arange(n, device="cuda") // int(s["M"])
        relation = group[:, None].eq(group[None, :])
        if torch.equal(relation, partition_relation(s, "correct")):
            raise RuntimeError("shuffled partition unexpectedly equals C0")
        return relation
    raise ValueError(kind)


def geometry(s: dict, kind: str) -> torch.Tensor:
    if kind == "correct":
        return s["pos"]
    if kind == "zero":
        return torch.zeros_like(s["pos"])
    if kind == "mismatch":
        generator = torch.Generator(device="cuda").manual_seed(SEED + 313)
        return s["pos"][torch.randperm(int(s["N"]), device="cuda", generator=generator)]
    raise ValueError(kind)


def relation_for(mode: str, relation: torch.Tensor | None) -> torch.Tensor | None:
    return relation if mode != "geometry_only" else None


def logits(model: OraclePartitionRoleDiagnostic, s: dict, state: torch.Tensor, *, geometry_kind="correct", partition_kind="correct") -> torch.Tensor:
    return model(
        z=s["z"], frac=geometry(s, geometry_kind), cell=s["cell"],
        role_z=s["role_z"], role_edge_index=s["role_edge_index"], role_bond_type=s["role_bond_type"],
        current_role=state, same_copy=relation_for(model.context_mode, partition_relation(s, partition_kind)),
    )


def hungarian_capacity(scores: torch.Tensor, s: dict) -> torch.Tensor:
    """Independent hard decoder, one Hungarian assignment per allowed element block."""
    result = torch.empty(int(s["N"]), dtype=torch.long, device=scores.device)
    for element in s["z"].unique(sorted=True):
        atoms = (s["z"] == element).nonzero().flatten()
        roles = (s["role_z"] == element).nonzero().flatten()
        slots = roles.repeat_interleave(int(s["Z"]))
        cost = -scores[atoms][:, slots].detach().float().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        if len(row) != len(atoms):
            raise RuntimeError("Hungarian did not return a complete element-block matching")
        result[atoms[torch.as_tensor(row, device=scores.device)]] = slots[torch.as_tensor(col, device=scores.device)]
    assert_legal(result, s["z"], s["role_z"], int(s["Z"]))
    return result


def automorphism_metrics(pred: torch.Tensor, s: dict, perms: list[list[int]]) -> dict:
    truth, copy = s["role"], s["copy"]
    aligned = [torch.as_tensor(perm, device="cuda", dtype=torch.long)[truth] for perm in perms]
    accuracy = torch.stack([(pred == item).float().mean() for item in aligned])
    best = int(accuracy.argmax())
    per_copy_total = 0.0
    per_copy_exact = True
    for c in range(int(s["Z"])):
        idx = (copy == c).nonzero().flatten()
        values = torch.stack([(pred[idx] == item[idx]).float().mean() for item in aligned])
        per_copy_total += float(values.max()) * len(idx)
        per_copy_exact &= bool(values.max() == 1)
    per_role = {str(r): float((pred[truth == r] == r).float().mean()) for r in range(int(s["M"]))}
    pair = torch.isin(truth, torch.tensor([1, 2], device="cuda"))
    return {
        "literal_accuracy": float((pred == truth).float().mean()),
        "global_aligned_accuracy": float(accuracy.max()),
        "literal_exact": bool(torch.equal(pred, truth)),
        "global_exact_orbit": bool(torch.equal(pred, aligned[best])),
        "per_copy_aligned_accuracy_diagnostic": per_copy_total / int(s["N"]),
        "per_copy_exact_orbit_diagnostic": per_copy_exact,
        "capacity_valid": True,
        "element_valid": bool(torch.equal(s["z"], s["role_z"][pred])),
        "roles_1_2_literal_accuracy": float((pred[pair] == truth[pair]).float().mean()),
        "singleton_roles_literal_accuracy": float((pred[~pair] == truth[~pair]).float().mean()),
        "per_role_literal_accuracy": per_role,
    }


@torch.no_grad()
def evaluate(model: OraclePartitionRoleDiagnostic, s: dict, states: list[torch.Tensor], perms: list[list[int]], *, geometry_kind="correct", partition_kind="correct") -> dict:
    start = time.perf_counter(); rows = []
    for state in states:
        pred = hungarian_capacity(logits(model, s, state, geometry_kind=geometry_kind, partition_kind=partition_kind), s)
        rows.append(automorphism_metrics(pred, s, perms))
    numeric = ("literal_accuracy", "global_aligned_accuracy", "per_copy_aligned_accuracy_diagnostic", "roles_1_2_literal_accuracy", "singleton_roles_literal_accuracy")
    out = {key: sum(float(row[key]) for row in rows) / len(rows) for key in numeric}
    out.update({
        "evaluations": len(rows), "exact_R": sum(row["literal_exact"] for row in rows),
        "global_exact_orbit": sum(row["global_exact_orbit"] for row in rows),
        "per_copy_exact_orbit_diagnostic": sum(row["per_copy_exact_orbit_diagnostic"] for row in rows),
        "capacity_valid": sum(row["capacity_valid"] for row in rows), "element_valid": sum(row["element_valid"] for row in rows),
        "per_role_literal_accuracy": {str(role): sum(row["per_role_literal_accuracy"][str(role)] for row in rows) / len(rows) for role in range(int(s["M"]))},
        "wall_seconds": time.perf_counter() - start,
        "geometry_kind": geometry_kind, "partition_kind": partition_kind,
    })
    return out


def train(mode: str, s: dict, states: list[torch.Tensor], perms: list[list[int]]) -> tuple[OraclePartitionRoleDiagnostic, dict]:
    seed_all(SEED)
    model = OraclePartitionRoleDiagnostic(context_mode=mode, hidden=HIDDEN, layers=LAYERS).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    checkpoint_dir = OUT / "checkpoints" / mode; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "step": 0, "mode": mode}, checkpoint_dir / "step_0.pt")
    # The same deterministic terminal-state sequence is used by every context mode.
    order = torch.randint(0, len(states), (STEPS,), generator=torch.Generator().manual_seed(SEED + 73)).tolist()
    curve, nonfinite, best = [], 0, (-1.0, -1, None)
    started = time.perf_counter()
    for step, state_index in enumerate(order, start=1):
        score = logits(model, s, states[state_index])
        soft = capacity_sinkhorn(score, s["role_z"], s["z"], int(s["Z"]))
        loss = -soft[torch.arange(int(s["N"]), device="cuda"), s["role"]].clamp_min(1e-30).log().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Sinkhorn loss in {mode} at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite gradient in {mode} at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 100 == 0 or step == STEPS:
            model.eval(); metric = evaluate(model, s, states, perms); model.train()
            record = {"step": step, "loss": float(loss), "validation": metric}; curve.append(record)
            key = (metric["literal_accuracy"], metric["exact_R"])
            if key > best[:2]:
                best = (*key, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"state_dict": final_state, "step": STEPS, "mode": mode}, checkpoint_dir / "final.pt")
    model.load_state_dict(best[2]); torch.save({"state_dict": model.state_dict(), "step": "best_by_literal_accuracy_then_exact", "mode": mode}, checkpoint_dir / "best.pt")
    model.eval(); final = evaluate(model, s, states, perms)
    return model, {"mode": mode, "steps": STEPS, "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY}, "hidden": HIDDEN, "layers": LAYERS, "nonfinite": nonfinite, "wall_seconds": time.perf_counter() - started, "curve": curve, "best_final": final}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(exist_ok=True)
    seed_all(SEED); s = load(); states = terminal_states(s)
    perms = json.loads(AUT.read_text())["permutations"]
    if (int(s["N"]), int(s["M"]), int(s["Z"]), s["id"], s["split"]) != (40, 10, 4, "RHODIN01|4|gener|9b810e76ec9d286", "val"):
        raise ValueError("fixed-sample identity mismatch")
    for state in states: assert_legal(state, s["z"], s["role_z"], int(s["Z"]))
    audit = {
        "sample": {"id": s["id"], "split": s["split"], "N": int(s["N"]), "M": int(s["M"]), "K": int(s["Z"])},
        "seed": SEED, "modes": list(MODES), "shared": {"hidden": HIDDEN, "layers": LAYERS, "steps": STEPS, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "terminal_states": [1000 + i for i in range(32)]},
        "oracle_boundary": {"geometry_only": "no mol_copy_id or C0", "oracle_same_copy": "C0 only as same/inter pair relation; no numeric copy ID/embedding", "oracle_copy_local": "C0 only gates local crystal message edges; no numeric copy ID/embedding"},
        "decoder": "element-block capacity Sinkhorn during training; independent element-block Hungarian MAP during evaluation",
    }
    (OUT / "config_audit.json").write_text(json.dumps(audit, indent=2))
    trained, outputs = {}, {}
    for mode in MODES:
        model, training = train(mode, s, states, perms); trained[mode] = model; outputs[mode] = training
        (OUT / f"{mode}_metrics.json").write_text(json.dumps(training, indent=2))
    # Required named metric files keep the mode names obvious to downstream readers.
    (OUT / "geometry_only_metrics.json").write_text(json.dumps(outputs["geometry_only"], indent=2))
    (OUT / "oracle_same_copy_metrics.json").write_text(json.dumps(outputs["oracle_same_copy"], indent=2))
    (OUT / "oracle_copy_local_metrics.json").write_text(json.dumps(outputs["oracle_copy_local"], indent=2))
    same = trained["oracle_same_copy"]
    partition_ablation = {kind: evaluate(same, s, states, perms, partition_kind=kind) for kind in ("correct", "shuffled", "zero")}
    (OUT / "shuffled_partition_ablation.json").write_text(json.dumps(partition_ablation, indent=2))
    geometry_ablation = {mode: {kind: evaluate(model, s, states, perms, geometry_kind=kind) for kind in ("correct", "zero", "mismatch")} for mode, model in trained.items()}
    per_role = {mode: outputs[mode]["best_final"]["per_role_literal_accuracy"] for mode in MODES}
    comparison = {
        "modes": {mode: outputs[mode]["best_final"] for mode in MODES},
        "delta_samecopy_literal_accuracy": outputs["oracle_same_copy"]["best_final"]["literal_accuracy"] - outputs["geometry_only"]["best_final"]["literal_accuracy"],
        "delta_copylocal_literal_accuracy": outputs["oracle_copy_local"]["best_final"]["literal_accuracy"] - outputs["geometry_only"]["best_final"]["literal_accuracy"],
        "geometry_ablation": geometry_ablation,
    }
    (OUT / "per_role_metrics.json").write_text(json.dumps(per_role, indent=2)); (OUT / "condition_comparison.json").write_text(json.dumps(comparison, indent=2))
    oracle_success = any(x["best_final"]["exact_R"] >= 28 and x["best_final"]["literal_accuracy"] >= .99 for x in outputs.values())
    context_separation = partition_ablation["correct"]["literal_accuracy"] > partition_ablation["shuffled"]["literal_accuracy"] and partition_ablation["correct"]["literal_accuracy"] > partition_ablation["zero"]["literal_accuracy"]
    if oracle_success and context_separation:
        decision = "A_copy_context_key_missing_variable"
        next_step = "diagnostic evidence supports joint/alternating R,Q inference; not implemented here"
    elif comparison["delta_copylocal_literal_accuracy"] > comparison["delta_samecopy_literal_accuracy"] + .05:
        decision = "C_copylocal_only_effective"
        next_step = "intermolecular-neighbor contamination is the leading diagnostic; retain STOP"
    else:
        decision = "B_copy_context_limited"
        next_step = "prefer global element-block structured assignment and richer relational geometry/molecule encoding"
    report = f"""# Oracle copy-partition identifiability diagnostic\n\nFixed sample: `{s['id']}` ({s['split']}), N=40, M=10, K=4, seed=17. This is a fixed-sample oracle diagnostic only: it is not a deployable generator and it does not run Q→C.\n\nDecision: **{decision}**. {next_step}\n\nGeometry-only literal accuracy: {outputs['geometry_only']['best_final']['literal_accuracy']:.6f}; oracle same-copy: {outputs['oracle_same_copy']['best_final']['literal_accuracy']:.6f}; oracle copy-local: {outputs['oracle_copy_local']['best_final']['literal_accuracy']:.6f}. Deltas vs geometry-only: same-copy {comparison['delta_samecopy_literal_accuracy']:.6f}, copy-local {comparison['delta_copylocal_literal_accuracy']:.6f}.\n\nSame-copy C ablation (literal): correct {partition_ablation['correct']['literal_accuracy']:.6f}, shuffled {partition_ablation['shuffled']['literal_accuracy']:.6f}, zero {partition_ablation['zero']['literal_accuracy']:.6f}.\n\nThe two oracle modes receive only a permutation-invariant C equality relation, never `mol_copy_id` values or embeddings. All evaluation decodes use element-block Hungarian matching and assert role capacity/element validity.\n\nSTOP remains in force: no Q→C formal generation, three-seed study, or D2.\n"""
    (OUT / "identifiability_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
