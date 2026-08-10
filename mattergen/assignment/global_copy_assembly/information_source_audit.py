"""Frozen-checkpoint information-source / ordering-leakage audit for O2.

NO TRAINING.  All decode paths run under ``torch.no_grad()`` with ``model.eval()``.
``mol_copy_id`` / ``C0`` are metrics-only (never model inputs).
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .orbit_attachment import (
    attachment_map_margins,
    balanced_attachment_dp,
)
from .orbit_membership import collapse_roles_to_orbit_membership
from .orbit_metrics import evaluate_orbit_assembly
from .orbit_module import OrbitAwareCopyAssembly
from .orbit_targets import (
    OrbitAttachmentTarget,
    OrbitAwareAssemblyTarget,
    build_orbit_aware_target,
)
from .singleton_backbone import SingletonBackbone
from .targets import AssemblyTarget, target_state_indices
from .tree_crf import TreeCRF
from .permutations import enumerate_permutations, identity_index


ATOM_PERM_SEEDS = (0, 1, 2, 3, 4, 17, 42, 123)
SHUFFLE_SEEDS = (0, 1, 2, 3, 4, 17, 42, 123)
NEAR_TIE_TOLS = (1e-8, 1e-6)


def freeze_model(model: OrbitAwareCopyAssembly) -> OrbitAwareCopyAssembly:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.buffers():
        # buffers stay as-is; no training updates
        pass
    return model


def assert_no_oracle_inputs(model: OrbitAwareCopyAssembly) -> dict[str, bool]:
    cfg = model.config
    if cfg.use_copy_id_as_input or cfg.use_oracle_copy_relation:
        raise RuntimeError("oracle copy input flags must be false for audit")
    return {
        "use_copy_id_as_input": bool(cfg.use_copy_id_as_input),
        "use_oracle_copy_relation": bool(cfg.use_oracle_copy_relation),
        "use_oracle_role_assignment": bool(cfg.use_oracle_role_assignment),
        "oracle_C_as_input": False,
    }


def geometry_bundle(
    sample: dict,
    mode: str,
    *,
    mismatch_sample: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Build geometry tensors for a mode. Always reconstructs from provided frac/cell.

    Modes
    -----
    correct_geometry:
        original frac + cell
    legacy_zero_geometry:
        zeros frac, **retains original cell** (matches evaluate_global_copy_assembly_orbit_o2)
    strict_zero_geometry:
        zeros frac + identity cell (no original lattice leakage)
    mismatched_geometry:
        mismatch sample pos + cell (same N required)
    """
    pos = sample["pos"]
    cell = sample["cell"]
    if mode == "correct_geometry":
        frac, use_cell = pos, cell
        note = "original fractional coords + original cell; crystal encoder rebuilds edges from (frac,cell)"
    elif mode == "legacy_zero_geometry" or mode == "zero_geometry":
        frac, use_cell = torch.zeros_like(pos), cell
        note = (
            "LEGACY: frac=0 but cell retained; ContextCrystalEncoder rebuilds neighbor list "
            "from zero frac with original lattice — not a cache of correct edges, but lattice leaks"
        )
    elif mode == "strict_zero_geometry":
        frac = torch.zeros_like(pos)
        use_cell = torch.eye(3, dtype=pos.dtype, device=pos.device)
        note = (
            "STRICT: frac=0 and cell=I_3; all geometric features rebuilt; no original lattice; "
            "neighbor graph from zero geometry only"
        )
    elif mode == "mismatched_geometry":
        if mismatch_sample is None:
            raise ValueError("mismatched_geometry requires mismatch_sample")
        if mismatch_sample["pos"].shape != pos.shape:
            raise ValueError("mismatch pos shape incompatible")
        frac = mismatch_sample["pos"].to(device=pos.device, dtype=pos.dtype)
        use_cell = mismatch_sample["cell"].to(device=cell.device, dtype=cell.dtype)
        note = "mismatch frac+cell; geometric features rebuilt from mismatch geometry"
    else:
        raise ValueError(f"unknown geometry mode {mode!r}")
    return {"frac": frac, "cell": use_cell, "geometry_mode": mode, "geometry_note": note}


def audit_legacy_vs_strict_zero_features(sample: dict) -> dict[str, object]:
    """Static/code-level audit of zero-geometry leakage (no model call)."""
    from mattergen.common.role_partition_diffusion.oracle_partition import periodic_edges

    pos = sample["pos"]
    cell = sample["cell"]
    zero = torch.zeros_like(pos)
    eye = torch.eye(3, dtype=pos.dtype)
    # Rebuild edges from each geometry (same API as ContextCrystalEncoder)
    e_corr, d_corr = periodic_edges(pos, cell, cutoff=6.0)
    e_leg, d_leg = periodic_edges(zero, cell, cutoff=6.0)
    e_str, d_str = periodic_edges(zero, eye, cutoff=6.0)
    # correct edges should not equal legacy zero edges in general
    legacy_reuses_correct_edges = bool(torch.equal(e_corr, e_leg))
    strict_reuses_correct_edges = bool(torch.equal(e_corr, e_str))
    classification = []
    if legacy_reuses_correct_edges:
        classification.append("GEOMETRY_LEAKAGE_IN_LEGACY_ZERO_ABLATION")
    # lattice leak even if edges differ
    lattice_retained_in_legacy = True
    return {
        "crystal_encoder_rebuilds_edges_each_forward": True,
        "cached_pair_scores_in_o2_forward": False,
        "legacy_zero_geometry": {
            "frac": "zeros_like(pos)",
            "cell": "original sample cell RETAINED",
            "edge_equal_to_correct": legacy_reuses_correct_edges,
            "num_edges_correct": int(e_corr.sum().item()),
            "num_edges_legacy_zero": int(e_leg.sum().item()),
            "mean_distance_legacy": float(d_leg[e_leg].mean()) if e_leg.any() else None,
            "lattice_retained": lattice_retained_in_legacy,
        },
        "strict_zero_geometry": {
            "frac": "zeros_like(pos)",
            "cell": "identity I_3",
            "edge_equal_to_correct": strict_reuses_correct_edges,
            "num_edges_strict_zero": int(e_str.sum().item()),
            "mean_distance_strict": float(d_str[e_str].mean()) if e_str.any() else None,
        },
        "geometry_derived_quantities_rebuilt_each_forward": [
            "frac coordinates (mode-dependent)",
            "cell/lattice (mode-dependent)",
            "edge_index / neighbor list via periodic_edges(frac,cell)",
            "PBC displacements and distances",
            "distance RBF",
            "crystal node embeddings (no cache across calls)",
            "pair potential RBF from instance positions",
            "orbit attachment PBC distances",
        ],
        "not_geometry_but_structural": [
            "role_assignment / bar_R orbit membership",
            "molecular role_edge_index / role_bond_type",
            "singleton backbone molecular tree (topology)",
            "z / role_z element tables",
        ],
        "classification_hints": classification,
    }


def permute_atom_indices(
    sample: dict,
    role_assignment: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Apply atom permutation σ to all atom-indexed quantities. Returns (sample', roles', inv)."""
    n = int(sample["N"])
    if sigma.numel() != n:
        raise ValueError("sigma length must equal N")
    inv = torch.empty_like(sigma)
    inv[sigma] = torch.arange(n, device=sigma.device, dtype=sigma.dtype)
    out = dict(sample)
    for key in ("pos", "z", "role", "copy"):
        if key in sample and torch.is_tensor(sample[key]) and sample[key].shape[0] == n:
            out[key] = sample[key][sigma]
    # role_assignment is crystal-atom labels
    roles_p = role_assignment[sigma]
    return out, roles_p, inv


def shuffle_singleton_role_instance_order(
    o2_target: OrbitAwareAssemblyTarget,
    *,
    seed: int,
) -> OrbitAwareAssemblyTarget:
    """Independently permute instance order q within each singleton role set.

    Physical atoms and C0 unchanged; only enumeration gauge of V_r changes.
    Permutations P* are rewritten so supervision still matches true copies.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    st = o2_target.singleton_target
    K = st.K
    new_sets: dict[int, torch.Tensor] = {}
    new_perms: dict[int, torch.Tensor] = {}
    for role, nodes in st.role_sets.items():
        tau = torch.randperm(K, generator=g)
        new_nodes = nodes[tau]
        # old: nodes[q] -> perm[q]; new: new_nodes[q']=nodes[tau[q']] so P'[q']=P[tau[q']]
        old_p = st.permutations[role]
        new_p = old_p[tau]
        new_sets[role] = new_nodes
        new_perms[role] = new_p
    # re-establish anchor identity: if anchor perm is not identity, this can happen
    # when tau is not identity — P_anchor must be identity by convention.
    # After shuffle, sorted order is gone; gauge labels q are still copy labels via P.
    # Anchor convention P_anchor[q]=q requires anchor nodes ordered by copy label.
    # Rebuild anchor order by sorting instances by their copy label P.
    anchor = st.anchor_role
    # Map each node to copy label under new_p
    # Ensure anchor P is identity by reordering anchor nodes by increasing copy label.
    a_nodes = new_sets[anchor]
    a_p = new_perms[anchor]
    order = torch.argsort(a_p)
    a_nodes = a_nodes[order]
    # After ordering by copy label, P should be 0..K-1 if bijective
    a_p_sorted = a_p[order]
    if not torch.equal(a_p_sorted, torch.arange(K, device=a_p.device)):
        # remap all perms through inverse of a_p_sorted if needed
        inv = torch.empty_like(a_p_sorted)
        inv[a_p_sorted] = torch.arange(K, device=a_p.device)
        for role in new_perms:
            new_perms[role] = inv[new_perms[role]]
        a_p_sorted = inv[a_p_sorted]
    new_sets[anchor] = a_nodes
    new_perms[anchor] = a_p_sorted
    new_st = AssemblyTarget(
        role_sets=new_sets,
        permutations=new_perms,
        anchor_role=anchor,
        K=K,
        M=st.M,
    )
    return OrbitAwareAssemblyTarget(
        partition=o2_target.partition,
        bar_r=o2_target.bar_r,
        singleton_roles=o2_target.singleton_roles,
        singleton_target=new_st,
        local_to_role=o2_target.local_to_role,
        role_to_local=o2_target.role_to_local,
        orbit_targets=o2_target.orbit_targets,
        K=o2_target.K,
        N=o2_target.N,
    )


def shuffle_orbit_candidate_order(
    o2_target: OrbitAwareAssemblyTarget,
    *,
    seed: int,
) -> OrbitAwareAssemblyTarget:
    """Randomly reorder V_12 candidate list; rewrite pairs_local accordingly."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    new_orbits = []
    for ot in o2_target.orbit_targets:
        n = int(ot.atom_indices.numel())
        tau = torch.randperm(n, generator=g)
        inv = torch.empty_like(tau)
        inv[tau] = torch.arange(n, device=tau.device)
        new_atoms = ot.atom_indices[tau]
        new_pairs = []
        for i, j in ot.pairs_local:
            i2, j2 = int(inv[i]), int(inv[j])
            new_pairs.append((i2, j2) if i2 < j2 else (j2, i2))
        new_orbits.append(
            OrbitAttachmentTarget(
                orbit_index=ot.orbit_index,
                atom_indices=new_atoms,
                pairs_local=tuple(new_pairs),
                atoms_per_copy=ot.atoms_per_copy,
            )
        )
    return OrbitAwareAssemblyTarget(
        partition=o2_target.partition,
        bar_r=o2_target.bar_r,
        singleton_roles=o2_target.singleton_roles,
        singleton_target=o2_target.singleton_target,
        local_to_role=o2_target.local_to_role,
        role_to_local=o2_target.role_to_local,
        orbit_targets=tuple(new_orbits),
        K=o2_target.K,
        N=o2_target.N,
    )


@torch.no_grad()
def singleton_score_of_states(
    factors: dict[tuple[int, int], torch.Tensor],
    tree_edges: tuple[tuple[int, int], ...],
    state_indices: dict[int, int],
) -> torch.Tensor:
    ref = next(iter(factors.values()))
    return sum(
        (factors[e][state_indices[e[0]], state_indices[e[1]]] for e in tree_edges),
        start=ref.new_zeros(()),
    )


@torch.no_grad()
def singleton_map_margins(
    factors: dict[tuple[int, int], torch.Tensor],
    tree,
    num_states: int,
    identity_state: int,
    map_states: dict[int, int],
    target_states: dict[int, int] | None,
    *,
    near_tie_tol: float = 1e-6,
) -> dict[str, object]:
    """MAP score, one-role-flip second best, logZ, target score, near-tie count."""
    crf = TreeCRF(tree, num_states=num_states, identity_state=identity_state)
    log_z = float(crf.log_partition(factors).detach())
    best = float(singleton_score_of_states(factors, tree.tree_edges, map_states).detach())
    # Approximate second-best: flip any non-root role to any other state
    second = float("-inf")
    n_near = 0
    for role in tree.preorder:
        if role == tree.root:
            continue
        for s in range(num_states):
            if s == map_states[role]:
                continue
            alt = dict(map_states)
            alt[role] = s
            sc = float(singleton_score_of_states(factors, tree.tree_edges, alt).detach())
            if sc > second:
                second = sc
            if abs(sc - best) <= near_tie_tol:
                n_near += 1
    gap = best - second if second > float("-inf") else float("inf")
    out: dict[str, object] = {
        "best_score": best,
        "second_best_score": second,
        "map_gap": gap,
        "log_partition": log_z,
        "approx_second_best_method": "single_non_root_role_state_flip",
        "number_of_near_ties_vs_map_among_one_flips": n_near,
        "near_tie_tol": near_tie_tol,
    }
    if target_states is not None:
        tscore = float(singleton_score_of_states(factors, tree.tree_edges, target_states).detach())
        out["target_score"] = tscore
        out["target_log_probability"] = tscore - log_z
    # structured entropy upper bound from logZ scale is not exact without full Z table
    out["structured_entropy"] = None
    out["structured_entropy_note"] = "exact tree entropy not enumerated; logZ and target_log_probability reported"
    return out


@torch.no_grad()
def decode_o2(
    model: OrbitAwareCopyAssembly,
    o2_target: OrbitAwareAssemblyTarget,
    backbone: SingletonBackbone,
    sample: dict,
    *,
    geometry_mode: str = "correct_geometry",
    mismatch_sample: dict | None = None,
    attachment_pair_order: str = "default",
    attachment_tie_seed: int = 0,
    near_tie_tol: float = 0.0,
    collect_margins: bool = True,
) -> dict[str, object]:
    """Frozen MAP decode with optional margins / tie-break. Metrics use sample['copy']."""
    assert_no_oracle_inputs(model)
    geo = geometry_bundle(sample, geometry_mode, mismatch_sample=mismatch_sample)
    frac, cell = geo["frac"], geo["cell"]
    hx, hm = model.encode(
        z=sample["z"],
        frac=frac,
        cell=cell,
        role_z=sample["role_z"],
        role_edge_index=sample["role_edge_index"],
        role_bond_type=sample["role_bond_type"],
    )
    singleton_target = o2_target.singleton_target
    factors, states, scores = model.singleton_factors(
        singleton_target=singleton_target, backbone=backbone, hx=hx, hm=hm, frac=frac, cell=cell
    )
    crf = TreeCRF(backbone.tree, num_states=len(states), identity_state=identity_index(states))
    decoded = crf.map_decode(factors)
    N, K = o2_target.N, o2_target.K
    G = torch.zeros(N, K, dtype=torch.float32, device=hx.device)
    for role, nodes in singleton_target.role_sets.items():
        perm = states[decoded.state_indices[role]].long()
        G[nodes, perm] = 1.0
    singleton_mask = torch.zeros(N, dtype=torch.bool, device=hx.device)
    for nodes in singleton_target.role_sets.values():
        singleton_mask[nodes] = True

    target_states = target_state_indices(singleton_target, states)
    sing_margin = (
        singleton_map_margins(
            factors,
            backbone.tree,
            len(states),
            identity_index(states),
            decoded.state_indices,
            target_states,
            near_tie_tol=max(near_tie_tol, 1e-6),
        )
        if collect_margins
        else {}
    )

    attachment_maps: dict[int, object] = {}
    orbit_margins: dict[int, object] = {}
    for ot in o2_target.orbit_targets:
        F = model.build_attachment_scores_from_G(
            orbit_atoms=ot.atom_indices.to(hx.device),
            G_singleton=G,
            hx=hx,
            frac=frac,
            cell=cell,
            singleton_mask=singleton_mask,
            orbit_id=ot.orbit_index,
            mode="max",
        )
        result = balanced_attachment_dp(
            F,
            atoms_per_copy=ot.atoms_per_copy,
            pair_order=attachment_pair_order,
            tie_break_seed=attachment_tie_seed,
            near_tie_tol=near_tie_tol,
        )
        attachment_maps[ot.orbit_index] = {
            "map_pairs": result.map_pairs,
            "map_score": float(result.map_score.detach()),
        }
        atoms = ot.atom_indices.to(hx.device)
        for k, (i, j) in enumerate(result.map_pairs):
            G[int(atoms[i].item()), k] = 1.0
            G[int(atoms[j].item()), k] = 1.0
        if collect_margins:
            orbit_margins[ot.orbit_index] = attachment_map_margins(
                F,
                atoms_per_copy=ot.atoms_per_copy,
                near_tie_tol=max(near_tie_tol, 1e-6),
                target_pairs=list(ot.pairs_local),
            )

    if not torch.allclose(G.sum(-1), torch.ones(N, device=G.device)):
        raise AssertionError("each atom must have exactly one copy in G")
    C = G @ G.T
    metrics = evaluate_orbit_assembly(
        G=G,
        C=C,
        copy=sample["copy"],
        bar_r=o2_target.bar_r,
        partition=o2_target.partition,
        role_assignment_for_projection=sample["role"],
        role_edge_index=sample["role_edge_index"],
        role_bond_type=sample["role_bond_type"],
        M=int(sample["M"]),
    )
    return {
        "G": G,
        "C": C,
        "metrics": metrics,
        "geometry": geo,
        "singleton_tree_energy": float(decoded.score.detach()),
        "singleton_state_indices": decoded.state_indices,
        "singleton_margins": sing_margin,
        "orbit_attachments": attachment_maps,
        "orbit_margins": orbit_margins,
        "attachment_pair_order": attachment_pair_order,
        "status": "CLEAN_GEOMETRY_ORBIT_AWARE_O2_AUDIT",
    }


def unpermute_C(C_perm: torch.Tensor, inv_sigma: torch.Tensor) -> torch.Tensor:
    """Map C' on permuted atoms back to original indices: P^T C' P with P=sigma."""
    # atoms in C_perm are ordered by sigma positions; original i maps to position inv[i]?
    # sample' has atom at new index j = old sigma^{-1}? 
    # We set out[key] = sample[key][sigma], so new index j has old atom sigma[j].
    # C'[j,k] concerns old atoms sigma[j], sigma[k].
    # C_orig[a,b] = C'[inv[a], inv[b]] where inv[sigma[j]]=j.
    return C_perm[inv_sigma][:, inv_sigma]


def summarize_seed_metrics(rows: list[dict]) -> dict[str, object]:
    if not rows:
        return {"n": 0}
    exact = [1.0 if r.get("exact_C") else 0.0 for r in rows]
    f1 = [float(r.get("copy_pair_f1", 0.0)) for r in rows]
    gaps_s = [float(r["singleton_map_gap"]) for r in rows if r.get("singleton_map_gap") is not None]
    gaps_o = [float(r["orbit_map_gap"]) for r in rows if r.get("orbit_map_gap") is not None]

    def _stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        t = torch.tensor(xs, dtype=torch.float64)
        return {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)),
            "min": float(t.min()),
            "median": float(t.median()),
            "max": float(t.max()),
        }

    return {
        "n": len(rows),
        "exact_C_rate": sum(exact) / len(exact),
        "copy_pair_f1": _stats(f1),
        "singleton_map_gap": _stats(gaps_s),
        "orbit_map_gap": _stats(gaps_o),
    }


def classify_audit(summary: dict[str, Any]) -> list[str]:
    """Heuristic multi-label classification from audit tables (no pretrained claims)."""
    labels: list[str] = []
    atom = summary.get("atom_permutation", {})
    role = summary.get("role_instance_shuffle", {})
    zero_feat = summary.get("zero_geometry_feature_audit", {})
    margins = summary.get("map_margin_results", {})
    tie = summary.get("tie_break", {})

    # ordering leakage
    if atom.get("correct_exact_C_rate", 1.0) < 1.0 or atom.get("zero_exact_C_rate", 1.0) < 1.0:
        labels.append("ORDERING_LEAKAGE")
    if role.get("correct_exact_C_rate", 1.0) < 1.0 or role.get("zero_exact_C_rate", 1.0) < 1.0:
        if "ORDERING_LEAKAGE" not in labels:
            labels.append("ORDERING_LEAKAGE")

    hints = zero_feat.get("classification_hints") or []
    labels.extend(hints)

    # tie-breaking
    tb = tie.get("zero_changes_with_tie_break")
    if tb is True:
        labels.append("TIE_BREAKING_ARTIFACT")
    # near-zero gap on zero geometry
    zg = margins.get("strict_zero_geometry") or margins.get("legacy_zero_geometry") or {}
    if zg.get("orbit_MAP_gap") is not None and float(zg["orbit_MAP_gap"]) <= 1e-6:
        if "TIE_BREAKING_ARTIFACT" not in labels:
            labels.append("TIE_BREAKING_ARTIFACT")

    correct = summary.get("baseline", {}).get("correct_geometry", {})
    strict = summary.get("baseline", {}).get("strict_zero_geometry", {})
    legacy = summary.get("baseline", {}).get("legacy_zero_geometry", {})
    mismatch = summary.get("baseline", {}).get("mismatched_geometry", {})

    def _exact(d):
        return bool(d.get("exact_C")) if isinstance(d, dict) and "exact_C" in d else None

    c_ok = _exact(correct)
    s_ok = _exact(strict)
    l_ok = _exact(legacy)
    m_ok = _exact(mismatch) if isinstance(mismatch, dict) and mismatch.get("status") != "MISMATCH_SAMPLE_NOT_AVAILABLE" else None

    ordering_ok = (
        atom.get("correct_exact_C_rate", 0.0) == 1.0
        and role.get("correct_exact_C_rate", 0.0) == 1.0
        and "ORDERING_LEAKAGE" not in labels
    )
    if (
        ordering_ok
        and c_ok
        and s_ok
        and "GEOMETRY_LEAKAGE_IN_LEGACY_ZERO_ABLATION" not in labels
        and "TIE_BREAKING_ARTIFACT" not in labels
    ):
        labels.append("SINGLE_INSTANCE_MEMORIZATION")

    if (
        ordering_ok
        and c_ok
        and s_ok is False
        and (m_ok is False or m_ok is None)
        and float((margins.get("correct_geometry") or {}).get("orbit_MAP_gap") or 0.0)
        > float((margins.get("strict_zero_geometry") or {}).get("orbit_MAP_gap") or 0.0) + 1e-3
    ):
        labels.append("GEOMETRY_DEPENDENT_ASSEMBLY")

    if not labels:
        labels.append("INCONCLUSIVE")
    # de-dup preserve order
    seen = set()
    out = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# O2 Information-Source / Ordering-Leakage Audit",
        "",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Classifications: `{', '.join(summary.get('classifications', []))}`",
        f"- Oracle input flags: `{json.dumps(summary.get('oracle_input_audit', {}))}`",
        "",
        "## Condition table",
        "",
        "| condition | exact C | pair F1 | ARI | projected bond F1 | singleton MAP gap | orbit MAP gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    table = summary.get("condition_table") or []
    for row in table:
        lines.append(
            f"| {row.get('condition')} | {row.get('exact_C')} | {row.get('copy_pair_f1')} | "
            f"{row.get('ARI')} | {row.get('projected_bond_f1')} | {row.get('singleton_map_gap')} | "
            f"{row.get('orbit_map_gap')} |"
        )
    lines.extend(["", "## Notes", ""])
    for note in summary.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
