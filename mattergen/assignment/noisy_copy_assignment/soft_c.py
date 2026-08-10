"""Conditional-on-singleton-MAP structured soft C for N1.

This is **not** the full joint structured marginal over (singleton tree-CRF
× orbit attachment).  Soft same-copy probabilities are formed by:

1. fixing the singleton backbone to its hard MAP groups ``G_singleton``;
2. taking a Boltzmann average over exact balanced orbit attachments
   conditional on that fixed backbone.

Report this object as:

    conditional-on-singleton-MAP structured soft C

Do not describe it as unconstrained pair-sigmoid soft C, nor as exact
full-joint ``P(g_i=g_j)`` under the complete O2 distribution.
"""
from __future__ import annotations

import torch

from mattergen.assignment.global_copy_assembly.orbit_attachment import (
    enumerate_balanced_attachment_scores,
)

# Canonical name for reports / provenance (keep string stable for grep).
SOFT_C_KIND = "conditional-on-singleton-MAP structured soft C"


@torch.no_grad()
def soft_c_from_singleton_map_and_attachment(
    *,
    G_singleton: torch.Tensor,
    singleton_mask: torch.Tensor,
    orbit_atoms: torch.Tensor,
    F_attach: torch.Tensor,
    atoms_per_copy: int = 2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build conditional-on-singleton-MAP structured soft C.

    Singleton–singleton blocks are hard indicators from ``G_singleton`` MAP.
    Orbit–orbit (and singleton–orbit) blocks use exact enumeration marginals
    of balanced attachments **conditioned on** that fixed singleton MAP.
    """
    n = int(G_singleton.shape[0])
    k = int(G_singleton.shape[1])
    device = G_singleton.device
    C = torch.zeros(n, n, device=device, dtype=torch.float32)
    # singleton hard same-copy
    for a in range(n):
        if not bool(singleton_mask[a]):
            continue
        ga = int(G_singleton[a].argmax().item())
        for b in range(n):
            if not bool(singleton_mask[b]):
                continue
            gb = int(G_singleton[b].argmax().item())
            C[a, b] = 1.0 if ga == gb else 0.0
    # orbit soft
    scored = enumerate_balanced_attachment_scores(F_attach, atoms_per_copy=atoms_per_copy)
    if not scored:
        raise RuntimeError("empty attachment landscape for soft C")
    scores = torch.tensor([s / max(temperature, 1e-8) for s, _ in scored], device=device)
    log_z = torch.logsumexp(scores, dim=0)
    probs = torch.exp(scores - log_z)
    n_o = int(orbit_atoms.numel())
    # accumulate P(same copy) for orbit atom pairs
    pair_same = torch.zeros(n_o, n_o, device=device)
    for p, (_, pairs) in zip(probs, scored):
        group = torch.full((n_o,), -1, device=device, dtype=torch.long)
        for copy_id, (i, j) in enumerate(pairs):
            group[i] = copy_id
            group[j] = copy_id
        same = group[:, None].eq(group[None, :]).float()
        pair_same = pair_same + p * same
    # write into full C
    for ii in range(n_o):
        ai = int(orbit_atoms[ii].item())
        for jj in range(n_o):
            aj = int(orbit_atoms[jj].item())
            C[ai, aj] = pair_same[ii, jj]
        # singleton–orbit: use hard singleton copy of attachment MAP under mean
        # use expected copy occupancy
        occ = torch.zeros(k, device=device)
        for p, (_, pairs) in zip(probs, scored):
            for copy_id, (i, j) in enumerate(pairs):
                if i == ii or j == ii:
                    occ[copy_id] = occ[copy_id] + p
        # soft link to singleton atoms
        for b in range(n):
            if not bool(singleton_mask[b]):
                continue
            gb = int(G_singleton[b].argmax().item())
            C[ai, b] = occ[gb]
            C[b, ai] = occ[gb]
    # diagonal
    C.fill_diagonal_(1.0)
    # symmetrize numerically
    C = 0.5 * (C + C.T)
    C = C.clamp(0.0, 1.0)
    return C


def soft_c_metrics(C_soft: torch.Tensor, C0: torch.Tensor) -> dict[str, float | str]:
    """Metrics for conditional-on-singleton-MAP structured soft C (off-diagonal)."""
    off = ~torch.eye(C_soft.shape[0], dtype=torch.bool, device=C_soft.device)
    p = C_soft[off].clamp(1e-6, 1 - 1e-6)
    t = C0[off].float()
    # Brier
    brier = float(((p - t) ** 2).mean())
    # log loss
    ll = float((-(t * p.log() + (1 - t) * (1 - p).log())).mean())
    # AUC approx via ranking
    pos = p[t > 0.5]
    neg = p[t < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        auc = float("nan")
    else:
        # Mann-Whitney
        # sample if large
        if pos.numel() * neg.numel() > 5_000_000:
            pos = pos[torch.randperm(pos.numel())[:2000]]
            neg = neg[torch.randperm(neg.numel())[:2000]]
        # P(pos > neg) + 0.5 P(eq)
        diff = pos[:, None] - neg[None, :]
        auc = float(((diff > 0).float() + 0.5 * (diff == 0).float()).mean())
    entropy = float((-(p * p.log() + (1 - p) * (1 - p).log())).mean())
    return {
        "soft_C_kind": SOFT_C_KIND,
        "same_copy_pair_AUC": auc,
        "same_copy_Brier": brier,
        "same_copy_log_loss": ll,
        "soft_C_entropy": entropy,
    }
