"""Soft-C channels, group context, and PBC edge audit for N2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .gates import NoiseGateConfig, noise_gate_value, pair_gate, pairwise_confidence


@dataclass
class EdgeFeedbackAudit:
    num_edges: int = 0
    num_nonzero_image_edges: int = 0
    num_self_image_edges: int = 0
    num_nonzero_self_image_edges: int = 0
    num_duplicate_atom_pair_images: int = 0
    num_zeroed_self_image_same_copy: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_edges": self.num_edges,
            "num_nonzero_image_edges": self.num_nonzero_image_edges,
            "num_self_image_edges": self.num_self_image_edges,
            "num_nonzero_self_image_edges": self.num_nonzero_self_image_edges,
            "num_duplicate_atom_pair_images": self.num_duplicate_atom_pair_images,
            "num_zeroed_self_image_same_copy": self.num_zeroed_self_image_same_copy,
            **self.metadata,
        }


def shuffle_soft_c(c: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Legacy global atom shuffle: C' = P C P^T.

    Prefer :func:`orbit_preserving_shuffle_soft_c` for B5 control.
    """
    n = c.shape[0]
    if generator is None:
        perm = torch.randperm(n, device=c.device)
    else:
        perm = torch.randperm(n, generator=generator, device="cpu").to(c.device)
    return c[perm][:, perm]


def orbit_preserving_shuffle_soft_c(
    c: torch.Tensor,
    orbit_labels: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    max_attempts: int = 64,
) -> torch.Tensor:
    """Orbit-preserving soft-C shuffle for B5 control.

    For each orbit o with atom set V_o, draw an independent permutation
    σ_o : V_o → V_o and compose into a global permutation P. Then

        C_shuffle = P C P^T   (implemented as C[perm][:, perm]
                               with perm[i] = σ(i))

    Geometry and H^A are left untouched by the caller; only soft-C atom-pair
    correspondence is broken. Permutations never mix different orbits.

    Avoids the identity / no-op permutation when at least one orbit has |V_o|≥2.
    """
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"soft C must be square, got {tuple(c.shape)}")
    n = int(c.shape[0])
    labels = orbit_labels.long().reshape(-1).to(device="cpu")
    if labels.numel() != n:
        raise ValueError(f"orbit_labels length {labels.numel()} != N={n}")

    orbit_ids = sorted({int(x) for x in labels.tolist()})
    shufflable = [o for o in orbit_ids if int((labels == o).sum()) >= 2]
    if not shufflable:
        raise RuntimeError(
            "orbit_preserving_shuffle_soft_c: no orbit with size>=2; cannot break copy correspondence"
        )

    def _draw_perm() -> torch.Tensor:
        perm = torch.arange(n, dtype=torch.long)
        for o in orbit_ids:
            idx = torch.nonzero(labels == o, as_tuple=False).flatten()
            k = int(idx.numel())
            if k <= 1:
                continue
            if generator is None:
                local = torch.randperm(k)
            else:
                local = torch.randperm(k, generator=generator)
            perm[idx] = idx[local]
        return perm

    identity = torch.arange(n, dtype=torch.long)
    perm = _draw_perm()
    attempts = 1
    while torch.equal(perm, identity) and attempts < max_attempts:
        perm = _draw_perm()
        attempts += 1
    if torch.equal(perm, identity):
        # Force a non-trivial swap inside the largest orbit
        o_big = max(shufflable, key=lambda o: int((labels == o).sum()))
        idx = torch.nonzero(labels == o_big, as_tuple=False).flatten()
        perm = identity.clone()
        perm[idx[0]], perm[idx[1]] = idx[1].item(), idx[0].item()

    perm = perm.to(device=c.device)
    # C'[i,j] = C[σ(i), σ(j)]
    return c[perm][:, perm]


def build_edge_soft_c_channels(
    *,
    soft_c: torch.Tensor,
    edge_index: torch.Tensor,
    cell_offsets: torch.Tensor | None,
    t_fraction: float,
    noise_gate_cfg: NoiseGateConfig,
    pairwise_confidence_enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, EdgeFeedbackAudit]:
    """Lookup soft-C on GemNet edges and build gated intra/inter/signed channels.

    Self-image edges (i==j and nonzero PBC offset): force same-copy feedback to 0
    (do not treat C_ii=1 as same-molecule for periodic self-images).
    """
    soft_c = soft_c.detach().float()
    n = soft_c.shape[0]
    src = edge_index[0].long()
    dst = edge_index[1].long()
    e = int(src.numel())
    audit = EdgeFeedbackAudit(num_edges=e)

    if e == 0:
        z = soft_c.new_zeros(0)
        return z, z, z, audit

    p = soft_c[src, dst].clamp(0.0, 1.0)
    a = pair_gate(
        p,
        t_fraction,
        noise_gate_cfg,
        pairwise_confidence_enabled=pairwise_confidence_enabled,
    )

    # PBC audit
    if cell_offsets is not None:
        off = cell_offsets.reshape(e, -1)
        nonzero_img = (off.abs().sum(dim=-1) > 0)
        audit.num_nonzero_image_edges = int(nonzero_img.sum().item())
        self_edge = src.eq(dst)
        audit.num_self_image_edges = int(self_edge.sum().item())
        self_img = self_edge & nonzero_img
        audit.num_nonzero_self_image_edges = int(self_img.sum().item())
        # zero same-copy feedback on nonzero self-image edges
        if self_img.any():
            a = a.clone()
            a[self_img] = 0.0
            p = p.clone()
            # keep p for logging but channels use a=0
            audit.num_zeroed_self_image_same_copy = int(self_img.sum().item())
        # duplicate atom-pair images
        keys = src * (n + 1) + dst
        # count pairs that appear more than once (any image)
        uniq, counts = torch.unique(keys, return_counts=True)
        audit.num_duplicate_atom_pair_images = int((counts > 1).sum().item())
        audit.metadata["cell_offsets_shape"] = list(cell_offsets.shape)
    else:
        audit.metadata["cell_offsets"] = None

    c_intra = a * p
    c_inter = a * (1.0 - p)
    c_signed = a * (2.0 * p - 1.0)
    # When p=0.5 or g=0, a=0 ⇒ all channels zero
    return c_intra, c_inter, c_signed, audit


def expected_same_copy_representation(
    *,
    h_a: torch.Tensor,
    soft_c: torch.Tensor,
    t_fraction: float,
    noise_gate_cfg: NoiseGateConfig,
    molecules_atoms_m: int,
    pairwise_confidence_enabled: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute h_group and g_group from Pass-A hiddens H^A and soft C.

    w_ij = g_noise * q_ij * p_ij, w_ii = 0
    h_group_i = sum_j w_ij H_j / (sum_j w_ij + eps)
    g_group_i = clip(m_i / (M-1), 0, 1)
    """
    soft_c = soft_c.detach().float()
    h_a = h_a.detach()  # Pass A frozen; never backprop through H^A into C
    n, hid = h_a.shape
    device = h_a.device
    p = soft_c.clamp(0.0, 1.0)
    g = noise_gate_value(t_fraction, noise_gate_cfg).to(device=device, dtype=p.dtype)
    if pairwise_confidence_enabled:
        q = pairwise_confidence(p)
    else:
        q = torch.ones_like(p)
    w = (g * q * p).clone()
    # exclude self
    w.fill_diagonal_(0.0)
    mass = w.sum(dim=-1)  # m_i
    # h_group = w @ H / (m + eps)
    h_group = (w @ h_a) / (mass.unsqueeze(-1) + eps)
    # when mass≈0, h_group ~ 0; adapter gated by g_group anyway
    denom = max(int(molecules_atoms_m) - 1, 1)
    g_group = (mass / float(denom)).clamp(0.0, 1.0)
    meta = {
        "mean_mass": float(mass.mean().item()),
        "mean_g_group": float(g_group.mean().item()),
        "g_noise": float(g.reshape(-1)[0].item()) if g.numel() else float(g),
        "molecules_atoms_m": int(molecules_atoms_m),
        "num_atoms": n,
        "hidden_dim": hid,
    }
    return h_group, g_group, mass, meta
