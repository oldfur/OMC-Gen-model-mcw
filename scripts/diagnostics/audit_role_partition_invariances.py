"""Numerical invariance/equivariance audit for the role-partition heads."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from mattergen.common.role_partition_diffusion.role_diffusion import RolePartitionDiffusion
from mattergen.common.role_partition_diffusion.matching_head import StructuredMatchingHead
from mattergen.common.role_partition_diffusion.targets import build_targets


OUT = Path("outputs/assignment_diffusion_mvp/role_partition_discrete_constrained")
SRC = Path("outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt")


def err(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().max().cpu())


def err_allowed(x: torch.Tensor, y: torch.Tensor, allowed: torch.Tensor) -> float:
    return float((x[allowed] - y[allowed]).abs().max().cpu())


@torch.no_grad()
def main() -> None:
    torch.manual_seed(17)
    s = torch.load(SRC, map_location="cpu")
    s = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
    net = RolePartitionDiffusion().cuda().eval()
    role_ckpt = torch.load(OUT / "checkpoints/role_best.pt", map_location="cpu")
    net.load_state_dict(role_ckpt.get("state_dict", role_ckpt))
    qckpt = torch.load(OUT / "checkpoints/matching_best.pt", map_location="cpu")
    qnet = RolePartitionDiffusion().cuda().eval(); qnet.load_state_dict(qckpt["encoder"])
    qhead = StructuredMatchingHead().cuda().eval(); qhead.load_state_dict(qckpt["head"])

    hx = net.crystal_encoder(s["z"], s["pos"], s["cell"])
    hm = net.molecule_encoder(s["role_z"], s["role_edge_index"], s["role_bond_type"])
    g = torch.Generator(device="cuda").manual_seed(123)
    prow = torch.randperm(s["N"], device="cuda", generator=g)
    pcol = torch.randperm(s["M"], device="cuda", generator=g)
    noisy = torch.full((s["N"],), -1, dtype=torch.long, device="cuda")
    hard = s["z"][:, None].eq(s["role_z"][None, :])
    score = net.role_head(hx, hm, noisy, 32, hard)
    row_score = net.role_head(hx[prow], hm, noisy[prow], 32, hard[prow])
    col_score = net.role_head(hx, hm[pcol], noisy, 32, hard[:, pcol])
    both_score = net.role_head(hx[prow], hm[pcol], noisy[prow], 32, hard[prow][:, pcol])
    translated = net.crystal_encoder(s["z"], (s["pos"] + torch.tensor([.177, .331, .719], device="cuda")) % 1, s["cell"])

    targets = build_targets(s["role"], s["copy"], s["role_z"], s["Z"])
    anchor, role = 0, 1
    ai, ti, _ = targets.q(anchor)[role]
    hxq = qnet.crystal_encoder(s["z"], s["pos"], s["cell"])
    hmq = qnet.molecule_encoder(s["role_z"], s["role_edge_index"], s["role_bond_type"])
    dist = s["pos"][ti, None] - s["pos"][ai][None, :]
    dist = torch.linalg.norm((dist - torch.round(dist)) @ s["cell"], dim=-1)
    partial = torch.full((s["Z"],), -1, dtype=torch.long, device="cuda")
    sq = qhead(hxq[ai], hxq[ti], hmq[anchor], hmq[role], dist, partial, 32)
    pr = torch.randperm(s["Z"], device="cuda", generator=g); pc = torch.randperm(s["Z"], device="cuda", generator=g)
    sqp = qhead(hxq[ai][pc], hxq[ti][pr], hmq[anchor], hmq[role], dist[pr][:, pc], partial[pr], 32)
    result = {
        "device": "cuda", "dtype": str(score.dtype),
        "crystal_translation_max_abs": err(hx, translated),
        "R_row_max_abs": err_allowed(row_score, score[prow], hard[prow]),
        "R_column_max_abs": err_allowed(col_score, score[:, pcol], hard[:, pcol]),
        "R_joint_max_abs": err_allowed(both_score, score[prow][:, pcol], hard[prow][:, pcol]),
        "Q_joint_row_column_max_abs": err(sqp, sq[pr][:, pc]),
        # Wrapped-PBC graph construction is float32. Translation changes only
        # round-off at the periodic boundary, not the representation semantics.
        "threshold": 2e-4,
    }
    result["PASS"] = all(v <= result["threshold"] for k, v in result.items() if k.endswith("max_abs"))
    (OUT / "encoder_invariance_metrics.json").write_text(json.dumps(result, indent=2))
    (OUT / "role_head_equivariance.json").write_text(json.dumps({"row_max_abs": result["R_row_max_abs"], "column_max_abs": result["R_column_max_abs"], "joint_max_abs": result["R_joint_max_abs"], "PASS": result["PASS"]}, indent=2))
    (OUT / "matching_head_equivariance.json").write_text(json.dumps({"joint_row_column_max_abs": result["Q_joint_row_column_max_abs"], "PASS": result["PASS"]}, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
