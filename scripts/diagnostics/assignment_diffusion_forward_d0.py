"""D0-only finite-dimensional tests for the gauge-fixed assignment DDPM."""
import json
from pathlib import Path

import torch

from mattergen.common.assignment_diffusion import AssignmentDiffusion, gauge_center, sinkhorn


def residual(x, m):
    x = x.masked_fill(~m, 0)
    return float(x.sum(1).abs().max()), float(x.sum(0).abs().max())


def projector(m, dtype):
    allowed = m.flatten().nonzero().flatten()
    columns = []
    for j in allowed:
        e = torch.zeros_like(m, dtype=dtype).flatten()
        e[j] = 1
        columns.append(gauge_center(e.reshape_as(m), m).flatten()[allowed])
    return torch.stack(columns, 1), allowed


def one_case(mask, dtype, samples=4096):
    p, allowed = projector(mask, dtype)
    rank = int(torch.linalg.matrix_rank(p, atol=1e-8 if dtype == torch.float64 else 1e-5))
    eps = torch.randn(samples, allowed.numel(), dtype=dtype)
    projected = eps @ p.T
    mean = projected.mean(0).abs().max()
    cov = projected.T @ projected / samples
    cov_err = (cov - p).abs().max()
    y = torch.randn_like(mask, dtype=dtype)
    g = gauge_center(y, mask)
    r, c = residual(g, mask)
    return dict(dtype=str(dtype).split('.')[-1], shape=list(mask.shape), allowed=int(allowed.numel()), rank=rank,
                row_residual=r, column_residual=c, idempotence=float((gauge_center(g, mask)-g).abs().max()),
                covariance_max_error=float(cov_err), mean_max_abs=float(mean))


def main():
    torch.manual_seed(20260804)
    # Complete bipartite blocks 3x3, 2x2, 1x1; expected rank 4+1+0=5.
    mask = torch.zeros(6, 6, dtype=torch.bool)
    mask[:3, :3] = True; mask[3:5, 3:5] = True; mask[5, 5] = True
    cases = [one_case(mask, torch.float32), one_case(mask, torch.float64)]
    expected_rank = (3 - 1) ** 2 + (2 - 1) ** 2 + (1 - 1) ** 2
    d = AssignmentDiffusion(steps=16)
    a0 = torch.eye(6); l0 = gauge_center(d.kappa * (2 * a0 - 1), mask)
    endpoints = []
    for t in [0, 1, 4, 8, 15]:
        eps = gauge_center(torch.randn_like(l0), mask); ab = d.alpha_bar[t]
        lt = gauge_center(ab.sqrt()*l0 + (1-ab).sqrt()*eps, mask)
        rec = (lt - ab.sqrt()*l0) / (1-ab).sqrt()
        a = sinkhorn(lt, mask)
        entropy = float(-(a[a > 0] * a[a > 0].log()).mean())
        endpoints.append(dict(t=t, reconstruction_error=float((rec-eps).abs().max()),
                              eps_residual=max(residual(eps,mask)), lt_residual=max(residual(lt,mask)),
                              signal_norm=float((ab.sqrt()*l0).norm()), noise_norm=float(((1-ab).sqrt()*eps).norm()), entropy=entropy,
                              sinkhorn_error=float((a.sum(1)-1).abs().max().maximum((a.sum(0)-1).abs().max())), forbidden_mass=float(a[~mask].abs().max())))
    # Row and column equivariance, plus exact gauge-shift invariance.
    y = torch.randn(6,6); row = torch.tensor([2,0,1,4,3,5]); col = torch.tensor([1,2,0,4,3,5])
    yp = y[row][:, col]; mp = mask[row][:, col]
    equiv = float((sinkhorn(yp, mp)-sinkhorn(y,mask)[row][:,col]).abs().max())
    shift = torch.randn(6,1)+torch.randn(1,6)
    shift_inv = float((sinkhorn(y,mask)-sinkhorn(y+shift,mask)).abs().max())
    _, a1, tr1 = d.reverse_ddpm(mask, lambda x,t: torch.zeros_like(x), seed=77)
    _, a2, tr2 = d.reverse_ddpm(mask, lambda x,t: torch.zeros_like(x), seed=77)
    _, a3, _ = d.reverse_ddpm(mask, lambda x,t: torch.zeros_like(x), seed=78)
    out = dict(seed=20260804, expected_projector_rank=expected_rank, projector_cases=cases, forward_endpoints=endpoints,
               row_column_equivariance_error=equiv, gauge_shift_sinkhorn_error=shift_inv,
               reverse_fixed_seed_error=float((a1-a2).abs().max()), reverse_different_seed_difference=float((a1-a3).abs().max()),
               reverse_final_error=tr1[-1]['marginal_error'], reverse_steps=len(tr1))
    p = Path('outputs/assignment_diffusion_mvp'); p.mkdir(parents=True, exist_ok=True)
    p.joinpath('d0_forward_metrics.json').write_text(json.dumps(out, indent=2))
    print(p/'d0_forward_metrics.json')


if __name__ == '__main__':
    main()
