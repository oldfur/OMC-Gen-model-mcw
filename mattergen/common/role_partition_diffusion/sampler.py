from __future__ import annotations
import torch
from scipy.optimize import linear_sum_assignment
from .structured_matching import sample_completion

def capacity_hungarian(scores,visible,capacity,gumbel=False,generator=None):
    slots=torch.repeat_interleave(torch.arange(len(capacity),device=scores.device),capacity);masked=(visible<0).nonzero().flatten();out=visible.clone()
    if len(masked)==0:return out
    value=scores[masked][:,slots]
    if gumbel:value=value-torch.log(-torch.log(torch.rand(value.shape,device=value.device,generator=generator).clamp_min(1e-8)))
    rows,cols=linear_sum_assignment((-value.detach().cpu().numpy()));out[masked[torch.tensor(rows,device=scores.device)]]=slots[torch.tensor(cols,device=scores.device)];return out

def reverse_roles(scores,noisy,capacity_diffusion,t,generator=None):
    visible=noisy>=0;used=torch.bincount(noisy[visible],minlength=scores.shape[1]);K=(scores.shape[0]//scores.shape[1]);capacity=(K-used).long()
    proposal=capacity_hungarian(scores,noisy,capacity,gumbel=True,generator=generator);rho=capacity_diffusion.schedule.rho(t);choose=(torch.rand(noisy.shape,device=noisy.device,generator=generator)<rho)&(~visible);return torch.where(choose,proposal,noisy)

def reverse_matching(scores,noisy,diffusion,t,generator=None):
    proposal=sample_completion(scores,noisy,generator);rho=diffusion.schedule.rho(t);choose=(torch.rand(noisy.shape,device=noisy.device,generator=generator)<rho)&(noisy<0);return torch.where(choose,proposal,noisy)
