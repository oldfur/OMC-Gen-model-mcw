from __future__ import annotations
import torch
from .mask_schedule import AbsorbingMaskSchedule
from .structured_matching import structured_nll

class PermutationMatchingDiffusion:
    def __init__(self,steps=32):self.schedule=AbsorbingMaskSchedule(steps)
    def forward(self,perm,t,generator=None):
        keep=torch.rand(perm.shape,device=perm.device,generator=generator)<self.schedule.bar_alpha[t].to(perm.device);return torch.where(keep,perm,torch.full_like(perm,-1))
    def loss(self,scores,truth,noisy):return structured_nll(scores,truth,noisy)/max(1,int((noisy<0).sum()))
