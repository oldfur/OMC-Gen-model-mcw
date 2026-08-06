from __future__ import annotations
import torch
from torch import nn

class RoleCompatibilityHead(nn.Module):
    def __init__(self,hidden=256,layers=4,steps=32):
        super().__init__();self.vis=nn.Linear(hidden,hidden);self.mask=nn.Parameter(torch.zeros(hidden));self.time=nn.Embedding(steps+1,hidden);self.inp=nn.Sequential(nn.Linear(6*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden));self.blocks=nn.ModuleList(nn.Sequential(nn.Linear(5*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers));self.out=nn.Linear(hidden,1)
    def forward(self,hx,hm,noisy_roles,t,mask):
        visible=torch.where((noisy_roles>=0)[:,None],self.vis(hm[noisy_roles.clamp_min(0)]),self.mask[None]);a=hx[:,None,:].expand(-1,len(hm),-1);b=hm[None,:,:].expand(len(hx),-1,-1);time=self.time(torch.as_tensor(t,device=hx.device)).view(1,1,-1).expand_as(a);u=self.inp(torch.cat([a,b,visible[:,None,:].expand_as(a),(a-b).abs(),a*b,time],-1)).masked_fill(~mask[...,None],0)
        for block in self.blocks:
            mean_r=(u*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];max_r=u.masked_fill(~mask[...,None],torch.finfo(u.dtype).min).max(1).values;mean_c=(u*mask[...,None]).sum(0)/mask.sum(0).clamp_min(1)[:,None];max_c=u.masked_fill(~mask[...,None],torch.finfo(u.dtype).min).max(0).values
            u=(u+block(torch.cat([u,mean_r[:,None,:].expand_as(u),max_r[:,None,:].expand_as(u),mean_c[None,:,:].expand_as(u),max_c[None,:,:].expand_as(u)],-1))).masked_fill(~mask[...,None],0)
        return self.out(u).squeeze(-1).masked_fill(~mask,float('-inf'))
