from __future__ import annotations
import torch
from torch import nn

class StructuredMatchingHead(nn.Module):
    """KxK axial score head; partial tokens are MATCHED/EXCLUDED/UNKNOWN only."""
    def __init__(self,hidden=256,layers=3,steps=32,rbf_dim=32):
        super().__init__();self.state=nn.Embedding(3,hidden);self.time=nn.Embedding(steps+1,hidden);self.register_buffer('centres',torch.linspace(0,6,rbf_dim));self.inp=nn.Sequential(nn.Linear(10*hidden+rbf_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden));self.blocks=nn.ModuleList(nn.Sequential(nn.Linear(3*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers));self.out=nn.Linear(hidden,1)
    def forward(self,h_anchor,h_target,hma,hmr,dist,partial,t):
        """Score target-role rows against anchor columns.

        ``partial[q]`` stores an anchor column for target row ``q`` and
        ``dist`` consequently has shape ``[target, anchor]``.
        """
        k=len(h_anchor);state=torch.full((k,k),2,device=dist.device,dtype=torch.long);visible=partial>=0
        if visible.any():
            rows=visible.nonzero().flatten();used=partial[visible]
            state[rows,:]=1;state[:,used]=1;state[rows,used]=0
        target=h_target[:,None,:].expand(-1,k,-1);anchor=h_anchor[None,:,:].expand(k,-1,-1);mol=torch.cat([hma,hmr,(hma-hmr).abs(),hma*hmr],-1).view(1,1,-1).expand(k,k,-1);rbf=torch.exp(-((dist[...,None]-self.centres)/(.1875))**2);time=self.time(torch.as_tensor(t,device=dist.device)).view(1,1,-1).expand(k,k,-1);u=self.inp(torch.cat([target,anchor,(target-anchor).abs(),target*anchor,mol,self.state(state),time,rbf],-1))
        allowed=state!=1
        for block in self.blocks:
            r=u.masked_fill(~allowed[...,None],0).sum(1)/allowed.sum(1).clamp_min(1)[:,None];c=u.masked_fill(~allowed[...,None],0).sum(0)/allowed.sum(0).clamp_min(1)[:,None];u=u+block(torch.cat([u,r[:,None,:].expand_as(u),c[None,:,:].expand_as(u)],-1))
        return self.out(u).squeeze(-1).masked_fill(~allowed,float('-inf'))
