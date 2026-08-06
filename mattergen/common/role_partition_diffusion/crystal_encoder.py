from __future__ import annotations
import torch
from torch import nn

class PeriodicCrystalEncoder(nn.Module):
    """Scalar PBC-invariant assignment-side encoder; no baseline score head is used."""
    def __init__(self,hidden=256,layers=4,rbf_dim=64,cutoff=6.,max_neighbors=64):
        super().__init__();self.atom=nn.Embedding(119,hidden);self.register_buffer('centres',torch.linspace(0,cutoff,rbf_dim));self.cutoff=cutoff;self.max_neighbors=max_neighbors;self.edge=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden+rbf_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers));self.node=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers))
    def forward(self,z,frac,cell):
        delta=frac[None,:,:]-frac[:,None,:];image=-torch.round(delta);cart=(delta+image)@cell;d=torch.linalg.norm(cart,dim=-1);edge=(d>0)&(d<self.cutoff)
        if self.max_neighbors<z.numel():
            nearest=torch.topk(d.masked_fill(~edge,float('inf')),self.max_neighbors,largest=False).indices;keep=torch.zeros_like(edge);keep.scatter_(1,nearest,True);edge&=keep
        rbf=torch.exp(-((d[...,None]-self.centres)/(self.cutoff/max(1,len(self.centres))))**2)*edge[...,None];h=self.atom(z)
        for e,v in zip(self.edge,self.node):
            msg=e(torch.cat([h[:,None,:].expand(-1,len(z),-1),h[None,:,:].expand(len(z),-1,-1),rbf],-1))*edge[...,None];pool=msg.sum(1)/edge.sum(1,keepdim=True).clamp_min(1);h=h+v(torch.cat([h,pool],-1))
        return h
