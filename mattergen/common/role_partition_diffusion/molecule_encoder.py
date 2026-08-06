from __future__ import annotations
import torch
from torch import nn

class MolecularGraphEncoder(nn.Module):
    def __init__(self,hidden=256,layers=4):
        super().__init__();self.atom=nn.Embedding(119,hidden);self.bond=nn.Embedding(8,hidden);self.edge=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers));self.node=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers))
    def forward(self,z,edge_index,bond_type):
        h=self.atom(z)
        for e,v in zip(self.edge,self.node):
            pool=torch.zeros_like(h)
            if edge_index.numel():
                src,dst=edge_index;m=e(torch.cat([h[src],self.bond(bond_type.clamp(0,7))],-1));pool.index_add_(0,dst,m);pool/=torch.bincount(dst,minlength=len(z)).to(h.dtype)[:,None].clamp_min(1)
            h=h+v(torch.cat([h,pool],-1))
        return h
