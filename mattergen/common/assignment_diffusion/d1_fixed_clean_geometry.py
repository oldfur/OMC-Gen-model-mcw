"""D1-only clean-geometry assignment predictor.

It has no copy, packed-slot, canonical-role index, or clean-assignment input.
All operations are set/graph operations and therefore equivariant to independent
crystal-row and molecular-role permutations.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .assignment_diffusion import gauge_center, sinkhorn

class RBF(nn.Module):
    def __init__(self, width=32, cutoff=7.0):
        super().__init__(); self.register_buffer("centres",torch.linspace(0,cutoff,width)); self.width=width; self.cutoff=cutoff
    def forward(self,d):
        return torch.exp(-((d[...,None]-self.centres)/(self.cutoff/self.width))**2) * (d[...,None] < self.cutoff)

class PeriodicCrystalEncoder(nn.Module):
    def __init__(self, hidden=96, layers=2):
        super().__init__();self.atom=nn.Embedding(119,hidden);self.rbf=RBF();self.msg=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden+32,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers));self.up=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers))
    def forward(self,z,pos,cell):
        # Fractional relative coordinates make global fractional translation invisible.
        delta=pos[:,None,:]-pos[None,:,:]; delta=delta-torch.round(delta); distance=torch.linalg.norm(delta@cell,dim=-1); edge=(distance>0)&(distance<self.rbf.cutoff); h=self.atom(z)
        for message,update in zip(self.msg,self.up):
            hi=h[:,None,:].expand(-1,len(z),-1);hj=h[None,:,:].expand(len(z),-1,-1);m=message(torch.cat([hi,hj,self.rbf(distance)],-1))*edge[...,None]; pooled=m.sum(1)/edge.sum(1,keepdim=True).clamp_min(1);h=h+update(torch.cat([h,pooled],-1))
        return h

class MolecularRoleEncoder(nn.Module):
    def __init__(self,hidden=96,layers=2):
        super().__init__();self.atom=nn.Embedding(119,hidden);self.bond=nn.Embedding(8,hidden);self.msg=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers));self.up=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers))
    def forward(self,z,edge_index,bond_type):
        h=self.atom(z)
        for message,update in zip(self.msg,self.up):
            aggregate=torch.zeros_like(h)
            if edge_index.numel():
                src,dst=edge_index; m=message(torch.cat([h[src],self.bond(bond_type.clamp(0,7))],-1));aggregate.index_add_(0,dst,m)
                count=torch.bincount(dst,minlength=len(z)).to(h.dtype)[:,None].clamp_min(1);aggregate=aggregate/count
            h=h+update(torch.cat([h,aggregate],-1))
        return h

class D1FixedCleanGeometryPredictor(nn.Module):
    """Periodic crystal graph + molecular graph + masked axial interactions."""
    def __init__(self,hidden=96,interaction_layers=3,steps=128,sinkhorn_iters=100):
        super().__init__();self.steps=steps;self.iters=sinkhorn_iters;self.xtal=PeriodicCrystalEncoder(hidden);self.mol=MolecularRoleEncoder(hidden);self.time=nn.Sequential(nn.Linear(1,hidden),nn.SiLU(),nn.Linear(hidden,hidden));self.pair=nn.Sequential(nn.Linear(3*hidden+1,hidden),nn.SiLU(),nn.Linear(hidden,hidden));self.row=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(interaction_layers));self.col=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(interaction_layers));self.out=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
    def forward(self,l_t,t_a,z_xtal,pos,cell,z_role,role_edge_index,role_bond_type,mask,*,geometry=True,edges=True):
        if l_t.ndim!=2 or l_t.shape!=mask.shape or l_t.shape[0]!=l_t.shape[1]: raise ValueError("D1 logits/mask must be square")
        if not torch.isfinite(l_t).all(): raise ValueError("D1 logits contain NaN/Inf")
        # No role/copy/slot metadata enters this function.
        h_i=self.xtal(z_xtal,pos,cell) if geometry else torch.zeros((len(z_xtal),self.xtal.atom.embedding_dim),device=l_t.device,dtype=l_t.dtype)
        h_r=self.mol(z_role,role_edge_index if edges else role_edge_index[:,:0],role_bond_type if edges else role_bond_type[:0])
        if l_t.shape[1] % h_r.shape[0]:raise ValueError("assignment columns must be whole molecular-role copies")
        # Repeated copy columns only replicate the role representation; their order
        # is supplied by noisy L_t, never by a copy label.
        repeated=h_r.repeat(l_t.shape[1]//h_r.shape[0],1)
        t=self.time(torch.as_tensor(t_a,dtype=l_t.dtype,device=l_t.device).reshape(1,1)/self.steps).expand(len(z_xtal),len(repeated),-1)
        h=self.pair(torch.cat([h_i[:,None,:].expand(-1,len(repeated),-1),repeated[None,:,:].expand(len(z_xtal),-1,-1),t,l_t[...,None]],-1)).masked_fill(~mask[...,None],0)
        for row_layer,col_layer in zip(self.row,self.col):
            r=(h*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];c=(h*mask[...,None]).sum(0)/mask.sum(0).clamp_min(1)[:,None]
            h=(h+row_layer(torch.cat([h,r[:,None,:].expand_as(h)],-1))+col_layer(torch.cat([h,c[None,:,:].expand_as(h)],-1))).masked_fill(~mask[...,None],0)
        return gauge_center(self.out(h).squeeze(-1),mask)
