from __future__ import annotations
import torch

def decode_connectivity(R:torch.Tensor, anchor:int, q_by_role:dict[int,tuple[torch.Tensor,torch.Tensor,torch.Tensor]]):
    n,m=R.shape;k=int(R[:,anchor].sum());G=torch.zeros(n,k,dtype=R.dtype,device=R.device);a=(R[:,anchor]==1).nonzero().flatten()
    for p,node in enumerate(a):G[node,p]=1
    for role,(_,nodes,q) in q_by_role.items():
        for row,node in enumerate(nodes):G[node,q[row].argmax()]=1
    C=G.float()@G.float().T
    return G,C
