from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class RolePartitionTargets:
    role: torch.Tensor; copy: torch.Tensor; role_atomic_numbers: torch.Tensor; K:int; M:int
    @property
    def N(self):return self.role.numel()
    def R(self):
        x=torch.zeros(self.N,self.M,device=self.role.device,dtype=torch.long);x[torch.arange(self.N,device=self.role.device),self.role]=1;return x
    def validate(self,crystal_z):
        r=self.R()
        if self.N!=self.K*self.M:raise ValueError('N != K*M')
        if not bool((r.sum(1)==1).all()) or not bool((r.sum(0)==self.K).all()):raise ValueError('invalid role capacity target')
        if not torch.equal(crystal_z,self.role_atomic_numbers[self.role]):raise ValueError('role target violates element identity')
    def q(self,anchor):
        # The candidate order is the current presentation order, never an order
        # derived from the true copy label.  ``copy`` is used only here to build
        # the clean target.  In particular, sorting by copy would turn every
        # clean Q into an identity matrix and leak the answer through ordering.
        out={};anchor_idx=(self.role==anchor).nonzero().flatten()
        for role in range(self.M):
            if role==anchor:continue
            idx=(self.role==role).nonzero().flatten()
            # q[target-row, anchor-column] is one iff copy labels agree.
            q=torch.zeros(self.K,self.K,dtype=torch.long,device=self.role.device)
            for target_row,node in enumerate(idx):q[target_row,(self.copy[anchor_idx]==self.copy[node]).nonzero().item()]=1
            out[role]=(anchor_idx,idx,q)
        return out

def build_targets(role,copy,role_atomic_numbers,K=None):
    if role.dtype not in (torch.int64,torch.int32) or copy.dtype not in (torch.int64,torch.int32):raise ValueError('role/copy IDs must be integer tensors')
    M=int(role.max())+1;K=int(copy.max())+1 if K is None else K
    target=RolePartitionTargets(role.long(),copy.long(),role_atomic_numbers.long(),K,M)
    return target
