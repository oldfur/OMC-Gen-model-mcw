from __future__ import annotations
import torch
from torch import nn
from .mask_schedule import AbsorbingMaskSchedule
from .crystal_encoder import PeriodicCrystalEncoder
from .molecule_encoder import MolecularGraphEncoder
from .role_head import RoleCompatibilityHead
from .swap_gibbs import SwapGibbsRoleDiffusion, SwapScoreHead

class CapacityRoleDiffusion:
    def __init__(self,steps=32):self.schedule=AbsorbingMaskSchedule(steps)
    def forward(self,role,t,generator=None):
        keep=torch.rand(role.shape,device=role.device,generator=generator)<self.schedule.bar_alpha[t].to(role.device)
        return torch.where(keep,role,torch.full_like(role,-1))
    def loss(self,scores,truth,noisy):
        masked=noisy<0
        return (scores.sum()*0 if not masked.any() else nn.functional.cross_entropy(scores[masked],truth[masked]))

class RolePartitionDiffusion(nn.Module):
    """Assignment-only R/Q container. It intentionally owns no baseline diffusion module."""
    def __init__(self,hidden=256,role_steps=32,matching_steps=32,role_diffusion_type="masked_capacity",swap_terminal_randomization_steps=128,role_context_mode="geometry_only"):
        super().__init__()
        if role_diffusion_type not in {"masked_capacity", "swap_gibbs"}: raise ValueError(f"unsupported role_diffusion_type={role_diffusion_type!r}")
        if role_context_mode not in {"geometry_only", "oracle_same_copy", "oracle_copy_local"}: raise ValueError(f"unsupported role_context_mode={role_context_mode!r}")
        self.crystal_encoder=PeriodicCrystalEncoder(hidden=hidden);self.molecule_encoder=MolecularGraphEncoder(hidden=hidden)
        self.role_diffusion_type=role_diffusion_type
        self.role_context_mode=role_context_mode
        self.role_head=RoleCompatibilityHead(hidden=hidden,steps=role_steps) if role_diffusion_type=="masked_capacity" else None
        self.swap_head=SwapScoreHead(hidden=hidden,steps=role_steps) if role_diffusion_type=="swap_gibbs" else None
        self.role_diffusion=CapacityRoleDiffusion(role_steps) if role_diffusion_type=="masked_capacity" else SwapGibbsRoleDiffusion(role_steps,swap_terminal_randomization_steps)
        self.role_steps=role_steps;self.matching_steps=matching_steps
    def loss(self,batch):
        # Generic DiffusionModule integration requires an explicitly prepared role
        # partition batch; silent fallback to old full assignment is prohibited.
        required=('role_partition_role','role_partition_frac_pos','role_partition_role_z','role_partition_edge_index','role_partition_bond_type')
        missing=[k for k in required if not hasattr(batch,k)]
        if missing:raise ValueError(f'role_partition mode requires prepared fields: {missing}')
        raise NotImplementedError('use role_partition_diffusion diagnostics/trainer for structured R,Q loss')
