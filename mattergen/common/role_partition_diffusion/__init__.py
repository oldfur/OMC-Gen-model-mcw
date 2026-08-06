"""Discrete constrained role-partition diffusion (R,Q -> C)."""
from .role_diffusion import RolePartitionDiffusion
from .targets import RolePartitionTargets, build_targets
from .decoder import decode_connectivity
from .swap_gibbs import SwapGibbsRoleDiffusion, SwapScoreHead, SwapStopHead
from .oracle_partition import OraclePartitionRoleDiagnostic, capacity_sinkhorn

__all__ = ["RolePartitionDiffusion", "RolePartitionTargets", "build_targets", "decode_connectivity", "SwapGibbsRoleDiffusion", "SwapScoreHead", "SwapStopHead", "OraclePartitionRoleDiagnostic", "capacity_sinkhorn"]
