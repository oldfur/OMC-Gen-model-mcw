"""Global structured copy assembly (clean geometry; oracle-R or geometry hard-R).

Supports full canonical ``R ∈ {0,1}^{N×M}`` with capacity ``|V_r|=K``.
Molecular automorphisms are a gauge for audit/evaluation only — never an
orbit-collapsed representation and never a silent rewrite of predicted R.
Does not implement assignment diffusion or the pos/cell denoiser.
"""

from .module import GlobalCopyAssemblyConfig, GlobalStructuredCopyAssembly
from .orbit_module import OrbitAwareAssemblyConfig, OrbitAwareCopyAssembly
from .orbit_membership import build_orbit_partition, collapse_roles_to_orbit_membership
from .targets import (
    AssemblyTarget,
    PredictedRoleAudit,
    build_assembly_target,
    build_assembly_target_from_predicted_roles,
    orbit_role_metrics,
    per_copy_automorphism_equivalent,
    permutations_to_group,
    validate_uniform_batch_k,
)
from .tree_crf import TreeCRF

__all__ = [
    "AssemblyTarget",
    "GlobalCopyAssemblyConfig",
    "GlobalStructuredCopyAssembly",
    "OrbitAwareAssemblyConfig",
    "OrbitAwareCopyAssembly",
    "PredictedRoleAudit",
    "TreeCRF",
    "build_assembly_target",
    "build_assembly_target_from_predicted_roles",
    "build_orbit_partition",
    "collapse_roles_to_orbit_membership",
    "orbit_role_metrics",
    "per_copy_automorphism_equivalent",
    "permutations_to_group",
    "validate_uniform_batch_k",
]
