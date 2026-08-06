"""Global structured copy assembly with an oracle role assignment.

This package is intentionally clean-geometry/oracle-R only.  It neither
implements assignment diffusion nor calls the pos/cell denoiser.
"""

from .module import GlobalCopyAssemblyConfig, GlobalStructuredCopyAssembly
from .targets import AssemblyTarget, build_assembly_target, permutations_to_group, validate_uniform_batch_k
from .tree_crf import TreeCRF

__all__ = ["AssemblyTarget", "GlobalCopyAssemblyConfig", "GlobalStructuredCopyAssembly", "TreeCRF", "build_assembly_target", "permutations_to_group", "validate_uniform_batch_k"]
