"""N1: MatterGen-noisy geometry → orbit-aware copy assignment (observational branch).

Primary hidden source: frozen pretrained molecular-CSP GemNetTDenoiser
(node_embeddings). ContextCrystalEncoder is ablation-only.

Does not modify MatterGen geometry denoising scores or reverse sampling.
"""

from .mattergen_noise_adapter import MatterGenNativeNoiseAdapter, build_default_mattergen_corruption
from .module import NoisyCopyAssignmentConfig, NoisyCopyAssignmentN1
from .soft_c import SOFT_C_KIND, SOFT_C_SEMANTICS

__all__ = [
    "MatterGenNativeNoiseAdapter",
    "build_default_mattergen_corruption",
    "NoisyCopyAssignmentConfig",
    "NoisyCopyAssignmentN1",
    "SOFT_C_KIND",
    "SOFT_C_SEMANTICS",
]
