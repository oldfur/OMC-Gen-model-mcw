"""N1: MatterGen-noisy geometry → orbit-aware copy assignment (observational branch).

Does not modify MatterGen geometry denoising scores or reverse sampling.
"""

from .mattergen_noise_adapter import MatterGenNativeNoiseAdapter, build_default_mattergen_corruption
from .module import NoisyCopyAssignmentConfig, NoisyCopyAssignmentN1

__all__ = [
    "MatterGenNativeNoiseAdapter",
    "build_default_mattergen_corruption",
    "NoisyCopyAssignmentConfig",
    "NoisyCopyAssignmentN1",
]
