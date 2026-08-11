"""N2: soft-C feedback into MatterGen geometry denoising (observational C → geometry)."""

from .gates import NoiseGateConfig, noise_gate, pairwise_confidence, pair_gate
from .module import SoftCFeedbackConfig, SoftCGeometryFeedbackN2

__all__ = [
    "NoiseGateConfig",
    "noise_gate",
    "pairwise_confidence",
    "pair_gate",
    "SoftCFeedbackConfig",
    "SoftCGeometryFeedbackN2",
]
