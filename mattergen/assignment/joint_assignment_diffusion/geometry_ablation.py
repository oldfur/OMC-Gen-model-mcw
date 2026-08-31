"""Original vs Oracle-G vs Clean-G geometry ablation flags (no learned G).

Oracle G_t is the known forward corruption of the clean target assignment.
Clean-G always feeds SCF the exact GT partition A_0, never noisy A_t.
"""
from __future__ import annotations

from typing import Any

ORACLE_G_CONSTRUCTION = (
    "A_t = simulate_forward_ctmc(GT_A0, uniform legal pi, fixed-exit beta_R/beta_G)"
    ".state_at(t); same t as X_t/L_t. SCF consumes C/copy_of/orbit_of from this A_t."
)
ORACLE_G_NOT = (
    "not GJumpHead prediction; not clean G_0/A_0 reused at t>0 unless the forward "
    "path had no jumps before t"
)
CLEAN_G_CONSTRUCTION = (
    "A_cond = A_0^GT (symmetry-augmented clean role/copy partition) at every t. "
    "X_t/L_t still use the native geometry noise at t. SCF never sees "
    "traj.state_at(t) / forward-noised A_t."
)
CLEAN_G_NOT = (
    "not noisy-trajectory oracle A_t; not GJumpHead; not learned G"
)


def _as_bool(v, default: bool = True) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def resolve_geometry_ablation_arm(
    *,
    ablation_arm: str | None = None,
    geometry_assignment_conditioning: Any = None,
    train_assignment_heads: Any = None,
    yaml_geom_cond: bool = True,
) -> dict[str, Any]:
    """Map CLI/yaml onto geometry-assignment ablation arms.

    original:      SCF off,  L_geom only
    oracle_g:      SCF on,   L_geom only, A_t = forward(GT)  (noisy-trajectory oracle)
    clean_g:       SCF on,   L_geom only, A_cond = A_0^GT at every t
    g_conditioned: SCF on,   L_geom + L_G/L_R (learned G; not this task)
    """
    arm = str(ablation_arm or "").strip().lower() or None
    if arm in ("orig", "original"):
        return {
            "ablation_arm": "original",
            "geometry_assignment_conditioning": False,
            "train_assignment_heads": False,
            "oracle_g": False,
            "clean_g": False,
        }
    if arm in ("oracle_g", "oracle-g", "oracle"):
        return {
            "ablation_arm": "oracle_g",
            "geometry_assignment_conditioning": True,
            "train_assignment_heads": False,
            "oracle_g": True,
            "clean_g": False,
        }
    if arm in ("clean_g", "clean-g", "clean_g_oracle", "clean"):
        return {
            "ablation_arm": "clean_g",
            "geometry_assignment_conditioning": True,
            "train_assignment_heads": False,
            "oracle_g": False,
            "clean_g": True,
        }
    if arm in ("g_conditioned", "g-conditioned", "learned_g"):
        return {
            "ablation_arm": "g_conditioned",
            "geometry_assignment_conditioning": True,
            "train_assignment_heads": True,
            "oracle_g": False,
            "clean_g": False,
        }
    geom = (
        yaml_geom_cond
        if geometry_assignment_conditioning is None
        else _as_bool(geometry_assignment_conditioning, True)
    )
    if train_assignment_heads is None:
        train_a = bool(geom)
    else:
        train_a = _as_bool(train_assignment_heads, False)
    train_a = bool(train_a) and bool(geom)
    if geom and not train_a:
        return {
            "ablation_arm": "oracle_g",
            "geometry_assignment_conditioning": True,
            "train_assignment_heads": False,
            "oracle_g": True,
            "clean_g": False,
        }
    return {
        "ablation_arm": "g_conditioned" if geom else "original",
        "geometry_assignment_conditioning": bool(geom),
        "train_assignment_heads": bool(train_a),
        "oracle_g": False,
        "clean_g": False,
    }
