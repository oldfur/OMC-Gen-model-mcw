"""Instantiation-only audit proving assignment latent modes do not co-instantiate."""
from __future__ import annotations
from argparse import Namespace
import json
from pathlib import Path
import warnings

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import VPSDE
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.model_target import ModelTarget


def module(**kw):
    return DiffusionModule(
        model=lambda batch, t: batch,
        corruption=MultiCorruption(sdes={"x": VPSDE()}),
        loss_fn=Namespace(model_targets={"x": ModelTarget.score_times_std}),
        **kw,
    )


def main() -> None:
    none = module(assignment_latent_mode="none")
    full = module(assignment_latent_mode="full_assignment")
    role = module(assignment_latent_mode="role_partition")
    swap = module(assignment_latent_mode="role_partition", role_diffusion_type="swap_gibbs", matching_diffusion_type="masked_permutation")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conflict = module(assignment_diffusion_enabled=True, assignment_latent_mode="role_partition")
    result = {
        "none": {"assignment": none.assignment_diffusion is not None, "role_partition": none.role_partition_diffusion is not None},
        "full_assignment": {"assignment": full.assignment_diffusion is not None, "role_partition": full.role_partition_diffusion is not None},
        "role_partition": {"assignment": role.assignment_diffusion is not None, "role_partition": role.role_partition_diffusion is not None},
        "role_partition_swap": {"assignment": swap.assignment_diffusion is not None, "role_partition": swap.role_partition_diffusion is not None, "role_diffusion_type": swap.role_diffusion_type, "matching_diffusion_type": swap.matching_diffusion_type},
        "conflict_warning": [str(w.message) for w in caught],
        "conflict_mode": conflict.assignment_latent_mode,
    }
    result["PASS"] = result["none"] == {"assignment": False, "role_partition": False} and result["full_assignment"] == {"assignment": True, "role_partition": False} and result["role_partition"] == {"assignment": False, "role_partition": True} and result["role_partition_swap"] == {"assignment": False, "role_partition": True, "role_diffusion_type": "swap_gibbs", "matching_diffusion_type": "masked_permutation"} and bool(result["conflict_warning"])
    out = Path("outputs/assignment_diffusion_mvp/role_partition_discrete_constrained")
    (out / "mode_router_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
