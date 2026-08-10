"""Adapter that **only** calls MatterGen-native forward corruption.

Provenance
----------
Training geometry noising for pos/cell is performed by:

    DiffusionModule._corrupt_batch
      → UniformTimestepSampler (or configured TimestepSampler)
      → MultiCorruption.sample_marginal(batch, t)
          → per-field Corruption.sample_marginal
              pos: NumAtomsVarianceAdjustedWrappedVESDE (wrapped fractional VESDE)
              cell: LatticeVPSDE

This adapter constructs the **same** MultiCorruption classes used by
``mattergen/conf/lightning_module/diffusion_module/corruption/csp.yaml`` /
``default.yaml`` and calls ``sample_marginal`` — it does **not** reimplement
schedules.

noise_source = mattergen_native_forward_process
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    NumAtomsVarianceAdjustedWrappedVESDE,
)
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.timestep_samplers import UniformTimestepSampler


# Exact class paths used by MatterGen configs (for audit metadata).
PROVENANCE = {
    "noise_source": "mattergen_native_forward_process",
    "diffusion_module_path": "mattergen.diffusion.diffusion_module.DiffusionModule._corrupt_batch",
    "multi_corruption_path": "mattergen.diffusion.corruption.multi_corruption.MultiCorruption.sample_marginal",
    "pos_sde_path": "mattergen.common.diffusion.corruption.NumAtomsVarianceAdjustedWrappedVESDE",
    "cell_sde_path": "mattergen.common.diffusion.corruption.LatticeVPSDE",
    "timestep_sampler_path": "mattergen.diffusion.timestep_samplers.UniformTimestepSampler",
    "config_reference": "mattergen/conf/lightning_module/diffusion_module/corruption/csp.yaml",
    "independent_noise_implementation": False,
}


def build_default_mattergen_corruption(
    *,
    limit_density: float = 0.05,
    pos_sigma_max: float = 5.0,
    pos_sigma_min: float = 0.01,
    cell_beta_min: float = 0.1,
    cell_beta_max: float = 20.0,
    limit_var_scaling_constant: float = 0.25,
) -> MultiCorruption:
    """Instantiate MultiCorruption matching CSP config field classes/hparams.

    Atomic-number D3PM is intentionally omitted: N1 only needs pos/cell noise
    for geometry.  SDEs for pos/cell are the same classes as MatterGen training.
    """
    pos_sde = NumAtomsVarianceAdjustedWrappedVESDE(
        wrapping_boundary=1.0,
        sigma_min=pos_sigma_min,
        sigma_max=pos_sigma_max,
        limit_info_key="num_atoms",
    )
    cell_sde = LatticeVPSDE(
        beta_min=cell_beta_min,
        beta_max=cell_beta_max,
        limit_density=limit_density,
        limit_var_scaling_constant=limit_var_scaling_constant,
    )
    return MultiCorruption(sdes={"cell": cell_sde, "pos": pos_sde})


@dataclass
class MatterGenNoisyGeometry:
    """Result of native MatterGen forward corruption for one crystal."""

    frac_coords_t: torch.Tensor
    lattice_t: torch.Tensor
    t: torch.Tensor
    sigma_x: torch.Tensor
    sigma_l: torch.Tensor
    log_snr_x: torch.Tensor
    log_snr_l: torch.Tensor
    mean_coeff_x: torch.Tensor | None = None
    mean_coeff_l: torch.Tensor | None = None
    noise_source: str = PROVENANCE["noise_source"]
    provenance: dict | None = None


class _BatchView:
    """Minimal BatchedData-like object for MultiCorruption.sample_marginal.

    Must implement ``__contains__``: MultiCorruption.apply does
    ``field_name in batch`` (see multi_corruption.apply).  Objects that only
    define ``__getitem__`` fall back to the old sequence protocol and probe
    ``batch[0], batch[1], ...``, which raises ``KeyError: 0`` for a mapping.
    ``SimpleBatchedData`` implements the same ``__contains__`` contract.
    """

    def __init__(self, store: dict):
        self._store = store

    def __getitem__(self, key: str):
        return self._store[key]

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __iter__(self):
        return iter(self._store)

    def keys(self):
        return self._store.keys()

    def get_batch_size(self) -> int:
        return int(self._store["num_atoms"].shape[0])

    def get_batch_idx(self, field_name: str) -> torch.Tensor | None:
        if field_name == "pos":
            # single crystal: all atoms belong to batch 0 (sparse / per-node field)
            n = int(self._store["pos"].shape[0])
            return torch.zeros(n, dtype=torch.long, device=self._store["pos"].device)
        if field_name == "cell":
            # dense graph-level field (matches ChemGraphBatch / LatticeVPSDE)
            return None
        if field_name == "num_atoms":
            return None
        return None

    def replace(self, **kwargs):
        store = dict(self._store)
        store.update(kwargs)
        return _BatchView(store)


class MatterGenNativeNoiseAdapter:
    """Call MatterGen MultiCorruption.sample_marginal — no custom schedule math."""

    def __init__(
        self,
        corruption: MultiCorruption | None = None,
        *,
        min_t: float = 1e-5,
        max_t: float | None = None,
        limit_density: float = 0.05,
    ):
        self.corruption = corruption or build_default_mattergen_corruption(limit_density=limit_density)
        T = float(self.corruption.T)
        self.timestep_sampler = UniformTimestepSampler(min_t=min_t, max_t=max_t if max_t is not None else T)
        self.provenance = dict(PROVENANCE)
        self.provenance["max_t"] = T
        self.provenance["min_t"] = min_t

    @property
    def T(self) -> float:
        return float(self.corruption.T)

    def sample_t(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        device = device or torch.device("cpu")
        return self.timestep_sampler(batch_size=batch_size, device=device)

    def _field_std(self, field: str, x0: torch.Tensor, t: torch.Tensor, batch: _BatchView) -> tuple[torch.Tensor, torch.Tensor]:
        sde = self.corruption.sdes[field]
        batch_idx = batch.get_batch_idx(field)
        mean, std = sde.marginal_prob(x=x0, t=t, batch_idx=batch_idx, batch=batch)
        return mean, std

    def corrupt_fixed_sample(
        self,
        *,
        frac_coords_0: torch.Tensor,
        lattice_0: torch.Tensor,
        num_atoms: int | torch.Tensor,
        t: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> MatterGenNoisyGeometry:
        """Corrupt one crystal using native MultiCorruption.sample_marginal.

        If ``generator`` is provided, we set global RNG state for the call so that
        SDE ``torch.randn_like`` draws are reproducible (MatterGen SDEs do not
        accept generators directly).
        """
        device = frac_coords_0.device
        if t is None:
            t = self.sample_t(1, device=device)
        else:
            t = t.to(device=device).reshape(-1)
            if t.numel() != 1:
                raise ValueError("fixed-sample N1 adapter expects scalar t per crystal")

        n_atoms = int(num_atoms) if not torch.is_tensor(num_atoms) else int(num_atoms.reshape(-1)[0].item())
        cell = lattice_0
        if cell.ndim == 2:
            cell = cell.unsqueeze(0)
        batch = _BatchView(
            {
                "pos": frac_coords_0,
                "cell": cell,
                "num_atoms": torch.tensor([n_atoms], device=device, dtype=torch.long),
            }
        )

        if generator is not None:
            # Align CPU/GPU RNG with MatterGen's torch.randn_like draws.
            state_cpu = torch.random.get_rng_state()
            state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                noisy = self.corruption.sample_marginal(batch, t)
            finally:
                torch.random.set_rng_state(state_cpu)
                if state_cuda is not None:
                    torch.cuda.set_rng_state_all(state_cuda)
        else:
            noisy = self.corruption.sample_marginal(batch, t)

        frac_t = noisy["pos"]
        cell_t = noisy["cell"]
        if cell_t.ndim == 3 and cell_t.shape[0] == 1:
            cell_t_out = cell_t[0]
        else:
            cell_t_out = cell_t

        # std / SNR diagnostics via the **same** SDE.marginal_prob (not a copy of formulas).
        mean_x, std_x = self._field_std("pos", frac_coords_0, t, batch)
        mean_l, std_l = self._field_std("cell", cell, t, batch)
        # representative scalars
        sx = std_x.reshape(-1)[0]
        sl = std_l.reshape(-1)[0]
        # logSNR ~ log(mean_coeff^2 / std^2); for VE mean_coeff may be 1
        def _log_snr(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            # use ratio of squared scales of first element
            m = mean.reshape(-1)[0].abs().clamp_min(1e-12)
            s = std.reshape(-1)[0].abs().clamp_min(1e-12)
            return 2 * (m.log() - s.log())

        return MatterGenNoisyGeometry(
            frac_coords_t=frac_t,
            lattice_t=cell_t_out,
            t=t.reshape(()),
            sigma_x=sx.detach(),
            sigma_l=sl.detach(),
            log_snr_x=_log_snr(mean_x, std_x).detach(),
            log_snr_l=_log_snr(mean_l, std_l).detach(),
            mean_coeff_x=mean_x.reshape(-1)[0].detach(),
            mean_coeff_l=mean_l.reshape(-1)[0].detach(),
            provenance=dict(self.provenance),
        )

    def corrupt_at_fraction(
        self,
        *,
        frac_coords_0: torch.Tensor,
        lattice_0: torch.Tensor,
        num_atoms: int,
        t_fraction: float,
        generator: torch.Generator | None = None,
    ) -> MatterGenNoisyGeometry:
        """t = t_fraction * T using native T from MultiCorruption."""
        t = torch.tensor([float(t_fraction) * self.T], device=frac_coords_0.device, dtype=torch.float32)
        t = t.clamp(min=1e-5, max=self.T)
        return self.corrupt_fixed_sample(
            frac_coords_0=frac_coords_0,
            lattice_0=lattice_0,
            num_atoms=num_atoms,
            t=t,
            generator=generator,
        )
