"""Reuse MatterGen native geometry loss for N2 (denoising score matching)."""
from __future__ import annotations

from typing import Any

import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.losses import Loss


def mattergen_geometry_loss(
    *,
    loss_fn: Loss,
    corruption: MultiCorruption,
    clean_batch,
    noisy_batch,
    score_model_output,
    t: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Call the same Loss object used by DiffusionModule.calc_loss for pos/cell.

    score_model_output must be a BatchedData-like object with field keys matching
    the loss_fn (typically pos, cell; atomic_numbers if present).
    """
    loss, metrics = loss_fn(
        multi_corruption=corruption,
        batch=clean_batch,
        noisy_batch=noisy_batch,
        score_model_output=score_model_output,
        t=t,
        node_is_unmasked=None,
    )
    return loss, {k: (float(v.detach()) if torch.is_tensor(v) else v) for k, v in metrics.items()}


class ScoreOutputView:
    """Minimal BatchedData-like wrapper so SummedFieldLoss can index pos/cell."""

    def __init__(self, store: dict, batch_idx: dict | None = None):
        self._store = store
        self._batch_idx = batch_idx or {}

    def __getitem__(self, key: str):
        return self._store[key]

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def get_batch_idx(self, field_name: str):
        return self._batch_idx.get(field_name)

    def replace(self, **kwargs):
        s = dict(self._store)
        s.update(kwargs)
        return ScoreOutputView(s, self._batch_idx)
