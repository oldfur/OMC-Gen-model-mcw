# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Generic, TypeVar
import warnings

import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.losses import Loss
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.diffusion.score_models.base import ScoreModel
from mattergen.diffusion.timestep_samplers import TimestepSampler, UniformTimestepSampler

T = TypeVar("T", bound=BatchedData)
BatchTransform = Callable[[T], T]


def identity(x: T) -> T:
    return x


class DiffusionModule(torch.nn.Module, Generic[T]):
    """Denoising diffusion model for a multi-part state"""

    def __init__(
        self,
        model: ScoreModel[T],
        corruption: MultiCorruption[T],
        loss_fn: Loss,
        pre_corruption_fn: BatchTransform | None = None,
        timestep_sampler: TimestepSampler | None = None,
        assignment_diffusion_enabled: bool | None = None,
        assignment_latent_mode: str | None = None,
        role_partition_diffusion_type: str = "discrete_constrained",
        role_diffusion_type: str = "masked_capacity",
        matching_diffusion_type: str = "masked_permutation",
        role_context_mode: str = "geometry_only",
        global_copy_assembly: dict | None = None,
        assignment_diffusion_loss_weight: float = 1.0,
        assignment_diffusion_steps: int = 1000,
        assignment_clean_logit_scale: float = 8.0,
        assignment_sinkhorn_tau_max: float = 1.0,
        assignment_sinkhorn_tau_min: float = 0.10,
        assignment_sinkhorn_max_iter: int = 300,
        assignment_sinkhorn_tol: float = 1e-6,
        assignment_prediction_type: str = "epsilon",
    ) -> None:
        super().__init__()
        self.model = model
        self.corruption = corruption
        self.loss_fn = loss_fn
        self.pre_corruption_fn = pre_corruption_fn or identity
        self.model_targets = {k: ModelTarget(v) for k, v in loss_fn.model_targets.items()}

        self.timestep_sampler = timestep_sampler or UniformTimestepSampler(
            min_t=1e-5,
            max_t=corruption.T,
        )
        valid_modes = {"none", "full_assignment", "role_partition"}
        if assignment_latent_mode is not None and assignment_latent_mode not in valid_modes:
            raise ValueError(f"assignment_latent_mode must be one of {sorted(valid_modes)}, got {assignment_latent_mode!r}")
        if role_partition_diffusion_type != "discrete_constrained":
            raise ValueError("only role_partition_diffusion_type='discrete_constrained' is implemented")
        if role_diffusion_type not in {"masked_capacity", "swap_gibbs"}:
            raise ValueError("role_diffusion_type must be 'masked_capacity' or 'swap_gibbs'")
        if matching_diffusion_type != "masked_permutation":
            raise ValueError("only matching_diffusion_type='masked_permutation' is implemented")
        if role_context_mode not in {"geometry_only", "oracle_same_copy", "oracle_copy_local"}:
            raise ValueError("role_context_mode must be geometry_only, oracle_same_copy, or oracle_copy_local")
        if assignment_latent_mode is None:
            assignment_latent_mode = "full_assignment" if assignment_diffusion_enabled else "none"
        elif assignment_diffusion_enabled is not None and assignment_diffusion_enabled != (assignment_latent_mode == "full_assignment"):
            warnings.warn(
                "assignment_latent_mode takes precedence over conflicting legacy "
                "assignment_diffusion_enabled",
                stacklevel=2,
            )
        self.assignment_latent_mode = assignment_latent_mode
        self.role_partition_diffusion_type = role_partition_diffusion_type
        self.role_diffusion_type = role_diffusion_type
        self.matching_diffusion_type = matching_diffusion_type
        # This is metadata/default routing only.  Oracle modes are implemented
        # by the fixed-sample diagnostic and are never used by normal sampling.
        self.role_context_mode = role_context_mode
        # Reserved standalone-research configuration.  The clean/oracle-R
        # assembly package is deliberately not instantiated here, so disabled
        # (and even enabled) config cannot alter pos/cell diffusion behaviour.
        self.global_copy_assembly_config = {"enabled": False} if global_copy_assembly is None else dict(global_copy_assembly)
        self.assignment_diffusion_enabled = assignment_latent_mode == "full_assignment"
        self.assignment_diffusion_loss_weight = assignment_diffusion_loss_weight
        if assignment_latent_mode == "full_assignment":
            from mattergen.common.assignment_diffusion import AssignmentDiffusion
            self.assignment_diffusion = AssignmentDiffusion(steps=assignment_diffusion_steps, clean_logit_scale=assignment_clean_logit_scale, tau_max=assignment_sinkhorn_tau_max, tau_min=assignment_sinkhorn_tau_min, sinkhorn_max_iter=assignment_sinkhorn_max_iter, sinkhorn_tol=assignment_sinkhorn_tol, prediction_type=assignment_prediction_type)
        else:
            self.assignment_diffusion = None
        if assignment_latent_mode == "role_partition":
            from mattergen.common.role_partition_diffusion import RolePartitionDiffusion
            self.role_partition_diffusion = RolePartitionDiffusion(role_diffusion_type=role_diffusion_type, role_context_mode=role_context_mode)
        else:
            self.role_partition_diffusion = None

        # Check corruption for nn.Modules and register them here.
        self._register_corruption_modules()

    def _register_corruption_modules(self):
        """
        Register corruptions that are instances of `torch.nn.Module`s for proper device, parameter,
        etc handling.
        """
        assert isinstance(self.corruption, MultiCorruption)
        for idx, (key, _corruption) in enumerate(self.corruption._corruptions.items()):
            if isinstance(_corruption, torch.nn.Module):
                self.register_module(f"MultiCorruption:{idx}:{key}", _corruption)

    def calc_loss(
        self, batch: T, node_is_unmasked: torch.LongTensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate loss and metrics given a batch of clean data which may include
        context/conditioning fields. Add noise, predict score using score model, then calculate
        loss.

        Args:
            batch: batch of training data
            node_is_unmasked: mask that has a value 1 for nodes that are included in the loss, and
                a value of 0 for nodes that should be ignored. If None, all nodes are included.

        Returns:
            loss: the loss for the batch
            metrics: a dictionary of metrics for the batch
        """
        batch = self.pre_corruption_fn(batch)

        noisy_batch, t = self._corrupt_batch(batch)

        score_model_output = self.model(noisy_batch, t)
        loss, metrics = self.loss_fn(
            multi_corruption=self.corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
            node_is_unmasked=node_is_unmasked,
        )
        if self.assignment_diffusion is not None:
            assignment_loss, marginal_error = self.assignment_diffusion.loss(batch, t)
            loss = loss + self.assignment_diffusion_loss_weight * assignment_loss
            metrics = {**metrics, "assignment_diffusion_loss": assignment_loss.detach(), "assignment_diffusion_marginal_error": marginal_error.detach()}
        if self.role_partition_diffusion is not None:
            role_loss, role_metrics = self.role_partition_diffusion.loss(batch)
            loss = loss + role_loss
            metrics = {**metrics, **{f"role_partition_{k}": v.detach() for k, v in role_metrics.items()}}
        assert loss.numel() == 1

        return loss, metrics

    def _corrupt_batch(
        self,
        batch: T,
    ) -> tuple[T, torch.Tensor]:
        """
        Corrupt a batch of data for use in a training step:
        - sample a different timestep for each sample in the batch
        - add noise according to the corruption process

        Args:
            batch: Batch of clean states

        Returns:
            noisy_batch: batch of noisy samples
            t: the timestep used for each sample in the batch

        """
        # Sample timesteps
        t = self.sample_timesteps(batch)

        # Add noise to data
        noisy_batch = self.corruption.sample_marginal(batch, t)

        return noisy_batch, t

    def score_fn(self, x: T, t: torch.Tensor) -> T:
        """Calculate the score of a batch of data at a given timestep

        Args:
            x: batch of data
            t: timestep

        Returns:
            score: score of the batch of data at the given timestep
        """
        model_out: T = self.model(x, t)
        fns = {k: convert_model_out_to_score for k in self.corruption.sdes.keys()}

        scores = apply(
            fns=fns,
            model_out=model_out,
            broadcast=dict(t=t, batch=x),
            sde=self.corruption.sdes,
            model_target=self.model_targets,
            batch_idx=self.corruption._get_batch_indices(x),
        )

        return model_out.replace(**scores)

    def sample_timesteps(self, batch: T) -> torch.Tensor:
        """Sample the timesteps, which will be used to determine how much noise
        to add to data.

        Args:
           batch: batch of data to be corrupted

        Returns: sampled timesteps
        """
        return self.timestep_sampler(
            batch_size=batch.get_batch_size(),
            device=self._get_device(batch),
        )

    def _get_device(self, batch: T) -> torch.device:
        return next(batch[k].device for k in self.corruption.sdes.keys())
