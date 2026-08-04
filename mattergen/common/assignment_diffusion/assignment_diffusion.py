"""Masked, gauge-fixed assignment DDPM.

This module is deliberately independent from the coordinate/cell diffusion path.
In particular, role columns are canonical molecular roles, never packed crystal
slots.  The public helpers below make that contract testable before a trajectory
or predictor is allowed to consume a packed batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn


def _error(message: str) -> None:
    raise ValueError(f"invalid packed assignment input: {message}")


def _integer_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        _error(f"{name} must be a tensor")
    if value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        _error(f"{name} must have integer dtype, got {value.dtype}")


def gauge_center(y: torch.Tensor, mask: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Project allowed logits onto the row/column-zero masked gauge subspace."""
    if y.shape != mask.shape:
        _error(f"logits shape {tuple(y.shape)} != mask shape {tuple(mask.shape)}")
    if y.ndim != 2 or mask.dtype is not torch.bool:
        _error("logits/mask must be a 2-D tensor and bool mask")
    if not torch.isfinite(y).all():
        _error("logits contain NaN or Inf")
    if not bool(mask.any(1).all()) or not bool(mask.any(0).all()):
        _error("assignment mask has empty row/column")
    x = y.masked_fill(~mask, 0.0)
    for _ in range(iterations):
        x = (x - x.sum(1, keepdim=True) / mask.sum(1, keepdim=True)).masked_fill(~mask, 0.0)
        x = (x - x.sum(0, keepdim=True) / mask.sum(0, keepdim=True)).masked_fill(~mask, 0.0)
    return x


def sinkhorn(y: torch.Tensor, mask: torch.Tensor, iters: int = 300, tol: float = 1e-6) -> torch.Tensor:
    """Masked log-domain Sinkhorn; forbidden entries retain exactly zero mass."""
    del tol  # fixed iteration count is intentional and deterministic
    if y.shape != mask.shape:
        _error(f"logits shape {tuple(y.shape)} != mask shape {tuple(mask.shape)}")
    if y.ndim != 2 or mask.dtype is not torch.bool:
        _error("logits/mask must be a 2-D tensor and bool mask")
    if not torch.isfinite(y).all():
        _error("logits contain NaN or Inf")
    if not bool(mask.any(1).all()) or not bool(mask.any(0).all()):
        _error("assignment mask has empty row/column")
    dtype = torch.float64 if y.dtype in (torch.float16, torch.bfloat16, torch.float32) else y.dtype
    neg = torch.finfo(dtype).min
    log_p = y.to(dtype).masked_fill(~mask, neg)
    for _ in range(iters):
        log_p = (log_p - torch.logsumexp(log_p, 1, keepdim=True)).masked_fill(~mask, neg)
        log_p = (log_p - torch.logsumexp(log_p, 0, keepdim=True)).masked_fill(~mask, neg)
    return log_p.exp().masked_fill(~mask, 0.0).to(y.dtype)


@dataclass(frozen=True)
class PackedAssignmentSample:
    sample: int
    slots: torch.Tensor
    copies: torch.Tensor
    roles: torch.Tensor
    role_slots: torch.Tensor
    role_atomic_numbers: torch.Tensor
    role_degree: torch.Tensor


def _get(batch: Any, name: str) -> Any:
    if not hasattr(batch, name):
        _error(f"missing {name}")
    return getattr(batch, name)


def _batch_index(batch: Any, n: int) -> torch.Tensor:
    try:
        value = batch.get_batch_idx("pos")
    except (AttributeError, KeyError):
        value = getattr(batch, "batch", None)
    if value is None:
        value = getattr(batch, "batch", None)
    if value is None:
        _error("missing packed batch index for pos")
    _integer_tensor(value, "batch index")
    if value.ndim != 1 or value.numel() != n:
        _error(f"packed batch index length {value.numel()} != node count {n}")
    if n == 0:
        _error("entire batch is empty")
    if (value < 0).any():
        _error("batch index is negative")
    if bool((value[1:] < value[:-1]).any()):
        _error("batch index is non-monotonic")
    unique = torch.unique_consecutive(value)
    expected = torch.arange(unique.numel(), device=value.device, dtype=value.dtype)
    if not torch.equal(unique, expected):
        _error("batch index is non-contiguous or out of range")
    return value.long()


def validate_packed_assignment_batch(batch: Any, *, full: bool = True) -> list[PackedAssignmentSample]:
    """Validate packed OMC assignment metadata and return canonical per-sample views.

    ``full=False`` retains the O(N) structural checks used in production.  The
    full mode also validates graph endpoints and optional declared graph size;
    it is used by diagnostics before loss/predictor execution.
    """
    pos = _get(batch, "pos")
    atomic_numbers = _get(batch, "atomic_numbers")
    copy_id = _get(batch, "mol_copy_id")
    role_id = _get(batch, "mol_atom_id")
    if not isinstance(pos, torch.Tensor) or pos.ndim != 2 or pos.shape[1] != 3:
        _error("pos must have shape [N, 3]")
    n = pos.shape[0]
    if n == 0:
        _error("entire batch is empty")
    if not torch.isfinite(pos).all():
        _error("coordinates contain NaN or Inf")
    _integer_tensor(atomic_numbers, "atomic_numbers")
    _integer_tensor(copy_id, "mol_copy_id")
    _integer_tensor(role_id, "mol_atom_id")
    for name, value in (("atomic_numbers", atomic_numbers), ("mol_copy_id", copy_id), ("mol_atom_id", role_id)):
        if value.ndim != 1 or value.numel() != n:
            _error(f"{name} length {value.numel()} != node count {n}")
    batch_index = _batch_index(batch, n)
    num_samples = int(batch_index[-1]) + 1
    cell = _get(batch, "cell")
    if not isinstance(cell, torch.Tensor) or cell.ndim < 2 or cell.shape[0] != num_samples:
        _error(f"cell leading dimension must equal packed batch size {num_samples}")
    if not torch.isfinite(cell).all():
        _error("cell contains NaN or Inf")
    declared_z = getattr(batch, "mol_num_molecules", None)
    if declared_z is not None:
        _integer_tensor(declared_z, "mol_num_molecules")
        if declared_z.numel() != num_samples:
            _error("mol_num_molecules length does not equal packed batch size")
    edge = getattr(batch, "mol_bond_edge_index", None)
    if full and edge is not None:
        _integer_tensor(edge, "mol_bond_edge_index")
        if edge.ndim != 2 or edge.shape[0] != 2:
            _error("mol_bond_edge_index must have shape [2, E]")
        if edge.numel() and ((edge < 0).any() or (edge >= n).any()):
            _error("mol_bond_edge_index contains out-of-range node index")
        if edge.numel() and not torch.equal(batch_index[edge[0]], batch_index[edge[1]]):
            _error("mol_bond_edge_index crosses packed samples")
    declared_m = getattr(batch, "target_molecular_graph_num_nodes", None)
    if declared_m is not None:
        _integer_tensor(declared_m, "target_molecular_graph_num_nodes")
        if declared_m.numel() != num_samples:
            _error("target_molecular_graph_num_nodes length does not equal packed batch size")

    result: list[PackedAssignmentSample] = []
    for sample in range(num_samples):
        slots = (batch_index == sample).nonzero().flatten()
        if slots.numel() == 0:
            _error(f"sample {sample} is empty")
        copies = copy_id[slots].long()
        roles = role_id[slots].long()
        if (copies < 0).any():
            _error(f"sample {sample} has negative/illegal mol_copy_id")
        if (roles < 0).any():
            _error(f"sample {sample} has negative/illegal mol_atom_id")
        unique_copies = torch.unique(copies, sorted=True)
        expected_copies = torch.arange(unique_copies.numel(), device=slots.device)
        if not torch.equal(unique_copies, expected_copies):
            _error(f"sample {sample} mol_copy_id must be exactly [0, Z)")
        unique_roles = torch.unique(roles, sorted=True)
        m, z = int(unique_roles.numel()), int(unique_copies.numel())
        expected_roles = torch.arange(m, device=slots.device)
        if not torch.equal(unique_roles, expected_roles):
            _error(f"sample {sample} mol_atom_id must be exactly [0, M)")
        if int(slots.numel()) != z * m:
            _error(f"sample {sample} requires N = Z * M, got N={slots.numel()}, Z={z}, M={m}")
        if declared_z is not None and int(declared_z[sample]) != z:
            _error(f"sample {sample} declared Z={int(declared_z[sample])} but found {z}")
        if declared_m is not None and int(declared_m[sample]) != m:
            _error(f"sample {sample} target molecular graph node count != M ({int(declared_m[sample])} != {m})")
        role_slots: list[int] = []
        reference_elements: torch.Tensor | None = None
        for copy in range(z):
            copy_local = (copies == copy).nonzero().flatten()
            copy_roles = roles[copy_local]
            if copy_local.numel() != m:
                _error(f"sample {sample} copy {copy} has wrong role count")
            counts = torch.bincount(copy_roles, minlength=m)
            if bool((counts == 0).any()):
                _error(f"sample {sample} copy {copy} is missing a canonical role")
            if bool((counts > 1).any()):
                _error(f"sample {sample} copy {copy} has a duplicate canonical role")
            if not torch.equal(torch.sort(copy_roles).values, expected_roles):
                _error(f"sample {sample} copy {copy} role set differs from canonical role set")
            local_by_role = torch.empty(m, device=slots.device, dtype=torch.long)
            local_by_role[copy_roles] = copy_local
            elems = atomic_numbers[slots[local_by_role]].long()
            if reference_elements is None:
                reference_elements = elems
            elif not torch.equal(elems, reference_elements):
                _error(f"sample {sample} copy {copy} has different element composition by role")
            if copy == 0:
                role_slots = slots[local_by_role].tolist()
        assert reference_elements is not None
        degree = torch.zeros(m, device=slots.device, dtype=pos.dtype)
        if edge is not None and edge.numel():
            sample_edge = edge[:, batch_index[edge[0]] == sample]
            if sample_edge.numel():
                # Degree is a canonical-role feature: average the repeated copies.
                degree.scatter_add_(0, role_id[sample_edge[0]].long(), torch.ones(sample_edge.shape[1], device=slots.device, dtype=pos.dtype))
                degree /= z
        result.append(PackedAssignmentSample(sample, slots, unique_copies, unique_roles, torch.tensor(role_slots, device=slots.device), reference_elements, degree))
    return result


class AssignmentDiffusion(nn.Module):
    def __init__(self, hidden_dim: int = 128, steps: int = 1000, clean_logit_scale: float = 8.0, tau_max: float = 1.0, tau_min: float = 0.1, sinkhorn_max_iter: int = 300, sinkhorn_tol: float = 1e-6, prediction_type: str = "epsilon"):
        super().__init__()
        if prediction_type != "epsilon":
            raise ValueError(f"unsupported assignment prediction_type={prediction_type!r}; expected 'epsilon'")
        self.steps, self.kappa, self.tau_max, self.tau_min = steps, clean_logit_scale, tau_max, tau_min
        self.iters, self.tol, self.prediction_type = sinkhorn_max_iter, sinkhorn_tol, prediction_type
        beta = torch.linspace(1e-4, 0.02, steps)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))
        self.atom = nn.Embedding(119, hidden_dim)
        self.role = nn.Embedding(119, hidden_dim)
        self.geometry = nn.Sequential(nn.Linear(12, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.role_graph = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(nn.Linear(3 * hidden_dim + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _target(self, batch: Any, info: PackedAssignmentSample, *, copy_permutation: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return clean target in a random copy gauge and canonical column metadata."""
        slots, m, z = info.slots, info.roles.numel(), info.copies.numel()
        local_copy = batch.mol_copy_id[slots].long()
        local_role = batch.mol_atom_id[slots].long()
        clean = torch.zeros((slots.numel(), slots.numel()), device=slots.device, dtype=torch.float32)
        clean[torch.arange(slots.numel(), device=slots.device), local_copy * m + local_role] = 1
        if copy_permutation is None:
            copy_permutation = torch.randperm(z, device=slots.device)
        _integer_tensor(copy_permutation, "copy permutation")
        if copy_permutation.ndim != 1 or copy_permutation.numel() != z or not torch.equal(torch.sort(copy_permutation).values, torch.arange(z, device=slots.device)):
            _error("copy permutation is not a permutation of [0, Z)")
        columns = torch.cat([torch.arange(copy * m, (copy + 1) * m, device=slots.device) for copy in copy_permutation])
        role_slots = info.role_slots.repeat(z)[columns % m]
        role_z = info.role_atomic_numbers.repeat(z)[columns % m]
        return clean[:, columns], batch.atomic_numbers[slots].long(), role_z, role_slots

    def predict_epsilon(self, y_t: torch.Tensor, *, crystal_atomic_numbers: torch.Tensor, role_atomic_numbers: torch.Tensor, role_degree: torch.Tensor, crystal_pos: torch.Tensor, cell: torch.Tensor, assignment_timestep: int | torch.Tensor, crystal_timestep: float | torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Actual epsilon predictor; equivariant to independent row/column permutations.

        It uses no flattened matrix, absolute slot embedding, copy index, or
        canonical-role-id embedding.  The only cross-entry aggregation is the
        columnwise soft assignment aggregate, which permutes with columns.
        """
        n = y_t.shape[0]
        if y_t.ndim != 2 or y_t.shape[1] != n or mask.shape != y_t.shape:
            _error("assignment logits/mask must be square and shape-aligned")
        for name, value in (("crystal_atomic_numbers", crystal_atomic_numbers), ("role_atomic_numbers", role_atomic_numbers), ("role_degree", role_degree)):
            if value.ndim != 1 or value.numel() != n:
                _error(f"{name} must have shape [N] matching assignment rows/columns")
        if crystal_pos.shape != (n, 3):
            _error("crystal_pos must have shape [N, 3]")
        if cell.numel() != 9:
            _error("cell must contain exactly 9 values")
        if not torch.isfinite(crystal_pos).all() or not torch.isfinite(cell).all() or not torch.isfinite(role_degree).all():
            _error("predictor features contain NaN or Inf")
        _integer_tensor(crystal_atomic_numbers, "crystal_atomic_numbers")
        _integer_tensor(role_atomic_numbers, "role_atomic_numbers")
        gauge_center(y_t, mask)  # validates mask/logits without changing caller values
        at = sinkhorn(y_t, mask, self.iters, self.tol)
        hi = self.atom(crystal_atomic_numbers) + self.geometry(torch.cat([crystal_pos, cell.reshape(1, 9).expand(n, -1)], -1))
        hr = self.role(role_atomic_numbers) + self.role_graph(role_degree[:, None])
        aggregate = (at.T @ hi) / at.sum(0)[:, None].clamp_min(1e-8)
        assignment_time = torch.as_tensor(assignment_timestep, device=y_t.device, dtype=y_t.dtype) / self.steps
        crystal_time = torch.as_tensor(crystal_timestep, device=y_t.device, dtype=y_t.dtype)
        extra = torch.stack([y_t, at, assignment_time.expand_as(y_t), crystal_time.expand_as(y_t)], -1)
        result = self.net(torch.cat([hi[:, None].expand(-1, n, -1), hr[None].expand(n, -1, -1), aggregate[None].expand(n, -1, -1), extra], -1)).squeeze(-1)
        if not torch.isfinite(result).all():
            _error("predictor produced NaN or Inf")
        return result.masked_fill(~mask, 0.0)

    def posterior(self, y_t: torch.Tensor, x0: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form q(y[t-1] | y[t], y[0]), before gauge projection."""
        if not 0 <= step < self.steps:
            raise ValueError(f"step {step} outside [0, {self.steps})")
        if step == 0:
            return x0, y_t.new_zeros(())
        ab = self.alpha_bar[step].to(dtype=y_t.dtype, device=y_t.device)
        previous = self.alpha_bar[step - 1].to(dtype=y_t.dtype, device=y_t.device)
        alpha_t, beta_t = ab / previous, 1 - ab / previous
        denominator = (1 - ab).clamp_min(torch.finfo(y_t.dtype).eps)
        coefficient_x0 = previous.sqrt() * beta_t / denominator
        coefficient_xt = alpha_t.sqrt() * (1 - previous) / denominator
        variance = (beta_t * (1 - previous) / denominator).clamp_min(0)
        return coefficient_x0 * x0 + coefficient_xt * y_t, variance

    def reverse_step(self, y_t: torch.Tensor, mask: torch.Tensor, predict_epsilon: Any, step: int, *, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One projected reverse DDPM transition, exposed for algebraic oracle tests."""
        eps = gauge_center(predict_epsilon(y_t, torch.tensor(step, device=y_t.device)), mask)
        ab = self.alpha_bar[step].to(dtype=y_t.dtype, device=y_t.device)
        x0 = gauge_center((y_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(torch.finfo(y_t.dtype).eps), mask)
        if step == 0:
            return x0, x0, y_t.new_zeros(())
        mean, variance = self.posterior(y_t, x0, step)
        if noise is None:
            noise = torch.randn_like(y_t)
        projected_noise = gauge_center(noise, mask)
        return gauge_center(mean + variance.sqrt() * projected_noise, mask), x0, variance

    def loss(self, batch: Any, t_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        infos = validate_packed_assignment_batch(batch, full=False)
        if t_x.ndim != 1 or t_x.numel() != len(infos):
            _error("crystal timestep count does not equal packed batch size")
        total, errors = [], []
        for info in infos:
            a0, z, role_z, role_slots = self._target(batch, info)
            mask = z[:, None].eq(role_z[None, :])
            y0 = gauge_center(self.kappa * (2 * a0 - 1), mask)
            assignment_t = torch.randint(self.steps, (1,), device=info.slots.device)
            ab = self.alpha_bar[assignment_t].to(y0)
            eps = gauge_center(torch.randn_like(y0), mask)
            y_t = gauge_center(ab.sqrt() * y0 + (1 - ab).sqrt() * eps, mask)
            prediction = self.predict_epsilon(y_t, crystal_atomic_numbers=z, role_atomic_numbers=role_z, role_degree=info.role_degree.repeat(info.copies.numel())[torch.arange(y_t.shape[1], device=y_t.device) % info.roles.numel()], crystal_pos=batch.pos[info.slots], cell=batch.cell[info.sample], assignment_timestep=assignment_t, crystal_timestep=t_x[info.sample], mask=mask)
            total.append((prediction - eps)[mask].square().mean())
            at = sinkhorn(y_t, mask, self.iters, self.tol)
            errors.append((at.sum(1) - 1).abs().max().maximum((at.sum(0) - 1).abs().max()))
        return torch.stack(total).mean(), torch.stack(errors).max()

    @torch.no_grad()
    def reverse_ddpm(self, mask: torch.Tensor, predict_epsilon: Any, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
        generator = torch.Generator(device=mask.device).manual_seed(seed)
        y = gauge_center(torch.randn(mask.shape, device=mask.device, generator=generator), mask)
        trace = []
        for step in range(self.steps - 1, -1, -1):
            noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator) if step else None
            y, _, _ = self.reverse_step(y, mask, predict_epsilon, step, noise=noise)
            assignment = sinkhorn(y, mask, self.iters, self.tol)
            trace.append({"step": float(step), "entropy": float(-(assignment[assignment > 0] * assignment[assignment > 0].log()).mean()), "marginal_error": float((assignment.sum(1) - 1).abs().max().maximum((assignment.sum(0) - 1).abs().max()))})
        return y, sinkhorn(y, mask, self.iters, self.tol), trace
