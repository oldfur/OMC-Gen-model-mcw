"""Deterministic orbit membership collapse from canonical hard R.

Canonical geometry-only decoder still emits ``R ∈ {0,1}^{N×M}``.  Before
orbit-aware assembly we deterministically form

    R̄_{i o} = Σ_{r ∈ o} R̂_{i r}

so each molecular automorphism orbit is one column.  No orbit-collapsed
decoder is trained; this is a pure post-process of hard labels / one-hot R.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class OrbitPartition:
    """Orbit partition of molecular roles 0..M-1."""

    orbits: tuple[tuple[int, ...], ...]  # length J; each orbit is sorted roles
    role_to_orbit: tuple[int, ...]  # length M
    orbit_sizes: tuple[int, ...]  # m_o = |o|

    @property
    def J(self) -> int:
        return len(self.orbits)

    @property
    def M(self) -> int:
        return len(self.role_to_orbit)

    def singleton_orbit_indices(self) -> list[int]:
        return [j for j, size in enumerate(self.orbit_sizes) if size == 1]

    def non_singleton_orbit_indices(self) -> list[int]:
        return [j for j, size in enumerate(self.orbit_sizes) if size > 1]

    def singleton_roles(self) -> list[int]:
        return [self.orbits[j][0] for j in self.singleton_orbit_indices()]

    def non_singleton_roles(self) -> list[int]:
        roles: list[int] = []
        for j in self.non_singleton_orbit_indices():
            roles.extend(self.orbits[j])
        return roles


def build_orbit_partition(role_orbits: list[list[int]] | list[tuple[int, ...]]) -> OrbitPartition:
    """Build a dense orbit partition from per-role orbit lists or unique orbits.

    Accepts either:
    * unique orbit lists e.g. ``[[0],[1,2],[3],...]``
    * per-role orbit membership lists aligned to roles 0..M-1
      (as in ``role_orbits.json`` values sorted by role key).
    """
    if not role_orbits:
        raise ValueError("role_orbits must be non-empty")
    # Detect per-role form: length == max_role+1 and each entry contains that role.
    max_role = max(max(orbit) for orbit in role_orbits if orbit)
    looks_per_role = len(role_orbits) == max_role + 1 and all(
        i in orbit for i, orbit in enumerate(role_orbits)
    )
    if looks_per_role:
        seen: set[tuple[int, ...]] = set()
        unique: list[tuple[int, ...]] = []
        for orbit in role_orbits:
            key = tuple(sorted(int(r) for r in orbit))
            if key not in seen:
                seen.add(key)
                unique.append(key)
        orbits = tuple(sorted(unique, key=lambda o: (o[0], o)))
    else:
        orbits = tuple(tuple(sorted(int(r) for r in orbit)) for orbit in role_orbits)
        # stable unique
        seen = set()
        uniq: list[tuple[int, ...]] = []
        for orbit in orbits:
            if orbit not in seen:
                seen.add(orbit)
                uniq.append(orbit)
        orbits = tuple(uniq)

    M = max(max(orbit) for orbit in orbits) + 1
    role_to_orbit = [-1] * M
    for j, orbit in enumerate(orbits):
        for role in orbit:
            if role_to_orbit[role] != -1:
                raise ValueError(f"role {role} appears in multiple orbits")
            role_to_orbit[role] = j
    if any(x < 0 for x in role_to_orbit):
        missing = [i for i, x in enumerate(role_to_orbit) if x < 0]
        raise ValueError(f"orbit partition missing roles {missing}")
    sizes = tuple(len(orbit) for orbit in orbits)
    return OrbitPartition(orbits=orbits, role_to_orbit=tuple(role_to_orbit), orbit_sizes=sizes)


def _labels(role_assignment: torch.Tensor, M: int) -> torch.Tensor:
    if role_assignment.ndim == 1:
        labels = role_assignment.long()
    elif role_assignment.ndim == 2:
        if role_assignment.shape[1] != M:
            raise ValueError(f"one-hot R must have shape [N,{M}]")
        labels = role_assignment.argmax(-1).long()
    else:
        raise ValueError("role assignment must be labels [N] or one-hot [N,M]")
    if labels.numel() == 0:
        raise ValueError("empty role assignment")
    if int(labels.min()) < 0 or int(labels.max()) >= M:
        raise ValueError("role assignment out of range")
    return labels


def collapse_roles_to_orbit_membership(
    role_assignment: torch.Tensor,
    partition: OrbitPartition,
) -> torch.Tensor:
    """Return hard orbit membership ``R̄ ∈ {0,1}^{N×J}`` from canonical R.

    Does not modify the geometry-only decoder; pure deterministic collapse.
    """
    labels = _labels(role_assignment, partition.M)
    n = int(labels.numel())
    j = partition.J
    mapping = torch.as_tensor(partition.role_to_orbit, dtype=torch.long, device=labels.device)
    orbit_labels = mapping[labels]
    bar = torch.zeros(n, j, dtype=torch.float32, device=labels.device)
    bar[torch.arange(n, device=labels.device), orbit_labels] = 1.0
    # Row one-hot.
    if not torch.allclose(bar.sum(-1), torch.ones(n, device=labels.device)):
        raise AssertionError("orbit membership rows must be one-hot")
    return bar


def orbit_atom_sets(bar_r: torch.Tensor, partition: OrbitPartition) -> dict[int, torch.Tensor]:
    """Map orbit index -> sorted crystal atom indices in that orbit."""
    if bar_r.ndim != 2 or bar_r.shape[1] != partition.J:
        raise ValueError(f"bar_r must have shape [N,{partition.J}]")
    return {
        j: (bar_r[:, j] > 0.5).nonzero(as_tuple=False).flatten().sort().values
        for j in range(partition.J)
    }


def validate_orbit_global_capacity(bar_r: torch.Tensor, partition: OrbitPartition, *, K: int) -> dict[str, object]:
    """Check global orbit capacities: |V_o| = K * |o|."""
    sets = orbit_atom_sets(bar_r, partition)
    sizes = {j: int(nodes.numel()) for j, nodes in sets.items()}
    expected = {j: K * partition.orbit_sizes[j] for j in range(partition.J)}
    ok = all(sizes[j] == expected[j] for j in range(partition.J))
    return {"valid": ok, "sizes": sizes, "expected": expected}


def validate_orbit_copy_capacity(G: torch.Tensor, bar_r: torch.Tensor, partition: OrbitPartition) -> dict[str, object]:
    """Check ``R̄^T G = m 1_K^T`` with m_o = |o|."""
    if G.ndim != 2 or bar_r.ndim != 2 or G.shape[0] != bar_r.shape[0]:
        raise ValueError("G and bar_r must share atom dimension N")
    counts = bar_r.transpose(0, 1) @ G  # [J,K]
    expected = torch.as_tensor(partition.orbit_sizes, dtype=counts.dtype, device=counts.device).unsqueeze(-1)
    expected = expected.expand_as(counts)
    ok = bool(torch.allclose(counts, expected))
    return {
        "valid": ok,
        "counts": counts.detach().cpu().tolist(),
        "expected_per_copy": list(partition.orbit_sizes),
    }
