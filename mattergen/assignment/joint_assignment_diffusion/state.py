"""Joint assignment state A[i,o,k] with derived R-bar, G, C."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mattergen.assignment.global_copy_assembly.orbit_membership import OrbitPartition


@dataclass
class JointAssignmentState:
    """Legal joint assignment A ∈ {0,1}^{N×J×K}.

    A[i,o,k]=1 means atom i occupies orbit o and copy k.
    Constraints (must always hold):
      sum_{o,k} A[i,o,k] = 1          (each atom one joint slot)
      sum_i A[i,o,k] = m_o            (orbit capacity per copy)
    """

    A: torch.Tensor  # [N, J, K] float {0,1}
    partition: OrbitPartition
    atomic_numbers: torch.Tensor  # [N] long
    element_by_orbit: torch.Tensor  # [J] representative Z for orbit (for legality)

    @property
    def N(self) -> int:
        return int(self.A.shape[0])

    @property
    def J(self) -> int:
        return int(self.A.shape[1])

    @property
    def K(self) -> int:
        return int(self.A.shape[2])

    def bar_r(self) -> torch.Tensor:
        """R-bar [N, J]."""
        return self.A.sum(dim=-1)

    def G(self) -> torch.Tensor:
        """G [N, K]."""
        return self.A.sum(dim=1)

    def C(self) -> torch.Tensor:
        g = self.G()
        return g @ g.T

    def orbit_of(self) -> torch.Tensor:
        return self.bar_r().argmax(dim=-1)

    def copy_of(self) -> torch.Tensor:
        return self.G().argmax(dim=-1)

    def clone(self) -> "JointAssignmentState":
        return JointAssignmentState(
            A=self.A.clone(),
            partition=self.partition,
            atomic_numbers=self.atomic_numbers.clone(),
            element_by_orbit=self.element_by_orbit.clone(),
        )

    def validate(self, tol: float = 1e-5) -> dict[str, bool | int]:
        A = self.A
        row = A.reshape(self.N, -1).sum(-1)
        ok_row = bool(torch.allclose(row, torch.ones_like(row), atol=tol))
        caps_ok = True
        for o, m in enumerate(self.partition.orbit_sizes):
            for k in range(self.K):
                if abs(float(A[:, o, k].sum()) - float(m)) > tol:
                    caps_ok = False
        binary = bool(((A - A.round()).abs() < tol).all() and ((A.round() >= 0) & (A.round() <= 1)).all())
        return {
            "row_ok": ok_row,
            "capacity_ok": caps_ok,
            "binary_ok": binary,
            "legal": ok_row and caps_ok and binary,
        }

    def apply_swap(self, i: int, j: int) -> "JointAssignmentState":
        """Swap joint labels of atoms i and j (preserves capacities)."""
        out = self.clone()
        ai = out.A[i].clone()
        out.A[i] = out.A[j]
        out.A[j] = ai
        return out


def build_element_by_orbit(
    partition: OrbitPartition,
    role_z: torch.Tensor,
) -> torch.Tensor:
    """Representative atomic number per orbit (all roles in orbit share element)."""
    z = []
    for o, roles in enumerate(partition.orbits):
        r0 = roles[0]
        z.append(int(role_z[r0].item()) if role_z.ndim == 1 else int(role_z[r0]))
    return torch.tensor(z, dtype=torch.long, device=role_z.device if torch.is_tensor(role_z) else "cpu")


def a_from_role_and_copy(
    *,
    role: torch.Tensor,
    copy: torch.Tensor,
    partition: OrbitPartition,
    atomic_numbers: torch.Tensor,
    role_z: torch.Tensor,
    K: int,
) -> JointAssignmentState:
    """Build A from per-atom role∈[0,M) and copy∈[0,K)."""
    n = int(role.numel())
    j = partition.J
    A = torch.zeros(n, j, K, dtype=torch.float32, device=role.device)
    for i in range(n):
        r = int(role[i].item())
        o = partition.role_to_orbit[r]
        k = int(copy[i].item())
        A[i, o, k] = 1.0
    elem = build_element_by_orbit(partition, role_z.to(role.device))
    return JointAssignmentState(
        A=A, partition=partition, atomic_numbers=atomic_numbers.long(), element_by_orbit=elem
    )


def sample_uniform_legal_prior(
    *,
    partition: OrbitPartition,
    atomic_numbers: torch.Tensor,
    role_z: torch.Tensor,
    K: int,
    generator: torch.Generator | None = None,
) -> JointAssignmentState:
    """A_T ~ Uniform(Ω_A): element-compatible random bijection onto joint slots."""
    device = atomic_numbers.device
    n = int(atomic_numbers.numel())
    j = partition.J
    elem = build_element_by_orbit(partition, role_z.to(device))
    # slots: list of (o,k) with capacity m_o each
    slots: list[tuple[int, int]] = []
    for o, m in enumerate(partition.orbit_sizes):
        for k in range(K):
            for _ in range(m):
                slots.append((o, k))
    assert len(slots) == n
    # group atoms by Z
    atoms_by_z: dict[int, list[int]] = {}
    for i in range(n):
        z = int(atomic_numbers[i].item())
        atoms_by_z.setdefault(z, []).append(i)
    # slots by required Z
    slots_by_z: dict[int, list[tuple[int, int]]] = {}
    for o, k in slots:
        z = int(elem[o].item())
        slots_by_z.setdefault(z, []).append((o, k))
    A = torch.zeros(n, j, K, device=device)
    for z, atom_ids in atoms_by_z.items():
        z_slots = slots_by_z.get(z, [])
        if len(z_slots) != len(atom_ids):
            raise RuntimeError(f"element Z={z}: |atoms|={len(atom_ids)} != |slots|={len(z_slots)}")
        if generator is None:
            perm = torch.randperm(len(atom_ids), device=device)
        else:
            perm = torch.randperm(len(atom_ids), generator=generator)
        for t, ai in enumerate(perm.tolist()):
            o, k = z_slots[t]
            A[atom_ids[ai], o, k] = 1.0
    st = JointAssignmentState(
        A=A, partition=partition, atomic_numbers=atomic_numbers.long(), element_by_orbit=elem
    )
    v = st.validate()
    if not v["legal"]:
        raise RuntimeError(f"uniform prior produced illegal A: {v}")
    return st
