"""Orbit [multi-role] balanced copy attachment: scores + exact bitmask DP.

For an orbit o with |o|=m and |V_o|=m K, assign atoms to K copies with
exactly m atoms per copy.  RHODIN01 orbit {1,2}: m=2, |V|=8, K=4.

Gauge marginalization over local canonical roles 1/2:

    F({i,j}, k) = logsumexp(S_12(i,j;k), S_21(i,j;k)) - log(2)

so F is invariant to swapping (i,j).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
import torch
from torch import nn


class OrbitAttachmentHead(nn.Module):
    """Learnable score for attaching an unordered atom pair to a singleton copy.

    Forward API contains no mol_copy_id / C0 / canonical 1-vs-2 identity as input
    features (only gauge-symmetric geometry and embeddings).
    """

    def __init__(
        self,
        hidden: int = 256,
        rbf_dim: int = 32,
        cutoff: float = 6.0,
        score_scale: float = 20.0,
        gauge_marginalization: str = "logsumexp",
    ):
        super().__init__()
        if score_scale <= 0:
            raise ValueError("score_scale must be positive")
        if gauge_marginalization not in {"logsumexp", "max"}:
            raise ValueError("gauge_marginalization must be logsumexp or max")
        self.score_scale = float(score_scale)
        self.gauge_marginalization = gauge_marginalization
        self.cutoff = cutoff
        self.register_buffer("centres", torch.linspace(0.0, cutoff, rbf_dim))
        # Features: h_i, h_j, h_copy, |hi-hj|, hi*hj, rbf_ij, rbf_i_copy, rbf_j_copy, orbit_emb, gauge_token
        in_dim = 5 * hidden + 3 * rbf_dim + hidden + hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.orbit_embedding = nn.Embedding(16, hidden)
        self.gauge_embedding = nn.Embedding(2, hidden)  # 0: 12, 1: 21 — marginalized out
        self.copy_pool = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def _rbf(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((distance.unsqueeze(-1) - self.centres) / (self.cutoff / max(1, len(self.centres)))) ** 2)

    def _pbc_distance(self, frac_a: torch.Tensor, frac_b: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        delta = frac_b - frac_a
        delta = delta - torch.round(delta)
        return torch.linalg.norm(delta @ cell, dim=-1)

    def score_ordered_pair(
        self,
        *,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        frac_i: torch.Tensor,
        frac_j: torch.Tensor,
        h_copy: torch.Tensor,
        frac_copy: torch.Tensor,
        cell: torch.Tensor,
        orbit_id: int,
        gauge_id: int,
    ) -> torch.Tensor:
        """Scalar score S for ordered local gauge (gauge_id 0=12, 1=21)."""
        if h_copy.ndim != 2:
            raise ValueError("h_copy must be [M_sing,H]")
        h_pool = self.copy_pool(h_copy.mean(0, keepdim=False))
        # mean distance from i/j to copy atoms
        d_ij = self._pbc_distance(frac_i, frac_j, cell)
        d_i = self._pbc_distance(frac_i.unsqueeze(0).expand(len(frac_copy), -1), frac_copy, cell).mean()
        d_j = self._pbc_distance(frac_j.unsqueeze(0).expand(len(frac_copy), -1), frac_copy, cell).mean()
        rbf = torch.cat([self._rbf(d_ij), self._rbf(d_i), self._rbf(d_j)], dim=-1)
        orbit = self.orbit_embedding(
            torch.as_tensor(orbit_id, device=h_i.device, dtype=torch.long).clamp(0, self.orbit_embedding.num_embeddings - 1)
        )
        gauge = self.gauge_embedding(
            torch.as_tensor(gauge_id, device=h_i.device, dtype=torch.long).clamp(0, 1)
        )
        feat = torch.cat(
            [h_i, h_j, h_pool, (h_i - h_j).abs(), h_i * h_j, rbf, orbit, gauge],
            dim=-1,
        )
        raw = self.net(feat).squeeze(-1)
        return self.score_scale * torch.tanh(raw / self.score_scale)

    def pair_score(
        self,
        *,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        frac_i: torch.Tensor,
        frac_j: torch.Tensor,
        h_copy: torch.Tensor,
        frac_copy: torch.Tensor,
        cell: torch.Tensor,
        orbit_id: int = 0,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Gauge-marginalized unordered pair score F({i,j}, k)."""
        mode = self.gauge_marginalization if mode is None else mode
        s12 = self.score_ordered_pair(
            h_i=h_i, h_j=h_j, frac_i=frac_i, frac_j=frac_j,
            h_copy=h_copy, frac_copy=frac_copy, cell=cell, orbit_id=orbit_id, gauge_id=0,
        )
        s21 = self.score_ordered_pair(
            h_i=h_j, h_j=h_i, frac_i=frac_j, frac_j=frac_i,
            h_copy=h_copy, frac_copy=frac_copy, cell=cell, orbit_id=orbit_id, gauge_id=1,
        )
        if mode == "max":
            return torch.maximum(s12, s21)
        # logsumexp marginalization, normalized by log(2)
        return torch.logsumexp(torch.stack([s12, s21], dim=0), dim=0) - math.log(2.0)


@dataclass(frozen=True)
class AttachmentResult:
    log_partition: torch.Tensor
    map_score: torch.Tensor
    map_pairs: tuple[tuple[int, int], ...]  # length K, local indices into V_o
    target_score: torch.Tensor | None = None


def _pair_list(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def balanced_attachment_dp(
    F: torch.Tensor,
    *,
    target_pairs: list[tuple[int, int]] | None = None,
    atoms_per_copy: int = 2,
) -> AttachmentResult:
    """Exact balanced attachment via bitmask DP.

    Parameters
    ----------
    F:
        Tensor of shape ``[K, n, n]`` with ``F[k,i,j] = F[k,j,i]`` and
        diagonal unused.  ``n`` must equal ``atoms_per_copy * K``.
    target_pairs:
        Optional list of length K of local index pairs for structured NLL.
    """
    if F.ndim != 3 or F.shape[1] != F.shape[2]:
        raise ValueError("F must have shape [K,n,n]")
    K, n, _ = F.shape
    if n != atoms_per_copy * K:
        raise ValueError(f"n={n} must equal atoms_per_copy*K={atoms_per_copy * K}")
    if atoms_per_copy != 2:
        raise NotImplementedError("MVP bitmask DP implements atoms_per_copy=2 only")
    device = F.device
    dtype = F.dtype
    n_masks = 1 << n
    neg = torch.tensor(float("-inf"), device=device, dtype=dtype)
    # log-sum DP and max DP
    log_dp = torch.full((K + 1, n_masks), float("-inf"), device=device, dtype=dtype)
    max_dp = torch.full((K + 1, n_masks), float("-inf"), device=device, dtype=dtype)
    back_i = torch.full((K + 1, n_masks), -1, device=device, dtype=torch.long)
    back_j = torch.full((K + 1, n_masks), -1, device=device, dtype=torch.long)
    log_dp[0, 0] = torch.zeros((), device=device, dtype=dtype)
    max_dp[0, 0] = torch.zeros((), device=device, dtype=dtype)

    pairs = _pair_list(n)
    for k in range(K):
        for mask in range(n_masks):
            if not torch.isfinite(log_dp[k, mask]):
                continue
            base_log = log_dp[k, mask]
            base_max = max_dp[k, mask]
            for i, j in pairs:
                bit = (1 << i) | (1 << j)
                if mask & bit:
                    continue
                new_mask = mask | bit
                score = F[k, i, j]
                # logsumexp transition
                cand_log = base_log + score
                log_dp[k + 1, new_mask] = torch.logaddexp(log_dp[k + 1, new_mask], cand_log)
                # max transition
                cand_max = base_max + score
                if cand_max > max_dp[k + 1, new_mask]:
                    max_dp[k + 1, new_mask] = cand_max
                    back_i[k + 1, new_mask] = i
                    back_j[k + 1, new_mask] = j

    full = (1 << n) - 1
    log_z = log_dp[K, full]
    map_score = max_dp[K, full]
    if not torch.isfinite(log_z) or not torch.isfinite(map_score):
        raise FloatingPointError("balanced attachment DP produced non-finite logZ/MAP")

    # reconstruct MAP pairs (from last copy backward)
    pairs_rev: list[tuple[int, int]] = []
    mask = full
    for k in range(K, 0, -1):
        i = int(back_i[k, mask].item())
        j = int(back_j[k, mask].item())
        if i < 0 or j < 0:
            raise RuntimeError(f"missing backpointer at copy stage {k}")
        pairs_rev.append((i, j) if i < j else (j, i))
        mask ^= (1 << i) | (1 << j)
    map_pairs = tuple(reversed(pairs_rev))
    if mask != 0:
        raise RuntimeError("MAP reconstruction did not consume full mask")

    target_score = None
    if target_pairs is not None:
        if len(target_pairs) != K:
            raise ValueError("target_pairs must have length K")
        used = 0
        acc = torch.zeros((), device=device, dtype=dtype)
        for k, (i, j) in enumerate(target_pairs):
            a, b = (i, j) if i < j else (j, i)
            bit = (1 << a) | (1 << b)
            if used & bit:
                raise ValueError("target_pairs are not disjoint")
            used |= bit
            acc = acc + F[k, a, b]
        if used != full:
            raise ValueError("target_pairs must cover all orbit atoms")
        target_score = acc

    return AttachmentResult(
        log_partition=log_z,
        map_score=map_score,
        map_pairs=map_pairs,
        target_score=target_score,
    )


def brute_force_balanced_logz_map(F: torch.Tensor, *, atoms_per_copy: int = 2) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    """O(n!)-style enumeration for tiny n; test oracle for bitmask DP."""
    K, n, _ = F.shape
    if n != atoms_per_copy * K or atoms_per_copy != 2:
        raise ValueError("brute force helper expects n=2K")
    atoms = list(range(n))
    best_score = None
    best_assign = None
    scores = []
    # Partition 2K atoms into K unordered pairs, then assign pairs to ordered copies K!.
    # Generate perfect matchings then permute onto copies.
    def perfect_matchings(items: list[int]) -> list[list[tuple[int, int]]]:
        if not items:
            return [[]]
        a = items[0]
        out = []
        for idx in range(1, len(items)):
            b = items[idx]
            rest = items[1:idx] + items[idx + 1 :]
            pair = (a, b) if a < b else (b, a)
            for matching in perfect_matchings(rest):
                out.append([pair] + matching)
        return out

    for matching in perfect_matchings(atoms):
        for ordered in itertools.permutations(matching, K):
            score = F.new_zeros(())
            for k, (i, j) in enumerate(ordered):
                score = score + F[k, i, j]
            scores.append(score)
            if best_score is None or score > best_score:
                best_score = score
                best_assign = tuple(ordered)
    log_z = torch.logsumexp(torch.stack(scores), dim=0)
    assert best_score is not None and best_assign is not None
    return log_z, best_score, best_assign
