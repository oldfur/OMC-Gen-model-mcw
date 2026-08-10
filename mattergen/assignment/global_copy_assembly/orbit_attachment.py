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
        # Ordered features: (h_i, h_j, ...) define local Aut gauge 1/2 by argument order.
        # Unordered F({i,j}) marginalizes both orders — exact swap invariance.
        # Features: h_i, h_j, h_copy, |hi-hj|, hi*hj, rbf_ij, rbf_i_copy, rbf_j_copy, orbit_emb
        in_dim = 5 * hidden + 3 * rbf_dim + hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.orbit_embedding = nn.Embedding(16, hidden)
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
    ) -> torch.Tensor:
        """Scalar score S for one ordered local Aut gauge (argument order = 1/2 gauge)."""
        if h_copy.ndim != 2:
            raise ValueError("h_copy must be [M_sing,H]")
        h_pool = self.copy_pool(h_copy.mean(0, keepdim=False))
        # mean distance from i/j to copy atoms
        d_ij = self._pbc_distance(frac_i, frac_j, cell)
        d_i = self._pbc_distance(frac_i.unsqueeze(0).expand(len(frac_copy), -1), frac_copy, cell).mean()
        d_j = self._pbc_distance(frac_j.unsqueeze(0).expand(len(frac_copy), -1), frac_copy, cell).mean()
        rbf = torch.cat([self._rbf(d_ij), self._rbf(d_i), self._rbf(d_j)], dim=-1)
        orbit = self.orbit_embedding(
            torch.as_tensor(orbit_id, device=h_i.device, dtype=torch.long).clamp(
                0, self.orbit_embedding.num_embeddings - 1
            )
        )
        feat = torch.cat(
            [h_i, h_j, h_pool, (h_i - h_j).abs(), h_i * h_j, rbf, orbit],
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
        """Gauge-marginalized unordered pair score F({i,j}, k).

        S_12 = S(i→j), S_21 = S(j→i).  Then
        F = logsumexp(S_12, S_21) - log(2)  (or max),
        which is *exactly* invariant to swapping the call arguments (i,j)↔(j,i).
        """
        mode = self.gauge_marginalization if mode is None else mode
        s12 = self.score_ordered_pair(
            h_i=h_i,
            h_j=h_j,
            frac_i=frac_i,
            frac_j=frac_j,
            h_copy=h_copy,
            frac_copy=frac_copy,
            cell=cell,
            orbit_id=orbit_id,
        )
        s21 = self.score_ordered_pair(
            h_i=h_j,
            h_j=h_i,
            frac_i=frac_j,
            frac_j=frac_i,
            h_copy=h_copy,
            frac_copy=frac_copy,
            cell=cell,
            orbit_id=orbit_id,
        )
        if mode == "max":
            return torch.maximum(s12, s21)
        return torch.logsumexp(torch.stack([s12, s21], dim=0), dim=0) - math.log(2.0)


@dataclass(frozen=True)
class AttachmentResult:
    log_partition: torch.Tensor
    map_score: torch.Tensor
    map_pairs: tuple[tuple[int, int], ...]  # length K, local indices into V_o
    target_score: torch.Tensor | None = None


def _pair_list(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _neg_inf_scalar(like: torch.Tensor) -> torch.Tensor:
    return like.new_tensor(float("-inf"))


def _balanced_attachment_log_partition(F: torch.Tensor, *, atoms_per_copy: int = 2) -> torch.Tensor:
    """Autograd-safe exact logZ via layer-wise list DP (no tensor inplace writes)."""
    if F.ndim != 3 or F.shape[1] != F.shape[2]:
        raise ValueError("F must have shape [K,n,n]")
    K, n, _ = F.shape
    if n != atoms_per_copy * K:
        raise ValueError(f"n={n} must equal atoms_per_copy*K={atoms_per_copy * K}")
    if atoms_per_copy != 2:
        raise NotImplementedError("MVP bitmask DP implements atoms_per_copy=2 only")
    n_masks = 1 << n
    pairs = _pair_list(n)
    # Python list of scalar tensors: rebinding entries is not an inplace Tensor op.
    log_dp: list[torch.Tensor] = [_neg_inf_scalar(F) for _ in range(n_masks)]
    log_dp[0] = F.new_zeros(())
    for k in range(K):
        new_dp: list[torch.Tensor] = [_neg_inf_scalar(F) for _ in range(n_masks)]
        for mask in range(n_masks):
            base = log_dp[mask]
            # Skip unreachable masks (hard -inf). Use detach so the check is not graph-facing.
            if float(base.detach()) == float("-inf"):
                continue
            for i, j in pairs:
                bit = (1 << i) | (1 << j)
                if mask & bit:
                    continue
                new_mask = mask | bit
                cand = base + F[k, i, j]
                # List rebinding (not Tensor.__setitem__) keeps autograd version counters clean.
                new_dp[new_mask] = torch.logaddexp(new_dp[new_mask], cand)
        log_dp = new_dp
    log_z = log_dp[(1 << n) - 1]
    if not torch.isfinite(log_z.detach()):
        raise FloatingPointError("balanced attachment DP produced non-finite logZ")
    return log_z


@torch.no_grad()
def _balanced_attachment_map(
    F: torch.Tensor,
    *,
    atoms_per_copy: int = 2,
    pair_order: str = "default",
    tie_break_seed: int = 0,
    near_tie_tol: float = 0.0,
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    """Exact MAP under no_grad (backpointers; not needed for structured NLL grads).

    ``pair_order`` controls deterministic/random enumeration when scores tie
    within ``near_tie_tol`` (Gate F).  Does not alter model scores.
    """
    if F.ndim != 3 or F.shape[1] != F.shape[2]:
        raise ValueError("F must have shape [K,n,n]")
    K, n, _ = F.shape
    if n != atoms_per_copy * K:
        raise ValueError(f"n={n} must equal atoms_per_copy*K={atoms_per_copy * K}")
    if atoms_per_copy != 2:
        raise NotImplementedError("MVP bitmask DP implements atoms_per_copy=2 only")
    if pair_order not in {"default", "reverse", "random"}:
        raise ValueError("pair_order must be default|reverse|random")
    device = F.device
    dtype = F.dtype
    n_masks = 1 << n
    max_dp = torch.full((K + 1, n_masks), float("-inf"), device=device, dtype=dtype)
    back_i = torch.full((K + 1, n_masks), -1, device=device, dtype=torch.long)
    back_j = torch.full((K + 1, n_masks), -1, device=device, dtype=torch.long)
    max_dp[0, 0] = 0.0
    pairs = _pair_list(n)
    if pair_order == "reverse":
        pairs = list(reversed(pairs))
    elif pair_order == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(tie_break_seed))
        order = torch.randperm(len(pairs), generator=g).tolist()
        pairs = [pairs[i] for i in order]
    for k in range(K):
        for mask in range(n_masks):
            base = max_dp[k, mask]
            if not torch.isfinite(base):
                continue
            for i, j in pairs:
                bit = (1 << i) | (1 << j)
                if mask & bit:
                    continue
                new_mask = mask | bit
                cand = base + F[k, i, j]
                best = max_dp[k + 1, new_mask]
                # default: strict improvement only.
                # reverse/random: also accept near-ties so enumeration order can change MAP.
                take = False
                if not torch.isfinite(best) and torch.isfinite(cand):
                    take = True
                elif cand > best + near_tie_tol:
                    take = True
                elif (
                    pair_order != "default"
                    and torch.isfinite(best)
                    and abs(float(cand - best)) <= near_tie_tol
                ):
                    take = True
                if take:
                    max_dp[k + 1, new_mask] = cand
                    back_i[k + 1, new_mask] = i
                    back_j[k + 1, new_mask] = j
    full = (1 << n) - 1
    map_score = max_dp[K, full]
    if not torch.isfinite(map_score):
        raise FloatingPointError("balanced attachment MAP produced non-finite score")
    pairs_rev: list[tuple[int, int]] = []
    mask = full
    for k in range(K, 0, -1):
        i = int(back_i[k, mask].item())
        j = int(back_j[k, mask].item())
        if i < 0 or j < 0:
            raise RuntimeError(f"missing backpointer at copy stage {k}")
        pairs_rev.append((i, j) if i < j else (j, i))
        mask ^= (1 << i) | (1 << j)
    if mask != 0:
        raise RuntimeError("MAP reconstruction did not consume full mask")
    return map_score, tuple(reversed(pairs_rev))


@torch.no_grad()
def enumerate_balanced_attachment_scores(
    F: torch.Tensor,
    *,
    atoms_per_copy: int = 2,
) -> list[tuple[float, tuple[tuple[int, int], ...]]]:
    """Enumerate all ordered balanced attachments with scores (exact landscape).

    Returns descending ``(score, map_pairs)`` list.  For RHODIN orbit n=8,K=4
    this is tractable (perfect matchings × K!).
    """
    if F.ndim != 3 or F.shape[1] != F.shape[2]:
        raise ValueError("F must have shape [K,n,n]")
    K, n, _ = F.shape
    if n != atoms_per_copy * K or atoms_per_copy != 2:
        raise ValueError("enumeration helper expects n=2K and atoms_per_copy=2")
    atoms = list(range(n))

    def perfect_matchings(items: list[int]) -> list[list[tuple[int, int]]]:
        if not items:
            return [[]]
        a = items[0]
        out: list[list[tuple[int, int]]] = []
        for idx in range(1, len(items)):
            b = items[idx]
            rest = items[1:idx] + items[idx + 1 :]
            pair = (a, b) if a < b else (b, a)
            for matching in perfect_matchings(rest):
                out.append([pair] + matching)
        return out

    scored: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    for matching in perfect_matchings(atoms):
        for ordered in itertools.permutations(matching, K):
            score = 0.0
            for k, (i, j) in enumerate(ordered):
                score += float(F[k, i, j].detach())
            scored.append((score, tuple(ordered)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


@torch.no_grad()
def attachment_map_margins(
    F: torch.Tensor,
    *,
    atoms_per_copy: int = 2,
    near_tie_tol: float = 1e-6,
    target_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    """Top-1 / top-2 scores, gaps, ties, logZ, optional target probability."""
    scored = enumerate_balanced_attachment_scores(F, atoms_per_copy=atoms_per_copy)
    if not scored:
        raise RuntimeError("empty attachment landscape")
    best_score, best_pairs = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else float("-inf")
    gap = best_score - second_score if len(scored) > 1 else float("inf")
    n_exact_ties = sum(1 for s, _ in scored if abs(s - best_score) <= 0.0)
    n_near_ties = sum(1 for s, _ in scored if abs(s - best_score) <= near_tie_tol)
    log_z = _balanced_attachment_log_partition(F, atoms_per_copy=atoms_per_copy)
    log_z_f = float(log_z.detach())
    # entropy of uniform over near-tie set is not full distribution; use full Boltzmann
    scores_t = torch.tensor([s for s, _ in scored], dtype=F.dtype, device=F.device)
    log_z_enum = torch.logsumexp(scores_t, dim=0)
    probs = torch.exp(scores_t - log_z_enum)
    entropy = float((-(probs * (scores_t - log_z_enum))).sum().clamp_min(0.0))
    out: dict[str, object] = {
        "best_attachment_score": best_score,
        "second_best_attachment_score": second_score,
        "attachment_MAP_gap": gap,
        "map_pairs": best_pairs,
        "logZ": log_z_f,
        "logZ_enumeration": float(log_z_enum.detach()),
        "entropy": entropy,
        "number_of_exact_ties": int(n_exact_ties),
        "number_of_near_ties": int(n_near_ties),
        "near_tie_tol": near_tie_tol,
        "num_complete_assignments": len(scored),
    }
    if target_pairs is not None:
        tscore = 0.0
        for k, (i, j) in enumerate(target_pairs):
            a, b = (i, j) if i < j else (j, i)
            tscore += float(F[k, a, b].detach())
        out["target_score"] = tscore
        out["target_log_probability"] = tscore - log_z_f
        out["target_probability"] = float(torch.exp(torch.tensor(tscore - log_z_f)).clamp_max(1.0))
    return out


def balanced_attachment_dp(
    F: torch.Tensor,
    *,
    target_pairs: list[tuple[int, int]] | None = None,
    atoms_per_copy: int = 2,
    pair_order: str = "default",
    tie_break_seed: int = 0,
    near_tie_tol: float = 0.0,
) -> AttachmentResult:
    """Exact balanced attachment via bitmask DP.

    Parameters
    ----------
    F:
        Tensor of shape ``[K, n, n]`` with ``F[k,i,j] = F[k,j,i]`` and
        diagonal unused.  ``n`` must equal ``atoms_per_copy * K``.
    target_pairs:
        Optional list of length K of local index pairs for structured NLL.

    Notes
    -----
    ``log_partition`` / ``target_score`` are autograd-safe (no inplace Tensor
    writes on the forward graph).  MAP uses a separate no_grad max-DP.
    """
    if F.ndim != 3 or F.shape[1] != F.shape[2]:
        raise ValueError("F must have shape [K,n,n]")
    K, n, _ = F.shape
    if n != atoms_per_copy * K:
        raise ValueError(f"n={n} must equal atoms_per_copy*K={atoms_per_copy * K}")
    if atoms_per_copy != 2:
        raise NotImplementedError("MVP bitmask DP implements atoms_per_copy=2 only")

    log_z = _balanced_attachment_log_partition(F, atoms_per_copy=atoms_per_copy)
    map_score, map_pairs = _balanced_attachment_map(
        F,
        atoms_per_copy=atoms_per_copy,
        pair_order=pair_order,
        tie_break_seed=tie_break_seed,
        near_tie_tol=near_tie_tol,
    )

    target_score = None
    if target_pairs is not None:
        if len(target_pairs) != K:
            raise ValueError("target_pairs must have length K")
        used = 0
        full = (1 << n) - 1
        terms: list[torch.Tensor] = []
        for k, (i, j) in enumerate(target_pairs):
            a, b = (i, j) if i < j else (j, i)
            bit = (1 << a) | (1 << b)
            if used & bit:
                raise ValueError("target_pairs are not disjoint")
            used |= bit
            terms.append(F[k, a, b])
        if used != full:
            raise ValueError("target_pairs must cover all orbit atoms")
        target_score = torch.stack(terms).sum()

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
