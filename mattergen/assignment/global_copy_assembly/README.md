# Global Structured Copy Assembly MVP

Standalone clean-geometry copy assembly. It does not implement assignment
diffusion, masked-Q repair, noisy geometry, iterative refinement, a joint
sampler, or pos/cell denoiser integration.

## Representation

Role assignment is always full canonical shape

\[
R\in\{0,1\}^{N\times M}
\]

(or integer labels of length \(N\)). Capacity is strict:

\[
R\mathbf{1}_M=\mathbf{1}_N,
\qquad
R^\top\mathbf{1}_N=K\mathbf{1}_M,
\]

so every crystal atom has one role and every role set satisfies

\[
|V_r|=K.
\]

For the fixed RHODIN01 MVP: \(N=40\), \(M=10\), \(K=4\).

### No orbit collapse

Molecular automorphisms **do not** merge roles (e.g. roles 1/2 are never
collapsed into a single \(2K\)-instance orbit role). The assembly state remains

\[
P_r\in S_K
\]

for each of the \(M\) canonical roles.

## Gauge semantics

Canonical role indices are a computational gauge. Literal equality
\(\widehat R=R_0\) is **not** required.

If for every true molecule copy \(k\) there exists \(\pi_k\in\operatorname{Aut}(G_{\rm mol})\)
such that the predicted roles on that copy equal the oracle roles composed with
\(\pi_k\), then

\[
\widehat R\sim R_0 \pmod{\operatorname{Aut}(G_{\rm mol})^K}
\]

and the assignment is classified `GAUGE_EQUIVALENT_R`.

Independent per-copy swaps such as roles \(1\leftrightarrow 2\) are physical
equivalences when \(\{1,2\}\) is an automorphism orbit. The following remain
**illegal** (classified `STRUCTURALLY_INCORRECT_R`):

* capacity violations (`role 1 = K+1`, `role 2 = K-1`);
* global capacity correct but a copy shows `1,1` while another shows `2,2`
  (not explainable by per-copy \(\operatorname{Aut}\)).

Automorphism quotient is used only for:

* correctness definition;
* audit classification;
* supervision *interpretation*;
* evaluation.

It is **never** used to rewrite predicted R before training.

## Convention and target isolation

For the stable crystal-index-sorted instances
`V_r=(v[r,0],...,v[r,K-1])` taken from the **role assignment in use**
(oracle \(R_0\) or geometry hard \(\widehat R\)),

\[
P_r[q]=k
\]

means instance `q` of role `r` has unified copy-gauge label `k`. The anchor
role must be in a singleton molecular-automorphism orbit. Its stable atom
order fixes gauge, so `P_anchor[q]=q`.

`mol_copy_id` is used **only** by target builders to form supervised `P*`:
it answers which true molecule copy a predicted instance belongs to. No
predictor API accepts `mol_copy_id`, `C0`, or `Q0`.

### Predicted-R path (no canonicalization)

When `role_source=geometry_only_hard_r`:

* \(R_{\rm effective}=R_{\rm artifact}\) exactly;
* forbidden: Aut-alignment rewrite, best-automorphism canonicalization toward
  \(R_0\), silent fallback to `sample["role"]`;
* \(P^*\) is built on \(V_r(\widehat R)\);
* if a role set has duplicate/missing true copies, the builder returns
  `TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR` and does **not** substitute
  oracle \(R_0\).

For legal gauge-swapped \(\widehat R\),

\[
P^*\to G^*\to C^*=C_0.
\]

## Model

The standalone periodic crystal encoder consumes clean `(X0,L0,z)` and the
molecular encoder consumes element/bond graph information. For a molecular
edge `(r,s,b)`, `BondPairPotential` produces a K-by-K scalar matrix
`w_rs(q,q')`. It contains crystal pair embeddings, molecular role embeddings,
bond type and minimum-image PBC-distance RBFs, but no copy information.

For each deterministic BFS spanning-tree edge and permutation states `u,v`:

\[
\Phi_{rs}(u,v)=\sum_q w_{rs}\bigl(q,v^{-1}(u(q))\bigr).
\]

The inverse table is precomputed once for all `K!` states. Non-tree graph
edges are recorded by `tree_builder` but are not used by this MVP CRF.

Unlike independent anchor-star masked Q, all role permutations are variables
of one CRF and every tree factor is scored jointly during one exact MAP or
sum-product pass. The anchor fixes only copy-label gauge; it does not turn
each non-anchor role into an independent prediction problem.

`TreeCRF` performs tensorized exact sum-product with `torch.logsumexp`. The
root is forced to identity. The structured loss is
`(logZ - S(P*)) / number_of_tree_edges` plus optional bidirectional pair CE.
`map_decode` uses exact max-product and stored backpointers; it never performs
independent per-role Hungarian matching.

## Decoding and evaluation (gauge-invariant)

Decoded permutations form `G[N,K]`, then `C=G G^\top`. Primary metrics:

* `exact C`
* copy-pair precision / recall / F1
* projected molecular-bond precision / recall / F1
* projected graph exact
* complete-copy rate
* copy graph-isomorphism rate
* cross-copy false molecular-edge rate

Literal exact R is **DIAGNOSTIC_ONLY** and is not a primary PASS gate.

Projected bonds use

\[
\widehat B = C\odot R B^{\rm mol} R^\top
\]

with the same \(R\) that defined assembly role sets (predicted or oracle).

## Geometry-only predicted-R extension

Configured via:

```yaml
mode: clean_geometry_predicted_r
role_source: geometry_only_hard_r
use_oracle_role_assignment: false
```

Pipeline:

```text
geometry-only checkpoint
  → hard-R artifact R̂ (raw Hungarian MAP, no canonicalize)
  → audit (Aut^K gauge + capacity)
  → P* on V_r(R̂)
  → tree-CRF train / MAP
  → G → C
```

### Current experimental hypothesis (canonical predicted-R path)

Remote runs may check whether a geometry hard assignment that is
**not** literally \(R_0\) but is equivalent modulo
\(\operatorname{Aut}(G_{\rm mol})^K\) still yields \(\widehat C=C_0\).

If \(\widehat R\) is structurally incorrect (outside that gauge class), the
**canonical-role** target construction fails explicitly; there is no oracle-R
fallback.

## Experiment O2 — orbit-aware full copy assembly

Geometry-only hard \(R\) can be orbit-role exact and bond-correct under oracle
\(C_0\) while still failing per-copy Aut equivalence (cross-copy compensation
on roles 1/2). O2 therefore does **not** require

\[
P_r\in S_K \quad\text{for non-singleton roles.}
\]

### Orbit membership (deterministic collapse)

Decoder still emits canonical \(\widehat R\in\{0,1\}^{N\times M}\). Before
assembly:

\[
\bar R_{io}=\sum_{r\in o}\widehat R_{ir},
\qquad
\bar R\in\{0,1\}^{N\times J}.
\]

RHODIN01: \(J=9\), orbits `[[0],[1,2],[3],…,[9]]`, so \(|V_{12}|=2K=8\).

### Stage A — singleton backbone

Singleton orbits keep \(|V_r|=K\) and reuse tree-CRF permutations \(P_r\in S_K\).
The singleton molecular graph may be disconnected after removing non-singleton
roles; `singleton_backbone` adds virtual edges from full-graph shortest paths
(path length + bond-type sequence) and builds a deterministic BFS tree.

### Stage B — exact orbit attachment

For orbit \(\{1,2\}\), attach 8 atoms to 4 copies with 2 atoms/copy using
bitmask DP:

\[
DP(k,S)=\max_{|U|=2,\,U\cap S=\emptyset}\big[DP(k-1,S\setminus U)+F_{12}(U,k)\big].
\]

Pair scores \(F_{12}\) are gauge-marginalized:

\[
F_{12}(\{i,j\},k)=\operatorname{logsumexp}(S_{12},S_{21})-\log 2,
\]

invariant to swapping \((i,j)\). Structured NLL uses the same DP with
`logsumexp` transitions for \(\log Z\).

### Final \(G,C\)

Merge singleton rows and orbit rows into \(G\in\{0,1\}^{N\times K}\), enforce
\(\bar R^\top G = m\mathbf{1}_K^\top\), then \(C=GG^\top\).

### Entrypoints

* config: `configs/assignment_diffusion_mvp/global_copy_assembly_orbit_aware_o2.yaml`
* train: `scripts/assignment_diffusion_mvp/train_global_copy_assembly_orbit_o2.py`
* eval: `scripts/assignment_diffusion_mvp/evaluate_global_copy_assembly_orbit_o2.py`
* remote: `scripts/assignment_diffusion_mvp/run_global_copy_assembly_orbit_o2_remote.sh`

### Information-source audit (frozen checkpoint, no training)

When `correct_geometry` and `zero_geometry` both yield exact \(C\), run:

* `scripts/assignment_diffusion_mvp/audit_orbit_o2_information_source.py`
* remote: `scripts/assignment_diffusion_mvp/run_orbit_o2_information_source_audit_remote.sh`

Gates: atom-index permutation equivariance; singleton \(V_r\) order shuffle;
orbit candidate order shuffle; legacy vs strict zero geometry feature rebuild;
MAP top-1/top-2 margins; tie-break perturbation; optional mismatch sample.
Writes under `.../global_copy_assembly_orbit_aware_o2/information_source_audit/`
without overwriting O2 training metrics.

## N1 — MatterGen-noisy geometry → orbit-aware assignment

Observational branch only (`geometry_feedback: false`). Noise is **only** via
MatterGen-native `MultiCorruption.sample_marginal` (pos
`NumAtomsVarianceAdjustedWrappedVESDE`, cell `LatticeVPSDE`), not a custom
schedule.

* package: `mattergen/assignment/noisy_copy_assignment/`
* config: `configs/assignment_diffusion_mvp/noisy_copy_assignment_n1.yaml`
* train: `scripts/assignment_diffusion_mvp/train_noisy_copy_assignment_n1.py`
* curves: `scripts/assignment_diffusion_mvp/evaluate_noisy_copy_assignment_curve_n1.py`
* remote: `scripts/assignment_diffusion_mvp/run_noisy_copy_assignment_n1_remote.sh`

Default freezes the feature backbone; assignment heads reuse O2 tree-CRF +
bitmask attachment. Does not modify pos/cell denoising scores.

## Future-only output contract

Authorized runs write under the configured output directory:
`config_audit.json`, `anchor_and_tree.json`, `permutation_convention.json`,
`training_trace.jsonl`, `checkpoint_selection.json`,
`geometry_only_hard_r.jsonl`, `geometry_only_r_audit.json`,
`map_evaluation_metrics.json`, `evaluation_metrics.json`, and the report
markdown. This document alone does not generate those artifacts.
