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

### Current experimental hypothesis

Remote runs should check whether a geometry hard assignment that is
**not** literally \(R_0\) but is equivalent modulo
\(\operatorname{Aut}(G_{\rm mol})^K\) still yields \(\widehat C=C_0\).

If \(\widehat R\) is structurally incorrect (outside that gauge class), target
construction fails explicitly; there is no oracle-R fallback.

## Future-only output contract

Authorized runs write under the configured output directory:
`config_audit.json`, `anchor_and_tree.json`, `permutation_convention.json`,
`training_trace.jsonl`, `checkpoint_selection.json`,
`geometry_only_hard_r.jsonl`, `geometry_only_r_audit.json`,
`map_evaluation_metrics.json`, `evaluation_metrics.json`, and the report
markdown. This document alone does not generate those artifacts.
