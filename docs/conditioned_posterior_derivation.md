# Posterior-correct multiscale (block) update — derivation

**Status:** v2 / M8 follow-up. Derived before implementation, per the project rule
"derive first, then implement, then prove."
**Scope:** Why the repeated-conditioning red-black sweep does **not** target the
posterior (even with the M8 `global_field` and the `null_space` prior terms), what
the mathematically correct local update is, and the exact acceptance rule for it.

---

## 1. Notation and target

Global state is the discretised log-permeability field on the `n = nx·ny`
cell grid (`n = 2304` at the default `48×48`). Fortran-order flattening
throughout (`reshape(vec, ny, nx, order="F")`).

* Global KLE: `Ψ = Φ diag(√λ) ∈ R^{n×N}`, `N = n_global_modes = 90`, where the
  columns of `Φ ∈ R^{n×N}` are orthonormal eigenvectors of the discretised
  exponential covariance `C ∈ R^{n×n}`, `C_{ab} = σ² exp(−‖x_a−x_b‖/ℓ)`, and
  `λ` the leading `N` eigenvalues. Because `ΦᵀΦ = I_N`, we have `ΨᵀΨ = diag(λ)`.
* Generative prior on coefficients: `θ ~ N(0, I_N)`, field `G = Ψ θ`.
  Equivalently the field prior is the rank-`N` Gaussian `G ~ N(0, C_N)`,
  `C_N = Ψ Ψᵀ = Σ_{m≤N} λ_m φ_m φ_mᵀ`.
* Likelihood / misfit: `Φ_mis(G) = ‖R(p(e^G)) − Y‖² / (2 σ_obs²)`, with `p(·)` the
  TPFA Darcy solve and `R` the sensor restriction. `L(Y|G) ∝ exp(−Φ_mis(G))`.
* Target posterior over the field:
  `π(G | Y) ∝ exp(−Φ_mis(G)) · N(G; 0, prior)`.

The repository's "global pCN" baseline samples this posterior in the `N`-mode
coefficient parametrisation and is stable (rel-k plateau ≈ 0.60, `‖θ‖ ≈ √N`).
The multiscale sweep is meant to be a faster sampler of the *same* posterior.

---

## 2. The existing local move, written exactly

Per subdomain `i` (coarse cell, core cells `S_i` that **tile** the grid;
extended region `Ω̂_i = core ∪ buffer ring`):

1. Build a **local** KLE `(Φ^{ext}_i, λ^{loc}_i)` of size `N_ext = N_c + M_b`
   from the covariance restricted to `Ω̂_i` — *a different Gaussian model from
   the global one.*
2. Build `A_i ∈ R^{M_b×N_ext}`, `(A_i)_{jℓ} = √λ^{loc}_ℓ φ^{ext}_ℓ(y_j)`, mapping
   local coefficients to field values at `M_b` buffer seam points `y_j`; set
   `c = G_old(y_j)` (frozen neighbour values).
3. Decompose `θ_loc = θ_p(c) + Z η`, `A_i θ_p = c`, `Z` an orthonormal basis of
   `null(A_i)` (`dim = N_c`), `θ_p = A_i⁺ c` (min-norm).
4. pCN the **free** part: `η' = √(1−β²) η + β ζ`, `ζ~N(0,I_{N_c})`; rebuild the
   local field `G_loc' = Φ^{ext}_i diag(√λ^{loc}) (θ_p(c) + Z η')`; **copy the
   core cells** `G_loc'[core]` into a candidate global field `G'`.
5. Accept on one of the implemented ratios:
   * `likelihood_only`: `Δlog L`.
   * `global_field` (M8): `Δlog L + log π_0^{proj}(G') − log π_0^{proj}(G) + Δq`,
     with `log π_0^{proj}(G) = −½‖ΦᵀG / √λ‖²` (project field onto the **global**
     KLE) and `Δq = ½(‖η'‖² − ‖η‖²)`.
   * `null_space`: `Δlog L − ½(‖η'‖² − ‖η‖²)`.

This is a Metropolis-within-Gibbs sweep where each block is the **core cells of
one subdomain**, the complement is frozen, and the proposal is the
projection-plus-pCN map of step 4.

---

## 3. Why none of the three rules targets `π(G|Y)`

### 3.1 The structural fact: incompatible conditionals

A Metropolis-within-Gibbs sweep is invariant for a global target `π` **iff each
block kernel is `π`-invariant**, i.e. iff each block kernel is an MH (or exact
Gibbs) step whose stationary law is the *true full conditional*
`π(G_{S_i} | G_{−S_i}, Y)`. The full conditional is fixed by the **one** global
prior `N(0, C_N)` and the likelihood:

```
π(G_S | G_R, Y) ∝ exp(−Φ_mis(G)) · p_prior(G_S | G_R),      R = complement of S.
```

The existing move never uses `p_prior(G_S | G_R)`. Its block proposal is built
from a **per-subdomain, independent local-KLE Gaussian** conditioned on `M_b`
*point* values. That family of block conditionals does **not** arise from any
single joint prior on `G`. They are **incompatible conditionals**: a Gibbs-type
sweep over incompatible conditionals has, in general, *no* stationary
distribution equal to the intended one (Besag's compatibility condition fails).
So there is no acceptance correction of the likelihood-only rule that recovers
`π(G|Y)` — the proposal family itself is inconsistent. This is the root cause,
independent of the prior term that is bolted on.

### 3.2 Why hard pointwise matching is the wrong conditional (framing B question)

Even granting the global prior `N(0, C_N)`, the correct prior conditional of the
core block `S` given the exterior `R` is Gaussian with a **covariance-weighted
(regression) mean and a Schur-complement covariance**:

```
G_S | G_R  ~  N( m_S , Σ_S ),
m_S = C_{S,R} C_{R,R}^{-1} G_R ,          (regression on ALL exterior cells)
Σ_S = C_{S,S} − C_{S,R} C_{R,R}^{-1} C_{R,S}.
```

Hard seam-matching `A θ_loc = c` is **not** this object:

* it conditions on `M_b` *discrete* buffer points, not the whole exterior `R`;
* it forces those points to equality (**zero** conditional variance there),
  whereas the true conditional has positive variance everywhere except at
  measured cells;
* it uses the **local** KLE covariance, not the global `C` regression;
* the free part keeps the *local* prior variance, not `Σ_S`.

So hard matching is a heuristic interpolation, not the prior conditional — it
cannot give a globally prior-consistent sweep. This is exactly the failure the
spec's framing (B) anticipated.

### 3.3 Why the `global_field` prior term does not rescue it

`log π_0^{proj}` penalises only the projection of `G` onto the `N=90` smooth
global modes. But the local update injects structure from the **local** KLE,
much of which is higher spatial frequency than 90 global modes can represent
over a `12×12` core. That structure is **in the kernel of the projection**:
`ΦᵀG` barely sees it. The penalty is therefore *blind to the very directions the
move excites*. Empirically (NOTES, resolving run) the accepted local norm grows
`13.5 → 20.5`, `‖θ_p‖` grows `10.3 → 13.5`, the null norm grows `10.1 → 14.8`,
the misfit keeps falling, yet rel-k reverses `0.46 → 0.886`: the chain fits the
sparse sensors with prior-invisible local wiggles and walks away from the truth.
This is textbook over-fitting enabled by (i) an over-rich, per-block basis,
(ii) a weak/smoothing likelihood with only 64 sensors, and (iii) a prior that
does not see the excited modes.

### 3.4 Why the `null_space` term does not rescue it

It bounds `‖η‖` (the *free* coordinates) but puts **no** prior force on `θ_p(c)`
/ the pinned part, and is still built on the inconsistent local-KLE conditional.
NOTES confirms: `‖η‖` is bounded (≈4.8) yet rel-k still reverses (`0.635 →
0.818`). Controlling the free part is provably insufficient because the drift
lives in the pinned/global part.

**Conclusion.** The construction is *structurally* approximate. No bolt-on prior
term fixes it. We must replace the block proposal with the true full conditional
of one global prior. We adopt framing (B), made exact.

---

## 4. The corrected update: precision-based block Gibbs

### 4.1 A proper, full-rank global prior consistent with the model

The rank-`N` prior `N(0, C_N)` cannot support cell-wise block updates: fixing the
`≈2160` exterior cells over-determines a `90`-DOF field, so the conditional is
degenerate. We therefore use the **proper** (full-rank) field prior

```
G ~ N(0, C_τ),     C_τ = C_N + τ² I = Ψ Ψᵀ + τ² I,     τ² > 0  (nugget / jitter).
```

This is the generative KLE prior plus a tiny independent per-cell variance —
standard GP "jitter". It is full rank and invertible, and its marginal variance
along smooth mode `φ_m` is `λ_m + τ² ≈ λ_m` (for `τ² ≪ λ_N`), while any
direction **outside** `span(Φ)` (the rough/local structure the heuristic
over-fit with) has variance `τ²` — i.e. precision `1/τ²`, strongly suppressed.
The truth `G_true = Ψ θ_true ∈ span(Ψ)` has high prior density, so recovery is
not biased; only the prior-implausible rough directions are penalised. As
`τ → 0` the model returns to the degenerate KLE prior.

The precision has a **closed form** (Woodbury), so we never form an `n×n`
inverse. With `ΨᵀΨ = diag(λ)`:

```
Q := C_τ^{-1} = (1/τ²) [ I − Φ diag( λ/(λ+τ²) ) Φᵀ ].
```

Matvec `Q g = (1/τ²)[ g − Φ ( d ⊙ (Φᵀ g) ) ]`, `d = λ/(λ+τ²)`, costs `O(nN)`.
The principal submatrix on a core block `S` is
`Q_SS = (1/τ²)[ I_{|S|} − Φ_S diag(d) Φ_Sᵀ ]` (a `|S|×|S|` SPD matrix, `|S|=144`),
formed once per subdomain.

### 4.2 The block full conditional (precision form)

For block `S`, complement `R`, the prior conditional `G_S | G_R ~ N(m_S, Q_SS^{-1})`
with

```
m_S = − Q_SS^{-1} Q_{S,R} G_R = G_S − Q_SS^{-1} (Q G)_S .
```

(The second equality avoids `Q_{S,R}`: `(QG)_S = Q_SS G_S + Q_{S,R} G_R`.)
`(QG)_S` uses the closed-form matvec; `Q_SS^{-1}·` is a `144`-dim Cholesky solve.
This `m_S` is exactly the covariance-weighted regression of §3.2, and
`Q_SS^{-1}` is exactly the Schur complement `Σ_S` (Schur ↔ precision-submatrix
identity).

### 4.3 The proposal and acceptance

pCN **around the conditional prior** (not around the origin):

```
G_S' = m_S + √(1−β²) (G_S − m_S) + β w,     w ~ N(0, Q_SS^{-1}),
```

drawn as `w = L^{-T} ζ`, `ζ ~ N(0, I_{|S|})`, where `Q_SS = L Lᵀ` (Cholesky).
Replace `G[core_i]` by `G_S'` to form the candidate `G'`.

**Acceptance is likelihood-only and exact.** The block target is
`π(G_S | G_R, Y) ∝ exp(−Φ_mis(G)) · N(G_S; m_S, Q_SS^{-1})`. The pCN kernel is
reversible w.r.t. its base Gaussian `N(m_S, Q_SS^{-1})`:

```
N(G_S; m_S, Σ_S) q(G_S→G_S') = N(G_S'; m_S, Σ_S) q(G_S'→G_S).
```

Hence in the MH ratio the Gaussian factors and the proposal factors cancel,
leaving only the likelihood:

```
log α = Φ_mis(G) − Φ_mis(G') = log L(Y|G') − log L(Y|G).
```

`m_S` and `Σ_S` depend on `G_R`, which is **fixed during this block step**, so
the same base Gaussian appears in both directions and the cancellation is exact.
Each block kernel is therefore invariant for the true full conditional; a sweep
over the core blocks (which **partition** all `n` cells) is invariant for
`π(G | Y) ∝ exp(−Φ_mis) N(0, C_τ)`. **The sweep provably targets the posterior.**

Notes:
* No `A`, `Z`, `θ_p`, local KLE, or overlap is needed for correctness — overlap
  was a device to hide seams in the heuristic. Cores tiling the grid is what
  makes the Gibbs sweep cover every variable.
* Block order is irrelevant to invariance; sequential (red-black or lexicographic)
  with the *current* exterior each step is standard systematic-scan Gibbs.
* This is the spec's framing (A) "correct proposal density" and framing (B)
  "correct covariance-weighted conditional" reconciled into a single exact rule;
  the proposal density that M8 mis-specified is here the conditional Gaussian
  `N(m_S, Σ_S)`, and the likelihood-only cancellation is its rigorous form.

### 4.4 Limiting / reduction cases (used by the invariance gate)

1. **No data** (`Φ_mis ≡ const`): every block step is exact conditional Gibbs of
   `N(0, C_τ)`; a sweep leaves `N(0, C_τ)` invariant. (Gate: start from `N(0,C_τ)`
   samples, sweep, check mean→0 and covariance→`C_τ`.)
2. **`β = 1`**: `G_S' = m_S + w` is an independent draw from the conditional
   prior; with likelihood-only acceptance this is the exact
   Metropolis-within-Gibbs "prior as proposal" step.
3. **One block = whole domain** (`S = all cells`, `R = ∅`): `m_S = 0`,
   `Σ_S = C_τ`, and the move is global pCN of the full posterior — recovering the
   validated global-pCN behaviour and reducing the acceptance to the M3 engine's
   prior-preserving likelihood-only rule.
4. **Fixed exterior + prior-preserving move ⇒ likelihood-only**, matching the v1
   rule exactly — the required "reduces to the known-correct rule" property.

### 4.5 Linear-Gaussian exactness (strong gate)

If the forward map is replaced by a **linear** `R(p(e^G)) → H G` (tiny grid),
the posterior is exactly Gaussian
`π(G|Y) ∝ exp(−½‖HG−Y‖²/σ_obs² − ½ GᵀQG) = N(μ_post, Σ_post)`,
`Σ_post = (Q + HᵀH/σ_obs²)^{-1}`, `μ_post = Σ_post Hᵀ Y/σ_obs²`. The block-Gibbs
sweep above must reproduce `μ_post, Σ_post` empirically. This is the correctness
gate that a method merely lowering rel-k cannot pass.

---

## 5. Cost

Per block move: one closed-form `Q g` matvec (`O(nN)`), one `Φᵀg` (`O(nN)`), a
`|S|`-dim Cholesky solve (`|S|=144`), and **one forward solve** — the same
forward-solve count as the existing red-black move. The forward solve dominates,
so per-iteration cost is essentially unchanged. One-time setup: the `N`-mode KLE
(already computed) and one `|S|×|S|` Cholesky per core (16 of them). No `n×n`
inverse is ever formed. Correctness-first, and here also not more expensive.

---

## 6. Parameter `τ` (nugget)

`τ²` trades prior fidelity (`τ² ≪ λ_N` keeps retained modes intact) against
suppression of rough directions (variance `τ²`). Exposed as `Config.block_nugget`
(absolute `τ²`, default `1e-4`) and overridable per run. Smaller `τ` ⇒ stronger
anti-overfitting and stiffer conditionals; too small over-shrinks the smallest
retained modes. Tuned in `exp08c`.

---

## 7. What is implemented and tested

* `conditioning/gaussian_block.py`: closed-form precision operator, per-core
  `Q_SS`/Cholesky, conditional mean, pCN-on-conditional proposal (pure, float64,
  explicit `Generator`).
* `sampler.py::block_gibbs_sampler`: the sweep, new function — existing
  `conditioned_sampler` / `red_black_conditioned_sampler` and the
  `likelihood_only` / `global_field` / `null_space` modes are untouched.
* Tests (`tests/test_gaussian_block.py`):
  * Woodbury precision matches `C_τ^{-1}` (dense) and `Q_SS` matches its
    submatrix.
  * pCN-on-conditional reversibility (detailed balance) of the base Gaussian.
  * **Invariance, no data** (case 1): sweep preserves `N(0, C_τ)` mean/cov.
  * **Linear-Gaussian exactness** (case 4.5): sweep recovers `μ_post, Σ_post`.
  * Reduction to likelihood-only with fixed exterior / whole-domain block.
* `exp08c`: new opt-in scheme `block_gibbs` reporting the rel-k trajectory vs
  `global_field` and global pCN, to show the reversal is cured.
