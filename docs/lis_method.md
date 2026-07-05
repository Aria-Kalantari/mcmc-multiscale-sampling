# Beating global pCN with a Likelihood-Informed Subspace (LIS) sampler

**Status:** prototype + measured results (M12). **Code:** `src/mcmc_multiscale/lis.py`,
`experiments/exp10_lis_acceleration.py`, gates in `tests/test_lis.py`.
**Question (project brief §10):** find a *correct* sampler that beats global pCN per forward
solve for the smooth-prior, sparse-data Darcy inverse problem at `N = 64–90`.

## TL;DR recommendation

Block the KLE coordinates **by spectrum, not space**. Build a likelihood-informed subspace from
the prior-preconditioned Gauss–Newton Hessian (via a cheap Darcy adjoint at the MAP) and sample with
**operator-weighted pCN preconditioned by the full Laplace covariance** — pCN about
`N(MAP, (I+H)^{-1})`, Metropolis-corrected to the exact posterior. This is the textbook DILI/LIS
accelerator and it directly attacks pCN's real bottleneck (the low-dimensional informed directions),
unlike the spatial block-Gibbs sampler, which fought the prior's correlations.

It is **correct** (passes the project's linear-Gaussian-exactness and no-data-invariance gates) and
it **converges measurably faster than global pCN per forward solve** on the directions that matter.
The improvement here is **solid but moderate** (~1.3–1.7× on convergence/robust mixing, ~2.7× on the
misfit and ‖θ‖ effective-sample rate), and it is **largest exactly where pCN is worst** — the most
informative regime. The win is bounded by a single structural effect (one joint accept/reject couples
the easy and hard directions); the concrete levers for a bigger win are in §7.

On the two forks you picked: **(1) gradients — "compare both": the gradient-free pilot route cannot
bootstrap the subspace in this regime** (a global-pCN pilot is itself stuck on the informed
directions, so its sample covariance does not identify them — principal angles ~60° vs the true
subspace). The **adjoint is cheap and exact** (one forward + `N_obs` adjoint solves, ≈528 one-time
solves at `N=64`), so **relaxing "gradient-free" is worth it.** **(2) Prototype — done**, in your
harness, measured against global pCN with the existing diagnostics.

---

## 1. The setup and the bottleneck

State is the whitened KLE coefficient vector `θ ~ N(0, I_N)`; `G = Φ Λ^{1/2} θ`, `k = exp(G)`, a
TPFA Darcy solve gives pressures, and `N_obs = 64` sensors give data with noise `σ_obs`. One sparse
SPD solve is the cost unit.

Global pCN is dimension-robust under the prior but **slow in the likelihood-informed directions**:
where the data constrains the field, the posterior is far narrower than the prior, so a prior-scale
pCN step overshoots and is rejected there. We confirmed this is a genuinely **low-rank** structure by
forming the prior-preconditioned Gauss–Newton Hessian `H = JᵀJ/σ_obs²` and looking at the posterior
standard deviation `1/√(1+μ_i)` per Hessian eigen-mode (`fig1_spectrum.png`):

| `σ_obs` | informed modes (`μ>1`) | top posterior std | tail (uninformed) |
|---|---:|---|---:|
| 0.01 (default) | 41 of 64 | 0.017, 0.034, 0.040, 0.055, … | → 1.0 |
| 0.03 | 18 of 64 | larger, fewer | → 1.0 |

At `σ_obs=0.03` the informed dimension is **18**, matching the meeting notes' "near-Gaussian,
~dimension-20 posterior". The tightest mode at `σ_obs=0.01` has posterior std 0.017, so a prior-scale
pCN step (β=0.05 → ~0.05) overshoots it ~3×; that single direction sets the chain's autocorrelation
time. **Fixing the informed directions is the whole game; spatial blocking is orthogonal to it.**

## 2. The method (operator-weighted pCN / DILI)

Split `R^N = U ⊕ U^⊥`, `U` = top-`r` eigenvectors of `H`. Because `U` and `U^⊥` are orthogonal
subspaces of the **isotropic** prior, they are prior-independent — no Schur complement, none of the
dense-precision coupling that killed spatial block-Gibbs (brief §5d). Propose:

- **informed block** — pCN about the Laplace approximation `N(m_r, diag s_i²)` on `U`
  (`s_i = 1/√(1+μ_i)`); `β_informed = 1` is an independence sampler, `< 1` is robust pCN-about-Laplace;
- **complement** — ordinary pCN about the prior on `U^⊥`.

The product kernel is reversible w.r.t. the Gaussian reference `q* = N(m_r, diag s²)_U ⊗ N(0,I)_⊥`, so
the Metropolis ratio targets the **true** posterior exactly at **one forward solve per step**:

```
log α = [log π(θ') − log q*(θ')] − [log π(θ) − log q*(θ)]
```

Taking `r = N` makes `q*` the **full Laplace covariance** `(I+H)^{-1}` and removes the split entirely
— this was the best variant in the ablation (§5). With `r = 0` it reduces *exactly* to global pCN.

**Reference point.** The Laplace proposal is a *local* sampler: excellent near the mode, useless far
from it (from an over-dispersed start the tight reference makes `π/q*` astronomically large and the
chain sticks). So we anchor it at the **MAP**, found by a damped/​line-searched Gauss–Newton from
`θ=0` (a plain GN step diverges because `k=exp(G)` is wildly nonlinear in the tails). At `N=64` this
is **528 one-time solves**, charged to the LIS budget.

## 3. Correctness (the gates — `tests/test_lis.py`, all pass)

- **adjoint == finite-difference Jacobian** to `3.9e-8` (the cheap-gradient route is the same operator);
- **no-data invariance**: with no likelihood the kernel leaves `N(0,I)` invariant (and `r=0` *is* pCN);
- **linear-Gaussian exactness**: on a linear forward map the Laplace reference is exact, so every
  proposal is accepted (`α≡1`) and the chain reproduces the analytic Gaussian posterior to MC error.

This is the same verification discipline used for the precision block-Gibbs sampler; the LIS
construction is, if anything, simpler to prove because the blocks are prior-orthogonal.

## 4. Headline result — convergence on the bottleneck (`fig2_rhat.png`)

`N=64`, `σ_obs=0.01`, 4 chains, matched starts drawn from the Laplace posterior, equal sampling
budget (550 iters/chain). R̂ is the worst over the leading informed-direction projections and ‖θ‖
(the directions that actually mix slowly):

| method | accept | end R̂ (worst informed) | ESS/1k solves | ESS/sec | rel-k(meanG) |
|---|---:|---:|---:|---:|---:|
| global pCN (β=0.05) | 0.13 | **4.07** | 8.00 | 0.526 | 0.479 |
| LIS full-Laplace (β=0.5) | 0.26 | 1.88 | **9.32** | **0.810** | 0.477 |
| LIS full-Laplace (β=0.6) | 0.21 | **1.36** | 8.18 | 0.695 | 0.477 |

At equal forward-solve budget LIS drives the worst informed direction to R̂≈1.4 while pCN is stuck at
~4; ESS/sec is ~1.5×. **pCN's chains sit at their starting informed-coordinates — it cannot move the
tight modes — so they disagree (high R̂); LIS samples them and the chains agree.**

**Measurement-hygiene caveat (brief §7).** Because pCN does not move the informed directions, a single
pCN chain looks *low-variance* there and its per-chain ESS is **inflated** (a near-constant series has
IAT≈1). So ESS-per-solve is only trustworthy once R̂≤1.1, which is why R̂-vs-solves is the honest
headline. On a single chain started *at* a true posterior sample (no stuck artifact), LIS beats pCN
**2.7–2.9× on the misfit and ‖θ‖ effective-sample rate** and is comparable on the individual leading
informed eigendirections.

## 5. Rank ablation — use the *full* Laplace, not a thin subspace

| informed rank `r` | accept | end R̂ |
|---|---:|---:|
| 10 | 0.00 | 32.0 |
| 20 | 0.03 | 3.54 |
| 41 (`μ>1`) | 0.18 | 1.58 |
| 64 (full Laplace) | 0.21 | **1.36** |

Low rank is *worse*: it leaves moderately-informed modes in the complement, where prior-scale pCN
overshoots them and acceptance collapses. The informed block must contain **every** mode the data
touches; the clean choice is the full Laplace covariance (`r=N`).

## 6. Regime dependence — LIS wins most where pCN fails most

At `σ_obs=0.03` (18 informed modes, tightest modes less tight), a tuned pCN reaches R̂≈1.78 while LIS
reaches ≈1.34 over ~2000 solves — LIS still ahead but the gap narrows, because the informed
directions are easier for pCN. **LIS's relative advantage is largest in the most informative regime
(`σ_obs=0.01`), exactly where pCN is most stuck.** Posterior-mean fields are equally plausible at
equal budget (rel-k ≈ 0.48 for both; `fig3_fields.png`), and both recover the large-scale truth.

## 7. Honest assessment and the levers for a bigger win

The win is **real, correct, and directly on the bottleneck**, but **moderate** in this problem. The
binding constraint is structural: a *single* joint accept/reject (acceptance ~0.2–0.5, capped by the
posterior's mild non-Gaussianity) couples the many easy uninformed modes to the few hard informed
ones — a rejection driven by the informed part also blocks the uninformed update. Three concrete,
correctness-preserving levers:

1. **Metropolis-within-Gibbs with a near-free complement refresh.** Update the uninformed complement
   by an *independent prior draw* (β=1) as its own MH step: since the likelihood barely depends on
   those modes, acceptance ≈ 1 and they decorrelate in **one** step. This decouples the easy modes
   from the informed accept/reject and should lift overall ESS/solve at the cost of one extra solve
   per sweep (still cheap; the brief's cost model counts solves but this buys near-iid complement).
2. **Gradient-based geometric MCMC on the informed block.** The adjoint already gives the misfit
   gradient for ≈ one extra solve; a manifold-MALA / stochastic-Newton step *inside* `U` raises
   informed-direction acceptance above what pCN-about-Laplace allows — this is the lever that attacks
   the bottleneck itself, not just the easy modes. (Relaxing "gradient-free" is the enabler.)
3. **A better reference than the Laplace Gaussian** (heavier-tailed or a cheap transport map fit from
   the pilot/MAP curvature) to push informed-block acceptance toward 1 where the posterior is
   non-Gaussian.

Lever 1 helps the uninformed many; lever 2/3 attack the informed few that set the convergence rate.

## 8. How to reproduce

```bash
# one-time setup (MAP + adjoint Hessian, cached) and the comparison:
python -m experiments.exp10_lis_acceleration --setup       --modes 64
python -m experiments.exp10_lis_acceleration --method global_pcn  --modes 64 --iters 550 --beta 0.05
python -m experiments.exp10_lis_acceleration --method lis_adjoint  --modes 64 --iters 550 --rank 64 --beta-informed 0.6
python -m experiments.exp10_lis_acceleration --aggregate   --modes 64
pytest tests/test_lis.py -q     # the correctness gates
```

Chains start from Laplace-posterior draws (LIS is a local sampler — see §2); the MAP/Hessian setup
cost is charged to the LIS solve budget. Figures land in `outputs/exp10/`.

## 9. Files delivered

- `src/mcmc_multiscale/lis.py` — adjoint Jacobian + GN-Hessian, MAP (Levenberg–Marquardt),
  informed-subspace builders (Hessian **and** gradient-free pilot), `principal_angles`, and the
  operator-weighted pCN proposal (`make_lis_proposal`).
- `experiments/exp10_lis_acceleration.py` — compute-fair multi-chain harness (R̂-vs-solves,
  ESS/1k-solves, ESS/sec, rank ablation, field recovery).
- `tests/test_lis.py` — the four correctness gates.
- `outputs/exp10/` — `fig1_spectrum.png` (informed spectrum), `fig2_rhat.png` (convergence headline),
  `fig3_fields.png` (field recovery).
