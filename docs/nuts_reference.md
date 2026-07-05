# A NUTS gold-standard reference for the Darcy posterior

**Status:** implemented + converged reference run (M13). **Code:**
`src/mcmc_multiscale/nuts.py`, `experiments/exp11_nuts_reference.py`, gates in
`tests/test_nuts.py`. **Question (SPEC §3.10 / M13):** the M9 convergence
experiments stalled (nothing reached R̂ ≲ 1.1), so the posterior-mean rel-k
"floor" (~0.5–0.8) and interval coverage could not be attributed — *is 0.6 the
true posterior floor, or an unconverged chain?* Build a trusted, gradient-based
sampler that converges, and read off the reference numbers.

## TL;DR

Hand-rolled **No-U-Turn Sampler** (Hoffman–Gelman 2014, multinomial/log-weight
variant à la Betancourt 2017) on the whitened potential
`U(θ) = ½‖θ‖² + Φ(θ)`, with the exact gradient `∇U = θ + Jᵀr/σ_obs²` from the
validated Darcy adjoint. HMC/NUTS is a *global* sampler with no
prior-scale-step bottleneck, so — unlike pCN, block-Gibbs, or LIS — it moves the
tight likelihood-informed directions freely and **converges to R̂ ≤ 1.01**.

The gradient is cheap: the scalar-misfit gradient needs **one forward + one
adjoint solve** (`--gradient single_solve`, 2 solves/gradient), reusing the LIS
adjoint machinery and gated to `1e-8` against the validated full Jacobian. The
role is **trust, not speed**: NUTS is the converged ground truth the accelerators
are measured against, not a per-solve-efficiency claim.

**Key finding.** The converged reference posterior-mean **rel-k floor is ≈ 0.49
(N=64) → 0.52 (N=90)**, which *brackets* what global pCN and LIS reported at
equal budget (exp10: rel-k(meanG) ≈ 0.477–0.479). So the ~0.5 rel-k is the **true
posterior floor**, set by the smooth prior + sparse data (64 sensors) + KLE
truncation — **not** an unconverged-chain artifact. The M9 ambiguity is resolved:
the accelerators' posterior *mean* was already right; what they could not do was
*prove* convergence (R̂ ≤ 1.01). NUTS proves it.

**One deviation from the plan, exactness-preserving.** We agreed to an identity
mass matrix (θ is whitened, so the prior is isotropic). But the `σ_obs = 0.01`
*likelihood* makes the posterior sharply anisotropic — the Laplace precision
`I+H` has condition number **~3.7e3** at `N=64` — so identity-metric NUTS
collapses to a tiny step and maxes out its tree depth (~1000 gradient solves per
iteration → the run would take ~2 days). We instead sample in whitened
coordinates `φ` with `θ = C φ`, `C = chol((I+H)^{-1})` from a cheap adjoint
Gauss-Newton Hessian at the MAP. **A linear reparametrisation (equivalently a
fixed dense mass matrix `M = I+H`) leaves the target distribution *exactly*
unchanged** — it is a preconditioner, not an approximation — so `θ = Cφ` are
exact posterior draws and the sampler remains a gold standard. This drops the
tree depth from 10 (maxed) to ~3 and the run to ~1–1.5 h/mode. The one-time
MAP + Hessian cost (**528 solves** at `N=64`) is charged to the solve budget,
exactly as exp10 (LIS) does. `--precondition none` recovers the raw
identity-metric sampler (correct but impractically slow here).

---

## 1. The setup and the role

State is the whitened KLE coefficient vector `θ ~ N(0, I_N)`; `G = Φ Λ^{1/2} θ`,
`k = exp(G)`, a TPFA Darcy solve gives pressures, and `N_obs = 64` sensors give
data with noise `σ_obs = 0.01`. The exact posterior is
`π(θ) ∝ exp(−U(θ))`, `U(θ) = ½‖θ‖² + Φ(θ)`,
`Φ(θ) = ½/σ_obs² ‖R(p(θ)) − y‖²` (`U = −log_posterior`).

pCN, block-Gibbs and LIS are all held back by the same structural fact: where the
data constrains the field, the posterior is far narrower than the prior, so a
prior-scale move overshoots and is rejected. NUTS sidesteps this entirely by
following the gradient with Hamiltonian dynamics: it does not propose from the
prior, so there is no prior-scale-step penalty in the informed directions. That
is exactly why it can *converge* here where the M9 samplers could not, and why it
is the right tool to pin the reference numbers rather than to win on speed.

## 2. The method

**Potential and gradient.** `U(θ) = ½‖θ‖² + Φ(θ)` reuses `bayes.misfit`.
The gradient `∇U(θ) = θ + Jᵀr/σ_obs²`, `r = R(p(θ)) − y`, has two backends
(`make_potential(..., gradient=...)`):

- `full_jacobian` — forms the validated `J = lis.darcy_jacobian_adjoint`
  (1 + N_obs solves) and returns `θ + Jᵀr/σ_obs²`. The correctness anchor.
- `single_solve` (default in exp11) — the scalar-misfit gradient needs only the
  *contraction* `Jᵀr`, not all of `J`. Since `Jᵀr = −Gmatᵀ (T^{-1}(R r))`, one
  adjoint solve `w = T^{-1}(R r)` (the TPFA operator `T` is symmetric, so it
  reuses the forward `splu`) plus the per-mode residual directional derivatives
  `lis._resid_dir_deriv` gives `∇U = θ − gm/σ_obs²` in **2 solves**. The leading
  minus is the `−(Lamᵀ Gmat)` sign of `darcy_jacobian_adjoint`.

**Leapfrog** (diagonal metric `M`, momentum `r ~ N(0,M)`, kinetic `½ rᵀM⁻¹r`):
`r½ = r − (ε/2)∇U(θ)`, `θ' = θ + ε M⁻¹ r½`, `r' = r½ − (ε/2)∇U(θ')`. The new
gradient is returned and threaded through the tree so `∇U` is never recomputed at
a visited state — each gradient is the whole solve cost.

**No-U-turn tree.** A recursive balanced tree doubles the trajectory in a random
time direction until either end makes a U-turn
(`(θ⁺−θ⁻)·M⁻¹r ≥ 0` fails at either endpoint) or a leaf diverges
(`H − H₀ > 1000` or non-finite). Trajectory sampling is **multinomial** in
log-weights (`w = exp(−(H−H₀))`, combined by `logaddexp`), with biased
progressive selection favouring the newer subtree — higher ESS than the original
slice sampler and no `exp(−H)` underflow bookkeeping.

**Dual-averaging step size** (Nesterov; `γ=0.05, t₀=10, κ=0.75`,
target-accept 0.9 for the reference run), initialised by the Alg. 4
reasonable-ε heuristic and frozen to the dual-averaged value at the end of
warmup.

**Preconditioning** (see TL;DR): sample `φ`, report `θ = Cφ`. **The metric `C` is
computed once during setup (the MAP plus a single adjoint Gauss-Newton Hessian)
and then frozen — it never adapts to the samples — so `θ = Cφ` is a constant
linear change of variables and the sampled target `exp(−U(θ))` is left exactly
invariant; the gold-standard claim is therefore manifest, not approximate.**
Chains start over-dispersed in whitened coordinates, `φ₀ = φ_MAP + s·z`,
`z ~ N(0,I)`, `s = 2` — dispersed by 2 posterior-stds (so R̂ is meaningful) yet
in-basin (no `exp(G)` overflow). A per-gradient divergence guard returns `U = ∞`
when a leapfrog overshoot makes `k = exp(G)` overflow or the TPFA factor
singular, so the integrator shrinks the step instead of crashing.

## 3. Correctness (the gates — `tests/test_nuts.py`, all pass)

The project's standard verification discipline for a new sampler (mirrors
`tests/test_lis.py`):

| gate | assertion | tolerance |
|---|---|---|
| **∇U == finite difference** | `‖∇U_adjoint − ∇U_fd‖/‖∇U_fd‖` | `< 1e-5` |
| **single == full gradient** | `‖∇U_single − ∇U_full‖/‖∇U_full‖` | `< 1e-8` |
| **no-data ⇒ N(0,I)** | `U(θ)=½‖θ‖²` recovers prior mean/cov | mean `< 0.1`, cov−I `< 0.15` |
| **linear-Gaussian exactness** | chain reproduces analytic `μ_post, Σ_post` | max‑abs `< 0.05` |
| **determinism** | fixed seed → bit-for-bit identical chain | exact |

The linear-Gaussian gate is the decisive trust gate: on a linear forward map `L`
with `U(θ) = ½‖θ‖² + ½‖Lθ−y‖²/σ²`, NUTS reproduces `Σ_post = (I + LᵀL/σ²)^{-1}`,
`μ_post = Σ_post Lᵀy/σ²` to Monte-Carlo error — the same gate that anchors trust
in `test_gaussian_block.py` and `test_lis.py`.

## 4. Headline reference numbers

`σ_obs = 0.01`, 4 chains, matched over-dispersed starts, `single_solve` gradient,
Laplace-preconditioned NUTS, **target-accept 0.9**, target R̂ ≤ 1.01 (worst over
misfit, ‖θ‖, and the leading KLE coordinates). Full run: 2000 sampling iters +
600 warmup per chain (8000 post-warmup samples per mode). Numbers below are from
`outputs/exp11/run_log.txt`.

| N | end R̂ | ESS (min scalar) | accept | tree depth | divergences | rel-k floor | 90% coverage | misfit floor (noise floor 32) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | **1.0010** | 2187 | 0.91 | 3.0 | 0 / 8000 | **0.491** | 0.953 | 29.8 |
| 90 | **1.0002** | 2041 | 0.91 | 3.0 | 0 / 8000 | **0.516** | 0.968 | 32.9 |

**Reading the numbers.** R̂ ≤ 1.01 with ESS ≈ 2000 is a *converged* chain — the
thing M9 never had. The **rel-k floor ≈ 0.49 (N=64) → 0.52 (N=90)** brackets the
exp10 pCN/LIS value (0.477–0.479): the smooth-prior + sparse-data posterior
simply cannot recover `k` below ~0.5 relative error at this sensor budget, and
that is a property of the *posterior*, now confirmed by a trusted sampler, not of
any one algorithm. (rel-k rises slightly from N=64 to N=90 because the extra 26
weakly-informed modes add prior-scale variance to the mean field without new data
to constrain them.) The **misfit floor ≈ 30–33** sits right at the noise floor
`N_obs/2 = 32` (the data are fit to the noise level, not overfit to 0). **90%
coverage ≈ 0.96** is slightly conservative — the credible intervals cover the
truth at ~the nominal rate.

**Divergences → 0.** At target-accept 0.9 both modes report **0/8000**
divergences (down from 9 and 26 at 0.8). Crucially, R̂, ESS, rel-k, coverage and
the misfit floor are all unchanged within Monte-Carlo error between the two runs
(rel-k Δ ≤ 0.006, coverage Δ ≈ 0.01 across two independent 8000-sample
estimates) — as they must be, since raising target-accept only shortens the
integrator step, never the target. That agreement is itself evidence the chain
is genuinely converged: the reference numbers are a property of the posterior,
not of the step size.

## 5. Cost and honest positioning

Per gradient: **2 solves** (`single_solve`); per NUTS iteration ≈ `2·n_leapfrog`
solves with `n_leapfrog ≈ 15` at tree depth 3 after preconditioning. One-time
setup (MAP + adjoint Hessian for the preconditioner) is **528 solves** at `N=64`,
charged to the budget. This makes the per-solve accounting honest, but NUTS is
**not** offered as an accelerator: it does more solves per effective sample than
LIS. Its value is that it *converges* and therefore *defines* the reference —
rel-k floor, coverage, misfit floor — that block-Gibbs, global pCN and LIS are
benchmarked against. The R̂-vs-solves curve uses `cumsum(n_leapfrog)·2 + setup +
warmup` on the x-axis (variable leapfrogs/iter), not `n_chains·iter`.

## 6. How to reproduce

```bash
python -m experiments.exp11_nuts_reference --setup     --modes 64,90
# the reference run (the numbers in §4):
python -m experiments.exp11_nuts_reference --run --modes 64,90 \
    --iters 2000 --warmup 600 --n-chains 4 --target-accept 0.9
python -m experiments.exp11_nuts_reference --aggregate  --modes 64,90
pytest tests/test_nuts.py -q     # the correctness gates
```

`--gradient full_jacobian` swaps in the validated-Jacobian gradient (slower,
correctness anchor); `--precondition none` runs the raw identity-metric sampler.
Figures and the run log land in `outputs/exp11/`.

## 7. Files delivered

- `src/mcmc_multiscale/nuts.py` — potential/gradient closures (both backends),
  leapfrog, recursive no-U-turn multinomial tree, dual-averaging step-size
  adaptation, optional diagonal mass, the `nuts_sample` driver, `NUTSState`.
- `experiments/exp11_nuts_reference.py` — multi-chain reference harness with
  Laplace preconditioning, R̂-vs-solves, ESS, rel-k floor, 90% coverage, and the
  data-misfit floor.
- `tests/test_nuts.py` — the five correctness gates.
- `outputs/exp11/` — `exp11_rhat_vs_solves_N{64,90}.png`,
  `exp11_misfit_trace_N{64,90}.png`, `exp11_fields_N{64,90}.png`, and
  `run_log.txt`.
