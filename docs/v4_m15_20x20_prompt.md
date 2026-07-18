# M15 — reproduce Aidan's 10× with a cheap (adjoint) basis

Paste-ready task spec for Claude Code. This is the first result of the preconditioning paper.
`SPEC.md` §3.12 (posterior-informed basis) and §3.13 (metrics) are authoritative; read them first.

---

## The one question this experiment answers

Aidan gets ~10× by building a basis from a **converged pilot** (64 chains × 100k iterations, MPSRF
converging ~40k), then sampling preconditioned in ~3–3.5k. The pilot costs more than the speedup
saves. **Does a basis built from the adjoint Gauss-Newton Hessian at the MAP (~hundreds of solves,
no pilot) reproduce the same 10×?** If yes, the speedup becomes real *and* practical.

Everything below is on **Aidan's exact config**: 20×20 grid, **squared-exponential** kernel, ℓ=0.16,
24 modes.

## Branch

`feature/m15-posterior-informed-basis`, experiment `experiments/exp13_posterior_informed.py`.

## Standing rules

- `python -m pytest` (not plain `pytest`), `ruff check .`, `black --check .` all clean before PR.
- float64; explicit `numpy.random.Generator` threaded everywhere; never global `np.random.*`.
- Experiments write figures + tables to `outputs/exp13/` (gitignored); `matplotlib.use("Agg")`.
- **Reuse the existing correct machinery; do not reinvent it.** In particular do NOT write a new
  preconditioned sampler — use `lis.make_lis_proposal`, which is already Metropolis-corrected to the
  exact posterior and already gated in `tests/test_lis.py`. Every new numerical function ships a test.

## Two small library additions first (each with a test)

1. **Squared-exponential covariance** in `covariance.py`, alongside `exp_covariance`:
   `sqexp_covariance(points, sigma, corr_length)` → `sigma**2 * exp(-||x-x'||^2 / (2 corr_length^2))`.
   Test: symmetric, unit diagonal for sigma=1, PSD, matches a hand-computed 2-point value.

2. **MPSRF** (multivariate potential scale reduction factor, Brooks & Gelman 1998) in
   `diagnostics.py`. This is the statistic Aidan reports, so we must speak it — scalar Gelman-Rubin
   is not enough. Signature `mpsrf(chains)` where `chains` is `(n_chains, n_samples, n_params)`;
   returns the scalar `sqrt((n-1)/n + (m+1)/m · λ_max(W^{-1} B/n))` per the standard definition.
   Tests: → 1 for many well-mixed identical-distribution chains; > 1 for deliberately offset chains;
   reduces to scalar `gelman_rubin` when `n_params == 1`.

## The experiment (`exp13_posterior_informed.py`)

Structure it like `exp10_lis_acceleration.py` (build KLE + truth + data, then drive the generic
`metropolis_hastings` engine) — NOT like the conditioned samplers. Steps:

1. **Setup.** Build the SE-kernel KLE at 20×20 with 24 modes (`sqexp_covariance` → `top_eigenpairs`).
   Fix a synthetic truth `theta_true`, generate pressure data with noise. **Noise is an assumption
   to flag** — Aidan didn't state σ_obs for this run; default to something reasonable (his MATLAB
   used absolute 1e-3), expose it as a CLI flag, and print it prominently as "provisional, confirm
   with Aidan." The 10× gap should be robust to the exact level.

2. **Baseline — global pCN in standard coordinates.** `make_pcn_proposal` + `metropolis_hastings`
   with `log_posterior`. Over-dispersed starts. Track **MPSRF vs iteration** across chains; record
   the iteration where MPSRF first drops below **1.2** (Aidan's threshold — his plot lines are at
   1.2 and 1.0). Target: ~40k, confirming our setup matches his. Keep the post-convergence samples —
   they seed the empirical basis in step 4.

3. **Cheap basis (ours).** `lis.gauss_newton_map` for the MAP, then
   `lis.build_informed_subspace_adjoint` at the MAP with **full rank (rank = 24, do NOT truncate** —
   SPEC §3.12 / pitfall 12). Record the solve count it charges (both functions return it) — this is
   the honest basis cost.

4. **Pilot basis (his), for comparison.** From the converged baseline samples,
   `lis.informed_subspace_from_samples` at **full rank**. This is expensive (it needs the pilot) —
   that's the point — but on 20×20 it's affordable and gives the key comparison in step 5.

5. **Principal angles.** `lis.principal_angles(U_cheap, U_pilot)`. Small angles ⇒ the cheap basis
   *is* the pilot basis, obtained for hundreds of solves instead of millions of iterations. **This is
   the single cleanest number in the experiment** — report it even if the speed comparison is noisy.

6. **Preconditioned pCN, cheap basis.** `make_lis_proposal(subspace_cheap, beta_informed,
   beta_complement)` (full Laplace, rank=24) + `metropolis_hastings`. Same chain count and
   over-dispersed starts as the baseline. MPSRF vs iteration; convergence iteration at MPSRF < 1.2.
   Target: ~3–4k. **Speedup = baseline_iters / preconditioned_iters.**

7. **(Optional) Preconditioned pCN, pilot basis** — should land near Aidan's ~3.25k, cross-checking
   that our harness reproduces his number with his own basis.

## Chain count / budget

20×20 with 24 modes is tiny (400 cells; a forward solve is ~ms). Matching Aidan's 64 chains locally
is a stretch (64 × 40k ≈ 2.6M solves), so make `--n-chains` a flag, default to something feasible
(8–16) that still gives a stable MPSRF, and note that 64 is a cluster stretch-goal. The qualitative
10× gap should appear with fewer chains. Cap baseline iterations (e.g. 60k) so a non-converging run
fails fast rather than hanging.

## Outputs to `outputs/exp13/`

- `exp13_mpsrf.png` — MPSRF vs iteration, baseline vs cheap-basis (and pilot-basis if run) on shared
  axes, with the 1.2 and 1.0 reference lines. **This is the figure for Monday.**
- `exp13_table.md` — convergence iteration per method, speedup ratio, **basis cost** (cheap: N
  solves; pilot: n_chains × pilot_iters), principal angles (min/max/mean), and the σ_obs used with
  its "provisional" flag. Print to stdout too.

## Honest accounting (the paper's core point — get this right)

The table must put the costs side by side in comparable units: cheap basis = *setup solves*; pilot
basis = *pilot iterations*. Do not quietly drop the pilot cost. If the cheap basis reproduces the
speedup, the sentence the table supports is: "same 10×, basis for ~N solves instead of ~2.6M
iterations." If it does **not**, that is equally a result — report how far the principal angles are
from zero and where it breaks down (SPEC §3.12: the Laplace basis only matches when the posterior is
near-Gaussian).

## Correctness

No new correctness gates needed for the sampler — `make_lis_proposal` is already gated (no-data
invariance, linear-Gaussian exactness, determinism) in `tests/test_lis.py`, and those are
config-independent. Do add the tests for the two new library functions (SE covariance, MPSRF) above.

## Verification

- `python -m pytest`, `ruff check .`, `black --check .` clean.
- `python -m experiments.exp13_posterior_informed` produces the figure + table in `outputs/exp13/`.
- Sanity: baseline converges near ~40k (setup matches Aidan); cheap-basis principal angles are small
  if the posterior is near-Gaussian at this config.

## Explicitly not in scope

M14(c) freeze, M16, M17. The 48×48 exponential-kernel reference set. exp12. No changes to the
conditioned samplers or the forward solver.

---

**Paste-to-Claude-Code opener:**

> Read `SPEC.md` §3.12 and §3.13, and `docs/v4_m15_20x20_prompt.md`. Implement M15 on branch
> `feature/m15-posterior-informed-basis` exactly as that prompt specifies — the two library
> additions first (squared-exponential covariance, MPSRF), each with tests, then
> `experiments/exp13_posterior_informed.py` reproducing Aidan's 20×20 config and comparing a cheap
> adjoint-Hessian basis against a pilot basis. Reuse `lis.make_lis_proposal` /
> `build_informed_subspace_adjoint` / `informed_subspace_from_samples` / `principal_angles` — do not
> write a new preconditioned sampler. **Show me your plan before writing any code.**
