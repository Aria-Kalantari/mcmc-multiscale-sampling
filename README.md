# MCMC Multiscale Sampling with Overlapping Subdomains

This repository is a Python research prototype for Bayesian inversion of a
spatially varying log-permeability field. It combines Gaussian random fields,
Karhunen-Loeve expansions, TPFA Darcy flow, Metropolis-Hastings sampling, and
overlapping-subdomain conditioning to study stability in multiscale MCMC
updates.

The main numerical finding is that an arbitrary LU/pivot particular solution
for the local hard-conditioning system can carry hidden null-space content that
accumulates under repeated updates. SVD/minimum-norm conditioning and stabilized
LU remove that hidden component. Soft conditioning provides a tunable
residual-versus-stability tradeoff, and red-black sweeps extend the update
schedule across the coarse partition with deterministic frozen snapshots.

## Install

```bash
python -m pip install -r requirements.txt
```

## Verify

```bash
python -m pytest
python -m ruff check .
python -m black --check .
```

## Experiments

```bash
python -m experiments.exp01_static_conditioning
python -m experiments.exp02_forward_bayes_sanity
python -m experiments.exp03_mcmc_gaussian_sanity
python -m experiments.exp04_reproduce_instability
python -m experiments.exp05_stability_fixes
python -m experiments.exp06_red_black_updates
python -m experiments.exp07_posterior_recovery
python -m experiments.exp08_convergence_diagnostics
python -m experiments.exp08c_recovery_decision
```

## Acceptance Modes

`Config.acceptance` defaults to `posterior`. The M8 posterior baseline adds a
projected global-KLE field prior and the hard-null pCN proposal correction.
The low-level conditioned samplers retain an explicit `likelihood_only` mode so
the M4/M5 instability studies remain reproducible. `exp07_posterior_recovery`
compares both modes for single-subdomain and red-black updates.

`exp08_convergence_diagnostics` adds multi-chain convergence and recovery
diagnostics: IAT, ESS, Gelman-Rubin R-hat, 90% log-field credible-interval
coverage, and a plain global-pCN reference chain for the same synthetic
posterior.

Use the opt-in convergence deep-dive for a longer compute-fair comparison:

```bash
python -m experiments.exp08_convergence_diagnostics --long
```

The deep-dive reports conservative ESS per 1000 forward solves and ESS per
wall-second alongside recovery and coverage. The plain exp08 command keeps its
responsive defaults for routine development.

For the opt-in recovery decision, compare red-black directly against global
pCN with matched unit-scale starts, trajectory checkpoints, and a wall cap:

```bash
python -m experiments.exp08c_recovery_decision --decide
```

This excludes the structurally frozen single-subdomain harness from the
recovery verdict and reports whether the requested proposal budget or wall cap
ended each scheme.

For the resolving comparison, use the larger opt-in profile:

```bash
python -m experiments.exp08c_recovery_decision --resolve
```

The resolving report classifies the final relative-k checkpoint windows as
flattened or still moving, reports whether each data-misfit tail is descending
toward the noise floor, and keeps R-hat, ESS, coverage, solve counts, and wall
time visible as supporting diagnostics. Budgets remain configurable with
`--sweeps`, `--pcn-iters`, `--max-seconds`, and `--checkpoints`; the fast
default command remains unchanged.

## Dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

The dashboard visualizes the single-subdomain sampler, red-black sweeps via an
`Update scheme` control, LU/SVD stability comparisons, and a compact M5
stability-fix comparison. Red-black defaults to stable SVD hard conditioning.

## Repository Structure

```text
src/mcmc_multiscale/          numerical library
src/mcmc_multiscale/forward/  TPFA pressure solver
src/mcmc_multiscale/conditioning/
                              hard, stabilized, and soft conditioning
experiments/                  headless reproducibility experiments
app/                          Streamlit dashboard
tests/                        pytest verification suite
```

## Limitations

- Research prototype, not production simulation software.
- Single-machine sequential implementation only.
- Synthetic examples only; no private data are included.
- The M8 global-prior route is a baseline. Its short default comparison remains
  only a modest recovery improvement; the constrained-manifold route in the
  project specification remains future work.
- The 2-color red-black schedule is deterministic frozen-snapshot scheduling.
  It is not an exact same-color parallel-independence guarantee under overlap.
  Exact independence would need a stronger coloring strategy, such as
  4-coloring, or additional overlap analysis.
- The Streamlit dashboard is a batch-run viewer; it does not implement
  pause/resume streaming.
