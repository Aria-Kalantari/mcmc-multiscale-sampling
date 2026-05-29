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
```

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
- The 2-color red-black schedule is deterministic frozen-snapshot scheduling.
  It is not an exact same-color parallel-independence guarantee under overlap.
  Exact independence would need a stronger coloring strategy, such as
  4-coloring, or additional overlap analysis.
- The Streamlit dashboard is a batch-run viewer; it does not implement
  pause/resume streaming.
