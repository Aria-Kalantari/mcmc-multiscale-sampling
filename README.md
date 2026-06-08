# MCMC Multiscale Sampling with Overlapping Subdomains

**A Python research prototype for Bayesian inversion of a spatially varying log-permeability field — combining Gaussian random fields, Karhunen–Loève expansions, TPFA Darcy flow, and Metropolis–Hastings sampling to study the numerical stability of overlapping-subdomain MCMC updates.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Lint](https://img.shields.io/badge/style-ruff%20%2B%20black-000000)

---

## Overview

This project tackles a Bayesian inverse problem from subsurface flow: given sparse pressure observations, recover the underlying spatially varying permeability field and quantify the uncertainty in that estimate. The forward physics is a two-point flux approximation (TPFA) Darcy solver; the unknown field is represented with a Gaussian random field and a Karhunen–Loève expansion; and the posterior is explored with Metropolis–Hastings MCMC using preconditioned Crank–Nicolson (pCN) proposals.

The research focus is **numerical stability of localized, overlapping-subdomain updates** — a building block for scaling MCMC on multiscale fields. The headline result is a concrete failure mode and its fix:

> An arbitrary LU/pivot particular solution for the local hard-conditioning system can carry **hidden null-space content** that accumulates under repeated updates and silently corrupts the chain. **SVD / minimum-norm conditioning and stabilized LU remove that hidden component.** Soft conditioning then offers a tunable residual-versus-stability trade-off, and red-black sweeps extend the schedule across the coarse partition with deterministic frozen snapshots.

It is presented honestly as a research prototype: limitations and open questions are documented, and unverified speedups are *not* claimed.

---

## What it demonstrates

- **Bayesian inverse modeling** end to end: prior (GRF + KLE) → forward model (TPFA) → likelihood → posterior sampling.
- **Numerical-linear-algebra depth**: null-space analysis, SVD/minimum-norm vs LU-pivot particular solutions, stabilized factorizations, and the practical consequences of each for an iterative sampler.
- **Rigorous convergence diagnostics**: integrated autocorrelation time (IAT), effective sample size (ESS), Gelman–Rubin R-hat, 90% credible-interval coverage, plus compute-fair metrics (ESS per 1000 forward solves and ESS per wall-second) against a plain global-pCN reference chain.
- **Reproducible experiment design**: nine headless experiments (`exp01`–`exp08c`) isolate sanity checks, instability reproduction, stability fixes, red-black scheduling, posterior recovery, and a decision-grade convergence comparison.
- **Engineering maturity**: a packaged `src/` library, ~25 pytest modules, ruff + black enforcement, a derivation document, and an interactive Streamlit dashboard.

---

## Method at a glance

| Component | Implementation |
|---|---|
| Unknown field | Log-permeability as a Gaussian random field (`field.py`, `covariance.py`) |
| Dimensionality reduction | Karhunen–Loève expansion (`kle.py`) |
| Forward model | TPFA Darcy pressure solver (`forward/tpfa.py`) |
| Sampler | Metropolis–Hastings with pCN proposals (`mcmc.py`, `proposals.py`, `sampler.py`) |
| Local updates | Overlapping-subdomain hard (SVD/min-norm + stabilized LU), soft, and red-black conditioning (`conditioning/`) |
| Diagnostics | IAT, ESS, R-hat, coverage (`diagnostics.py`) |
| Interface | Streamlit dashboard (`app/streamlit_app.py`) |

---

## Repository structure

```text
mcmc-multiscale-sampling/
├── src/mcmc_multiscale/
│   ├── forward/tpfa.py            # TPFA Darcy pressure solver
│   ├── conditioning/             # hard (SVD/LU), soft, null-space, projection
│   ├── field.py, covariance.py   # Gaussian random field
│   ├── kle.py                    # Karhunen–Loève expansion
│   ├── mcmc.py, proposals.py     # Metropolis–Hastings + pCN
│   ├── sampler.py, subdomain.py  # multiscale / overlapping-subdomain sampler
│   └── diagnostics.py            # IAT, ESS, R-hat, coverage
├── experiments/                  # exp01–exp08c reproducibility scripts
├── app/streamlit_app.py          # interactive dashboard
├── docs/conditioned_posterior_derivation.md
└── tests/                        # ~25 pytest modules
```

## Quickstart

```bash
python -m pip install -r requirements.txt          # numpy, scipy, matplotlib, streamlit, pytest, ruff, black

python -m pytest                                   # verify
python -m ruff check . && python -m black --check . # lint/format

python -m experiments.exp08_convergence_diagnostics   # ESS / R-hat / coverage
python -m streamlit run app/streamlit_app.py          # interactive dashboard
```

Opt-in deep comparisons:

```bash
python -m experiments.exp08_convergence_diagnostics --long
python -m experiments.exp08c_recovery_decision --decide
python -m experiments.exp08c_recovery_decision --resolve
```

---

## Limitations

Research prototype, not production simulation software · single-machine, sequential · synthetic examples only (no private data) · the M8 global-prior route is a baseline with a modest recovery improvement, and the constrained-manifold route in the spec remains future work · the 2-color red-black schedule is deterministic frozen-snapshot scheduling, not an exact parallel-independence guarantee under overlap.

---

## Skills demonstrated

Bayesian inference & MCMC (Metropolis–Hastings, pCN) · numerical linear algebra (SVD, LU, null-space, stabilized factorizations) · scientific computing & PDE forward models (TPFA Darcy flow) · uncertainty quantification & convergence diagnostics (ESS, R-hat, coverage) · reproducible experiment design · Python library packaging, testing (pytest), and linting (ruff/black).

---

*Author: Arya Kalantari · [github.com/Aria-Kalantari](https://github.com/Aria-Kalantari)*
