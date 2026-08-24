# MCMC Multiscale Sampling with Overlapping Subdomains

**A Python research prototype for Bayesian inversion of spatially varying permeability fields, with Gaussian random fields, Karhunen–Loève expansions, TPFA Darcy flow, posterior-correct MCMC, and convergence-focused sampler comparisons.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Lint](https://img.shields.io/badge/style-ruff%20%2B%20black-000000)

---

## Overview

This project studies a Bayesian inverse problem from subsurface flow: recover a spatially varying log-permeability field from sparse, noisy pressure observations and quantify the uncertainty in that recovery.

The numerical pipeline is:

1. define a Gaussian-random-field prior for log permeability;
2. represent the field in whitened Karhunen–Loève (KLE) coordinates;
3. map log permeability to permeability with `k = exp(G)`;
4. solve the Darcy pressure equation with a cell-centered two-point flux approximation (TPFA);
5. evaluate the observation likelihood; and
6. explore the posterior with global, spatial-block, and likelihood-informed MCMC methods.

The project began as a study of numerical instability in overlapping-subdomain conditioning and grew into a broader comparison of posterior-correct multiscale sampling, likelihood-informed proposals, and a converged NUTS reference.

---

## Main scientific findings

### 1. Hidden null-space content can destabilize repeated conditioning

An arbitrary LU/pivot particular solution of a local hard-conditioning system can contain a component in the constraint null space. Repeated local updates can accumulate this hidden component and silently corrupt the chain.

SVD minimum-norm conditioning and stabilized LU remove that particular failure mode. Soft conditioning provides a residual-versus-stability trade-off, and deterministic red-black sweeps extend the diagnostic schedule across the coarse partition.

### 2. Stabilizing the linear solve does not make the original repeated-conditioning route exact

The later convergence experiments separate numerical stability from posterior correctness. Refreshing or standardizing the local conditioning may delay or obscure the observed drift, but it does not repair the target distribution. These paths remain useful mechanism probes, not production posterior samplers.

### 3. Precision block-Gibbs provides a posterior-correct spatial-block update

The corrected block sampler uses the proper full-rank prior

`C_tau = Phi diag(lambda) Phi.T + tau^2 I`

and evaluates its precision with a Woodbury identity. Each core block is proposed with pCN around its exact Gaussian full conditional. The Gaussian conditional and proposal terms cancel, so the Metropolis step uses a likelihood-only ratio while still targeting the correct posterior under the stated `C_tau` prior.

### 4. NUTS resolves the posterior-recovery ambiguity

The hand-written, adjoint-gradient NUTS implementation acts as a trusted reference rather than a speed claim. In the documented four-chain runs it reaches `R-hat <= 1.01`, with zero divergences at target acceptance 0.9.

| KLE modes | End R-hat | Minimum scalar ESS | Relative-permeability error floor | 90% coverage |
|---:|---:|---:|---:|---:|
| 64 | 1.0010 | 2187 | 0.491 | 0.953 |
| 90 | 1.0002 | 2041 | 0.516 | 0.968 |

The approximately `0.5` recovery floor is therefore a property of the smooth-prior, sparse-data posterior—not merely an unconverged-chain artifact.

### 5. A cheap adjoint basis captures the expensive pilot-informed subspace

On the 20x20, 24-mode squared-exponential benchmark, the M15 posterior-informed experiment builds an adjoint Gauss–Newton/Laplace basis for 528 forward-solve equivalents. Its leading five principal angles to the pilot basis are approximately `1.3–9.2` degrees, while the pilot costs about 160,000 solves to construct.

In the main provisional `sigma_obs = 0.02` run, single-step-size global pCN does not reach MPSRF 1.2 within 40,000 iterations, while the cheap-basis proposal reaches it in about 10,000 iterations. This is reported as a greater-than-4x lower-bound improvement—not as a completed reproduction of the external 10x target.

---

## Sampling methods and intended use

| Method | Implementation | Target status | Intended use |
|---|---|---|---|
| Global pCN / random walk | `mcmc.py`, `proposals.py` | Exact with the matching MH ratio | Baseline and sanity checks |
| Repeated local conditioning | `conditioned_sampler` | Diagnostic; known failure modes | Reproduce and isolate instability |
| Red-black local conditioning | `red_black_conditioned_sampler` | Diagnostic under the legacy construction | Scheduling and drift experiments |
| Precision block-Gibbs | `block_gibbs_sampler`, `conditioning/gaussian_block.py` | Exact for the KLE-plus-nugget prior | Posterior-correct spatial-block sampling |
| LIS / Laplace pCN | `lis.py` | Exact after Gaussian-reference MH correction | Accelerate likelihood-informed directions |
| NUTS | `nuts.py` | Exact up to numerical integration and validated adaptation | Converged reference posterior |

The distinction between diagnostic and correctness-bearing paths is deliberate. A numerically stable chain is not automatically sampling the intended posterior.

---

## Repository structure

```text
mcmc-multiscale-sampling/
├── src/mcmc_multiscale/
│   ├── conditioning/
│   │   ├── constraints.py        # local hard-conditioning systems
│   │   ├── particular.py        # SVD and LU particular solutions
│   │   ├── gaussian_block.py     # posterior-correct precision block updates
│   │   ├── nullspace.py          # null-space bases and projections
│   │   └── soft.py               # regularized conditioning maps
│   ├── forward/tpfa.py           # sparse TPFA Darcy pressure solver
│   ├── bayes.py                  # prior, likelihood, posterior, and misfit
│   ├── covariance.py             # exponential and squared-exponential kernels
│   ├── field.py, grid.py         # field reconstruction and grid conventions
│   ├── kle.py                    # truncated KLE construction
│   ├── mcmc.py, proposals.py     # generic MH engine and pCN/RW proposals
│   ├── sampler.py                # conditioned, red-black, and block samplers
│   ├── lis.py                    # adjoint GN/Laplace informed proposals
│   ├── nuts.py                   # multinomial NUTS reference sampler
│   └── diagnostics.py            # IAT, ESS, R-hat, MPSRF, and coverage
├── experiments/                  # exp01 through exp13 research harnesses
├── tests/                        # numerical gates and regression tests
├── docs/                         # derivations, method notes, and handoffs
├── outputs/                      # selected experiment tables and figures
└── app/streamlit_app.py          # interactive diagnostic dashboard
```

All field vectors use MATLAB-compatible Fortran ordering: zero-based cell `(row, col)` maps to `row + col * ny`.

---

## Experiment map

| Experiments | Purpose |
|---|---|
| `exp01`–`exp03` | Static conditioning, forward/Bayesian sanity, and Gaussian MCMC validation |
| `exp04`–`exp06` | Instability reproduction, stabilization methods, and red-black scheduling |
| `exp07`–`exp08c` | Posterior recovery, convergence diagnostics, and decision-grade comparisons |
| `exp09` | Compute-fair acceleration study across local and global methods |
| `exp10` | Likelihood-informed-subspace acceleration and rank ablation |
| `exp11` | Adjoint-gradient NUTS gold-standard reference |
| `exp12` | Conditioning diagnostics, including standardization and refresh-period probes |
| `exp13` | Cheap posterior-informed basis versus global pCN and pilot-informed proposals |

Selected technical write-ups:

- [`docs/conditioned_posterior_derivation.md`](docs/conditioned_posterior_derivation.md)
- [`docs/lis_method.md`](docs/lis_method.md)
- [`docs/nuts_reference.md`](docs/nuts_reference.md)
- [`outputs/exp13/exp13_table.md`](outputs/exp13/exp13_table.md)
- [`outputs/exp13/exp13_sigma_sweep_table.md`](outputs/exp13/exp13_sigma_sweep_table.md)

---

## Quickstart

Python 3.11 or newer is required.

```bash
python -m pip install -r requirements.txt

python -m pytest
python -m ruff check .
python -m black --check .
```

Run representative experiments:

```bash
python -m experiments.exp08_convergence_diagnostics
python -m experiments.exp10_lis_acceleration --help
python -m experiments.exp11_nuts_reference --help
python -m experiments.exp13_posterior_informed --help
```

Launch the dashboard:

```bash
python -m streamlit run app/streamlit_app.py
```

Some reference experiments are intentionally expensive. Use each script's `--help` output and reduced/test modes before starting a full multi-chain run.

---

## Verification philosophy

Posterior-sampling claims are backed by mechanism-specific gates, including:

- no-data invariance of the stated prior;
- linear-Gaussian recovery of the analytic posterior;
- fixed-seed determinism;
- adjoint gradients versus finite differences;
- detailed balance for Gaussian block proposals;
- exact reduction cases, such as rank-zero LIS to global pCN; and
- golden fixtures for behavior-preserving sampler changes.

Run the full suite with `python -m pytest` so the `experiments` package is importable consistently.

---

## Limitations and current roadmap

This is research software, not production reservoir-simulation software. It is single-machine and mostly sequential, uses synthetic data, assumes a specific TPFA boundary-value problem, and relies on dense covariance/KLE operations that will not scale directly to very large grids.

The reported M15 observation noise and sensor configuration are provisional pending confirmation against the external reference setup. NUTS is positioned as a trusted reference and is more expensive per iteration than the accelerator methods.

Planned work includes a boundary-conditioned multiscale prior with constraint-preserving coordinates and a unified benchmark in which every method is compared against a target-matched reference posterior.

---

## Skills demonstrated

Bayesian inverse problems and MCMC · Gaussian random fields and KLE · numerical linear algebra and null-space analysis · sparse PDE solvers and Darcy flow · adjoint differentiation · pCN, Gibbs, LIS, and NUTS · convergence diagnostics and uncertainty quantification · reproducible scientific experiments · Python packaging and testing.

---

*Author: Arya Kalantari · [github.com/Aria-Kalantari](https://github.com/Aria-Kalantari)*
