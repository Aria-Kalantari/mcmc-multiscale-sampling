# MCMC Multiscale Sampling with Overlapping Subdomains

This repository is a from-scratch Python research implementation. The current
state implements **M1 + M2 + M3**: the static local-conditioning core ported
from `reference/matlab/local_conditioning_project.m`, the TPFA forward solver
and basic Bayesian observation/misfit utilities, plus a generic
Metropolis-Hastings engine with random-walk and pCN proposals.

Implemented now:

- grid construction with MATLAB-compatible Fortran-order flattening
- exponential covariance matrices
- dense KLE eigenpairs
- log-field reconstruction helpers
- target overlapping subdomain construction
- hard conditioning constraints
- SVD minimum-norm particular solution and null-space basis
- static conditioning experiment `exp01`
- sparse TPFA Darcy pressure solver
- synthetic pressure observations
- prior, likelihood/misfit, and posterior helpers
- random-walk and pCN proposal kernels
- generic Metropolis-Hastings generator

Not implemented yet: conditioned sampler integration, LU instability
reproduction, diagnostics, soft constraints, Streamlit, or Phase 2
generalization.

## Run

```bash
python -m pytest
python -m experiments.exp01_static_conditioning
python -m experiments.exp02_forward_bayes_sanity
python -m experiments.exp03_mcmc_gaussian_sanity
python -m ruff check .
python -m black --check .
```
