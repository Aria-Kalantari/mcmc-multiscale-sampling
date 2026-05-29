# MCMC Multiscale Sampling with Overlapping Subdomains

This repository is a from-scratch Python research implementation. The current
state implements **Phase 1 / M1 only**: the static local-conditioning core ported
from `reference/matlab/local_conditioning_project.m`.

Implemented now:

- grid construction with MATLAB-compatible Fortran-order flattening
- exponential covariance matrices
- dense KLE eigenpairs
- log-field reconstruction helpers
- target overlapping subdomain construction
- hard conditioning constraints
- SVD minimum-norm particular solution and null-space basis
- static conditioning experiment `exp01`

Not implemented yet: TPFA, observations, Bayes, MCMC, sampler integration,
diagnostics, soft constraints, Streamlit, or Phase 2 generalization.

## Run

```bash
python -m pytest
python -m experiments.exp01_static_conditioning
```
