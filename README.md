# MCMC Multiscale Sampling with Overlapping Subdomains

This repository is a from-scratch Python research implementation. The current
state implements **M1 + M2 + M3 + M4 + M5 + M6**: the static local-conditioning core
ported from `reference/matlab/local_conditioning_project.m`, the TPFA forward
solver and basic Bayesian observation/misfit utilities, a generic
Metropolis-Hastings engine with random-walk and pCN proposals, and a headless
single-subdomain conditioned sampler that reproduces the LU/pivot
repeated-conditioning instability, compares the M5 stability fixes, and
provides a Streamlit dashboard for interactive viewing.

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
- LU/pivot arbitrary particular solution
- single-subdomain conditioned sampler diagnostics
- LU-vs-SVD instability reproduction experiment
- stabilized LU hard conditioning
- c=0 hard-conditioning diagnostic mode
- soft/proximal conditioning utilities
- stability-fix comparison experiment `exp05`
- Streamlit dashboard `app/streamlit_app.py`
- shared Streamlit-free app summary helpers

Not implemented yet: Phase 2 generalization.

## Run

```bash
python -m pytest
python -m experiments.exp01_static_conditioning
python -m experiments.exp02_forward_bayes_sanity
python -m experiments.exp03_mcmc_gaussian_sanity
python -m experiments.exp04_reproduce_instability
python -m experiments.exp05_stability_fixes
python -m streamlit run app/streamlit_app.py
python -m ruff check .
python -m black --check .
```
