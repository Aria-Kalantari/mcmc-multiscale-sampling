# Agent Notes

This repository is public research code. Do not add private transcripts,
credentials, private archives, generated `.mat`/`.npz` data, or large binary
outputs.

## Phase 1 / M1 Scope

M1 includes only the static local-conditioning port:

- `config.py`
- `grid.py`
- `covariance.py`
- `kle.py`
- `field.py`
- `subdomain.py`
- `conditioning/constraints.py`
- `conditioning/particular.py`
- `conditioning/nullspace.py`
- `conditioning/project.py`
- `experiments/exp01_static_conditioning.py`
- M1 pytest coverage

When working specifically on M1, do not implement TPFA, observations, Bayes,
MCMC, integrated sampler, diagnostics, soft constraints, Streamlit, or Phase 2
unless the task explicitly asks for a later milestone.

## Phase 2 / M2 Scope

M2 adds only:

- `forward/tpfa.py`
- `observations.py`
- `bayes.py`
- TPFA, observation, and Bayes tests
- optional `experiments/exp02_forward_bayes_sanity.py`

Do not implement MCMC, pCN proposals, conditioned sampler integration,
diagnostics, LU instability reproduction, soft constraints, Streamlit, or Phase
2 generalization unless a later task explicitly asks for them.

## Phase 3 / M3 Scope

M3 adds only:

- `proposals.py`
- `mcmc.py`
- proposal and MCMC tests
- optional `experiments/exp03_mcmc_gaussian_sanity.py`

Do not implement the integrated conditioned sampler, LU/pivot instability
reproduction, SVD fix experiments inside MCMC, soft constraints, Streamlit, or
Phase 2 abstractions unless a later task explicitly asks for them.

## Phase 4 / M4 Scope

M4 adds only:

- `diagnostics.py`
- `sampler.py`
- completed `conditioning/particular.py::lu_pivot`
- diagnostics, LU/pivot, mechanism, and sampler smoke tests
- `experiments/exp04_reproduce_instability.py`

The M4 sampler is a headless single-subdomain debug harness. Do not add M5 soft
constraints, a c=0 experiment, Streamlit, or Phase 2 abstractions unless a
later task explicitly asks for them.

## Phase 5 / M5 Scope

M5 adds only:

- stabilized LU hard conditioning
- SVD/minimum-norm as the stable hard-conditioning baseline
- c=0 hard-conditioning diagnostic mode
- soft/proximal conditioning utilities
- sampler support for M5 comparison modes
- `experiments/exp05_stability_fixes.py`

Do not add Streamlit, Phase 2 abstractions, private data, or full-sampler
architecture changes unless a later task explicitly asks for them.

## Phase 6 / M6 Scope

M6 adds only:

- `app/streamlit_app.py`
- Streamlit-free app/reporting helpers in `src/mcmc_multiscale/`
- lightweight helper tests
- documentation for launching the dashboard

The app is a viewer/controller for existing M1-M5 code. It must not add new
conditioning formulas, sampler logic, TPFA logic, Bayesian logic, Streamlit
imports outside `app/streamlit_app.py`, private data, or Phase 2 abstractions.

## Commands

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

The MATLAB file in `reference/matlab/local_conditioning_project.m` is a
read-only reference. Never call MATLAB at runtime.
