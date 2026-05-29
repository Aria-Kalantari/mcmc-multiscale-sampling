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

## Commands

```bash
python -m pytest
python -m experiments.exp01_static_conditioning
python -m experiments.exp02_forward_bayes_sanity
python -m ruff check .
python -m black --check .
```

The MATLAB file in `reference/matlab/local_conditioning_project.m` is a
read-only reference. Never call MATLAB at runtime.
