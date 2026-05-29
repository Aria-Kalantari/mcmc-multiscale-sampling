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

Do not implement TPFA, observations, Bayes, MCMC, integrated sampler,
diagnostics, soft constraints, Streamlit, or Phase 2 unless the task explicitly
asks for a later milestone.

## Commands

```bash
python -m pytest
python -m experiments.exp01_static_conditioning
```

The MATLAB file in `reference/matlab/local_conditioning_project.m` is a
read-only reference. Never call MATLAB at runtime.
