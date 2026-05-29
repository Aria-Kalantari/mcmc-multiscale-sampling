# Notes

## M1 Implementation

- The Python port uses zero-based indices but preserves MATLAB vector ordering:
  vectors are flattened from `(ny, nx)` arrays with `order="F"`.
- `Config.target_row` and `Config.target_col` are one-based coarse-subdomain
  identifiers, matching the MATLAB parameter convention.
- `top_eigenpairs` uses SciPy dense `eigh` with a top-eigenpair subset. This is
  deterministic and sign-agnostic tests avoid comparing raw eigenvector signs.
- `lu_pivot` was left unused in M1; it is implemented in M4 as the arbitrary
  pivot-column particular solution for instability reproduction.

## M2 Implementation

- TPFA solves `-div(k grad p) = f` on the cell-centered unit-square grid with
  harmonic internal-face permeability averages.
- # CONFIRM: The default pressure boundary convention is Dirichlet `p=1` on
  the left boundary, Dirichlet `p=0` on the right boundary, and homogeneous
  no-flow Neumann on the top and bottom. The default source is `f=0`.
- Dirichlet boundaries use half-cell transmissibilities. For constant
  permeability `k=1`, the numerical pressure reproduces the cell-center profile
  `p(x)=1-x`; the M2 sanity run measured max error `1.465494e-14` on a
  `48 x 48` grid.
- Manufactured-source verification uses a smooth positive permeability and an
  exact pressure satisfying the same left/right Dirichlet and top/bottom
  no-flow conditions. L2 errors for `16 x 16`, `32 x 32`, and `64 x 64` grids
  were `4.148090e-04`, `1.036890e-04`, and `2.592169e-05`, respectively,
  showing approximately second-order refinement behavior.
- # CONFIRM: The default pressure observation layout is a regular
  `n_obs_x x n_obs_y` grid of cell-centered sensors, defaulting to `8 x 8`,
  with Gaussian noise standard deviation `sigma_obs=0.01`.
- The M2 sanity run measured noiseless synthetic-truth misfit
  `0.000000e+00` at `theta_true`.

## M3 Implementation

- `proposals.py` implements symmetric Gaussian random-walk proposals and pCN
  proposals. pCN returns `prior_preserving=True` and zero proposal-density
  terms because the MH engine applies the prior correction.
- `metropolis_hastings` consumes `ProposalResult.prior_preserving`. For pCN or
  any prior-preserving proposal, it requires `log_prior_fn` and subtracts
  `log_prior_prop - log_prior_current` from the full-target ratio. This makes
  pCN safe when `log_density_fn` is a full posterior
  `log_likelihood + log_prior`.
- Using a prior-preserving proposal without `log_prior_fn` raises `ValueError`;
  the engine does not silently double-count the prior.
- Random-walk Gaussian sanity checks verify ordinary MH. The M3 sanity run
  sampled a 2D target with empirical mean `[1.00865736, -0.49784125]`,
  empirical covariance `[[1.01111593, 0.30073261], [0.30073261, 0.57914015]]`,
  and acceptance rate `0.5651` for target mean `[1.0, -0.5]` and covariance
  `[[1.0, 0.3], [0.3, 0.6]]`.
- pCN Gaussian checks verify prior-preserving MH behavior. With a standard
  normal target and `beta=1`, pCN acceptance was `1.0000`, empirical mean was
  `[0.0015592, 0.02359554]`, and empirical covariance was
  `[[0.98578901, 0.02788268], [0.02788268, 0.96100299]]`.
- The pCN full-posterior correction sanity run targeted posterior mean `0.8`
  and variance `0.36`; it measured empirical mean `0.8143`, variance `0.3480`,
  and acceptance rate `0.6939`.
- Random-walk acceptance responds to scale: in the M3 sanity run, `beta=0.2`
  gave acceptance `0.9373` and `beta=3.0` gave acceptance `0.3765`.
- One-step stationarity sanity from standard-normal initial states measured
  max mean shift `6.7805e-03` and max covariance shift `1.0228e-02`.
- No conditioned sampler, LU/pivot instability reproduction, soft constraints,
  or Streamlit code has been added yet.

## M4 Implementation

- Experiment numbering shift: the original SPEC names the instability
  experiment `exp02_reproduce_instability.py`, but this repository already uses
  `exp02` and `exp03` for M2/M3 sanity checks. The M4 instability experiment is
  therefore `experiments/exp04_reproduce_instability.py`.
- `lu_pivot` is intentionally arbitrary and unstable: it selects pivot columns
  by QR with column pivoting, solves the pivot block by LU factorization, sets
  non-pivot coordinates to zero, and does not remove hidden null-space content.
- The M4 sampler carries a current accepted local extended coefficient vector
  `theta_local_accepted` in `R^{N_ext}`. Each iteration proposes from it with
  pCN by default:
  `sqrt(1 - beta**2) * theta_local_accepted + beta * xi`.
- The conditioned candidate uses the Pereira/Aidan-style repeated-conditioning
  mechanism `theta_p + Z @ (Z.T @ theta_proposed)`. It deliberately does not
  use the shifted affine projection `theta_p + Z @ (Z.T @ (theta - theta_p))`
  on the instability path, because that projection can cancel the hidden
  null-space component of an arbitrary particular solution.
- Default `Config.beta` is `0.2`. Smaller beta can exaggerate LU drift because
  the retained null component decays slowly between repeated conditioning
  steps.
- In the single-subdomain M4 harness, only the core cells are copied back into
  the global field. The conditioning points are in the buffer, so the RHS `c`
  is effectively constant across iterations. This isolates the repeated local
  coefficient mechanism.
- The initial `theta_local_accepted` is a deterministic standard-normal draw.
  It does not need to be range-consistent with `G_current`, because each
  conditioning step discards the range component and replaces it with the
  current particular solution.
- M4 uses a likelihood-ratio accept/reject step on pressure data only:
  `log_alpha = log_like_candidate - log_like_current`. This is a debug harness,
  not the final mathematically polished sampler.
- Likelihood-only acceptance has no prior penalty on `||theta||`, so nothing
  pulls the null component back toward the prior. This is intentional for M4.
- Candidate and accepted/current norms are both tracked. MH rejection may
  suppress accepted-chain drift even when the generation step is unstable.
- The sampler yields/saves the actually generated candidate field and the
  accepted/current field every iteration because the process is path-dependent.
- The primary M4 instability flag is relative:
  `LU max candidate theta norm / SVD max candidate theta norm > 2.0`. Accepted
  ratios and absolute ratios to the expected Gaussian norm are also reported.
- The default observed run `python -m experiments.exp04_reproduce_instability`
  triggered the criterion: LU/SVD max candidate norm ratio `3.5386`, LU/SVD
  max accepted norm ratio `3.5306`, LU candidate max norm `43.8965`, SVD
  candidate max norm `12.4050`, expected Gaussian norm `6.7456`, LU acceptance
  `0.260`, SVD acceptance `0.263`, and accepted-chain LU drift was visible.
- No M5 soft constraints, c=0 experiment, Streamlit app, or Phase 2
  abstractions have been added.

## Deviations From SPEC.md

- The M1 experiment prints the required static-conditioning table only; it does
  not generate MATLAB-style figures.
- The structural M1 parity checks match the MATLAB reference: `rank(A)=Mb`,
  `NullDim=30`, machine-precision conditioned residuals, increasing `cond(A)`,
  and strongly reduced interface jumps. The most ill-conditioned case is
  sensitive in the last selected modes: Python reports `CondA=1.7681e+03` for
  `Mb=64` versus the SPEC table's rounded `1.393e+03`.
- M2 follows the prompt's intentional API divergences: `make_truth` returns a
  `TruthData` dataclass, and Bayes `misfit`/likelihood/posterior functions use
  explicit `Phi`, `lam`, `ForwardModel`, sensor, noise, and grid arguments.
