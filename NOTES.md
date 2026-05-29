# Notes

## M1 Implementation

- The Python port uses zero-based indices but preserves MATLAB vector ordering:
  vectors are flattened from `(ny, nx)` arrays with `order="F"`.
- `Config.target_row` and `Config.target_col` are one-based coarse-subdomain
  identifiers, matching the MATLAB parameter convention.
- `top_eigenpairs` uses SciPy dense `eigh` with a top-eigenpair subset. This is
  deterministic and sign-agnostic tests avoid comparing raw eigenvector signs.
- `lu_pivot` is intentionally a Phase 2/M4 placeholder and is not used in M1.

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
