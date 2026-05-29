# Notes

## M1 Implementation

- The Python port uses zero-based indices but preserves MATLAB vector ordering:
  vectors are flattened from `(ny, nx)` arrays with `order="F"`.
- `Config.target_row` and `Config.target_col` are one-based coarse-subdomain
  identifiers, matching the MATLAB parameter convention.
- `top_eigenpairs` uses SciPy dense `eigh` with a top-eigenpair subset. This is
  deterministic and sign-agnostic tests avoid comparing raw eigenvector signs.
- `lu_pivot` is intentionally a Phase 2/M4 placeholder and is not used in M1.

## Deviations From SPEC.md

- The M1 experiment prints the required static-conditioning table only; it does
  not generate MATLAB-style figures.
- The structural M1 parity checks match the MATLAB reference: `rank(A)=Mb`,
  `NullDim=30`, machine-precision conditioned residuals, increasing `cond(A)`,
  and strongly reduced interface jumps. The most ill-conditioned case is
  sensitive in the last selected modes: Python reports `CondA=1.7681e+03` for
  `Mb=64` versus the SPEC table's rounded `1.393e+03`.
