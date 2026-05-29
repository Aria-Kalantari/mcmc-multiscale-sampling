# SPEC — MCMC Multiscale Sampling with Overlapping Subdomains (Python, from scratch)

**Owner:** Arya Kalantari · **Advisor:** Dr. Luis Felipe Pereira · **Collaborator:** Aidan
**Status:** v0 spec for a from-scratch implementation · **Last updated:** 2026-05-28

---

## 1. What this project is (domain primer)

We are solving a **Bayesian inverse problem** for subsurface (Darcy) flow. The unknown is a spatially varying **log-permeability field** $G(x)=\log k(x)$, modeled as a Gaussian random field and represented by a truncated **Karhunen–Loève expansion (KLE)**. Given pressure measurements, we want the posterior distribution over $G$ (equivalently over the KLE coefficients $\theta$).

The posterior is explored with **Markov Chain Monte Carlo (MCMC)** (Metropolis–Hastings, preconditioned Crank–Nicolson proposals). MCMC mixes slowly in high dimensions, so the research idea is to **accelerate it** by updating the field **one overlapping subdomain at a time**, while **conditioning** each local update so it stays continuous with its frozen neighbors (no visible seam at subdomain interfaces).

**The central research problem this code must expose and fix.** When the local conditioning step is applied **repeatedly inside the MCMC loop**, the KLE coefficient norm $\lVert\theta\rVert$ **grows without bound** and the chain **drifts off the prior**. Because the Darcy forward map is smoothing / ill-posed, these oversized fields can still be *accepted*. The leading hypothesis (from the project notes) is that an **arbitrary particular solution** of the conditioning system carries a hidden null-space component that accumulates; the **minimum-norm (SVD / Moore–Penrose) particular solution** should remove the drift. This code must (a) reproduce the explosion and (b) test the fixes.

If you want the full narrative, read `aidan_session_extracted_notes.md` in this folder. The math is summarized in §3 below.

---

## 2. Goals and non-goals

### Goals
- A clean, tested Python library that implements the full pipeline: Gaussian field via KLE → Darcy forward solve (TPFA finite volume) → likelihood/posterior → Metropolis–Hastings/pCN MCMC → multiscale conditioned subdomain updates → diagnostics.
- Faithful **port** of the validated MATLAB conditioning core, proven by parity tests.
- A **reproduction** of the $\lVert\theta\rVert$ explosion, and an **experiment harness** that tests the candidate fixes (minimum-norm $\theta_p$, the $c=0$ diagnostic, soft constraints).
- A **Streamlit app** to watch sampling happen in real time: live field heatmaps, live metric traces, interactive controls.

### Non-goals (for now)
- HPC / cluster parallelism, MPI, GPU. (Design so 64 independent chains *could* later run in parallel, but single-machine is fine.)
- Matching Aidan/Borges's exact MRST code byte-for-byte. We target the *same scheme family* (cell-centered TPFA), not their file format. Their code, when it arrives, is a cross-check, not a dependency.
- Multi-phase flow, time dependence, 3D. Single-phase, steady, 2D.
- Automatic differentiation / gradient samplers (MALA/HMC). The whole point is a gradient-free accelerator. Leave a clean seam if we revisit this.

---

## 3. The mathematics (precise, implement against this)

### 3.1 Field model
Domain $\Omega=[0,1]^2$, cell-centered uniform grid $n_x\times n_y$ (default $48\times48$), cell centers $x_{ij}=((i-0.5)/n_x,\ (j-0.5)/n_y)$.

Gaussian log-permeability with exponential covariance:
$$
C(x,x') = \sigma^2 \exp\!\big(-\lVert x-x'\rVert_2 / \ell\big),\qquad \sigma=1.0,\ \ell=0.18.
$$
KLE from the eigenpairs $(\lambda_m,\varphi_m)$ of $C$ (descending $\lambda$):
$$
G(x)=\sum_{m=1}^{N}\sqrt{\lambda_m}\,\theta_m\,\varphi_m(x),\qquad \theta_m\sim\mathcal N(0,1)\ \text{i.i.d.},\qquad k(x)=\exp\big(G(x)\big).
$$
Global truncation $N=N_{\text{global}}=90$ modes. In matrix form on the grid: $G = \Phi\,(\sqrt{\lambda}\odot\theta)$, where $\Phi$ has the discretized eigenvectors as columns. (Matches MATLAB `topEigenpairs`, `GoldVec`.)

### 3.2 Forward model (Darcy, single-phase, steady) — TPFA finite volume **[NEW]**
$$
u=-k(x)\nabla p,\qquad \nabla\cdot u = f \ \text{ in }\Omega,
$$
with boundary conditions (**[CONFIRM]** default, consistent with the meeting's "slab, roughly $1-x$" pressure): Dirichlet $p=1$ on the left edge $x=0$, $p=0$ on the right edge $x=1$, no-flow (homogeneous Neumann) on top/bottom, $f=0$. Discretize with **cell-centered two-point flux approximation (TPFA)**: face transmissibility uses the **harmonic mean** of the two adjacent cell permeabilities; assemble a sparse SPD system $\mathbf{T}\,\mathbf p = \mathbf b$ and solve with `scipy.sparse.linalg` (direct `spsolve` is fine at $48^2$). See §6.5 for the API.

### 3.3 Bayesian inverse problem
- **Truth:** fix $\theta_{\text{true}}$ → $G_{\text{true}}$ → $k_{\text{true}}$ → solve Darcy → $p_{\text{true}}$ → sample at $N_{\text{obs}}$ sensor locations → add noise: $Y = \mathcal R(p_{\text{true}}) + \varepsilon,\ \varepsilon\sim\mathcal N(0,\sigma_{\text{obs}}^2 I)$. $\mathcal R$ is the restriction to sensor cells.
- **Prior:** $\theta\sim\mathcal N(0,I)$.
- **Likelihood / misfit:**
$$
L(Y\mid\theta)\propto \exp(-\Phi(\theta)),\qquad \Phi(\theta)=\frac{1}{2\sigma_{\text{obs}}^2}\big\lVert \mathcal R\big(p(\theta)\big)-Y\big\rVert_2^2 .
$$
- **Posterior:** $\pi(\theta\mid Y)\propto L(Y\mid\theta)\,\pi_0(\theta)$.

### 3.4 MCMC
**pCN proposal** (dimension-robust, prior-preserving — preferred):
$$
\theta' = \sqrt{1-\beta^2}\,\theta + \beta\,\xi,\qquad \xi\sim\mathcal N(0,I),\ \beta\in(0,1].
$$
Also provide a plain random-walk proposal $\theta'=\theta+\beta\xi$ for comparison. **Acceptance** (for pCN the symmetric prior terms cancel, so it reduces to the likelihood ratio; implement the general MH ratio and let the proposal declare whether it is prior-preserving):
$$
\alpha=\min\!\Big(1,\ \tfrac{L(Y\mid\theta')\,\pi_0(\theta')\,q(\theta\mid\theta')}{L(Y\mid\theta)\,\pi_0(\theta)\,q(\theta'\mid\theta)}\Big),\quad \text{accept if } u<\alpha,\ u\sim U(0,1).
$$
Work in log space ($\log\alpha$) for numerical stability. Detailed balance + ergodicity + aperiodicity ⇒ samples target the posterior.

### 3.5 Subdomains and local conditioning **[PORT for the static core]**
Partition the fine grid into a coarse $n_{cx}\times n_{cy}$ grid (default $4\times4$). Pick a **core** subdomain $\Omega_i$ (default row 2, col 2) and **enlarge** it by `overlapCells` (default 2) cells on each side to get the overlapping region $\widehat\Omega_i$. Cells in $\widehat\Omega_i\setminus\Omega_i$ are the **buffer**.

Build a **local KLE** on $\widehat\Omega_i$ (same covariance), keep $N_{\text{ext}}=N_c+M_b$ modes ($N_c=30$ base; $M_b$ = number of buffer conditioning points). Choose $M_b$ buffer points (the reference spreads them by angle around the core centroid — PORT `selectConditioningPoints` exactly).

**Constraint system** (match the frozen neighbor field at the conditioning points $y_j$):
$$
G_i(y_j)=c_j:=G_{\text{old}}(y_j)\ \Longrightarrow\ A\theta=c,\qquad A_{j\ell}=\sqrt{\lambda_\ell}\,\varphi_\ell(y_j),\quad A\in\mathbb R^{M_b\times N_{\text{ext}}}.
$$
$A$ is short-and-wide, full row rank $r=M_b$; $\dim\mathrm{Null}(A)=N_{\text{ext}}-M_b=N_c$ (preserved for all $M_b$).

**Decomposition:** $\theta=\theta_p+\theta_n$, with $A\theta_p=c$ and $\theta_n\in\mathrm{Null}(A)$.

**Particular solution — two constructions (this is the crux):**
- **(A) Minimum-norm / SVD / Moore–Penrose — STABLE [PORT].** With $A=U\Sigma V^\top$, $\theta_p = V_r\Sigma_r^{-1}U_r^\top c = A^\top(AA^\top)^{-1}c = A^+c$. This $\theta_p\perp\mathrm{Null}(A)$ → no hidden null-space component. (PORT `solveConditioningSVD`.)
- **(B) Arbitrary / LU on pivot columns — UNSTABLE [NEW, to reproduce the bug].** Row-reduce $A$, keep $r$ pivot columns to form a square $B$, LU-factor $B$, solve $LU\,\theta_p=c$ with non-pivot coordinates set to 0. This $\theta_p$ generally has a null-space component. This is what the conference paper / Aidan's code use and what we believe causes the drift.

**Null space + stochastic part:** orthonormal $Z$ with $\mathrm{Range}(Z)=\mathrm{Null}(A)$; draw $\eta\sim\mathcal N(0,I)$:
$$
\theta_n=ZZ^\top\eta,\qquad \theta=\theta_p+\theta_n\quad(\text{equivalently }\theta=\theta_p+Z\eta).
$$
**Affine projection view** (used when conditioning an existing proposal $\theta$):
$$
\Pi_S(\theta)=\theta_p+ZZ^\top(\theta-\theta_p),\qquad S=\{\theta:A\theta=c\}=\theta_p+\mathrm{Null}(A).
$$
**Stabilizer** (remove the null part of any particular solution): $\theta_p^\star=(I-ZZ^\top)\theta_p^{\text{any}}$.

Then reconstruct the local field $G_i=\Phi_{\text{ext}}(\sqrt\lambda\odot\theta)$ and copy the **core** cells back into the global field (overlap/buffer cells are discarded — PORT the `coreLocalIdx`/`coreGlobalIdx` logic).

### 3.6 Diagnostics for stability
- **Coefficient norm** $\lVert\theta^{(m)}\rVert$ vs iteration $m$. For $\theta\sim\mathcal N(0,I_n)$, $\mathbb E\lVert\theta\rVert\approx\sqrt n$; the chain is healthy if the norm stays near $\sqrt n$, **drifting** if it climbs. This is the primary stability metric.
- **Constraint residual** $\lVert A\theta-c\rVert/\max(1,\lVert c\rVert)$ (should be ~1e-16 in float64).
- **RMS interface jump** across the core boundary (PORT `interfaceJumpRMS`).
- **Acceptance rate, relative error** $\lVert k-k_{\text{true}}\rVert/\lVert k_{\text{true}}\rVert$ (and on pressure), **integrated autocorrelation time** (mixing).

### 3.7 Soft / regularized conditioning **[NEW]**
Replace the hard equality $A\theta=c$ with a penalty so the seam values may move slightly while $\lVert\theta\rVert$ is controlled:
$$
\min_\theta\ \tfrac12\lVert\theta\rVert^2 + \tfrac{\rho}{2}\lVert A\theta-c\rVert^2 ,
$$
or, equivalently, fold the constraint into the likelihood as a Gaussian "soft observation" with variance $1/\rho$. Sweep $\rho$; large $\rho$ → hard constraint.

---

## 4. Reference numbers to reproduce (from the validated MATLAB run)

These are the trusted outputs of the static core. Parity tests (§11) check against them. cond(A) is the effective condition number $s_1/s_r$.

| $M_b$ | $N_{\text{ext}}$ | rank(A) | nulldim | cond(A) | $\sigma_{\min}$ | RMS jump (conditioned) |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 31 | 1  | 30 | 1.000 | 0.947 | 0.991 |
| 2  | 32 | 2  | 30 | 1.287 | 0.828 | 0.724 |
| 4  | 34 | 4  | 30 | 1.881 | 0.690 | 0.566 |
| 8  | 38 | 8  | 30 | 3.559 | 0.480 | 0.507 |
| 16 | 46 | 16 | 30 | 8.119 | 0.305 | 0.284 |
| 32 | 62 | 32 | 30 | 40.87 | 0.083 | 0.255 |
| 64 | 94 | 64 | 30 | 1393  | 0.0034| 0.228 |

Trends to assert (not exact equality — RNG differs across MATLAB/NumPy): cond(A) increases monotonically with $M_b$; nulldim stays 30; conditioned constraint residual ~1e-16; conditioned RMS jump well below unconditioned (~72% reduction by $M_b=16$).

---

## 5. Tech stack and dependencies

- Python ≥ 3.11. NumPy, SciPy (sparse + linalg), Matplotlib. **Streamlit** for the app. **pytest** for tests. `scipy.io` to read any `.mat` files Aidan sends.
- Optional: `numba` only if profiling shows a hot loop; do not add it preemptively.
- Packaging: `pyproject.toml`, installable as `mcmc_multiscale`. Pin versions in `requirements.txt`.
- Reproducibility: a single `numpy.random.Generator` (`default_rng(seed)`) threaded through everything; never call the legacy global `np.random.*`. Default seed 7 (matches MATLAB `rng(7)`), configurable.

---

## 6. Module specifications

Repository layout:
```
.
├─ SPEC.md                     # this file
├─ NOTES.md                    # agent's running decision log (create it)
├─ pyproject.toml / requirements.txt
├─ src/mcmc_multiscale/
│  ├─ config.py        grid.py        covariance.py   kle.py        field.py
│  ├─ subdomain.py     observations.py               bayes.py
│  ├─ forward/tpfa.py
│  ├─ conditioning/{constraints.py, particular.py, nullspace.py, project.py, soft.py}
│  ├─ proposals.py     mcmc.py        sampler.py      diagnostics.py   io_utils.py
├─ app/streamlit_app.py
├─ experiments/exp0{1..5}_*.py
└─ tests/test_*.py
```

### 6.1 `config.py`
A frozen `@dataclass` `Config` holding all parameters with the defaults from §3/§4 (`nx, ny, n_coarse_x, n_coarse_y, target_row, target_col, overlap_cells, sigma, corr_length, n_global_modes, Nc, Mb, sigma_obs, beta, seed, theta_p_method ∈ {"svd","lu"}, conditioning ∈ {"hard","soft","zero"}, rho, update_scheme ∈ {"single","red_black"}`). One source of truth; everything takes a `Config`.

### 6.2 `grid.py` **[PORT]**
`cell_centered_grid(nx, ny) -> (x_vec, y_vec, X, Y, points)`. Mirror `makeCellCenteredGrid`. **Decide and document the flatten order** (MATLAB is column-major / Fortran order via `reshape(.,ny,nx)`); use `order="F"` consistently or convert. Parity test must confirm point ordering matches MATLAB `globalPts`.

### 6.3 `covariance.py` **[PORT]**
`exp_covariance(points, sigma, ell) -> C` = `sigma**2 * exp(-dist/ell)`, symmetrized, with `+1e-12*I` jitter. Mirror `covarianceMatrix`.

### 6.4 `kle.py` **[PORT]**
`top_eigenpairs(C, k) -> (Phi, lam)`: top-$k$ eigenpairs, descending, nonneg-clipped. Mirror `topEigenpairs` (use `scipy.linalg.eigh` or `scipy.sparse.linalg.eigsh`; eigenvector **sign is arbitrary** — parity tests must be sign-agnostic, e.g. compare $|\Phi^\top\Phi_{\text{ref}}|$ or compare reconstructed fields/covariance, not raw eigenvectors).

### 6.5 `field.py` **[PORT]**
`field_from_theta(Phi, lam, theta) -> G_vec` = `Phi @ (sqrt(lam) * theta)`; helpers to reshape to `(ny,nx)` and to `k = exp(G)`.

### 6.6 `subdomain.py` **[PORT]**
`make_subdomain(cfg) -> Subdomain` with `core_cols, core_rows, hat_cols, hat_rows, local_global_idx, core_local_idx, buffer_local_idx`. Mirror `makeSubdomain` (watch 1-based→0-based indexing and the flatten order). Also `interface_jump_rms(G, sub)` (PORT `interfaceJumpRMS`).

### 6.7 `forward/tpfa.py` **[NEW]**
```
class ForwardModel:
    def __init__(self, cfg): ...          # precompute geometry, BC structure
    def solve(self, k_field_2d) -> p_2d   # TPFA FV Darcy solve, returns pressure
```
Cell-centered TPFA, harmonic face transmissibilities, sparse SPD assembly, Dirichlet left/right + no-flow top/bottom by default (**[CONFIRM]**). Keep the BC and source `f` configurable. **Verify** (§11): (i) constant $k$ ⇒ linear pressure profile matching the analytic 1-D solution to ~1e-12; (ii) symmetry; (iii) refinement convergence on a manufactured solution.

### 6.8 `observations.py` **[NEW]**
Build synthetic truth and data: choose sensor cells (default: a regular grid of $N_{\text{obs}}$ pressure sensors; **[CONFIRM]** count/layout — the notes say "many pressure measurements"), `make_truth(cfg) -> (theta_true, Y, sensor_idx)`, and `restrict(p_2d, sensor_idx) -> vector` = $\mathcal R(p)$.

### 6.9 `bayes.py` **[NEW]**
`log_prior(theta)`, `misfit(theta, fwd, kle, Y, sensor_idx, sigma_obs) -> Phi`, `log_likelihood = -Phi`, `log_posterior`. Pure functions of `theta`; cache nothing global.

### 6.10 `conditioning/` (the heart)
- `constraints.py` **[PORT]**: `select_conditioning_points(local_pts, core_local_idx, buffer_local_idx, Mb)` (PORT exactly — angular ordering + `linspace` spread + the top-up loop), `build_A(PhiExt, sqrt_lam, cond_local_idx)` = `PhiExt[cond_idx,:] * sqrt_lam` (broadcast over columns), `build_c(G_old_vec, ...)`.
- `particular.py`: `svd_min_norm(A, c) -> (theta_p, Z, info)` **[PORT** of `solveConditioningSVD`, including `info.rankA/condEffective/minNonzeroSingular`]; `lu_pivot(A, c) -> (theta_p, info)` **[NEW** — RREF/pivot-column + LU, the arbitrary solution]. Selected by `cfg.theta_p_method`.
- `nullspace.py`: `null_basis(A) -> Z` (orthonormal, e.g. via SVD right vectors with zero singular values or `scipy.linalg.null_space`); `project_null(Z, eta) -> Z @ (Z.T @ eta)`.
- `project.py`: `affine_project(theta, theta_p, Z)` = $\Pi_S$; `stabilize(theta_p, Z)` = $(I-ZZ^\top)\theta_p$.
- `soft.py` **[NEW]**: `soft_condition(A, c, rho, eta, ...)` solving the regularized problem of §3.7.

### 6.11 `proposals.py` **[NEW]**
`pcn(theta, beta, rng)` and `random_walk(theta, beta, rng)`, each returning the proposal and a flag `prior_preserving: bool` plus any proposal-density terms needed for the MH ratio.

### 6.12 `mcmc.py` **[NEW]**
A clean Metropolis–Hastings engine: takes `log_posterior`, a proposal, an initial `theta`, `n_iter`, `rng`; yields per-iteration `MCMCState` (theta, logpost, accepted, plus hooks). Log-space acceptance. Must support a **callback per iteration** (the app and diagnostics subscribe to it) and run as a **generator** so the UI can stream.

### 6.13 `sampler.py` **[NEW]** — multiscale conditioned sampler (the integration)
This wires conditioning into MCMC. One step:
1. propose (pCN) a fresh local stochastic part for the chosen subdomain(s);
2. **condition** it (`cfg.conditioning`/`cfg.theta_p_method`): build `A,c`, get `theta_p`, draw/project `theta_n`, form `theta = theta_p + theta_n` (or $\Pi_S$);
3. rebuild $G_i$, copy core back into the global field, form candidate $Y$-field;
4. forward solve, compute $\Phi$, MH accept/reject;
5. **save the actually-generated field** each iteration (see pitfall §12.2);
6. repeat. Support `update_scheme`: **single** (one subdomain, optionally revisited several times while `theta_p` is solved **once**) and **red_black** (checkerboard — update all "black" subdomains with "white" interfaces frozen, then swap). Record the §3.6 diagnostics every iteration.

### 6.14 `diagnostics.py` **[NEW]**
`theta_norm`, running acceptance, `relative_error(k, k_true)`, `integrated_autocorr_time(chain)`, `interface_jump`. Return tidy arrays/records the app and experiment scripts can plot.

### 6.15 `io_utils.py`
`load_mat(path)` (scipy.io) for Aidan's `.mat` inputs — **[CONFIRM]** exact variable names; the notes warn that the Python side expects MATLAB variable names with **exact capitalization** or it errors. Save/load chains and fields (`.npz`), and config (`.json`).

---

## 7. The Streamlit app (`app/streamlit_app.py`) **[NEW]**

A dashboard to **watch sampling in real time**.

Layout:
- **Sidebar (controls):** sliders/inputs for `Mb`, `beta`, `sigma_obs`, `n_iter`, `seed`; selectboxes for `theta_p_method` (svd / lu), `conditioning` (hard / soft / zero), `update_scheme` (single / red_black), `rho`. **Start / Pause / Reset / Step** buttons.
- **Main panel, left column (fields):** live heatmap of the current global $G$ (or $k$) with the core/overlap rectangles and conditioning points overlaid (mirror the MATLAB visualization); side-by-side "current vs truth" and a residual/error map.
- **Main panel, right column (live metrics, time series):** $\lVert\theta^{(m)}\rVert$ with a reference line at $\sqrt n$ (the headline stability plot); acceptance rate; relative error; constraint residual; current `cond(A)`. Update at a throttled cadence (e.g. every K iterations) so the UI stays responsive.
- **Bottom:** a small results table and a "download chain (.npz)" button.

Implementation notes:
- Run the sampler as the §6.12 **generator**; advance it inside a loop guarded by Streamlit session state (`st.session_state`) and use `st.fragment`/periodic rerun (or a manual step loop with `time.sleep`) to stream updates. Keep all state in `st.session_state` (no globals).
- The app must add **zero numerical logic** — it only drives the library and renders. Anything it computes must already exist (and be tested) in `src/`.
- Performance: throttle redraws; downsample long traces; cap stored history.

A headless CLI (`experiments/`) must reproduce every app result without Streamlit — the app is a viewer, not the source of truth.

---

## 8. Experiments (`experiments/`) **[NEW]**

1. `exp01_static_conditioning.py` — reproduce §4 table (the port sanity check; no MCMC).
2. `exp02_reproduce_instability.py` — run the conditioned sampler with the **LU/pivot** $\theta_p$; show $\lVert\theta\rVert$ exploding while fields still get accepted. **This is the key deliverable result.**
3. `exp03_minnorm_fix.py` — same run with **SVD min-norm** $\theta_p$; show $\lVert\theta\rVert$ stays near $\sqrt n$; quantify error/acceptance/mixing vs exp02.
4. `exp04_c_zero.py` — set $c=0$; confirm stable-but-biased (notes: ~20% faster, ~15% higher error, true field not recovered).
5. `exp05_soft_constraints.py` — sweep $\rho$; seek stability without the $c=0$ bias.

Each writes figures + a small results table to `outputs/` and is runnable as `python -m experiments.expNN_...`.

---

## 9. (Phase 2) Generalization — design seams to leave now

v1 is the focused reproduce-and-fix above. Phase 2 generalizes the validated pieces into a reusable framework: pluggable `ForwardModel`, `Prior`, `Proposal`, and `ConditioningStrategy` interfaces; multiple subdomains / full coarse-grid sweep; many parallel chains; a config/CLI system; richer diagnostics (R-hat across chains, ESS). **Do not build Phase 2 now** — just keep the v1 interfaces small and dependency-light so generalization is a refactor, not a rewrite. Note Phase-2 hooks with `# PHASE2:` comments.

---

## 10. Milestones and acceptance criteria (build in this order)

**M1 — Port the static core.** grid, covariance, kle, field, subdomain, conditioning(constraints + svd_min_norm + nullspace). ✅ when `exp01` reproduces the §4 trends and `tests/test_conditioning_parity.py` passes (residual ~1e-16, nulldim 30, cond(A) monotone).

**M2 — Forward + Bayes.** tpfa, observations, bayes. ✅ when the TPFA verification tests pass (§6.7) and a synthetic-truth misfit is ~0 at $\theta_{\text{true}}$.

**M3 — MCMC.** proposals, mcmc engine. ✅ when sampling a known Gaussian target recovers its mean/covariance within MC error; acceptance responds sensibly to $\beta$.

**M4 — Integrated conditioned sampler.** sampler (single-subdomain), diagnostics, the "save generated field" mechanism, the **LU/pivot** path. ✅ when `exp02` **reproduces the $\lVert\theta\rVert$ explosion**.

**M5 — The fix.** SVD min-norm path + $c=0$ + soft constraints. ✅ when `exp03` shows a **stable norm** and `exp04`/`exp05` behave as described.

**M6 — Streamlit app.** ✅ when it streams live fields + metrics and the controls switch methods on the fly, reproducing M4/M5 visually.

**M7 — Red–black + polish.** ✅ when checkerboard updates run and the README documents how to run everything.

---

## 11. Verification & testing (a module is not done until its test passes)

- **Parity (vs MATLAB) — the most important tests.** `test_grid_parity`, `test_kle_parity` (sign-agnostic: compare reconstructed covariance/fields, not raw eigenvectors), `test_conditioning_parity` (A shape $M_b\times N_{\text{ext}}$, residual ~1e-16, nulldim = $N_c$, cond(A) trend from §4). Use the **same seed (7)** and the same RNG-free quantities where possible; for RNG-dependent quantities compare *statistics/trends*, not exact values.
- **Forward solver:** analytic constant-$k$ linear pressure (~1e-12), manufactured-solution convergence, SPD/symmetry of $\mathbf T$.
- **MCMC:** recovers a known Gaussian; detailed-balance sanity (a reversible chain leaves the target invariant).
- **Conditioning math:** $A\theta=c$ to ~1e-16 for both $\theta_p$ routes; $\theta_p^{\text{svd}}\perp\mathrm{Null}(A)$ (i.e. $Z^\top\theta_p\approx0$) while $\theta_p^{\text{lu}}$ generally is not; $\Pi_S$ idempotent.
- **Instability characterization:** a test asserting exp02's norm grows and exp03's stays bounded (this *encodes the research finding* as a regression test).
- **Determinism:** same seed ⇒ identical chain.
- Provide `make test` / a documented `pytest` command; target the §10 milestones.

---

## 12. Known pitfalls (the project already hit these)

1. **Precision.** float64 only. Assert the constraint residual is ~1e-16; a residual ~1e-6 means single precision leaked in (this exact bug happened in MATLAB). Never rescale $\theta_p$ after solving — it breaks $A\theta_p=c$.
2. **Save the generated field.** The $\theta$ used to *build* a field is not the $\theta$ stored in the chain (the process is path-dependent), so you cannot reconstruct fields after the fact. **Persist each generated field as you run.**
3. **Don't form inverses.** For min-norm, solve via the SVD (or solve $AA^\top\lambda=c$ by Cholesky/QR), not by explicitly forming $(AA^\top)^{-1}$.
4. **Indexing & flatten order.** MATLAB is 1-based and column-major; `reshape(vec, ny, nx)`. Pick one convention (recommend Fortran order to match) and assert it in a parity test, or subtle off-by-one/transpose bugs will silently corrupt fields.
5. **Eigenvector signs/degeneracy** are arbitrary and differ MATLAB↔NumPy — never compare raw eigenvectors; compare invariant quantities.
6. **cond(A) blows up with $M_b$** (→1393 at $M_b=64$): more conditioning points reduce the seam but worsen conditioning. Favor moderate $M_b$ (≈16–32) and pair extra points with the min-norm solve or soft constraints.
7. **MATLAB I/O capitalization.** If/when loading Aidan's `.mat` files, variable names must match exactly (capitalization included) or the load fails. **[CONFIRM]** names on arrival.
8. **`cfg.conditioning="zero"` is a diagnostic, not a fix** — it is stable but biased (forces seams flat, wrong field).

---

## 13. Coding conventions
- Type hints everywhere; small pure functions; `Config` is the only global state.
- Docstrings that state the math (cross-reference §3) and units.
- No hidden randomness — pass `rng` explicitly.
- Keep `src/` import-light (NumPy/SciPy only); Matplotlib/Streamlit live in `app/`+`experiments/`.
- Format with `ruff`/`black`; lint clean. Every `[NEW]` numerical function ships with a test.

---

## 14. Reference materials in this folder
- `reference/matlab/local_conditioning_project.m` — **the validated static-conditioning code to PORT** (residual→machine precision, the §4 table). Authoritative for grid/covariance/KLE/subdomain/conditioning.
- `aidan_session_extracted_notes.md` — full notes from the research meeting: the instability narrative, the two $\theta_p$ constructions, the fixes, the MCMC embedding, the open questions. **Read for context.**
- `NLA-Project-Report.pdf` — the report the MATLAB code came from (method write-up + §4 results).
- `Conf25-V07.pdf` (Pereira) — the conditioning method; **Algorithm 1 = the LU/pivot-column construction** (the arbitrary $\theta_p$ to reproduce). Null-space projector $P=QQ^\top$.
- `Published-multiscale-sampling.pdf` (Ali, Al-Mamun, Pereira, Rahunanthan, JCP 2024) — the base multiscale sampler (no conditioning): MCMC algorithm, likelihood, diagnostics. Reference for the MCMC + forward-model conventions.
- `cond_by_projection_notes_stable.pdf` — the minimum-norm / affine-projection notes: $\theta_p^\star=A^\top(AA^\top)^{-1}c$, $\theta=\theta_p^\star+Z\eta$, the $(I-ZZ^\top)$ stabilizer, and the "don't form inverses / don't rescale" warnings.

---

## 15. Glossary
- **KLE** — Karhunen–Loève expansion: spectral representation of the Gaussian field from covariance eigenpairs.
- **$\theta$** — KLE coefficient vector (the MCMC state). Prior $\mathcal N(0,I)$.
- **$G,\ k$** — log-permeability $G=\log k$; permeability $k=\exp G$.
- **Core / overlap / buffer** — the non-overlapping subdomain / its enlarged region / the extra ring of cells.
- **$M_b,\ N_c,\ N_{\text{ext}}$** — buffer conditioning points / base local stochastic dimension (30) / local modes kept $=N_c+M_b$.
- **$A\theta=c$** — conditioning system pinning the local field to neighbor values at buffer points.
- **$\theta_p,\ \theta_n,\ Z$** — particular solution / null-space part / orthonormal null-space basis.
- **pCN** — preconditioned Crank–Nicolson proposal (prior-preserving).
- **TPFA** — two-point flux approximation (cell-centered finite-volume Darcy discretization).
- **Drift / explosion** — $\lVert\theta\rVert$ growing far past $\sqrt n$; the bug to fix.

---

## 16. Open questions to resolve with Aidan / Dr. Pereira (don't block on these)
- Exact **boundary conditions / source** for the Darcy solve, and the **sensor layout / count** and $\sigma_{\text{obs}}$ (and whether any interior permeability points are measured, $M_k>0$).
- The intended **soft-constraint** form (penalty vs Gaussian soft-observation) and how the acceptance ratio changes.
- Whether the drift is believed to come from the **arbitrary $\theta_p$** or from **re-conditioning** itself; whether $\theta_p$ is recomputed every visit or solved once.
- The agreed **metric for "fixed"** (norm stability threshold, acceptance, relative error, IAT) and the target deliverable.
- `.mat` **variable names/format** for the code Aidan will send (so `io_utils.load_mat` matches).
