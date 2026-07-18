# v4 handoff — task specs for Claude Code

Working notes for handing M14–M17 to Claude Code. **`SPEC.md` is authoritative**; this file is
just the order of work and the paste-ready briefs. Read `SPEC.md` §0 `[V4]` first — it says what is
closed and what is actually open.

## Order of work

| day | milestone | what | why this order |
|---|---|---|---|
| 1 | **M14 (a,b)** | standardization probe; `cond_refresh_period` | trivial, experiment-only, clears Aidan's asks |
| 2 | **M14 (c)** | optimization-based frozen background | reuses `lis.gauss_newton_map`; Aidan's stated priority |
| 3–4 | **M15** | posterior-informed basis accelerator | **the star** — spectral, exact, mostly already in `lis.py` |
| 5 | **M16** | boundary-conditioned prior + NUTS-under-MSM | Aidan's construction; needs `B`/`N_B` |
| 6 | **M17** | unified benchmark | one table, target-matched |

Nothing here blocks on Aidan's replies. GLAM / `RE` / config-match only bite at M16–M17.

## Standing rules (every task)

- Feature branch → PR. `SPEC.md` and `outputs/` are gitignored; never commit figures.
- Land the imported module before the importer (`nuts.py` imports `lis.py`).
- Tests: `python -m pytest` (plain `pytest` misses the `experiments` package). Lint: `ruff check .`
  and `black --check .`.
- float64 only; explicit `numpy.random.Generator` threaded everywhere, never global `np.random.*`.
- Every `[V4]` numerical function ships with a test.
- **Anything claiming to sample a posterior ships all three gates** (no-data invariance,
  linear-Gaussian exactness, determinism). M14 claims nothing, so it ships diagnostic assertions
  instead — see below.
- Do not modify `conditioned_sampler` / `red_black_conditioned_sampler` / `block_gibbs_sampler`
  semantics. Additive only; existing defaults must reproduce current output bit-for-bit.

---

## Day 1 — M14 (a) + (b)

### Brief (paste to Claude Code)

> Read `SPEC.md` §0 `[V4]`, §8 (exp12), §10 (M14), §11 `[V4]`, §12 pitfalls 11–15.
>
> Implement M14(a) and M14(b) on branch `feature/m14-conditioning-diagnostics`. Both are
> **experiment-only diagnostics**. They must not alter any correctness-bearing sampler path, and
> existing defaults must reproduce current output bit-for-bit.
>
> **(a) Standardization probe.** Add an opt-in flag (default off) that, before the forward solve,
> replaces the log-perm field `G` by `(G − mean(G)) / std(G)`. This **changes the target** and is a
> diagnostic only — SPEC §12 forbids any recovery claim from it. Ship a test asserting it does
> **not** target `π(G|Y)` (e.g. on a tractable/linear-Gaussian case, show the stationary
> distribution differs from the analytic posterior, or that the induced map is not measure-
> preserving). Encode "diagnostic only" in the test name and docstring.
>
> **(b) `cond_refresh_period`.** Add an integer parameter to `red_black_conditioned_sampler` that
> rebuilds the conditioning RHS `c` (and `θ_p`) only every `K` sweeps instead of every sweep.
> Default must reproduce current behaviour bit-for-bit — ship a regression test for that.
>
> Then `experiments/exp12_conditioning_diagnostics.py` with subcommands for (a) and (b). For (b),
> sweep `K ∈ {1, 2, 4, 8, 16}` and plot rel-k trajectory per `K`; the hypothesis is that the
> reversal onset is delayed as `K` grows (mitigation, not a cure — SPEC §0 `[V4]` closure 1).
> Figures + a small table to `outputs/exp12/`.
>
> Run `python -m pytest`, `ruff check .`, `black --check .` before opening the PR.

## Day 2 — M14 (c)

> Implement M14(c) on `feature/m14-frozen-background`. Reuse `lis.gauss_newton_map` for the MAP
> background — do not write a new optimizer. Freeze all non-core cells at the MAP field, sample the
> core only.
>
> **This samples `π(core | background = ĝ, Y)`, a background-conditional, NOT the joint posterior.**
> Say so in the docstring, the experiment output, and `NOTES.md`. Do not compare its rel-k to the
> NUTS reference as if it were the joint posterior.
>
> Assertions (diagnostic, not exactness gates): `‖θ‖` stays bounded near `√n`, and no rel-k
> reversal. That is the point — it demonstrates freezing removes the run-away, supporting the
> §0 diagnosis that the drift lives in the exterior/pinned part.
>
> Add to `experiments/exp12_conditioning_diagnostics.py` as a third subcommand. Note in `NOTES.md`
> that the defensible framing (interior permeability known at the conditioning points) is
> **[CONFIRM]** with Aidan — SPEC §16.

## Days 3–4 — M15 (the star)

> Read `SPEC.md` §3.12, §10 (M15), §11 `[V4]`, §12 pitfalls 11–13. Branch
> `feature/m15-posterior-informed-basis`.
>
> **Extend `lis.py`; do not fork it.** `informed_subspace_from_samples` and `make_lis_proposal` are
> already the machinery. The deltas: (i) source samples from the **converged M13 NUTS reference**
> (not a pCN pilot — pitfall 13), (ii) **retain full rank** and rescale by posterior std (pitfall
> 12), (iii) use as the Gaussian reference `q*` with the Metropolis correction so the target is
> exactly preserved (pitfall 11 — the basis is a preconditioner, never a new prior; it must be
> frozen, never adapted mid-chain).
>
> **Gates first, in `tests/test_posterior_basis.py`, before the experiment:**
> 1. no-data ⇒ recovers `N(0, I)`;
> 2. linear-Gaussian ⇒ acceptance ≡ 1 and reproduces analytic `μ_post, Σ_post`;
> 3. determinism (fixed seed ⇒ identical chain);
> 4. basis is frozen (assert it does not adapt to samples mid-chain);
> 5. full-rank regression — truncating to dominant PCA directions degrades R̂ (encodes §3.12).
>
> Then `experiments/exp13_posterior_informed.py`: posterior-informed basis vs global pCN vs LIS vs
> NUTS at **equal forward solves**, reusing the compute-fair harness in `exp08c`
> (`_TruthReplayGenerator` matched starts, `_consume_states` / `_finish_chain`, `solve_offset`).
> Report R̂-vs-solves, ESS/1k-solves, rel-k, coverage. Quote any speedup at a **single stated
> convergence threshold** (SPEC §3.13 — his MPSRF ≤ 1.2 and our R̂ ≤ 1.01 are not the same bar).
>
> Target: reproduce Aidan's ~10×. A null result is a real result — report it honestly.

## Day 5 — M16

> Read `SPEC.md` §3.11 and `New_Conditioning_Idea.pdf` (project folder). Branch
> `feature/m16-boundary-conditioned-prior`.
>
> Offline: build `B` from shared-cell equalities `Φ_i(x_{i,k})θ_i − Φ_j(x_{j,k})θ_j = 0` across
> neighbouring subdomains; compute an orthonormal null basis `N_B` (SVD). Online: pCN in
> `z ~ N(0, I_q)`, `θ' = N_B z'`, stitch the global field, **likelihood-only acceptance** (exact —
> pCN is prior-preserving w.r.t. `N(0, I_q)`, so the prior cancels).
>
> **Blocking must be in `z`** (or constraint-preserving moves in `N(B)`). A naive `θ_ℓ` update
> violates `Bθ = 0` — SPEC §3.11, Aidan's §8.
>
> Gates: `B N_B = 0` and `Bθ' = 0` per proposal (~1e-16); no-data ⇒ `z ~ N(0, I_q)`;
> linear-Gaussian exactness **against the MSM target**; determinism; reduction (one block over all
> of `z` ⇒ plain global pCN in `z`).
>
> Then NUTS under the MSM prior: `U(z) = ½‖z‖² + Φ_mis(N_B z)`,
> `∇U(z) = z + N_Bᵀ (∂G/∂θ)ᵀ Jᵀr/σ_obs²`, reusing the validated adjoint for `Jᵀr`. Gate `∇U`
> against finite differences as usual.
>
> `experiments/exp14_boundary_conditioned_prior.py` reports the **MSM-vs-global posterior gap**
> (posterior-mean field difference, rel-k under each target, misfit floor). Remember: this targets a
> **different posterior** than M13 — that is the point, not a bug (pitfall 14).

## Day 6 — M17

> `experiments/exp15_unified_benchmark.py`: one table, every scheme vs the **target-matched** NUTS
> reference (global-prior schemes vs M13; MSM-prior schemes vs the M16 reference), metrics per
> SPEC §3.13, single convergence threshold. Record in `NOTES.md`, null results included.
>
> Blocked on Aidan for metric definitions (`RE`) and config match — see SPEC §16. Report ours
> unambiguously in the meantime.
