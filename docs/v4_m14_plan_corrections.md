# M14 plan — corrections to apply before implementing

Paste-ready correction prompt for Claude Code. Applies to the
`feature/m14-conditioning-diagnostics` plan.

---

Your plan is approved in shape — the §0 framing is right, and moving probe (a) entirely out of
`src/` is the correct call. Four corrections before you write code. Two are load-bearing.

## 1. The bit-for-bit regression as specified is a tautology — fix it

`test_cond_refresh_period_default_reproduces_current_behaviour` compares the **default** against
**explicit `K=1`**. But the default *is* 1 — both take the identical new code path. That test only
proves the default value is 1. It proves nothing about whether the refactor changed behaviour, and
a caching bug that changes values *deterministically* would pass it.

Your RNG-ordering argument ("`build_c`/`_solve_local_conditioning` consume no RNG, so `K=1` always
recomputes → identical stream") is sound **reasoning**, but reasoning is not a test. Likewise,
`test_red_black_sampler_is_deterministic_for_fixed_seed` proves determinism of the *new* code, not
equivalence to the *old* code. The two together do **not** prove bit-for-bit invariance. Drop that
claim.

**What to do instead — order matters:**

1. **Before touching `sampler.py`**, on current `main`, capture a golden fixture: run
   `red_black_conditioned_sampler` on a small reduced config at a fixed seed for a few sweeps, and
   dump the accepted-field trajectory (or `G_accepted` at each sweep + the accept flags) to
   `tests/data/m14_red_black_golden.npz`. Commit the fixture.
2. **Then** make the `cond_refresh_period` change.
3. The regression test loads the fixture and asserts the new default path reproduces it exactly
   (`np.testing.assert_array_equal`, rtol=0/atol=0).

Keep your `K=1` vs `K=4` divergence test and the `K=0 → ValueError` test as-is; they're fine.

## 2. Wrong regime — `likelihood_only` does not produce the reversal

You picked `acceptance="likelihood_only"` for the `refresh` sweep. That regime **blows up** — NOTES
records rel-k `~1e13` and `max‖θ‖/expected ≈ 21` for hard-SVD red-black under likelihood-only. That
is the *runaway*, a different failure mode.

The **reversal** you want to probe (rel-k `0.4629 → 0.8855` while misfit keeps falling) is a
`global_field` phenomenon. Use:

```
acceptance="posterior", prior_mode="global_field", theta_p_method="svd"
```

Keep both CLI-overridable, but that must be the default for the `refresh` subcommand. As specified,
the experiment would have measured a runaway and labelled it a reversal.

## 3. Budget — the reversal is slow; use the reduced grid

Per NOTES, on the 48×48 default the reversal bottoms out at **~14,187 local updates/chain**
(≈887 sweeps at 16 subdomains/sweep) and only reaches 0.8855 by **32,000** (≈2000 sweeps). Five
values of `K` at that budget is many hours.

Use the **reduced 16×16 config** already used by exp08c and the block-Gibbs work (40 modes, 36
sensors, misfit floor 18) as the default for the `refresh` subcommand, with the full config
available behind a flag. Otherwise you will get a null result for *budget* reasons and be unable to
distinguish it from a real one.

State the per-`K` update budget in the output table so the null-vs-underpowered distinction is
visible.

## 4. Make the standardization test exact, not statistical

Your part (i) ("the standardized log-target is not affine-equivalent to the Gaussian posterior") is
vague and hard to assert; part (ii) (MH chain mean far from `μ_post`) is Monte Carlo and needs a
hand-tuned threshold.

There is an exact proof. Standardization is **invariant under affine rescaling of the field**: for
`a > 0`,

```
standardize(aG + b) = (aG + b - a·mean(G) - b) / (a·std(G))
                    = (G - mean(G)) / std(G)
                    = standardize(G)
```

In θ-coordinates `G = Φ√λ θ`, so `θ → aθ` gives `G → aG`. Therefore assert, to machine precision,
for several random `a > 0` and random `θ`:

```
Phi_mis(standardize(field_from_theta(Phi, lam, a*theta))) ==
Phi_mis(standardize(field_from_theta(Phi, lam,   theta)))
```

Deterministic, no tuning, no flakiness. **This is the gate.** It proves the target changed
*structurally*: the standardized likelihood is constant along the ray `θ → aθ`, so the standardized
posterior reverts to the **prior** in the amplitude direction — the data says nothing about field
scale. Name it something like
`test_standardization_makes_likelihood_scale_invariant_diagnostic_only`.

Keep the linear-Gaussian MH comparison as a secondary check if you want, but it is not the gate.

**Record the finding in `NOTES.md` in these terms:** *standardization does not fix the drift — it
makes the drift invisible to the likelihood.* It blinds the data to exactly the `‖θ‖` scale the
drift lives in. That is the scientific payoff of probe (a), and a stronger statement than "it
changes the target."

## 5. Minor

Pitfall citation: you cite §12 pitfalls 8/11. Pitfall 11 is the basis/prior one — irrelevant here.
You want **9** (likelihood-only acceptance is a debug harness, not a sampler).

---

Everything else in the plan stands. Re-post the revised plan before writing code.
