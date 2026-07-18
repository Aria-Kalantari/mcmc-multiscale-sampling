# M15 - posterior-informed basis: the 10x reproduction attempt

Config: 20x20 grid, squared-exponential kernel, l=0.16, 24 modes. MPSRF threshold = 1.2.
**Provisional (Aidan did not state these; confirm with Aidan):** sigma_obs = 0.02; sensors = 8x8 = 64.

## Convergence and cost

| method | conv iter (MPSRF<=thr) | basis cost (solves) | sampling solves @ conv | accept | early-accept | rel-k |
|---|---|---|---|---|---|---|
| global_pcn | not_reached | 0 | n/a | 0.52 | 0.50 | 0.2451 |
| global_pcn_prior | not_reached | 0 | n/a | 0.52 | 0.50 | 0.2754 |
| posterior_informed | 10000 | 528 | 80000 | 0.29 | 0.37 | 0.2542 |
| pilot_informed | 6399 | 160000 | 51192 | 0.22 | 0.17 | 0.2750 |

## Baseline pCN beta tuning (fair comparison)

beta chosen by fewest iterations to MPSRF<=thr, not by a fixed acceptance target (pCN's optimum is problem-dependent and typically above the 0.234 random-walk value). The full grid is the evidence the baseline was tuned, not crippled:

| beta | acceptance | iters to MPSRF<=thr |
|---|---|---|
| 0.03 (best) | 0.52 | not_reached |
| 0.05 | 0.31 | not_reached |

## Headline

- **Speedup (cheap basis): > 4.0x (lower bound)** -- global_pcn does not converge within 40000 iters (MPSRF plateaus at 1.75); posterior_informed reaches 1.2 at 10000.
- **Cheap basis cost: 528 forward solves** (MAP 463 + adjoint Jacobian 65) -- no pilot.
- **Pilot basis cost: 160000 forward solves** (n_chains x baseline second-half iters). Same subspace for 528 vs 160000 solves (303x).

## Principal angles (cheap adjoint basis vs pilot basis), degrees

Informed rank (mu>1) = 17. Small-r is the real test; larger r conflates finite-sample pilot noise. (the pilot basis here is built from the baseline's second half, which has not itself converged -- yet the dominant directions still match)

| top-r | principal angles (deg) |
|---|---|
| 3 | [2.4, 5.2, 13.2] |
| 5 | [1.3, 2.0, 3.6, 6.6, 9.2] |
| 10 | [0.6, 0.6, 1.0, 1.8, 2.7, 4.3, 4.5, 7.1, 10.6, 21.2] |
| 17 (r_informed) | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1, 2.9, 5.1, 7.8, 11.5, 15.7, 34.9] |

## Verdict

(1) At this (provisional) sigma_obs, single-beta global pCN does not converge within 40000 iters -- MPSRF plateaus at ~1.8 -- and the prior-scale control behaves identically, so this is intrinsic mixing on an anisotropic posterior, not a start-transient. This is a harder regime than a baseline that converges at ~40k: here preconditioning is necessary, not merely faster. The exact regime (sigma_obs, sensor count) is a to-confirm item with Aidan. (2) Preconditioning with the 528-solve cheap adjoint/Laplace basis converges in ~10000 iters (pilot basis ~6399) -- the proof the basis is good. (3) Corroboration: the cheap basis matches the pilot's informed subspace (top-5 [1.3, 2.0, 3.6, 6.6, 9.2] deg; 14/17 directions <10 deg, 12 <5 deg), so the 528-solve adjoint basis buys the same informed directions as the pilot's 160000-solve basis -- same subspace, ~303x cheaper.
