# M15 regime map: preconditioning across sigma_obs

20x20 SE l=0.16, 24 modes, sensors 8x8. MPSRF threshold 1.2. Baseline beta tuned per sigma (fewest iters). This maps the regime and pre-empts 'sigma_obs=0.02 was cherry-picked'.

| sigma_obs | informed rank | tuned beta | baseline accept | baseline conv (final MPSRF) | cheap conv | pilot conv | cheap-vs-pilot top-5 angles (deg) |
|---|---|---|---|---|---|---|---|
| 0.02 | 17 | 0.06 | 0.23 | plateau (2.88) | n/a | 8000 | [2.3, 2.9, 3.7, 6.2, 7.9] |
| 0.05 | 12 | 0.25 | 0.08 | plateau (1.24) | 1677 | 2621 | [2.2, 2.9, 3.8, 5.5, 20.6] |
| 0.1 | 8 | 0.25 | 0.32 | 5969 | 859 | 859 | [1.4, 2.7, 5.1, 6.3, 20.3] |

Reading: as sigma_obs shrinks the posterior grows more anisotropic; single-beta pCN crosses from converging (loose sigma) to plateauing within budget (tight sigma), while the 528-solve cheap adjoint basis keeps matching the pilot subspace to a few degrees in every regime -- so the basis result is not an artifact of the sigma_obs choice. (Budgets here are lighter than the main sigma=0.02 run; see exp13_table.md for the detailed sigma=0.02 result.)