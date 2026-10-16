# T8 · GPR: kernel-parameterised error covariance (design approved 2026-10-10)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

Design, kill criteria and scope: BACKLOG.md, "GPR". Code: `analysis/T8/t8.py` (DGP, truths, HB-MNL +
RFC-S, decisions, summary), `analysis/T8/t8_gpu.py` (GHK fit with kernel Σ, one T4 per fit).

- **Results (2026-10-10): FAIL on the primary criterion (truth b).** Guard (a) PASS; (c) recorded
  PASS but carried by one replicate. `out/gate.json`, `out/full_{a,b,c}/results.json`,
  `out/summary.json`, `gate.log`, `run_{a,b,c}.log`. GPU: gate 249 s + main 11,642 T4-s
  (~3.3 T4-h, ~$2.00); HB-MNL local.
  - Recovery gate (truth b, N = 1,000, SEs on): PASS. Weights within 2 SE (max |z| 1.11, flavor);
    implied correlation max |error| 0.085 (bar 0.1); ℓ 0.248 (SE 0.042) vs 0.3.

| truth | kernel regret / flips | var-share regret / flips | HB-MNL + RFC-S regret / flips | kernel − var-share (paired SE) | verdict |
|---|---|---|---|---|---|
| (a) categorical | 0.00% / 0 | 0.00% / 0 | 8.03% / 29 | +0.00 (0.00) | guard PASS |
| (b) kernel | 0.51% / 8 | 0.56% / 15 | 4.50% / 37 | −0.05 (0.24) | **FAIL** (bar: > 2 SE) |
| (c) merge | 11.74% / 29 | 12.38% / 31 | 12.40% / 38 | −0.64 (0.64) | recorded PASS |

  - (b): the kernel beats HB-MNL + RFC-S by 3.99 pts (SE 0.35) and has fewer flips than variance
    shares (8 vs 15), but its regret is not lower than variance shares': the arms differ in 8 of 20
    replicates, 7 by −0.75 pts (kernel better) and 1 by +4.17 (variance shares better). Both MVP
    arms are near the truth's decisions; terciles capture most of the sweetness structure at J ≤ 6.
  - (a): the two MVP arms made identical decisions in all 20 replicates; w_rbf costs nothing.
  - (c): the arms differ in 1 replicate only (kernel −12.7 pts); the tie with SE = |mean| is that
    single replicate. All three arms are far off under merging (≈ 12% regret).
  - New-SKU (report only). Held-out correlation MAE, kernel − variance shares: (b) −0.088
    (SE 0.006), better by > 2 SE; (a) +0.030 (SE 0.008) and (c) +0.016 (SE 0.007), worse. Extension
    decision matches the truth in 19/19 (a), 20/20 (b), 20/20 (c) replicates for both arms: meets
    ≥ 16/20, does not separate the arms.
  - Reading: under a continuous-attribute truth the kernel recovers the new SKU's correlations
    clearly better, but at this J and these decisions (price, delist, extension at h = 2 pts) that
    doesn't move regret beyond variance shares with terciles. Not tested (OUT): larger J, decisions
    that hinge on a near-neighbour's substitution, finer continuous structure than terciles absorb.
