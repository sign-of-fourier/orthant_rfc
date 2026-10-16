# T7 · Engine vs shared-draw simulator for TURF search (design approved 2026-10-10)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

- **Why now.** The engine-moat argument depends on it (BACKLOG, first up after T6). In scenario
  scoring the shared-draw simulator was ~25× faster than the GPU engine (ORTHANT_PLAN S2), because
  every scenario needed ~1,500 orthants (6 SKUs × Q taste draws). TURF reach is one orthant per
  portfolio, 1 − P(buys none of S), with no taste integration: the engine's best case. The
  simulator's best case: one draw set scores every portfolio.
- **Hypothesis.** On T6's fitted model, TURF search driven by the GPU engine reaches the
  reference-best portfolio (no material regret) in less wall time than search driven by a
  best-practice shared-draw simulator at matched regret.
- **Kill criteria (user, 2026-10-10).**
  - Material regret: 0.5 reach pts vs the reference best (T6's flip threshold). 0.1 pts reported as
    secondary, not gating.
  - Speed bar: engine ≥ 5× faster wall time (including setup) than the simulator at the smallest N
    that achieves regret ≤ 0.5 pts, at the same k.
  - Where: primary = exhaustive k = 6 AND a majority of greedy k ∈ {6, …, 12}. Secondary (reported,
    not gated): fraction of near-tie pairs (within 0.5 pts of the best) correctly ordered at matched
    time.
- **Model check (done).** T6's fit has no household covariates: intercept-only marginals
  μ_j = Φ⁻¹(p̂_j) and one R (`t6.fit_ifm`), so reach(S) is one orthant per portfolio, shared by every
  household. (Had marginals varied by household: stop and ask.)
- **Simulator (best practice; a naive simulator invalidates the test).** Draw N simulated households
  once from the fitted model (latent z ~ N(μ, R), item accepted if z_j > 0). Store acceptance
  bit-sliced: for each item j a bitvector over households packed into 64-bit words, B[j] (N/64
  words). reach(S) = popcount(OR_{j∈S} B[j]) / N. Common draws for all portfolios; on GPU (torch
  int64 bitwise OR, popcount by byte lookup), portfolios batched. Equivalent to the per-household
  30-bit-mask form mean[(mask & S_mask) ≠ 0], 64 households per word op.
- **Scope.**
  - In: T6's fitted model (soft drinks, J = 30, window-1 IFM fit, recomputed deterministically; no
    new fit). Searches: exhaustive k = 3–6 (C(30, 6) = 594k at k = 6); greedy k = 3–12. Scorers on
    one T4: (a) GPU engine (`modal_gp_api.orthant_prob`, order 1, float64, as score_bench);
    (b) bitset simulator, N = 2^12 … 2^20; (c) GHK-QMC (T6's integrator), M = 2^10 … 2^14.
  - Reference: GHK at M = 2^16 re-scores the union of each method's top 100 per k (all scorers and
    settings). **Assumption: the true best is in that union.**
  - OUT: refitting or new data, the engine bias fix, J scaling beyond 30, pricing, posterior
    uncertainty.
- **Tests (minimal).**
  1. Exhaustive k = 3–6: per scorer and setting, regret of its best vs the reference best (0.5 and
     0.1 pts), wall time including setup.
  2. Greedy k = 3–12: regret of each scorer's greedy portfolio vs the reference best of the union at
     that k; wall time.
  3. Secondary: near-tie pairs (within 0.5 pts of the reference best) correctly ordered, per scorer at
     matched time.
- **Results (2026-10-10): FAIL. The bitset simulator is faster than the engine at matched regret at
  every k.** `analysis/T7/t7.py`, `out/results.json`, `run.log`. One T4, 302 s wall including
  container start (~$0.05); reference 1.0 s.

| search | k | engine regret / time | sim smallest N with regret ≤ 0.5 | sim regret / time at that N | engine ÷ sim time |
|---|---|---|---|---|---|
| exhaustive | 3 | 0.00 / 0.03 s | 4,096 | 0.40 / 0.02 s | 1.6× slower |
| exhaustive | 4 | 0.00 / 0.08 s | 4,096 | 0.43 / 0.01 s | 15× slower |
| exhaustive | 5 | 0.00 / 0.36 s | 4,096 | 0.00 / 0.02 s | 16× slower |
| exhaustive | 6 | 0.00 / 1.04 s | 4,096 | 0.00 / 0.09 s | 12× slower |
| greedy | 6–12 | 0.00–0.06 / 0.02–0.04 s | 4,096 | ≤ 0.27 / < 0.01 s | 9–10× slower |

  - Primary: exhaustive k = 6 engine 12× *slower* (bar: 5× faster); greedy k = 6–12 engine ≥ 5×
    faster at 0 of 7. FAIL.
  - Accuracy is not the issue: the engine had 0 regret except 0.06 pts at greedy k = 10. The
    simulator needs only N = 4,096 households to stay within 0.5 pts (max 0.43); at N = 16,384
    (regret ≤ 0.11, the 0.1-pt secondary nearly met) it is still ~3× faster than the engine at k = 6.
  - Secondary, near-tie ordering (greedy k = 7–12, 7–23 portfolios within 0.5 pts of the best):
    engine 0.72–0.90 of pairs ordered as the reference; sim N = 65,536 0.85–0.93 at ≤ the engine's
    time; GHK M = 1,024 0.98–1.00 at about the engine's time. The engine's bias costs it near-tie
    order; GHK-QMC dominates it on both speed and order.
  - GHK (T6's integrator) is slower than the simulator for exhaustive search (k = 6: 9.7 s at
    M = 1,024) but has 0 regret everywhere.
  - Reading: for TURF on a fitted intercept-only MVP, one orthant per portfolio is still more work
    than one OR + popcount over 4k–16k simulated households. The engine-moat argument does not hold
    for TURF search at J = 30, k ≤ 12. Not tested (OUT): J > 30, much larger k, household covariates
    (one orthant per household × portfolio would change the arithmetic for both).
