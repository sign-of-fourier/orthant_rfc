# T6 · Correlated-reach TURF on public basket data

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

- **Hypothesis.** Reach computed as 1 − orthant under an estimated correlation (MVP) predicts
  held-out portfolio reach better than the independence formula 1 − ∏(1 − p_j), and changes which
  portfolio greedy selects, toward the one with higher held-out reach.
- **Kill criteria and design fixes (user, 2026-10-09).**
  - **Design fixes.**
    - k = 1 is a sanity check only (both methods must give identical reach). Prediction is scored
      on k = 2–6.
    - The independence formula uses MVP's own fitted marginals, so only the correlation differs
      between methods.
    - All held-out reach numbers carry a household-bootstrap SE.
    - The category cluster (commodity + J item list) is fixed below before running. No category
      changes after results.
  - **Control (GATE, run first; failure = stop and report, unlike T5's G0).** Simulated independent
    truth with the same marginals, households and trips. Pass: zero counted flips vs the
    independence formula, AND MVP's added reach error ≤ max(0.2 pts, 5% relative). Report max
    |ρ̂| but don't gate on it.
  - **Prediction.** MVP's mean absolute held-out reach error ≥ 20% lower than the independence
    formula's, pooled over k = 2–6; also holds at k = 5–6; and MVP not worse than the independence
    formula by > 2 SE at any k.
  - **Decision.** A flip counts if the held-out reach difference ≥ max(0.5 pts, 2 SE). Pass: ≥ 2 of
    k ∈ {3, 4, 5, 6} have counted flips; MVP wins the majority of counted flips; no counted flip
    loses. Record the first k where the greedy picks diverge (paths are nested; flips aren't
    independent).
- **Unit: household-level reach (design change, user 2026-10-09).** Originally trip-level, as T3. At
  trip-level product rates (~1%) correlation moves pairwise reach by ~0.1 pts, so the decision
  test would be near-null by construction; at household rates (~5–20%) it moves it by ~5 pts.
  - Observation = household × window binary vector over the J items (bought the item at least once
    in the window).
  - Time split: two equal-length windows, weeks 1–27 and 28–53. Households active in both: 2,338.
  - Fit marginals + Σ (IFM) on window 1; score observed window-2 reach = share of households buying
    ≥ 1 item of S.
  - Bootstrap over households. Control = simulated independent household vectors with the window-1
    marginals and the same household count.
  - Estimator: IFM (full ML stays in the backlog).
- **Category cluster (fixed 2026-10-09, before any fit).** SOFT DRINKS (`product_category`), top 30
  products by window-1 household rate (ranked on window 1 only, so the item choice doesn't see the
  scoring window). Pre-fit check: window-1 rates 4.4–22.2%, 5 of 30 below 5% (window 2: 11 of 30), so
  not "most below 5%": J = 30 kept (rule: drop to top 20 if most fall below ~5%; bread only if that
  check fails). Manufacturers: 1208 (10), 103 (9), 2224 (8), 69 private label (3). "Same brand" =
  same `manufacturer_id` (dunnhumby `brand` is only National / Private); variants are formats (12-pack
  cans, 2 L, 20 oz single, multipack). Category reach (any soft drink) 87% / 86% per window.
  - Product IDs: 5569230, 1053690, 8090521, 8090537, 844165, 5569471, 1092026, 1085604, 5569845,
    1110572, 916381, 879755, 1132770, 6534480, 13511722, 8090509, 893501, 6534035, 8090532, 1138189,
    5569374, 6534077, 868764, 1037894, 1076875, 882441, 1107553, 1074524, 947798, 991951.
  - Caveat for the write-up: formats are partly occasion-driven (a 12-pack for the home, a 20 oz single
    on the go), so same-manufacturer correlation mixes taste similarity with shopping-occasion mix.
- **Scope.**
  - In: dunnhumby Complete Journey; one category cluster with natural near-duplicates (same
    brand, different flavor or pack), J = 15–30. Household × window vectors (above); reuse the
    T3 / rho_test loader and IFM machinery ([design](../../analysis/T3/DESIGN.md)). Reach estimators: (a) independent
    product (MVP's fitted marginals), (b) MVP orthant, (c) the empirical / deterministic baseline
    only if trivial.
  - OUT: MaxDiff data, engine-vs-simulator timing ([backlog](../../BACKLOG.md), first up), posterior
    uncertainty, B&B, pricing.
- **Tests (minimal), in order.**
  0. Control gate: simulated independent household vectors (window-1 marginals, same household
     count). Stop if it fails.
  1. Prediction: absolute error vs observed window-2 household reach for random candidate sets, k = 2–6
     (k = 1 sanity check: identical reach for both).
  2. Decision: greedy portfolio per estimator at k = 3–6 (selected on the window-1 fit), scored by
     window-2 household reach. Report counted
     flips, reach difference with SE, and the first k where the greedy paths diverge.
- **Assumption (state in the write-up).** Removing an item does not change the remaining items'
  utilities. That is standard for TURF; counterfactual delisting isn't validated here.
- **Control gate result (2026-10-10): FAIL, waived by the user.** SciPy integrator, B = 200, 200 sets
  per k (`analysis/T6/out/control.json`). Counted flips 1 (k = 5: MVP's portfolio −1.63 pts window-2
  reach, SE 0.66; k = 6 differs by −0.60, SE 0.63, not counted); added reach error +0.007 pts (limit
  0.2), pass; k = 1 identical; max |ρ̂| 0.905 (one pair with 0 co-purchases at the −0.99 bound,
  pulled to −0.905 by the PSD projection). User: a fluke; an independent truth can only show MVP
  recovering the identity matrix, so the gate tests a tautology; proceed to the real stage
  (`t6.py real --control-waived`).
- **Integrator.** MVP reach by GHK (scrambled Sobol, M = 4,096) on one Modal T4, checked against
  SciPy on the main fit's sets (max diff 0.0014 pts in the wiring check; tolerance 0.01).
- **Results (2026-10-10): PASS (prediction and decision).** Real stage, B = 200, 200 random sets
  per k, `analysis/T6/out/real_gpu.json`, `real.log`. 146 s wall; GPU 192k orthants in 8.8 T4-s
  (~$0.01); GPU vs SciPy max diff 0.003 pts. IFM ρ: median +0.22, range −0.23..+0.71; PSD projection
  moved 0.005. k = 1 sanity: identical.

| k | independence: mean abs reach error, pts (SE) | MVP (SE) | MVP − indep. SE |
|---|---|---|---|
| 1 | 1.81 (0.14) | 1.81 (0.14) | — (sanity) |
| 2 | 2.67 (0.20) | 2.40 (0.18) | 0.06 |
| 3 | 3.06 (0.25) | 2.27 (0.18) | 0.19 |
| 4 | 4.45 (0.42) | 2.49 (0.23) | 0.30 |
| 5 | 5.20 (0.55) | 2.24 (0.20) | 0.49 |
| 6 | 7.20 (0.64) | 2.16 (0.29) | 0.49 |
| pooled 2–6 | 4.52 | 2.31 | 0.30 |

  Prediction: MVP's error 49% lower pooled (rule ≥ 20%), 57% at k = 5 and 70% at k = 6; MVP not worse
  at any k. PASS. The independence error grows with k (it double-counts overlapping buyers); MVP's
  stays flat at ~2.2–2.5 pts (window-to-window drift).

| k | MVP greedy portfolio | independence greedy portfolio | window-2 reach MVP − indep., pts (SE) | counted flip | winner |
|---|---|---|---|---|---|
| 3 | 5569230, 844165, 1053690 | 5569230, 1053690, 8090521 | +4.36 (0.64) | yes | MVP |
| 4 | + 8090521 | + 8090537 | +3.81 (0.59) | yes | MVP |
| 5 | + 5569471 | + 844165 | +1.75 (0.52) | yes | MVP |
| 6 | + 1085604 | + 5569471 | +1.15 (0.46) | yes | MVP |

  Decision: counted flips at k = 3, 4, 5, 6; MVP wins all 4; none lost. PASS. Greedy paths first
  diverge at k = 2 (independence takes 1053690 second, MVP 844165). Reach 39.5 vs 35.2% at k = 3.
  Mechanism: independence stacks near-duplicates from one manufacturer (8090521 and 8090537, both
  manufacturer 103 12-pack cans) whose buyers overlap; MVP sees the correlation and takes a
  different format (844165, manufacturer 103 2 L) that reaches new households. Caveat (as noted
  above): formats are partly occasion-driven, so "near-duplicate" here mixes taste and occasion.

- **Follow-up (2026-10-17): [T16](../T16/TEST.md) FAIL.** Against per-household independence (the practitioner TURF incumbent)
  MVP gains 0.13 pts (95% CI [−0.06, +0.32]) and picks the same portfolios; HB per-household independence closes 94%
  of this test's gap. The PASS above holds against pooled independence only.
