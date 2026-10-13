# Plan: where multivariate probit beats incumbent choice methods

Status 2026-10-10: T5 PASS, T6 PASS, T7 FAIL (engine vs simulator for TURF), T8 FAIL (GPR kernel Σ,
primary criterion), T9 FAIL (MACML ties GHK: both at zero regret at N = 600). 2026-10-11: T10 Part A
inconclusive (as labelled); T10 Part B PAUSED (user, 2026-10-10) at the step-0 stop-and-ask; side
study OVUS vs the engine done (T10, below). T11 FAIL (joint chance constraint = independent product on every
decision). T13 KILL (6 of 20: Joint = Independent on all 18 decisions). 2026-10-13: T14 PASS (A: CS-F 207 vs MVP-IFM 50 flips at k = 20/30; B, 24-item baking cluster: holdout backs MVP 3, CS-F 0). Model catalog: [MODELS.md](MODELS.md). Backlog: BACKLOG.md. T1–T3 completed, T4 paused. Everything else is in
[BACKLOG.md](BACKLOG.md). Kill criteria are set by the user; nothing runs until they are filled in. Scope is
strict: if a test suggests expanding it, stop and ask (no new arms, truths or metrics).
The earlier product and working notes moved to `PRODUCT_NOTES.md`. They cover
machine limits, the orthant backend, and style rules, and still apply.

## Context

- **Engine.** A GPU MVN orthant engine: a deterministic approximation, about
  70–150× faster than QMC Sobol for n = 4 to about 128. It is deployed on a
  model backend API. For small tasks, the same code ships as a local `.so`
  (`multivariate_probit.orthant`; see `rfc.orthant_cdf`).
- **Library.** `multivariate_probit` 0.2.3: sklearn-style, cross-fit IFM.
- **Aim.** Find where multivariate probit (MVP) beats the methods incumbent
  platforms use (Sawtooth: HB-MNL, RFC simulation, MBC combinatorial coding),
  and where the engine is what makes MVP feasible.
- **Settled, not re-tested.** The engine envelope: smoothness, accuracy at
  negative correlation, and error on small probabilities. The engine is used
  as-is. No accuracy sweeps of the engine.

## Three questions (kept separate in every write-up)

- **Q1. Model.** Is MVP better than, or as good as, the incumbent's model (RFC
  in CBC, coded subsets in MBC)? Parity counts if MVP is explainable or faster.
- **Q2. Estimator.** Is full MLE better than IFM? IFM does not apply to CBC
  single-choice data. There, the probit is fitted by full MLE or a substitute
  (GHK, Gibbs data augmentation, MACML).
- **Q3. Speed.** What does the engine buy in time-to-decision against scalable
  incumbent methods?

## Terminology

- **Linked set:** items whose choices are actually dependent in the data. A
  property of the data.
- **Coded subset:** items the analyst codes combinatorially (Sawtooth MBC),
  capped at about 5 binary items (≤ ~36 combinations recommended; ≥ ~75 too
  sparse). A modeling choice.
- **Decision layer:** the business decision a model output feeds (price, line
  extension, bundle, assortment, a cost-weighted metric). All kill criteria
  are decision-layer tradeoffs set by the user. Designs propose candidate
  metrics and leave the cutoffs blank.

## The five items under test

1. **Shelf CBC with 20+ SKUs.** Logit is used because there is no scalable
   probit; IIA is tolerated under that constraint, not preferred. The binding
   constraints are computation and identifiability. An attribute-structured Σ
   (as in RFC) fixes identifiability.
2. **MBC linked sets with k ≥ 6.** These can't be modeled even when the
   analyst knows the items are linked.
3. **σ distortion.** RFC reproduces near-duplicate effects by inflating σ
   (about 4× in an earlier simulation; unverified, not used as a reference). The answer may be wrong. Even when
   shares are right, a wrong σ means a wrong explanation and wrong downstream
   use.
4. **Specification across subsets.** Before the fit, the analyst must declare
   where independence holds by choosing coded subsets. After the fit, any
   decision search over candidates that span subsets assumes independence
   between them. MVP learns the structure, and every combination is a query.
5. **RFC plus an estimated correlation matrix**, for mapping to business
   decisions (cost-weighted, margin-weighted). Depends on item 3: a wrong σ
   gives wrong weights.
## Active

### T5 · N-sweep (static sample savings)

- **Hypothesis.** Under correlated-error truths, MVP reaches the decision error that HB-MNL +
  RFC achieves at large N with materially smaller N. Under a logit truth, MVP is not materially
  worse at small N.
- **Kill criteria (user, 2026-10-09).**
  - "Materially smaller N": N_match ≤ 720, i.e. at least 40% fewer completes than RFC-S at
    N = 1,200. Anything under about 25% is hard to market. (On the grid this means N_match ≤ 600.)
  - "Matches": MVP's mean decision flips (or cost regret) within one replicate-SE of RFC-S's at
    N = 1,200.
  - G0 guard: at every N, MVP's regret is no worse than RFC-S's by more than one SE. If it fails
    only at N = 150, record the failure; don't kill on it.
- **Scope.**
  - In: T1's DGP and arms ([design](analysis/T1/DESIGN.md)): G0 (logit control), G1; G2 only if
    cheap. Methods: HB-MNL + RFC-S and MVP (GPU GHK). N ∈ {150, 300, 600, 1,200}, 20 replicates
    per cell.
  - OUT: adaptive or sequential fielding ([backlog](BACKLOG.md)), SEs, new truths, J scaling.
- **Step 0 (before T5 runs).** `analysis/T1/run.py` now fits MVP with the total error variance
  fixed (`--scale total`, default; the 2026-10-08 fix from `mvp_refit.py`); `--scale snu` keeps the
  σ_ν-pinned path for exact T1 reproduction. Rerun one T1 cell (G1, 3 replicates) with the fix and
  confirm it reproduces the 2026-10-08 refit results (`out/full_G1/results_mvp_total.json`):
  σ, S1, S2, decisions.
  - *Done 2026-10-09, PASS.* `run.py --truth G1 --reps 3 --out-name step0_G1`: MVP σ, S1, S2,
    log-likelihood, S4 and all decision fields identical to `results_mvp_total.json` (max |Δ| ≤ 1e-16);
    RFC-S arm identical to the original `full_G1/results.json` up to float rounding (~1e-15). 3 fits
    on T4 in 168 s wall; ~11 min total; GPU cost ~$0.10.
- **Tests (minimal).** For each truth × N × replicate: simulate, fit both methods, score the T1
  decisions (price change, line extension, cost-weighted metric) against the truth.
  - Metrics: decision flips and cost regret against the truth.
  - Result: error-vs-N curves per method, plus N_match = the smallest N at which MVP matches
    RFC-S's error at N = 1,200.
- **Results (2026-10-09).** Full tables: [RESULTS.md](analysis/T5/RESULTS.md). PASS on the kill rules
  (N_match 150 ≤ 720 for G1 and G2, both metrics); G0 guard fails only at N = 150 (recorded, not a
  kill). ~$3.40 GPU.

| truth | N | MVP flips (price / ext / cost) | MVP cost regret % | RFC-S flips | RFC-S cost regret % |
|---|---|---|---|---|---|
| G0 | 150 | 1 / 0 / 1 | 0.37 | 3 / 0 / 0 | 0.00 |
| G0 | 300 | 1 / 0 / 0 | 0.00 | 2 / 0 / 0 | 0.00 |
| G0 | 600 | 0 / 0 / 0 | 0.00 | 0 / 0 / 0 | 0.00 |
| G0 | 1,200 | 0 / 0 / 0 | 0.00 | 0 / 0 / 0 | 0.00 |
| G1 | 150 | 0 / 0 / 2 | 0.17 | 3 / 6 / 9 | 3.77 |
| G1 | 300 | 0 / 0 / 2 | 0.31 | 2 / 9 / 10 | 4.23 |
| G1 | 600 | 0 / 0 / 0 | 0.00 | 1 / 6 / 8 | 3.69 |
| G1 | 1,200 | 0 / 0 / 0 | 0.00 | 0 / 9 / 10 | 4.61 |
| G2 | 150 | 0 / 0 / 1 | 0.10 | 0 / 7 / 10 | 4.11 |
| G2 | 300 | 0 / 0 / 2 | 0.31 | 1 / 8 / 10 | 3.77 |
| G2 | 600 | 0 / 0 / 0 | 0.00 | 1 / 6 / 7 | 3.12 |
| G2 | 1,200 | 0 / 0 / 0 | 0.00 | 0 / 9 / 10 | 4.46 |

| truth | RFC-S error at N = 1,200 (flips per rep / cost regret %) | N_match (MVP), both metrics |
|---|---|---|
| G1 | 0.95 / 4.61 | 150 |
| G2 | 0.95 / 4.46 | 150 |

  - **Reading.** RFC-S does not improve with N (about 4% cost regret from 150 to 1,200): its error is
    bias from its form, not sampling noise. So the result is "MVP at N = 150 beats RFC-S at any N",
    not "MVP matches RFC-S with fewer completes". N_match = 150 is the grid floor; the true N_match
    may be lower.
  - **G0 guard.** N = 150: MVP cost regret +0.37 vs RFC-S (SE 0.37), one replicate (7.4%); an exact
    tie with 1 SE, counted as a failure, recorded per the rule. N ≥ 300: ok. No invented similarity.
  - N = 600 rows reproduce T1 (G1 RFC-S 1 / 6 / 8, 3.69%; G2 14 flips), as expected with the same seeds.

### T6 · Correlated-reach TURF on public basket data

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
    T3 / rho_test loader and IFM machinery ([design](analysis/T3/DESIGN.md)). Reach estimators: (a) independent
    product (MVP's fitted marginals), (b) MVP orthant, (c) the empirical / deterministic baseline
    only if trivial.
  - OUT: MaxDiff data, engine-vs-simulator timing ([backlog](BACKLOG.md), first up), posterior
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

### T7 · Engine vs shared-draw simulator for TURF search (design approved 2026-10-10)

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

### T8 · GPR: kernel-parameterised error covariance (design approved 2026-10-10)

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

### T9 · MACML as an incumbent probit arm (criteria set 2026-10-10)

Background, hypothesis and scope: BACKLOG.md, "MACML".

- **Kill criteria (user, 2026-10-10).**
  - Recovery gate (T1 G1, one dataset, vs the GHK fit with SEs; `analysis/T1/out/step0_G1`):
    coefficients and variance shares within 2 SE of the GHK fit, SE of the difference
    √(SE_GHK² + SE_MACML²), MACML SEs from the Godambe sandwich; implied correlation max |diff|
    ≤ 0.1 → **changed (user, 2026-10-10, after the gate diagnosis below): every distinct implied
    error correlation's gap within 2 SE of the difference** (delta method on each fit's sandwich).
    Failure = STOP and diagnose (a mis-implemented baseline invalidates the test).
  - Win (G2): gate on regret only. GHK mean cost regret < MACML's by > 2 paired-SE. Flips
    reported, not gating. G1 reported alongside (does the gap grow with correlation).
  - G0/G1 no-loss: recorded. Flag if MACML beats GHK by > 2 SE on G1 (suggests GHK simulation
    bias).
  - Time: GHK ≤ 10 min/fit on one T4 at T1 size (T5 N = 600 cells: 57–118 s, already met). MACML
    time reported with the implementation caveat; no relative speed claim. MACML feasibility kill:
    30 min/fit, timed on one dataset first.
  - J = 20: stop and ask after J = 3–6 regardless of result.
  - Implementation: (1) pybhatlib pairwise CML over panel tasks if supported; (2) else pairwise CML
    written here on CPU with pybhatlib's MVNCD (OVUS), in scope if ≈ 1 day; (3) no pybhatlib MSL
    panel substitute; (4) no GPU MACML on our MVNCD; (5) if (2) exceeds ~1 day, drop and document.
- **Implementation check (2026-10-10).** (1) is out: pybhatlib 0.4.0 has no composite likelihood;
  `models.mnp` is cross-sectional with one Σ over labelled alternatives, `models.mnpkercp` is panel
  MSL. So (2): per respondent, all 66 task pairs (T = 12); each pair term is a 6-dim orthant (two
  tasks × 3 differenced alternatives) with the cross-task covariance from the random coefficients.
  Gradient: pybhatlib's analytic OVUS gradient (∂P/∂a, ∂P/∂Σ) chained to θ (21 parameters).
  Timing, this box (2 cores): OVUS value + gradient 0.28 ms per orthant → 11 s per likelihood
  evaluation (39,600 orthants at N = 600), ~6 s on 2 processes. Finite-difference gradients would
  be ~2 min per evaluation (over the kill). Projected fit 60–150 evaluations ≈ 6–28 min, Godambe
  sandwich ≈ 3 min.
- **Implementation as built (2026-10-10).** All on GPU (user: port OVUS to torch). `analysis/T9/macml.py`:
  batched torch port of pybhatlib 0.4.0 `_mvncd_ovus` with pybhatlib's torch Genz BVN; autograd
  gradients; Godambe sandwich. `t9.py check`: per orthant vs pybhatlib max |Δlog P| 1.8e-15, except
  exact |a/sd| ties (23 of 1,320; up to 1.6e-3): OVUS depends on the variable order and pybhatlib's
  tie order is numpy's unstable default argsort; the port breaks ties stably. Two fixes: (i) pybhatlib's
  torch BVN high-correlation branch gave NaN gradients at equal limits (sqrt'(0) in a branch
  torch.where discards); a copy with sqrt floored at 1e-300 gives bit-identical values. (ii) OVUS re-sorts
  variables at every θ, so the composite likelihood jumps where an order flips (line scan: jumps
  0.03–0.055 vs smooth second differences 1e-7) and L-BFGS stalled (max|grad| 14–28, start-dependent
  optima). Fix (user, option A): freeze each orthant's order, fit, re-sort, repeat until no order
  changes (gate fit: 3 rounds, 39,558 → 3,690 → 0 re-sorted); at the fit every orthant is pybhatlib's
  OVUS value; both starts reach the same optimum.
- **Gate (2026-10-10).** First form FAIL: correlation gap 0.143 > 0.1 (flavor share MACML .240 vs GHK
  .383, truth .45), all coefficient and share z within 2. Reps 1–2 gaps 0.056, 0.026: the 0.1 bar is
  ~1 SE of the difference at N = 600. Revised form (above) **PASS**: max correlation |z| 1.46,
  coefficients |z| ≤ 0.30, shares |z| ≤ 1.34. `out/gate.json`, `gate.log`, `out/diag.json`, `diag.log`.
  MACML fit 34 s + Godambe 17 s on a T4. GHK covariance from a rerun of step0 rep 0 with SEs
  (θ reproduced exactly).
- **Arms.** MACML (new, `analysis/T9/`) vs MVP GPU GHK reused from T5's N = 600 cells (same seeds,
  no new GPU). Metrics as T5: S1, flips, cost regret, fit time.

- **Results (2026-10-10): FAIL on the win criterion: no decision difference at all.** `out/full_{G0,G1,G2}/
  results.json`, `out/summary.json`, `run_G*.log`. GPU: 1,762 T4-s main + ~400 gate/diagnosis
  (~36 T4-min, ~$0.35).

| truth | GHK regret / flips | MACML regret / flips | MACML − GHK (paired SE) | S1 error GHK / MACML | fit s median GHK / MACML | verdict |
|---|---|---|---|---|---|---|
| G0 | 0.00% / 0 | 0.00% / 0 | 0.00 (0.00) | +0.044 / +0.026 | 74 / 26 | recorded |
| G1 | 0.00% / 0 | 0.00% / 0 | 0.00 (0.00) | −0.024 / −0.022 | 87 / 32 | recorded; no flag |
| G2 | 0.00% / 0 | 0.00% / 0 | 0.00 (0.00) | −0.006 / −0.020 | 65 / 28 | **FAIL** (no GHK win) |

  - Both arms made the truth's decisions in all 60 replicates (as GHK did in T5 at N = 600: 0 flips,
    0 regret). T1's three decisions are saturated at N = 600, so the test cannot separate the two
    probit estimators; the gap does not grow with correlation because there is no gap.
  - MACML's pair correlation S1 is as close to the truth as GHK's (|error| ≤ 0.026 vs ≤ 0.044).
  - Time: GHK 57–118 s/fit (cap 10 min: met). MACML 18–50 s/fit on a T4 as our GPU port; per the
    criteria no relative speed claim (pybhatlib's own CPU code would be ~10–30 min/fit here).
  - Not tested (OUT / stop-and-ask): J = 20, smaller N where decisions are not saturated, decisions
    beyond T1's three.

### T10 · MACML vs GHK follow-up: smaller N, and J = 20 (Part A done, inconclusive; Part B PAUSED 2026-10-10)

User, 2026-10-10: follow up T9 (decisions saturated at N = 600, J = 3) with smaller N, and harder at
J = 20. Kill criteria: BLANKS for the user. Nothing runs until they are filled in.

- **Part A: smaller N, T1 DGP (J = 3).** T9's code unchanged; GHK arm reused from T5's N = 150 / 300
  cells (same seeds). Power warning: T5's GHK made only 1–2 flips per 20 replicates at N = 150–300
  (cost regret 0.10–0.37%, SE 0.1–0.4), so a > 2 paired-SE regret gap needs MACML to flip often.
  Estimate: no build; MACML 60 fits per N ≈ 30 T4-min ≈ $0.30 per N.
- **Part B: J = 20, T4 shelf DGP** (`analysis/T4/t4.py`: brand 5 × flavor 4 × pack 3 grid, J = 20
  subset, 12 shared error components; 13 decisions per dataset: 6 price, delist, 6 extensions;
  decisions scored by 10⁶-draw simulation, flip material at 0.5% regret).
  - MACML: T9's pair likelihood at 2J = 40 dims per pair (66 pairs × N).
  - GHK: T1's exact panel GHK at T·J = 240 dims (T4's MVP-G nested simulation was the costly path).
    Not yet validated at this size; GHK noise at 240 dims with M = 4,096 is a risk.
  - Step 0, timing probe: one likelihood + gradient per arm at J = 20, N = 600 (build ~2 h, ~$0.05).
  - Rough estimate before the probe (d² scaling from T1, unverified): GHK ~45–90 T4-min per fit,
    MACML ~5–20; S1 truth × 10 reps ≈ $8–15; adding S0 doubles it.
- **Kill criteria (user, 2026-10-10).**
  - Step 0 likelihood (user): the 240-dim panel orthant arises only from the person-level tastes
    β_n (ζ and ν are fresh per task), so GHK uses the conditioned likelihood
    P(y_n) = ∫ Π_t P(y_nt | β_n) φ(β_n) dβ_n: QMC over β_n × per-task 20-dim GHK (T4's MVP-G).
    Full 240-dim GHK only for across-task occasion errors (not this model). The 240-dim probe was
    stopped unfinished.
  - Recovery gate (J = 20, one shelf-truth dataset, each arm vs the TRUTH): coefficients and
    variance shares within 2 SE (own SEs; Godambe for MACML); implied correlation max |error| ≤ 0.1.
    MACML failure: record, drop the arm, report. GHK failure: stop.
  - Win: mean regret over the 13 decisions per dataset; GHK < MACML by > 2 paired-SE. Reported, not
    gating: by decision type and flips at 0.5%; delist is underpowered (1 per dataset).
  - Truths / replicates: shelf (S1) and null (S0), 10 each. Pre-registered extension: paired gap
    between 1 and 2 SE → extend to 20; otherwise stop at 10.
  - Time caps: GHK ≤ 90 T4-min per fit, feasibility kill if the probe projects > 3 h; MACML ≤ 30
    min per fit, kill if > 2 h. On a kill: stop and ask.
  - Part A: N = 150 and 300, T9's criteria unchanged. Labelled in advance as expected inconclusive
    (it tests sample efficiency, not correlation strength).

- **Part A results (2026-10-10): inconclusive, as labelled.** `analysis/T9/out/full_G*_N{150,300}/`,
  `summary_N150.json`, `summary_N300.json`, `analysis/T9/run_G*_N*.log`. ~33 T4-min (~$0.33).

| N | truth | GHK regret / flips | MACML regret / flips | MACML − GHK (paired SE) |
|---|---|---|---|---|
| 150 | G0 | 0.37% / 2 | 0.37% / 3 | 0.00 (0.00) |
| 150 | G1 | 0.17% / 2 | 0.25% / 3 | +0.08 (0.15), no flag |
| 150 | G2 | 0.10% / 1 | 0.65% / 3 | +0.55 (0.45), no win (1.2 SE) |
| 300 | G0 | 0.00% / 1 | 0.00% / 1 | 0.00 (0.00) |
| 300 | G1 | 0.31% / 2 | 0.31% / 2 | 0.00 (0.00), no flag |
| 300 | G2 | 0.31% / 2 | 0.31% / 2 | 0.00 (0.00), no win |

- **Step 0 probe (2026-10-10), one T4, J = 20, S1, N = 600.** `analysis/T10/probe.py`,
  `out/probe.json`, `out/probe_macml_chunks.json`. Evaluations per fit from T4's gate fits.
  - GHK conditioned (MVP-G) 256 × 128: 40 s / eval × ~55 evals ≈ 37 min fit (+ ~31 min SEs).
    T4's gate (N = 1,000) at this size under-recovered shares (u z −2.4 / −2.5).
  - GHK conditioned 512 × 256: 160 s / eval (chunk 4; chunk 8 OOM on 16 GB) × ~78 ≈ 3.5 h fit:
    over the 3 h feasibility kill. This is the size at which T4's gate recovered the shares.
  - MACML (40-dim pairs): 9 s / eval at chunk 60 (1.5 GiB) → ~15–25 min per fit with order rounds.
  - Stopped and asked (kill rule) 2026-10-10. Options put to the user: (1) gate GHK at 256 × 128
    (in the cap, known share under-recovery); (2) an intermediate 320 × 160; (3) stop Part B.
  - **PAUSED (user, 2026-10-10: "Lets pause on this") with that choice open.** No Part B fits run.
  - **Bigger-GPU probe (user, 2026-10-12, cap $1).** `probe_gpu.py`, `out/probe_gpu.json`, `probe_gpu.log`. Same
    512 × 256 evaluation as above, chunk 32 (peak 65 GiB): A100-80GB 22.4 s / eval (7.2× the T4), H100 12.8 s (12.5×).
    At ~78 evaluations: A100 ~29 min, H100 ~17 min per fit (SEs not timed), under the 90-min cap. At list prices
    (T4 $0.59/h, A100-80GB $2.50/h, H100 $3.95/h) about $2.10 / $1.20 / $1.10 per fit. Probe ~2 min wall, ~$0.15.

- **Side study: OVUS vs the engine (user, 2026-10-10, while Part B is paused).** Not a kill-criteria
  test; measurements only. Accuracy by |ΔP| (absolute, user: ratios alone mislead), by dimension.
  - Accuracy on T9/T10's own orthants (`ovus_vs_engine.py`, `out/ovus_vs_engine.json`; 100 orthants
    per set; reference GHK scrambled Sobol 2^15 × 8). |ΔP| median (max):

| set | d | median P | engine high | engine low | OVUS |
|---|---|---|---|---|---|
| T9 pair, G2 | 6 | 0.087 | 6.8e-3 (3.1e-2) | 2.0e-2 (1.5e-1) | 5.2e-4 (3.5e-3) |
| T9 pair, G1 | 6 | 0.084 | 6.9e-3 (3.1e-2) | 1.9e-2 (1.5e-1) | 5.0e-4 (3.4e-3) |
| J = 20 task given tastes | 20 | 0.19 | 2.7e-3 (1.3e-2) | 9.9e-3 (4.0e-2) | 2.1e-3 (1.9e-2) |
| J = 20 MACML pair | 40 | 0.0085 | 3.6e-3 (1.4e-2) | 2.9e-3 (1.7e-2) | 8.0e-4 (3.1e-3) |

  - Speed, CPU, 2 threads (`speed_sweep.py`, `out/speed_sweep.json`), µs per orthant at the largest
    batch run: engine (compiled) 15 / 103 / 424 at d = 6 / 20 / 40; OVUS torch 6.3 / 38 / 163. The
    engine is flat in batch; OVUS falls with batch (fixed per-call overhead amortised). The engine's
    torch version matches the compiled one to ≤ 2.2e-16.
  - Speed, GPU, one T4, numpy in / out (`gpu_sweep.py`), µs per orthant at batch 10⁵: first run
    (`out/gpu_sweep_before_fix.json`) engine 1.9 / 25 / 1,456 vs OVUS 0.70 / 4.3 / 26. The d = 40
    engine figure was a GPU batched-kernel limit above d = 32 in one preprocessing step; fixed in the
    engine (results unchanged to ≤ 7e-18 on T10's orthants). Rerun (`out/gpu_sweep.json`): engine
    1.9 / 25 / 32 vs OVUS 0.72 / 4.3 / 28. GPU equals CPU to ≤ 3.3e-16 for both methods.
  - On generated mixed-sign two-factor correlation matrices (500 per d, GHK reference SE ≤ 2.2e-5),
    |ΔP| median (max): OVUS 8.7e-4 (1.6e-2) / 8.2e-4 (6.7e-3) / 4.5e-5 (2.1e-3) at d = 6 / 20 / 40;
    engine high 5.6e-3 (4.9e-2) / 4.1e-3 (3.6e-2) / 2.3e-4 (7.0e-3). Cost at 10⁵ about equal at d = 40,
    OVUS cheaper at d ≤ 20. On T10's positive-leaning orthants the gap is smaller (d = 20 about equal).
  - GPU ~25 T4-min in all (~$0.25).

### T11 · Joint chance-constrained product line (FAIL 2026-10-11)

- **Hypothesis.** A joint chance constraint on line survival, P(every kept SKU's share ≥ t) ≥ 1 − α,
  evaluated as one d-dim orthant under a delta-method normal share distribution, (1) selects
  different assortment/price decisions than point-estimate or independent-product constraints,
  (2) those decisions are backed by realized survival across replicates, (3) orthant evaluation
  reaches the same decisions as nested MC materially faster.
- **Setup (proposed concretions, user to confirm).**
  - Fits: T5's MVP GHK fits, G1 and G2, N = 150 and 600, 20 reps (80 datasets, same seeds). No refits.
  - Market: competitors B, D, E fixed. Owned (J = 6): A, C (T1's owned pair) + A′, A″, F, G (T1's
    candidates). Price grid on A: {4.50, 5.00, 5.50} (A″ stays at 4.70). 127 lines (32 subsets with
    A × 3 prices + 31 without). d = kept SKUs, 1–6. Objective Σ_kept (p − 1.80)·ŝ_j (T1 economics).
  - Shares ŝ(θ̂): population shares by T1's GPU GHK share simulator with fixed draws, so ŝ is
    deterministic in θ and ∂ŝ/∂θ comes by autograd. Σ_s = J V Jᵀ.
  - t and α: α = 0.10; t picked from {2, 3, 5, 8, 10}% as the value whose binding rate under D is
    nearest 50% on G1 N = 600 reps 0–4 (fitted models only, never the truth), then fixed for all
    cells. "Binds" = arm A's line is infeasible under R.
  - R: θ_b ~ N(θ̂, V), B = 2,000 (MC SE of P_R ≤ 0.007 at P = 0.9); report B and P_R's MC SE.
    Same share simulator per draw.
  - F: OVUS GPU port from T9, variable order fixed per orthant at evaluation (order frozen across
    the price grid in the probe).
- **Step 0 (needed: T5 stored θ̂ only, no V).** Sandwich V at the stored θ̂ (Hessian + scores at
  the fit; no optimisation). Step0_G1 timing: 72 s at N = 600. Estimate 40 × ~72 s + 40 × ~20 s
  ≈ 60 T4-min ≈ $0.60. Then a 1-dataset timing probe of R (~$0.05) before the full run.
- **Estimate (before the probe).** Build ~half a day. D/E/F searches seconds per dataset. R:
  80 × 189 lines × 2,000 draws of a 10-alternative share simulation, rough 1–3 T4-h ($0.60–1.80).
  Total paid ≈ $1.30–2.50.
- **Kill criteria (proposed, approved by user 2026-10-11).**
  - Binding gate: if the constraint binds in < 20% or > 80% of the 80 dataset searches, stop and
    ask before scoring.
  - (1) Different decisions: PASS if, among binding searches, D's line ≠ B's in ≥ 15% and
    D ≠ C in ≥ 10%. KILL if D ≠ C in < 5% (joint adds nothing over independence).
  - (2) Survival: PASS if D's realized survival ≥ (1 − α) − 2 binomial SE pooled per truth × N,
    AND D's survival exceeds B's by > 2 paired SE, AND D's profit regret ≤ C's + 2 paired SE.
    KILL if D's realized survival < (1 − α) − 2 SE at N = 600.
  - (3) Speed: PASS if the fastest of D/E/F has decision agreement with R ≥ 95% and full-search wall
    time ≤ R/10. D vs E and D vs F reported, no gate (at d ≤ 6 all three are microsecond-scale; the
    gap to R is the delta method, not the integrator).
  - Normality flag (not a kill): median |P_delta − P_R| > 0.05 in a cell → that cell's (2) is
    reported with the caveat; N = 150 expected to be the weak cell.
  - Overall: PASS = (1) and (2) PASS; (3) separate. INCONCLUSIVE if (1) passes and (2) is within 2 SE.
- **Report-only probe.** One assortment (all six kept, d = 6), A's price on a 41-point grid
  4.50–5.50: P from D, E, F; max step jump / local slope; finite differences vs the Plackett
  gradient (∂P/∂a and ∂P/∂ρ chained through ∂Σ_s/∂p), per integrator. No price search built.
- **Out of scope.** J ≥ 20, posterior draws, gradient search, new truths, real data, G0.
- **Approved as drafted (user, 2026-10-11).** Code: `analysis/T11/t11.py` (ghk_gpu.py: sandwich and
  draws factored out, fit unchanged). Local wiring check OK.
- **Pilot (2026-10-11), G1 N = 600 rep 0, one T4: 147 s wall** (V 67 s, R 62 s at B = 2,000, delta step
  1.4 s, orthant arms < 1 s). nll at T5's θ̂ reproduces T5 (Δ 2e-4); V positive definite. D/E/F agree
  to ≤ 1.3e-3; |P_R − P_E| ≤ 0.016. Projected full run ≈ 2.5 T4-h ≈ $1.50 (in the estimate).
  - **Binding gate trips by construction (stop-and-ask).** With no per-SKU fixed cost, arm A keeps all
    six SKUs, and C's share is ~1.5% (fit and truth), below every t in {2..10}%: A's line is infeasible
    at every t, so binding ≈ 100% (> 80%). Full run not launched; awaiting user.
  - **User (2026-10-11): option 1, t grid extended with 0.5%, 1%, 1.5%;** same 50% selection rule. Arm names
    in results: Unconstrained (A), Point (B), Independent (C), Joint-engine (D), Joint-GHK (E), Joint-OVUS (F),
    Nested MC (R).
  - **User (2026-10-11): binding is not a gate** (Unconstrained failing the hurdle is its expected behaviour; a
    different test). **Every t scored, no selection**; overall PASS needs (1) and (2) at a majority of the 8 t.
- **Results (2026-10-11): FAIL. (1) KILL at all 8 t: the joint rule never picks a different line from
  Independent (0 of 640 searches).** `analysis/T11/out/summary.json`, `out/G*_N*/rep*.json`, `out/price_probe.json`,
  `run.log`. GPU: 80 jobs 973 s wall, ~2.5 T4-h (~$1.50); pilot + probe ~$0.10.

| t | NMC ≠ Point | Joint ≠ Point | Joint ≠ Independent | (1) | (2) | (3): agree with NMC (D = E = F) | speed vs NMC |
|---|---|---|---|---|---|---|---|
| 0.5% | 0.07 | 0.15 | 0.00 | KILL | fail | 0.90 | 45× |
| 1% | 0.25 | 0.33 | 0.00 | KILL | fail | 0.80 | 45× |
| 1.5% | 0.64 | 0.65 | 0.00 | KILL | fail | 0.78 | 45× |
| 2% | 0.36 | 0.36 | 0.00 | KILL | fail | 0.88 | 45× |
| 3% | 0.33 | 0.34 | 0.00 | KILL | fail | 0.79 | 45× |
| 5% | 0.81 | 0.80 | 0.00 | KILL | PASS | 0.76 | 45× |
| 8% | 0.44 | 0.44 | 0.00 | KILL | fail | 0.95 | 45× |
| 10% | 0.69 | 0.70 | 0.00 | KILL | PASS | 0.89 | 45× |

  - **Why joint = independent.** Near the 90% boundary, risk sits in one SKU: at t = 5%, 350 of 399 borderline
    lines have exactly one SKU with z < 2.5. With one risky SKU, the joint probability equals the product
    whatever the correlations (here −0.95 to 0.91; 68% of Σ_s mixed-sign). Joint vs Independent feasibility
    differs on ≤ 3 of 9,520 lines per t (max |ΔP| 0.01–0.11).
  - **Integrators.** Joint-engine, Joint-GHK and Joint-OVUS make identical decisions at every t;
    |P − P_NMC| medians identical to 4 digits by d.
  - **Survival (2).** The uncertainty-aware rules (Independent = Joint) keep realized survival 0.80–1.00
    (floor 0.77) in every cell; Point 0.30–1.00. Gain over Point > 2 SE at 5% and 10% in all cells, and
    in some cells at 2% and 8%. Price: regret 0.2–11% vs Point's −0.5 to 6% (Point lines often break the
    hurdle in truth); conservative at N = 150 (survival 0.95–1.00, regret up to 11%).
  - **Speed (3).** Full search ~1.4 s (delta step 1.35 s, orthants ≤ 0.06 s) vs NMC 62 s at B = 2,000: 45×,
    from the delta method, not the integrator (Independent costs the same). Agreement with NMC 0.76–0.95;
    passes only at 8%.
  - **Normality.** Median |P_delta − P_NMC| ≤ 0.03 (N = 150, t 1–2%), ~0 at N = 600 for t ≥ 3%; max 0.10–0.34
    in every cell. Joint max = marginal max: the gap is non-normal marginal shares near zero, not dependence.
  - **Price probe (report only)**, all six kept, G1 N = 600 rep 0, 41 prices: informative only at t ≤ 2%
    (C at ~1.5% puts P ≈ 0 above). Max step jump vs E's local slope: D 0.03 / 0.10 / 0.05 / 0.20 at
    0.5 / 1 / 1.5 / 2%, E and F ≤ 0.15 (curvature, not roughness). FD vs Plackett (max |Δ| / max |g|):
    E, F ≤ 2.3e-5; D 3e-6 / 2e-4 / 1.7e-3 / 1.5e-2, growing as P falls.
  - **Reading.** The hypothesis fails at its core: on this DGP joint and independent chance constraints
    make the same decisions. What helps is the uncertainty-aware constraint itself (vs Point), and any
    product-of-marginals version gets that.

### T13 · Cardinality-constrained line with joint survival, 6 of 20 (KILL 2026-10-11)

- **Model (fixed).** T4 recovery-gate fit: J = 20 shelf, MVP-G 512×256, N = 1,000, S1 truth dataset
  (`analysis/T4/out/gate_g/fit_512x256.json`: θ (25) and sandwich covariance). No new fits. Caveat: pack0 is
  ~−3.4 SE in every T4 fit (likely this dataset).
- **Decision.** Exactly 6 of 20 SKUs, C(20,6) = 38,760 lines; prices fixed at T4's base prices; cost $3.00
  (T4). Shelf = the 6 kept SKUs + "none" (no other alternatives). Shares are exact population orthants
  (normal tastes and errors), 7 per line, d = 6. Log-share delta method: Σ_log = J V Jᵀ, J = ∂log ŝ/∂θ.
- **Proposed t (from the fitted model, engine on CPU, no cost).** Unconstrained optimum: SKUs 3, 4, 6, 11, 15, 18,
  shares 6.5 / 7.6 / 8.7 / 4.7 / 16.6 / 7.9%, none 48%; log-share SDs 0.044–0.075; pair correlations −0.28 to 0.50.
  "Within reach" = 1% < P(miss) < 99%:

  | t | P(miss) per kept SKU | in reach | certain miss |
  |---|---|---|---|
  | 6.5% | .47 / .01 / 0 / 1.0 / 0 / 0 | 2 | 1 |
  | 7.0% | .83 / .11 / 0 / 1.0 / 0 / .03 | 3 | 1 |
  | 7.5% | .97 / .40 / .03 / 1.0 / 0 / .20 | 4 | 1 |

  The 4.7% SKU misses for certain at every t with 2+ in reach (SDs are tight at N = 1,000), so the unconstrained
  optimum is infeasible at every α; the constrained optimum is a different line.
- **Proposed floors for (2):** 90%, 95%, 98% of the unconstrained max expected profit (top lines are within
  ~2% of each other, so 98% is tight).
- **Arms.** Point, Independent, Joint-OVUS, Joint-GHK (reference integrator, scrambled Sobol 2^14 × 4),
  Joint-engine, Nested MC (θ ~ N(θ̂, V), shares per draw by GHK with fixed draws). Cells: (1) 3 t × 3 α = 9,
  (2) 3 t × 3 floors = 9.
- **Search.** Exhaustive (reference). Greedy: (1) add the SKU with the highest expected profit whose partial
  line meets the constraint, else the highest P; (2) profit-greedy to 6. Then best-improvement 1-swap
  (6 × 14 = 84 neighbours per step) on the cell's objective among feasible neighbours ((1): if infeasible,
  ascend P first). Report whether each reaches the exhaustive optimum, per arm.
- **Step 0 (run first, stop and report).** Delta step for all 38,760 lines, Independent and Joint (all three
  integrators) for all lines and t. Metrics as specified: kept SKUs with P(miss) > 1% at each arm's optimum and
  their shared brand / flavor / pack; Joint vs Independent feasibility disagreement under (1); top-20 rank
  changes under (2); pairwise excess Σ_{i<j}[P(miss_i ∩ miss_j) − p_i p_j] at the optimum by sign of the pair
  correlation (exact bivariate normal on log shares).
  - Cost: ~10–15 T4-min (delta step ~7 min, Joint-GHK on 116k orthants ~2 min) ≈ $0.15.
- **After Step 0.** Nested MC on all lines, B = 10,000 (MC SE 0.001 at 1 − α = 0.99, 0.002 at 0.95), GHK M = 256
  per draw. Estimate from T11's throughput: 2.5–5 T4-h ≈ $1.50–3.00, split across T4s; a 1-line-batch timing
  probe (~$0.05) first. If the probe projects > $5: stop and ask (fallback: Nested MC on the top 2,000 lines
  per cell). Local search timed per arm, with orthant batch sizes (exhaustive 38,760 × 3 t; greedy 20 − k;
  swap 84). True shares (T4 S1 truth) of each chosen line: reported, not scored.
- **Kill criteria (approved 2026-10-11).**
  - Step 0 stop: in every (1) cell Joint and Independent disagree on feasibility for < 0.5% of lines and pick
    the same line, AND in every (2) cell the top 20 lines have ≤ 1 rank change → STOP (report, no Nested MC).
  - H1 (decisions differ): Joint-GHK's line ≠ Independent's in ≥ 2 of the 9 cells of (1) or ≥ 2 of the 9 of (2).
  - H2 (backed by Nested MC): in the cells where they differ, Nested MC agrees with Joint in a majority;
    under (1) Joint's line has P_NMC ≥ 1 − α − 2 MC SE; under (2) Joint's P_NMC exceeds Independent's by
    > 2 MC SE.
  - PASS = H1 and H2. KILL = Joint = Independent in all 18 cells. Otherwise INCONCLUSIVE (one dataset, no
    replication).
  - Integrators: report agreement of Joint-engine and Joint-OVUS with Joint-GHK; flag any cell where an
    integrator alone changes the decision.
  - Speed and search: report only (T11: the time saving vs Nested MC is the delta step).
- **Out of scope.** Prices as decisions, new fits, other J, gradients, P(π ≥ L), replication over datasets.
- **Approved as drafted (user, 2026-10-11).** Code: `analysis/T13/t13.py`.
- **Step 0 (2026-10-11): one T4, 345 s wall (~$0.06).** `out/step0.npz`, `out/step0_report.json`. Delta step 209 s
  for all 38,760 lines (51 share evaluations each, GHK M = 1,024; checked against M = 2^15: ≤ 1.7e-4 abs). Orthant
  time per t over all lines: Independent 0.007 s, Joint-engine 0.08 s, Joint-OVUS 0.03 s, Joint-GHK 37.5 s.
  - Unconstrained optimum on GHK shares: SKUs 3, 4, 6, 11, 15, 18 (as sized; the CPU engine shares used for sizing
    were off by up to 0.009 abs, 17% rel on a small share, without changing the optimum).
  - **(1): Joint and Independent disagree on feasibility for 0 of 38,760 lines in all 9 cells (all three
    integrators), and pick the same line in all 9.** The constraint is near-deterministic: only 0–6 lines per
    cell have P within ±0.01 of 1 − α (log-share SDs 0.04–0.08 at N = 1,000). Joint and Independent differ by up
    to 0.12 on single lines, but only on lines far from the boundary (low P). t = 7.5%, α = 1%: no feasible line.
  - **(2): the chosen line is the same for Joint and Independent in all 9 cells.** Top-20 ranks do change at
    floors 90% and 95% (7–18 positions, 0–4 lines swapped into the top 20); at 98% only 4 lines qualify, no change.
  - At the optima: Joint/Independent optima keep 0–2 SKUs with P(miss) > 1% under (1) and 3–4 under (2), sharing
    pack 2 at most. Pairwise excess at the (2) optimum (t = 7%): +0.014 (positive pairs), −0.071 (negative pairs):
    negative pairs dominate, so Joint < Independent there (0.257 vs 0.300), without moving the argmax.
  - **Against the criteria.** Step 0 stop rule as written is not met (top-20 changes at 90% / 95% floors), but
    H1 is 0 of 18 cells: Joint = Independent on every decision, i.e. the KILL condition, which Nested MC cannot
    change. Stopped and asked whether to run Nested MC (~$1.50–3) for the reference agreement anyway.
  - **User (2026-10-11): stop, KILL.** No Nested MC. Total T13 GPU ~$0.06.

### T14 · MBC linked sets at k > 12 (approved 2026-10-11; Part A PASS 2026-10-12, Part B PASS 2026-10-13)

- **Part A, simulated (T2 extension).** T2's DGP and truths (L0 null, L1 factor, L2 two-cluster, LC
  complementarity) at k = 20 and 30, menu M = k (T2's k = 12 cell was M = k), 3 price levels, own-price effects
  only, no heterogeneity; N = 1,000 × 10 menus + 4 holdout menus; 20 replicates per (truth, k) = 160 datasets.
  L2 splits the linked set in halves (10 + 10, 15 + 15). LC: T2's two pairs + one triple.
  - Arms: CS-F (analyst lift rule, cap 5), CS-O (oracle blocks), IL, MVP-IFM (`multivariate_probit`, margins in all
    M log prices; nearest-PSD repair of R: rate, ‖R − R̃‖_F, min eigenvalue before), MVP-ML (GPU GHK, the `mvp_fit`
    service as in T2, time-capped).
  - Decisions as T2, scored against the truth by GHK-QMC: best pair, best triple, TURF-4 (C(20,4) = 4,845 /
    C(30,4) = 27,405 sets), D-price errors. Holdout pattern LL by per-observation GHK (2^k enumeration is out).
- **Part B, real (T3 extension).** Proposed cluster: **home baking, 24 items** (eggs, sugar, flour, butter, cream
  cheese, cake mix, frosting, brownie mix, muffin/cornbread mix, chocolate chips, marshmallows, baking nuts, oil,
  cooking spray, shortening, baking powder/soda, extracts, pie filling, pie crust, whipped topping, cake decor/icing,
  graham crackers, aluminum foil, refrigerated cookie dough). Basket rates 0.25–9.5%; raw lifts median 4.8, 98% of
  276 pairs > 2 (raw lifts are inflated by trip size; T3's gate decides). Same households, covariates and
  time-split holdout as T3.
  - Arms: CS-F, CS-O, IL, MVP-IFM, MVP-ML if within the cap. Decisions: best pair, best triple, TURF-4, and the
    recommendation A → B for each of the 24 items (27 decisions). Holdout adjudication as T3 (household bootstrap).
  - Build change: T3 enumerates all 2^M patterns; at M = 24 decisions use 2–4-dim marginals and the holdout LL uses
    per-trip GHK on GPU.
- **Report only.** Per-observation likelihood cost (value + gradient) at d = 20 / 30, GHK (M draws as used) vs OVUS
  (T9's GPU port), one T4.
- **ML time cap (proposed).** 60 T4-min per fit. Step 0 times one ML fit at k = 30 (L2) and at M = 24 (real); if
  either projects > 60 min: stop and ask.
- **Kill criteria (approved 2026-10-11, adjusted below).**
  - A, Q1: pool L1 + L2 at k = 20, 30 (80 datasets × 3 decisions = 240). PASS if CS-F flips exceed MVP-IFM's by
    ≥ 48 (20 pts) and CS-F's mean regret exceeds MVP-IFM's by > 2 paired SE. KILL if CS-F is within 24 flips (10 pts)
    of MVP-IFM. Else INCONCLUSIVE.
  - A, guards (caveats, not kills): L0 median |ρ̂| ≤ 0.05; LC reported.
  - A, Q2 (report): IFM-vs-ML decision agreement; PSD repair rate and size, and whether repaired datasets flip more.
  - B gate (as T3): ≥ 6 items connected at ρ ≥ 0.2 after covariates; fail → stop and ask.
  - B: PASS if, on decisions where CS-F and MVP-IFM disagree, the holdout backs MVP on ≥ 3 more than CS-F and on a
    majority of the resolved ones. KILL if the holdout backs CS-F at least as often as MVP.
  - B controls: T3's two simulated controls, IFM only.
- **Estimate.** Build ~1 day. Step 0 (two capped ML timing fits + the OVUS/GHK microbench): ≤ 2 T4-h, ≤ $1.20.
  A without ML (CPU containers, as T2): ~$2. A with ML: all 160 fits at 20–60 min ≈ $30–95; or **ML on the kill
  pool only (L1, L2 × k 20, 30), 10 reps = 40 fits ≈ $8–24 (proposed)**. B: CPU ~$0.50, one ML fit ≤ $0.60, holdout
  LL on GPU ~$0.10.
- **Out of scope.** Engine in the likelihood, Gibbs, new truths.
- **Approved with adjustments (user, 2026-10-11).**
  - Baking cluster approved. Staples (eggs, sugar, oil, foil) test false linking: trip size must be a covariate
    (T3's trip-size covariate kept, alongside household propensity); post-covariate ρ for staple pairs reported
    separately, so a pass cannot rest on staple links.
  - M = k approved (partial menus are a separate question).
  - 60-min ML cap approved. If ML projects over the cap at a k: IFM only there, IFM-vs-ML agreement recorded as
    unanswered at that k (ML is not in the kill criteria).
  - ML on L1/L2, 10 reps (40 fits) approved.
  - Part A: the 240 decisions are best pair, best triple, TURF-4 (80 datasets × 3); price-change errors reported
    separately. CS-O reported alongside CS-F (if oracle subsets also lose, the claim is about the method).
  - L0 guard: also report max |ρ̂| and whether PSD repair ever triggers under L0 (it should not; if it does, a red
    flag on the pairwise step).
  - Part B: results by decision type (pair, triple, TURF-4, follow-ons). "Resolved" = held-out difference
    ≥ max(0.5 pts, 2 SE) under the household bootstrap (T6's rule), fixed up front. PASS also requires MVP not to
    lose on any of the 3 set decisions (pair, triple, TURF-4). Follow-ons are not independent tests (under
    independence CS-F tends to pick the most popular item for every follow-on).
- **Step 0, first attempt (2026-10-11): lost.** The bench OOM'd (one autograd graph over all chunks) and the error
  discarded both ML results after ~63 min wall (~$1.20, T4 mvp_fit service); ≥ 1 fit ran to the 60-min T4 cap.
- **Step 0 rerun (user 2026-10-12: ML on an H100 via mvp_fit's own code in an ephemeral app; ≤ ~$0.40).**
  `t14.py step0`, `out/step0/*.json`, `step0_rerun.log`; each part saved as it arrives. 130 s wall, ~$0.15.
  - ML on one H100 (512 draws, float32 GHK, no SEs), max_iter 1 then 5 in one container (the 5-iteration call ran
    3 joint iterations: L-BFGS's eval cap is 1.25 × max_iter):
    sim L2 k = 30 (10,000 obs, 30 covariates): 4.2 s (incl. CUDA start) / 0.69 s, 5.1 GiB peak;
    baking (112,319 obs, 24 items, 47 covariates): 6.9 s / 5.1 s, 19 GiB peak.
    Upper bound per likelihood + gradient: ~0.1 s (sim), ~0.85 s (baking). Even at L-BFGS's 625-eval limit:
    ≤ ~1 min (sim), ≤ ~9 min (baking) per fit, well under the 60-min cap. T4 would be ~7–12× slower, so the lost
    T4 run's cap hit was most likely the baking fit.
  - Bench (report only, one T4, value + gradient, 512 draws): GHK 137 / 347 µs per obs at d = 20 / 30; OVUS
    48 / 80 (2.8× / 4.3× cheaper). Cost only; accuracy not measured here.
- **Part A results (2026-10-12): PASS.** `run_a.py`, `out/A_full/` (summary.txt, summary.json, per-job JSON), 160
  datasets, 0 errors, ~20 min wall, ~$4. Truth decisions GHK-QMC 2^16 (CPU); arms' probit scoring GHK-QMC 2^14 on a T4
  (checked identical to the CPU path on the selftest); ML (mvp_fit's code) on H100s, 5–8 s per fit, 40/40 converged.
  - Kill pool (L1 + L2, k 20/30, 240 decisions): flips CS-F 207, CS-O 119, MVP-IFM 50; gap 157 (PASS ≥ 48). Mean
    regret gap CS-F − IFM 5.2% of the best per decision, paired SE 0.30 over 80 datasets. In absolute points: flips
    with regret > 0.5 pts CS-F 185, CS-O 115, IL 235, IFM 2, ML 1; mean absolute regret CS-F 1.90 pts (max 10.1),
    CS-O 0.89 (max 3.6), IFM 0.037 (max 0.87), ML 0.033 (max 0.53). Oracle blocks also lose: the cap of 5 cannot
    hold a 20–30-item linked set.
  - L0 guard: IFM |ρ̂| median 0.013, max 0.091; PSD repair never triggered (0/40, nor on any truth). IFM flips 34 of
    120 under L0, 2 with regret > 0.5 pts (max 0.78 pts); CS 19, none > 0.5 pts.
  - LC (reported): CS-F / CS-O 0 flips > 0.5 pts (max 0.33); IFM 3 (max 0.71); IL 81.
  - Q2: IFM and ML pick the same set on 115/120 decisions (flips IFM 24, ML 23 on those datasets).
  - Price-change errors (reported separately): marginal-change MAE IFM / ML 0.34–0.42 pts vs CS-F 0.05–0.11 at every
    truth: the MVP margins carry all k log prices (spurious cross-price slopes); CS fits own + screened in-subset
    cross terms. A margin-specification effect, not the correlation.
- **Part B selftest (2026-10-12, 300 households, ~$0.40).** `run_b.py`, `out/B_selftest/`. Pipeline runs end to end
  (probit set quantities via a 2^24 latent-draw histogram on a T4, checked against brute force; coded-subset set
  quantities checked against simulation, max |Δ| 4e-3 at MC noise). Results not meaningful at this size: holdout
  covariates sit far from training (log-price columns at +1.35 / −1.31 training SDs on the later weeks), so predicted
  holdout rates are 2–13× the observed for every arm (eggs IL: 0.23 vs 0.10); raw IFM R min eigenvalue −1.50 (repair
  ‖R − R̃‖_F 2.28); ML hit its 500-iteration cap with a singular R. Fixes: IFM fit with the library's repair on and
  the raw pairwise R recomputed for the record; correlation matrices shrunk 1e-6 toward I when singular (scoring only).
- **Part B results (2026-10-13): PASS, at the threshold (3 − 0 = 3).** `run_b.py`, `out/B_full/` (gate.json,
  per-job JSON), `run_b_full.log`. 1,869 households, 149,789 trips (112,319 training), 45 covariates. ML 130 s on one
  H100 (284 iterations, converged, R min eigenvalue 0.036). Fits and controls on Modal CPU (2 vCPU, 8 GiB), scoring on
  T4s; 93 min wall.
  - Gate: PASS, 22 of 24 items linked at ρ ≥ 0.2; 19 with staple pairs zeroed. Post-covariate ρ, staple pairs median
    0.16 (max 0.48, 36% ≥ 0.2); non-staple median 0.16 (max 0.92, 41% ≥ 0.2). The pass does not rest on staples.
    Raw IFM R min eigenvalue −0.020, repair ‖R − R̃‖_F 0.024.
  - CS-F subsets: {eggs, muffin/cornbread mix, cooking spray}, {sugar, chocolate chips, marshmallows, baking nuts,
    graham crackers}, {flour, shortening, baking powder/soda, extracts, pie filling}, {butter, cream cheese, pie
    crust, whipped topping, aluminum foil}, {cake mix, frosting, brownie mix, oil, cake decor/icing},
    {refrig. cookie dough}.

| decision type | differ | backs MVP | backs CS-F | unresolved |
|---|---|---|---|---|
| best pair | 1 / 1 | 0 | 0 | 1 |
| best triple | 0 / 1 | 0 | 0 | 0 |
| TURF-4 | 0 / 1 | 0 | 0 | 0 |
| follow-on | 5 / 24 | 3 | 0 | 2 |

  - Resolved (holdout, MVP − CS-F, household-bootstrap SE): extracts → sugar vs → flour +22.2 pts (4.0); pie crust →
    eggs vs → cream cheese +10.7 (3.4); graham crackers → eggs vs → marshmallows +14.1 (3.8). Unresolved: best pair
    eggs + cream cheese 1.19% vs cake mix + frosting 0.87%, +0.32 pts (SE 0.08; under the 0.5-pt bar); chocolate
    chips → eggs vs → sugar −1.9 (3.0); baking nuts → eggs vs → sugar +2.7 (2.9). MVP loses no set decision.
  - Q2 (report): ML vs IFM differ on 1 of 27 (holdout backs IFM); ML vs CS-F differ on 5: ML 2, CS-F 0, 3 unresolved.
  - Controls (10 reps per truth, flips of 27 / material > 0.5 pts): S-MVP: IFM 21 / 19, CS-F 44 / 34, CS-O 67 / 55,
    IL 40 / 30. S-CS: IFM 37 / 27, CS-F 47 / 34, CS-O 47 / 33, IL 101 / 80. IFM flips least under both truths,
    including CS-F's own.
  - Holdout LL per trip: ML −1.524, CS-O −1.563, CS-F −1.569, IL −1.626. IFM −32.9 (S-CS controls −1,814): a few
    holdout patterns get near-zero probability under the repaired, near-singular IFM R. Not in the criteria; IFM's
    LL is not quoted until checked.
  - Invalid first run (kept as `out/B_full_badprice/`, `run_b_full_badprice.log`): `t3.load` dropped constant price
    columns by full-period SD, so prices constant in training (pie filling, extracts) were standardised by ~1e-15 and
    reached 2.8e13 on 5,598 holdout trips. Fixed: the SD rule uses training rows. T3's taco data is unchanged by the
    fix (same 16 columns, holdout max |z| 3.3).

## Paused

### T4 · Shelf scaling (item 1). Paused 2026-10-09 on cost

Reframe (2026-10-09): not a full joint correlation at J = 60 (1,770 correlations, not
identifiable from N = 600 × 12 tasks on any hardware). The claim is structured Σ (brand / flavor /
pack variance shares, CDT-shaped) recovered at J = 20–60 in practical time, changing delist, line
and price decisions vs HB-MNL + RFC; the (J − 1)-dim integral and its compute stay. See
[T4 design, Claim](analysis/T4/DESIGN.md#claim-reframed-2026-10-09-while-paused). Follow-ups are
in [BACKLOG.md](BACKLOG.md#t4-follow-ups).

**T4. Shelf scaling (item 1).** Simulate CBC at J = 20–60 with an
attribute-structured Σ. Fit the probit by engine MLE against GHK, Gibbs and
MACML; compare against logit plus tuned RFC. Record the number of shared
components K against J−1 (orthants vs conditioning on factors). Measure
time to fit against J and the decision difference against logit plus RFC.
Kill: the incumbents scale adequately, or decisions match logit plus RFC.

Status at pause:

| Test | Items | Status | Decision | Write-up |
|---|---|---|---|---|
| T4 | 1 | design approved 2026-10-08 (A–C as recommended); core (t4.py), MVP-F/G fitter (mvp_gpu.py, nested taste × inner-draw simulation) and recovery gate (gate.py) run 2026-10-09: FAIL as specified (128×128 collapses to product-only; 512×256 recovers shares but 29 min fit + 31 min SE at J = 20, b: pack0 z −3.2). Gauss-Hermite for the chosen nu is worse at equal compute (noise is in the 12 shared terms); per-task GHK at J = 20 is ~20× less noisy per unit time, so the gate reran with MVP-G as the main fit at J = 20 (user 2026-10-09: decisions first, scaling later, ~1 h per fit acceptable). GHK gate on L4 (256×128, 512×256): 512×256 hit the 3 h Modal limit at 04:37 UTC; the script saved nothing, so the finished 256×128 result was lost too (fixed: gate.py now saves each fit; fit timeout now 24 h, L4). Rerun 04:50–13:10 UTC on L4: 256×128 fit 34 min + SE 29 min, shares .11/.30/.29/.31 (u z −0.8/−2.4/−2.5); 512×256 fit 4.5 h + SE 3.6 h, shares .13/.35/.29/.23 (u z +0.2/−0.6/−1.2), pack0 −3.4, none −2.2, two taste SDs −2.7/−2.2; sizes agree to 1.8 SE. Strictly FAIL (2-SE and 1-SE rules); shares recovered at 512×256; pack0 ≈ −3 in every fit so far (likely this dataset). T4 PAUSED 2026-10-09 on cost (user): full run est. J = 20 only ~$45 at 512×256 (~$6 at 256×128), all J ~$600; SEs not needed for decisions (move S4 to the SE test) | — | [design](analysis/T4/DESIGN.md) |

## Completed: T1–T3

D1, the SE test, unit economics, kill-rule blanks and all follow-ups moved to
[BACKLOG.md](BACKLOG.md) on 2026-10-09. The original test plan and history are kept below.

### Original test plan (2026-10-07)

#### Tests

Proposed order: **T1 → D1 → T2 → T3 → T4** (to confirm).

| Test | Items | Q1 | Q2 | Q3 | One line |
|---|---|---|---|---|---|
| T1 | 3, 5 | σ recovery and decisions, probit vs HB-MNL + tuned RFC | — (CBC: engine MLE only) | not a goal | Is RFC's σ wrong, and does it move decisions? |
| D1 | 5 | presentation | — | — | RFC shares + estimated Σ → cost-weighted decision. Only if T1 recovers σ usably. |
| T2 | 2, 4 | coded subsets vs MVP, k = 5 to 10+ | IFM vs engine MLE vs GHK / Gibbs / MACML | time-to-decision vs k | Do incumbent methods scale with the linked set? |
| T3 | 2, 4 | coded subsets vs MVP on dunnhumby | IFM vs MLE if in the approved design | analyst + compute time | Same as T2 on revealed data. Skipped if T2 is killed. |
| T4 | 1 | probit vs logit + tuned RFC at J = 20–60 | engine MLE vs GHK / Gibbs / MACML | time-to-fit vs J | Does probit scale on the shelf, and does it change decisions? |

**T1. σ recovery (items 3, 5).** Simulate from a probit with per-attribute
similarity. Fit HB-MNL with tuned RFC against a probit fitted by engine MLE.
Measure σ recovery and three downstream decisions: a price change, a line
extension, and a cost-weighted metric. Kill: σ is recovered equally well and
the decisions match.

**T2. Scaling, simulated MBC (items 2, 4).** Sweep linked-set size k from 5 to
10+, plus ρ and sample size, including linked sets that cross the analyst's
coded subsets. Methods: coded subsets with independence for the rest; MVP by
IFM; MVP by engine MLE; MLE by GHK; Gibbs; MACML. Measure the decision
difference and time-to-decision as a function of k, per method. Hypothesis:
the incumbent methods don't scale. Kill: no decision difference and no
meaningful time saving. If killed, T3 is skipped.

**T3. Real data (items 2, 4).** Reuse the canonical `analysis/rho_test/run.py`
pipeline (dunnhumby Complete Journey). Find a natural cluster of 8+
complements; compare coded subsets against MVP. This absorbs the pending
`analysis/joint_vs_proxy/` head-to-head and keeps its fairness controls, but
parity is a pass, not a fallback. Pass: a different decision, or the same
decision in significantly less time (analyst specification plus compute).
Caveat: revealed data, not stated MBC data.

### Status

| Test | Items | Status | Decision | Write-up |
|---|---|---|---|---|
| T1 | 3, 5 | G0, G1, G2 × RFC-S, RFC-G done 2026-10-08; MVP refit (total variance) | PASS (MVP better) with caveats, 2026-10-08 | [design](analysis/T1/DESIGN.md), [results](analysis/T1/RESULTS.md) |
| T2 | 2, 4 | done 2026-10-08 (L0, L1, L2, LC × k 5/8/12 × 20) | PASS on decisions (MVP 28 vs CS 146 flips of 240); no time saving at k ≤ 12; T3 goes ahead | [design](analysis/T2/DESIGN.md), [results](analysis/T2/RESULTS.md) |
| T3 | 2, 4 | done 2026-10-08 (taco cluster, 8 items, 1,869 households) | PASS (MVP better): 6 of 11 decisions differ, holdout supports MVP on 4, CS on 0; controls pass; no time saving | [design](analysis/T3/DESIGN.md), [results](analysis/T3/RESULTS.md) |

### History (tracked next steps, through 2026-10-08)

- **T1 extensions (G0, G2 done 2026-10-07; RFC-G done 2026-10-08):** G0 (null truth: does MVP invent
  σ?), G2 (a stronger-similarity truth than G1), RFC-G (one σ per attribute,
  the generous proxy). Each needs your go.
- **Engine accuracy on T1's panel problems.** On 3 of T1's 6-dim panel orthants
  the engine was 0.004–0.007 off SciPy (3–9% relative), within its documented
  1–2 digits. Not re-tested; the T1 recovery check is the fit-for-purpose gate.
  If it fails, options go to you.
- **T1 MVP likelihood (was blocking; resolved 2026-10-07 by GHK on GPU, below).** The exact 36-dim panel likelihood jumps
  with parameter steps of 3e-3 on respondents with very small likelihoods
  (about e^-19): engine error on small probabilities, as documented, not a
  merge effect (same with dup_corr = 0). The finite-difference L-BFGS-B fit
  stopped after 14 iterations, far from the truth. Options are with the user.
- **T1 uses the engine at 3 dims only (option 1, 2026-10-06).** Expected
  pushback: "RFC already handles 3 variables fine." Answer to keep in the
  write-up: T1 is about Q1 (is the estimated σ right?); RFC tunes one σ to
  holdout shares and does not estimate it from choices. T1 makes no Q3 claim:
  at 3 dims GHK or MC would also do. The engine-specific claim is for T2/T4
  (high-dim). Option 2 (pairs of tasks, 6 dims) is the fallback.
- **Products per task (user's hypothesis, 2026-10-06).** RFC's σ inflation
  may be smaller when tasks have only 3 products (fewer near-duplicates in
  holdouts). T1 uses 3 per task; decision scenarios have 6. Candidate
  follow-up: 3 vs 5+ products per task.
- **Margin model for T2/T3 (2026-10-06).** `multivariate_probit` takes any
  classifier per item (η_j = Φ⁻¹(p̂_j)). T1/T4 (CBC) use a linear index fitted
  jointly by MLE; plug-in margins do not apply there. For T2 (linear DGP) linear
  margins are correctly specified. For T3 decide: one fixed margin model for
  both MVP and the incumbent's per-item terms, or report linear and flexible
  separately, so a better margin is not credited to the joint structure.
  Flexible margins also favour IFM over full MLE (Q2).
- **Flexible-margin regime (candidate, not approved).** Arbitrary margins need
  per-item binary outcomes and IFM; incumbents (HB-MNL/RFC, coded subsets,
  MACML/GHK/Gibbs) estimate a parametric index jointly. Expected lift: low in
  CBC (not applicable) and stated MBC; moderate-high in observational
  multi-outcome data (Bhat-style multivariate binary/ordered, baskets) for
  prediction, low for pricing on non-randomized prices. Strengthens IFM in Q2;
  richer margins shrink Σ (rho_test: median |ρ| 0.30 → 0.08 with trip size).
  Options: a linear-vs-flexible variant in T3, or a new test on a dataset you
  choose.
- **T1 selftest result (2026-10-06).** Recovery check at N = 1,000 passed the
  pre-registered rule (σ within 2 SE: z = +0.66, −1.41, −1.40), but 16 of 21
  parameters have negative z and all 9 taste spreads ω are low (z −0.04 to
  −2.08); implied S1 0.80 vs 0.90 true. Possible causes (not separated): MSL
  simulation bias at R = 200, engine 3-dim values about 0.6% low, early stop
  (max |grad| 5.4). Fit took 2.1 h + 0.5 h SEs on CPU.
  **Diagnostics (`analysis/T1/diag.py`, same dataset):**
  - *Optimizer ruled out.* A refit at R = 200 started at the truth reached the
    same optimum (nll 13607.81 vs 13607.84; σ 0.48/0.72/0.72; S1 0.800).
  - *Simulation bias at R = 200.* The fit beats the truth by 27.7 nll at R = 200,
    more than sampling noise explains (about 10.5 expected for 21 parameters;
    99th percentile about 19.5). Rescored at R = 500, the gap is 7.4, inside noise.
  - The full run already uses R = 500. Not yet shown: that a fit *at* R = 500
    recovers S1 ≈ 0.90.
- **T1 MVP by GHK on GPU (2026-10-07, resolves the blocking item).** The exact 36-dim
  panel likelihood, P(y_n) = one orthant of dim T(J−1) = 36 with covariance
  Δ(X diag(ω²) X' + blockdiag Σ_a σ_a² S_a)Δ', evaluated by GHK with scrambled Sobol
  draws fixed for the whole fit (smooth objective; autograd gradients and Hessian;
  sandwich SEs). No taste draws, so no MSL bias. Not the engine (its 36-dim values were
  too rough; the engine still does pop_shares/decisions at 3–4 dims). `run.py --mvp ghk`
  (default) fits all replicates in parallel on T4s before the CPU arms; `--mvp msl` keeps
  option 1. Code: `analysis/T1/ghk_gpu.py` (ephemeral Modal app, nothing deployed).
  - *Likelihood accuracy* (scratch study, R = 600, own T1-like DGP, at the
    truth): bias of the total log-likelihood −26 (MSL R = 200), −9.4 (R = 500), −1.0
    (GHK M = 1024), +0.19 (M = 4096, within noise).
  - *Recovery, selftest dataset (N = 1,000):* GHK M = 4096: 35 iters, fit 126 s + SEs
    120 s on a T4 (MSL: 2.1 h + 0.5 h CPU). σ = 0.50/1.09/1.06 vs 0.36/0.86/0.86,
    z = +1.09/+1.00/+0.91 (pass). **S1 0.891 vs 0.900** (MSL R = 200: 0.80). S2 shares
    0.096/0.456/0.435/0.013 vs 0.08/0.45/0.45/0.02. M = 1024 agrees (S1 0.887), so
    simulation error is small. nll beats the truth by 7.6 (≈ 10.5 expected).
  - *All 21 estimates sit ~+1 SE in magnitude* (σ, ω, |b| all ×1.2–1.4). That is one
    direction: the overall scale, pinned only by σ_ν = 0.18 (2% of error variance). MSL
    erred along the same direction the other way. Scale-free metrics (S1, S2, shares,
    decisions) are unaffected; raw σ_a recovery and S4 coverage are scale-sensitive. If
    S4 matters, consider G1 with a larger σ_ν, or report σ_a / total error sd.
- **multivariate-probit 0.3.0 `fitter="modal"` (noted 2026-10-06, not adopted).**
  Full ML for the pick-any binary probit on GPU: GHK with fixed Sobol draws, exact
  gradients and Hessian, SEs. Not T1's model (CBC). Candidate GHK/full-ML arm for
  T2/T3 (no respondent random effects: SEs need clustering). Revisit after T1.
- **T1 full G1 run (2026-10-07), [RESULTS.md](analysis/T1/RESULTS.md).** R = 20, N = 600.
  S1 median |error| MVP 0.033 vs RFC-S 0.243 (S1 0.869 / 0.657 vs 0.900); paired gap
  0.380 ± 0.118. Flips (price/ext/cost of 20): MVP 0/0/0, RFC-S 1/6/8; cost regret
  0.00% vs 3.69%. MVP S4 coverage 0.90/0.90/0.95; scale biased up ~1 SE (σ_ν caveat
  above). Total ~1 h (MVP 8 min on 20 T4s; HB/RFC the rest on CPU).
- **T1 G0 and G2 (2026-10-07), [RESULTS.md](analysis/T1/RESULTS.md).** G0 (logit null): MVP
  median S1 0.025 (max 0.20), 0 flips; no invented similarity. G2 (pair corr .97): MVP S1
  0.961, 0 flips; RFC-S 0.640, 14 flips. RFC-S's S1 is capped at 2/3 by its form. One G2
  replicate ran away in scale (×18; σ_ν-only normalisation), scale-free outputs fine.
- **T1 RFC-G and scale fix (2026-10-08), [RESULTS.md](analysis/T1/RESULTS.md).** RFC-G (per-
  attribute σ) doesn't close the gap: S1 0.62/0.71 on G1/G2, 40 flips of 120 (RFC-S 29, MVP
  0); holdout-MAE tuning doesn't identify the error structure. MVP refit with total error
  variance fixed: no runaways, σ recovered (G1 0.40/0.83/0.86 vs 0.36/0.86/0.86), S4
  0.90–1.00 except G2 brand 0.75 (σ near 0). Use the total-variance normalisation going
  forward. T1 verdict: PASS (MVP better) with caveats.
- **Kill-rule blanks for T1** left unset: the user chose "PASS with caveats" (2026-10-08).
- **Open (T1 code):** `run.py` still fits MVP with σ_ν pinned; the total-variance version is
  in `mvp_refit.py` / `ghk_gpu.sig2_of(tvar=...)`. Fold it into `run.py` before reuse in T4.

## Baselines

- Speed is always measured at matched error on the decision-layer output, as
  wall-clock time to decision. Never against QMC Sobol alone.
- If a baseline has no usable implementation, stop and ask before writing
  one.
- The Sawtooth proxy must be faithful: tuned RFC, own- and cross-effects, and
  coded subsets chosen as an analyst would from counts and lift. Every
  deviation from Sawtooth practice is documented.

## Prior work in this repo (inputs, not conclusions)

- `tests/rfc_vs_probit/`: near-duplicates, oracle utilities. Tuned
  RFC-style S2 had 5.6–6.4 points of sibling source-of-volume error against
  1.6–3.2 for the probit kernel. No decision flips for S2. Relevant to T1.
- `tests/joint_menu/`: pick-any menu, k = 6. Independent logits picked the
  wrong bundle under correlated truth; MVP and coded patterns did not.
  Relevant to T2.
- `analysis/rho_test/`: dunnhumby pipeline, 8 items, GO on pasta × pasta
  sauce (ρ_B = 0.70). The data source for T3.
- `analysis/joint_vs_proxy/run.py`: pending head-to-head (coded subsets
  C1 = 3 items, C2 = 2 items, vs MVP), no results yet. Absorbed by T3.
- `research/similarity_kernel/`: kernel on product error, one replicate run.

## Environment and baselines (2026-10-06)

- R 4.3.3 and bayesm 3.1-6 installed (Ubuntu packages). HB-MNL proxy for
  T1/T4: `bayesm::rhierMnlRwMixture`, a documented deviation from Sawtooth
  CBC/HB. Gibbs for T2/T4 can use `bayesm::rmnpGibbs` / `rmvpGibbs`.
- No local GPU. GHK and MACML have no ready implementation here; each is
  raised at its test's design step before any code is written.
- The "σ about 4×" figure is unverified and is not used as a reference.
