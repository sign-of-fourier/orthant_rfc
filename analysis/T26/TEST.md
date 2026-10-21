# T26 · Tiered screens on an exactly enumerable assortment problem: engine screen vs simulator screen

Status 2026-10-21: main run done (CPU only, about $0.60 with the pilot). Results below; out/results.csv, out/convergence.png.

## Question

As algorithms, under identical execution conditions: on a decision whose exact optimum is known, how much work
does each tiered screen need to reach a solution, and how much true margin does it lose along the way?
- **C:** the engine, from cheap to accurate settings.
- **F:** a shared-draw simulator, from few to many draws.

Per tier, what does each screen throw away?

Framing note (earlier rule): the engine's speed claim is against OVUS, not simulators; the engine losing to a
simulator is acceptable. This test compares the two practical screens head to head because the user asked for it.
An OVUS arm can be added (option O1 below).

## Problem (reused from T25 v2)

- T25 decision 2 (build and place): client manufacturer 103 (17 SKUs), 5 store clusters, build at most 10 SKUs
  (incl. the top 2), each cluster's line 1-6 of the build. Objective: client margin per 1,000 trips, summed over
  clusters by trip share.
- Same segment utilities (LC-MNL with the pooled prior, K = 8-10), competitor shelves, margins and stocking rule
  as T25 v2.
- **Exact enumeration.** A solution's value is the sum of its cluster shelves' values, so:
  - score every distinct (cluster, line) shelf once: 21,777 lines x 5 clusters = **108,885 shelves**, about
    5.3M pick-one orthants per truth (K x line size per shelf, dimension 18-31);
  - for each of the 6,435 size-10 builds, take each cluster's best line among the build's subsets (about 27M
    lookups, seconds on CPU). Size 10 is optimal: a bigger build never hurts.
  - This gives each screen's chosen solution and the exact optimum under the truth.

## Truths (two)

- **rho 0.5 (T25 "stress", renamed "moderate"):** rho = 0.5 within near-duplicate pairs, 0.25 within
  manufacturer, otherwise 0. The difference covariances have no negative correlations.
- **rho 0.8 (new, "high"):** 0.8 within near-duplicates, 0.25 within manufacturer. Negative correlations appear
  in the difference covariances (two near-duplicates of the target, weakly correlated with each other:
  0.25 - 0.8 - 0.8 + 1 < 0).
- T25's estimated truth (Sigma = I) is left out: no correlation structure to screen on.

## Reference (exact optimum under each truth)

1. An unbiased shared-draw simulator with 2^18 shoppers per segment, competitor collapse, on all 108,885 shelves.
   Its SE on a 1% share is about 0.0002.
2. Shortlist the top 500 solutions by value, plus every solution within 4 SE of the best.
3. Rescore their shelves with GHK, Genz ordering and R = 20,000.
4. Best = the reference optimum. Report the gap to the 501st solution in SEs as evidence it is exact.

The engine is never the reference: at rho 0.8 it is one of the arms under test.

## Arms (only the scorer differs; same enumeration and lookup)

### C · engine screen (user's design)

| tier | scorer | candidates |
|---|---|---|
| 1 | resolution "low", float32, a preprocessing step skipped | all 108,885 shelves |
| 2 | resolution "high", the same step skipped | survivors of tier 1 |
| 3 | resolution "highest", full | survivors of tier 2 |
| 4 | QMC (GHK, Genz, R = 20,000) | finalists |

- Sizing goes backwards from the budget: the final QMC tier gets about 20% of the time budget, which fixes the
  number of finalists; earlier keeps follow from the rules below.
- Screening here acts on solutions, not shelves. After each tier, the per-build lookup ranks solutions, and the
  survivors are the shelves of the kept solutions (a shelf survives if any kept solution uses it).

### F · simulator screen

| tier | scorer | candidates |
|---|---|---|
| 1 | 256 shoppers per segment | all 108,885 shelves |
| 2 | 4,096 | survivors |
| 3 | 65,536 | survivors |
| 4 | QMC (GHK, R = 20,000) | finalists |

- Fresh draws per tier (independent of the reference's draws). Competitor collapse throughout: each shopper's
  best competitor is precomputed, and a shelf reads only its 1-6 line columns.
- Same sizing and keep rules as C.

### A · logit (model-regret baseline)

- LC-MNL closed form on all shelves, CPU, one tier. Shows what ignoring correlation costs at each rho.

### Options (not in the base cost)

- **O1:** an OVUS tiered arm (the engine's stated competitor).
- **O2:** a GPU rerun of both screens, with the same algorithms and a hardware note only, if the user later wants
  it.

## Keep rules (Claude's, user 2026-10-20: "use Claude's thumbs"), frozen after the pilot

1. **Error band:** keep every solution within 3 sigma of the tier's best. Sigma = the SD of (tier value - QMC
   value) among top solutions, measured in the pilot. It is measured near the top, because uniform bias does not
   change rankings.
2. **Cost cap:** a tier may cost at most about 1/3 of the previous one. If the band asks for more, keep what the
   cap allows and flag "recall at risk".
3. **Final tier:** about 20% of the budget; at least 20 finalists.
- Starting guess: about 3% -> 15% -> about 100 finalists.

## Pilot (calibration only; frozen before the main run)

- About 2,000 solutions per truth: 1,000 random plus 1,000 from logit-greedy with perturbations (to cover the
  top).
- Score every tier and QMC. Record sigma near the top, cost per candidate per tier, and rank agreement in the top
  50.
- Set the keeps; write them into this file before the main run.

## Apples to apples (user 2026-10-20: "apples to apples as an algorithm; I don't want to test the power of the T4")

- **Same execution model for both screens:** one CPU core, numpy, float64, the same batch size per call, no GPU,
  no process pool. Parallel scaling and GPU fit are hardware questions, not algorithm questions, and stay out.
  - The engine runs the library's numpy build at the tier's settings.
  - The simulator runs plain numpy argmax.
- **Same algorithmic effort:** each screen gets its best known algorithmic form, applied the same way.
  - The simulator's competitor collapse is in, because it changes the operation count, not the hardware use.
  - Any engine-side reuse across targets that changes the operation count is in too.
  - Both read the same precomputed per-cluster inputs (utilities, Sigma, competitor sets); setup is timed
    separately and reported.
- **Cost is counted two ways:**
  1. Single-core CPU seconds: the same machine for both, so the ratio is meaningful even though absolute times
     are machine-specific.
  2. Hardware-neutral operation counts per tier, from each algorithm's formulas: normal CDF / inverse-CDF
     evaluations, multiply-adds and comparisons. These show where the time goes and survive a hardware change.
- **Budgets are in work, not seconds:** each budget is a single-core CPU-second allowance (e.g. the time one full
  order-1 pass takes, and 1/4 of it), identical for both screens.
- **Same final tier:** both screens finish with the same QMC scorer (GHK, Genz ordering, R = 20,000), so only the
  screening tiers differ.
- **The reference is not part of the comparison:** it may run on any hardware (GPU or many cores), since it is
  computed once and never timed.

## Runs

- **Full span only:** every screen starts from all 108,885 shelves (no sampled pools; user 2026-10-20).
- **Work-budget sweep, per truth (2):** each screen runs at 6 work budgets, each sized backwards. Unit W = the
  single-core time of one full order-1 pass over all shelves (measured in the pilot); budgets = W/16, W/8, W/4,
  W/2, W, 2W. Plus one unconstrained run.
  - Each budget gives one point: (work used, true regret of the pick). Together they make the convergence curve.
  - Per-tier picks inside each run give extra points.
- **Repeats:**
  - C is deterministic: 3 timing repeats, median reported.
  - F: 3 draw seeds.
- A (logit) runs once per truth for model regret; it is not timed against the screens.

## Deliverable and verdict (user 2026-10-20)

- **No pass/fail thresholds.** The verdict is read off the convergence curves: the pick's true regret ($ per
  1,000 trips and % of the exact optimum) vs work (single-core seconds; operation counts on a second axis), one
  line per screen, one panel per truth, with ranges over repeats/seeds.
  - **Option 1, work to target:** where each curve first stays at or below a regret line. Reported for 0.5%,
    0.1% and the exact optimum.
  - **Option 2, regret at equal work:** the vertical gap between curves at each budget.
- **Reported alongside:**
  - recall per tier: is the exact optimum still kept? If not, the true regret of the best survivor;
  - time and operations per tier;
  - A's model regret at each rho.

## Cost estimate (to verify in the pilot)

| step | estimate |
|---|---|
| Pilot (both truths, single core) | minutes of CPU, < $0.05 |
| Reference (2 truths): simulator 2^18 x 108,885 shelves + GHK on the top 500 (any hardware, untimed) | ~30-60 GPU-min, ~$0.30-0.60 (most uncertain) |
| C and F screens on single CPU cores: unconstrained + 6 budgets x 3 repeats x 2 truths. About 5.3M order-0 orthants per full pass, ~10-20 min single-core; runs in parallel containers | ~$0.60-1.20 CPU |
| A, enumeration lookups, summaries (CPU) | ~$0.10 |
| **Total** | **~30-60 GPU-min plus about $1-1.50 CPU, about $1.10-2.10** |

- Proposed caps: 60 GPU-min (reference only), $5 in total.

## What carries over from T25

- Code: T25's evaluators, truths and runners. The engine-dependent scripts (T25's evaluators and runners,
  and this test's driver) are kept out of the public repo.
- New code:
  - the enumeration plus per-build lookup;
  - the tiered-screen drivers;
  - competitor collapse for the simulator;
  - the rho 0.8 truth.
- Ledger lesson: book GPU time as container time, not spawn -> sequential collection.

## Results v1 (2026-10-21, superseded; out/v1/)

- Survivors carried only each build's best line, so no screen reached the exact optimum. Final regret was 0.09-0.32%.
- Budgets came from the contended pilot (about 3x too large).

## Results v2 (2026-10-21): top-3 lines per cluster carried; fixed budget grid 10-640 single-core s

- **Reference optimum:** 58.44 (rho 0.5), 57.94 (rho 0.8) $ per 1,000 trips.
- Table: true regret % of the final pick (mean over 3 seeds), with picks at the exact optimum in brackets.

| budget (s) | C, rho 0.5 | F dense, rho 0.5 | F collapsed, rho 0.5 | C, rho 0.8 | F dense, rho 0.8 | F collapsed, rho 0.8 |
|---|---|---|---|---|---|---|
| 10 | 8.06 | 4.05 | 1.67 | 6.55 | 4.13 | 0.75 (2/3) |
| 20 | 5.70 | 1.75 | **0 (3/3)** | 5.93 | 3.19 | 0.12 (2/3) |
| 40 | 3.83 | 0.54 | 0 (3/3) | 3.70 | 0.90 | **0 (3/3)** |
| 80 | 2.06 | **0 (3/3)** | 0 | 2.86 | **0 (3/3)** | 0 |
| 160 | 1.07 | 0 | 0 | 1.28 | 0 | 0 |
| 320 | 0.38 | 0 | 0 | 0.36 (2/3) | 0 | 0 |
| 640 | 0.26 (0/3; done at about 220 s) | 0 | 0 | **0 (3/3)**, done at about 200 s | 0 | 0 |

- **Time to the exact optimum in all 3 seeds:**
  - rho 0.5: F collapsed about 20 s, F dense about 75 s. C never: it stalls at 0.26% (0.15 $/1k trips) with the
    wrong build set.
  - rho 0.8: F collapsed about 35 s, F dense about 68 s, C about 200 s.
- **Where the engine's time goes:** the "low" tier over all 108,885 shelves takes about 150-200 s on one core. That
  tier alone costs more than the simulators' whole screens.
- The line fix removed the floor: F reaches the exact optimum in every complete run, and C does at rho 0.8.
- Spend: v2 about $0.21. T26 in total about $0.80, all CPU.
