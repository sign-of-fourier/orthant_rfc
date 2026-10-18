# Where the advantage is

Last updated 2026-10-18 (through T18). Maintained alongside [PLAN.md](PLAN.md); each test's full write-up is in
`analysis/T<n>/TEST.md`.

The project is about scale: unlocking problems that are too big or too slow for the incumbents, (1) with the
engine and (2) with the GPU. This page lists only the cases where we have an advantage. Speed claims are against
the right incumbent: the GPU probit pipeline against HB-MNL + RFC (logit IIA), the engine against OVUS (the other
analytic orthant approximation). Errors are absolute (probability or share points); decisions are compared in
regret points or flipped decisions against a known truth.

**Enabled** means the technique was impractical before (too slow, too costly, or impossible for the incumbent's
method) and is now practical, and it beats the incumbent at something.

## At a glance

| problem | incumbent: time / error | ours: time / error | enabled | beats the incumbent at |
|---|---|---|---|---|
| **MBC beyond the coded-subset cap**, k = 6–30 linked items | coded subsets: combinations grow as 2^k, capped near k = 5. Flips 146 of 240 decisions at k = 6–12 (T2), 207 of 240 at k = 20–30 (T14) | MVP by IFM, CPU, seconds. Flips 28 of 240 (T2), 50 of 240 (T14) | **yes** | decisions: 118–157 fewer flips of 240 (sim); real holdouts back MVP 4–0 (T3, 8 items) and 3–0 (T14, 24 items) |
| **Shelf CBC, multinomial probit**, J = 20, N = 1,000, 12 tasks | HB-MNL + RFC-S, CPU: 3–4 min | MNP before T18: 4.5 h (L4), 19–27 min (H100). After T18: **0.35–2.2 min on an H100**, $0.02–0.15 per fit | **yes** (T18) | assortment regret 1.85 pts lower, 95% CI [1.30, 2.40] (T15, sim, 5 datasets) |
| **Orthant screening in decision search** (population shares, dim = shelf size) | OVUS | engine **3.6–20× faster** at equal decision adequacy, gap growing with dimension (T15) | — | speed |
| **Scoring pattern probabilities** on a near-singular Σ, d = 8–16 | OVUS: 1.8–3.2e-4 s/row, median \|ΔP\| 0.0007–0.0015. SciPy: 1.1–1.8 s/row | engine "high": **4–8e-5 s/row**, median \|ΔP\| 0.0034–0.0048; "highest": 0.9–2.1e-4 s/row, 0.0022–0.0031 | — | speed: 4–5× OVUS, 21,000–28,000× SciPy ([mvprobit benchmarks](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/benchmarks.md)) |
| **Few high-dimensional decision queries** (MBC) | QMC | engine about **500× faster** (T2) | — | speed |
| **Respondents needed** (CBC, similarity truths) | HB-MNL + RFC-S: N ≥ 720 | MVP: N = 150 for the same decision error (T5, sim) | — | sample size |

## CBC

### Shelf CBC with near-duplicates, J = 20 (enabled)

- **Speed/scale.** The multinomial probit fit used to take 4.5 h on an L4 (T4) or 19–27 min on an H100 (T15),
  against 3–4 min for HB-MNL on CPU. T18 replaced the nested likelihood with one Genz-ordered GHK orthant per
  respondent: 0.35 min (1k draws) to 2.2 min (16k draws) on an H100, $0.02–0.15 per fit, 1.5–10× faster than HB
  in wall time.
- **Win.** Assortment regret 1.85 pts lower than HB-MNL + RFC-S, 95% CI [1.30, 2.40] (T15, sim). Under a logit
  truth it ties or slightly beats the incumbent, so it does not invent similarity.
- **Accuracy guard.** T18's fit recovers the true parameters (joint Wald p 0.85–0.88); with non-normal tastes a
  Gaussian copula keeps it to one orthant with share error within 0.014 pts of the exact arm.
- **Evidence:** simulation. Real shelf data is the open gap.

### Large shelves, J = 40–60+ and big decision spaces (enabled, win not yet tested)

- **Speed/scale.** MNP fit 2.2 / 6.5 / 14 min for J = 20 / 40 / 60 on an H100 (J = 40 from timed evaluations), while
  each respondent's orthant grows from 240 to 720 dimensions. The full T4 study (2 truths × 3 shelf sizes × 10
  replicates) drops from about $600 to about $35. HB-MNL at J = 60 is not measured (estimated about 10 min on CPU).
- **Why it should win.** More SKUs mean more near-duplicates for IIA to mishandle, and the search space grows
  combinatorially, which is where the engine's 3.6–20× screening speed over OVUS pays.
- **Open.** One J = 60 recovery miss, likely that dataset; a second dataset settles it before the T4 run.

### Explanation

- RFC's σ is wrong by construction (S1 capped at 2/3, tuned on share fit), so its "why" is wrong even when its shares
  are right. MVP recovers σ and the error correlation (T1). A model advantage with no speed cost at small J.

### Sample size

- MVP at N = 150 matched the decision error HB-MNL + RFC-S needs N ≥ 720 for, on both correlated truths (T5, sim).

## MBC (pick-any menus, linked items)

### Beyond the coded-subset cap, k = 6–30 (enabled)

- **Speed/scale.** Coded subsets enumerate 2^k combinations and Sawtooth caps them near k = 5, so they cannot represent
  a linked set past that. MVP by IFM fits on CPU in seconds at k = 6–12 and stays in seconds to minutes at k = 20–30.
- **Win.**
  - k = 6–12: 28 vs 146 decision flips of 240 (T2, sim); the real dunnhumby holdout (8 items) backed MVP on 4
    disagreements, coded subsets on 0 (T3).
  - k = 20–30: 50 vs 207 flips of 240 (T14 Part A, sim); a real 24-item baking cluster backed MVP 3 to 0 (T14 Part B).
- **Untested.** The same comparison against an HB coded-subset incumbent (Sawtooth fits MBC by HB).

## Engine

- **Screening speed.** 3.6–20× faster than OVUS at equal decision adequacy in shelf search, rising with orthant size
  (T15, H100 and T4).
- **Scoring speed.** On a near-singular Σ at d = 8–16: "high" 4–8e-5 s/row (median |ΔP| 0.0034–0.0048), "highest"
  0.9–2.1e-4 s/row (0.0022–0.0031), OVUS 1.8–3.2e-4 s/row (0.0007–0.0015), SciPy 1.1–1.8 s/row. All three
  approximations sit inside a 0.005 decision floor at the median.
- **High-dimensional queries.** About 500× faster than QMC for a few high-dimensional MBC decision queries (T2).
- **Its niche.** Many high-dimensional queries in a huge search space, where speed is a sample-size multiplier at a
  fixed budget.

## The common thread

Every advantage here is in a case practitioners can't keep small, either because the coded-subset cap binds or
because the shelf and its decision space are too big. That is also where the GPU fit (T18) and the engine's speed
matter most.
