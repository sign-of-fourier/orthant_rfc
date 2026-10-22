# Where the advantage is

Last updated 2026-10-22 (through T27 and its follow-up). Maintained alongside [PLAN.md](PLAN.md); each test's full write-up is in
`analysis/T<n>/TEST.md`.

The project is about scale: unlocking problems that are too big or too slow for the incumbents, (1) with the
engine and (2) with the GPU. This page lists only the cases where we have an advantage. Speed claims are against
the right incumbent: the GPU probit pipeline against HB-MNL + RFC (logit IIA), the engine against OVUS (the other
analytic orthant approximation). Errors are absolute (probability or share points); decisions are compared in
regret points or flipped decisions against a known truth.

**Enabled** means the technique was impractical before (too slow, too costly, or impossible for the incumbent's
method) and is now practical, and it beats the incumbent at something.

**Source** says what the advantage comes from:

- **Model**: the probit model itself (MVP for pick-any, MNP for single choice), whose correlated errors represent
  near-duplicates and linked items that logit IIA and coded subsets cannot.
- **GPU**: the probit fit on a GPU (GHK likelihood in torch on Modal), which makes a slow fit practical.
- **Engine**: the analytic orthant approximation, used for scoring and screening decisions, and (exploratory, T27)
  as the likelihood of a cross-sectional MNP fit. Panel likelihoods with respondent random effects stay with GHK.

## At a glance

| problem | source | incumbent: time / error | ours: time / error | enabled | beats the incumbent at |
|---|---|---|---|---|---|
| **MBC beyond the coded-subset cap**, k = 6–30 linked items | Model | coded subsets: combinations grow as 2^k, capped near k = 5. Flips 146 of 240 decisions at k = 6–12 (T2), 207 of 240 at k = 20–30 (T14) | MVP by IFM, CPU, seconds. Flips 28 of 240 (T2), 50 of 240 (T14) | **yes** | decisions: 118–157 fewer flips of 240 (sim); real holdouts back MVP 4–0 (T3, 8 items) and 3–0 (T14, 24 items); real takeaway baskets, 21 groups vs coded subsets over all of them: +0.11 to +0.36 nats per order, top-10 bundles 0.16–1.31 pts closer (T21) |
| **Shelf CBC, multinomial probit**, J = 20, N = 1,000, 12 tasks | Model + GPU; engine screens the decision search | HB-MNL + RFC-S, CPU: 3–4 min | MNP before T18: 4.5 h (L4), 19–27 min (H100). After T18: **0.35–2.2 min on an H100**, $0.02–0.15 per fit | **yes** (T18) | assortment regret 1.85 pts lower, 95% CI [1.30, 2.40] (T15, sim, 5 datasets). Real shelves (Trivago, J = 25, 20k-user panel, T24): no error correlation beyond tastes; MVP ties a panel mixed logit (±0.007 nats per choice) |
| **Small CBC**, J ≤ 6, mixed MNP | Model | HB-MNL + RFC-S: about 5% cost regret at every N from 50 to 1,500, a bias floor that more respondents do not lower | MVP fit 10–30 s on a T4 at N = 50–200; 0.2–1.6% cost regret already at **N = 50** | — | decisions at any sample size: RFC-S at N = 1,500 is still worse than MVP at N = 50, so the sample-size ratio is unbounded (> 30× at N = 50, grid-limited) (T20, sim); recovers σ and the error correlation that RFC gets wrong by construction (T1). Real stated choice (Swissmetro, J = 3, T22): MVP best holdout log-likelihood at N = 100–468 but by only 0.001–0.043 nats per choice over RFC-S; no error correlation in that data, so no floor |
| **Dependence across subsets in decision search** (choose m of N: baskets in MBC, assortments in shelf CBC) | Model + engine | the analyst declares groups before the fit (coded subsets in MBC; nests, or RFC's attribute errors, in CBC) and independence across them; each candidate is then a cheap closed form | the full error correlation is estimated, so no grouping is declared; each candidate is one orthant per item or pattern, and the engine makes brute-force search over candidates practical (3.6–20× OVUS, T15; about 500× QMC, T2) | — | decisions on candidates that span groups: MBC T2/T14 (118–157 fewer flips of 240); real MBC (T21, takeaway baskets): coded subsets equal independence on pairs that span blocks, where MVP is 0.13–0.27 pts better; inside a block coded subsets are 0.03–0.24 pts better. CBC vs RFC T15/T20. Not vs a structural nested logit (untested); a cross-effects logit that estimates every pair tied MVP (T19) |
| **Orthant screening in decision search** (population shares, dim = shelf size) | Engine | OVUS | engine **3.6–20× faster** at equal decision adequacy, gap growing with dimension (T15) | — | speed. Real d = 24 shelf shares (Trivago, T24): "high" 2.4× OVUS with lower median \|ΔP\| (0.0011 vs 0.0036); "highest" 0.00024, the only method with max share error under 0.005 |
| **Scoring pattern probabilities** on a near-singular Σ, d = 8–16 | Engine | OVUS: 1.8–3.2e-4 s/row, median \|ΔP\| 0.0007–0.0015. SciPy: 1.1–1.8 s/row | engine "high": **3–6e-5 s/row**, median \|ΔP\| 0.0034–0.0048; "highest": 0.9–2.0e-4 s/row, 0.0022–0.0029 | — | speed: 5–6× OVUS, 28,000–37,000× SciPy ([mvprobit benchmarks](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/benchmarks.md); engine times remeasured 2026-10-19). **Real d = 21 (T21, takeaway basket patterns, median P 5e-5): 3× OVUS, but median \|ΔP\| 1.2–1.5e-5 vs OVUS 0.4–1.5e-6, and "high" returns P = 0 on 4 of 200: fails the accuracy guard for likelihood scoring** |
| **Few high-dimensional decision queries** (MBC) | Engine | QMC | engine about **500× faster** (T2) | — | speed |
| **Brand-side line and build-and-place search** (GA over lines of up to 6 in 5 store clusters; orthants of d = 18–31) | Model; engine as the GA's evaluator | logit (LC-MNL share of preference): 0.91 / 2.32% regret at 300 s (est / stress truth). RFC on 8 CPU cores: 2.49 / 3.60% | MNP, engine on a T4: **0.35 / 0.75%** at 300 s; 0.35 / 1.87% already at 60 s, where GHK on 8 cores is at 2.08 / 4.50% | — | decisions (T25, dunnhumby soft drinks, 2 seeds, every gap under 3 $ per 1,000 trips). Logit keeps more near-duplicate client SKUs on the same shelf: cannibalization 15–31% vs 13–25% under the stress truth |
| **MNP likelihood evaluation** (one choice orthant per row, d = J − 1 = 4–64) | Engine | OVUS (MACML's approximation), same CPU core: 104 µs → 1.4 ms per row, median \|ΔP\| 5e-4 → 2.4e-3; 90th-percentile \|Δlog P\| 0.011 → 0.23 | the release engine ("highest"): **49 µs → 1.4 ms**; the fast engine (private prototype): **~45 µs → ~2.0 ms**; both median \|ΔP\| 2e-4 → 4e-5, 90th-percentile \|Δlog P\| ≤ 0.007 at every d | — | speed: the release engine 1.5–2.1× OVUS up to d = 32 (equal at 64); the fast engine ~2.3× OVUS at d = 8, ~1.2× at 32, ~1.5× slower at 64. Lower error at every d for both. GHK needs 6–23× the engine's time for the same accuracy from d = 16; SciPy 1.9 s per row at d = 64. Fit bias against the known truth (J = 10, a near-substitute nest at r = 0.97, none declared): fast engine 0.42 SE worst parameter, OVUS 0.48, release engine 1.05; part-worths ≤ 0.15 SE for all three. An engine fit (J = 5) is within 0.28–0.33 SE of the GHK MLE (T27, sim, exploratory) |

## CBC

### Shelf CBC with near-duplicates, J = 20 (enabled)

- **Source.** The model wins the decisions; the GPU makes the fit practical; the engine speeds up the decision
  search that follows. The engine is not used in the fit: on the 240-dimensional panel orthant it is about 5 nats
  per respondent off (T18 side check).
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

- **Source.** Model + GPU for the fit; the engine for the search.
- **Speed/scale.** MNP fit 2.2 / 6.5 / 14 min for J = 20 / 40 / 60 on an H100 (J = 40 from timed evaluations), while
  each respondent's orthant grows from 240 to 720 dimensions. The full T4 study (2 truths × 3 shelf sizes × 10
  replicates) drops from about $600 to about $35. HB-MNL at J = 60 is not measured (estimated about 10 min on CPU).
- **Why it should win.** More SKUs mean more near-duplicates for IIA to mishandle, and the search space grows
  combinatorially, which is where the engine's 3.6–20× screening speed over OVUS pays.
- **Open.** One J = 60 recovery miss, likely that dataset; a second dataset settles it before the T4 run.

### Brand-side line and build-and-place search (T25)

- **Source.** Model; the engine is the GA's evaluator.
- **Speed/scale.** Same GA, same wall-clock budget, only the share evaluator differs. Shelves of 17–25 competitors
  plus a line of up to 6 give orthants of d = 18–31. In 60 s the engine (T4) completes about 18k shelf evaluations
  and 600 generations; GHK on 8 CPU cores completes 11k and about 50.
- **Win.** Regret at 300 s, estimated / stress truth: engine 0.35 / 0.75%, logit 0.91 / 2.32%, RFC (CPU) 2.49 /
  3.60%. The engine leads GHK early (0.35 vs 2.08% at 60 s on the estimated truth); GHK ties by 300 s on the
  estimated truth and is still behind on stress (1.52%). Logit and RFC are limited by the model, not the hardware:
  RFC on a T4 lands where RFC on CPU does.
- **Caveats.** Two seeds (ranges, no CIs). Every gap is under 3 $ per 1,000 trips. The estimated truth has Σ = I,
  so its probit-vs-logit gap is the error distribution only; the stress truth is assumed.

### Small CBC, J ≤ 6

- **Source.** Model only; at this size the fit is not the bottleneck. At N = 200, MVP takes 23–29 s on a T4 GPU, against
  HB-MNL 25–28 s plus RFC-S tuning 9–18 s on one CPU core (T20).
- **Sample size.** HB-MNL + RFC-S has a cost-regret floor of about 5% (fitted 4.95 and 5.02 on the two correlated
  truths) that does not fall between N = 50 and 1,500. MVP is at 0.2–1.6% from N = 50, 3–5 points better on the
  same datasets at N = 50–200, with every CI above 0. There is no crossing, so no finite "fewer respondents"
  multiplier: RFC-S at N = 1,500 is still worse than MVP at N = 50. Under a logit truth neither method has an edge
  (T20, sim, R = 10; supersedes T5's two-point reading, whose "N ≥ 720" was a pass threshold).
- **The floor is not a tuning artefact.** RFC-S's tuning often pins the attribute σ at zero, which collapses it to
  plain HB logit. Wider bounds give identical decisions. A generous per-attribute RFC (not Sawtooth practice)
  narrows the gap on the strong-similarity truth only. Even RFC tuned to the true shares of the decision scenarios
  stays at 3.7–4.0% against MVP's 0 (T20 floor check, 5 replicates).
- **Not yet tested against a structural incumbent** fitted to the choices (nested or cross-nested logit). In MBC,
  the structural cross-effects logit tied MVP (T19).
- **Explanation.** RFC's σ is wrong by construction (S1 capped at 2/3, tuned on share fit), so its "why" is wrong
  even when its shares are right. MVP recovers σ and the error correlation (T1).

## MBC (pick-any menus, linked items)

### Beyond the coded-subset cap, k = 6–30 (enabled)

- **Source.** Model. The fit is on CPU in seconds; the engine scores the high-dimensional decision queries.
- **Speed/scale.** Coded subsets enumerate 2^k combinations and Sawtooth caps them near k = 5, so they cannot represent
  a linked set past that. MVP by IFM fits on CPU in seconds at k = 6–12 and stays in seconds to minutes at k = 20–30.
- **Win.**
  - k = 6–12: 28 vs 146 decision flips of 240 (T2, sim); the real dunnhumby holdout (8 items) backed MVP on 4
    disagreements, coded subsets on 0 (T3).
  - k = 20–30: 50 vs 207 flips of 240 (T14 Part A, sim); a real 24-item baking cluster backed MVP 3 to 0 (T14 Part B).
- **Untested.** The same comparison against an HB coded-subset incumbent (Sawtooth fits MBC by HB).

## Across subsets (CBC and MBC)

### Dependence across subsets in decision search

- **Source.** Model + engine. The incumbents keep each candidate cheap by declaring independence:
  - coded subsets in MBC are exact within a block of ≤ 5 items and independent between blocks;
  - a nested logit allows one tree of nests, with no correlation across nests;
  - RFC adds one attribute-structured error and is tuned on holdout shares.

  The grouping is chosen before the fit, and every later decision search inherits it.

  MVP and MNP estimate the error correlation instead, so a candidate that spans groups is still scored with the
  dependence intact. Scoring costs one orthant query per item (shelf shares) or per pattern (MBC). Choosing m of N
  is C(N, m) candidates, so brute force is practical only if each query is fast: that is the engine's role.
- **Speed/scale.** The engine screens 3.6–20× faster than OVUS at equal decision adequacy, rising with orthant size
  (T15). It is about 500× faster than QMC for high-dimensional MBC queries (T2).
- **Win.**
  - MBC: once a linked set exceeds the cap it has to be split across subsets, and decisions flip: 146 vs 28 of 240
    (T2), 207 vs 50 (T14 A). Real holdouts back MVP 4–0 (T3) and 3–0 (T14 B).
  - CBC: against HB-MNL + RFC-S, assortment regret is 1.85 pts lower (T15). In small CBC, RFC-S has a ~5% floor
    (T20).
- **Boundary.** An incumbent that estimates the dependence from the choices is a different case. The MBC
  cross-effects logit (210 pairwise parameters) tied MVP on real takeaway baskets (T19). The structural incumbent
  for shelf CBC, an estimated nested or cross-nested logit, has not been tested.
  - With similarity on crossed attributes (flavor and pack in T1/T5/T20), one tree can hold only one of them. A
    cross-nested logit needs the allocations declared by hand.

## Engine

- **Screening speed.** 3.6–20× faster than OVUS at equal decision adequacy in shelf search, rising with orthant size
  (T15, H100 and T4).
- **Scoring speed.** On a near-singular Σ at d = 8–16: "high" 3–6e-5 s/row (median |ΔP| 0.0034–0.0048), "highest"
  0.9–2.0e-4 s/row (0.0022–0.0029), OVUS 1.8–3.2e-4 s/row (0.0007–0.0015), SciPy 1.1–1.8 s/row. All three
  approximations sit inside a 0.005 decision floor at the median.
- **High-dimensional queries.** About 500× faster than QMC for a few high-dimensional MBC decision queries (T2).
- **MNP likelihood (T27, exploratory).** On one CPU core, the release engine ("highest") costs 49 µs → 1.4 ms per
  choice row at d = 4–64 and the fast engine (private prototype) ~45 µs → ~2.0 ms, against OVUS 104 µs → 1.4 ms.
  Both engines have lower error at every d: median |ΔP| 2e-4 → 4e-5 vs 5e-4 → 2.4e-3, and 90th-percentile
  |Δlog P| at most 0.007 vs OVUS's 0.011 → 0.23 (the likelihood sums log P). Matching their accuracy costs GHK
  6–23× the time from d = 16. An L-BFGS fit converges in 35–36 iterations and lands within 0.28–0.33 SE of the
  GHK MLE (J = 5, N = 1,000). Against the known truth at J = 10 (a nest at r = 0.97, none declared), the fast
  engine's worst fit bias is 0.42 SE, OVUS's 0.48 and the release engine's 1.05; part-worths ≤ 0.15 SE for all
  three. The release engine's near-duplicate cutoff stalled an actual L-BFGS fit at J = 10.
  - Near-substitute nests (correlation near 1) are the weak case for the release engine: up to 5 SE bias on a
    3-alternative nest at r = 0.97. A private prototype with an exact rewrite of analyst-declared nests brings every
    case tested to 0.06–0.21 SE, where OVUS reaches 5.35 SE on a 3-alternative nest at r = 0.9. Each rewritten term
    is a within-nest winner's probability, so near-substitutes stay in the model rather than being dropped.
- **Its niche.** Many high-dimensional queries in a huge search space, where speed is a sample-size multiplier at a
  fixed budget, and cross-sectional MNP likelihoods (T27). Panel likelihoods with respondent random effects stay
  with GHK (T18).
- **Boundary.**
  - Against a shared-draw simulator on an exactly enumerable shelf problem, on one core, the simulator's screen
    reaches the exact optimum in 68–75 s; the engine's screen takes about 200 s or stalls 0.26% short (T26). In a
    GA under a fixed wall-clock budget on a T4, the engine ended at least as good as the simulator, which stalls
    on sampling noise (T25).
  - Sign-frustrated and low-rank small-P matrices (MVP-like patterns) can come back at P ≤ 0, where OVUS and GHK do
    not (T21, T27). MNP choice orthants are mostly positive.

## Candidates (untested)

Cells we expect to show an advantage but have not tested. Listed only where the expected advantage is medium or
higher.

| cell | source | expected advantage | what would test it |
|---|---|---|---|
| **HB-MVP for stated MBC** (respondent random effects) | Model + GPU | Medium–high. The T2/T14 wins should carry over, and individual pick-any patterns are no longer pooled. The per-respondent panel likelihood is the same compute problem T18 solved for MNP (one orthant per respondent, tastes integrated analytically) | T14-style simulation with respondent heterogeneity, against HB coded subsets |
| **Real stated MBC** (designed menus, not baskets) | Model | Likely the T14 pattern. Menus are designed and prices randomised, so cleaner evidence than revealed baskets | a real stated MBC dataset with a holdout |
| **Shelf CBC vs an estimated nested / cross-nested logit** | Model | Medium. A single nest tree cannot hold similarity on crossed attributes; a cross-nested logit can, but its allocations are declared by hand, which is the across-subsets problem again. The T19 tie says an incumbent that estimates the dependence can match MVP | T20 or T15 data with a nested logit (one tree per attribute) and a cross-nested logit fitted to the choices |
| **MNP fit with the engine as the likelihood**, large J | Engine | Medium. T27 beats OVUS on one CPU core at d = 4–64 and lands within 0.3 SE of the GHK MLE at J = 5; against the known truth at J = 10 the fast engine matches OVUS (0.42 vs 0.48 SE) with no nests declared, while the release engine stalls in an actual fit near its cutoff. Beyond J = 5 the evidence is predicted bias, not completed fits (the J = 10 fit was abandoned). Untested at scale, on a GPU, on real data, or against a full MACML fit's wall time | a T27-style fit at J = 20–60 against MACML (OVUS) and GHK, time to the same SE-level accuracy |
| **Near-clone line extensions** (a new SKU nearly identical to one on the shelf; cannibalization within the family) | Model + engine | Medium–high. This is the near-1 nest case: the release engine reached 5 SE bias on a 3-alternative nest at r = 0.97, and the prototype with declared nests 0.14 (T27). Each rewritten term is a family member's win probability against the rest of the shelf, so within-family cannibalization is computed directly instead of being inflated into σ (RFC) or dropped by the analyst | a T15 / T25-style shelf with clone SKUs at ρ = 0.95–0.99: cannibalization and line-decision regret for the release engine, the prototype, OVUS and GHK, against the truth |
| **Consistent screening across assortments with near-duplicate families** | Engine | Medium. With its fixed near-duplicate cutoff, the release engine changes how it treats a family when a candidate shelf adds or drops a near-duplicate, so neighbouring candidates can be scored by different approximations; the prototype has no cutoff and costs the same (0.88–1.46×, T27) | a T15 / T26 screen on shelves with near-clone families: ranking flips against GHK, release engine vs prototype vs OVUS at equal time |
| **Gradient-based price and margin optimization on near-duplicate shelves** | Engine | Medium. Shares from the prototype are smooth in prices and correlations, so an optimizer can use exact gradients. The release engine's shares have jumps around near-duplicate families (T27 scans), and OVUS's variable ordering can switch as prices move. Caveat: the prototype still makes one discrete internal choice, which flipped under tiny parameter changes on rewritten nest terms (T27 follow-up); on shelves with negative correlations it could flip as prices move | L-BFGS on shelf margin over prices with near-duplicates: time to the optimum and regret, prototype vs release engine vs OVUS vs GHK |
| **Standard errors for an MNP fit** | Engine | Medium. Clients want SEs. A deterministic, smooth likelihood gives a Hessian without simulation noise; with GHK the SE pass can dominate the fit's time (about 18 of 20 min at d = 12 for MVP). Caveat: the one J = 10 prototype fit ended in an undiagnosed line-search failure (finite-difference gradients), so smoothness at the Hessian scale is unverified | SE coverage and time to SEs, prototype engine vs GHK sandwich vs MACML, on a T27-style MNP |
| **HB-MNP / HB-MVP by Gibbs** (fully Bayesian) | GPU | Medium. Gibbs data augmentation is truncated-normal draws, which parallelise on a GPU; the model case is the same as the frequentist fits above | timing against a CPU Gibbs sampler at J = 20 |

## The common thread

Every advantage here is in a case practitioners can't keep small:
- the coded-subset cap binds;
- the analyst has to declare independence across groups to keep each candidate cheap;
- or the shelf and its decision space are too big.

That is also where the GPU fit (T18) and the engine's speed matter most. The advantage holds against incumbents
that bolt similarity on or declare it away. Against one that estimates the dependence from the choices, it has
so far been a tie (T19).
