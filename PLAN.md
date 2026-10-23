# Plan: where multivariate probit beats incumbent choice methods

Status 2026-10-21: T1–T3, T5, T6, T14 PASS. T7, T8, T9, T11 FAIL. T13 KILL. T10 Part A inconclusive, Part B
paused. T4 paused. T15: D1 PASS / D4 SOFT FAIL, S0 guard pass. T16 FAIL (T6's win is heterogeneity). T17 stopped (power check FAIL). T18 PASS (A and B). T19: MVP ≫ independence, ≈ cross-effects logit (no win). T20–T22, T24:
see Open. T23 has no write-up yet. T25: MNP evaluators beat logit and RFC. T26: the simulator screen beats the
engine screen. T27 (exploratory): the engine as an MNP likelihood beats OVUS; near-1 nests need a fix.

- **Each test's full write-up** (design, kill criteria, user decisions, results) is in `analysis/T<n>/TEST.md`.
  T1–T3 are in their own DESIGN.md / RESULTS.md, with the original plan and history in [HISTORY.md](HISTORY.md).
- **Related documents.** Model catalog: [MODELS.md](MODELS.md). Backlog: [BACKLOG.md](BACKLOG.md). Search and
  error vs speed: [SEARCH.md](SEARCH.md).
- **Rules.**
  - Kill criteria are set by the user; nothing runs until they are filled in.
  - Scope is strict: if a test suggests expanding it, stop and ask (no new arms, truths or metrics).
  - The earlier product and working notes moved to `PRODUCT_NOTES.md`. They cover machine limits, the orthant
    backend, and style rules, and still apply.

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

## Tests

### Open

- **T27 · The engine as an MNP likelihood (2026-10-21, exploratory): on one CPU core it is faster and more accurate
  than OVUS at d = 4–64; an engine fit lands within 0.33 SE of the GHK MLE; near-substitute nests need a fix.**
  - Speed: "highest" 1.5–2.1× OVUS up to d = 32, equal at 64, with lower |ΔP| at every d (OVUS 5e-4 → 2.4e-3).
    GHK needs 6–23× the engine's time for matched accuracy from d = 16; SciPy reaches 1.9 s per row at d = 64.
  - Fit (J = 5, N = 1,000): L-BFGS converges in 35–36 iterations; 0.28–0.33 SE from the GHK MLE.
  - Near-1 nests: the release engine's fixed near-duplicate cutoff makes the likelihood jump, and a 3-alternative
    nest at r = 0.97 gives 5 SE bias. A private prototype (smooth) plus an exact rewrite of analyst-declared nests
    brings every case to 0.06–0.21 SE; OVUS reaches 5.35 SE on the 3-alternative nest at r = 0.9.
  - Ornery matrices: the engine returns P ≤ 0 on sign-frustrated / low-rank small-P rows where OVUS and GHK do not;
    OVUS breaks on AR(1) chains.
  - $0, local. [Full write-up](analysis/T27/TEST.md).
- **T26 · Tiered screens on an exactly enumerable assortment problem (2026-10-21): the shared-draw simulator reaches
  the exact optimum in 68–75 s (dense) on one core; the engine screen needs about 200 s (rho 0.8) or never (rho 0.5).**
  - T25 decision 2, enumerated exactly (108,885 shelves). Same core, same enumeration; only the scorer differs.
  - The engine's cheapest tier over all shelves (150–200 s) costs more than the simulator's whole screen. Competitor
    collapse makes the simulator faster still (20–35 s), but that is a shortcut, not the algorithm.
  - About $0.80, CPU only. [Full write-up](analysis/T26/TEST.md).
- **T25 · Brand-side line and build-and-place on dunnhumby soft drinks (2026-10-20): the same GA with six share
  evaluators; MNP evaluators beat logit and RFC; the engine leads GHK early and ends at least as good as the
  simulator.**
  - Estimated Σ = I in all 7 clusters; a stress truth (ρ 0.5 near-duplicates) is assumed. 2 seeds.
  - Decision 2 regret at 300 s (est / stress): engine 0.35 / 0.75%, simulator 0.35 / 1.45% (stalls on sampling
    noise), GHK CPU 0.35 / 1.52%, logit 0.91 / 2.32%, RFC CPU 2.49 / 3.60%. All gaps under 3 $ per 1,000 trips.
  - $2.44 (GPU 152 min under a 160-min cap). [Summary](analysis/T25/SUMMARY.md), [design](analysis/T25/TEST.md).
- **T24 · Trivago hotel shelves, a real panel with ~150 properties (2026-10-20): no error correlation beyond tastes;
  on 24-dimensional shelf-share orthants the engine is faster and more accurate than OVUS.**
  - 24,879 users with 2–8 sessions (59,176 last-clickout choices, 25 hotels per shelf); 64 coefficients (position,
    price, stars, rating, type, 54 amenities); 20% of users held out.
  - A: MVP's five shared error components (type, star, rating, price tertile, amenity cluster) are all 0 at full N;
    iid takes 0.999. MVP − MXL: +0.005 [+0.004, +0.007] nats per choice for a new user, −0.007 [−0.010, −0.005]
    per user's sequence: a tie in substance. The mixed models beat MNL by 0.08–0.09 only on a user's sequence.
  - B: engine "high" 95 µs per orthant vs batched OVUS 225 µs; median |ΔP| 0.0011 vs 0.0036 (clicked hotel,
    median P 0.096). Only "highest" (300 µs) keeps the max share error under 0.005. No zeros.
  - About $2 of Modal time (MVP 48 min on an A100). [Full write-up](analysis/T24/TEST.md).
- **T22 · Swissmetro, real stated choice (2026-10-19): MVP has the best holdout log-likelihood at every training
  size (100–468), but by little; the rail error correlation is 0 once random tastes are in; no T20-style floor.**
  - 623 respondents × 9 tasks, J = 3 (train, SM, car); 25% of respondents held out, population predictions.
  - MVP − RFC-S: +0.001 to +0.043 nats per choice (5 of 7 CIs above 0). Binned shares: MVP beats RFC-S by
    0.4–2.4 pts (5 of 7 above 0), and ties plain HB-MNL (HB lower in 6 of 7, 1 CI excludes 0). Pooled MNL and NL
    are worst; NL's λ is unstable at N = 100.
  - The RFC tuning (σ_l 1.6–2.8) worsens share calibration vs plain HB in all 7 fits.
  - Under $0.05 of T4 time. [Full write-up](analysis/T22/TEST.md).
- **T21 · T19's takeaway baskets vs the real incumbents (2026-10-19): MVP beats coded subsets over all 21 groups
  in every cell; the engine is 3× OVUS on real d = 21 patterns but fails the accuracy guard.**
  - A: coded subsets (T3's analyst rule, and blocks chosen from MVP's R), at most 5 groups per block. M3 − CS: +0.11
    to +0.36 nats per order. On pairs that span blocks MVP is 0.13–0.27 pts better; inside a block CS is 0.03–0.24
    better. On the observed top-10 bundles, mostly spanning blocks, MVP is 0.16–1.31 pts better. M1 still ties MVP.
  - B: 200 holdout basket patterns at d = 21, raw (min eig 1e-8) and shrunk R. Engine "high" is 3× faster than
    batched OVUS, but its |ΔP| is about 10× OVUS's and it returns P = 0 on 4 of 200 (reference P 1e-11 to 7e-6).
    The near-singularity is not the cause.
  - $0, local. [Full write-up](analysis/T21/TEST.md).
- **T20 · Sample-size curves, small CBC (2026-10-19): HB-MNL + RFC-S has a ~5% cost-regret floor at every N from
  50 to 1,500; MVP at N = 50 is already below it. No crossing, so N_RFC-S / N_MVP is unbounded (model-free > 30 at
  N_MVP = 50).**
  - T5's generator, models and metric; R = 10. RFC-S at all 11 N; MVP at 50 / 100 / 200 only (binary search).
  - G1 / G2: RFC-S − MVP +3.1 to +5.1 cost-regret points at N = 50–200, every CI above 0. G0 logit control: no
    MVP advantage.
  - RFC-S tuning hits a bound in about 40% of fits (same as T5). The floor check (wide box, RFC-G, tuning on true
    shares) shows the floor is not a tuning artefact.
  - Replaces the "N ≥ 720 vs 150" reading of T5, where 720 was the pass threshold, not a measurement.
  - About $2 compute. [Full write-up](analysis/T20/TEST.md).
- **T19 · MVP on real takeaway baskets: bundle cannibalization (2026-10-19): MVP ≫ independence; MVP ≈ MBC-style
  cross-effects logit, and the logit is better on small orders.**
  - Kaggle UK Indian takeaways (2 restaurants, 13k and 20k orders, 2016–2019), 21 item groups, date split 75/25.
    No price ever changed (the multi-price items are protein variants sold side by side), so the bundle math is
    at current prices only.
  - Decision test (546 entrée + side bundles, full-bundle rate): MVP beats independent logits by 0.08–0.19 pts
    over all bundles and 0.35–1.6 pts on the 10 most common. Against M1 (chained cross-effects logits, the same
    210 dependence parameters) the gap is +0.007/+0.008 pts on all orders (immaterial) and −0.03 to −0.06 pts on
    small orders and with the size covariate. The top-10 ranking is the same for both. Joint holdout log-lik: M1
    is ahead of MVP by 0.05 nats per order on all orders and 0.4 on small orders.
  - The pairwise-IFM correlation lands on the PD boundary; s = 0.05 shrinkage was needed for 21-dim GHK.
  - [Full write-up](analysis/T19/TEST.md).
- **T18 · One panel orthant per respondent vs the nested simulated likelihood (2026-10-17): Part A PASS; Part B (lean):
  copula ≈ exact; PASS (relabelled by the user).**
  - With the Genz variable order, one 240-dim GHK orthant per respondent beats T4's nested 512 × 256 likelihood on
    log-lik error (median 0.08–0.29 vs 0.67 nats) and predicted parameter bias (0.7–2.8 vs 5.2 SE), at 3.6–20× the
    speed. It is smooth. Natural order is a HARD FAIL (heavy-tailed weights).
  - A2 fits: 21 s (1k) and 2.2 min (16k) on an H100 against 4.5 h for the nested fit on an L4. No parameter is
    significantly off the truth (joint Wald p 0.88 / 0.85; nested 1.6e-13). The rule's agreement-with-nested test
    failed because nested is the biased one; relabelled PASS by the user.
  - Part B (lean, non-normal tastes): one Gaussian-copula orthant per respondent matches the exact orthant arm
    on shares (within 0.014 pts; LN SOFT FAIL as immaterial, MIX PASS) and decisions (0 flips, 0% regret both;
    saturated), fitting 3–5× faster. About $7.55 in all for T18.
  - [Full write-up](analysis/T18/TEST.md).
- **Further interest (paused 2026-10-17, not designed): twin-strength sweep on synthetic data.** T15-style shelf,
  heterogeneous tastes plus within-person twin correlation swept from 0 (IIA) to strong; MNP vs HB-MNL + RFC.
  Question: how strong must twin substitution be before MNP's edge is material? Gives a threshold to compare
  real data against (T17's soft-drink shelf sits near zero). A nested-logit arm was considered and set aside
  (rare in conjoint practice; RFC is the real incumbent). No kill criteria yet.
- **T15 · Engine pipeline on the shelf, J = 20–60 (2026-10-16): D1 PASS, D4 SOFT FAIL; S0 guard pass.**
  - **Main question (MNP via the M*P pipeline vs HB-MNL + RFC-S, regret against the truth, 5 datasets).**
    - D1 PASS: assortment, Δ +1.85 pts [+1.30, +2.40].
    - D4 SOFT FAIL: joint line + extension + price, +0.20 [+0.13, +0.27], reliably better but below the 0.5 floor.
  - **Pipeline vs accurate-only search:** a tie at J = 20.
  - **Engine vs OVUS as the screener:** 3.6–20× faster.
  - **S0 guard:** under a logit truth M*P ties or slightly beats the incumbent, with no invented similarity.
  - **Cost:** M*P costs more per decision (H100 fit about 20–27 min vs HB 3–4 min on CPU). About $13 in all.
  - [Full write-up](analysis/T15/TEST.md), [design](analysis/T15/DESIGN.md).
- **T10 · MACML vs GHK follow-up: Part A inconclusive, Part B paused (2026-10-10).**
  - Part A: at N = 150 / 300, MACML and GHK made nearly the same decisions, as labelled in advance.
  - Part B (J = 20): paused at the Step 0 probe. GHK at the draw count that recovers the shares projected over the
    3 h cap on a T4. An H100 brings it to about 17 min per fit (2026-10-12). The choice of draw count is open.
  - Includes the OVUS vs engine side study. [Full write-up](analysis/T10/TEST.md).
- **T4 · Shelf scaling: paused on cost (2026-10-09).**
  - The recovery gate failed as specified. Shares were recovered only at 512 × 256 draws, at 4.5 h + 3.6 h SEs on
    an L4. pack0 is about −3 SE in every fit.
  - The full run was estimated at about $45 (J = 20) to $600 (all J). T13 and T15 reuse its J = 20 fit.
  - [Full write-up](analysis/T4/TEST.md), [design](analysis/T4/DESIGN.md).

### Closed

- **T31 · B5, MNP fit at large J vs HB-MNL: Part A PASS, Part B PASS at m = 6 (2026-10-23).**
  - Per-task probit fit (GHK-1k, H100) at J = 20 / 40 / 60: 1.1–1.8 / 6.5–7.1 / 8.6–15 min. HB-MNL misplaces
    1.6–2.2× MNP's holdout share in all 14 fits. Line + price decision, 10 reps: 1–2% more profit than HB-MNL + SoP
    at m = 6 (Holm p 0.02, both truths), 2–4% more than + RFC-S; not shown at m = 8 vs SoP. A calibrated
    compartmentalized incumbent (one study per pack format) is no better than full-shelf HB-MNL.
  - Sawtooth is limited to logit by its software, not by J or m; the GPU makes the better model usable. All truths
    are our probit generator (mixed-logit truth not run). [Full write-up](analysis/T31/TEST.md).
- **T30 · Line + prices beyond enumeration, greedy + swap vs wide search: no search effect (2026-10-22).**
  - T29 plus 5 price levels per SKU (2.1e7 lines at m = 6, 2.0e9 at m = 8). Greedy + swap matches wide search for
    every arm at m = 4, 6, 8: line profit has diminishing returns, so the search never forces a shortcut. The
    "larger m or K breaks the incumbent" path is paused. No write-up (`analysis/T30/t30.py`, `out/`).
- **T29 · Client line from a candidate set, HB-MNL + SoP / RFC-S vs MNP (2026-10-22): superseded by T31.**
  - Manufacturer 103's 17 SKUs, m = 4–8, exhaustive search, rep 0 only. A small m = 8 edge for MNP (one rep, no
    CI); its 720-dim panel fit is biased (taste sds 17–23% too wide), which T31's per-task fit replaces.

- **T28 · TURF reach line at scale, stepwise vs exhaustive search: KILL at Gate 0 (2026-10-22).**
  - dunnhumby soft drinks, J = 30–100, m = 4–15: an off-the-shelf ILP (HiGHS) proves the optimum in 0.05–1.1 s per
    cell on one CPU core; stepwise is within 0.13 pts of it everywhere, stepwise + swap within 0.04. Holdout gaps
    are noise (−0.30 to +0.73 pts). The optimal line lives in the top ~30 items, so J doesn't change it. $0.
  - No reach or speed claim: the incumbent's wall is the software, not the problem.
  - [Full write-up](analysis/T28/TEST.md).
- **T17 · Within-person substitution on dunnhumby soft drinks: stopped, power check FAIL (2026-10-17).**
  - Even with a nested-logit truth, the twin test can't separate NL from MNL: Δ +0.02 pts, 95% CI [−0.07, +0.10]
    (materiality 1.0). Pooled nesting by manufacturer is weak (λ 0.82–1.0) and the price coefficient is
    positive (promotions endogenous). About $0.01.
  - The C4 question (twin substitution given heterogeneous tastes) stays open; this shelf can't answer it.
  - [Full write-up](analysis/T17/TEST.md).
- **T16 · T6 against per-household independence (practitioner TURF): FAIL (2026-10-17).**
  - HB per-household independence closes 94% of T6's gap: pooled reach error MVP 2.25 vs HB-IND 2.38 pts (Δ +0.13,
    95% CI [−0.06, +0.32]); greedy portfolios identical at every k. Control gate passed. About $0.16.
  - T6's PASS was against pooled independence; it is not an advantage over Sawtooth-style TURF.
  - [Full write-up](analysis/T16/TEST.md).
- **T14 · MBC linked sets at k > 12: PASS (2026-10-12 / 13).**
  - Part A (simulated, k = 20 / 30): coded subsets flipped 207 of 240 decisions, MVP-IFM 50.
  - Part B (24-item baking cluster, real): the holdout backs MVP on 3 disagreements, coded subsets on 0, at
    the threshold.
  - [Full write-up](analysis/T14/TEST.md).
- **T13 · Line of 6 of 20 with joint survival: KILL (2026-10-11).** The joint and independent rules picked the
  same line on all 18 decisions. About $0.06. [Full write-up](analysis/T13/TEST.md).
- **T11 · Joint chance-constrained product line: FAIL (2026-10-11).** The joint constraint never chose a
  different line from the independent one (0 of 640 searches). What helps is the uncertainty-aware constraint
  itself. About $1.60. [Full write-up](analysis/T11/TEST.md).
- **T9 · MACML as an incumbent probit arm: FAIL (2026-10-10).** MACML and GHK made identical decisions (0 flips,
  0 regret, all truths, N = 600), so there was no GHK win. About $0.35. [Full write-up](analysis/T9/TEST.md).
- **T8 · GPR kernel error covariance: FAIL on the primary criterion (2026-10-10).** Under a continuous-attribute
  truth the kernel tied variance shares (0.51% vs 0.56% regret, inside 2 SE). The no-loss guard passed. About
  $2. [Full write-up](analysis/T8/TEST.md).
- **T7 · Engine vs shared-draw simulator for TURF search: FAIL (2026-10-10).** At matched regret the simulator
  was 9–16× faster than the engine at every k. The engine's picks had essentially zero regret. About $0.05.
  [Full write-up](analysis/T7/TEST.md).
- **T6 · Correlated-reach TURF on public basket data: PASS (2026-10-10).** Against pooled independence only; T16 shows the gain is household heterogeneity.
  - MVP's held-out reach error was 49% lower than the independence formula's.
  - Greedy portfolios flipped at k = 3–6, and MVP won all 4 on held-out reach.
  - About $0.01. [Full write-up](analysis/T6/TEST.md).
- **T5 · N-sweep (static sample savings): PASS (2026-10-09).** MVP matched the decision error that HB-MNL + RFC-S
  needs at N ≥ 720 already at N = 150, on both correlated truths. RFC-S did not improve with N. About $3.40.
  [Full write-up](analysis/T5/TEST.md), [results](analysis/T5/RESULTS.md).
- **T1–T3: PASS (2026-10-08).**
  - T1: MVP beats RFC, with caveats.
  - T2: 28 vs 146 flips of 240.
  - T3: the holdout backs MVP on 4, coded subsets on 0.
  - Results in `analysis/T1`–`T3`; [HISTORY.md](HISTORY.md).

## Baselines

- Speed is always measured at matched error on the decision-layer output, as
  wall-clock time to decision. Never against QMC Sobol alone.
- If a baseline has no usable implementation, stop and ask before writing
  one.
- Reporting (user, 2026-10-13; from T14 on): every compared decision gets the absolute difference in points with a
  95% CI, then the relative lift with its CI, and a multiplicity-adjusted p (Holm over the decisions compared).
  Points lead; relative lift is never reported alone (small bases inflate it). Statistical significance and
  materiality (the pre-set point floor) are reported as separate labels. Pre-registered kill rules stay as written;
  this changes the reporting, not the verdicts.
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
