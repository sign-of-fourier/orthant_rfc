# Plan: where multivariate probit beats incumbent choice methods

Status 2026-10-09: narrowed to two active items, **T5** (N-sweep) and **T6** (correlated-reach
TURF). T1–T3 completed, T4 paused. Everything else is in [BACKLOG.md](BACKLOG.md). Kill criteria
are set by the user (T5 set 2026-10-09; T6 blank); nothing runs until they are filled in. Scope is
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
- **Kill criteria (blanks are yours).**
  - Prediction: reach-error gap that counts as "better": ___
  - Decision: flips / reach regret that count: ___
  - Control: invented correlation allowed under the independent truth: ___
- **Scope.**
  - In: dunnhumby Complete Journey; one category cluster with natural near-duplicates (same
    brand, different flavor or pack), J = 15–30. Hold out the last 25% of trips, as in T3; reuse
    the T3 machinery ([design](analysis/T3/DESIGN.md)). Reach estimators: (a) independent
    product, (b) MVP orthant, (c) the empirical / deterministic baseline only if trivial.
  - OUT: MaxDiff data, engine-vs-simulator timing ([backlog](BACKLOG.md), first up), posterior
    uncertainty, B&B, pricing.
- **Tests (minimal).**
  1. Prediction: absolute reach error on held-out trips for random candidate sets, k = 1–6.
  2. Decision: greedy portfolio per estimator at k = 3–6, scored by held-out reach. Report flips
     and reach regret.
  3. Control: a simulated independent truth where MVP must not invent correlation.
- **Assumption (state in the write-up).** Removing an item does not change the remaining items'
  utilities. That is standard for TURF; counterfactual delisting isn't validated here.
- **Results.**

| k | estimator | mean abs reach error (held out) |
|---|---|---|
| | | |

| k | estimator | greedy portfolio | held-out reach | flip vs best | reach regret % |
|---|---|---|---|---|---|
| | | | | | |

| control (independent truth) | MVP max \|ρ̂\| | reach error, MVP vs independent |
|---|---|---|
| | | |

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
