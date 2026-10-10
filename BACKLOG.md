# Backlog

Paused 2026-10-09: the broad test plan narrowed to two active items (PLAN.md: T5, T6). Everything
else lives here. Nothing here runs without the user's go. Each entry: what, why it matters, what it
depends on, where it came from.

## Next up after T5/T6 (user, 2026-10-09)

Order (user, 2026-10-09): T5 and T6, then the TURF engine-vs-simulator item (below, under Wedges),
then GPR, then MACML. Kill criteria are blanks for the user in both. Scope is a draft for the
design step.

### GPR: kernel-parameterised error covariance

- **What.** Σ_ij = k(x_i, x_j): a GP kernel over product attributes in place of the per-attribute
  variance shares. The current model is the special case of indicator kernels (same brand / flavor
  / pack) plus a nugget; the kernel adds continuous attributes (pack size, price tier, embeddings)
  and smooth similarity. Hyperparameters fitted by the probit likelihood (errors are latent; not
  GP regression on observed data).
- **Why.** Few parameters at any J; predicts the correlation of an SKU that doesn't exist yet
  (line extensions, CDT). Does not reduce the integral: each choice is still a (J − 1)-dim orthant
  (a low-rank kernel is a factor model, i.e. MVP-F, which failed recovery at affordable draws).
- **Hypothesis.** When similarity runs through a continuous or perceptual attribute, a kernel Σ
  recovers substitution and decisions (delist, extension, price) better than variance shares and
  HB-MNL + RFC; under categorical-only similarity it matches variance shares (no loss).
- **Kill criteria.** ___ (blanks for the user).
- **Scope.**
  - In: T1's CBC DGP and GHK GPU fitter (J = 3–6 per task, so the integral stays cheap); truths
    (a) categorical shares (T1 G1, control), (b) a continuous-attribute kernel truth,
    (c) `research/similarity_kernel/` "merge" (perceptual merging, kernel misspecified); arms:
    variance shares, kernel Σ (one kernel family chosen at design), HB-MNL + RFC-S.
  - OUT: J scaling / J = 60 (compute, not parameters, is the J = 60 limit), real data, SEs,
    kernel search over many families.
- **Tests (minimal).** 1. Recovery of the kernel hyperparameters (one dataset, the T1 gate).
  2. Decisions vs truth (flips, regret) per truth × arm, 20 replicates. 3. New-SKU test: hold
  one SKU out of fitting, predict its correlation and extension incrementality.
- **Prior work.** `research/similarity_kernel/` (Laplace kernel on part-worth distance vs a
  perceptual-merging truth; one replicate run). Finishing it is the cheap first step (local CPU).
- **Cost (rough).** Same per-fit cost as T1/T5 GHK: 3 truths × 3 arms × 20 reps at N = 600, about
  $2–4 of T4 GPU plus ~3 h local HB; the similarity_kernel replicates are local only.
- **Depends on.** T5's `run.py` (--scale total, --no-se); a kernel-Σ option in `ghk_gpu.sig2_of`.
- **From.** User, 2026-10-09.

### MACML: Bhat's estimator as an incumbent probit arm

- **What.** Maximum approximate composite marginal likelihood (Bhat): pairwise composite
  likelihood over a respondent's choice tasks, each term by Bhat's analytic MVNCD approximation.
  The fast analytic incumbent probit; listed as an arm in T2 and T4 but never implemented.
- **Why.** Our GPU GHK full ML has to beat the best existing probit estimator, not only
  HB-MNL + RFC and Gibbs. Known risk for MACML: bias at larger J and strong correlation (our
  near-duplicate regime). The only comparison so far is integrator-level
  (`multivariate_probit/docs/benchmarks.md`: Bhat OVUS 2–5× more accurate than the engine on a
  near-singular Σ, 4–7× slower), not a fit.
- **Hypothesis.** At J = 3–6 MACML matches GHK ML on decisions in less time; in the near-duplicate
  regime (G2, pair corr .97) and as J grows, its approximation bias moves σ and decisions, and GHK
  ML wins on decisions at a time the user accepts.
- **Kill criteria.** ___ (blanks for the user).
- **Scope.**
  - In: T1 DGP, truths G0 / G1 / G2, N = 600, 20 replicates; arms: MACML vs MVP (GPU GHK, the T5
    fit); metrics as T1/T5 (S1, flips, cost regret) plus fit time. J = 20 (T4 shelf, structured Σ)
    only if the J = 3–6 result leaves it open.
  - OUT: writing MACML from scratch (see implementation), SEs, real data, new truths.
- **Implementation (check first).** pybhatlib 0.4.0 (installed in an earlier session's scratch
  venv, not in this repo's environment): `models.mnp` (MNP with `mix` random coefficients,
  analytic MVNCD, analytic gradients) and `models.mnpkercp` (panel MNP by MSL over Halton draws with
  the MVNCD kernel). Whether it does composite marginal likelihood over panel task pairs with a
  structured Σ is unverified. If it doesn't fit T1's model, stop and ask before writing one
  (PLAN.md baselines rule).
- **Tests (minimal).** 1. Recovery on the T1 selftest dataset (one fit, vs the GHK fit).
  2. G0 / G1 / G2 × 20 replicates, decisions and time vs GHK ML.
- **Cost (rough).** MACML is CPU: pybhatlib loops over rows in Python, so time per fit is unknown
  (timed on one dataset first). GPU only for the GHK arm, reusable from T5's N = 600 cells (same
  seeds) at ~$0.
- **Depends on.** T5 (its N = 600 GHK fits are the comparison arm); pybhatlib's model coverage.
- **From.** User, 2026-10-09; T2/T4 designs ("MACML (backlog, no implementation)").

## Wedges (new, 2026-10-09)

- **Engine vs shared-draw simulator for TURF search.** **Done as T7 (2026-10-10): FAIL; see PLAN.md.** Greedy and
  branch-and-bound TURF search, including near-tie portfolios, scored by the engine vs a
  shared-draw (common random numbers) simulator.
  - Why: the engine-moat argument depends on it; for scenario scoring the QMC simulator was ~25x
    faster than the GPU engine (ORTHANT_PLAN S2), so the engine needs a search where it wins.
  - Depends on: T6 PASS.
  - From: user, 2026-10-09.
- **Adaptive fielding with early stopping.** Refit per wave; stop when the decision settles.
  - Why: the GPU-dependent half of the sample-savings wedge (fast refits make per-wave decisions
    practical). T5's N-sweep is the static half and is active.
  - Depends on: T5 result (a static N saving first); fast GPU fits.
  - From: user, 2026-10-09.
- **CDT from probit.** The estimated similarity structure as a probabilistic consumer decision
  tree for retail line reviews.
  - Why: puts the correlation matrix in a form category managers already use.
  - Depends on: a fitted Σ on shelf or basket data (T1/T3 outputs, or T6).
  - From: user, 2026-10-09.
- **Decision uncertainty as the output unit.** Report P(decision wrong) and P(A beats B).
  - Why: decision-layer output instead of parameter SEs.
  - Depends on: posterior or sampling distribution of the fit (SE value-proposition test below).
  - From: user, 2026-10-09.
- **Similarity-identifying designs.** Choice / MaxDiff designs that identify error correlation
  (Bayesian optimal design).
  - Why: T4's weak identification of share parameters (pack0 ≈ −3 SE) points at the design, not
    only the fitter.
  - Depends on: a working probit fitter for the design family (T4 MVP-G or the library fitter).
  - From: user, 2026-10-09.
- **Synthetic-panel audit.** Does an LLM respondent panel reproduce human substitution structure?
  - Why: synthetic panels are sold as a sample substitute; the probit Σ is a direct test of
    whether they get substitution right, not just shares.
  - Depends on: a human dataset with a fitted Σ to compare against (T1-style CBC or T3 baskets).
  - From: user, 2026-10-09.

## Moved from PLAN.md

### T4 follow-ups

- **Second dataset for pack0.** pack0 ≈ −3 in every fit so far (256×128: u z −0.8/−2.4/−2.5;
  512×256: pack0 −3.4, none −2.2, two taste SDs −2.7/−2.2); "likely this dataset". A second
  simulated dataset separates the dataset from the fitter.
  - Why: the gate is strictly FAIL on the 2-SE and 1-SE rules; this decides whether that is real.
  - Depends on: T4 un-paused (cost: ~$6 at 256×128, ~$45 at 512×256 for J = 20).
  - From: PLAN.md T4 status, 2026-10-09; [T4 design](analysis/T4/DESIGN.md).
- **Draw setting.** 256×128 vs 512×256 (taste × inner draws): 512×256 recovers shares (.13/.35/
  .29/.23) but fit 4.5 h + SE 3.6 h on L4; 256×128 fit 34 min + SE 29 min.
  - Why: sets the cost of every T4 fit.
  - Depends on: T4 un-paused.
  - From: PLAN.md T4 status, 2026-10-09.
- **Cheaper SEs.** "SEs not needed for decisions (move S4 to the SE test)".
  - Why: SE time ≈ fit time at J = 20; "`mvp_fit`'s SE step took ~18 of ~20 min per fit at
    d = 12, 222 parameters (T2 ran with `se=False`)."

  - Depends on: the SE value-proposition test below.
  - From: PLAN.md T4 status, 2026-10-09.
- **Panel probit fitter in the library.** "A proper panel probit fitter (MixedProbit: QMC over
  tastes, exact per-task orthant) in the library or API rather than a one-off; the engine as its
  inner integrator."
  - Why: T4's `mvp_gpu.py` is a one-off.
  - Depends on: engine fitting accuracy (ORTHANT_PLAN: not accurate enough for fitting yet; GHK
    meanwhile).
  - From: PLAN.md follow-ups (T4).

### D1, SE test, unit economics

- **D1 demo (item 5).** "Built from T1 outputs: RFC shares plus an estimated similarity/
  correlation matrix mapped to a cost-weighted decision. Presentation, not science."
  - Why: the sales artifact for item 5.
  - Depends on: T1 (done, PASS with caveats).
  - From: PLAN.md tests.
- **SE value proposition (user, 2026-10-08; after T3/T4).** "Time-to-SE and 95% coverage, ours vs
  best practice, on T2 L1/L2 k = 12 (40 datasets). Arms: deployed `mvp_fit` SEs (T4, float64
  Hessian); fast variants (per-row gradients only, Hessian on A100, float32 Hessian); bayesm Gibbs
  20k posterior SDs; IFM + 200-resample bootstrap; optional R `mvProbit` (1 h cap). Metrics:
  coverage of R, slopes, pair joints, TURF-4 reach; width; P(chosen bundle is best). Est. 45–60 min
  wall, ~$8 GPU. Check whether Sawtooth's simulator SEs reflect only cross-respondent spread before
  claiming it."
  - Why: SEs were ~18 of ~20 min per `mvp_fit` at d = 12; also feeds decision uncertainty.
  - Depends on: T2 datasets (exist).
  - From: PLAN.md follow-ups.
- **Unit economics (user, 2026-10-09).** "Run-time stats per fit to price a fit. Every T4 fit
  already records seconds per likelihood evaluation, evaluations, fit and SE time, GPU. Add a
  short benchmark (`mvp_gpu.time_eval`, one evaluation + gradient) on L4 vs 2 x H100 (and A100) at
  J = 20/40/60, then cost per fit = evaluations x seconds/eval x $/GPU-hour + SE pass. Few minutes
  of GPU; not run yet (H100 deferred on cost)."
  - Why: pricing a fit.
  - Depends on: nothing (H100 deferred on cost).
  - From: PLAN.md follow-ups.

### Kill-rule blanks for T1–T4

- Unset in every test: T1 ([design](analysis/T1/DESIGN.md), "PASS with caveats" chosen
  2026-10-08), T2 ([design](analysis/T2/DESIGN.md), guard thresholds too), T3
  ([design](analysis/T3/DESIGN.md): minimum holdout support level, default 95%), T4
  ([design](analysis/T4/DESIGN.md): what "practical" means, default under 30 min per fit, and the
  flip allowance).
  - Why: the verdicts so far rest on large gaps, not pre-registered tolerances.
  - Depends on: the user (decision-layer tradeoffs).
  - From: PLAN.md status; T1–T4 DESIGN.md.

### T1–T3 follow-ups

Decided 2026-10-08: get through T1–T4 first; each test's follow-ups wait. (The N sweep, "N ∈
{300, 3000} at k = 8", moved out of this list into active work as T5.)

- **T1:** "3 vs 5+ products per task (RFC σ inflation); fold the total-variance normalisation into
  `run.py`; an engine-based fitter for the panel likelihood (engine still in progress)." Open code
  item: "`run.py` still fits MVP with σ_ν pinned; the total-variance version is in `mvp_refit.py` /
  `ghk_gpu.sig2_of(tvar=...)`. Fold it into `run.py` before reuse in T4." (T5 reuses T1; see T5.)
  - From: PLAN.md follow-ups and next steps; [T1 results](analysis/T1/RESULTS.md).
- **T2:** "respondent heterogeneity (random intercepts; panel methods for every arm); MACML
  (backlog, no implementation); engine-based full MLE; decision search over many scenarios (where
  the engine's ~500× decision speed matters)."
  - From: PLAN.md follow-ups; [T2 results](analysis/T2/RESULTS.md).
- **T3:** "cookout cluster (fallback, unused); random household effects; engine for many-trip
  decisions is slower than one simulated pattern distribution (44 s vs 5–7 s), so a
  batched/aggregated path if the engine is to be used there."
  - From: PLAN.md follow-ups; [T3 results](analysis/T3/RESULTS.md).
- **Margin model / flexible-margin regime (candidate, not approved).** As in PLAN.md next steps
  (2026-10-06): one fixed margin model for MVP and incumbents, or linear and flexible reported
  separately; a linear-vs-flexible variant in T3 or a new test on a dataset you choose.
  - From: PLAN.md next steps.
- **multivariate-probit 0.3.0 `fitter="modal"`** as a GHK/full-ML arm for T2/T3 (noted
  2026-10-06, not adopted).
  - From: PLAN.md next steps.

### Engine follow-ups

From [ORTHANT_PLAN.md](ORTHANT_PLAN.md) and PLAN.md. Why, for all: the engine is not accurate
enough for fitting (bulk bias) and is the slowest scenario scorer (S2); these decide whether it
has a role. Depends on: nothing beyond GPU minutes, except the bias fix ("out of scope until 1–2
say whether it is worth it").

- **Warm start (option 1).** "Engine fit, then QMC from there. Needs no accuracy rule (QMC has the
  last word). Measure: QMC iterations cold vs warm (cold was 50-80 in stage A)."
- **Harder screening set.** "A harder candidate set (many near-ties) if S1 is to be leaned on."
- **Low resolution and float32.** "Not tried: the engine at resolution "low" (order 0, faster),
  engine in float32."
- **Bulk-bias fix (option 3).** The engine underestimates common choice probabilities (P = 0.1–1)
  by 0.1% (weak) to 4% (near-dup), same sign everywhere.
- **6 Oct log-likelihood jumps.** "The exact 36-dim panel likelihood jumps with parameter steps of
  3e-3 on respondents with very small likelihoods (about e^-19): engine error on small
  probabilities, as documented, not a merge effect (same with dup_corr = 0)." Jumps of 0.42–0.66
  in log-likelihood per 3e-3 parameter move. A smoothness issue relevant to pricing and gradient
  relaxations.
- **Engine accuracy at ρ ~ 0.8 and small 12-dim patterns** (service): "Engine accuracy at
  correlations ~0.8 (2 points in 4-item reach) and for 12-dim patterns near 1e-4 (returns 0):
  candidates for the QMC-fallback router."
- **Weak / J = 12 reference check** (open caveat): QMC reference shares (.18/.20/.24/.37) far from
  the truth (.05/.10/.10/.75); check before leaning on that cell.
- From: ORTHANT_PLAN.md options and status; PLAN.md follow-ups (service) and next steps.
