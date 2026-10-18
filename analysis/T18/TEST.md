# T18 · One panel orthant per respondent instead of the nested simulated likelihood (design, 2026-10-17)

Status: **Part A PASS (Genz order; A2 relabelled by the user). Part B (lean) done 2026-10-17: copula ≈ exact
orthant on shares and decisions (LN share SOFT FAIL by 0.014 pts, MIX PASS, regret PASS both). Part B relabelled PASS by the user 2026-10-18.**

## Question

With normal tastes, can the nested simulated likelihood for a panel probit (T4's MVP-G: QMC over tastes ×
per-task GHK) be replaced by one GHK orthant of dimension d = T(J − 1) per respondent, at equal or better
accuracy and lower cost? If so, does the idea extend to non-normal tastes (Part B)?

## Background and prior decisions

- **Model.** β_n ~ N(b, Ω), within-task errors ε_t ~ N(0, Σ) redrawn each task. A respondent's stacked utility
  differences D = A U are N(μ, Ω_D) with μ = A X b and Ω_D = A (X Ω X′ + I_T ⊗ Σ) A′. The whole choice history
  is P(D > 0): one orthant with cutoffs k_i = μ_i / √Ω_D,ii and correlation R.
- **It is the same integral as the nested loop.** The panel orthant integrates the tastes analytically, and the
  nested loop integrates them by QMC. In Part A any difference between the two arms is simulation error only,
  with no model difference.
- **T1 already does this at d = 36** (`analysis/T1/ghk_gpu.py`, M = 4,096, fixed scrambled Sobol). It recovered
  the parameters with no MSL bias, and the fit median was 73 s at N = 600 on a T4.
- **T10 (2026-10-10), user decision:** "the 240-dim panel orthant arises only from the person-level tastes ...
  GHK uses the conditioned likelihood (QMC over β_n × per-task GHK). Full 240-dim GHK only for across-task
  occasion errors." The 240-dim probe was stopped before it finished. **T18 reopens that decision on purpose**
  and measures the accuracy question it left open.
- **T4 gate (2026-10-09), the nested benchmark.** J = 20, S1, N = 1,000, rep 0, nested 512 × 256 (Q taste draws ×
  R inner GHK draws), seed 1. It took 78 evaluations and 207 s per evaluation on an L4 (4.5 h fit, 3.6 h SEs).
  Variance shares came out .13/.35/.29/.23 against a truth of .10/.35/.35/.20, and pack0 sits at about −3.4 SE in
  every fit. The θ̂, SEs and nll are saved in `analysis/T4/out/gate_g/fit_512x256.json`. The H100 runs the same
  evaluation 12.5× faster (T10 probe), which is about 17 s per evaluation and about 22 min per fit at N = 1,000.
- **Compute.** Per respondent and per evaluation, the panel arm costs about M · d²/2 and the nested arm about
  Q · R · T · (J − 1)²/2 (inner products only). The panel arm is cheaper when M < Q · R / T = 512 · 256 / 12 ≈ 10.9k.
  Counting the log Φ / Φ⁻¹ calls per path tilts this further toward the panel arm (Q · R · T · (J − 1) of those
  against M · d). The timings are measured in Step 0, not assumed.

## The risk being tested

- **Log of an average remains.** Panel GHK still takes log(mean of M weights). Its bias is about −CV²/(2M), where
  CV is the coefficient of variation of the GHK weights. The panel arm removes the *outer* log-of-average layer;
  it does not remove the bias itself. At d = 240 with P ≈ e⁻²⁰ to e⁻⁴⁰ (the T4 gate nll is −26.7 per respondent
  on average), the weights may be heavy-tailed. In that case CV is large and the 1/M rate does not hold at
  affordable M.
- **Variable order.** GHK error at high d depends on the integration order. The natural order (task by task) is
  the default. A per-respondent order (most-constrained first, chosen once at θ₀ and held fixed for the whole
  fit so the likelihood stays smooth) is open point B. It is standard GHK practice, not a new integrator.

## Part A · Normal tastes (the main question)

### Data

- **A-240:** T4's gate dataset, `t4.make_data("S1", 20, 1000, 12, 0)`: J = 20 SKUs plus none, T = 12, so
  d = 12 × 20 = 240. The covariance comes from `ghk_gpu.panel_arrays` with T4's levels, as in T10's probe.
- **A-36 (sanity anchor):** T1's G1 recovery dataset at N = 1,000, d = 36, where the panel arm is known to work.
  If T18's code fails here, it is a bug, not a finding.

### Arms (GHK only; the engine is excluded, as settled in T1)

- **Panel GHK at M ∈ {1k, 4k, 16k}:** fixed scrambled Sobol, one random shift per respondent, as in T1. Autograd
  gives exact gradients. Each step accumulates rank-1 updates, so backward memory is O(M · d) per respondent
  rather than O(M · d²).
- **Nested at T4's sizes, 512 × 256:** T4's `mvp_gpu.py` method G, unchanged. 256 × 128 is reported for context
  only.
- **Reference (not an arm):** panel GHK at M = 2¹⁸ with 8 independent random shifts. Its CI comes from the
  spread across shifts, on the P scale, then mapped to log. It is cross-checked against a nested reference at
  4,096 × 2,048. The two are the same integral, so if they disagree beyond their CIs, I stop and report: the
  reference is suspect and nothing below means anything.

### A1 · Accuracy at fixed true θ (no fitting)

All per-respondent metrics are absolute errors in log-likelihood (nats), with 95% CIs by bootstrap over
respondents. Comparisons of each panel M against nested 512 × 256 are paired over respondents, Holm over the
3 values of M. Significance and materiality are labelled separately.

- **E1. Per-respondent log-lik error:** median and 90th percentile |Δ log L|, and the mean signed error (bias).
  Each arm is run with 8 independent shifts, which separates bias from noise (SD across shifts).
- **E2. Error on the sum over N:** the total log-lik bias and its SD. The fit sees the sum.
- **E3. Score bias, which predicts the fit:** Δg = Σ_n (∇ log L̃_n − ∇ log L_n^ref) at the truth, mapped to a
  predicted parameter shift Δθ ≈ H⁻¹ Δg in units of the T4 gate SEs. Report the max over parameters and the
  value for the 3 variance-share parameters. This predicts the A2 fit bias before paying for A2.
- **E4. Weight diagnostics (descriptive):** effective sample size (Σw)² / Σw² divided by M, per respondent. Also
  whether E1's bias falls as 1/M from 1k → 4k → 16k (×4 per step). If it doesn't, the weights are heavy-tailed.
- **E5. Smoothness (the T1 failure mode).** Along 8 random unit directions in θ, plus each variance-share
  parameter, at steps h ∈ {1e-4, 3e-4, 1e-3, 3e-3} around the truth:
  - the max per-respondent |Δ log L| at h = 3e-3 (T1's engine jumped by 0.4–0.7 here);
  - roughness, the max |second difference| / h² compared with the reference curvature;
  - the relative gap between the autograd gradient and a central difference.
  With fixed draws, GHK is smooth by construction, so this is a check that it really is.
- **E6. Time:** likelihood plus gradient for all N = 1,000 on one H100, per arm, as the median of 3 runs. Peak
  memory and chunk size are recorded too.

### A2 · One full fit each, T4 gate dataset

- **Panel fit:** at the M that you choose after A1 (open point C). Same L-BFGS, θ₀ and parameterisation as the T4
  gate (b, log ω, variance-share softmax, total error variance fixed).
- **Nested fit:** reuse the 2026-10-09 512 × 256 fit (same dataset, draws and seed; $0). Alternatively, rerun it on
  an H100 so the wall time is measured on the same hardware (about 22 min, about $1.50; open point D).
- **Metrics:**
  - z = (θ̂ − θ_true) / SE per parameter. Both fits use the gate's 512 × 256 sandwich SEs (SE is a property of
    the dataset; the SE run is skipped as in T4, open point D).
  - Variance shares against the truth and against the nested fit, in absolute points.
  - Panel−nested agreement, |Δθ| / SE.
  - Holdout and population shares at θ̂ against the truth (absolute pts, 95% CI), and T4's 13 decisions
    (flips at 0.5% regret) for panel vs nested. Descriptive, since there is one dataset.
  - nll at θ̂ and at the truth under the reference.
  - Wall time, evaluations, GPU type and cost.

### Verdict rules (proposed 2026-10-17, significance first; awaiting your approval)

User, 2026-10-17: a near miss should not count as a fail; statistical significance is the criterion. T18 asks
whether the panel arm can *replace* the nested one, so the tests are non-inferiority tests. The panel arm fails on
accuracy only if it is **significantly worse** than nested 512 × 256. A difference that the data cannot
distinguish from zero is a pass. Materiality m only grades a significant loss (SOFT vs HARD). Significance means
Holm-adjusted p < 0.05 over the panel arms compared (3 values of M × 2 orders = 6), using two-sided paired tests
over respondents. CIs are 95%, unadjusted, and reported with the points.

**Accuracy table** (E1 and E3 separately). Δ = nested error − panel error, so positive means the panel arm is
better.

| label | condition |
|---|---|
| PASS (better) | Δ > 0, significant |
| PASS (no worse) | not significant |
| SOFT FAIL (worse, immaterial) | Δ < 0, significant, \|Δ\| < m |
| HARD FAIL (worse, material) | Δ < 0, significant, \|Δ\| ≥ m |

- **E1, per-respondent error.** e_n = mean over the 8 shifts of |log L̃_n − log L_n^ref|. The test is paired over
  the 1,000 respondents. Suggested m = **0.05 nats** per respondent, about a 5% relative error in that
  respondent's likelihood.
- **E3, predicted parameter shift.** Per shift s, the statistic is max over parameters of |Δθ_s| / SE. H is the
  outer product of the reference scores (OPG) at the truth, and the SEs come from H⁻¹. The test is Welch's over
  the 8 shifts per arm. Suggested m = **0.25 SE**: simulation bias of 0.25 SE adds about 3% to the RMSE.
- **E2 and E4** are reported only.

**Guards** (a failure stops the test; these are not graded).

- **Reference cross-check:** nested reference − panel reference, paired over respondents. If the difference is
  significant (p < 0.05) and larger than 0.01 nats on average, stop: the reference or the code is wrong.
- **Smoothness (E5):** the median Taylor residual |ℓ(θ + h e) − ℓ(θ) − h g·e| must shrink like h². Pass if the
  log-log slope over h ∈ {1e-4 … 3e-3} is ≥ 1.5. T1's jumps would give a slope near 0. The relative gap between
  the gradient and a central difference ≤ 1e-3. This runs in float64, with float32 noise reported separately.

**Speed (E6).** The speedup is nested time ÷ panel time, both float32 on the same H100, with likelihood plus
gradient over N = 1,000 and 3 repeats each; the CI is a t-interval on the log ratio.

| label | condition |
|---|---|
| PASS | the CI is above 1 and the speedup is ≥ 2× (suggested materiality) |
| SOFT | the CI is above 1 and the speedup is < 2× |
| INCONCLUSIVE | the CI contains 1 |
| HARD FAIL | the CI is below 1 (the panel arm is slower) |

**Choosing the A2 arm (pre-registered, which resolves open point C).** Take the cheapest panel arm (M, order)
that is PASS on both E1 and E3 and not HARD FAIL on speed. **A1 overall:** PASS if such an arm exists and its
speed label is PASS or SOFT. INCONCLUSIVE if the accuracy labels pass only at an M whose speed is INCONCLUSIVE.
HARD FAIL if no M passes on accuracy, or every M that does is slower.

**A2, one fit each** (panel at the chosen arm; the nested fit reused):
- **Agreement.** Wald test of θ̂_panel − θ̂_nested with the gate's sampling covariance V (25 df). The two fits
  differ only by simulation error, so this asks whether that error is detectable against sampling noise. PASS if
  p ≥ 0.05. The max |Δθ| / SE is reported too.
- **Recovery against the truth.** Per parameter, z = (θ̂ − θ_true) / SE, Holm over the 25 parameters. PASS if
  every parameter the panel fit rejects (Holm p < 0.05) is also rejected by the nested fit. This handles the
  dataset's pack0 ≈ −3.4 without excluding it (open point E). The joint Wald test of the 3 variance shares
  against the truth is reported for both fits.
- **A2 overall:** PASS if both pass. Wall time and cost are reported, and the speedup is labelled as in E6.
- **Gating:** A2 runs only if A1 is PASS or INCONCLUSIVE (INCONCLUSIVE means you decide). Part B runs only if A2
  passes.

## Part B · Non-normal tastes (only if A passes)

The tastes are no longer jointly normal, so the panel orthant is no longer exact. The question is which
orthant-based approximation stays close to the nested loop, which is still exact up to simulation.

- **Truths (T4 S1 errors, J = 20, N = 1,000, one dataset each):**
  - **B-LN:** price coefficient −exp(N(m, s²)), with its mean and SD matched to T4's normal slope (−0.975, 0.41);
    the other tastes as T4.
  - **B-MIX:** two segments, π = 0.4 / 0.6, differing in brand-0 preference and price sensitivity; within each
    segment β ~ N(b_k, Ω), with a common Ω equal to T4's ω². The values are fixed in Step 0 so that the
    segments' shares differ by at least 5 pts on the firm's SKUs.
- **Arms:**
  - **Nested loop:** QMC draws from the true taste distribution × per-task GHK at 512 × 256. This is the arm the
    others are scored against, and also its own reference at 4,096 × 2,048.
  - **Mixture-of-normals sum of orthants:** P = Σ_k π_k P_k(D > 0), one panel orthant per component. This is
    exact for B-MIX. For B-LN, the lognormal is approximated by a K-component normal mixture (K ∈ {3, 5}, fitted
    once to the true marginal). Report the mass of the approximation that sits at positive price slopes.
  - **Gaussian copula:** k_i = Φ⁻¹(P(D_i > 0)) from each difference's true marginal (1-D quadrature over the
    non-normal coefficient), and R from the normal part (open point F); then P ≈ Φ_R(k) by panel GHK.
- **Metrics:**
  - **B1, fixed θ:** E1–E3 as in A1, with the nested reference as truth.
  - **B2, one fit per arm per truth:** shares (absolute pts, 95% CI) and T4's 13 decisions (flips, regret)
    against the truth simulator. Each arm minus nested is paired over decisions, Holm over arms.
- **Verdict (proposed):** the same non-inferiority table, with the nested loop as the comparator. B1 uses Δ in
  per-respondent log-lik error with m = 0.05 nats. B2 uses Δ in share error (m = 0.5 pts) and decision regret
  (m = 0.5 pts of best profit, T15's m), Holm over arms.

### Part B implementation (2026-10-17, `t18b.py`; user: "build part B")

- **Truth values.**
  - **B-LN:** |β_price| = exp(m + s z), m = −0.1067, s = 0.4035, giving mean −0.975 and SD 0.41.
  - **B-MIX:** δ = (+1.25 brand0, −0.75 price). The segment means are b + 0.6δ and b − 0.4δ, so the population
    mean stays T4's b; π = 0.4. The firm's 6-SKU share is 65.9 vs 23.0 pts by segment (gap 42.9, need ≥ 5).
  - Own data seeds (T4's SeedSequence plus a family code). The "none" share is 15.0% (LN) and 16.3% (MIX).
- **Parameters.**
  - B-LN: T4's 25, with the price slots read as (m, log s).
  - B-MIX: 28 = segment-2 mean, common log SDs, 3 shares, δ_brand0, δ_price, logit π.
  - Fits start from T4's θ₀; B-MIX starts at δ = (0.5, −0.5), π = 0.5.
- **Arms** (orthant arms at M = 16k with the Genz order fixed at θ₀, A2's more accurate size):
  - **nested:** 512 × 256 with the true family's taste draws (B-MIX: per segment, weighted).
  - **mixture:** B-MIX is exact (2 orthants). For B-LN, z is split into K equal-probability strata (K = 3, 5),
    each a normal component with the stratum's mean and variance of the lognormal. This is smooth in (m, s).
  - **copula:** cutoffs from each difference's true marginal (B-LN by 32-node Gauss–Hermite over z; B-MIX in
    closed form), and R the correlation of D under the moment-matched taste covariance (open point F, default).
- **B1 reference (changed from the design's nested 4,096 × 2,048).** A showed the nested reference's error is
  about 0.11 nats at d = 240, which is too imprecise.
  - B-MIX: the exact 2-orthant mixture, panel Genz 2¹⁸ × 8 shifts.
  - B-LN: Gauss–Hermite over z_price, with a panel Genz orthant at each node (price coefficient fixed at the
    node). This is exact up to quadrature and GHK. B0 checks 32 vs 64 nodes, 2¹⁴ draws per node.
- **B2 shares and decisions:** first-choice simulation, 10⁶ draws, common random numbers across fitted models
  and the truth, T4's 43 scenarios and 13 decisions (`t4.decision_values` / `t4.score`). Local numpy.
  - Share error: per scenario, the mean |Δshare| in points, paired over the 43 scenarios.
  - Regret: the 7 price and delist decisions, paired.
  - Extensions: |incrementality error|, descriptive.

## Step 0 results (2026-10-17): code works, the d = 240 reference needs a variable order

**Run.** One H100 (info-29741), about 2.5 min wall, about $0.20. Files: `out/smoke_A36.json`,
`out/smoke_A240.json`, `out/smoke_A240_time.json`, `out/smoke.log`. The summary is `t18.py report smoke`.
Errors are per respondent (n = 20) against the natural-order panel reference (2¹⁸ draws, 1 shift). That is
indicative only: no CIs over shifts.

| d = 240 arm | median \|err\| (nats) | p90 | signed mean | ESS / M |
|---|---|---|---|---|
| panel natural 1k / 4k / 16k | 1.41 / 0.68 / 0.66 | 4.03 / 2.67 / 1.44 | −1.85 / −0.96 / −0.68 | 4.5e-3 / 1.4e-3 / 4.3e-4 |
| panel genz 1k / 4k / 16k | 0.24 / 0.18 / 0.14 | 0.66 / 0.74 / 0.46 | +0.13 / +0.12 / +0.16 | 1.3e-2 / 8.0e-3 / 6.0e-3 |
| nested 512 × 256 (T4 gate size) | 0.85 | 1.64 | −0.43 | 6.5e-3 (of Q) |

- **The reference is not good enough at d = 240.** The natural-order reference's ESS is 6.8e-5 × 2¹⁸ ≈ 18 draws.
  Natural-order ESS stays roughly constant as M grows (about 4.6, 5.7 and 7 at 1k, 4k and 16k), which is the
  signature of heavy-tailed GHK weights. The genz arms sit +0.12 to +0.16 above that reference. A log of an
  average is biased downward, so the reference is probably low. The cross-check nested − panel reference is
  +0.17 [−0.09, +0.44], with max |·| 2.28: not significant, but wide. Genz order raises ESS (13, 33 and 98 at 1k,
  4k and 16k) but still sublinearly in M.
- **At d = 36 everything is consistent.** References agree: +0.0008 [−0.053, +0.055]. Genz 16k has a median
  error of 0.0035 against nested's 0.11.
- **Smoothness guard: PASS at both d.** Taylor-residual slope 2.00; gradient vs central difference ≤ 4e-8; the
  max |Δ log L| at h = 3e-3 is 0.011 (panel) and 0.027 (nested), against 0.4–0.7 for T1's engine.
- **float32 = float64** to about 1e-6 nats in every arm.
- **Time, N = 1,000, likelihood + gradient, float32, H100:** panel 4k (natural) 1.01 s (3 repeats, ±0.002;
  peak 22 GiB) against nested 512 × 256 12.34 s (peak 45 GiB). That is 12× faster. The Genz order costs 0.1 s
  once per dataset.
- **Stopped before A1.** A1's E1–E3 are measured against the reference, so the reference must first be shown
  to be accurate at d = 240.

## Step 0b results (2026-10-17): the Genz-ordered reference converges

**Run.** One H100, 5 min wall, about $0.35. File: `out/refprobe_A240.json`; summary: `t18.py report refprobe`.
A-240, first 50 respondents, values only. Panel GHK uses the Genz order, fixed at θ₀, with 8 independent shifts
at each M. The pooled estimate is the log of the mean over the 8 shifts; its SE comes from the delta method on
the P scale.

| M per shift | per-shift SD of log P, median (p90) | pooled 8 × M SE, median (p90) | ESS median | time, 8 shifts |
|---|---|---|---|---|
| 2¹⁸ | 0.022 (0.047) | 0.0078 (0.017) | 1,235 | 11 s |
| 2²⁰ | 0.011 (0.024) | 0.0039 (0.0087) | 4,820 | 40 s |
| 2²² | 0.0045 (0.011) | 0.0016 (0.0041) | 18,116 | 160 s |

- **Converges at the √M rate.** ESS grows about 4× per 4× M, and the SD halves. That is the healthy regime, unlike
  natural order at d = 240, whose ESS stayed at about 18.
- **No drift across sizes.** Pooled 2²⁰ − 2¹⁸ is −0.0018 [−0.0044, +0.0007]; pooled 2²² − 2²⁰ is −0.0006
  [−0.0019, +0.0008]. Neither is significant. The median |·| is 0.0046 and 0.0017.
- **The Step 0 natural-order reference was biased low** by 0.19 nats on average (median |·| 0.18), so Step 0's
  error table at d = 240 is not valid. Its signs (natural arms low, genz arms slightly high) were reference bias.
- **The nested reference can't serve as a cross-check.** At 4096 × 2048 × 4 shifts its pooled SE is 0.11 (median).
  Nested − genz 2²² is −0.041 [−0.115, +0.033], with max |·| 1.10: consistent, but 70× less precise than the panel
  reference. The guard is replaced by the convergence check above.
- **Proposed A1 reference:** Genz order, 2²⁰ × 8 shifts (pooled SE about 0.004 nats, 12× below m = 0.05), with
  gradients, on all N = 1,000.

## A1 results (2026-10-17): PASS with Genz order, HARD FAIL with natural order

**Run.** 7 H100s in parallel, 12 min wall, about 58 GPU-min including start-up, about $3.80. Files:
`out/a1_*.json`, `out/a1_summary.json`, `out/a1.log`; summary: `t18.py report a1`. N = 1,000 at the truth.
Reference: Genz order, 2²⁰ × 8 shifts, pooled SE median 0.0039 nats (p90 0.010). E3 uses the OPG H at the truth;
OPG SEs are 0.83× the gate's sandwich SEs (median), so E3 shifts in sandwich-SE units are about 17% smaller.

| arm | E1 median \|err\| | E2 total bias over N [95% CI] | E3 mean max \|shift\| (SE) | time (s) | E1 vs nested | E3 vs nested | speed |
|---|---|---|---|---|---|---|---|
| nested 512 × 256 | 0.673 | −454 [−485, −423] | 5.20 | 12.26 | — | — | — |
| panel natural 1k | 1.638 | −1,672 [−1,700, −1,643] | 70.5 | 0.51 | HARD FAIL | HARD FAIL | 23.9× |
| panel natural 4k | 1.091 | −1,072 [−1,094, −1,050] | 47.9 | 1.00 | HARD FAIL | HARD FAIL | 12.2× |
| panel natural 16k | 0.746 | −676 [−700, −652] | 30.6 | 3.12 | HARD FAIL | HARD FAIL | 3.7× |
| **panel genz 1k** | 0.285 | −96 [−107, −86] | 2.78 | 0.51 | PASS (better) | PASS (better) | 20.5× [13.4, 31.4] |
| **panel genz 4k** | 0.154 | −33 [−37, −29] | 1.31 | 1.00 | PASS (better) | PASS (better) | 12.2× |
| **panel genz 16k** | 0.080 | −9.2 [−11.8, −6.5] | 0.73 | 3.37 | PASS (better) | PASS (better) | 3.6× [3.1, 4.2] |

- **E1, Δ = nested − panel error (positive = panel better).** Genz: +0.49 [+0.46, +0.53], +0.63 [+0.60, +0.66],
  +0.71 [+0.68, +0.74] nats; Holm p ≤ 1e-155. Natural: −1.12, −0.53, −0.11, all significant and material.
- **E3 on the variance shares (mean predicted shift, SE units [95% CI]).**
  - nested: flavor −1.69 [−2.38, −1.00], pack −1.77 [−2.38, −1.15], brand −0.45 [−1.26, +0.37]. This predicts
    understated shares, consistent with T4's gate.
  - genz 1k: about +2.5 on all three. genz 4k: about +1.1 to +1.2. genz 16k: +0.56 to +0.62, every CI above 0.
    The genz arms overstate the shares, with the opposite sign to nested.
  - natural: −26 to −70.
- **E4.** Genz bias shrinks about 3× per 4× M (−0.096 → −0.033 → −0.009 nats), close to the 1/M regime. Natural
  order shrinks only about 1.6× per step, which fits heavy-tailed weights.
- **E5 guard: PASS on all 7 arms.** Slope 2.00, gradient gap ≤ 5e-8, max |Δ log L| at h = 3e-3 ≤ 0.027.
- **Verdict by the pre-registered rule: A1 PASS.** All three Genz arms are PASS on E1 and E3, with speed PASS. The
  cheapest passing arm, and so the A2 arm, is **panel genz 1k**.
- **Caveat on the rule (for the user).** Non-inferiority is judged against nested, and nested is itself biased
  (max shift 5.2 SE; shares −1.7 SE). The chosen genz 1k beats nested but still predicts about +2.5 SE on the
  shares, which may fail A2's recovery test against the truth. Genz 16k predicts +0.6 SE. Whether to also fit
  16k is a scope question for the user.

## A2 results (2026-10-17): PASS (relabelled by the user); FAIL by the original rule (agreement with nested)

**Relabel (user, 2026-10-17): "relabel".** The pre-registered rule needed agreement with the nested fit and recovery
against the truth. Agreement failed because the nested fit is itself biased (A1 E3; joint Wald against the truth
p = 1.6e-13). Recovery against the truth decides, so A2 is **PASS** for both panel arms. The original label is kept
below for the record.

**Run.** 2 H100s in parallel, about 3 min wall, about $0.35. Files: `out/a2_genz_1024.json`,
`out/a2_genz_16384.json`, `out/a2_summary.json`; summary: `t18.py report a2`. Same dataset, θ₀, parameterisation
and L-BFGS settings as the T4 gate. Fixed draws (shift 0), Genz order fixed at θ₀. SEs are the gate's 512 × 256
sandwich SEs (open point D). Genz 1k is the rule's arm; Genz 16k was added by the user.

| fit | wall | iter / evals | shares b / f / p / product | Holm rejections vs truth | shares Wald vs truth p | agreement with nested: Wald (25 df), p |
|---|---|---|---|---|---|---|
| truth | | | .10 / .35 / .35 / .20 | | | |
| nested 512 × 256 (2026-10-09) | 270 min L4 (H100 est. 16 min) | 49 / 78 | .125 / .349 / .294 / .233 | pack0 | 0.35 | — |
| panel genz 1k | **0.3 min** H100 | 32 / 39 | .124 / .370 / .348 / .159 | none | 0.67 | 111, p 8e-13 → **FAIL** |
| panel genz 16k | **2.2 min** H100 | 35 / 42 | .154 / .351 / .350 / .145 | none | 0.38 | 108, p 3e-12 → **FAIL** |

- **Pre-registered verdict: A2 FAIL for both arms.**
  - Recovery: PASS. Neither panel fit has a Holm rejection, so the subset rule holds trivially; pack0 is z −2.32 /
    −2.37 against −3.4 for nested.
  - Agreement: FAIL. The panel fits differ from the nested fit by much more than sampling noise; max |Δθ| / SE is
    2.0–2.3.
- **Converged.** Final gradient max 0.08 / 0.07 (summed). Each fit's nll is below the truth's under its own
  simulator (by 10.7 and 9.7).
- **Post hoc, not pre-registered (reported for the user's decision):**
  - The two panel fits agree with each other: Wald 2.5 on 25 df (p ≈ 1), max |Δθ| / SE 0.7.
  - The joint Wald test of all 25 parameters against the truth gives panel 1k p = 0.88, panel 16k p = 0.85, and
    nested p = 1.6e-13. RMS z is 0.78 / 0.88 against 1.50 for nested.
  - With A1's E3 (nested predicted max shift 5.2 SE; Genz 0.7–2.8), this points to the nested fit as the biased
    one. The agreement criterion assumed the nested fit was close to correct, and it is not.
  - Product share: panel .159 / .145 against truth .20 (nested .233). The share z's are +1.0 to +1.7, within 2 SE.
- **Time:** about 46× (1k) and 7× (16k) faster than the nested fit's estimated H100 wall time, or 900× / 120×
  against its actual 4.5 h L4 run.
- **Not done (from the A2 design):** holdout/population shares and T4's 13 decisions, and nll under the
  reference. They are descriptive on one dataset; to be added if wanted.

### Part B leaned (user, 2026-10-17: "lean it up")

- **Hypothesis:** for non-normal tastes, one Gaussian-copula orthant per respondent is as good as an exact
  likelihood.
- **Dropped:**
  - the nested arm (Part A showed it is the slow and biased comparator);
  - B1, the fixed-θ diagnostics (B0 already shows every arm runs, is smooth and is roughly accurate);
  - mixture K = 3.
- **B2 (lean).** Per truth, one fit of the copula and one of the exact orthant arm: B-MIX the 2-orthant mixture
  (exact up to GHK), B-LN the K = 5 strata mixture (near exact; K = 3 ≈ K = 5 in B0). Then shares and decisions
  against the truth (10⁶ draws, common random numbers).
- **Verdict:** the non-inferiority table, Δ = exact − copula (positive = copula better). It is applied to the
  share error (paired over the 43 scenarios, m = 0.5 pts) and to regret (paired over the 7 price and delist
  decisions, m = 0.5 pts), Holm over the 4 comparisons. Parameter recovery against the truth is reported only.
- **Cost:** 4 H100 fits in parallel, about 20 GPU-min, about $1.50.

## Part B, B0 smoke (2026-10-17): every arm works, smooth, references converge

**Run.** 4 H100s, about 2.5 min wall, about 10 GPU-min, about $0.70. Files: `out/b0_{LN,MIX}{,_time}.json`;
summary: `t18b.py report smoke`. There are 20 respondents per truth and errors are against the reference, with 1
shift per arm (indicative; no CIs).

| truth / arm | median \|err\| (nats) | p90 | signed mean | time, N = 1,000, lik + grad |
|---|---|---|---|---|
| LN nested 512 × 256 | 0.738 | 1.713 | −0.581 | 12.40 s |
| LN copula | 0.162 | 0.393 | +0.000 | 3.15 s |
| LN mixture K = 3 | 0.119 | 0.362 | −0.004 | 9.39 s |
| LN mixture K = 5 | 0.123 | 0.361 | −0.005 | 15.64 s |
| MIX nested 512 × 256 | 0.539 | 1.176 | −0.142 | 24.65 s |
| MIX copula | 0.114 | 0.305 | −0.030 | 3.16 s |
| MIX mixture (exact) | 0.056 | 0.173 | −0.041 | 6.25 s |

- **The LN reference has converged in quadrature:** 32 vs 64 Gauss–Hermite nodes differ by less than 1e-4 nats.
  B1 uses 32 nodes.
- **Smoothness:** slope 2.00 on every arm; gradient gap ≤ 1.2e-8; max |Δ log L| at h = 3e-3 ≤ 0.017.
- The arms' GHK noise (16k, 1 shift) is mixed with their approximation error here. B1's 8 shifts separate the bias.

## Part B lean B2 results (2026-10-17): the copula matches the exact orthant arm

**Run.** 4 H100 fits in parallel, 13 min wall (the LN exact fit is the long pole), about 28 GPU-min with start-up,
about $1.85 (quoted $1.50). Shares and decisions are local (10⁶ draws, common random numbers, 43 scenarios, 13
decisions). Files: `out/b2_{LN,MIX}_{copula,exact}.json`, `out/b2_summary.json`, `out/b2_report.txt`; summary:
`t18b.py report b2`.

| truth / arm | fit (H100) | iter / evals | share error vs truth (pts, mean over scenarios) | regret | flips | ext \|incr err\| |
|---|---|---|---|---|---|---|
| LN copula | **2.6 min** | 37 / 50 | 0.147 | 0.000% | 0/13 | 0.007 |
| LN exact (mixture K = 5) | 12.8 min | 40 / 49 | 0.133 | 0.000% | 0/13 | 0.005 |
| MIX copula | **2.2 min** | 36 / 43 | 0.251 | 0.000% | 0/13 | 0.022 |
| MIX exact (2-orthant mixture) | 6.3 min | 45 / 61 | 0.259 | 0.000% | 0/13 | 0.019 |

Δ = exact − copula (positive = copula better), Holm over the 4 comparisons:

| truth | metric | Δ [95% CI] | Holm p | label |
|---|---|---|---|---|
| LN | share error | −0.014 [−0.015, −0.013] pts | 6e-28 | SOFT FAIL (worse, immaterial; m = 0.5) |
| LN | regret | 0.000 | 1 | PASS (no worse) |
| MIX | share error | +0.008 [+0.008, +0.009] pts | 1e-27 | PASS (better) |
| MIX | regret | 0.000 | 1 | PASS (no worse) |

- **Shares and decisions:** the copula is within 0.014 pts of the exact arm on both truths, with no flips and 0%
  regret for either arm. Both arms' share errors against the truth are 0.13–0.26 pts.
- **Caveats.**
  - The share-error CIs pair the 43 scenarios of a single fit per arm, so they measure variation across
    scenarios, not replication across datasets. The tiny CIs overstate certainty about a repeatable difference.
  - The regret test has no power: decisions are saturated at N = 1,000 (0 flips for both arms, as in T9 / T13).
- **Speed:** the copula fits 2.9× (MIX) to 4.9× (LN) faster than the exact orthant arm, because it is one orthant
  per respondent against K = 2–5.
- **Parameters (reported only).**
  - Both arms land close together on the normal-part parameters and the error shares; both understate the pack
    share (LN .26–.27 against .35).
  - On B-MIX the copula's segment structure drifts more: logit π +0.37 against +0.09, δ_brand0 −0.25 against
    −0.14. The shares are unaffected.
- **Not tested (dropped by the lean design):** nested as a comparator, and fixed-θ log-lik / score-bias
  diagnostics.

## Steps, time and cost

All on Modal as ephemeral apps on profile info-29741, no keep-warm. Prices are the T10 list prices (H100
$3.95/h). Estimates scale T1's 73 s fit (d = 36, T4 GPU) by d² and are unverified until Step 0. I re-quote
before every paid step.

| step | what | where | wall | cost |
|---|---|---|---|---|
| build | `t18.py`: one generic GHK (blocked, rank-1 offsets) for panel, nested and references; order; smoothness; timing | — | done | $0 |
| 0 · smoke | A-36 and A-240 on 20 respondents: both references, panel {1k, 4k, 16k} × {natural, genz}, nested, f32 vs f64, smoothness on 10; then full-N (1,000) timing of panel 4k and nested at A-240 | 1 H100 | 10–20 min incl. image build | ≤ $1.30 |
| A1 | N = 1,000, 6 panel arms × 8 shifts plus nested × 8 shifts, reference 2¹⁸ × 8 on all N, smoothness on 100 | 1–2 H100 | quoted from Step 0 | est. $3–6 |
| A2 | panel fit at the chosen arm; nested reused | 1 H100 | quoted from A1 | est. $0.70–2.70 |
| B | quoted after A | | | |

Part A total: about $5–10.

## Open points

Defaults taken 2026-10-17 so the test is ready to run; change any of them.

- **A. Verdict rules:** proposed above (significance first; m = 0.05 nats, 0.25 SE, 2× speed). Awaiting approval.
- **B. GHK order:** both orders are measured in A1 and are among the arms for the A2 choice.
- **C. M for A2:** the pre-registered rule above (the cheapest passing arm).
- **D. Nested fit and SEs:** reuse the 2026-10-09 L4 fit and its SEs. No panel SEs.
- **E. pack0:** handled by the Holm subset rule in A2. Not excluded.
- **F. Copula R (Part B):** moment-matched Pearson correlation of D. Decided before Part B.

## Rules carried over

- Report absolute errors with 95% CIs, Holm across comparisons, significance and materiality separately.
- Torch runs on Modal only (4 GB local box). Public repo: no engine material. The engine is not an arm.
- Scope is strict: no new arms, truths or metrics without asking. Commit only when asked.

## T4 cost check (2026-10-18): timing smoke at J = 20 / 40 / 60

`python3 t18.py jsmoke` / `report jsmoke` (out/jsmoke_*.json, out/jsmoke_report.txt). T4 S1 datasets, rep 0, N = 1,000,
Genz-ordered panel orthant, one H100 per J. Time = likelihood + gradient over all 1,000 respondents (median of 2);
error = per-respondent |log-lik error| vs a 2^18 Genz reference on the first 100 respondents. The reference's own
ESS falls with d (median about 1,280 / 180 / 94 at J = 20 / 40 / 60), so the errors at J = 40 and 60 are lower bounds.

| J (d) | M | s / eval | ESS / M | median abs error (nats) | signed mean | 40-eval fit |
|---|---|---|---|---|---|---|
| 20 (240) | 1k / 4k / 16k | 0.56 / 1.01 / 3.38 | .015 / .010 / .007 | 0.24 / 0.12 / 0.056 | −0.15 / −0.08 / −0.03 | 0.4 / 0.7 / 2.3 min |
| 40 (480) | 1k / 4k / 16k | 2.10 / 3.28 / 9.80 | .006 / .003 / .002 | 0.51 / 0.32 / 0.18 | −0.34 / −0.16 / −0.09 | 1.4 / 2.2 / 6.5 min |
| 60 (720) | 1k / 4k / 16k | 5.58 / 8.27 / 20.5 | .005 / .002 / .001 | 0.83 / 0.49 / 0.26 | −0.57 / −0.31 / −0.14 | 3.7 / 5.5 / 13.7 min |

- Time grows slower than d² (16k: 3.4 → 9.8 → 20.5 s). Accuracy at fixed M degrades with d; the error roughly
  halves per 4× draws. Matching J = 20's 16k accuracy would need roughly 4× draws at J = 40 and 16× at J = 60
  (extrapolated, not measured).
- Whether that matters for parameters and decisions is not shown by per-respondent error; a single J = 60 fit
  checked against the truth (as A2) answers it directly.
- Cost about $0.45 (6 GPU-min plus start-ups). The J = 60 dataset took about 22 min to build locally (4 GB box;
  built in 50-respondent chunks).

## Engine side check (2026-10-18, user request): smooth and fast, not accurate enough on the panel orthant

At the user's request, the engine (approximate mode, no extra preprocessing) on the J = 20 panel orthant
(d = 240), true θ, first 100 respondents, CPU, cost $0. Details are in the private notes.

- Speed: 0.9–6.6 ms per respondent on CPU (about 1–7 s for N = 1,000), no GPU needed.
- Smoothness: PASS. Second differences scale as h² (slope 2.00); central-difference gradients agree to 8e-8.
- Accuracy: FAIL for this use. Median |log-lik error| 5.0 nats per respondent (p90 11.5), biased high by
  +3.4 nats, against 0.056 for GHK at 16k draws. On a single task (d = 20) it is within about 0.05 nats of SciPy;
  with two tasks (d = 40), 0.07–0.33. The panel's correlation is block-structured (strong within a task, weak
  across tasks), and one shared factor does not represent it; the error grows with the number of tasks.
- Engine start, GHK finish (2026-10-18, J = 20, about $0.20): the engine-only fit does not converge. In 300
  L-BFGS iterations (9 s) every coefficient and SD grows without bound (price −253 vs −0.98; log SDs up to +7),
  at a log-likelihood that the GHK optimum puts far below. The engine's approximate likelihood keeps
  improving as the scale grows, so it has no usable optimum and gives no starting point (the covariance at
  its estimate is not positive-definite). J = 60 fits not run (held by the user).
- Two-stage GHK (2026-10-18, J = 20, about $0.20; `t18.py warm`): the 16k fit started from the 1k estimate
  needed 36 evaluations (1.9 min) against 42 (2.2 min) from θ0; with the 1k stage, 2.23 min in total, the same
  as cold. Same optimum (nll equal to 0.001; max gap 0.002 SE). No saving: L-BFGS spends most of its
  evaluations rebuilding curvature, not travelling. Not pursued.
- J = 60 fit (2026-10-18, about $1.10; `t18.py j60` / `report j60`): Genz order, 16k draws, cold, as A2. 13.7 min,
  41 evaluations. Recovery vs the truth (OPG SEs from per-respondent scores at the estimate) FAILS: variance
  shares .157 / .419 / .308 / .116 vs truth .10 / .35 / .35 / .20 (Wald p 7e-9); Holm rejects u_brand (z +5.6),
  u_flavor (+5.2) and logsd_none (+3.6); joint Wald p 6e-7. The product share is pushed down, consistent with the
  simulation bias seen in the timing smoke (−0.14 nats per respondent at 16k). One dataset; not yet separated
  from a dataset effect.
- Draws or dataset? (2026-10-18, about $0.20; `t18.py j60shift`): scores at the J = 60 estimate with 64k draws
  predict where a 64k fit would go (one step, OPG Hessian). Largest predicted move 0.60 SE (logsd_flavor1);
  shares .157 / .419 / .308 / .116 → .158 / .411 / .313 / .119, still rejected (p 2e-8; same Holm rejections). If
  the simulation bias falls as 1/M, 16k → 64k removes three-quarters of it, so the draws explain little of the miss:
  most of it is this dataset (as pack0 at J = 20). Not yet checked on a second dataset.
