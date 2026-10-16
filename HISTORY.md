Moved verbatim from PLAN.md on 2026-10-16. Summary: [PLAN.md](PLAN.md).

# T1–T3: original test plan, status and history

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
