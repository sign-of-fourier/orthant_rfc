# T1. σ recovery: design (approved 2026-10-06, reduced scope)

**Approved scope:** truth G1 only (σ values as below; max pair correlation 0.90
accepted) and RFC-S only. G0, G2 and RFC-G are next steps, tracked in `PLAN.md`.

Items 3 and 5. No analysis code has been written for this test.

## Question

When choice data come from a probit with per-attribute similarity, does a probit
fitted by full MLE with the engine recover the similarity σ, and the decisions
that depend on it, better than HB-MNL with tuned RFC?

## Non-goals

- Engine accuracy or speed sweeps (settled).
- IFM. CBC single-choice data has no IFM; the probit is fitted by full MLE only.
- GHK, Gibbs or MACML baselines (T2/T4).
- A time-to-decision claim. Wall-clock time is recorded but not concluded on (Q3
  belongs to T2–T4).
- Explaining *why* RFC's σ moves (the "4×" mechanism). T1 measures it; it
  does not model it.
- D1. It is built only after T1 concludes, and only if the D1 gate below passes.

## DGP

The market from `tests/rfc_vs_probit` is reused so results line up with it.

- **Attributes.** Brand (4 levels), flavor (3), pack (3), price (5 levels,
  $3.50–$5.50), effects-coded: 8 part-worths plus a price slope and a "none"
  constant, 10 parameters.
- **Respondents.** N = 600. β_n ~ N(b, diag(ω²)), with b and ω from
  `tests/rfc_vs_probit` (`MU_LEVEL`, `SD_COEF`); log-normal price slope.
- **Design.** T = 12 training tasks per respondent, 3 products plus "none".
  The design is random with balanced overlap: levels may repeat within a task,
  as in Sawtooth's balanced-overlap method, because σ_a is identified only from
  shared levels. Plus 6 fixed holdout tasks, the same for everyone, as
  Sawtooth uses for RFC tuning. 2 of the 6 contain a near-duplicate pair.
- **Utility, per respondent n, task t, alternative j:**
  U = x_jt'β_n + Σ_a σ_a ζ_{n,t,a,level_a(j)} + ν_njt.
  Products sharing a level of attribute a in a task share that ζ draw. ζ and ν
  are redrawn every task. "None" has its own independent error with the same
  total variance.
- **Truths (candidates; pick which to run):**
  - **G0 null.** σ_a = 0, ν iid Gumbel (logit truth). A good method should
    find σ ≈ 0. This is the false-positive check.
  - **G1 moderate.** σ on flavor and pack, small on brand: the
    `tests/rfc_vs_probit` D2 values (σ_b, σ_f, σ_p, σ_ν = 0.36, 0.86, 0.86,
    0.18). Normal ν.
  - **G2 strong.** G1 with flavor/pack σ scaled so a same-flavor, same-pack
    pair has error correlation about 0.9.
- **Price slope** is normal, N(−0.975, 0.41²), with the mean and sd of
  `tests/rfc_vs_probit`'s log-normal slope. It must be normal so a
  respondent's tasks are jointly normal for the exact panel likelihood.
- **Replicates.** R = 20 datasets per truth, with fixed seeds.

## Methods

**Incumbent: HB-MNL plus tuned RFC (the Sawtooth proxy).**
- **HB-MNL.** Normal mixing distribution, full covariance, a Sawtooth-like
  prior (prior variance 1, df = 5). 20,000 iterations, the last 10,000 kept.
  Per-respondent point estimates are posterior means, as in Sawtooth's
  utilities file. Implementation: `bayesm::rhierMnlRwMixture` (bayesm 3.1-6,
  R 4.3.3, Ubuntu packages), one mixture component. bayesm's default prior
  differs in detail from Sawtooth's; the difference is documented in RESULTS.
- **RFC.** Run on the HB point estimates, first choice, 10⁵ draws per scenario
  with common random numbers, tuned to minimise share MAE on the 6 holdout
  tasks (Sawtooth's practice). Two variants:
  - **RFC-S (faithful).** One attribute-variability σ shared by all
    attributes, plus product variability (Gumbel) and the exponent. This is
    our understanding of Sawtooth's simulator controls, not verified.
  - **RFC-G (generous).** One σ_a per attribute, plus product variability and
    the exponent. It can represent the truth's structure. **This is a
    deviation from Sawtooth practice** and is labelled as such.

**MVP: probit by full MLE with the engine.**
- The model has the same form as the DGP (normal ν): b, diagonal ω, σ_b, σ_f,
  σ_p, with the scale fixed by σ_ν. Under G0 it is misspecified, by design.
- **Likelihood (revised 2026-10-06, option 1).** Simulated panel likelihood.
  Each respondent gets R scrambled-Halton taste draws β_nr = b + ω·z_nr
  (R = 200 selftest, 500 full). Given β, each task's choice probability is a
  3-dimension orthant from the engine; the within-task error covariance does not
  depend on β. P(y_n) = mean over r of the product over t. Taste spread (fixed
  across tasks) and σ_a (redrawn every task) are separated by the panel
  structure, as before.
  - *Why not the exact 36-dimension panel orthant (the original plan):* in
    the selftest, engine values on respondents with likelihoods near e⁻¹⁹
    jumped by 0.4–0.7 in log-likelihood for 3e-3 parameter steps (not the
    near-duplicate merge; same with dup_corr = 0). The finite-difference fit
    stopped after 14 iterations, far from the truth.
- **Likelihood (revised 2026-10-07, default).** The exact panel orthant (36 dims) by GHK
  with fixed scrambled Sobol draws (M = 4,096 per respondent), on Modal GPUs
  (`ghk_gpu.py`; `run.py --mvp ghk`). The taste integral is exact, so the MSL bias of
  option 1 is gone; gradients and the Hessian are exact (autograd); SEs are sandwich.
  **Deviation:** the likelihood is not evaluated by the engine. T1 is a Q1 test, so the
  integrator does not affect its conclusion. The option-1 MSL below remains as
  `--mvp msl`.
- **Gradient (option 1 only).** Analytic for b and log ω (dP/du_i = φ(u_i)·Φ₂(conditional),
  the engine at 2 dimensions; the formula matches SciPy to 6 digits).
  Forward differences for log σ. The engine's 3-dimension values are about
  0.6% off SciPy on a checked problem, so the analytic gradient and the engine's
  objective disagree by 1–3%; convergence is judged by max |gradient|.
- **Optimizer.** L-BFGS-B, then sandwich SEs (per-respondent scores and a
  forward-difference Hessian of the summed score).

## Metrics (candidates; you set the tradeoffs)

The scales differ (probit vs logit), so every σ metric is scale-free.

**σ recovery**
- **S1.** Within-respondent error correlation of a near-duplicate pair
  (A and A′, same flavor and pack), implied by each fitted simulator, against
  the truth. This is the quantity that drives cannibalization.
- **S2.** Per-attribute variance share σ_a² / total error variance, against
  the truth (RFC-G and MVP only; RFC-S has one σ).
- **S3.** The ratio of fitted to true correlation (S1). The earlier "about 4×"
  figure is unverified and is not used as a reference.
- **S4.** For MVP only: 95% CI coverage of σ_a across replicates.

**Decisions** (each scored against the truth simulator)
- **D-price.** Pick A's price from the 5 levels to maximise A's margin, given
  costs. Report the chosen level, a flip vs the truth, and profit regret (% of
  true optimal profit).
- **D-ext.** Add the line extension A″ (A at −$0.30). Report incrementality
  error and a launch flip at hurdle h.
- **D-cost.** Pick, among 4 candidate SKUs, the one whose addition maximises
  portfolio contribution Σ(p − c)·share with fixed unit costs. Report the
  flip and profit regret.
- **Shares (descriptive only).** Holdout share MAE.

## Speed protocol

Recorded, not concluded: wall-clock time for HB, RFC tuning and MVP MLE, with
machine, backend (local `.so` vs API) and versions.

## Quick mode vs full run

**`--selftest` (CPU, local `.so`)**
- G1 only, R = 1, N = 100, HB 2,000 iterations, coarse RFC grid.
- Fail-fast checks:
  - raw truth shares sum to 1 within 0.035 per task (the engine's recorded
    scenario-sum range; shares are then renormalized);
  - MVP log-likelihood at the fit ≥ at the truth;
  - MVP at N = 1,000 (reduced from 2,000 for CPU time) on one dataset
    recovers every σ_a within 2 sandwich SEs;
  - HB log-likelihood drift from the 3rd to the 4th quarter of kept draws
    < 1% (bayesm does not report acceptance rates);
  - no RFC tuning parameter at a bound;
  - RAM under 1 GB.
- Per-batch log lines.

**Full run**
- G0, G1, G2 × R = 20 × N = 600.
- MVP fits on GPU via the API. HB and RFC run on CPU.
- I ask before launching.

## Expected runtime

Rough figures, refined after quick mode.

- **Selftest:** under 15 min on CPU.
- **Local `.so` timing (MSL):** one score evaluation (value, analytic b/ω
  gradient, 3 σ differences) at N = 20, R = 200 takes 1.5 s, so about 7.5 s
  at N = 100. At N = 600, R = 500 it is about 110 s, so a fit of about 100
  iterations takes about 3 h on CPU, or about 60 h for 20 fits. The full run
  needs the GPU API; its time will be measured before I ask to launch.
- **HB:** measured at this size (N = 600, T = 12, 4 alternatives, 10
  parameters): 2,000 iterations in 9.7 s, so about 100 s per fit and about
  1.6 h for 60 fits. RFC tuning
  takes under 1 h in total.
- **Full run:** HB and RFC about 2 h on CPU; MVP time on GPU to be measured.

## Kill / pass rule (blanks are yours)

On truths G1 and G2, pooled over replicates.

- **Equal σ recovery.** The median |S1 error| of the better RFC variant is
  within ___ of MVP's.
- **Decisions match.** Across D-price, D-ext (hurdle h = ___) and D-cost:
  - MVP has at most ___ fewer flips than RFC (of 3 decisions × R replicates);
  - the mean profit-regret difference is under ___ % of true profit.
- **KILL** if both hold.
- **PASS** (MVP better) if either fails in MVP's favour.
- **G0** is a guard, not part of the kill. If MVP's median S1 under G0
  exceeds ___, MVP invents similarity. That is reported as a caveat against
  any PASS.
- **INCONCLUSIVE** if the replicate-to-replicate spread in S1 is wider than
  the MVP–RFC gap. Concretely: the 95% CI of the paired difference over
  replicates contains both 0 and ±___.

**D1 gate (separate).** D1 is built only if MVP's S1 error on G1 is within ___
and its S4 coverage is at least ___.

## Resolved points

- **A. HB-MNL.** R and bayesm are installed; `rhierMnlRwMixture` is used. If it
  causes problems, rebuilding is to be discussed, not done unasked.
- **D. "About 4×".** Not a blocker; the ratio S3 is reported without a
  reference value.
- **E. Costs.** Placeholder unit cost of 40% of the middle price ($1.80) for
  every SKU. Costs are needed for D-price (margin) and D-cost; D-ext does not
  use them. Hurdle h stays a blank for you.

## Open points

- **B. Truths.** G0 (no similarity), G1 (moderate) and G2 (strong): run all
  three? Recommended: all three.
- **C. RFC-G.** It gives RFC one σ per attribute, so a loss for RFC cannot be
  put down to Sawtooth having a single σ. One extra tuning run per dataset.
  Recommended: keep it, labelled as a deviation.
