# T37 · T34's assortment win vs logits that estimate the error dependence (2026-10-24): WIN IS ESTIMATING THE DEPENDENCE

## Why

T34: on T15's D1 assortment (choose k = 6, 8, 10 of 20 SKUs) MNP beats the incumbent's best readout (HB-MNL + RFC-S
over posterior draws) by 1.74 points. The residual is substitution structure: 80% of S1's error variance is shared
by brand / flavor / pack level, and RFC's one tuned σ is declared, not estimated. T19 found that a logit which
estimates the dependence (cross-effects) tied MVP. This test asks the same of T34's win: is it the probit, or
estimating the dependence at all? Neither challenger is what Sawtooth users run; both are in Apollo / Biogeme.

## Design

- **Data and decision:** T15 S1 reps 0-4 (J = 20, N = 1,000, 12 tasks), D1 exhaustive at k = 6, 8, 10 at base
  prices. T34's truth best, MNP picks (rescored) and RFC-S-over-draws arm reused; truth values of new picks by GHK
  at 65,536 points at the true parameters (T15's scorer).
- **Error-component mixed logit (EC):** diagonal normal tastes (11 means, 11 sds) + a normal error component per
  brand / flavor / pack level per choice occasion (one sd per attribute) + one on "none", logit kernel. Simulated
  ML on an H100: 250 taste draws per respondent × 500 error-component draws per occasion. Decision: 20,000
  population taste draws × 5 error-component draws, logit within.
- **Pooled cross-nested logit (CNL):** one nest per brand / flavor / pack level (12), "none" alone; allocation per
  attribute (3, softmax) and nest parameter per attribute (3) estimated; pooled tastes. Decision: closed-form shares
  per candidate.
- **Checks:** both on the logit truth S0 (rep 0): EC error-component sds should go to 0.

## Kill criteria (proposed by Claude, accepted by the user 2026-10-24)

Δ = EC regret − MNP regret, points of % best profit, paired t over the 5 per-dataset means.

- **WIN IS THE PROBIT:** Δ > 2 SE and Δ ≥ 0.5.
- **WIN IS ESTIMATING THE DEPENDENCE:** Δ < 0.5.
- **Else INCONCLUSIVE.**

## Results (2026-10-24): WIN IS ESTIMATING THE DEPENDENCE

**Files.** `t37.py`; `out/S1_rep{0..4}.json`, `out/check_S0.json`, `out/summary.json`; `run_rep0.log`,
`run_rep1-4.log`, `check_S0.log`; `out/pilot/` (rep 0 at 200 × 50 draws, see below).

5 datasets × 3 instances; Δ = arm − MNP (positive = MNP better), t-interval over per-dataset means. MNP regret is
0.00 on every instance.

| arm | regret | Δ [95% CI] | p | profit forecast error | fit |
|---|---|---|---|---|---|
| HB-MNL + RFC-S over draws (T34, incumbent's best) | 1.74 | +1.74 [+1.09, +2.38] | 0.002 | −12.8% | HB 3.5-4.7 min CPU + tuning |
| **Error-component mixed logit** | **0.35** | **+0.35 [−0.18, +0.88]** | 0.14 | −8.0 to +1.2% | 1.6-3.8 min H100 |
| Pooled cross-nested logit | 0.27 | +0.27 (same picks every rep) | | +10.9 to +18.4% | < 1 s |
| MNP (T15) | 0.00 | | | +0.5% (−1.1 to +2.6) | 0.35-2.2 min H100 (T18) |

- **Verdict: WIN IS ESTIMATING THE DEPENDENCE.** EC is 0.35 points behind MNP, below the 0.5 materiality and not
  significant. Both logits close 80-85% of the incumbent's 1.74-point gap.
- **Per instance.** At k = 6, where every incumbent arm makes T15's 2.8-4.8% pick error, EC is right on 4 of 5
  datasets and CNL on 5 of 5. The remaining misses are near misses: EC 0.08-0.57 at k = 8, 0.26 at k = 10 (rep 4:
  0); CNL 0.33 and 0.48 on every dataset.
- **MNP is still never worse.** 0.00 on all 15 instances, and its profit forecast is the only one within 3%. CNL's
  forecasts are 11-18% high on every instance: right picks, wrong level.
- **Recovery (EC, model scale ≈ 1.8-1.9× the truth's from price):** flavor 1.07-1.37 and pack 1.12-1.45 against
  about 1.4 expected; brand 0-0.19 against about 0.75 (the smallest share, 10%; not recovered); "none" 1.4-2.2.
- **Draws matter.** At 200 taste × 50 error-component draws (pilot, rep 0) EC collapsed the error components to
  0.0-0.46 and regret was 1.54 (k = 6 miss). Simulated log-likelihood is biased against error variance when the inner
  integral is under-sampled; 250 × 500 recovered them. The same nested-integral problem as T4's MNP fit.
- **S0 check.** EC on the logit truth: all four error-component sds go to 0 (pass). CNL on the logit truth: nest
  parameters 0.74-0.81 (1 = no nesting): the pooled CNL invents nesting from taste heterogeneity. Its decision cost
  on S0 is untested.
- **What this does to T34's claim.** It holds against the incumbent (Sawtooth's tools have neither model). It is not
  a probit advantage: any model that estimates the error structure gets most of it. MNP's remaining edge here is
  immaterial on picks (0.27-0.35) and material on profit forecasts.
- **Cost.** About $1 of H100 time (6 fits).
