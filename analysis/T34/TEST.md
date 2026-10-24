# T34 · T15's assortment win vs HB over posterior draws (2026-10-24): HOLDS

## Why

T15 D1 (choose k = 6, 8, 10 of 20 SKUs) is the largest remaining CBC win: MNP regret 1.85 points lower than HB-MNL
+ RFC-S, 95% CI [1.30, 2.40] (5 datasets). The incumbent simulated on respondent point estimates. At J = 60, HB
over posterior draws closed the share and decision gaps (T31). Two signs point the same way here: the incumbent
under-forecasts the profit of its own pick by about 17%, and it makes the same pick error on every dataset (drops
SKU 3, keeps SKU 12; exhaustive at k = 6 and still 2.8-4.8% short). Against that, T15's truth (T4 S1) puts 80% of
the error variance on errors shared by brand / flavor / pack level (T31 stress: about 50%). Draws fix overconfidence
about tastes; they add no correlated errors.

## Design

- **Data:** T15 S1 reps 0-4 (T4 `make_data`, J = 20, N = 1,000, 12 tasks, 6 holdout tasks).
- **HB refit:** same sampler, seed and length as T15 (bayesm, 20,000 iterations, keep 10), plus 100 draws per
  respondent evenly spaced over the second half (`analysis/T1/hb.R`, optional 6th argument).
- **Incumbent arms** (all searched exhaustively over D1, fixed base prices):
  - `sop_means`: SoP on point estimates.
  - `rfc_means`: RFC-S on point estimates, 200 error draws per respondent, σ and scale tuned on the holdout (T15's
    arm, refitted; sanity check against T15's incumbent).
  - `sop_draws`: SoP averaged over the 100 posterior draws.
  - `rfc_draws`: RFC-S over the draws, 2 error draws per posterior draw, tuned on the holdout.
- **MNP:** T15's pipeline picks and regrets, unchanged. Truth values of all picks by GHK at 65,536 points at the true
  parameters (T15's scorer); the true best is T15's, raised if any new pick beats it.
- **Also recorded:** each arm's profit forecast error, (own value − true value) / true value.

## Kill criteria (proposed by Claude, accepted by the user 2026-10-24)

Δ = arm regret − MNP regret, points of % best profit, paired t over the 5 per-dataset means, Holm over the two
draws arms.

- **HOLDS:** the better draws arm has Δ > 2 paired SE and Δ ≥ 0.5.
- **KILL:** the better draws arm has Δ < 0.5.
- **Else INCONCLUSIVE.**

## Results (2026-10-24): HOLDS

**Files.** `t34.py`; `out/S1_rep{0..4}.json`, `out/summary.json`; `run.log`.

5 datasets × 3 instances. Regret in points of % best profit; Δ = arm − MNP (positive = MNP better), t-interval over
the per-dataset means. MNP regret is 0.00 on every instance (T15).

| incumbent arm | regret | Δ [95% CI] | Holm p | profit forecast error |
|---|---|---|---|---|
| SoP, point estimates | 2.68 | +2.68 [+2.25, +3.12] | | −18.1% |
| RFC-S, point estimates (T15's arm, refitted) | 1.85 | +1.85 [+1.30, +2.40] | | −14.6% |
| SoP over draws | 1.93 | +1.93 [+1.53, +2.32] | 0.000 | −13.2% |
| **RFC-S over draws** (best draws arm) | **1.74** | **+1.74 [+1.09, +2.38]** | 0.002 | −12.8% |
| MNP (T15) | 0.00 | | | +0.5% (−1.1 to +2.6) |

- **Verdict: HOLDS.** The best draws arm is 1.74 points behind MNP, 7.6 paired SE and 3.5× the 0.5 materiality.
- **Sanity check.** The refit reproduces T15 exactly: same HB means, same tuned RFC σ, same picks and regrets
  (1.85).
- **What draws do.** They fix SoP's overconfidence (2.68 → 1.93; on point estimates RFC already did that, 1.85),
  and RFC over draws gains one instance (rep 4, k = 8: 2.18 → 0.57). Every other pick is unchanged. At k = 6 every
  incumbent arm on every dataset makes T15's pick error (2.8-4.8%).
- **Why, against T31.** T31's draws gain came from removing the point estimates' "none" bias. Here the residual is
  substitution structure: 80% of S1's error variance is shared by brand / flavor / pack level, and neither readout
  adds estimated correlation (RFC's one tuned σ is the same for all three attributes).
- **Forecasts.** Every incumbent arm under-forecasts the profit of its own pick by 11-20%; draws narrow it by 2-5
  points. MNP's forecast is within 3% (mean +0.5%).
- **Holdout fit.** Tuning over draws moves RFC's σ from 0.01-0.35 to 0.38-0.51 (rep 3 off the lower bound) and
  holdout share MAE by at most 0.0008; the holdout tasks barely separate the readouts.
- **Time and cost.** HB refits 3.5-4.7 min on Modal CPU (including 100 saved draws), tuning 1-4 min; the four
  exhaustive D1 searches (up to 184,756 candidates) take about 1 s each on the H100. About $1 in all.
