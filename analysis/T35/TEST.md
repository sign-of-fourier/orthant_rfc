# T35 · T20's RFC-S bias floor vs HB over posterior draws (2026-10-24): INCONCLUSIVE; draws halve the floor, do not remove it

## Why

T20: HB-MNL + RFC-S (point estimates) has a cost-regret floor of about 5% at every N from 50 to 1,500 on the
correlated truths; MVP at N = 50 is already below it. A floor that does not fall with N is what the point-estimate
readout would produce: each respondent's posterior uncertainty depends on their 12 tasks, not on N. At J = 60 HB
over posterior draws closed MNP's gaps (T31); on T15's assortment it did not (T34). T20's truths (G1, G2) have
error correlation HB-MNL cannot represent, as T34's.

## Design

- **Cells:** T20's G1 and G2 at N = 150, 600, 1,500, reps 0-9 (60 datasets), regenerated from T20's seeds.
- **HB refit:** T20's sampler, seed and length (bayesm, 20,000 iterations, keep 10), plus 100 draws per respondent
  evenly spaced over the second half.
- **Incumbent arms:**
  - `rfc_means`: T20's RFC-S on point estimates (T20's level draws and tuned σ, g), recomputed from the refit
    (sanity check: must reproduce T20's decisions).
  - `sop_means`: SoP on point estimates.
  - `sop_draws`: SoP over the 100 draws.
  - `rfc_draws`: RFC-S over the draws, 10 level draws per posterior draw (T20's 1,000 per respondent), σ and g
    tuned on the 6 holdout tasks as in T20.
- **MVP:** T20's fits and scores, unchanged. T20 ran MVP only at N = 50, 100, 200 (missed when this test was
  proposed), so the comparison is unpaired, against MVP at N = 50, which is T20's own claim (RFC-S at N = 1,500
  vs MVP at N = 50). Decisions and the metric as T20 (cost regret %, D-cost; truth by the
  orthant binary).

## Kill criteria (proposed by Claude, accepted by the user 2026-10-24)

Δ = best draws arm's cost regret − MVP's (as proposed: paired over the 10 reps of a cell; as run: against MVP at
N = 50, Welch SE).

- **FLOOR SURVIVES:** Δ > 2 paired SE and Δ ≥ 0.5 points in all 6 cells (both truths, all three N).
- **FLOOR IS THE READOUT:** Δ < 0.5 at N = 1,500 (either truth).
- **Else INCONCLUSIVE.**

## Results (2026-10-24): INCONCLUSIVE

**Files.** `t35.py`; `out/inc/*.json` (decisions per arm), `out/regrets.json`, `out/summary.json`; `run.log`.

Cost regret %, mean [95% CI] over 10 reps. MVP (T20): G1 1.56 [0.32, 2.79] / 0.17 / 0.61 at N = 50 / 100 / 200;
G2 0.63 [−0.39, 1.64] / 0.42 / 0.42.

| truth, N | RFC-S, point est. (T20) | SoP, point est. | SoP over draws | RFC-S over draws | RFC-S draws − MVP N = 50 (Welch p) |
|---|---|---|---|---|---|
| G1, 150 | 5.22 [2.07, 8.37] | 9.21 | 7.37 | 0.50 [−0.08, 1.07] | −1.06 (0.10) |
| G1, 600 | 4.77 [1.41, 8.14] | 9.21 | 9.21 | 4.02 [0.79, 7.25] | +2.46 (0.13) |
| G1, 1,500 | 6.45 [3.27, 9.63] | 9.21 | 9.21 | 6.03 [3.06, 8.99] | +4.47 (0.008) |
| G2, 150 | 6.24 [3.16, 9.32] | 8.91 | 8.02 | 3.97 [0.87, 7.06] | +3.34 (0.04) |
| G2, 600 | 4.46 [1.10, 7.82] | 8.91 | 8.91 | 3.77 [0.57, 6.96] | +3.14 (0.06) |
| G2, 1,500 | 5.35 [2.06, 8.64] | 8.91 | 8.91 | 2.87 [−0.14, 5.89] | +2.25 (0.14) |

- **Verdict by the rule: INCONCLUSIVE.** Not "survives" (G1 N = 150: RFC-S over draws ties MVP); not "readout"
  (at N = 1,500 RFC-S over draws is 4.5 and 2.3 points above MVP at N = 50).
- **Sanity check.** The refit reproduces T20's RFC-S decisions in all 60 datasets.
- **What the regret is.** D-cost has five options; a wrong pick costs about 9 points (or 1.7-4.5 for a near
  miss), so cell means are miss rates. Wrong picks (≈ 9 points) in 60 datasets: RFC-S on point estimates 35, RFC-S
  over draws 20 (Fisher p 0.01), MVP at N = 50-200 none (its misses are 1.7-4.5-point near misses, 6 of 60 at ≥ 4
  points; RFC-S over draws vs MVP, ≥ 4 points: 20 vs 6, Fisher p 0.003).
- **The floor stays flat in N with draws too.** Neither readout improves from N = 150 to 1,500; the incumbent's pick
  turns on σ and g tuned on 6 holdout tasks (19 parameter-at-bound flags over the 60 draws tunings). SoP, on either readout, gets the
  decision wrong on almost every dataset.
- **What this does to T20's claim.** Against HB over draws the floor about halves (≈ 5.4 → 3.5 points pooled) and
  is still above MVP: significant on G1 at N = 1,500 (+4.5, p 0.008), not on G2 (+2.3, p 0.14). "RFC-S at N = 1,500
  is worse than MVP at N = 50" holds on G1 and is unproven on G2; the unbounded sample-size ratio rests on G1.
- **Cost.** 60 HB refits with draws and RFC tuning on Modal CPU, about $2.
