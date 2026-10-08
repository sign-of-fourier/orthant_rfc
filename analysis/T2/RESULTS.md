# T2 results: linked-set scaling, simulated MBC (2026-10-08)

Run: `python analysis/T2/run.py` (4 truths × k ∈ {5, 8, 12} × R = 20 = 240 datasets,
N = 1,000 respondents × 10 menus of 12 items). All fits on Modal: CS/IL/IFM and Gibbs on CPU
containers, MVP-ML on the deployed `mvp_fit` GPU service (no SEs), scoring on CPU containers;
26 min wall. Raw: `out/full/` (`results.json`, `fits.json`, `log.txt`). Summary:
`python analysis/T2/summarize.py`. No errors; MVP-ML converged 240/240; Gibbs drift ≤ 0.002.

Decisions scored against the truth; probit decisions by GHK-QMC (the engine as a second
column, below). A flip is "material" when its regret exceeds 0.5%.

## Verdict

**PASS on decisions (MVP better); no time saving at k ≤ 12.** On the kill-rule pool (L1 + L2,
k ≥ 8; 80 datasets × 3 decisions), coded subsets flip 146 of 240 decisions (145 material);
every MVP estimator flips 27–29. The KILL rule needs both "no decision difference" and "no
time saving"; the first fails by a wide margin, so T3 goes ahead. Kill blanks unset.

| Pool (L1+L2, k ≥ 8) | flips / 240 (material) | TURF-4 regret | triple regret | median time |
|---|---|---|---|---|
| CS-F (analyst subsets) | 146 (145) | 1.24% | 5.39% | 9 s |
| CS-O (oracle subsets) | 161 (160) | 0.85% | 11.93% | 11 s |
| IL (independent logits) | 139 (139) | 3.19% | 8.07% | 0 s |
| MVP-IFM | 28 (28) | 0.02% | 0.33% | 9 s |
| MVP-ML (GPU) | 29 (29) | 0.02% | 0.33% | 113 s |
| MVP-Gibbs | 27 (27) | 0.01% | 0.33% | 359 s |

## Q1: coded subsets fail exactly when the linked set exceeds the cap

Flips (pair / triple / TURF-4, of 20) and pair-joint MAE (points):

| truth, k | CS-F | CS-O | MVP-IFM | MVP-ML | MVP-Gibbs |
|---|---|---|---|---|---|
| L1, 5 | 0/0/0, 0.33 | 0/0/0, 0.33 | 0/0/2, 0.25 | 0/0/2, 0.25 | 0/0/3, 0.24 |
| L1, 8 | 2/20/19, 1.33 | 20/20/16, 1.29 | 0/2/1, 0.21 | 0/2/1, 0.21 | 0/2/0, 0.20 |
| L1, 12 | 20/20/4, 3.40 | 20/20/4, 3.40 | 8/10/0, 0.24 | 9/10/0, 0.23 | 9/9/0, 0.23 |
| L2, 5 | 0/0/0, 0.29 | 0/0/0, 0.29 | 0/0/3, 0.24 | 0/0/3, 0.24 | 0/0/4, 0.24 |
| L2, 8 | 0/0/1, 0.72 | 0/0/1, 0.72 | 0/0/0, 0.22 | 0/0/0, 0.22 | 0/0/0, 0.21 |
| L2, 12 | 20/20/20, 2.43 | 20/20/20, 2.40 | 4/2/1, 0.25 | 4/2/1, 0.25 | 4/2/1, 0.24 |

- With k = 5 (L1, L2) or two clusters of 4 (L2, k = 8) the dependence fits inside ≤ 5-item
  subsets and coded subsets are as good as MVP. Once a cluster exceeds 5 (L1 k = 8, 12; L2
  k = 12), the cut pairs are treated as independent and the joint-dependent decisions flip
  in nearly every replicate (systematic, not noise). Pair-joint error grows 0.3 → 3.4 points.
- Knowing the true linked set does not help (CS-O ≈ CS-F, sometimes worse): the cap, not
  the analyst's rule, is the problem.
- MVP's error stays at 0.2–0.25 points for every k. Its remaining L1 k = 12 flips are
  near-ties in the truth (regret 0.7–0.9%).
- Holdout pattern log-likelihood: MVP better wherever subsets are cut (L1 k = 12: −6.01 vs
  −6.35; L2 k = 12: −5.78 vs −6.00), equal otherwise.

## Guards

- **L0 (no correlation).** MVP does not invent dependence (median |ρ̂| 0.012). It does flip
  more near-tie decisions than CS (e.g. k = 5: 6/9/5 vs 2/5/0, regret ~1%): the cost of
  estimating 66 near-zero correlations. Gibbs (own-price margins) shows the same, so it is
  the correlations, not the cross-price terms.
- **LC (complementarity; CS is the right model, MVP is not).** CS-O is best (k = 5: 0 flips).
  MVP flips the best triple in 9–11 of 20 at k = 5 (regret 5–6%), but CS-F flips it in 20 of 20
  (17%): the analyst's lift rule mis-groups the triple. At k = 8, 12 nothing is material.

## Q2: estimators

IFM, full ML and Gibbs give the same decisions (28 / 29 / 27 pool flips) and the same joint
errors. Full ML buys nothing here, as expected with correctly specified linear margins and
no heterogeneity. Gibbs's slightly lower holdout errors come from its own-price-only margins
(fewer parameters), not from the estimator. IFM is the fastest by 20–80×.

## Q3: speed

Median seconds (fit | decisions by QMC | decisions by engine):

| k | CS-F | MVP-IFM | MVP-ML (GPU service) | MVP-Gibbs |
|---|---|---|---|---|
| 5 | 2.7 | 4.5 \| 4.7 \| 0.009 | 85.6 \| 4.6 \| 0.009 | 351 \| 4.6 \| 0.009 |
| 8 | 2.3 | 4.7 \| 4.8 \| 0.009 | 97.6 \| 4.7 \| 0.009 | 355 \| 4.7 \| 0.009 |
| 12 | 5.7 | 5.0 \| 4.6 \| 0.009 | 122 \| 4.6 \| 0.009 | 352 \| 4.6 \| 0.009 |

- No time saving at k ≤ 12: coded subsets are cheap *because* the cap stops them modelling
  the dependence. MVP-IFM costs the same and gets the decisions right.
- The engine computes all decision quantities ~500× faster than QMC (0.009 vs 4.7 s per
  model) with 3 extra pool flips (31 vs 28). That matters for decision search over many
  scenarios (assortment, pricing grids), not for one decision set.
- MVP-ML time is the hosted service's (queue + container + fit). Its default SE step took
  ~18 of ~20 min per fit at 222 parameters; T2 ran without SEs (tracked as a service issue).

## Caveats

1. Kill blanks unset; the decision gap is large (146 vs 28 flips), the speed claim fails.
2. No respondent heterogeneity (follow-up). Sawtooth fits MBC sub-models by HB.
3. Truths L0–L2 are probit; LC is the only truth where CS is the right model.
4. The analyst rule (lift > 1.2 or < 0.8, cap 5) is ours, not Sawtooth's documented rule.
5. Design revisions during the build (D-TURF for D-promo, QMC scoring, own-price Gibbs):
   DESIGN.md, "Revisions".
6. Engine accuracy: up to 2 points off in 4-item reach at correlations ~0.8, and 0 for some
   12-dim patterns near 1e-4 (known envelope); hence QMC for scoring.
