# Conjointly TURF Experiment 7

**Status:** paused (Oct 2026)
**Script:** `orthant_rfc/analysis/conjointly_turf/turf_bootstrap.py` (numpy/pandas, seed 20261015). The output is in `output.txt` and `cluster_assignments.csv`. The Conjointly export is gitignored and stays out of the public repo.

## Purpose

This is a side-by-side test of an incumbent choice-modeling platform (Conjointly) on its own public demo study. It asks three questions:

1. What do the standard MaxDiff and TURF outputs tell a client?
2. What do they leave out?
3. Where would a model of the dependence structure (correlated preferences, with MVP as the end goal) change or sharpen the conclusions?

## Data

Conjointly's public MaxDiff demo is a color-preference study. It has 14 items (Red, Black, Orange, Yellow, Green, Blue, Purple, Violet, Maroon, Navy, Turquoise, Lime, White, Silver), shown in sets of 4, with N = 202 respondents. We used four exports.

- **Aggregate report JSON**
  - Zero-centered utilities with CIs
  - Probability-rescaled scores (baseline 0.25 for a set of 4)
  - All-14 logit preference share and first-choice share (neither has a CI)
  - Best/worst counts
- **Utility bar chart.** Used to map item IDs to color labels.
- **The TURF tool**
  - Reach defined as "item in respondent's top-k", computed on point-estimate utilities
  - Constraint options: must-include, any-of, exclude, prohibited pairs, "at least one high-ranking item"
- **Individual-level utilities**
  - One HB point estimate per respondent per item, with Red fixed at 0
  - Not posterior draws, and not the raw best/worst choices

## What we did

1. Read the aggregate outputs. Mapped IDs to colors and compared utility, logit share and first-choice share item by item.
2. Ran Conjointly's TURF.
   - A 2-item, top-1 run as a sanity check.
   - A 3-item threshold sweep at k = 2, 3, 4, with no constraints, so the model rather than analyst inputs drove the result.
3. Reproduced the TURF from the individual utilities. All shares, reach values, frequencies and ranks matched Conjointly to within 5e-8. No ties in anyone's top 6 meant tie handling didn't matter.
4. Bootstrapped respondents (B = 5000). At each k, enumerated all 364 triples per replicate and recorded:
   - P(win) for each triple
   - Reach CIs
   - Paired CIs on the reach gap between #1 and the other top-10 triples
5. Estimated the between-person preference structure.
   - Row-centered utilities, with and without per-respondent SD scaling.
   - Correlations compared to the −1/13 baseline that centering induces.
   - Average-linkage hierarchical clustering, checked against scipy.
6. Tested three structural predictions. All three had been made earlier by reading only Conjointly's aggregate outputs. The spec gave no pass rules for the Lime and Blue–Navy tests; the rules used were set during the analysis.

## Results

### Conjointly's aggregate outputs

- **Navy has the top mean utility but is rarely anyone's favorite.** Its utility is 0.44 and its first-choice share is 8.9%.
- **Black is polarizing but wins first choice.** 41% best and 18% worst; first choice for 27.7%.
- **Blue is the anomaly.**
  - Its utility (0.20) is similar to Maroon and Turquoise.
  - Its first choice is 1 of 202 respondents, against roughly 10–13% for those peers.
  - Its logit share is 4.7%, about 9× its first-choice rate. That gap is far larger than for items of similar utility, which is the IIA signature.
- **No uncertainty on decision outputs.** Utilities have CIs; shares and first choice do not. At N = 202, the middle items' first-choice shares are statistically indistinguishable, and the output doesn't say so.

### TURF threshold sweep (3 items)

| k | #1 triple | Reach | Lead over #2 |
|---|---|---|---|
| 2 | Black, Purple, Turquoise | 74.3% | 5.0 pts (~10 people) |
| 3 | Black, Purple, Turquoise | 88.1% | 3.0 pts (~6 people) |
| 4 | Black, Purple, Turquoise | 94.1% | 1.0 pt (2 people) |

The #2 triple is {Black, Navy, Turquoise} at every k. The winner is stable across thresholds, but its lead shrinks as k grows. At k = 4 the top 50 triples all fall between 84% and 94% reach.

### Bootstrap uncertainty (sampling only)

| k | P(#1 wins) | Runner-up P(win) | Triples tied with #1 | Paired gap to #2 (95% CI) |
|---|---|---|---|---|
| 2 | 0.82 | 0.05 | 4 | 0.050 (−0.015, +0.109) |
| 3 | 0.72 | 0.11 | 8 | 0.030 (−0.025, +0.084) |
| 4 | 0.53 | 0.24 | 13 | 0.010 (−0.025, +0.045) |

No paired gap to #2 excludes zero. These figures understate the uncertainty, because they ignore within-person posterior uncertainty.

### Preference structure

The four clusters are identical under both scalings:

| Cluster | Items |
|---|---|
| Neutral | Black, Maroon, White, Silver |
| Blue-green | Green, Blue, Navy |
| Cool | Purple, Violet, Turquoise |
| Warm | Red, Orange, Yellow, Lime |

Notable correlations:

- Positive: Blue–Navy +0.70, Black–Silver +0.69, Navy–Black +0.53.
- Negative: Navy–Violet −0.49, Lime–Black −0.54.
- Black is negatively correlated with Violet and Turquoise.

### Prediction tests

| Prediction (from aggregate outputs) | Outcome |
|---|---|
| Blue is a near-duplicate of Navy (from the first-choice anomaly) | **Pass, partial.** Navy ranks above Blue for 74% of respondents, and for 80% of the 50 who have both in their top 4. But Blue still adds 13 people to {Black, Navy}, against 17 to {Black, Turquoise}, so it is a correlated sibling, not a pure duplicate. |
| Dark/cool cluster: Navy overlaps with Purple and Violet (from the frequency column) | **Fail.** Navy–Purple is +0.06 and Navy–Violet is −0.49. Navy's overlap runs through Black: 61% of Navy-top-3 respondents also have Black in their top 3, against 46% overall. |
| Lime niche (from Lime's appearance in the k = 4 TURF) | **Fail, effectively.** Lime has the 4th-highest spread, and 12.4% (25 people) have it in their top 4. Of those 25, 6 are missed by the winning triple at k = 3: 24%, against 11.9% overall, about 2×, but only 6 people. The niche test also required Lime to be negatively correlated with all four core items, and Lime–Turquoise is +0.17. |

### Why the winning triple wins

- **Black anchors.** It is first choice for 28% and in the top 3 for 46%.
- **Purple and Turquoise reach the people Black misses.** Both are negatively correlated with Black. Of the 110 respondents without Black in their top 3, the pair reaches 86, with 30 reached by both.
- **Navy duplicates Black's audience.** It reaches only 34 of those 110. Because it is positively correlated with Black, Navy triples score higher frequency but lower reach. At k = 3, {Black, Navy, Turquoise} reaches 72 people with two or more of its items, against 47 for the winner.

The winner takes two items from one cluster and none from blue-green or warm. Correlation with the anchor, not cluster membership, explains the ranking. A "one item per cluster" rule would pick {Black, Navy, Turquoise}, the runner-up, which is statistically tied with #1 at every k. At N = 202 the distinction explains the ranking but does not change the decision.

## Interpretation

- **Conjointly's TURF gets the answer right.** Counting reach on individual-level utilities already captures within-person redundancy. The winning set is stable across thresholds. We should say this plainly in any comparison.
- **It is wrong about confidence.** It presents a clean ranking where the top triple wins only 53–82% of the time, depending on an arbitrary threshold, with 4 to 13 statistical ties. Because within-person uncertainty is ignored, the true P(win) is probably lower and the tie counts higher. The output has no uncertainty on any decision-level quantity.
- **It can't explain the result.** The reasons the set wins (an anchor plus anti-correlated items, with Navy's overlap running through Black) are invisible in the standard outputs.
- **Reading structure from aggregate outputs is unreliable.** Of three structural calls made from Conjointly's outputs, one held (Blue–Navy) and two failed. The frequency column and the marginals invite stories that the correlation structure contradicts. This is the strongest honest argument for estimating the dependence structure rather than inferring it.
- **Redundancy is graded, not binary.** Blue–Navy is a strong but partial overlap. A fitted Σ represents that; Conjointly's "prohibited pairs" toggle can only switch it on or off, and only if the analyst already knows to.
- **IIA hurts the share simulator, not the TURF.** Blue's inflated logit share is the IIA symptom. TURF on individual utilities is largely immune to it.

## Conclusions

- **Pitch:** "right on average, wrong about confidence, silent on why." Lead with P(#1 wins) and the tie band, then the explanation from the correlation structure. Don't claim the incumbent picks the wrong set; on this data it doesn't.
- **Client-facing finding:** correlation with the anchor, not cluster diversity, explains why a portfolio wins. On this data it does not change the choice, since the cluster-diverse set is statistically tied with the winner.
- **Process lesson:** test structural hypotheses on respondent-level data before putting them on a slide. Two of the three intuitive reads were wrong.

## Limitations

- **Point estimates only.** The bootstrap captures respondent sampling, not within-person uncertainty, so all P(win) values and tie counts understate the true uncertainty.
- **The utilities are HB-MNL-derived.** They already embed logit assumptions. The correlation structure is estimated after the fact, not inside the choice model.
- **MVP has not been tested here.** Fitting MVP and comparing it with HB-MNL needs the raw best/worst choices, which these exports don't include.
- **This is one demo study,** with color stimuli and N = 202. It is not a commercial dataset.

## Split-half test: MVP reach vs counting (design fixed 2026-10-15, before running)

**Question.** Conjointly counts reach on each respondent's point estimates. Does a fitted correlation structure (MVP, as in T6) predict reach for new respondents better than counting, and does it pick a triple with higher held-out reach?

**What it compares.** Reach estimators only. Both arms start from the same HB-MNL point estimates, so this is not MVP against HB-MNL.

**Data.** For each k in {2, 3, 4}, each respondent's top-k membership as 14 binary indicators (from the Step 1 rankings).

**Design.**
- R = 200 random splits of the 202 respondents into two halves of 101 (fixed seed). Train on one half, score on the other.
- Arms, all fitted on the training half:
  - **Count** (Conjointly's method): reach(S) = share of training respondents with at least one item of S in their top-k.
  - **MVP**: intercept-only IFM fit on the training indicators (`t6.fit_ifm`: probit margins, pairwise ρ from each 2×2 table, nearest valid correlation matrix), reach(S) = 1 − P(none of S in top-k), one 3-dim orthant per triple by GHK (scrambled Sobol, M = 4,096; T6's integrator, run locally on CPU).
  - **Independence** (reference only, not in the pass rule): 1 − ∏(1 − p̂_j) on the same margins.
- Target: observed reach in the held-out half.

**Measures (per k).**
- **Prediction:** mean absolute error over all 364 triples between training estimate and held-out reach, averaged over splits, in reach points.
- **Decision:** each arm's top triple on the training half, scored by held-out reach. Report how often the picks differ, and the mean held-out reach difference (MVP pick − Count pick).
- **Uncertainty:** splits overlap, so naive across-split SEs are too small. CIs use the corrected resampled t-test (Nadeau and Bengio 2003): variance × (1/R + n_test/n_train). Holm across the three k.
- **Materiality:** 1 respondent in the held-out half, about 1 reach point.

**Pass rule (fixed now).**
- **PASS:** MVP's prediction error is lower than Count's with Holm-adjusted p < 0.05 at 2 or more of the 3 k, Count is not significantly better at any k, and the MVP pick's held-out reach is not significantly worse than the Count pick's at any k.
- **FAIL:** Count is significantly better on prediction at any k, or the MVP pick is significantly and materially worse at any k.
- **INCONCLUSIVE:** anything else.

**Known caveats.**
- Misspecification: every respondent has exactly k items in their top-k, and MVP can't represent a fixed row sum.
- Small N: 101 per half, with gaps of 1–5 points at the top.
- Held-out reach is itself noisy; that noise is common to both arms, so it widens the CIs but doesn't bias the comparison.

**Script:** `analysis/conjointly_turf/mvp_reach.py`, output `mvp_reach_output.txt`. CPU only, $0.

### Result (2026-10-15): INCONCLUSIVE

R = 200 splits, 129 s on CPU. GHK checked against SciPy on one test orthant (difference 3e-6).

**Prediction:** mean absolute error vs held-out reach, all 364 triples, in reach points.

| k | Count | MVP | Independence | Count − MVP (95% CI) | Relative lift (95% CI) | Holm p |
|---|---|---|---|---|---|---|
| 2 | 5.16 | 5.36 | 5.50 | −0.20 (−0.89, +0.50) | −4.6% (−19.6, +10.4) | 0.68 |
| 3 | 5.17 | 5.55 | 6.41 | −0.38 (−0.99, +0.24) | −8.2% (−21.9, +5.6) | 0.68 |
| 4 | 4.83 | 5.12 | 6.50 | −0.29 (−0.82, +0.25) | −6.4% (−18.5, +5.7) | 0.68 |

**Decision:** each arm's top triple on the training half, scored by held-out reach.

| k | Picks differ | Count pick | MVP pick | MVP − Count (95% CI) | Holm p |
|---|---|---|---|---|---|
| 2 | 91 / 200 | 72.1 | 68.9 | −3.2 (−13.2, +6.7) | 1 |
| 3 | 82 / 200 | 85.2 | 84.6 | −0.6 (−9.0, +7.8) | 1 |
| 4 | 91 / 200 | 91.2 | 92.1 | +0.9 (−5.4, +7.2) | 1 |

**Reading.**
- No difference is significant. MVP's point estimates are slightly worse than counting on prediction at every k (0.2–0.4 points), and both beat independence at k = 3 and 4.
- The picks differ in 41–46% of splits, but the held-out reach of the two picks can't be told apart. With 101 respondents per half, a single split moves held-out reach by several points.
- **The MVP fit is degenerate here.** Many pairs never co-occur in a training half's top-k (the fixed row sum makes this common), so their IFM ρ sits at the −0.99 bound: on average 51 of 91 pairs per fit at k = 2, 23 at k = 3, 10 at k = 4. The nearest-correlation repair then moves correlations by a median of 0.89, 0.77 and 0.57. The MVP arm is therefore mostly the repair, not a fitted dependence structure, so this run is not a fair test of MVP reach.
- Top-k indicators from 14-item rankings are a poor input for a binary MVP. A fair comparison needs the raw best/worst choices, as in Next steps.

## Next steps (when resumed)

1. Check whether Conjointly's demo exports raw best/worst responses.
2. If it doesn't, run a small MaxDiff on Conjointly. That yields both their outputs and the raw data, and makes a reproducible, publishable dataset (a candidate for the Sawtooth 2028 submission).
3. Fit MVP on the raw choices.
   - Compare P(#1 wins) using full posterior uncertainty.
   - Check whether the four-cluster structure and the Blue–Navy dependence are recovered at fit time.
   - Compare share-simulator behavior on Blue/Navy against Conjointly's logit share.
4. Keep the Conjointly data out of the public orthant_rfc repo. Commit the script and summary results only.
