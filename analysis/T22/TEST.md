# T22 · Swissmetro, real stated choice: mixed MNP (MVP) vs HB-MNL + RFC-S vs nested logit (2026-10-19)

Status: **done. MVP has the best holdout log-likelihood at every training size, but the margin is small: +0.001 to
+0.043 nats per choice over RFC-S, with 5 of 7 CIs above 0. On binned shares, MVP beats RFC-S (5 of 7 CIs above 0)
and ties plain HB-MNL. The error correlation that motivates MVP is not there: the fitted rail (train + SM) error
share is 0 in all 7 fits, once random tastes are in. There is no T20-style floor on this data.**

## Why

This is the first real *designed* choice data in the series. The "Small CBC" row rests on simulation (T5, T20).
Swissmetro is the textbook real stated-choice panel, and the textbook nested logit case: rail and Swissmetro are
the similar pair. It tests MVP against Sawtooth's pipeline and the structural incumbent on real respondents.

## Data and split

- **Source.** The EPFL Swissmetro SP survey (`transp-or.epfl.ch/data/swissmetro.dat`, 1998), in `data/` (not
  committed).
- **Sample.** Commute and business trips (PURPOSE 1 or 3), CHOICE > 0, and a car available (so J = 3 in every
  task). That leaves 623 respondents × 9 tasks. Choices: train 8%, SM 60%, car 32%.
- **Variables** (same in every model).
  - ASC train and ASC SM (car is the reference).
  - Time / 100 min and headway / 100 min (car has no headway).
  - Cost / 100 CHF, set to 0 for train and SM when the respondent holds a GA season pass (the Biogeme convention).
- **Split.**
  - 25% of respondents (155) are held out and scored on all 9 tasks. Each model predicts with *population*
    probabilities, as a market simulator does for people not in the sample.
  - The training respondents' tasks 1–7 fit every model. Their tasks 8–9 are used only to tune RFC-S (Sawtooth's
    holdout-task tuning).
  - Training sizes: all 468 training respondents, plus 3 random subsets each of 100 and 200.

## Models

| model | what it is | fit |
|---|---|---|
| MNL | pooled logit | scipy, < 1 s |
| NL | pooled nested logit, nest {train, SM}, dissimilarity λ | scipy, < 1 s |
| HB-MNL | `bayesm::rhierMnlRwMixture`, 20,000 iterations (T1's `hb.R`); population probabilities average the respondents' posterior means | 10–43 s CPU |
| RFC-S | Sawtooth RFC on the HB estimates (details below) | tuning 18–448 s CPU |
| MVP | mixed MNP (details below) | 4–14 s on a T4, + 1–6 s SEs |

**RFC-S.** Swissmetro's attributes are alternative-specific and continuous, not shared levels as in T1. So the
attribute error is on each coefficient and shared by the task's alternatives (alternatives with similar time or
cost become correlated). The ASC errors are per alternative, and the product error is Gumbel with scale g. One
σ_l covers everything. It is tuned by maximising the population log-likelihood of tasks 8–9, using 100 draws per
respondent. No fit hit a bound. σ_l came out at 1.6–2.8, so unlike T20, tuning does not collapse RFC to plain HB.

**MVP.**
- Normal tastes with a diagonal sd on all 5 coefficients.
- Error = a rail component shared by train and SM + an iid component. The total is fixed at π²/6 (logit scale).
- Fit: the exact panel likelihood by GHK (T1's `ghk_gpu.fit`, M = 4,096). Each respondent's 7 tasks are one
  14-dimensional orthant.
- Holdout probabilities are exact bivariate orthants.

## Results (held-out respondents; positive differences = MVP better)

### Log-likelihood per choice

| train | MNL | NL | HB-MNL | RFC-S | MVP | MVP − NL | MVP − HB-MNL | MVP − RFC-S |
|---|---|---|---|---|---|---|---|---|
| all 468 | −0.735 | −0.732 | −0.716 | −0.709 | **−0.687** | +0.045 [+0.010, +0.096] | +0.029 [+0.005, +0.059] | +0.022 [+0.012, +0.032] |
| 200 s0 | −0.744 | −0.741 | −0.707 | −0.715 | **−0.689** | +0.052 [+0.020, +0.090] | +0.018 [−0.001, +0.039] | +0.026 [+0.013, +0.037] |
| 200 s1 | −0.731 | −0.729 | −0.709 | −0.708 | **−0.694** | +0.035 [−0.001, +0.103] | +0.015 [−0.010, +0.046] | +0.014 [+0.005, +0.024] |
| 200 s2 | −0.740 | −0.737 | −0.708 | −0.718 | **−0.691** | +0.045 [+0.010, +0.096] | +0.017 [−0.003, +0.037] | +0.026 [+0.014, +0.038] |
| 100 s0 | −0.756 | −0.756 | −0.722 | −0.743 | **−0.700** | +0.056 [+0.025, +0.102] | +0.022 [+0.008, +0.037] | +0.043 [+0.029, +0.057] |
| 100 s1 | −0.746 | −0.732 | −0.714 | −0.702 | **−0.702** | +0.031 [−0.011, +0.096] | +0.012 [−0.025, +0.073] | +0.001 [−0.012, +0.012] |
| 100 s2 | −0.759 | −0.750 | −0.742 | −0.700 | **−0.696** | +0.054 [+0.016, +0.097] | +0.046 [+0.013, +0.084] | +0.005 [−0.007, +0.015] |

- **MVP is best in all 7.**
  - Over RFC-S the gain is 0.001–0.043 nats per choice, with 5 of 7 CIs above 0. Over HB-MNL it is 0.012–0.046,
    with 3 of 7 CIs above 0.
  - For scale, 0.02 nats per choice is about 0.2 nats per respondent over 9 tasks: a real but modest
    predictive edge.
- **The pooled logits (MNL, NL) are worst.** They ignore heterogeneity, so they are not a fair structural
  incumbent. A mixed nested logit would be.

### Share calibration: MAE over 9 bins × 3 alternatives (pts)

The bins are tertiles of (SM cost − train cost) × tertiles of (car time − train time).

| train | MNL | NL | HB-MNL | RFC-S | MVP | HB-MNL − MVP | RFC-S − MVP |
|---|---|---|---|---|---|---|---|
| all 468 | 6.82 | 6.84 | **4.63** | 6.95 | 5.34 | −0.71 [−1.24, +0.38] | +1.61 [+0.66, +2.19] |
| 200 (3 subsets) | 6.5–7.5 | 6.5–7.6 | **5.1–5.4** | 6.8–7.4 | 5.6–6.0 | −0.73 to −0.40, all CIs span 0 | +0.75 to +1.85, all CIs above 0 |
| 100 (3 subsets) | 7.0–8.7 | 7.0–8.8 | **4.6–6.6** | 5.6–8.8 | 4.8–6.4 | −1.26 to +0.21, 1 of 3 CIs below 0 | +0.41 to +2.38, 1 of 3 CIs above 0 |

- **MVP vs RFC-S.** MVP beats RFC-S by 0.4–2.4 pts. RFC tuning makes the shares *worse* than plain HB in all 7
  fits: it fits the tuning tasks' log-likelihood by adding noise, which flattens the shares.
- **MVP vs plain HB-MNL.** HB-MNL is the best-calibrated model on binned shares in 6 of 7 fits, by 0.2–1.3 pts.
  Only 1 of 7 CIs excludes 0, so this is a tie, leaning HB.

### Policy query: SM cost × 1.2 (no ground truth)

| | MNL | NL | HB-MNL | RFC-S | MVP |
|---|---|---|---|---|---|
| SM share change (pts), all 468 | −5.2 | −5.6 | −8.0 | −6.4 | −6.4 |
| share of SM's loss going to train: all 468 / range over subsets | 20% / 19–28% | 27% / 25–68% | 21% / 11–31% | 30% / 15–36% | 20% / 14–28% |

The mixed models and MNL agree that about a fifth of SM's lost share goes to train. The nested logit says more,
and its answer is unstable at small N (λ ranges from 0.22 to 1.00 across the 100-respondent subsets).

### The rail correlation

MVP's fitted rail error share is 0 in all 7 fits (sandwich SE of its log-ratio 0.24–1.9). The random ASCs carry
the structure instead: taste sd 3–5 on ASC train and 2.6–3.6 on ASC SM. Within a choice, train and SM errors are
not correlated once persistent preference for each mode is modelled. The pooled NL's λ = 0.75 reflects
heterogeneity it cannot otherwise represent, not within-task similarity. This is the stated-choice version of
T16's finding: heterogeneity, not error correlation, carries the dependence.

## Reading

- **On real designed choice data, MVP predicts held-out respondents best.** The gain is a few hundredths of a nat
  per choice. It beats Sawtooth's pipeline (HB-MNL + RFC-S) on log-likelihood and on binned shares.
- **But the mechanism T1/T5/T20 rely on is absent here.** There is no within-task error correlation for RFC to
  get wrong, and RFC-S does not show T20's bias floor. Its deficit comes from tuning noise, and plain HB-MNL
  calibrates shares as well as MVP.
- **For the "Small CBC" row,** this is supportive but weak real-data evidence: MVP ≥ the incumbent at N = 100–468.
  It is not a replication of T20's unbounded sample-size ratio. That ratio needs crossed similarity (shared
  levels) that Swissmetro's three modes do not have.

## Limitations

- **J = 3, one similarity pair, alternative-specific attributes.** RFC's attribute-error construction had to be
  adapted (see Models); it is not Sawtooth's exact level-based form.
- **The structural incumbent is pooled.** A mixed nested logit (or HB with correlated ASCs, which bayesm's full
  covariance already allows) is the fairer version. MVP's taste covariance is diagonal, while HB's is full.
- **No ground truth for the policy query.** SP data from 1998, and the car-available subsample only.
- **The bootstrap is over held-out respondents,** with the models held fixed.

## Files and cost

- `t22.py`: `local` fits MNL, NL, HB-MNL and RFC-S, writing `out/local_*.json` and `out/hb_*.npy`. About 15 min of
  local CPU.
- `t22_modal.py` (`t22.py mvp`): MVP on Modal T4s, writing `out/mvp_*.json`. 7 fits of 4–14 s each, with 1–6 s
  for SEs; under $0.05.
- `t22_score.py` (`t22.py score`): writes `out/scores.json` and `out/report.md`.
