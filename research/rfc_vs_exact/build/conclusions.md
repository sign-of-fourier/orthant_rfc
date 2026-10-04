## Where RFC is good enough, and where it is not

These findings come from the outputs above, on this synthetic design (K = 5,
10 part-worths, σ_P = 0.5) and one seed. "Exact" means the exact
probabilities of model (a). `orthant_cdf` computes them approximately, and
its error against SciPy is listed separately.

**RFC is good enough when:**

- **The quantity is a market share averaged over many respondents or draws.**
  The Monte Carlo noise averages out across respondents, so with 1,000
  respondents about 250 draws per (respondent, draw) is enough for 0.1 share
  points. At 1,000 respondents × 500 draws × 10 scenarios, RFC took 205 s
  against 130 s for the exact shares. Its posterior-mean shares were within
  0.004 of exact. At this scale the two cost about the same.
- **Duplicates and near-duplicates, with normal product noise.** RFC (a)
  matched the exact combined twin share to 0.0002 and the near-twin share
  difference to 0.0002 (R = 10⁶). RFC's handling of duplicates belongs to the
  model and is not a Monte Carlo artifact.
- **Cross-price effects, if the simulator uses common random numbers** (a fixed
  seed shared by the base and changed scenarios). At R = 10⁵ with common
  draws, the matrix error was 0.0002 and no off-diagonal entry had the wrong
  sign. Even at R = 10⁴ the error was 0.0008, with 1.5% wrong signs.

- **Picking the single best product line, with common random numbers.**
  R = 10³ picked the same best line of 10⁶ as `orthant_cdf` (profit gap 0) in
  46 s, against 41 s for `orthant_cdf` (`product_line_fair.ipynb`). That line
  has near-duplicate products (difference correlation ≥ 0.9), where
  `orthant_cdf` is least accurate; in section 7, SciPy agreed on it.

**RFC is not good enough when:**

- **Each individual share must be accurate to 0.1 points.** That takes about
  250,000 draws per scenario. At 10⁶ respondents it is about 4 hours
  (extrapolated), against 15 s for `orthant_cdf`. Across the 150 test sets,
  10⁶ draws still left a worst-case share error of 0.0015.
- **Cross-price effects with independent draws per scenario.** At R = 10⁵,
  15% of the off-diagonal entries still had the wrong sign. These are the
  small entries (below 0.001), but a cannibalization table with "negative
  substitution" would be wrong.
- **Ranking product lines beyond the winner.** With common random numbers
  (`product_line_fair.ipynb`), R = 10⁴ draws took 436 s for 10⁶ lines and still got
  only 8 of the top 10 (Spearman 0.82 on the top 1,000). `orthant_cdf` took
  41 s, so any RFC setting that gets 9 of the top 10 together with the
  winner takes more than 10.6 times as long. At 0.1-point accuracy per share
  (R = 250,000, independent draws), RFC would take about 5 hours
  (extrapolated).
- **Fitting the variances.** On data from model (a), maximum likelihood on
  exact probabilities with a diagonal Σ_A reached holdout MAE 0.0136, against
  0.0129 at the true parameters and 0.019–0.020 for grid-searched RFC.
  - With the same two parameters as the grid, the exact fit reached 0.0158.
    So part of the gain comes from fitting more parameters, which a grid
    cannot do, and part from the coarse 8×8 grid: it picked (0.35, 0.47)
    where the optimum was near (0.40, 0.54).
  - On data from model (b), where the exact model is misspecified, the
    two-parameter exact fit and both RFC grids were tied (0.0165–0.0169).
    The diagonal exact fit was still best (0.0137).

**Other findings:**

- **Which error distribution RFC uses matters more than how many draws.**
  RFC (b), with Gumbel product noise, differs from model (a) by 0.0035 per
  share on average (0.014 at most), and 0.007 on combined twin shares. Beyond
  about R = 3·10⁴ the change of distribution dominates the Monte Carlo noise.
  Neither is "right"; they are different models.
- **`orthant_cdf` is not exact.**
  - On per-row covariances at d = 4 its share error was 0.0007 (median) and
    0.018 (max). That is better than RFC at R = 10⁴ but worse than RFC at
    R = 10⁶.
  - On product-line profit at d = 5 its error was up to 0.019, against a
    0.079 spread across the top 50 lines. The SciPy re-scoring agreed on the
    top six lines and on the winner.
  - Where every share must be within about 0.005, check it against SciPy.
- **HB section.** The interval widths follow from the synthetic population-mean
  uncertainty (sd 0.08) and are not a finding. Using posterior means only moved
  shares by 0.0095 on average (0.032 at most), with no ranking reversals among
  100 pairs. 22 of the 100 pairs had a 90% interval of the share difference
  that included 0. A point-estimate simulator cannot report that.

