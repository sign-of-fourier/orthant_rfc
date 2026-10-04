**Conclusion: the error is biased, so the regime below should be flagged.**

`orthant_cdf` underestimates choice shares. The bias is nonzero in every group
(clustered t-test p ≤ 1e-5):

- −0.0013 with an exact twin or a near-twin, and −0.0004 with no similar
  products.
- The 5 shares of a scenario sum to 0.995 on average, and as low as 0.969.
- The bias grows with share size: about −0.002 for shares above 0.3, against
  −0.0003 below 0.05.
- It grows with the largest correlation of the difference covariance. In the
  sweep it was −0.006 to −0.007 for correlations of 0.9–0.99, with errors up to
  0.044.

It does not average out. Across 1,000 respondents the market-share error
stayed at 0.0014 on average (0.0069 at worst, in a twin scenario). That is 3
to 120 times what independent noise would leave, and almost as large as the
per-respondent error of 0.0018.

The 0.018-size worst case is not in the twin group. Here the worst share
(0.019) was in a near-twin scenario, on a product that is *not* part of the
pair and has a true share of 0.52. Within twin and near-twin scenarios the pair
members' error is about a third of the other products' error.

**Flag:**

- Large shares (above about 0.15) in scenarios with a twin or near-twin.
- Any scenario whose difference covariance has a correlation of 0.9 or more.

Outside that regime the error is a small but systematic underestimate (about
−0.0004, at most 0.007). That is acceptable at a tolerance of about 0.01 but
not at 0.001. One possible correction is to renormalize each scenario's shares
to sum to 1; it is untested here.

**After renormalizing** (final section): shares rescaled to sum to 1 halve
both the market-share error (at most 0.0035 at 1,000 respondents) and the
largest share error in the stratified sets (0.009). A residual bias of about
0.001 remains: in scenarios with a twin or near-twin the pair members come out
too high and the other products too low, and this does not average out across
respondents. At max correlation ≥ 0.9 the error stays large (sd 0.0065,
max 0.034). Later sections use renormalized `orthant_cdf` shares, cite this
table, and flag max correlation ≥ 0.9.
