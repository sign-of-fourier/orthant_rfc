**Verdict: a residual bias of about 0.001 remains at every correlation, and
for max correlation ≥ 0.9 the error is large as well.** Below 0.9 that
residual is the only error left.

What renormalizing does:

- **It removes the overall underestimate.** It halves the market-share error
  at 1,000 respondents: 0.0069 → 0.0035 with a twin, 0.0038 → 0.0019 with a
  near-twin, 0.0031 → 0.0011 with no similar products. It also halves the
  largest share error in the stratified sets: 0.019 → 0.009.
- **It leaves a pattern across shares.** In twin and near-twin scenarios the
  pair members come out too high (+0.0011) and the other products too low
  (−0.0007), both at p < 1e-9. Shares of 0.3–0.5 are high by 0.0008 and shares
  above 0.5 low by 0.0008 (p < 1e-4). All 300 stratified scenarios have max
  correlation below 0.9, so the residual is not confined to high correlation.
- **Max correlation ≥ 0.9 (sweep):** sd 0.0065, max 0.034. The per-bin
  biases are not significant with 24–72 shares per bin; the spread is the
  problem.
- **Max correlation < 0.9 (sweep):** sd 0.0013, max 0.0041. No bias is
  detectable with 65 shares.
