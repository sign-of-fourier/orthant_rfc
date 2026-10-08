# Similarity kernel study: spec

Question: does a similarity kernel on product error fix the near-substitute
problem (near-copies over-counted in combined share), compared with no fix
and with RFC's attribute error alone?

The truth is never the kernel. Each truth's own probabilities are reported as
an oracle row (ceiling); the kernel is scored by the share of the
none-to-oracle gap it closes.

## Design (after Huber, Orme & Miller 1999)

- 6 attributes, effects-coded, 11 coefficients: brand 3, screen 3, sound 3,
  pip 2, blockout 2, price 4. Importances: price .35, brand .28, screen .16,
  sound .10, pip .07, blockout .04.
- beta_n ~ N(mu, diag(Omega_sd^2)), N = 400 (also 150, 800).
- Distance d_ij = sum_a |pw_a(l_i) - pw_a(l_j)| under mu (utility units).
  Twins at k = 0..3 (changed on the k least important attributes) have
  d = 0, 0.3, 0.8, 1.3. Dissimilar alternatives are at d >= 1.3, mostly >= 2.
- 20 holdouts × 5 alternatives:
  - for each k: 2 balanced, 1 twin pair above the rest (+0.6), 1 below (-0.6);
  - 2 triplets (3 near-copies at k = 1);
  - 2 repeats (test-retest).
  Twin |Δu| <= 0.3, alternating sign.
- Fit and evaluation are split **by respondent** (half and half), so every
  holdout type appears in both.

## Truths (total noise matched: expected holdout hit rate about 0.56–0.57)

| truth   | product error |
|---------|---------------|
| merge   | **soft perceptual merging**: P(i, j merged) = 1/(1+exp((d-d0)/s)), d0 = 0.6, s = 0.2; merged alternatives share one e_P draw |
| laplace | correlation τ'·exp(-d'/ℓ') on perceptual weights ≠ \|mu\| (τ' = 0.8, ℓ' = 1) |
| rfc     | iid normal (a good fix should give τ ≈ 0) |
| gumbel  | iid Gumbel, variance matched (a large τ here = false positive) |

All truths: e_A ~ N(0, 0.3² I), σ = 0.6. Merge is the headline.

## Fixes (exact probit, ML on the fit half)

- none: σ only
- rfc: s_A, σ (Σ_A = s_A² I)
- kernel: s_A, σ, τ, ℓ
- later: Monte Carlo RFC by grid search (1999 incumbent); ITEM as an
  alternative fix.

## Metrics (evaluation half)

- Primary: combined twin share / actual, by k and by kind (bal/up/down/triplet).
- Gap closed = (none - fix) / (none - oracle), on twin-share error.
- MAE / test-retest MAE; out-of-sample log-likelihood; fitted τ, ℓ by truth.

## Accuracy gate

At d → 0 and τ → 1 the difference covariances approach correlation 1, where
`orthant_cdf` is weakest and merges at ρ >= 0.97. Validate against SciPy on
the actual problems before trusting small-d results.

## Stages

1. Oracle beta_n, defaults, 1 replication: smoke check. **(now)**
2. 200 replications at the defaults; sweep d0 and s (merge softness).
3. Stage 2 (HB-like shrinkage beta_hat = mu + λ(beta - mu) + η), N sweep.
