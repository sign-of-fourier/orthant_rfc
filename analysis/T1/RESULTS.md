# T1 results: G0, G1, G2 × RFC-S vs MVP (2026-10-07)

## Verdict: PASS (MVP better), with caveats

Decided 2026-10-08 without numeric kill-rule blanks: the user chose "PASS with caveats".
Pooled over G1 and G2 (R = 40), MVP has 0 decision flips of 120 against RFC-S's 29, 0% vs
3.4% mean cost regret, and median |S1 error| 0.03 vs 0.24–0.33. The KILL condition ("RFC
within tolerance on S1 and on decisions") fails for any S1 tolerance below ~0.2 and any flip
allowance below 29 of 120. G0, the guard, is clean: MVP does not invent similarity.

Caveats:
1. **Blanks unset.** No pre-registered tolerances; the verdict rests on the gap being large,
   not on a threshold.
2. **Truths share MVP's form.** G1 and G2 are probit error-component truths, so MVP is
   correctly specified there. G0 (logit truth, MVP misspecified) is the only check against
   that, and it only tests "no similarity".
3. **RFC-S is our reading of Sawtooth**, not verified against Sawtooth software, and HB is
   bayesm's `rhierMnlRwMixture`, not CBC/HB. RFC-S hit a tuning bound in ~half the
   replicates.
4. **RFC-S's S1 cannot exceed 2/3** by its functional form. RFC-G removes the cap (one σ
   per attribute, a deviation from Sawtooth practice) and does not close the gap: its S1 is
   no better and its decisions are no better (see RFC-G). The gap comes from tuning to
   holdout share MAE, which does not identify the error structure, not only from the cap.
5. **Placeholder economics.** Unit cost $1.80 and hurdle h = 0.5 are placeholders; the
   flip counts depend on them, the S1 results do not.
6. **Scale.** Resolved by the refit with the total error variance fixed (see "Scale fix").
   The tables above use the original σ_ν-pinned fits; S1 and decisions are the same under
   both. S4 coverage after the fix: G1 0.90/0.95/1.00, G2 0.75/0.95/0.95 (brand, flavor,
   pack); G2 brand (true σ 0.13, near 0) is the one shortfall.
7. **MVP's likelihood is GHK, not the engine** (DESIGN.md deviation). The engine still
   computes MVP's shares and decisions.
8. **One design point.** N = 600, T = 12, 3 products + none, 20 replicates per truth.

Run: `python analysis/T1/run.py --reps 20` (G1, R = 20, N = 600, T = 12 + 6 holdout).
MVP: exact 36-dim panel likelihood by GHK, M = 4096, one T4 per replicate (all 20 in
parallel, 8 min wall; fit median 73 s). HB-MNL (bayesm, 20,000 iterations, ~94 s) and
RFC-S tuning on CPU, ~50 min in total. Raw output: `out/full_G1/` (`results.json`,
`log.txt`, `manifest.json`). Summary: `python analysis/T1/summarize.py out/full_G1/results.json`.

The selftest (N = 100) passed all fail-fast checks. Every MVP fit beats the truth's
log-likelihood (20/20).

## σ recovery (true S1 = 0.900)

|                     | MVP (GHK) | RFC-S |
|---------------------|-----------|-------|
| S1 median           | 0.869     | 0.657 |
| median \|S1 error\| | 0.033     | 0.243 |
| mean S1 error (sd)  | −0.024 (0.039) | −0.418 (0.267) |
| S3 median (fitted / true) | 0.965 | 0.731 |

Paired \|S1 error\| (RFC-S − MVP): mean 0.380, 95% CI ±0.118, so the gap is clear of
replicate noise (the CI [0.26, 0.50] excludes 0, so not INCONCLUSIVE).

MVP S2 (mean): 0.105 / 0.425 / 0.452 / 0.018 vs true 0.08 / 0.45 / 0.45 / 0.02.

S4 (MVP, 95% CI on log σ_a): coverage 0.90 / 0.90 / 0.95 (brand, flavor, pack).
Raw σ_a mean 0.67 / 1.34 / 1.36 vs 0.36 / 0.86 / 0.86. Mean z over replicates is ≈ +1.1
for every ω and σ (≈ 0 for b), so the overall scale is biased upward by ~1 SE in finite
samples. That is the scale direction, pinned only by σ_ν = 0.18 (see PLAN.md); it does
not reach S1, S2, shares or decisions.

## Decisions (hurdle h = 0.5 placeholder; costs placeholder)

|                       | MVP | RFC-S |
|-----------------------|-----|-------|
| flips: price / ext / cost (of 20 each) | 0 / 0 / 0 | 1 / 6 / 8 |
| mean profit regret: price / cost | 0.00% / 0.00% | 0.26% / 3.69% |
| D-ext incrementality error: mean / MAE | −0.004 / 0.020 | +0.137 / 0.187 |
| holdout share MAE     | 0.0132 | 0.0214 |

RFC-S overstates the line extension's incrementality (it under-models the A/A″
similarity), which is what flips D-ext and D-cost.

## Caveats

- RFC-S had a tuning parameter at a bound in 10/20 replicates (selftest fail-fast
  rule; in the full run it is recorded, not fatal). In several, σ_l → 0.01 or g → 0.05.
- h, costs and the kill-rule tolerances are placeholders; the verdict below is
  conditional on the user filling them.
- RFC-G is not run yet. G0 and G2 below.

## Reading against the kill rule

With any S1 tolerance under ~0.2 and any flip allowance under 15 of 60, the KILL
condition fails in MVP's favour: **PASS (MVP better) on G1**, pending G0 as the guard.
D1 gate: MVP S1 error −0.024 median-abs 0.033; S4 coverage 0.90–0.95.

# G0 (null, logit truth) and G2 (pair corr .97)

Run: `run.py --reps 20 --truth G0` and `--truth G2` side by side (2026-10-07, ~70 min wall
for both; MVP 6 / 14.5 min on GPU). G0: σ_a = 0, iid Gumbel errors, truth shares by mixed
logit (2^16 Sobol taste draws); the probit is misspecified by design and σ_ν is fixed at
√(π²/6). G2: variance shares .01/.485/.485/.02 (σ_ν share kept from G1).

| | G0 MVP | G0 RFC-S | G2 MVP | G2 RFC-S |
|---|---|---|---|---|
| true S1 | 0 | 0 | 0.970 | 0.970 |
| S1 median | 0.025 | 0.000 | 0.961 | 0.640 |
| median \|S1 error\| | 0.025 | 0.000 | 0.025 | 0.330 |
| S1 max over reps | 0.203 | 0.164 | — | — |
| flips price/ext/cost (of 20) | 0/0/0 | 0/0/0 | 0/0/0 | 1/6/7 |
| mean regret price/cost | 0/0% | 0/0% | 0/0% | 0.26/3.12% |
| ext incrementality MAE | 0.013 | 0.070 | 0.020 | 0.183 |
| holdout share MAE | 0.0137 | 0.0154 | 0.0135 | 0.0218 |

- **G0 guard:** MVP does not invent similarity (median S1 0.025; 6/20 exactly 0; worst
  0.20). RFC-S is slightly closer to 0 on S1, but MVP still makes no decision errors and
  has the lower incrementality error. Paired |S1 err| RFC − MVP: −0.016 ± 0.033 (a tie).
  S4 is undefined under G0 (σ_a = 0 sits on the boundary of log σ).
- **G2:** same picture as G1, stronger. Paired |S1 err| RFC − MVP 0.448 ± 0.109.
  MVP σ median 0.125/0.927/0.893 vs 0.128/0.893/0.893; S2 mean .023/.476/.483/.018.
- **RFC-S caps S1 at 2/3.** pair_corr_rfc = 2σ_l²/(3σ_l² + π²g²/6) → 2/3 as g → 0; 9 of
  20 G2 reps sit at 0.666. RFC-S cannot represent a near-duplicate correlation above 0.67
  however it is tuned, so its G1/G2 gap is structural, not a tuning failure.
- **Scale runaway, G2 rep 15.** MVP scaled the whole model ~×18 (b_price −17.1, σ_f, σ_p ≈
  16.6, σ_brand → 0, SE 0): with σ_ν fixed at 0.18 and the other σ free, the likelihood is
  nearly flat along the scale and the fit drifted out (nll still beats the truth by 11.6).
  Scale-free outputs were fine (S1 1.00, 0 flips, holdout MAE 0.015), but this replicate
  drives the G2 σ mean (1.78) and S4 (brand 0.60). A firmer normalisation (fix the price
  coefficient's mean or the total error variance instead of σ_ν) would remove it; that is
  a design choice, not made.

## Pooled over G1 and G2 (the kill rule's population)

MVP flips 0 of 120; RFC-S 29 of 120 (2 price, 12 ext, 15 cost). Mean cost regret 0.00% vs
3.4%. Median |S1 error| 0.03 vs 0.24–0.33. Verdict stays **PASS (MVP better)** under any
plausible blanks, with G0 clean as the guard.

# Scale fix: MVP refit with the total error variance fixed (2026-10-08)

`mvp_refit.py`: σ_b² + σ_f² + σ_p² + σ_ν² = π²/6 (true in every truth), estimating the split
u_a = log(σ_a²/σ_ν²) (each share floored at 10⁻⁶; a line search otherwise reached a singular
covariance). Same data, draws and M; 60 fits on T4s in 16.5 min wall. SEs of log σ_a by the
delta method from the sandwich covariance. Output: `out/full_<truth>/results_mvp_total.json`.

| | σ mean (b/f/p) | σ max | S4 coverage | S1 median | flips |
|---|---|---|---|---|---|
| G1 truth | 0.363 / 0.860 / 0.860 | | | 0.900 | |
| G1 σ_ν-pinned | 0.674 / 1.335 / 1.362 | 5.15 / 8.87 / 8.81 | 0.90 / 0.90 / 0.95 | 0.869 | 0 |
| G1 total fixed | 0.403 / 0.834 / 0.861 | 0.52 / 0.91 / 0.93 | 0.90 / 0.95 / 1.00 | 0.869 | 0 |
| G2 truth | 0.128 / 0.893 / 0.893 | | | 0.970 | |
| G2 σ_ν-pinned | 0.145 / 1.777 / 1.786 | 0.39 / 16.6 / 16.7 | 0.60 / 0.90 / 0.95 | 0.961 | 0 |
| G2 total fixed | 0.126 / 0.887 / 0.892 | 0.41 / 1.00 / 0.95 | 0.75 / 0.95 / 0.95 | 0.973 | 0 |

- No runaways: the price coefficient stays in [−1.03, −0.89] (truth −0.975) across all 60
  fits. G1 had a runaway too (σ to ~9), hidden in the σ_ν-pinned means.
- Mean z of b and log ω is now within ±0.7 (was ≈ +1.1 for every ω and σ).
- G0: S1 median 0.017 (max 0.20), 0 flips; z values are off by design (misspecified).
- **Recommendation:** use the total-variance normalisation for MVP from now on.

# RFC-G (2026-10-08)

`rfc_g.py`: RFC with σ_brand, σ_flavor, σ_pack and the Gumbel g, "none" error variance
Σσ_a²; tuned to holdout share MAE by Nelder-Mead from the RFC-S optimum and from the best of
a 3⁴ grid. HB rerun with the same seeds (each replicate checked by reproducing RFC-S's
stored holdout MAE exactly); tuning on Modal CPU, one container per replicate (~4 min each).

| | S1 median | median \|S1 err\| | flips p/e/c (of 20) | regret price/cost | ext MAE | holdout MAE |
|---|---|---|---|---|---|---|
| **G0** (S1 0) MVP | 0.017 | 0.017 | 0/0/0 | 0/0% | 0.013 | 0.0137 |
| RFC-S | 0.000 | 0.000 | 0/0/0 | 0/0% | 0.070 | 0.0154 |
| RFC-G | 0.000 | 0.000 | 0/0/0 | 0/0% | 0.051 | 0.0149 |
| **G1** (S1 0.90) MVP | 0.869 | 0.033 | 0/0/0 | 0/0% | 0.019 | 0.0132 |
| RFC-S | 0.657 | 0.243 | 1/6/8 | 0.26/3.69% | 0.187 | 0.0214 |
| RFC-G | 0.621 | 0.279 | 2/10/13 | 0.53/5.37% | 0.226 | 0.0197 |
| **G2** (S1 0.97) MVP | 0.973 | 0.014 | 0/0/0 | 0/0% | 0.020 | 0.0134 |
| RFC-S | 0.640 | 0.330 | 1/6/7 | 0.26/3.12% | 0.183 | 0.0218 |
| RFC-G | 0.706 | 0.264 | 1/5/9 | 0.26/3.67% | 0.192 | 0.0199 |

(MVP rows: total-variance refit.) Paired |S1 err| RFC-G − MVP: G1 0.350 ± 0.139, G2
0.316 ± 0.118, G0 0.005 ± 0.045.

- **Freedom doesn't help.** RFC-G fits the holdouts a little better than RFC-S (MAE 0.020
  vs 0.021) but its S1 is no closer to the truth and its decisions are no better (G1: 25
  flips vs 15). Its tuned σ split is unstable (G1 S2 mean 0.17/0.21/0.31/0.31 vs truth
  .08/.45/.45/.02; some replicates put all variance on brand, or none on flavor/pack).
- **Why:** 6 holdout tasks (2 with a near-duplicate pair) give share MAE too little
  information about the error structure. Matching holdout shares and getting
  cannibalisation right are different targets; MVP gets the structure from the 12
  training choices per respondent through the likelihood.
- RFC-G hit a bound in 30 of 60 replicates (RFC-S: ~30 of 60).

## Pooled verdict with RFC-G

Over G1 and G2 (R = 40), flips of 120: MVP 0, RFC-S 29, RFC-G 40. The better RFC variant
(RFC-S on decisions, either on S1) is still far from MVP. **PASS (MVP better) stands**;
caveat 4 is answered (the gap is not just RFC-S's 2/3 cap).
