# ORTHANT_PLAN: the engine as a probit evaluator (started 2026-10-09)

Kept apart from PLAN.md on purpose: PLAN.md (T1-T4, "is probit better") uses QMC (GHK with
scrambled Sobol) only and does not depend on anything here.

## Question

Where is the orthant engine (local `multivariate_probit.orthant`, deterministic approximation,
fast, no gradients) accurate enough to replace QMC, and where not?

- **Fitting** (likelihood of observed choices, log scale).
- **Scoring** (shares / revenue of candidate scenarios once a model is fitted, probability scale).

## Findings so far

Work in `analysis/engine_certify/` (DESIGN.md, ec.py, per_row.py, out/). Shrunk T4 shelf: J = 8
and 12 SKUs + none, brand / flavor / pack shared errors, no random tastes, N = 300, T = 8.
Engine with `dup_corr=0` (no 0.97 merge). Reference: GHK, R = 1,024 for fits, 8,192 per row.

**Fitting: the engine is not accurate enough at any correlation tested.** Distance of the engine
estimate from the QMC estimate, sqrt(2 x QMC log-likelihood lost), in SE units (pass <= 0.25):

| truth (shares brand/flavor/pack/product) | J = 8 | J = 12 |
|---|---|---|
| weak (.05/.10/.10/.75) | 0.55 | 2.55 |
| medium (.10/.25/.25/.40) | 2.18 | running |
| shelf (.10/.35/.35/.20) | 2.72 | running |
| near-dup (.05/.45/.45/.05) | 5.45 | running |

Direction is consistent: the engine understates similarity (product share too large: .67 vs QMC
.44 medium, .57 vs .34 shelf, .17 vs .02 near-dup at J = 8).

**Cause: a systematic bias in the bulk, not the tails** (per_row.py, J = 8, at the truth). The
engine underestimates common choice probabilities (P = 0.1-1, 80% of choices) by 0.1% (weak), 1%
(medium), 2% (shelf), 4% (near-dup); same sign everywhere, so it sums into a pull on the share
parameters. Relative error is largest at P < 0.01 (12-21% at shelf / near-dup) but those are ~1%
of choices and contribute little to the gradient error.

**Ruled out for fitting:**
- A correlation threshold ("engine until rho_max > rho*"): no safe rho* exists (fails at weak).
- Routing rows by probability size: the error is in the common rows.
- Gauss-Hermite for the chosen alternative's own error inside the factor method (T4 side note):
  worse at equal compute.
- The 0.97 merge makes no difference here (near-dup D_LR 5.45 on and off); with dup_corr = 0
  nothing broke.

**Open caveat:** at weak / J = 12 the QMC reference's own shares (.18/.20/.24/.37) are far from the
truth (.05/.10/.10/.75); weak identification at N = 300 or reference error. Check before leaning on
that cell.

## Options

1. **Warm start for fitting (speed only).** Engine fit, then QMC from there. Needs no accuracy rule
   (QMC has the last word). Measure: QMC iterations cold vs warm (cold was 50-80 in stage A).
2. **Screen-then-verify for scoring.** In a product-line / price search the engine scores every
   candidate; QMC rescores the top k%; the pick is QMC's best of the shortlist. Accuracy question:
   does the true best survive the screen (regret vs keep fraction), as with q-EI's PI screen.
   Reasons it may work where fitting failed: probability scale, not log; a bias common to all
   candidates cancels in a ranking. Reason it may not: the bias grows with correlation, so it is
   not common across candidates that differ in near-duplicates (line extensions, delistings),
   which are the decisions probit is for. T2: engine off by up to 2 points at rho ~ 0.8.
3. **Fix the bias in the engine.** Out of scope until 1-2 say whether it is worth it.

## Test S1: screen-then-verify for scoring (design, 2026-10-09)

The engine as a screen over candidate scenarios of a fitted model, QMC/exact only on the shortlist.

- **Model:** T4's J = 20 shelf and DGP, taste heterogeneity included, parameters = the truth
  (scoring does not need a fit). Error-share truths weak / medium / shelf (T4 S1) / near-dup.
- **Candidates (firm margin objective):**
  - A. Assortment x extension: keep or drop each of the firm's 6 SKUs (64) x no clone or a
    near-duplicate clone of one of them (7) = 448. This is where the correlation structure differs
    between candidates.
  - B. Prices: 400 random price vectors for the firm's 6 SKUs (5 levels each), full assortment.
- **Engine score:** population shares of the firm's SKUs = mean over Q = 256 scrambled-Sobol taste
  draws of the engine's choice probability (one ~20-dim orthant per firm SKU per draw),
  `dup_corr=0`.
- **Reference:** T4's frequency simulator, 10^6 draws, common random numbers across candidates;
  run twice with different seeds, and regret below the reference's own seed-to-seed noise counts
  as 0.
- **Metrics per truth x family:** regret % of the engine-alone pick; regret % of screen-then-verify
  (engine top k%, then the best of those by the reference) at k = 1, 5, 10%; rank of the true best
  in the engine's order; Spearman. Material = regret > 0.5% (T4's rule).
- **Pass:** zero material regret at k = 5% in every truth and family.
- **Cost:** local CPU, no fits; reference ~4 min per truth; engine time unknown at 20 dims (timed
  first; Q drops to 128 if needed). Under 1 GB.

### S1 results (2026-10-09): PASS

`analysis/engine_certify/screen.py`, out/screen/. Regret % vs the reference best (material > 0.5%);
reference seed-to-seed noise 0.16-0.36%.

| truth | family | engine pick regret | screen 1 / 5 / 10% | Spearman | engine margin vs ref |
|---|---|---|---|---|---|
| weak | assortment | 0 | 0 / 0 / 0 | 1.000 | -1.2% |
| weak | prices | 0 | 0 / 0 / 0 | 0.999 | -1.3% |
| medium | assortment | 0 | 0 / 0 / 0 | 1.000 | -1.2% |
| medium | prices | 0 | 0 / 0 / 0 | 0.999 | -1.3% |
| shelf (T4 S1) | assortment | 0 | 0 / 0 / 0 | 1.000 | -1.3% |
| shelf | prices | 0 | 0 / 0 / 0 | 1.000 | -1.3% |
| near-dup (.05/.45/.45/.05) | assortment | 0 | 0 / 0 / 0 | 0.999 | -1.7% |
| near-dup | prices | 0 | 0 / 0 / 0 | 0.999 | -1.6% |
| G1 (.08/.45/.45/.02; clone rho .98) | assortment | 0.50 | 0 / 0 / 0 | 0.999 | -1.7% |
| G1 | prices | 0 | 0 / 0 / 0 | 0.999 | -1.4% |

- The engine's margin error is a near-uniform -1.2 to -1.7% that cancels in the ranking (part is
  likely the 256-draw taste integration: it is -1.2% even at weak).
- The only miss: G1 assortment, the engine alone picks the second best (same assortment, clone of
  SKU 3 instead of SKU 0; 0.50% regret, at the reference noise level). Keeping 1% and verifying
  fixes it. This is the rho >= 0.97 region, with the merge off.
- Easy vs hard: assortment bests lead by 2.8-4.7%; prices have 4-5 candidates within 2%.
- Engine 0.13 s per candidate (848 in ~110 s, local CPU) vs the reference ~0.25 s per seed at 10^6
  draws; the speed comparison that matters (vs QMC scoring at matched accuracy) is not done.

### S2: GPU scoring speed, engine vs GHK-QMC vs QMC simulator (2026-10-09)

`boaz/modal/prototypes/score_bench.py` (the GPU engine lives in boaz; this repo does not import it);
results `analysis/engine_certify/out/gpu_bench/`. Same 848 candidates and 10^6-draw reference as
S1, shelf and G1 truths, one Tesla T4 each, times for all 848 after warm-up. The GPU engine code
reproduces the .so's margins to 2e-16 (same computation, merge off).

| scorer | cheapest zero-regret setting (both truths, pick alone) | T4 seconds / 848 | vs engine |
|---|---|---|---|
| GPU engine (float64, order 1) | Q = 64 (Q = 128/256 pick 0.50% at G1: bias, not draws) | 5.9-6.4 | 1x |
| GPU GHK-QMC (float32) | Q = 128, R = 64 | 1.3-1.4 | ~4.5x faster |
| GPU QMC simulator (float32) | N = 2^15 shoppers | 0.22-0.24 | ~25x faster |

- Screen-then-verify at 1% keep had zero regret for every setting of every scorer.
- The engine is the slowest scorer here: each candidate needs ~1,500 orthants (6 owned products x
  Q taste draws), while the simulator gets all shares from one set of shoppers. More taste draws do
  not remove its G1 miss (0.50% at Q = 128 and 256).
- Not tried: the engine at resolution "low" (order 0, faster), engine in float32.
- Cost is negligible for all three at this size (T4 time in seconds).

## Status

- Stage A stopped after 6 of 24 datasets (all fail; enough for the fitting conclusion); stage B
  (rho* hybrid) not run.
- S1 done (PASS). S2 done: for scenario scoring the QMC simulator is ~25x faster than the GPU
  engine at matched (zero) regret. Next: option 1 (warm start); speed of engine screening vs QMC scoring at
  matched accuracy; a harder candidate set (many near-ties) if S1 is to be leaned on.
- 2026-10-09: paused. The next steps above (warm start, harder screening set, low resolution /
  float32, bulk-bias fix) moved to [BACKLOG.md](BACKLOG.md#engine-follow-ups).
