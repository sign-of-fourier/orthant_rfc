# T31 · B5, MNP fit at large J: can anyone else fit it, can we, does it change decisions

Scoped with the user 2026-10-23 (bottleneck list, B5). Criteria confirmed by the user 2026-10-23.

- **Claim.** A probit with random tastes can be fitted on a realistic CBC shelf (J = 20–60) fast and accurately
  enough to use; nobody else can at this size; and it changes decisions vs HB-MNL + RFC.
- **World.** T29's: 60 dunnhumby soft-drink SKUs + none, N = 1,000, T = 12, prices base × {0.8 … 1.2}; truths est
  (Σ = I) and stress (mfr / sub / cell / product variance shares .25 / .02 / .23 / .50).
- **Fit (ours).** Per-task composite likelihood: tastes folded into each task's covariance, one d = J − 1 orthant
  per choice, GHK at 1,024 Sobol points on an H100, sandwich SEs. Replaces T29's 720-dim panel orthant, which is
  biased at M = 1,024 (taste sds 17–23% too wide) and only slightly helped by a first-order bias correction.
  Per-task GHK-1k error at the true θ: 0.002 nats per choice (both truths). Fits the mean and the total covariance,
  not the taste / error split; population shares need only the total.

## Gate 0 (dropped 2026-10-23, user)

Was: can an existing open-source MNP fitter already do this. A pybhatlib (MACML) fit ran on an 8-core CPU and was
stopped after about 11 iterations (80–120 s each; its GPU path fails on this spec). Dropped as off the B5 objective:
the claim is bottleneck + GPU fix vs the incumbent (HB-MNL + RFC), not what other tools might do.

## Part A: feasibility (ours)

- J ∈ {20, 40, 60} (nested SKU subsets) × est, stress; rep 0 at all J, 5 reps at J = 60. Panel fit alongside at
  J = 60 as the known-biased reference.
- Measures: fit time; joint Wald vs truth; holdout share error (pts).
- **Pass (confirmed 2026-10-23).** Holdout share error no worse than HB-MNL's. (The parameter condition is dropped:
  per choice, each SKU's mean and taste sd are not separately identified, so parameters drift along flat
  directions that leave shares unchanged.)

## Part B: value

- T30's line + price decision (m = 6, 8), greedy + swap. Arms: per-task MNP (simulation scoring), HB-MNL + SoP,
  HB-MNL + RFC-S. Regret vs truth, $ / 1,000 trips, CI + Holm over 5 reps per truth.
- **Pass (confirmed 2026-10-23).** MNP below the best incumbent arm by ≥ 1 $ / 1,000 trips, CI above 0, in at
  least one truth; guard: no loss under est. Truths est and stress.

## Results

### Assessment

Incumbent: Sawtooth CBC/HB on a CPU (no GPU). Us: a GPU, any method. Question: with the GPU, is there a method
that gives better answers? **Against HB-MNL simulated over its posterior draws: no, except profit forecasts under
correlated substitution** (see "HB over posterior draws" below; this overturns the earlier "yes, modestly", which
was against posterior means plugged into SoP). Earlier wording, kept for the record: **Yes, modestly.** Sawtooth users are limited to logit by the software (no probit is
offered), not by J or m: HB-MNL stays practical at every size tested, and neither line size nor shelf size widens
the gap here. What the GPU adds is a better model at a usable cost.

- **Method.** Per-task probit fit, 8.6–15 min on an H100 at J = 60 (about $0.60–1.00 per fit); 1.1–1.8 min at
  J = 20, 6.5–7.1 at 40.
- **Better answers.** Shares: HB-MNL misplaces 1.6–2.2× MNP's share on the holdout (3–4 pts more) in all 14
  fits, about half of it "none" (HB over-predicts it by 3–7 pts). Decisions (m = 6, 10 reps): 1–2% more profit
  than HB-MNL + SoP in both truths, 2–4% more than + RFC-S; at m = 8 only vs RFC-S under est. The calibrated
  compartmentalized workflow is no better than full-shelf HB-MNL (Part C).
- **Caveat.** Every truth is our probit generator; a mixed-logit truth (HB-MNL's own model) has not been run.
- **What the GPU adds.** A 15-minute fit instead of about 7.3 h on an 8-core CPU (projected, not run to the
  end), which matters once a study needs many runs or more parameters.

### Bottleneck: the same fit on CPU vs GPU (est, J = 60, rep 0)

Per-task composite GHK-1k, value + autograd gradient over 12,000 choices: 0.84 s on an H100 vs 50 s on an 8-core
CPU (Modal), about 60×. The full GPU fit took 7.5 min (534 evaluations, stopped at the 500-iteration cap); the same
fit on CPU projects to 7.3 h.

### Part A, rep 0, est

Holdout share error vs the truth (6 fixed holdout price vectors × 61 alternatives, pts):

| fit | time | MAE | max | none bias |
|---|---|---|---|---|
| per-task GHK-1k (H100) | 7.5 min | 0.183 | 1.22 | −0.48 |
| panel GHK-1k, 720-d (T29) | 10.6 min | 0.203 | 0.99 | −0.16 |
| HB-MNL, SoP (T4) | 1.4 min | 0.267 | 3.69 | +3.27 |
| observed holdout shares, N = 1,000 | — | 0.275 | 2.27 | −0.85 |

Per-task fit: price −1.157 (truth −1.2), price sd 0.33 (0.3), taste-sd median 1.62 (1.55; the panel fit gave
1.81). Each SKU's mean and taste sd are weakly separated per choice (near-singular information), so parameters
drift along flat directions; shares are unaffected. Judge Part A on shares, not parameters.

### Part A (2026-10-23): PASS in all 14 fits

Per-task GHK-1k fits on H100s with the convergence rule (step change < 1e-8 in mean nll, cap 1,000 iterations):
J = 20 1.1–1.8 min, J = 40 6.5–7.1 min, J = 60 8.6–15.0 min (the two rep-0 fits hit the cap). HB-MNL 1.3–1.8 min
on a T4. Holdout share MAE vs truth, pts (`t31.py partA`, `out/partA.json`):

| truth | J | MNP | HB-MNL | observed holdout | HB-MNL none bias |
|---|---|---|---|---|---|
| est | 20 | 0.288 | 0.624 | 0.508 | +4.89 |
| est | 40 | 0.216 | 0.431 | 0.354 | +5.30 |
| est | 60 (5 reps) | 0.153–0.181 | 0.243–0.298 | 0.271–0.296 | +3.27 to +5.18 |
| stress | 20 | 0.335 | 0.708 | 0.485 | +6.51 |
| stress | 40 | 0.167 | 0.370 | 0.325 | +5.08 |
| stress | 60 (5 reps) | 0.155–0.185 | 0.262–0.282 | 0.272–0.288 | +3.52 to +5.66 |

MNP's error is below HB-MNL's in every fit (about 35–55% lower) and below the sampling noise of a 1,000-person
holdout. HB-MNL's error is mostly one alternative: it over-predicts "none" by 3–7 pts in every fit (MNP's none
bias is −0.7 to +1.0).

MAE per alternative falls with J by construction (mean share ~ 1 / (J + 1)), so it does not compare across J.
Scale-free (2026-10-23), total share misplaced = ½ Σ |error| over all alternatives, pts, mean over holdout tasks:

| truth | J | MNP | HB-MNL | HB − MNP | HB / MNP |
|---|---|---|---|---|---|
| est | 20 | 3.0 | 6.5 | 3.5 | 2.2× |
| est | 40 | 4.4 | 8.8 | 4.4 | 2.0× |
| est | 60 (5 reps) | 5.2 | 8.4 | 3.2 | 1.6× |
| stress | 20 | 3.5 | 7.4 | 3.9 | 2.1× |
| stress | 40 | 3.4 | 7.6 | 4.2 | 2.2× |
| stress | 60 (5 reps) | 5.3 | 8.4 | 3.0 | 1.6× |

Both errors grow with J; the gap stays at 3–4 pts. About half of HB-MNL's excess is "none": over the SKUs alone
HB-MNL misplaces 1.3–1.4× MNP's share. Caveat: every truth here is our probit generator; a mixed-logit truth
(HB-MNL's own model) has not been run.

### Part B, 5 reps (2026-10-23): no cell significant; guard holds

T30's line + price decision, greedy + swap per arm, 5 reps, regret vs the truth optimum (T30's), $ / 1,000 trips
(`t31.py partB`, `out/partB.json`). Difference = best incumbent arm − MNP, paired by rep, t CI, Holm over 4 cells.

| truth | m | SoP | RFC-S | MNP | best − MNP [95% CI] | % of optimum | Holm p |
|---|---|---|---|---|---|---|---|
| est | 6 | 4.05 | 8.68 | 0.63 | +3.42 [−1.19, +8.04] | +2.1% | 0.33 |
| est | 8 | 3.43 | 8.54 | 1.63 | +1.80 [−2.74, +6.34] | +1.0% | 0.67 |
| stress | 6 | 2.34 | 3.82 | 1.02 | +1.32 [−0.14, +2.78] | +0.8% | 0.26 |
| stress | 8 | 1.83 | 4.54 | 2.02 | −0.19 [−4.46, +4.07] | −0.1% | 0.91 |

MNP has the lowest mean regret in 3 of 4 cells and passes the materiality bar (≥ 1 $) on points in 3, but no CI
excludes 0 at 5 reps: SoP's regret swings 0–8 $ across reps. Guard (no loss under est): holds. RFC-S is the
worst arm in every cell.

### Part B, 10 reps (2026-10-23): PASS (est m = 6, stress m = 6); guard holds

The user chose to add 5 reps (5–9) after the 5-rep result, so this is a second look at the data. Same design and
columns; difference = SoP (the best incumbent arm in every cell) − MNP.

| truth | m | SoP | RFC-S | MNP | best − MNP [95% CI] | % of optimum | reps 0–4 / 5–9 | MNP better / worse | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| est | 6 | 4.03 | 7.45 | 0.58 | +3.44 [+1.33, +5.55] | +2.1% | +3.42 / +3.46 | 9 / 1 | 0.020 |
| est | 8 | 3.79 | 7.57 | 1.61 | +2.18 [−0.37, +4.74] | +1.2% | +1.80 / +2.57 | 7 / 3 | 0.17 |
| stress | 6 | 3.10 | 4.35 | 1.36 | +1.74 [+0.61, +2.88] | +1.0% | +1.32 / +2.16 | 8 / 1 | 0.021 |
| stress | 8 | 1.72 | 3.76 | 1.72 | +0.01 [−1.72, +1.73] | +0.0% | −0.19 / +0.20 | 5 / 4 | 0.99 |

At m = 6 both truths pass: the difference is at least 1 $ and the CI is above 0. Doubling the Holm p to allow
for the two looks gives 0.040 and 0.042. The new reps alone repeat the first five's point estimates. At m = 8
neither passes; under stress it is a tie. Guard (no loss under est): holds. MNP fits for reps 5–9 ran on a second
Modal workspace, because the first had hit its spend limit.

**Against RFC-S** (Sawtooth's recommended simulator; our implementation: one attribute-error sd over the truth's
manufacturer / sub-commodity / cell levels, tuned to the holdout). Same 10 reps, RFC-S − MNP, Holm over 4 cells:

| truth | m | RFC-S | MNP | RFC-S − MNP [95% CI] | % of optimum | MNP better / worse | Holm p |
|---|---|---|---|---|---|---|---|
| est | 6 | 7.45 | 0.58 | +6.87 [+5.19, +8.55] | +4.2% | 10 / 0 | < 0.001 |
| est | 8 | 7.57 | 1.61 | +5.96 [+4.20, +7.73] | +3.2% | 10 / 0 | < 0.001 |
| stress | 6 | 4.35 | 1.36 | +2.99 [+0.83, +5.15] | +1.8% | 7 / 2 | 0.024 |
| stress | 8 | 3.76 | 1.72 | +2.04 [−0.19, +4.27] | +1.1% | 7 / 2 | 0.068 |

RFC-S is worse than SoP in every cell, which may reflect our implementation rather than Sawtooth's; SoP is the
conservative incumbent, RFC-S the recommended-practice one.

**What m does.** Regret of the incumbent does not grow from m = 6 to 8 (SoP est 4.03 → 3.79, stress 3.10 → 1.72);
MNP's rises (0.58 → 1.61, 1.36 → 1.72), mostly from one-step price misses (it nearly always picks the optimal
SKUs). Greedy + swap is near-optimal for every arm (T30), so m adds no search burden for anyone. m is the decision
setting for this guard, not a scale axis.

### Part C (2026-10-23, provisional): compartmentalized incumbent

The incumbent's realistic workflow: one survey (N = 1,000, 12 tasks) and one HB-MNL per pack-format compartment
(2-liter 23 SKUs, single-serve 8, multipacks 29), the line + price search over the summed compartment profits
(each compartment its own market with its own none), scored on the full shelf under the truth. m = 6, 10 reps;
MNP and full-shelf SoP from Part B. Regret, $ / 1,000 trips (`t31.py hbcomp`, `t31.py partC`, `out/partC.json`):

| truth | SoP-comp | RFC-S-comp | SoP (full shelf) | MNP | best comp − MNP [95% CI] | % of optimum | MNP better / worse | Holm p |
|---|---|---|---|---|---|---|---|---|
| est | 9.95 (max 18.5) | 13.52 | 4.03 | 0.58 (max 1.6) | +9.37 [+5.72, +13.02] | +5.7% | 10 / 0 | 0.0005 |
| stress | 6.63 (max 11.5) | 8.65 | 3.10 | 1.36 (max 4.8) | +5.27 [+2.35, +8.19] | +3.1% | 10 / 0 | 0.003 |

Caveat: the summed-compartment objective weights every compartment as a full market (the single-serve study
counts as much as the 2-liter one) and over-forecasts profit by 54–66%.

**Calibrated compartments** (`partC --weighted`): each compartment's profit scaled so its predicted purchase volume
at the surveyed shelf matches its share of category purchases (real occasions; 85% of trips buy), as a
practitioner calibrating separate studies to sales would.

| truth | SoP-comp-w | RFC-S-comp-w | SoP (full shelf) | MNP | SoP-comp-w − MNP [95% CI] | % of optimum | MNP better / worse | Holm p | SoP-comp-w − SoP [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| est | 6.56 (max 10.6) | 10.85 | 4.03 | 0.58 | +5.97 [+3.68, +8.26] | +3.7% | 10 / 0 | 0.0005 | +2.53 [−1.30, +6.36] |
| stress | 2.71 (max 3.4) | 6.58 | 3.10 | 1.36 | +1.35 [−0.17, +2.86] | +0.8% | 7 / 2 | 0.076 | −0.40 [−1.78, +0.99] |

Calibrated, the compartmentalized incumbent is not distinguishable from full-shelf HB-MNL in either truth; most of
the unweighted gap came from the naive combination, not from compartmentalizing. MNP beats it under est (10 / 0);
under stress the gap is +0.8% and not significant. Forecasts: −5% (est), −9% (stress), vs MNP's +2%.

### Fit time: the stopping rule (2026-10-23)

The per-task GHK fit's last two-thirds is a creep along flat directions (each SKU's mean against its taste sd),
which barely moves shares. est rep 0 from zero: within 8 nats of the final nll at 4.9 min, within 2 at 9.9 min,
final (44,058.3) at 14.9 min. A looser stop (`fit --tol 1e-6`, files suffixed `_tol1e-06`), est rep 0: **3.7 min** (264
evaluations) vs 14.9, nll +10.0, holdout share misplaced vs truth 5.46 pts vs 5.53 (MAE 0.179 vs 0.181). The two
fits' shares differ by 1.38 pts misplaced (max 0.29 on one alternative), inside either fit's error vs the truth.
Recommended rule for new fits: about 4 min, about $0.25 per J = 60 fit on an H100. (One rep checked.)

### Mixed-logit truth (2026-10-23)

Robustness check against home advantage: the data are generated from HB-MNL's own model (est's tastes, iid Gumbel
errors with the probit truths' total variance), so logit is correctly specified and the probit is not. Same
pipeline (`--truths mxl`): J = 60, 5 reps, Part A and Part B (m = 6, 8). No T30 optimum exists for this truth:
the optimum is the best of a truth greedy + swap and every arm's line. "None" takes 9.4% of survey choices (est
13.5%): with the same tastes, Gumbel errors give the best of 60 SKUs a heavier right tail.

Results, 5 reps (`partA`, `partB --truths mxl`). Holdout share misplaced vs truth, pts: MNP 4.6–5.6, HB-MNL
7.1–7.7, observed 1,000-person holdout 8.6–10.0; SKUs only: MNP 4.5–5.4, HB-MNL 5.5–5.8. HB-MNL over-predicts
"none" by +2.9 to +4.4 pts (MNP −0.7 to +0.4), the same bias as under the probit truths, although logit is the
true model here: the bias comes from estimation (HB posterior means + SoP, or our HB sampler), not from
misspecification. Unresolved: whether Sawtooth's HB shows it too.

Part B, regret $ / 1,000 trips, difference = arm − MNP, t CI over 5 reps, Holm over 4:

| m | SoP | RFC-S | MNP | SoP − MNP [95% CI] | RFC-S − MNP [95% CI] | Holm p (SoP / RFC-S) |
|---|---|---|---|---|---|---|
| 6 | 3.50 | 3.16 | 1.01 | +2.49 [−1.44, +6.42] | +2.15 [+0.61, +3.70] | 0.31 / 0.072 |
| 8 | 2.54 | 3.20 | 1.72 | +0.81 [−1.11, +2.74] | +1.48 [−0.39, +3.35] | 0.31 / 0.28 |

MNP does not lose when logit is the true model (guard holds); no difference is significant at 5 reps. Profit
forecast bias: SoP +3%, RFC-S +10–12%, MNP +3–5%.

### HB over posterior draws (2026-10-23): the incumbent's share and decision gaps close

The HB-MNL arms above plug each respondent's posterior mean into the logit formula (SoP). Averaging SoP over 100
posterior draws per respondent instead (`hbdraws`, `drawcheck`; same data and seeds) removes HB-MNL's "none" bias
and its share-error gap. Holdout share misplaced vs truth, pts, reps 0–4:

| truth | HB posterior means | HB over draws | MNP | HB "none" bias, means → draws |
|---|---|---|---|---|
| est | 7.4–8.8 | 4.1–5.6 | 4.7–5.5 | +3.0 to +5.1 → −1.8 to −0.3 |
| mxl | 6.9–8.1 | 4.1–5.4 | 4.6–5.6 | +2.7 to +4.6 → −1.4 to −0.1 |

Part B with the draws arm (SoP-draws), regret $ / 1,000 trips, reps 0–4, SoP-draws − MNP [95% CI], Holm over 6:

| truth | m | SoP | RFC-S | SoP-draws | MNP | SoP-draws − MNP | MNP better / worse | forecast bias, draws / MNP |
|---|---|---|---|---|---|---|---|---|
| est | 6 | 4.05 | 8.68 | 1.21 | 0.63 | +0.58 [−2.26, +3.43] | 1 / 1 | −0.9% / +3.5% |
| est | 8 | 3.43 | 8.54 | 1.69 | 1.63 | +0.05 [−2.21, +2.31] | 2 / 2 | −0.8% / +3.5% |
| stress | 6 | 2.34 | 3.82 | 0.00 | 1.02 | −1.02 [−2.50, +0.47] | 0 / 3 | −8.8% / +3.2% |
| stress | 8 | 1.83 | 4.54 | 0.24 | 2.02 | −1.77 [−3.76, +0.21] | 0 / 4 | −7.2% / +3.7% |
| mxl | 6 | 3.50 | 3.16 | 2.06 | 1.01 | +1.05 [−1.16, +3.26] | 2 / 3 | +1.5% / +3.3% |
| mxl | 8 | 2.54 | 3.20 | 2.39 | 1.72 | +0.66 [−1.77, +3.10] | 3 / 2 | +2.3% / +4.5% |

No cell differs significantly (Holm p ≥ 0.41); under stress HB over draws is ahead on points. What remains for
MNP: profit forecasts under correlated substitution (HB over draws under-forecasts by 7–9% under stress, MNP is
+3%). Open: whether Sawtooth's and Conjointly's simulators use posterior draws or point estimates by default.

### What incumbent users actually run (2026-10-23, documentation check)

- **Sawtooth:** its knowledge base recommends Randomized First Choice in general; simulating on 50–200 HB draws per
  respondent with share of preference is described as "another solid approach" that needs a manual edit of the
  .HBU file. Orme's 2020 paper (38 SKUs, brand / price CBC; repeated at 51 SKUs) found point estimates and draws
  give very similar sourcing, draws slightly stronger differential substitution, and recommends the simpler
  practice of simulating on point estimates (from the paper's abstract and search snippets; full text not read).
  So the default practice is point estimates with RFC or share of preference: the Part B arms SoP and RFC-S.
- **Conjointly:** MCMC HB multinomial logit; shares from "individual coefficients", share of preference by
  default (first choice optional, no RFC). Whether the coefficients are posterior means or draws is not stated.

Reading: against the incumbents' documented default (point estimates), T31's 10-rep Part B win stands (m = 6:
+1–2% vs SoP, +2–4% vs RFC-S). The same gap closes when HB is simulated over its draws, an option Sawtooth
documents but advises is unnecessary; so the advantage is real against practice but can be matched without a GPU
or a probit. Our results disagree with Orme's "very similar" on this shelf (HB "none" bias +3 to +5 pts on point
estimates, about −1 on draws); his comparison was on sourcing patterns, ours on absolute shares and decisions.
