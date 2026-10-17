# X1. Tree utilities vs linear utilities for conjoint: design (draft 2026-10-15)

**Status: PARKED (user, 2026-10-16).** Pooled phase complete: no method more share-accurate on the 40 held-out combinations (out/summary.txt). Heterogeneity phase piloted on split 0 only; two LC-P-XGB fixes (per-class early stopping, K grid to 6) noted, not made.

First experiment in the X series (exploratory, separate from the T tests). No pass/fail: this run is for
looking at the comparison. A pass rule, if any, is added afterwards and the experiment becomes a test.

## Question

Conjoint analysis is used to value products that don't exist yet. On real CBC data, with test tasks
containing attribute combinations never seen in training:

1. Do tree-based utilities (XGBoost) predict choices among unseen combinations better or worse than linear
   utilities?
2. How do XGBoost's SHAP-based part-worths compare with the linear models' betas?
3. Does modelling error correlation (probit Σ) matter more than the link function?

## Data

bayesm `camera`: real CBC, 332 respondents × 16 tasks = 5,312 tasks. Each task shows 4 cameras plus
"none" (chosen 25%).

| Attribute | Levels | Coding in the data |
|---|---|---|
| Brand | Canon, Sony, Nikon, Panasonic | 4 dummies vs "none" (so they double as brand constants) |
| Pixels, zoom, video, swivel, wifi | 2 each | 0/1 |
| Price | 0.79, 1.29, 1.79, 2.29, 2.79 | numeric |

No respondent covariates. Alternatives are **unlabeled**: positions carry no meaning and brands are
attributes (48% of tasks repeat a brand). Synthetic data is out of scope for X1 (see Later).

## Holdout: unseen level pairs

A split holds out one level pair. Every task in which any alternative has that pair is a test task; all
other tasks train. Whole tasks are removed, not single alternatives: dropping an alternative from a choice
set is valid only under logit (IIA), so it would bias the probit arms. Respondents are the same in train and
test; only products vary.

| Family | Pairs | Tasks removed per split | Role |
|---|---|---|---|
| Brand × price level (e.g., Nikon at 2.79) | 20 | 18–23% | Main: brand-specific price response |
| Brand × feature (e.g., Sony with wifi) | 20 | 43–53% | Main: brand-specific feature value |
| Feature × feature | — | 65–70% | Excluded: too little training data |

**Random-task reference.** For each split, a second split that removes the same number of tasks at random.
This separates "the combination is new" from "there is less training data". Without it, a drop in accuracy
can't be attributed to novelty.

**Tuning.** Any hyperparameter choice (tree depth, rounds, learning rate, RFC error scales) is made on a
validation pair held out from the training tasks, never on the test pair.

## Arms

All arms are pooled (one utility function for everyone) in X1. Heterogeneity comes later (see Later).

| Arm | Utility | Error structure | Purpose |
|---|---|---|---|
| **P-XGB** | XGBoost trees | Probit, variance-share Σ | Arm 1 |
| **P-XGB-I** | XGBoost trees | Probit, Σ = I | Isolates trees from Σ |
| **P-XGB-1** | XGBoost depth-1 (additive) | Probit, variance-share Σ | Trees without interactions |
| **P-LIN** | Linear | Probit, variance-share Σ | Arm 2 |
| **P-LIN-I** | Linear | Probit, Σ = I | Link-only control |
| **L-RFC** | Linear (aggregate MNL) | Logit, RFC simulator, tuned | Arm 3, pooled |
| **HB-RFC** | Linear, per respondent (HB-MNL) | Logit, RFC simulator, tuned | The incumbent (Sawtooth practice). Not pooled, so it is ranked in the heterogeneity phase; in the pooled phase it is shown for orientation only |

**Error structure.** Positions are unlabeled, so a "full" Σ over positions (as in `MultinomialProbit`) has
no meaning here. The probit arms use T1's variance-share Σ: each attribute a has a variance σ²_a shared
by alternatives in the same task with the same level of a, plus an independent term. The covariance is
built per task from which levels the alternatives share (`analysis/T1/run.py`, `err_cov`). Identification
as in T1: utilities relative to "none" (all features 0, utility 0), and the total error variance fixed.

**Link-only check.** With Σ = I, the probit is a logit with a different squashing function. If P-LIN-I and
L-RFC give the same predictions (after the ~1.6–1.8 scale factor), the link doesn't matter, and any
probit gain comes from Σ.

**XGBoost without softmax.** XGBoost with a custom objective: the negative MNP log-likelihood of each
observed choice. Per task:
- The trees give a utility for each of the 4 cameras.
- P(observed choice) is a 4-dimensional orthant under the task's Σ, by GHK with fixed scrambled Sobol
  draws (the integrator in `mnp_fit` and T6).
- The gradient for each alternative comes from autodiff through GHK in torch. The Hessian is the diagonal
  of the Fisher information, floored at a small positive value.
- Σ is fitted by alternating: trees with Σ fixed, then the σ_a with utilities fixed, using out-of-fold tree
  utilities (5-fold over respondents) so the Σ step doesn't see in-sample fits.

**L-RFC and HB-RFC.** RFC as in T1 (attribute-level and product-level error draws on top of the part-worths,
first choice per draw), error scales tuned on the validation pair. HB-MNL by bayesm `rhierMnlRwMixture`
(T1's `hb.R`). RFC predictions are simulated shares, so their log-likelihood uses enough draws that
simulation noise is below 0.1% of the total.

## Measures

Fixed before running. Per split, then pooled over the 20 splits of each family.

**Prediction on test tasks:**
- Mean log-likelihood per task (primary)
- Hit rate
- Share error on the held-out combination: predicted vs observed choice share of alternatives with the
  held-out pair, in share points (absolute first)
- "None" share error

Each with 95% CIs from a respondent bootstrap. Arm comparisons are paired (same splits, same bootstrap
resamples), reported as differences with CIs, relative lift with CI, and Holm-adjusted p across the
comparisons made. Significance and materiality are reported separately.

**Novelty penalty:** each metric on the level-pair split minus the same metric on its random-task
reference.

**Explanation (full-data fits, no holdout):**
- Linear arms: betas, attribute importance (range share), willingness to pay (part-worth ÷ price
  coefficient), each with bootstrap CIs.
- XGBoost arms: mean SHAP value per attribute level, centered per attribute, on the utility scale; the
  same importance and WTP from those; SHAP interaction values for every brand × feature and brand × price
  pair.
- Comparison on scale-free quantities only: importance shares, level order, WTP ratios. Raw SHAP and raw
  betas are on different scales.
- For each held-out pair: what interaction P-XGB implies for the combination it never saw, vs the additive
  prediction.

## Run order

1. **Wiring (full data, no holdout).** Fit every arm once. Checks: aggregate MNL matches bayesm's
   aggregate logit; P-LIN-I ≈ MNL × scale; P-XGB-1 SHAP part-worths are close to P-LIN's betas. If the
   last fails, the custom objective is wrong.
2. **Pilot:** 2 splits per family, all arms. Time and cost per split measured here and stated before the
   full run.
3. **Full run:** 40 level-pair splits + 40 random-task references.
4. **Heterogeneity phase** (below), only after the pooled results are written up.
5. **Write-up:** `analysis/X1/RESULTS.md`, and a summary in `CONJOINT_X_SERIES.md` (to be created).

## Implementation notes and deviations (2026-10-15)

Code: `core.py` (data, splits, probit likelihood, arms), `x1.py` (runner), `summarize.py`, `explain.py`,
`lc.py` (heterogeneity phase). Data: `Rscript export_camera.R out/data/camera_long.csv`.

- **Wiring (done, all PASS).** MNL matches bayesm's `llmnl` optimum to 6e-7; MNL / P-LIN-I ratio is a
  constant 1.45 (CV 1.4%); P-XGB-1 SHAP part-worths vs P-LIN betas correlate 1.000. GHK checked against
  SciPy (1e-5). The XGBoost gradient and Hessian match finite differences.
- **Σ for the tree arms is the linear probit's, held fixed** (no alternation, no out-of-fold Σ step). First
  version: shares refitted on the validation tasks with the early-stopped trees' utilities. That was
  unstable: validation tasks are picked by one level pair, so they are few and unrepresentative, and on
  split 68 (Nikon + wifi) the own-error share went to 0, the covariance became near-singular and the tree
  arms' test log-likelihood collapsed (−1.61 and −1.32 vs −1.23 for linear). All tree-arm results were
  refitted with P-LIN's Σ (`x1.py run --refit-trees`). P-XGB vs P-XGB-I still tests Σ with trees.
  5-fold out-of-fold fits were rejected on cost (about 3.5× the tree time on a 2-core CPU).
- **Own-error share floored at 1%,** so cameras identical on every attribute keep a nonsingular covariance
  (split 76 failed in the linear fit without it). Fitted own shares are 0.26–0.47, far from the floor.
- **Smoke test, not the full grid:** after 6 brand × price pairs and 4 brand × feature pairs (each with its
  random reference) the signal was read and the run stopped (user, 2026-10-15: smoke test first).
- **Boosting settings fixed, not tuned per split:** depth 3 (P-XGB, P-XGB-I) and 1 (P-XGB-1), learning rate
  0.1, rounds by early stopping (patience 30, cap 2000; the pilot's cap of 600 was hit by depth 1, so the
  pilot was rerun inside the full run). GHK draws: 128 for boosting gradients, 512 for linear fits and the
  Σ step, 4,096 for test probabilities. The exact Hessian diagonal is recomputed every 10 rounds.
- **RFC tuned by validation log-likelihood,** not by share MAE as in T1: X1's tasks differ per respondent, so
  there are no common holdout tasks to compute shares on. Tuning fits use training minus validation;
  final fits use all training tasks.
- **HB-MNL:** bayesm `rhierMnlRwMixture`, 20,000 iterations, keep every 10th, posterior means from the
  second half (T1's `hb.R`).
- **Pilot (8 splits):** 240–424 s per split on the local CPU; outputs archived in `out/pilot_v1/`.

## Heterogeneity phase: latent classes

P-XGB is our method; this phase extends it to K latent classes. `camera` has no respondent covariates,
so heterogeneity can come only from repeated choices: 16 tasks per person.

This is where the incumbent enters. Sawtooth practice is HB-MNL + RFC: a logit per respondent (IIA within
each person), with heterogeneity across people and RFC's error draws softening IIA in the aggregate shares.
The pooled phase can't include it fairly; this phase does.

**Arms (same splits):**
- **HB-RFC:** the incumbent. HB-MNL (continuous heterogeneity) + tuned RFC.
- **LC-P-XGB:** our method with K classes, each with its own tree ensemble; shared variance-share Σ; class
  shares π_k.
- **LC-P-LIN:** identical with linear utilities per class (trees vs linear at the same heterogeneity).
- **LC-L-RFC:** latent-class logit + tuned RFC at the same K (Sawtooth also sells latent-class MNL). Separates
  "continuous vs discrete heterogeneity" (HB-RFC vs LC-L-RFC) from "probit and trees vs logit and RFC"
  (LC-L-RFC vs LC-P-LIN vs LC-P-XGB).

The headline comparison is LC-P-XGB vs HB-RFC; the other two arms say why any gap exists.

**Fitting (EM, both arms):** person n's likelihood is Σ_k π_k ∏_t P(choice_nt | class k, Σ), a finite sum,
so there are no outer simulation draws.
- E-step: each person's class probabilities from their training-task choices (GHK per task).
- M-step: refit each class's utility with tasks weighted by those probabilities (XGBoost per-row weights
  in the custom objective); update π_k; refit Σ on out-of-fold utilities.
- K in {2, 3, 4}, chosen by validation-pair log-likelihood; several random starts per K.
- Test tasks belong to the same respondents, so prediction uses each person's posterior class
  probabilities from their training tasks.

**Respondent ID is not a feature.** As a number it is meaningless (trees would split on arbitrary ID
ranges). As a category it is a per-person fixed effect: 332 levels from 16 tasks each, which overfits and
cannot extend to new people.

**Guard against a distraction caused by our own design.** This phase may fail because our EM or objective
is wrong, not because the idea is. Before reading any comparison:
- K = 1 must reproduce the pooled P-LIN and P-XGB fits (log-likelihood equal within GHK noise).
- LC-P-LIN must agree with a standard latent-class logit on class sizes and part-worth ordering (after
  scale), at the same K.
- Time box: the pilot plus one full pass. If a wiring check fails or the time box runs out, stop and
  report the failure; no tuning toward a result.
- Results from this phase are labeled separately from the pooled comparison and don't change its
  conclusions.

**Preferred route, later:** heterogeneity explained by respondent covariates (demographics) on a dataset
that has them. Trees use covariates naturally (covariate × product-feature splits) and the model extends
to new people. Covariates usually explain only part of taste variation, so classes or random effects may
still be needed on top.

## Cost

Expected CPU only. Per task, one 4-dim GHK; 5,312 tasks. XGBoost's custom objective calls GHK once per
boosting round, so P-XGB is the slowest arm. The 4 GB local box limits batch size; if the pilot shows
more than about 10 minutes per fit, the GHK step moves to a Modal T4 and the cost is stated before
running.

## Later (not X1)

- **Continuous heterogeneity.** Respondent-level random effects in the probit arms. T4's panel probit cost
  about $45 per fit at J = 20 because its simulation was nested: outer draws for each person's random
  effects (512) times inner GHK draws per task (256), for every person on every likelihood evaluation. The
  missing scaling idea is probably to avoid that nesting. Candidates:
  - Bayesian data augmentation (Gibbs, as in bayesm `rmnpGibbs`): cost grows linearly in tasks, with no
    nested integral.
  - Laplace or variational approximation of each person's random effects in place of outer draws.
  - Stochastic gradients over minibatches of respondents.
  - MACML (analytic, no simulation; tied GHK in T9).
- **Synthetic augmentation.** Fit a model to `camera`, sample new respondents and tasks with the
  combinations a split needs. The generator must not belong to any arm's family (e.g., resampled real
  respondents, or a mixture or latent-class generator), or it hands that arm an advantage.
- **Demographics:** find a real CBC dataset with respondent covariates (see Heterogeneity phase).

## Comparisons (all kept, user 2026-10-15)

Each control changes one thing from a main arm:

| Comparison | Only difference | Question |
|---|---|---|
| P-LIN vs P-LIN-I | Σ removed | Does error correlation help? |
| P-LIN-I vs L-RFC | Probit vs logit link, no correlation | Does the squashing function matter? |
| P-XGB vs P-XGB-I | Σ removed, with trees | Does Σ still help once utilities are flexible? |
| P-XGB-1 vs P-LIN | Additive trees vs linear | Does flexible shape per attribute help? |
| P-XGB vs P-XGB-1 | Interactions allowed vs not | Do learned interactions help or hurt on unseen combinations? |

