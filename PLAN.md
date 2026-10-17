# Plan: where multivariate probit beats incumbent choice methods

Status 2026-10-16: T1–T3, T5, T6, T14 PASS. T7, T8, T9, T11 FAIL. T13 KILL. T10 Part A inconclusive, Part B
paused. T4 paused. T15: D1 PASS / D4 SOFT FAIL, S0 guard pass. T16 FAIL (T6's win is heterogeneity).

- **Each test's full write-up** (design, kill criteria, user decisions, results) is in `analysis/T<n>/TEST.md`.
  T1–T3 are in their own DESIGN.md / RESULTS.md, with the original plan and history in [HISTORY.md](HISTORY.md).
- **Related documents.** Model catalog: [MODELS.md](MODELS.md). Backlog: [BACKLOG.md](BACKLOG.md). Search and
  error vs speed: [SEARCH.md](SEARCH.md).
- **Rules.**
  - Kill criteria are set by the user; nothing runs until they are filled in.
  - Scope is strict: if a test suggests expanding it, stop and ask (no new arms, truths or metrics).
  - The earlier product and working notes moved to `PRODUCT_NOTES.md`. They cover machine limits, the orthant
    backend, and style rules, and still apply.

## Context

- **Engine.** A GPU MVN orthant engine: a deterministic approximation, about
  70–150× faster than QMC Sobol for n = 4 to about 128. It is deployed on a
  model backend API. For small tasks, the same code ships as a local `.so`
  (`multivariate_probit.orthant`; see `rfc.orthant_cdf`).
- **Library.** `multivariate_probit` 0.2.3: sklearn-style, cross-fit IFM.
- **Aim.** Find where multivariate probit (MVP) beats the methods incumbent
  platforms use (Sawtooth: HB-MNL, RFC simulation, MBC combinatorial coding),
  and where the engine is what makes MVP feasible.
- **Settled, not re-tested.** The engine envelope: smoothness, accuracy at
  negative correlation, and error on small probabilities. The engine is used
  as-is. No accuracy sweeps of the engine.

## Three questions (kept separate in every write-up)

- **Q1. Model.** Is MVP better than, or as good as, the incumbent's model (RFC
  in CBC, coded subsets in MBC)? Parity counts if MVP is explainable or faster.
- **Q2. Estimator.** Is full MLE better than IFM? IFM does not apply to CBC
  single-choice data. There, the probit is fitted by full MLE or a substitute
  (GHK, Gibbs data augmentation, MACML).
- **Q3. Speed.** What does the engine buy in time-to-decision against scalable
  incumbent methods?

## Terminology

- **Linked set:** items whose choices are actually dependent in the data. A
  property of the data.
- **Coded subset:** items the analyst codes combinatorially (Sawtooth MBC),
  capped at about 5 binary items (≤ ~36 combinations recommended; ≥ ~75 too
  sparse). A modeling choice.
- **Decision layer:** the business decision a model output feeds (price, line
  extension, bundle, assortment, a cost-weighted metric). All kill criteria
  are decision-layer tradeoffs set by the user. Designs propose candidate
  metrics and leave the cutoffs blank.

## The five items under test

1. **Shelf CBC with 20+ SKUs.** Logit is used because there is no scalable
   probit; IIA is tolerated under that constraint, not preferred. The binding
   constraints are computation and identifiability. An attribute-structured Σ
   (as in RFC) fixes identifiability.
2. **MBC linked sets with k ≥ 6.** These can't be modeled even when the
   analyst knows the items are linked.
3. **σ distortion.** RFC reproduces near-duplicate effects by inflating σ
   (about 4× in an earlier simulation; unverified, not used as a reference). The answer may be wrong. Even when
   shares are right, a wrong σ means a wrong explanation and wrong downstream
   use.
4. **Specification across subsets.** Before the fit, the analyst must declare
   where independence holds by choosing coded subsets. After the fit, any
   decision search over candidates that span subsets assumes independence
   between them. MVP learns the structure, and every combination is a query.
5. **RFC plus an estimated correlation matrix**, for mapping to business
   decisions (cost-weighted, margin-weighted). Depends on item 3: a wrong σ
   gives wrong weights.

## Tests

### Open

- **T15 · Engine pipeline on the shelf, J = 20–60 (2026-10-16): D1 PASS, D4 SOFT FAIL; S0 guard pass.**
  - **Main question (MNP via the M*P pipeline vs HB-MNL + RFC-S, regret against the truth, 5 datasets).**
    - D1 PASS: assortment, Δ +1.85 pts [+1.30, +2.40].
    - D4 SOFT FAIL: joint line + extension + price, +0.20 [+0.13, +0.27], reliably better but below the 0.5 floor.
  - **Pipeline vs accurate-only search:** a tie at J = 20.
  - **Engine vs OVUS as the screener:** 3.6–20× faster.
  - **S0 guard:** under a logit truth M*P ties or slightly beats the incumbent, with no invented similarity.
  - **Cost:** M*P costs more per decision (H100 fit about 20–27 min vs HB 3–4 min on CPU). About $13 in all.
  - [Full write-up](analysis/T15/TEST.md), [design](analysis/T15/DESIGN.md).
- **T10 · MACML vs GHK follow-up: Part A inconclusive, Part B paused (2026-10-10).**
  - Part A: at N = 150 / 300, MACML and GHK made nearly the same decisions, as labelled in advance.
  - Part B (J = 20): paused at the Step 0 probe. GHK at the draw count that recovers the shares projected over the
    3 h cap on a T4. An H100 brings it to about 17 min per fit (2026-10-12). The choice of draw count is open.
  - Includes the OVUS vs engine side study. [Full write-up](analysis/T10/TEST.md).
- **T4 · Shelf scaling: paused on cost (2026-10-09).**
  - The recovery gate failed as specified. Shares were recovered only at 512 × 256 draws, at 4.5 h + 3.6 h SEs on
    an L4. pack0 is about −3 SE in every fit.
  - The full run was estimated at about $45 (J = 20) to $600 (all J). T13 and T15 reuse its J = 20 fit.
  - [Full write-up](analysis/T4/TEST.md), [design](analysis/T4/DESIGN.md).

### Closed

- **T16 · T6 against per-household independence (practitioner TURF): FAIL (2026-10-17).**
  - HB per-household independence closes 94% of T6's gap: pooled reach error MVP 2.25 vs HB-IND 2.38 pts (Δ +0.13,
    95% CI [−0.06, +0.32]); greedy portfolios identical at every k. Control gate passed. About $0.16.
  - T6's PASS was against pooled independence; it is not an advantage over Sawtooth-style TURF.
  - [Full write-up](analysis/T16/TEST.md).
- **T14 · MBC linked sets at k > 12: PASS (2026-10-12 / 13).**
  - Part A (simulated, k = 20 / 30): coded subsets flipped 207 of 240 decisions, MVP-IFM 50.
  - Part B (24-item baking cluster, real): the holdout backs MVP on 3 disagreements, coded subsets on 0, at
    the threshold.
  - [Full write-up](analysis/T14/TEST.md).
- **T13 · Line of 6 of 20 with joint survival: KILL (2026-10-11).** The joint and independent rules picked the
  same line on all 18 decisions. About $0.06. [Full write-up](analysis/T13/TEST.md).
- **T11 · Joint chance-constrained product line: FAIL (2026-10-11).** The joint constraint never chose a
  different line from the independent one (0 of 640 searches). What helps is the uncertainty-aware constraint
  itself. About $1.60. [Full write-up](analysis/T11/TEST.md).
- **T9 · MACML as an incumbent probit arm: FAIL (2026-10-10).** MACML and GHK made identical decisions (0 flips,
  0 regret, all truths, N = 600), so there was no GHK win. About $0.35. [Full write-up](analysis/T9/TEST.md).
- **T8 · GPR kernel error covariance: FAIL on the primary criterion (2026-10-10).** Under a continuous-attribute
  truth the kernel tied variance shares (0.51% vs 0.56% regret, inside 2 SE). The no-loss guard passed. About
  $2. [Full write-up](analysis/T8/TEST.md).
- **T7 · Engine vs shared-draw simulator for TURF search: FAIL (2026-10-10).** At matched regret the simulator
  was 9–16× faster than the engine at every k. The engine's picks had essentially zero regret. About $0.05.
  [Full write-up](analysis/T7/TEST.md).
- **T6 · Correlated-reach TURF on public basket data: PASS (2026-10-10).** Against pooled independence only; T16 shows the gain is household heterogeneity.
  - MVP's held-out reach error was 49% lower than the independence formula's.
  - Greedy portfolios flipped at k = 3–6, and MVP won all 4 on held-out reach.
  - About $0.01. [Full write-up](analysis/T6/TEST.md).
- **T5 · N-sweep (static sample savings): PASS (2026-10-09).** MVP matched the decision error that HB-MNL + RFC-S
  needs at N ≥ 720 already at N = 150, on both correlated truths. RFC-S did not improve with N. About $3.40.
  [Full write-up](analysis/T5/TEST.md), [results](analysis/T5/RESULTS.md).
- **T1–T3: PASS (2026-10-08).**
  - T1: MVP beats RFC, with caveats.
  - T2: 28 vs 146 flips of 240.
  - T3: the holdout backs MVP on 4, coded subsets on 0.
  - Results in `analysis/T1`–`T3`; [HISTORY.md](HISTORY.md).

## Baselines

- Speed is always measured at matched error on the decision-layer output, as
  wall-clock time to decision. Never against QMC Sobol alone.
- If a baseline has no usable implementation, stop and ask before writing
  one.
- Reporting (user, 2026-10-13; from T14 on): every compared decision gets the absolute difference in points with a
  95% CI, then the relative lift with its CI, and a multiplicity-adjusted p (Holm over the decisions compared).
  Points lead; relative lift is never reported alone (small bases inflate it). Statistical significance and
  materiality (the pre-set point floor) are reported as separate labels. Pre-registered kill rules stay as written;
  this changes the reporting, not the verdicts.
- The Sawtooth proxy must be faithful: tuned RFC, own- and cross-effects, and
  coded subsets chosen as an analyst would from counts and lift. Every
  deviation from Sawtooth practice is documented.

## Prior work in this repo (inputs, not conclusions)

- `tests/rfc_vs_probit/`: near-duplicates, oracle utilities. Tuned
  RFC-style S2 had 5.6–6.4 points of sibling source-of-volume error against
  1.6–3.2 for the probit kernel. No decision flips for S2. Relevant to T1.
- `tests/joint_menu/`: pick-any menu, k = 6. Independent logits picked the
  wrong bundle under correlated truth; MVP and coded patterns did not.
  Relevant to T2.
- `analysis/rho_test/`: dunnhumby pipeline, 8 items, GO on pasta × pasta
  sauce (ρ_B = 0.70). The data source for T3.
- `analysis/joint_vs_proxy/run.py`: pending head-to-head (coded subsets
  C1 = 3 items, C2 = 2 items, vs MVP), no results yet. Absorbed by T3.
- `research/similarity_kernel/`: kernel on product error, one replicate run.

## Environment and baselines (2026-10-06)

- R 4.3.3 and bayesm 3.1-6 installed (Ubuntu packages). HB-MNL proxy for
  T1/T4: `bayesm::rhierMnlRwMixture`, a documented deviation from Sawtooth
  CBC/HB. Gibbs for T2/T4 can use `bayesm::rmnpGibbs` / `rmvpGibbs`.
- No local GPU. GHK and MACML have no ready implementation here; each is
  raised at its test's design step before any code is written.
- The "σ about 4×" figure is unverified and is not used as a reference.
