# T2. Linked-set scaling, simulated MBC: design (approved 2026-10-08)

Items 2 and 4. No analysis code has been written for this test.

## Question

In menu-based choice (pick-any of M binary items, prices varied by task), when the items
whose choices are dependent (the linked set) number k ≥ 6:

- **Q1.** Do decisions from coded subsets (Sawtooth MBC practice, ≤ 5 items per subset,
  independence between subsets) differ from those of a multivariate probit (MVP) over the
  whole menu?
- **Q2.** Among MVP estimators (IFM, full MLE, Gibbs), which reaches the decision, at what
  error?
- **Q3.** How does time-to-decision grow with k, per method, at matched decision error?

## Non-goals

- CBC / single choice (T1, T4). Real data (T3).
- Engine accuracy or speed sweeps (settled).
- Flexible margins. The DGP is linear, so linear margins are correctly specified for every
  method (PLAN.md, "Margin model").
- Building MACML (see open point B).

## DGP

Reuses the pick-any structure of `tests/joint_menu`.

- **Menu.** M = 12 binary items. The first k form the linked set, k ∈ {5, 8, 12}; the
  other M − k are independent of everything.
- **Prices.** Each item has 3 price levels (−20%, base, +20%), drawn independently per task
  (an MBC-style randomized design).
- **Utility, respondent n, task t, item j:**
  U_ntj = a_j + b_j · log(p_ntj / p_j0) + ε_ntj, pick j if U > 0.
  ε_nt ~ N(0, R), unit diagonal. Own-price effects only; no true cross-price terms.
- **Correlation R (truths):**
  - **L0 null.** R = I. Guard: does MVP invent correlation?
  - **L1 factor.** Linked items load on one factor, ρ_ij = λ_i λ_j with λ ~ U(0.5, 0.8)
    (ρ ≈ 0.25–0.64). Fixed λ per k, from a seed.
  - **L2 two-cluster.** Linked items split in two halves, within-half factor as L1, plus a
    weak shared factor (cross-half ρ ≈ 0.1–0.2). This is the case where a ≤ 5 cap must cut
    a real dependence.
  - **LC complementarity** (guard for the incumbent). R = I, plus pattern constants γ on
    two linked pairs and one triple (as `tests/joint_menu` J3). Coded subsets are correctly
    specified; MVP is not.
- **Intercepts a_j.** Base take rates 10–40%, fixed per k from a seed. Slopes b_j ≈ −1.5.
- **Respondents and tasks.** N = 1,000, T = 10 training menus + 4 fixed holdout menus.
  No respondent heterogeneity in v1 (open point A).
- **Replicates.** R = 20 datasets per (truth, k), fixed seeds.

## Methods

**Incumbents (Sawtooth MBC proxy).**
- **CS-F (faithful coded subsets).** An analyst rule on the training data: rank item pairs
  by |lift| from counts, merge greedily into subsets of at most 5 items (2^5 = 32 patterns
  ≤ the ~36 recommended), stop at lift below a threshold (from MBC guidance; open point F).
  Each subset is an MNL over its 2^s patterns: pattern constants, item price terms, and
  within-subset cross-price terms kept by a screen (p < 0.20, as `joint_vs_proxy`).
  Items outside subsets are binary logits. Independence between subsets and singletons.
- **CS-O (oracle coded subsets).** The analyst knows the true linked set and cuts it into
  the best blocks of ≤ 5 (by within-block |ρ|). Measures the cost of the cap itself.
- **IL.** Independent binary logits for every item (lower bound).

**MVP (one model, three estimators).** Binary probit margins, linear in all 12 log prices
(own and cross terms; the truth has own only), full 12 × 12 R (or factor-structured; open
point D).
- **MVP-IFM.** `multivariate_probit` (CPU): per-item probits, then pairwise R.
- **MVP-ML.** Full-information ML by GHK with fixed Sobol draws on GPU (the `mvp_fit`
  Modal endpoint, `fitter="modal"`), SEs.
- **MVP-Gibbs.** `bayesm::rmvpGibbs` (data augmentation), posterior means after
  identification by the correlation normalisation. On CPU containers.
- **MACML.** No implementation here (open point B).

**Decision layer.** Every MVP estimator's pattern probabilities come from the engine
(orthant of dim ≤ 12 per pattern and price scenario); CS from its MNLs × independence.

## Metrics (candidates; you set the tradeoffs)

**Joint structure (descriptive).**
- Pair-joint and triple-joint MAE (points) on holdout menus; P(basket size) distribution.
- Cross-subset pairs: joint error on pairs that CS-F placed in different subsets.

**Decisions** (each scored against the truth simulator).
- **D-bundle.** Pick the pair, and the triple, with the highest joint take (and highest
  lift) at base prices. Flip vs truth.
- **D-price.** Raise one linked item's price 20%: change in each other item's take and in
  "basket contains ≥ 1 linked item". Error, and a flip on the sign of the largest
  cross-effect.
- **D-promo.** Choose which one item to discount 20% to maximise expected menu
  contribution Σ (p − c)·P(take) with fixed unit costs. A cross-subset decision (item 4).
  Flip and regret (% of true optimum).
- **Holdout log-likelihood** of observed patterns (descriptive).

**Speed (Q3).** Wall-clock time to decision per method (fit + decision search), against k,
at matched decision error (PLAN.md baselines rule). Machine, backend and versions
recorded.

## Quick mode vs full run

- **`--selftest`:** k = 8, L1, R = 1, N = 300, all methods, fail-fast checks (truth
  pattern probabilities sum to 1; MVP-ML nll at fit ≤ at truth; Gibbs trace drift < 1%;
  RAM < 1 GB). Under 10 min.
- **Full run:** 4 truths × 3 k × R = 20 = 240 datasets. MVP-ML on Modal T4s in parallel;
  CS, IL, IFM local; Gibbs on Modal CPU containers (R + bayesm image), one per dataset.
  I state the projected time after the selftest and ask before launching.

## Expected runtime (rough; refined after selftest)

- CS / IL / IFM: seconds per dataset, ~10–20 min total locally.
- MVP-ML: ~1–3 min per fit on a T4; 240 fits in parallel, ~15 min wall.
- Gibbs: unmeasured; if ~2 min per dataset, ~10 min wall on parallel CPU containers.
- Decision layer: ≤ 2^12 patterns × scenarios per method, seconds with the local engine.
- **Full run: about 1 h wall**, most of it the local CS/IFM loop and the result
  collection. Gibbs run locally instead (2 cores, 240 datasets) would add several hours.

## Kill / pass rule (blanks are yours)

On truths L1 and L2, pooled over k ≥ 8 and replicates.

- **No decision difference.** CS-F has at most ___ more flips than the best MVP estimator
  (of 3 decisions × R × k), and mean D-promo regret difference under ___ %.
- **No time saving.** The best MVP estimator's time-to-decision is not more than ___×
  faster than CS-F at matched error.
- **KILL** if both hold. If killed, T3 is skipped.
- **PASS** if either fails in MVP's favour.
- **Guards** (reported as caveats, not part of the kill): L0, MVP's median |ρ̂| above ___
  means it invents dependence; LC, MVP's flips above ___ means misspecification costs.

## Decisions (2026-10-08)

Approved as proposed: A without heterogeneity (follow-up), B MACML to backlog, C engine in
the decision layer only (an engine fitter is a follow-up; the hypothesis under test is full
MLE in general), D–G as proposed. Kill blanks unset. Follow-ups tracked in PLAN.md.

## Open points (as drafted)

- **A. Heterogeneity.** v1 has none, which isolates the k question and keeps MVP's
  likelihood one k-dim orthant per menu. Sawtooth fits MBC sub-models by HB, so a
  random-intercept truth (α_n ~ N(a, Σ_α)) is more faithful but needs panel methods for
  every arm (the T1 problem, now k·T dims). Proposal: v1 without, a follow-up with.
- **B. MACML.** No usable implementation. Per the baselines rule I stop here: drop it, or
  approve writing one. Proposal: drop; Bhat's MACML is an analytic MVN-CDF approximation
  inside composite ML, the same role the engine plays, so MVP-ML / IFM with the engine
  covers its question.
- **C. Engine MLE.** No full-MLE fitter uses the engine; MVP-ML uses GHK on GPU. In T1 the
  engine was too rough to optimise at 36 dims; at k ≤ 12 per-menu probabilities are not
  tiny, so it might work. Proposal: engine in the decision layer only for v1 (Q3 then
  measures decision-time, not fit-time, for the engine).
- **D. R structure.** Full 12 × 12 R (66 correlations) for MVP, or a factor-structured R?
  Proposal: full R for IFM and ML (it's what a user would fit), factor R as a variant only
  if full R fails to converge at N = 1,000.
- **E. Grid.** k ∈ {5, 8, 12}, one N, ρ fixed by truth. The PLAN mentions sweeping ρ and N
  too; proposal: add N ∈ {300, 3000} at k = 8 only, after the core run.
- **F. CS lift threshold.** Need a source for the analyst's stopping rule (MBC docs or
  your practice). Proposal: merge while pair lift > 1.2 or < 0.8, cap 5.
- **G. Costs.** Placeholder unit cost 40% of base price for D-promo.
