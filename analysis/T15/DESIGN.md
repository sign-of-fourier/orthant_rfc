# T15. Engine pipeline on the shelf, J = 20–60: design (draft 2026-10-16)

Kill criteria set by the user 2026-10-16, with resolutions (see Kill criteria). Approval to start Step 0 is
pending. No analysis code has been written.
This is T4's decision test, reframed as a search-throughput test. T4 stays paused. T12 is unused; the next free
number after T14 is T15.

## Framing

- Accuracy gaps of 1e-3 to 1e-4 in orthant probabilities are ties. The engine's accuracy envelope is settled, so
  there are no engine accuracy sweeps here.
- The engine's claimed advantage is throughput. It shows up only through the **engine pipeline**:
  - **Stage 1.** The engine scores a very large candidate set, by exhaustive or wide search.
  - **Stage 2.** An accurate integrator (GHK-QMC) rescores the top K, then runs a local search from there.
- **Question.** At the same wall-clock budget, does scoring about 100× more candidates in stage 1 find better
  decisions than a search that uses only the accurate integrator?
- **Three questions, reported separately (PLAN.md):**
  - Q3, speed: arms (b) vs (c) vs (d) vs (e). This is the primary question.
  - Q1, model: arm (a) vs the MVP arms, scored against the truth.
  - Q2, estimator: out of scope (one fitted model).

## Model (fixed; confirmed)

- **Fit.** T4's J = 20 recovery-gate fit: `analysis/T4/out/gate_g/fit_512x256.json`.
  - MVP-G, 512 × 256 draws, N = 1,000, S1 truth dataset. It has θ (25 parameters) and a sandwich covariance.
  - It is the same fit T13 used.
  - Caveat: pack0 sits at about −3.4 SE in every T4 fit, likely because of this dataset.
- **Parameters.**
  - b (11): brand 4, flavor 3, pack 2, price, none.
  - Taste SDs w (11): diagonal Ω, normal on every coefficient, including price.
  - Variance shares: brand / flavor / pack / product, with total error variance π²/6.
- **Confirmed: population shares are orthants with tastes folded into the covariance.** For a shelf of K SKUs
  plus "none", the population utilities are jointly normal:
  - Mean: μ = X b.
  - Covariance: Cov = X diag(w²) Xᵀ + Σ_a s_a S_a + (π²/6 on "none").
  - The share of SKU j is one orthant of the K differences U_j − U_k, so dim = K = shelf size.
  - This is exactly `t13.shares()`, which already computes T4's model this way (6-SKU shelves).
  - It holds because the tastes are normal (the price slope too), prices enter linearly, and ζ and ν are normal.
  - It would fail with a lognormal price slope, or with any decision that needs per-respondent shares. Neither
    applies here.
- **Each share is its own orthant, with no renormalisation.** T13 normalised by the sum of all K + 1 shares. Here a
  firm objective needs only the firm's own SKUs, so renormalising would force extra orthants.
- **H100 refits.** Allowed by you, at about 17 min and about $1.10 per fit (2026-10-12 probe). Proposed only as an
  optional replication for Q1 (open point E). The speed question needs one fitted model.

## Decisions (proposed; you confirm)

The firm is brand 0. It has 6 SKUs on the J = 20 shelf; T4's other 14 SKUs are competitors at T4's base prices.
Brand 0 has 12 SKUs in the 5 × 4 × 3 grid, so the firm has 6 unshelved candidates for extensions.

Costs and margins (proposed):
- Unit cost by pack: $2.60 / $3.00 / $3.40.
- Fixed cost F per kept or launched SKU, in share-weighted dollars. F is set so that keeping everything is not
  optimal: in T11, with no fixed cost, the optimum was to keep every SKU.
- Objective: Σ_kept (p_j − c_j) s_j − F · (# kept).

| ID | Decision | Candidate space at J = 20 | d (orthant dim) | Orthants per candidate | At J = 40 / 60 |
|---|---|---|---|---|---|
| D1 | Assortment, category view: keep exactly k of J at base prices, category margin | C(20, k) = 38,760 / 125,970 / 184,756 for k = 6 / 8 / 10. Exhaustive feasible; borderline at the budget | k | k | C(40, 10) = 8.5 × 10⁸; C(60, 10) = 7.5 × 10¹⁰. Infeasible |
| D2 | Joint price grid: the firm's 6 SKUs, 9 levels each ($3.50–$5.50 by $0.25) | 9⁶ = 531,441 | 20 | 6 | Same space, larger d (40, 60) |
| D3 | Line extensions: each of the 6 brand-0 candidates is off or launched at one of 5 prices | 6⁶ = 46,656 | 20–26 | 6–12 | d up to 46 / 66 |
| D4 | Joint line + extension + price: all 12 brand-0 slots, each off or on at one of 5 prices | 6¹² = 2.2 × 10⁹. Infeasible | 14–26 | ≤ 12 | 6¹⁸ ≈ 10¹⁴ at J = 40 (if the firm has 12 on shelf) |

- The cost/margin weighting applies to every decision through the objective above.
- **Instances:** 3 per decision, giving 12 instances.
  - D1: k = 6, 8, 10.
  - D2–D4: three cost scenarios (base, pack costs ±20%, F doubled).
- **D2 is easy.** Shares move smoothly in price, so local search alone may find its optimum. It is included as a
  control: a decision where a wide stage 1 should not matter.

## Arms

All MVP arms use the same fitted θ̂ and the same search code; only the stage-1 scorer and the budget split differ.

| Arm | Stage 1 | Stage 2 |
|---|---|---|
| (a) HB-MNL + tuned RFC-S, incumbent | Sawtooth-style scenario search on the RFC simulator (see Search) | — |
| (b) MVP + engine pipeline | Engine (service default settings, float64, as T7), on a large candidate stream | GHK-QMC at M_acc rescores the top K, then local search |
| (c) MVP + accurate-only | GHK-QMC at M_acc on the stream, at the same wall-clock budget as (b) | Same local search |
| (d) MVP + shared-draw simulator | Common-draw first-choice simulator on the stream | Same as (b) |
| (e) MVP + low-M GHK stage 1 (user, 2026-10-16) | GHK-QMC at its adequacy M (below M_acc) on the stream | Same as (b) |

- **(a) Incumbent.**
  - HB-MNL is `bayesm::rhierMnlRwMixture` with T1/T4's prior and 20k iterations, fitted on the gate dataset's 12
    training tasks.
  - RFC-S is tuned on its 6 holdout tasks (`t4.tune_rfc`).
  - The gate dataset is regenerated with `t4.make_data` using the gate's seed. A wiring check reproduces the fit's
    nll at θ̂.
  - It runs on CPU. It answers Q1, so its time is reported but not budget-matched.
- **(d) Shared-draw simulator.**
  - Draw n = (β, ζ, ν) once for all 60 grid SKUs plus "none". A candidate's share comes from argmax over the
    present SKUs.
  - Prices only shift the price term (β_price,n × Δp), so one draw set scores every candidate.
  - Runs on GPU, batched.
  - **This is a deviation from "bitset as in T7".** T7's bitset form (OR + popcount) computes reach. A first-choice
    share needs argmax over masked utilities, so this is the best-practice equivalent: common draws, GPU, batched.
  - N is the smallest that reaches matched decision error in Step 0, as in T7.
- **Reference.**
  - GHK at high M (2^14 × 4 scrambles, as T13) at θ̂.
  - It rescores the union of every arm's shortlists (top K at every budget and seed) plus every final pick.
  - **Assumption: the true best is in that union.**
  - Check: on D1 at J = 20, an exhaustive pass at M_acc tests the assumption directly. This is cheap: at most
    184,756 × 10 orthants at d ≤ 10.
- **Truth.** T4's S1 parameters. Each pick's truth value is computed by the same high-M GHK at θ_true; population
  shares under the truth are orthants too. Regret against the truth includes estimation error from one dataset,
  which is shared by (b), (c) and (d).

## Search (same code for (b), (c), (d))

- **Candidate stream.** A seeded uniform random order over the decision space.
  - For D1 at J = 20 the stream is a random permutation of all subsets, so a long enough budget becomes
    exhaustive.
  - Any prefix of the stream is a uniform random sample, so one run at the largest budget also gives every smaller
    budget's stage 1 (anytime logging).
- **Stage 1.** Score the stream until a fraction f of the budget is spent (proposed f = 0.8), then keep the top K.
  K ∈ {10, 30, 100, 300} is reported as a recall curve; the run uses K = 100 (proposed).
- **Stage 2.**
  - GHK at M_acc rescores the top K.
  - Then best-improvement local search with GHK from the best candidate, until no improvement or the budget runs
    out.
  - Neighbourhoods: D1 is a 1-swap (k(J − k) neighbours); D2 moves one SKU by ±1 price step (12); D3 and D4 change
    one slot (≤ 60).
- **(c)** runs the same loop with GHK at M_acc as its stage-1 scorer, so its stream prefix is shorter by the
  throughput ratio.
- **(e)** runs the same loop with GHK at its own adequacy M (set in Step 0) as its stage-1 scorer.
- **Seeds and budgets (revised 2026-10-16).** 5 seeds per (arm, instance), giving 15 pairs per decision.
  - **Budgets are set from Step 0's measured throughput on the H100**, not fixed in seconds. They are the H100
    wall times in which the fastest adequate stage-1 scorer covers about 10⁴, 10⁵ and 10⁶ candidates. They
    replace {30, 100, 300} s.
  - A fourth budget, about 3.3× the largest, runs **only for arm (c)**, only in cells where (c) has not matched (b)
    at the largest base budget. It resumes from (c)'s saved state at the largest base budget. Its cost is
    reported before launch.
  - **A budget is a cap.** A run stops early once its stage-1 stream has covered the whole decision space and its
    local search has stopped improving. At J = 20 this covers D1–D3 for the fast scorers. Its regret is then
    carried forward to every larger budget.
- **Hardware (user, 2026-10-16).**
  - H100 for the main grid and the conditional fourth budget.
  - Step 0, the S0 refit and the S0 search may run on either GPU; whichever costs less is used.
  - All arms in one comparison run on the same GPU type.
- **(a)'s scenario search.**
  - Exhaustive when the space is at most 10⁵ (proposed); otherwise stochastic hill-climbing with restarts, in the
    style of Sawtooth's Advanced Simulation Module (grid, hill-climb, genetic).
  - Its budget is the same maximum wall time on CPU (proposed). Any deviation from ASM practice is documented.
- **M_acc** (the stage-2 GHK draw count) is set in Step 0: the smallest M whose objective error against the
  reference is below a fraction of the materiality floor. That fraction is blank.

## Metrics (cutoffs blank)

- **Decision regret.**
  - Against the reference: profit at θ̂ of the reference-best in the union minus the arm's pick, as % of the best.
  - Against the truth: the same quantity at θ_true.
  - Reported per instance and budget, with points (regret %) and a 95% CI over seeds and instances first, then
    relative lift with its CI, then Holm-adjusted p over decisions. Significance and materiality are separate
    labels (2026-10-13 rule).
- **Candidates evaluated** per budget, per arm and stage.
- **Recall.** Whether the reference-best is in the stage-1 shortlist, as a function of K, per arm and budget.
- **Regret vs budget curves** for (b) through (e), at the three throughput-set budgets on one H100, plus the
  fourth budget for (c) where triggered.
- **Catch-up curve.** The budget at which (c)'s regret matches (b)'s, interpolated on log-budget, if it ever
  does.
- **Wall time split by stage**, setup included.
- **Run details:** d, batch sizes, GPU type and cost per stage.
- **Speed is the wall-clock time to a decision at matched decision error** (PLAN.md baselines), never per-orthant
  speed alone.

## Kill criteria (user, 2026-10-16)

- **Materiality: 0.5% of max profit.** Accuracy differences below it are ties.
- **Scoring:**
  - The primary and attribution rules are scored against the reference: the high-M GHK best at the fitted θ̂.
  - The guards are scored against the truth.

**Step 0 gate (stop if it fails).**
- **Adequacy.** Each stage-1 scorer's M or N is sized so that the K = 100 shortlist contains a candidate within
  materiality of the reference best.
  - The required rate is ≥ 95% of all streams, pooled per decision.
  - Each instance gets 20 random streams. The reference best comes from a fixed random sample of 20,000 candidates
    per instance; D1 at J = 20 is exhaustive.
  - A stream is the scorer's own randomness (a fresh Sobol scramble, or a fresh simulator draw set) applied to the
    same 20,000 candidates. The engine is deterministic, so its 20 streams are identical. *(My interpretation;
    please confirm.)*
- **Simulator N** is the larger of two settings: the general adequacy rule, and correct ordering of adjacent price
  levels on D2.
- **KILL at J = 20** if the engine is < 3× faster in candidates/s than the best of the simulator and low-M GHK,
  each at its adequacy setting. Then stop and ask about a d = 40 timing probe.

**Primary: pipeline (b) vs accurate-only (c), scored on D4 and D1.**
- D2 and D3 are controls and are expected to tie. Only their time-to-optimum is reported.
- **PASS** if either holds:
  - (b)'s regret is lower than (c)'s by > 2 paired SE AND by ≥ materiality, at ≥ 3 of the 4 budgets. With 1,000 s
    run only where triggered, that means 3 of 3 for the base budgets.
  - (c) needs ≥ 5× (b)'s budget to match (b)'s regret (interpolated on log-budget), or never matches within the
    largest budget run.
- **KILL** if (b) vs (c) is a tie or a loss at every budget.
- **INCONCLUSIVE** if (b) wins at only 1–2 budgets, or the budget-to-match multiple is between 2× and 5×.
- **Single dataset.** The verdict must hold on both D1 and D4; otherwise INCONCLUSIVE, and then ask about H100
  refits.

**Attribution: the engine (b) vs the other stage-1 scorers (d) and (e), inside the same pipeline.** Pre-registered
labels:
- *Engine pipeline validated:* (b) beats each of (d) and (e) with ≥ 2× time-to-match.
- *Pipeline yes, engine not the reason:* (b) ties the best of (d) and (e).
- *Pipeline yes, other scorer better:* (b) loses to (d) or (e).

**Guards (record, don't kill).**
- **S0 truth.** One S0 dataset, one H100 refit (about $1.10) and HB-MNL on CPU. MVP's regret must be no worse than
  the incumbent's by more than materiality.
  - MVP's pick is the reference-best of the union at that fit's θ̂.
  - S0 search: (b) and (c) at 300 s, 1 seed each.
- **MVP vs HB-MNL + RFC (S1).** MVP's regret should be lower by > 2 SE and by a material amount on most of D1–D4.
  If it isn't, flag the contradiction with T5.

**Escalation to J = 40 / 60, only if either holds:**
- (c) fails to match (b) on D4 within the largest budget run; or
- the engine's speed ratio against the best competitor rises from d = 20 to d = 40, in one timing probe.
  - The model covers all 60 grid SKUs, so the probe can be built from the same fit.

## Risk stated up front (from T7 and T10; not measured here)

- **Engine vs GHK.**
  - The engine on a T4 GPU takes about 25 µs per orthant at d = 20 (T10, batch 10⁵).
  - GHK at d = 20 has not been timed on a GPU at the M values needed. A linear scaling of T13's Joint-GHK
    throughput (T4 GPU, d = 6, M = 2^14 × 4: 38,760 orthants in 37.5 s, about 1 ms each) gives about 200 µs at
    M = 4,096 and about 50 µs at M = 1,024.
  - If that holds, the engine is about 2–8× faster, not 100×. This is inferred, not measured. Step 0 measures it.
- **Engine vs simulator.**
  - The simulator costs N × d comparisons per candidate. That is about 8 × 10⁴ at N = 4,096, d = 20, which on a
    GPU is far below 6–12 engine orthants per candidate.
  - T7 found the simulator 9–16× faster at matched regret for reach.
  - Expected outcome: (d) beats (b) on throughput at J = 20. That changes only if matched decision error needs a
    very large N, which could happen with profit near-ties on shares around 5%.

## Step 0: timing probe (run first, then stop and report)

- **Hardware.** Throughput is measured on both a T4 and an H100, about 5 minutes on the H100 (about $0.35).
  - This sets the budgets and the H100's real speedup for these scorers.
  - The adequacy and reference passes are fixed work and run on whichever GPU is cheaper per unit of work.
  - If the H100's speedup for the stage-1 scorers is below 6.7× (its price ratio to a T4), the main grid costs
    more on the H100 than the $4 estimate. That is reported before launch.
- **Wiring.**
  - Regenerate the gate dataset; nll at θ̂ matches the fit to 1e-3.
  - `t13.shares()`-style orthant shares at θ̂ match the simulator at N = 2^20 within MC error, on 20 candidates
    from each decision.
- **Throughput.** Per candidate (not per orthant), at batch sizes {10³, 10⁴, 10⁵} and d ∈ {6, 10, 14, 20, 26}:
  - Engine.
  - GHK with M ∈ {256, 1,024, 4,096, 16,384}.
  - Simulator with N ∈ {2^12, …, 2^18}.
- **Adequacy (sets the M and N values).**
  - Per instance: the 20,000-candidate sample, plus D1 at J = 20 exhaustive, scored by the high-M reference.
  - Each scorer setting is run for 20 streams. M and N are stepped up until the adequacy rule holds.
  - The D2 adjacent-level ordering check is applied to the simulator.
  - M_acc for (c) and stage 2 comes from the same reference comparison.
- **Engine speed gate.** Candidates/s for the engine vs the better of the simulator and low-M GHK, each at its
  adequacy setting. If the ratio is < 3×: KILL, then stop and ask about the d = 40 probe.
- **Projection.** Candidates per budget per arm, and the projected cost of the full J = 20 run.
- **Cost (revised).** The adequacy rule is much larger than the original matched-error sample:
  - The reference alone is 12 × 20,000 candidates plus D1 exhaustive, about 2–3 million orthants at high M.
  - Estimate: 3–5 T4-hours, **about $2–3** (was $0.20).
  - If the projection after the reference pass exceeds $5, stop and ask.

## Cost (J = 20, before Step 0; revised 2026-10-16)

| Item | GPU | Estimate |
|---|---|---|
| Step 0: adequacy, reference, speed gate, T4 + H100 throughput | T4 or H100 | about $2.50–3.50 |
| Main grid: 12 instances × 4 arms × 5 seeds, three budgets, budget as a cap | H100 | **about $4**, if the 12.5× speedup holds |
| Fourth budget, arm (c) only, triggered cells, resumed | H100 | at most about $4 |
| S0 refit (MVP-G 512 × 256) | H100 | about $1.10 |
| S0 search: (b) and (c), largest budget, 1 seed | either | under $1 |
| HB-MNL (S0 and S1) and RFC search | CPU | under $1.50 |
| **Total** | | **about $10–15** |

- **Why GPU-hours are runs × budget.** Runs = instances × arms × seeds = 240. Each run lasts at most its largest
  budget, which already gives the smaller budgets.
- **Why budgets shrink on the H100.** The same candidate counts take less time. Per-candidate cost relative to a
  T4 is 6.7 / speedup.
- J = 40 / 60 come only through the escalation rule, after your review.

## Scope

**IN**
- T4's J = 20 gate fit; decisions D1–D4 at 3 instances each, with the cost/margin objective.
- Arms (a)–(e); the high-M GHK reference on the union of shortlists; truth scoring at θ_true.
- The S0 guard: one S0 dataset, one H100 refit, HB-MNL.
- The metrics listed above; Step 0 timing probe; J = 20 only.

**OUT**
- Engine accuracy sweeps and engine settings other than the service default.
- Engine bias fix; new estimators; Q2.
- J = 40 and 60 until you review J = 20 (stop and ask).
- Parameter uncertainty or joint survival (T11 and T13 killed it); gradient or relaxation search.
- Real data; new truths beyond S1 and the one S0 guard dataset.
- Task 2 (HB-MVP); per-respondent decisions; TURF.
- OVUS as a stage-2 variant.
- Replication over S1 datasets: only if the verdict is INCONCLUSIVE, and only after asking.

## Resolved (user, 2026-10-16)

- Arm (e), low-M GHK stage 1: added.
- S0 guard: added (one refit).
- Adequacy: 20 streams per instance, a 20,000-candidate reference sample per instance, ≥ 95% pooled per decision.
- Simulator N: the larger of the adequacy and D2-ordering rules.
- Budget-to-match: interpolated on log-budget.
- Seeds: 5. Budgets {30, 100, 300} s; 1,000 s only where (c) hasn't matched (b), with cost reported first.
- Primary and attribution scored against the reference; guards against the truth.
- INCONCLUSIVE: (b) wins at 1–2 budgets, or a budget-to-match multiple between 2× and 5×.

## Still open (defaults apply unless you change them)

- **Decisions and costs** as proposed: D1–D4; pack costs $2.60 / $3.00 / $3.40; fixed cost F; D2's 9-level grid.
- **Budget split:** f = 0.8, K = 100.
- **Incumbent search:** ASM-style on CPU, at the largest budget.
- **Streams:** "stream = scorer randomness on the fixed 20,000 sample" (Kill criteria, Step 0 gate).

## Stage-1 screening settings (user, 2026-10-16)

- **How they are chosen.** Per run and per hardware (CPU or GPU), with a logged trace of the problem and hardware
  conditions.
- **Flags.** A run whose conditions differ from those the settings were tuned under is flagged.
- **Status.** The rules are provisional, to be resolved or tuned later.
- **Where.** Details are in private notes, not in this repository.
