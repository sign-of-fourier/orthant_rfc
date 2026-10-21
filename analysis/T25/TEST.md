# T25 · Brand-side line and build-and-place search: same GA, six share evaluators, fixed wall-clock budgets

Status 2026-10-20: done (v2). Results in SUMMARY.md. v1 (invalid: never-stocked SKUs had ridge-to-0 utilities) is in out/v1/. GPU cap raised by the user from 90 to 160 min for the v2 rerun.

## Question

How much client margin is lost when the same genetic algorithm (GA) runs with different share evaluators under
the same wall-clock budget? Evaluation speed is separated from the search algorithm by holding the GA fixed.

## Data (dunnhumby Complete Journey, R package release)

- Full `get_transactions()` table: 1,469,307 rows, 2,469 households, 53 weeks (not the 75k-row sample). The
  102-week Kaggle CSV release is not on the box.
- Category: SOFT DRINKS, the **top 60 SKUs by purchase occasions** (user 2026-10-20). This replaces the
  ">= 30 occasions" rule, which leaves 356 SKUs. Coffee was the first choice, but no store has >= 200 coffee
  occasions in this release.
- Columns: COMMODITY_DESC = product_category, SUB_COMMODITY_DESC = product_type, BRAND = brand,
  CURR_SIZE_OF_PRODUCT = package_size, MANUFACTURER = manufacturer_id.
- SKU attributes:
  - Sub-commodity (5 levels) and manufacturer (4: 103 has 17 SKUs, 1208 has 15, 2224 has 10, 69 has 18).
    Private label is exactly manufacturer 69.
  - Size band (4 levels, from the estimated volume per unit; nearly the same as sub-commodity).
  - Average unit price.
- Near-duplicate pairs (same manufacturer + same sub-commodity + same or adjacent size band): **146 of 1,770**.
  Diet/regular pairs can't be identified: this release has no flavor or description field.
- Choice occasions:
  - A basket with >= 1 of the 60 SKUs is one occasion; the highest-spend line is the choice.
  - Outside option: baskets at the same store and week with no soft-drink purchase, subsampled to 3x.
  - Availability: an SKU is available at a store in a month if it sold there that month.
- Store clusters: k-means on stores' sales-mix shares, stores with >= 200 category occasions (65 stores).
  - k = 2 (decision 1): 12 and 53 stores.
  - k = 5 (decision 2): 3, 17, 10, 14 and 21 stores. The smallest has 851 category occasions, so the threshold
    did not need lowering to 100.
- **Stocked in a cluster**: the SKU is available in >= 25% of the cluster's store-months. The agreed rule was
  50%, but sales-based availability misses slow movers, and 50% leaves only 7-10 of 60 SKUs per cluster: a
  client line of 2-3 with 2 forced, and 3-6 competitors.

## Models

- **LC-MNL** (arm A's model; the utilities of every model):
  - K = 10 household-level segments per cluster; V_kj = a_kj - b_k price_j, with b_k >= 0, ridge 0.01 and the
    outside option at 0.
  - EM with 5 starts; segments under 3% weight are merged into the nearest segment.
  - Ridge 0.01 toward a pooled all-store MNL plus a free segment level (v2). v1 shrank toward 0, which left
    never-stocked SKUs at the outside option's level.
  - K = 10 vs 20 on 20% held-out households: holdout log-likelihood favours 20 by 0.013-0.026 per occasion,
    BIC favours 10 by 2.1k-4.5k, and K = 20 leaves 7 segments under 3%. Not clear support, so K = 10.
- **Truths** (MNP; all arms scored against them):
  - Utilities: mu_k = V_k x sqrt(6) / pi (logit scale to unit error variance).
  - Errors: e ~ N(0, Sigma), shared across segments; the outside option has its own uncorrelated unit error.
  - **est (estimated)**:
    - Sigma = unit-diagonal rescale of I + w_nd ND + w_mfr S_mfr + w_sub S_sub, clipped to positive definite.
    - The weights are fitted by household-panel probit likelihood on the real occasions, with the LC utilities
      held fixed. Identification comes from store-month assortment variation. GHK with Genz ordering and
      R = 100 fixed antithetic draws; Nelder-Mead.
    - The earlier switching-rate moment match was flat at the identity (user 2026-10-20: replaced).
    - The private-label and size-band factors were dropped as collinear with manufacturer and sub-commodity.
  - **stress (assumed)**: rho = 0.5 within near-duplicate pairs and 0.25 within manufacturer, otherwise 0.
- Margin: 0.30 x price for national brands, 0.40 x price for private label (assumption).

## Decisions

- Client = manufacturer 103 (17 of the 60 SKUs; user's choice over 69, which is the retailer's private label).
  Must-keep = its top 2 by occasions.
- Each cluster is a retailer. The competitors it stocks stay on its shelf and are not decision variables.
- Objective = client margin per 1,000 category trips, summed over clusters by trip share:
  sum_c tau_c x 1000 x sum_k w_ck x sum_{j in line_c} P_ck(j | line_c + competitors_c + outside) x margin_j.
  Under MNP, P is an orthant probability of dimension |line| + |competitors|.
- **Decision 1 (per-cluster line, k = 2)**:
  - Line size = half of the client SKUs stocked in the cluster (rounded); the top 2 are always in the line;
    unstocked client SKUs are allowed ("new to this retailer").
  - The space is tiny (C(15, 3) = 455 lines per cluster), so the optimum is found exactly by enumeration
    under the truth.
  - Reported: chosen line, regret and time to the optimum per arm (the model-regret check).
- **Decision 2 (build and place, k = 5; the main test)**:
  - The client produces at most 10 of its 17 SKUs, including the top 2. Each cluster gets a line of 1-6 of
    the produced SKUs.
  - The clusters are coupled through the build set, so one GA covers everything.
  - Best-known = max over the top 50 distinct solutions of each arm's final populations plus every checkpoint
    solution, rescored under the truth.

## GA (identical in all arms)

- **Encoding**: a boolean genome (1 + C) x 17: the build mask plus one line mask per cluster. Decision 1 has
  C = 1, a full build and a fixed line size.
- **Repair** (after crossover and after mutation):
  - Put the must-keep SKUs in, cap the build, keep each line inside the build, and enforce the line-size
    bounds.
  - Removals are random among non-must SKUs.
  - Additions are the highest-ranked SKU by the arm's single segment-weighted pass: each SKU's share against
    the outside option alone.
- **Population and selection**: 300; 150 offspring per generation; tournament of 2; (mu + lambda) elitist
  survival; no early stopping.
- **Crossover**: per row (build and each line), uniform on the union of the parents' SKUs, with the size taken
  from one parent at random.
- **Mutation**: each occupied slot is swapped with probability 2 / slots, about 2 swaps per child. Sawtooth's
  per-gene default of 0.5 is meant for attribute vectors of a few genes and is near-random search here
  (deviation).
  - Decision 2 also resizes each line by +-1 with probability 1 / C (line sizes are free there).
- **Initialization**, identical across arms (built from the LC-MNL, not from any arm's evaluator):
  - 150 targeted genomes: greedy add-one on a random (cluster, segment)'s logit utilities.
  - 150 random feasible genomes.
- **Evaluations**: shelf values are cached per (cluster, line); "evaluations" counts (cluster, line) shelves
  scored.
- **Timing**: evaluator setup (process pool, CUDA context, draws, one warm-up call) is outside the clock. The
  clock covers the ranking pass, the initial population and every generation. The run stops at the first
  scoring-chunk boundary past the budget.
- **Seeds**: 1 and 2, the same for every arm. The user cut seeds from 3 to 2 so both truths fit the GPU cap.

## Arms (only the evaluator differs)

| arm | evaluator | hardware |
|---|---|---|
| A | LC-MNL share of preference, closed form | Modal CPU, 8 cores |
| B | MNP truth, GHK, R = 500 fixed antithetic draws, Genz ordering, numpy process pool | Modal CPU, 8 cores |
| C | MNP truth, GPU orthant engine at its default resolution ("high", float64) | Modal T4 |
| D | RFC on LC-MNL utilities: Gumbel product error + normal attribute-level error, 1,000 draws per segment | Modal T4 |
| E | the same RFC in numpy (user 2026-10-20) | Modal CPU, 8 cores |
| F | MNP truth, shared-draw first-choice simulator: 4,096 people per segment drawn once, argmax per shelf (user 2026-10-20; T7's simulator idea, the probit counterpart of D) | Modal T4 |

- CPU arms run on 8-core Modal containers because the dev box has 2 cores.
- RFC attribute-error sd is tuned by the real occasions' log-likelihood over {0, 0.1, ..., 1}, not on the truth.
- A, D and E never see the truth, so they run once and are scored under both truths. B, C and F run per truth.
- SciPy `multivariate_normal.cdf` is timed on up to 100 decision-2 orthants for reference only.

## Budgets (user 2026-10-20, to fit the GPU cap with arm F)

- Decision 1 runs at 60 s only: in the 15 s smoke, every arm scored all 455 lines.
- Decision 2 runs at 60 s and 300 s, except D, whose 300 s runs are dropped (the spec's first cut).

## Scoring

- Each (cluster, line) shelf is scored under the truth by the engine at resolution "highest", in the
  engine library's numpy build on an 8-core Modal CPU container. It is the same rule as the GPU build; moving it
  to CPU keeps scoring off the GPU cap.
- Cross-check: GHK with Genz ordering and R = 20,000 fixed antithetic draws on every checkpoint shelf and on the
  top 20 solutions. Report max |diff|.
- Per problem, truth, arm and budget, mean and range over seeds of:
  - true objective;
  - regret in % of best-known;
  - regret in $ per 1,000 trips;
  - evaluations and generations completed;
  - GPU seconds and $.

## Caps

- GPU: <= 90 min and <= $15 in total, enforced before every launch from measured plus reserved T4 seconds.
  Reserve per GPU container = its budgets + overhead measured in the smoke; scoring is reserved per call.
- If the cap would be exceeded, D's 300 s runs are dropped first. Each GPU container's Modal timeout bounds it.
- Ledger: out/gpu_ledger.json.

## Code

- `t25.py` (truths, bundles) and `summary.py` are here. The engine-dependent scripts (the evaluators, the Modal
  runners and the truth scoring) are kept out of the public repo.
