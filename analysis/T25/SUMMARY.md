# T25 · Summary (2026-10-20)

Design and every deviation are in TEST.md. Full tables are in out/summary_tables.md, one row per run x checkpoint
in out/results.csv, and anytime curves in out/anytime.png. Two seeds per cell, so ranges are given, not CIs. All
regrets are against the best solution found by any arm, scored under the truth.

**Version note.** A first full run (v1, kept in out/v1/) was invalid. Per-cluster segment utilities were shrunk
toward 0, so SKUs never stocked in a cluster looked about 60x more attractive than real SKUs. "New to this
retailer" lines were then built from them, with client shares of 43-62%. v2 shrinks toward a pooled all-store
estimate plus a segment level; never-stocked SKUs now sit in the identified range. Model shares on each cluster's
stocked shelf (category 27-31%, client 11-12%) match the data (25%, 8-12%).

## Setup in one paragraph

- Client: manufacturer 103 (17 SKUs). Market: dunnhumby soft drinks, top 60 SKUs.
- Segment utilities: LC-MNL (K = 10) per store cluster.
- **Estimated truth**: MNP with Σ fitted by likelihood on store-month assortment variation. **Σ = I in all 7
  clusters** (likelihood ratio vs identity 0): beyond the segments, the data show no error correlation.
- **Stress truth**: assumed ρ = 0.5 near-duplicates, 0.25 same manufacturer.
- Decision 1: per-cluster line of 5 (top 2 forced), k = 2 clusters, 455 lines each, enumerated exactly.
- Decision 2: build at most 10 SKUs; each of 5 clusters gets a line of up to 6; one coupled GA.
- Truth scoring: engine at resolution "highest", cross-checked by GHK with 20,000 draws. Max |diff| over each problem's top 20 is
  0.033 $/1k trips, and GHK picks the same best solution every time.

## Decision 2 (the main test): regret, % of best-known (mean [min, max])

Best-known: 61.14 $/1k trips (estimated truth), 58.42 (stress truth).

| arm | est, 60 s | est, 300 s | stress, 60 s | stress, 300 s |
|---|---|---|---|---|
| A logit, CPU | 0.96 [0.91, 1.02] | 0.91 [0.91, 0.91] | 2.88 [2.29, 3.47] | 2.32 [2.29, 2.34] |
| B MNP GHK, CPU 8 cores | 2.08 [1.37, 2.79] | 0.35 [0.00, 0.70] | 4.50 [3.56, 5.45] | 1.52 [1.43, 1.61] |
| C MNP engine, T4 | **0.35** [0.00, 0.70] | **0.35** [0.00, 0.70] | 1.87 [1.86, 1.88] | **0.75** [0.00, 1.50] |
| D RFC, T4 | 2.52 [1.90, 3.14] | (dropped for the cap) | 4.16 [3.37, 4.95] | (dropped) |
| E RFC, CPU 8 cores | 2.07 [1.90, 2.25] | 2.49 [1.90, 3.09] | 4.07 [3.37, 4.77] | 3.60 [3.37, 3.84] |
| F MNP simulator, T4 | **0.35** [0.00, 0.70] | **0.35** [0.00, 0.70] | **1.45** [1.45, 1.45] | 1.45 [1.45, 1.45] |

In $ per 1,000 trips: 1% of best-known is about 0.6 $ (est) or 0.58 $ (stress). The largest gap in the table
(B vs F, stress, 60 s) is about 1.8 $/1k trips.

Anytime (300 s runs, mean regret %):

| truth | arm | 5 s | 15 s | 30 s | 60 s | 120 s | 300 s |
|---|---|---|---|---|---|---|---|
| est | A | 2.25 | 1.31 | 0.97 | 0.97 | 0.97 | 0.91 |
| est | B | 2.56 | 2.56 | 2.56 | 2.08 | 1.25 | 0.35 |
| est | C | 2.56 | 1.90 | 0.98 | 0.35 | 0.35 | 0.35 |
| est | F | 2.13 | 1.17 | **0.35** | 0.35 | 0.35 | 0.35 |
| stress | A | 4.68 | 3.26 | 2.88 | 2.88 | 2.88 | 2.32 |
| stress | B | 4.50 | 4.50 | 4.50 | 4.50 | 3.11 | 1.52 |
| stress | C | 4.50 | 4.50 | 2.60 | 1.87 | 1.43 | **0.75** |
| stress | F | 4.50 | 3.92 | **1.84** | **1.45** | 1.45 | 1.45 |

### The questions

- **C vs B (engine vs the strongest CPU probit at equal time): the engine wins early, B catches up late.**
  - At 60 s, C has 0.35% regret vs B's 2.08% (est) and 1.87% vs 4.50% (stress). B completes about 11k shelf
    evaluations and 45-54 generations in 60 s; C completes about 18k and about 600.
  - By 300 s B ties C on the est truth (0.35%). On stress it is still behind (1.52% vs 0.75%).
- **A vs C (what a practitioner leaves on the table): about 0.6% on the est truth, 1.6% on stress, at 300 s.**
  - In $/1k trips: 0.56 vs 0.22 (est) and 1.35 vs 0.44 (stress).
  - Even when the truth has Σ = I, logit (Gumbel errors) and probit (normal errors) rank lines differently.
  - A converges fastest in the first 15 s, then stops improving under the truth: it is optimizing the wrong
    model.
- **C vs D (the engine vs just using a GPU): the GPU alone doesn't help.**
  - D (RFC on the same T4) is at 2.52% / 4.16% regret at 60 s, vs C's 0.35% / 1.87%.
  - E (the same RFC on 8 CPU cores) lands in the same place as D: the RFC model is the limit, not the hardware.
- **C vs F (engine vs shared-draw MNP simulator on the same GPU): F converges faster, C ends at least as good.**
  - Est truth: F reaches 0.35% by 30 s, C by 60 s; they are identical after.
  - Stress truth: F leads through 60 s (1.45% vs 1.87%), then stalls at 1.45% for both seeds. C keeps improving
    to 0.75% (one seed 0%, one 1.50%).
  - F's 4,096 simulated shoppers per segment can't separate solutions closer than its sampling noise; the
    engine has no sampling noise.
- **Do A and C pick different lines?**
  - Yes, in every 300 s run. Seed 1: A's build set differs from C's and every cluster's line differs. Seed 2:
    the same build, with a different line in cluster 4.
  - Est truth, seed 1: client share on A's lines is up to 0.8 points lower (clusters 1, 2) and about equal
    elsewhere. Its cannibalization is higher in clusters 0 and 2.
  - Under the stress truth, A has higher cannibalization in every cluster: 15-31% vs 13-25% on C's lines. Logit
    keeps more near-duplicate client SKUs on the same shelf, because IIA doesn't see them stealing from each
    other.
  - SKU-level line differences are in out/summary_tables.md.

### Client lines (best-known, stress truth)

- Every cluster gets the full 6. Client share is 5.0-6.6% per cluster; mean cannibalization is 13-25%.
- Under the est truth: client share 5.3-6.9%, cannibalization 6-24%.

## Decision 1 (model-regret check, 455 lines per cluster, exact optimum)

- Every arm scores the whole space within about 1 s, so this isolates the model.
- **Cluster 0**: every arm finds the optimum under both truths.
- **Cluster 1, est truth**: all arms optimal except F, where one seed's noisy estimate preferred a line 0.54%
  (0.17 $/1k trips) worse.
- **Cluster 1, stress truth**: A, D and E (the logit-utility arms) pick a line 1.49% (0.45 $/1k trips) worse in
  4 of 6 runs. Two RFC runs, D seed 1 and E seed 1, hit the optimum. B, C and F find it in every run.

## Speed and cost

- **Evaluations in 60 s on decision 2**: A 21k, D 22k, E 21k, F 19k, C 18k, B 11k. Generations: A 1,938, F 1,120,
  D 1,162, E 825, C 624, B 48. B's few generations are why it trails early.
- **Scale**: these shelves have 17-25 competitors plus a line of up to 6, so orthants of dimension 18-31.
- **GPU**: 152.2 min in total under the raised 160-min cap. That is 79.3 for v1 (invalid, see the version note)
  and the smokes, and about 73 for v2. Total spend $2.44 including CPU containers.
  - Truth scoring ran on an 8-core CPU container (engine library at "highest", numpy) to stay off the GPU cap.

## Caveats

- **Two seeds.** Ranges overlap in several cells (C's stress 300 s range is 0-1.5%). No CIs or Holm tests are
  possible at n = 2; read the differences as directions.
- **Materiality.** All regrets are under 3 $ per 1,000 trips, roughly 5% of the objective at worst. Whether
  that is material is the user's call.
- **The estimated truth is Σ = I.** The probit-vs-logit difference under it is the error distribution only. The
  stress truth is assumed, not estimated.
- **Assumptions:**
  - Margins: 0.30 x price for national brands, 0.40 x price for private label.
  - Stocked: >= 25% of store-months, not the agreed 50% (sales-based availability; see TEST.md).
  - Availability = sold that month.
  - Mutation rate: about 2 swaps per child, not Sawtooth's 0.5 per gene.
  - D's 300 s runs and decision 1's 300 s runs were dropped to fit the GPU cap.
- **RFC tuning**: the attribute-error sd sits at the grid edge (1.0) for the 3-store cluster.
