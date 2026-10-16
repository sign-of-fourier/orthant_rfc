# T13 · Cardinality-constrained line with joint survival, 6 of 20 (KILL 2026-10-11)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

- **Model (fixed).** T4 recovery-gate fit: J = 20 shelf, MVP-G 512×256, N = 1,000, S1 truth dataset
  (`analysis/T4/out/gate_g/fit_512x256.json`: θ (25) and sandwich covariance). No new fits. Caveat: pack0 is
  ~−3.4 SE in every T4 fit (likely this dataset).
- **Decision.** Exactly 6 of 20 SKUs, C(20,6) = 38,760 lines; prices fixed at T4's base prices; cost $3.00
  (T4). Shelf = the 6 kept SKUs + "none" (no other alternatives). Shares are exact population orthants
  (normal tastes and errors), 7 per line, d = 6. Log-share delta method: Σ_log = J V Jᵀ, J = ∂log ŝ/∂θ.
- **Proposed t (from the fitted model, engine on CPU, no cost).** Unconstrained optimum: SKUs 3, 4, 6, 11, 15, 18,
  shares 6.5 / 7.6 / 8.7 / 4.7 / 16.6 / 7.9%, none 48%; log-share SDs 0.044–0.075; pair correlations −0.28 to 0.50.
  "Within reach" = 1% < P(miss) < 99%:

  | t | P(miss) per kept SKU | in reach | certain miss |
  |---|---|---|---|
  | 6.5% | .47 / .01 / 0 / 1.0 / 0 / 0 | 2 | 1 |
  | 7.0% | .83 / .11 / 0 / 1.0 / 0 / .03 | 3 | 1 |
  | 7.5% | .97 / .40 / .03 / 1.0 / 0 / .20 | 4 | 1 |

  The 4.7% SKU misses for certain at every t with 2+ in reach (SDs are tight at N = 1,000), so the unconstrained
  optimum is infeasible at every α; the constrained optimum is a different line.
- **Proposed floors for (2):** 90%, 95%, 98% of the unconstrained max expected profit (top lines are within
  ~2% of each other, so 98% is tight).
- **Arms.** Point, Independent, Joint-OVUS, Joint-GHK (reference integrator, scrambled Sobol 2^14 × 4),
  Joint-engine, Nested MC (θ ~ N(θ̂, V), shares per draw by GHK with fixed draws). Cells: (1) 3 t × 3 α = 9,
  (2) 3 t × 3 floors = 9.
- **Search.** Exhaustive (reference). Greedy: (1) add the SKU with the highest expected profit whose partial
  line meets the constraint, else the highest P; (2) profit-greedy to 6. Then best-improvement 1-swap
  (6 × 14 = 84 neighbours per step) on the cell's objective among feasible neighbours ((1): if infeasible,
  ascend P first). Report whether each reaches the exhaustive optimum, per arm.
- **Step 0 (run first, stop and report).** Delta step for all 38,760 lines, Independent and Joint (all three
  integrators) for all lines and t. Metrics as specified: kept SKUs with P(miss) > 1% at each arm's optimum and
  their shared brand / flavor / pack; Joint vs Independent feasibility disagreement under (1); top-20 rank
  changes under (2); pairwise excess Σ_{i<j}[P(miss_i ∩ miss_j) − p_i p_j] at the optimum by sign of the pair
  correlation (exact bivariate normal on log shares).
  - Cost: ~10–15 T4-min (delta step ~7 min, Joint-GHK on 116k orthants ~2 min) ≈ $0.15.
- **After Step 0.** Nested MC on all lines, B = 10,000 (MC SE 0.001 at 1 − α = 0.99, 0.002 at 0.95), GHK M = 256
  per draw. Estimate from T11's throughput: 2.5–5 T4-h ≈ $1.50–3.00, split across T4s; a 1-line-batch timing
  probe (~$0.05) first. If the probe projects > $5: stop and ask (fallback: Nested MC on the top 2,000 lines
  per cell). Local search timed per arm, with orthant batch sizes (exhaustive 38,760 × 3 t; greedy 20 − k;
  swap 84). True shares (T4 S1 truth) of each chosen line: reported, not scored.
- **Kill criteria (approved 2026-10-11).**
  - Step 0 stop: in every (1) cell Joint and Independent disagree on feasibility for < 0.5% of lines and pick
    the same line, AND in every (2) cell the top 20 lines have ≤ 1 rank change → STOP (report, no Nested MC).
  - H1 (decisions differ): Joint-GHK's line ≠ Independent's in ≥ 2 of the 9 cells of (1) or ≥ 2 of the 9 of (2).
  - H2 (backed by Nested MC): in the cells where they differ, Nested MC agrees with Joint in a majority;
    under (1) Joint's line has P_NMC ≥ 1 − α − 2 MC SE; under (2) Joint's P_NMC exceeds Independent's by
    > 2 MC SE.
  - PASS = H1 and H2. KILL = Joint = Independent in all 18 cells. Otherwise INCONCLUSIVE (one dataset, no
    replication).
  - Integrators: report agreement of Joint-engine and Joint-OVUS with Joint-GHK; flag any cell where an
    integrator alone changes the decision.
  - Speed and search: report only (T11: the time saving vs Nested MC is the delta step).
- **Out of scope.** Prices as decisions, new fits, other J, gradients, P(π ≥ L), replication over datasets.
- **Approved as drafted (user, 2026-10-11).** Code: `analysis/T13/t13.py`.
- **Step 0 (2026-10-11): one T4, 345 s wall (~$0.06).** `out/step0.npz`, `out/step0_report.json`. Delta step 209 s
  for all 38,760 lines (51 share evaluations each, GHK M = 1,024; checked against M = 2^15: ≤ 1.7e-4 abs). Orthant
  time per t over all lines: Independent 0.007 s, Joint-engine 0.08 s, Joint-OVUS 0.03 s, Joint-GHK 37.5 s.
  - Unconstrained optimum on GHK shares: SKUs 3, 4, 6, 11, 15, 18 (as sized; the CPU engine shares used for sizing
    were off by up to 0.009 abs, 17% rel on a small share, without changing the optimum).
  - **(1): Joint and Independent disagree on feasibility for 0 of 38,760 lines in all 9 cells (all three
    integrators), and pick the same line in all 9.** The constraint is near-deterministic: only 0–6 lines per
    cell have P within ±0.01 of 1 − α (log-share SDs 0.04–0.08 at N = 1,000). Joint and Independent differ by up
    to 0.12 on single lines, but only on lines far from the boundary (low P). t = 7.5%, α = 1%: no feasible line.
  - **(2): the chosen line is the same for Joint and Independent in all 9 cells.** Top-20 ranks do change at
    floors 90% and 95% (7–18 positions, 0–4 lines swapped into the top 20); at 98% only 4 lines qualify, no change.
  - At the optima: Joint/Independent optima keep 0–2 SKUs with P(miss) > 1% under (1) and 3–4 under (2), sharing
    pack 2 at most. Pairwise excess at the (2) optimum (t = 7%): +0.014 (positive pairs), −0.071 (negative pairs):
    negative pairs dominate, so Joint < Independent there (0.257 vs 0.300), without moving the argmax.
  - **Against the criteria.** Step 0 stop rule as written is not met (top-20 changes at 90% / 95% floors), but
    H1 is 0 of 18 cells: Joint = Independent on every decision, i.e. the KILL condition, which Nested MC cannot
    change. Stopped and asked whether to run Nested MC (~$1.50–3) for the reference agreement anyway.
  - **User (2026-10-11): stop, KILL.** No Nested MC. Total T13 GPU ~$0.06.
