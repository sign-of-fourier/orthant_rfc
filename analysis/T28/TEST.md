# T28 · TURF reach line at scale: stepwise vs exhaustive search

Design agreed with the user 2026-10-22. Use case A of the incumbent-wall list: past the exhaustive limit TURF
software switches to stepwise, and early stepwise picks may corner the search.

- **Question.** At a fixed budget, how much reach does stepwise leave on the table, and how large a J × m can we
  search exhaustively (GPU) where the incumbent cannot?
- **Data.** dunnhumby Complete Journey, SOFT DRINKS (fixed before any run: the most products at ≥ 1% window-1
  household rate, 183). Top J by window-1 household rate, J ∈ {30, 50, 70, 100}, nested; window-1 rates 22.2% down
  to 1.7%. The 2,338 households active in both windows (as T6). Acceptable set = bought the item at least once in
  the window. Search on window 1 (weeks 1–27); window 2 (weeks 28–53) is the holdout.
- **Arms.** Forward stepwise (strict); stepwise + best-improvement 1-swap (strong incumbent); max-coverage ILP
  (SciPy `milp` / HiGHS, the proven optimum, also the threat check); GPU exhaustive / wide search (ours, built
  only if Gate 0 passes).
- **Grid.** m ∈ {4, 6, 8, 10, 12, 15} × the four J, 24 cells.
- **Gate 0 (kill).** At J = 100, m = 15: stop if the ILP proves the optimum in under a minute AND stepwise + swap is
  within 0.5 pts of it.
- **Pass rules (not reached).** Counted gap = holdout reach difference ≥ max(0.5 pts, 2 SE); reach claim needs
  counted gaps vs strict stepwise in ≥ 3 cells and no counted loss; speed claim needs our largest J × m in an
  hour to beat both stepwise + swap and the ILP.

## Result (2026-10-22): KILL at Gate 0

`t28.py gate` / `gate --full`, `out/gate.json`, `out/grid_cpu.json`. CPU only, $0.

- **Gate cell, J = 100, m = 15.** The ILP proves the optimum (reach 63.26%) in 1.2 s; stepwise and stepwise + swap
  are both 0.04 pts short. On the holdout the optimum is 0.30 pts *worse* than stepwise.
- **Whole grid.** The ILP proves the optimum in every cell in 0.05–1.1 s (997–1,533 unique acceptable-set rows).
  In-sample gap to the optimum: strict stepwise 0.00 in 18 of 24 cells, 0.13 pts at m = 6 (all J), 0.04 at m = 15
  (J ≥ 50); stepwise + swap 0.00 except 0.04 at m = 15. Largest holdout gap 0.73 pts (m = 6, ~15 households of
  2,338, under 2 SE at this sample size); at m = 15 it goes either way (+0.56 at J = 30, −0.30 at J ≥ 70).
- **J doesn't matter.** The optimal lines are identical for J = 30–100 at every m except m = 15: the reach-optimal
  line lives in the top ~30 items, and the long tail (rates 2–5%) never enters. The search space grows 10⁹-fold
  from J = 30 to 100 and the answer does not change.

## Reading

- **Reach.** No reach claim: stepwise is within 0.13 pts of the proven optimum everywhere, and swap closes that.
  Reach is a coverage objective, where greedy is near-optimal in practice; the holdout gaps are household noise.
- **Speed.** No speed claim: the incumbent's wall is the software, not the problem. An off-the-shelf ILP solves
  max-coverage at J = 100, m = 15 exactly on one CPU core in about a second, so GPU exhaustive search would be
  slower than an exact method that already exists.
- **Not tested.** Larger panels (Instacart, ~200k users) make the ILP bigger, but greedy's near-optimality and the
  short-tail effect would not change. Weighted / frequency TURF objectives are still coverage-like. Neither was
  worth running.
