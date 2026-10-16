# T5 · N-sweep (static sample savings)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

- **Hypothesis.** Under correlated-error truths, MVP reaches the decision error that HB-MNL +
  RFC achieves at large N with materially smaller N. Under a logit truth, MVP is not materially
  worse at small N.
- **Kill criteria (user, 2026-10-09).**
  - "Materially smaller N": N_match ≤ 720, i.e. at least 40% fewer completes than RFC-S at
    N = 1,200. Anything under about 25% is hard to market. (On the grid this means N_match ≤ 600.)
  - "Matches": MVP's mean decision flips (or cost regret) within one replicate-SE of RFC-S's at
    N = 1,200.
  - G0 guard: at every N, MVP's regret is no worse than RFC-S's by more than one SE. If it fails
    only at N = 150, record the failure; don't kill on it.
- **Scope.**
  - In: T1's DGP and arms ([design](../../analysis/T1/DESIGN.md)): G0 (logit control), G1; G2 only if
    cheap. Methods: HB-MNL + RFC-S and MVP (GPU GHK). N ∈ {150, 300, 600, 1,200}, 20 replicates
    per cell.
  - OUT: adaptive or sequential fielding ([backlog](../../BACKLOG.md)), SEs, new truths, J scaling.
- **Step 0 (before T5 runs).** `analysis/T1/run.py` now fits MVP with the total error variance
  fixed (`--scale total`, default; the 2026-10-08 fix from `mvp_refit.py`); `--scale snu` keeps the
  σ_ν-pinned path for exact T1 reproduction. Rerun one T1 cell (G1, 3 replicates) with the fix and
  confirm it reproduces the 2026-10-08 refit results (`out/full_G1/results_mvp_total.json`):
  σ, S1, S2, decisions.
  - *Done 2026-10-09, PASS.* `run.py --truth G1 --reps 3 --out-name step0_G1`: MVP σ, S1, S2,
    log-likelihood, S4 and all decision fields identical to `results_mvp_total.json` (max |Δ| ≤ 1e-16);
    RFC-S arm identical to the original `full_G1/results.json` up to float rounding (~1e-15). 3 fits
    on T4 in 168 s wall; ~11 min total; GPU cost ~$0.10.
- **Tests (minimal).** For each truth × N × replicate: simulate, fit both methods, score the T1
  decisions (price change, line extension, cost-weighted metric) against the truth.
  - Metrics: decision flips and cost regret against the truth.
  - Result: error-vs-N curves per method, plus N_match = the smallest N at which MVP matches
    RFC-S's error at N = 1,200.
- **Results (2026-10-09).** Full tables: [RESULTS.md](../../analysis/T5/RESULTS.md). PASS on the kill rules
  (N_match 150 ≤ 720 for G1 and G2, both metrics); G0 guard fails only at N = 150 (recorded, not a
  kill). ~$3.40 GPU.

| truth | N | MVP flips (price / ext / cost) | MVP cost regret % | RFC-S flips | RFC-S cost regret % |
|---|---|---|---|---|---|
| G0 | 150 | 1 / 0 / 1 | 0.37 | 3 / 0 / 0 | 0.00 |
| G0 | 300 | 1 / 0 / 0 | 0.00 | 2 / 0 / 0 | 0.00 |
| G0 | 600 | 0 / 0 / 0 | 0.00 | 0 / 0 / 0 | 0.00 |
| G0 | 1,200 | 0 / 0 / 0 | 0.00 | 0 / 0 / 0 | 0.00 |
| G1 | 150 | 0 / 0 / 2 | 0.17 | 3 / 6 / 9 | 3.77 |
| G1 | 300 | 0 / 0 / 2 | 0.31 | 2 / 9 / 10 | 4.23 |
| G1 | 600 | 0 / 0 / 0 | 0.00 | 1 / 6 / 8 | 3.69 |
| G1 | 1,200 | 0 / 0 / 0 | 0.00 | 0 / 9 / 10 | 4.61 |
| G2 | 150 | 0 / 0 / 1 | 0.10 | 0 / 7 / 10 | 4.11 |
| G2 | 300 | 0 / 0 / 2 | 0.31 | 1 / 8 / 10 | 3.77 |
| G2 | 600 | 0 / 0 / 0 | 0.00 | 1 / 6 / 7 | 3.12 |
| G2 | 1,200 | 0 / 0 / 0 | 0.00 | 0 / 9 / 10 | 4.46 |

| truth | RFC-S error at N = 1,200 (flips per rep / cost regret %) | N_match (MVP), both metrics |
|---|---|---|
| G1 | 0.95 / 4.61 | 150 |
| G2 | 0.95 / 4.46 | 150 |

  - **Reading.** RFC-S does not improve with N (about 4% cost regret from 150 to 1,200): its error is
    bias from its form, not sampling noise. So the result is "MVP at N = 150 beats RFC-S at any N",
    not "MVP matches RFC-S with fewer completes". N_match = 150 is the grid floor; the true N_match
    may be lower.
  - **G0 guard.** N = 150: MVP cost regret +0.37 vs RFC-S (SE 0.37), one replicate (7.4%); an exact
    tie with 1 SE, counted as a failure, recorded per the rule. N ≥ 300: ok. No invented similarity.
  - N = 600 rows reproduce T1 (G1 RFC-S 1 / 6 / 8, 3.69%; G2 14 flips), as expected with the same seeds.
