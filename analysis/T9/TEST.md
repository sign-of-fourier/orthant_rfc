# T9 · MACML as an incumbent probit arm (criteria set 2026-10-10)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

Background, hypothesis and scope: BACKLOG.md, "MACML".

- **Kill criteria (user, 2026-10-10).**
  - Recovery gate (T1 G1, one dataset, vs the GHK fit with SEs; `analysis/T1/out/step0_G1`):
    coefficients and variance shares within 2 SE of the GHK fit, SE of the difference
    √(SE_GHK² + SE_MACML²), MACML SEs from the Godambe sandwich; implied correlation max |diff|
    ≤ 0.1 → **changed (user, 2026-10-10, after the gate diagnosis below): every distinct implied
    error correlation's gap within 2 SE of the difference** (delta method on each fit's sandwich).
    Failure = STOP and diagnose (a mis-implemented baseline invalidates the test).
  - Win (G2): gate on regret only. GHK mean cost regret < MACML's by > 2 paired-SE. Flips
    reported, not gating. G1 reported alongside (does the gap grow with correlation).
  - G0/G1 no-loss: recorded. Flag if MACML beats GHK by > 2 SE on G1 (suggests GHK simulation
    bias).
  - Time: GHK ≤ 10 min/fit on one T4 at T1 size (T5 N = 600 cells: 57–118 s, already met). MACML
    time reported with the implementation caveat; no relative speed claim. MACML feasibility kill:
    30 min/fit, timed on one dataset first.
  - J = 20: stop and ask after J = 3–6 regardless of result.
  - Implementation: (1) pybhatlib pairwise CML over panel tasks if supported; (2) else pairwise CML
    written here on CPU with pybhatlib's MVNCD (OVUS), in scope if ≈ 1 day; (3) no pybhatlib MSL
    panel substitute; (4) no GPU MACML on our MVNCD; (5) if (2) exceeds ~1 day, drop and document.
- **Implementation check (2026-10-10).** (1) is out: pybhatlib 0.4.0 has no composite likelihood;
  `models.mnp` is cross-sectional with one Σ over labelled alternatives, `models.mnpkercp` is panel
  MSL. So (2): per respondent, all 66 task pairs (T = 12); each pair term is a 6-dim orthant (two
  tasks × 3 differenced alternatives) with the cross-task covariance from the random coefficients.
  Gradient: pybhatlib's analytic OVUS gradient (∂P/∂a, ∂P/∂Σ) chained to θ (21 parameters).
  Timing, this box (2 cores): OVUS value + gradient 0.28 ms per orthant → 11 s per likelihood
  evaluation (39,600 orthants at N = 600), ~6 s on 2 processes. Finite-difference gradients would
  be ~2 min per evaluation (over the kill). Projected fit 60–150 evaluations ≈ 6–28 min, Godambe
  sandwich ≈ 3 min.
- **Implementation as built (2026-10-10).** All on GPU (user: port OVUS to torch). `analysis/T9/macml.py`:
  batched torch port of pybhatlib 0.4.0 `_mvncd_ovus` with pybhatlib's torch Genz BVN; autograd
  gradients; Godambe sandwich. `t9.py check`: per orthant vs pybhatlib max |Δlog P| 1.8e-15, except
  exact |a/sd| ties (23 of 1,320; up to 1.6e-3): OVUS depends on the variable order and pybhatlib's
  tie order is numpy's unstable default argsort; the port breaks ties stably. Two fixes: (i) pybhatlib's
  torch BVN high-correlation branch gave NaN gradients at equal limits (sqrt'(0) in a branch
  torch.where discards); a copy with sqrt floored at 1e-300 gives bit-identical values. (ii) OVUS re-sorts
  variables at every θ, so the composite likelihood jumps where an order flips (line scan: jumps
  0.03–0.055 vs smooth second differences 1e-7) and L-BFGS stalled (max|grad| 14–28, start-dependent
  optima). Fix (user, option A): freeze each orthant's order, fit, re-sort, repeat until no order
  changes (gate fit: 3 rounds, 39,558 → 3,690 → 0 re-sorted); at the fit every orthant is pybhatlib's
  OVUS value; both starts reach the same optimum.
- **Gate (2026-10-10).** First form FAIL: correlation gap 0.143 > 0.1 (flavor share MACML .240 vs GHK
  .383, truth .45), all coefficient and share z within 2. Reps 1–2 gaps 0.056, 0.026: the 0.1 bar is
  ~1 SE of the difference at N = 600. Revised form (above) **PASS**: max correlation |z| 1.46,
  coefficients |z| ≤ 0.30, shares |z| ≤ 1.34. `out/gate.json`, `gate.log`, `out/diag.json`, `diag.log`.
  MACML fit 34 s + Godambe 17 s on a T4. GHK covariance from a rerun of step0 rep 0 with SEs
  (θ reproduced exactly).
- **Arms.** MACML (new, `analysis/T9/`) vs MVP GPU GHK reused from T5's N = 600 cells (same seeds,
  no new GPU). Metrics as T5: S1, flips, cost regret, fit time.

- **Results (2026-10-10): FAIL on the win criterion: no decision difference at all.** `out/full_{G0,G1,G2}/
  results.json`, `out/summary.json`, `run_G*.log`. GPU: 1,762 T4-s main + ~400 gate/diagnosis
  (~36 T4-min, ~$0.35).

| truth | GHK regret / flips | MACML regret / flips | MACML − GHK (paired SE) | S1 error GHK / MACML | fit s median GHK / MACML | verdict |
|---|---|---|---|---|---|---|
| G0 | 0.00% / 0 | 0.00% / 0 | 0.00 (0.00) | +0.044 / +0.026 | 74 / 26 | recorded |
| G1 | 0.00% / 0 | 0.00% / 0 | 0.00 (0.00) | −0.024 / −0.022 | 87 / 32 | recorded; no flag |
| G2 | 0.00% / 0 | 0.00% / 0 | 0.00 (0.00) | −0.006 / −0.020 | 65 / 28 | **FAIL** (no GHK win) |

  - Both arms made the truth's decisions in all 60 replicates (as GHK did in T5 at N = 600: 0 flips,
    0 regret). T1's three decisions are saturated at N = 600, so the test cannot separate the two
    probit estimators; the gap does not grow with correlation because there is no gap.
  - MACML's pair correlation S1 is as close to the truth as GHK's (|error| ≤ 0.026 vs ≤ 0.044).
  - Time: GHK 57–118 s/fit (cap 10 min: met). MACML 18–50 s/fit on a T4 as our GPU port; per the
    criteria no relative speed claim (pybhatlib's own CPU code would be ~10–30 min/fit here).
  - Not tested (OUT / stop-and-ask): J = 20, smaller N where decisions are not saturated, decisions
    beyond T1's three.
