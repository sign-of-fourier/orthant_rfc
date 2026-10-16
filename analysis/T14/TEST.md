# T14 · MBC linked sets at k > 12 (approved 2026-10-11; Part A PASS 2026-10-12, Part B PASS 2026-10-13)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

- **Part A, simulated (T2 extension).** T2's DGP and truths (L0 null, L1 factor, L2 two-cluster, LC
  complementarity) at k = 20 and 30, menu M = k (T2's k = 12 cell was M = k), 3 price levels, own-price effects
  only, no heterogeneity; N = 1,000 × 10 menus + 4 holdout menus; 20 replicates per (truth, k) = 160 datasets.
  L2 splits the linked set in halves (10 + 10, 15 + 15). LC: T2's two pairs + one triple.
  - Arms: CS-F (analyst lift rule, cap 5), CS-O (oracle blocks), IL, MVP-IFM (`multivariate_probit`, margins in all
    M log prices; nearest-PSD repair of R: rate, ‖R − R̃‖_F, min eigenvalue before), MVP-ML (GPU GHK, the `mvp_fit`
    service as in T2, time-capped).
  - Decisions as T2, scored against the truth by GHK-QMC: best pair, best triple, TURF-4 (C(20,4) = 4,845 /
    C(30,4) = 27,405 sets), D-price errors. Holdout pattern LL by per-observation GHK (2^k enumeration is out).
- **Part B, real (T3 extension).** Proposed cluster: **home baking, 24 items** (eggs, sugar, flour, butter, cream
  cheese, cake mix, frosting, brownie mix, muffin/cornbread mix, chocolate chips, marshmallows, baking nuts, oil,
  cooking spray, shortening, baking powder/soda, extracts, pie filling, pie crust, whipped topping, cake decor/icing,
  graham crackers, aluminum foil, refrigerated cookie dough). Basket rates 0.25–9.5%; raw lifts median 4.8, 98% of
  276 pairs > 2 (raw lifts are inflated by trip size; T3's gate decides). Same households, covariates and
  time-split holdout as T3.
  - Arms: CS-F, CS-O, IL, MVP-IFM, MVP-ML if within the cap. Decisions: best pair, best triple, TURF-4, and the
    recommendation A → B for each of the 24 items (27 decisions). Holdout adjudication as T3 (household bootstrap).
  - Build change: T3 enumerates all 2^M patterns; at M = 24 decisions use 2–4-dim marginals and the holdout LL uses
    per-trip GHK on GPU.
- **Report only.** Per-observation likelihood cost (value + gradient) at d = 20 / 30, GHK (M draws as used) vs OVUS
  (T9's GPU port), one T4.
- **ML time cap (proposed).** 60 T4-min per fit. Step 0 times one ML fit at k = 30 (L2) and at M = 24 (real); if
  either projects > 60 min: stop and ask.
- **Kill criteria (approved 2026-10-11, adjusted below).**
  - A, Q1: pool L1 + L2 at k = 20, 30 (80 datasets × 3 decisions = 240). PASS if CS-F flips exceed MVP-IFM's by
    ≥ 48 (20 pts) and CS-F's mean regret exceeds MVP-IFM's by > 2 paired SE. KILL if CS-F is within 24 flips (10 pts)
    of MVP-IFM. Else INCONCLUSIVE.
  - A, guards (caveats, not kills): L0 median |ρ̂| ≤ 0.05; LC reported.
  - A, Q2 (report): IFM-vs-ML decision agreement; PSD repair rate and size, and whether repaired datasets flip more.
  - B gate (as T3): ≥ 6 items connected at ρ ≥ 0.2 after covariates; fail → stop and ask.
  - B: PASS if, on decisions where CS-F and MVP-IFM disagree, the holdout backs MVP on ≥ 3 more than CS-F and on a
    majority of the resolved ones. KILL if the holdout backs CS-F at least as often as MVP.
  - B controls: T3's two simulated controls, IFM only.
- **Estimate.** Build ~1 day. Step 0 (two capped ML timing fits + the OVUS/GHK microbench): ≤ 2 T4-h, ≤ $1.20.
  A without ML (CPU containers, as T2): ~$2. A with ML: all 160 fits at 20–60 min ≈ $30–95; or **ML on the kill
  pool only (L1, L2 × k 20, 30), 10 reps = 40 fits ≈ $8–24 (proposed)**. B: CPU ~$0.50, one ML fit ≤ $0.60, holdout
  LL on GPU ~$0.10.
- **Out of scope.** Engine in the likelihood, Gibbs, new truths.
- **Approved with adjustments (user, 2026-10-11).**
  - Baking cluster approved. Staples (eggs, sugar, oil, foil) test false linking: trip size must be a covariate
    (T3's trip-size covariate kept, alongside household propensity); post-covariate ρ for staple pairs reported
    separately, so a pass cannot rest on staple links.
  - M = k approved (partial menus are a separate question).
  - 60-min ML cap approved. If ML projects over the cap at a k: IFM only there, IFM-vs-ML agreement recorded as
    unanswered at that k (ML is not in the kill criteria).
  - ML on L1/L2, 10 reps (40 fits) approved.
  - Part A: the 240 decisions are best pair, best triple, TURF-4 (80 datasets × 3); price-change errors reported
    separately. CS-O reported alongside CS-F (if oracle subsets also lose, the claim is about the method).
  - L0 guard: also report max |ρ̂| and whether PSD repair ever triggers under L0 (it should not; if it does, a red
    flag on the pairwise step).
  - Part B: results by decision type (pair, triple, TURF-4, follow-ons). "Resolved" = held-out difference
    ≥ max(0.5 pts, 2 SE) under the household bootstrap (T6's rule), fixed up front. PASS also requires MVP not to
    lose on any of the 3 set decisions (pair, triple, TURF-4). Follow-ons are not independent tests (under
    independence CS-F tends to pick the most popular item for every follow-on).
- **Step 0, first attempt (2026-10-11): lost.** The bench OOM'd (one autograd graph over all chunks) and the error
  discarded both ML results after ~63 min wall (~$1.20, T4 mvp_fit service); ≥ 1 fit ran to the 60-min T4 cap.
- **Step 0 rerun (user 2026-10-12: ML on an H100 via mvp_fit's own code in an ephemeral app; ≤ ~$0.40).**
  `t14.py step0`, `out/step0/*.json`, `step0_rerun.log`; each part saved as it arrives. 130 s wall, ~$0.15.
  - ML on one H100 (512 draws, float32 GHK, no SEs), max_iter 1 then 5 in one container (the 5-iteration call ran
    3 joint iterations: L-BFGS's eval cap is 1.25 × max_iter):
    sim L2 k = 30 (10,000 obs, 30 covariates): 4.2 s (incl. CUDA start) / 0.69 s, 5.1 GiB peak;
    baking (112,319 obs, 24 items, 47 covariates): 6.9 s / 5.1 s, 19 GiB peak.
    Upper bound per likelihood + gradient: ~0.1 s (sim), ~0.85 s (baking). Even at L-BFGS's 625-eval limit:
    ≤ ~1 min (sim), ≤ ~9 min (baking) per fit, well under the 60-min cap. T4 would be ~7–12× slower, so the lost
    T4 run's cap hit was most likely the baking fit.
  - Bench (report only, one T4, value + gradient, 512 draws): GHK 137 / 347 µs per obs at d = 20 / 30; OVUS
    48 / 80 (2.8× / 4.3× cheaper). Cost only; accuracy not measured here.
- **Part A results (2026-10-12): PASS.** `run_a.py`, `out/A_full/` (summary.txt, summary.json, per-job JSON), 160
  datasets, 0 errors, ~20 min wall, ~$4. Truth decisions GHK-QMC 2^16 (CPU); arms' probit scoring GHK-QMC 2^14 on a T4
  (checked identical to the CPU path on the selftest); ML (mvp_fit's code) on H100s, 5–8 s per fit, 40/40 converged.
  - Kill pool (L1 + L2, k 20/30, 240 decisions): flips CS-F 207, CS-O 119, MVP-IFM 50; gap 157 (PASS ≥ 48). Mean
    regret gap CS-F − IFM 5.2% of the best per decision, paired SE 0.30 over 80 datasets. In absolute points: flips
    with regret > 0.5 pts CS-F 185, CS-O 115, IL 235, IFM 2, ML 1; mean absolute regret CS-F 1.90 pts (max 10.1),
    CS-O 0.89 (max 3.6), IFM 0.037 (max 0.87), ML 0.033 (max 0.53). Oracle blocks also lose: the cap of 5 cannot
    hold a 20–30-item linked set.
  - L0 guard: IFM |ρ̂| median 0.013, max 0.091; PSD repair never triggered (0/40, nor on any truth). IFM flips 34 of
    120 under L0, 2 with regret > 0.5 pts (max 0.78 pts); CS 19, none > 0.5 pts.
  - LC (reported): CS-F / CS-O 0 flips > 0.5 pts (max 0.33); IFM 3 (max 0.71); IL 81.
  - Q2: IFM and ML pick the same set on 115/120 decisions (flips IFM 24, ML 23 on those datasets).
  - Price-change errors (reported separately): marginal-change MAE IFM / ML 0.34–0.42 pts vs CS-F 0.05–0.11 at every
    truth: the MVP margins carry all k log prices (spurious cross-price slopes); CS fits own + screened in-subset
    cross terms. A margin-specification effect, not the correlation.
- **Part B selftest (2026-10-12, 300 households, ~$0.40).** `run_b.py`, `out/B_selftest/`. Pipeline runs end to end
  (probit set quantities via a 2^24 latent-draw histogram on a T4, checked against brute force; coded-subset set
  quantities checked against simulation, max |Δ| 4e-3 at MC noise). Results not meaningful at this size: holdout
  covariates sit far from training (log-price columns at +1.35 / −1.31 training SDs on the later weeks), so predicted
  holdout rates are 2–13× the observed for every arm (eggs IL: 0.23 vs 0.10); raw IFM R min eigenvalue −1.50 (repair
  ‖R − R̃‖_F 2.28); ML hit its 500-iteration cap with a singular R. Fixes: IFM fit with the library's repair on and
  the raw pairwise R recomputed for the record; correlation matrices shrunk 1e-6 toward I when singular (scoring only).
- **Part B results (2026-10-13): PASS, at the threshold (3 − 0 = 3).** `run_b.py`, `out/B_full/` (gate.json,
  per-job JSON), `run_b_full.log`. 1,869 households, 149,789 trips (112,319 training), 45 covariates. ML 130 s on one
  H100 (284 iterations, converged, R min eigenvalue 0.036). Fits and controls on Modal CPU (2 vCPU, 8 GiB), scoring on
  T4s; 93 min wall.
  - Gate: PASS, 22 of 24 items linked at ρ ≥ 0.2; 19 with staple pairs zeroed. Post-covariate ρ, staple pairs median
    0.16 (max 0.48, 36% ≥ 0.2); non-staple median 0.16 (max 0.92, 41% ≥ 0.2). The pass does not rest on staples.
    Raw IFM R min eigenvalue −0.020, repair ‖R − R̃‖_F 0.024.
  - CS-F subsets: {eggs, muffin/cornbread mix, cooking spray}, {sugar, chocolate chips, marshmallows, baking nuts,
    graham crackers}, {flour, shortening, baking powder/soda, extracts, pie filling}, {butter, cream cheese, pie
    crust, whipped topping, aluminum foil}, {cake mix, frosting, brownie mix, oil, cake decor/icing},
    {refrig. cookie dough}.

| decision type | differ | backs MVP | backs CS-F | unresolved |
|---|---|---|---|---|
| best pair | 1 / 1 | 0 | 0 | 1 |
| best triple | 0 / 1 | 0 | 0 | 0 |
| TURF-4 | 0 / 1 | 0 | 0 | 0 |
| follow-on | 5 / 24 | 3 | 0 | 2 |

  - Where the picks differ (holdout value, MVP − CS-F with 95% household-bootstrap CI, relative lift, Holm-adjusted
    p over the 6; bootstrap p floors at 0.001):

| decision | MVP pick | CS-F pick | MVP − CS-F, pts [95% CI] | relative [95% CI] | Holm p | rule |
|---|---|---|---|---|---|---|
| best pair | eggs + cream cheese 1.19% | cake mix + frosting 0.87% | +0.32 [+0.15, +0.48] | +37% [+17, +61] | 0.006 | unresolved (< 0.5 pts) |
| extracts → | sugar 44.3% | flour 22.2% | +22.2 [+14.9, +29.7] | +100% [+57, +171] | 0.006 | MVP |
| graham crackers → | eggs 25.7% | marshmallows 11.7% | +14.1 [+6.8, +21.4] | +121% [+46, +247] | 0.006 | MVP |
| pie crust → | eggs 32.7% | cream cheese 22.0% | +10.7 [+4.0, +17.3] | +49% [+16, +91] | 0.006 | MVP |
| baking nuts → | eggs 32.8% | sugar 30.1% | +2.7 [−3.2, +8.4] | +9% [−10, +31] | 0.78 | unresolved |
| chocolate chips → | eggs 34.8% | sugar 36.7% | −1.9 [−8.0, +3.7] | −5% [−20, +11] | 0.78 | unresolved |

  - Report: significant at Holm 5%, MVP 4, CS-F 0; 3 of the 4 also material (≥ 0.5 pts). The best pair is a real
    but small gap. The follow-ons share one fit and several hinge on eggs, so the 4 are not 4 independent results.
  - Q2 (report): ML vs IFM differ on 1 of 27 (holdout backs IFM); ML vs CS-F differ on 5: ML 2, CS-F 0, 3 unresolved.
  - Controls (10 reps per truth, flips of 27 / material > 0.5 pts): S-MVP: IFM 21 / 19, CS-F 44 / 34, CS-O 67 / 55,
    IL 40 / 30. S-CS: IFM 37 / 27, CS-F 47 / 34, CS-O 47 / 33, IL 101 / 80. IFM flips least under both truths,
    including CS-F's own.
  - Holdout LL per trip: ML −1.524, CS-O −1.563, CS-F −1.569, IL −1.626. IFM −32.9 (S-CS controls −1,814): a few
    holdout patterns get near-zero probability under the repaired, near-singular IFM R. Not in the criteria; IFM's
    LL is not quoted until checked.
  - Invalid first run (kept as `out/B_full_badprice/`, `run_b_full_badprice.log`): `t3.load` dropped constant price
    columns by full-period SD, so prices constant in training (pie filling, extracts) were standardised by ~1e-15 and
    reached 2.8e13 on 5,598 holdout trips. Fixed: the SD rule uses training rows. T3's taco data is unchanged by the
    fix (same 16 columns, holdout max |z| 3.3).
