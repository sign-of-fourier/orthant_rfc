# T11 · Joint chance-constrained product line (FAIL 2026-10-11)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

- **Hypothesis.** A joint chance constraint on line survival, P(every kept SKU's share ≥ t) ≥ 1 − α,
  evaluated as one d-dim orthant under a delta-method normal share distribution, (1) selects
  different assortment/price decisions than point-estimate or independent-product constraints,
  (2) those decisions are backed by realized survival across replicates, (3) orthant evaluation
  reaches the same decisions as nested MC materially faster.
- **Setup (proposed concretions, user to confirm).**
  - Fits: T5's MVP GHK fits, G1 and G2, N = 150 and 600, 20 reps (80 datasets, same seeds). No refits.
  - Market: competitors B, D, E fixed. Owned (J = 6): A, C (T1's owned pair) + A′, A″, F, G (T1's
    candidates). Price grid on A: {4.50, 5.00, 5.50} (A″ stays at 4.70). 127 lines (32 subsets with
    A × 3 prices + 31 without). d = kept SKUs, 1–6. Objective Σ_kept (p − 1.80)·ŝ_j (T1 economics).
  - Shares ŝ(θ̂): population shares by T1's GPU GHK share simulator with fixed draws, so ŝ is
    deterministic in θ and ∂ŝ/∂θ comes by autograd. Σ_s = J V Jᵀ.
  - t and α: α = 0.10; t picked from {2, 3, 5, 8, 10}% as the value whose binding rate under D is
    nearest 50% on G1 N = 600 reps 0–4 (fitted models only, never the truth), then fixed for all
    cells. "Binds" = arm A's line is infeasible under R.
  - R: θ_b ~ N(θ̂, V), B = 2,000 (MC SE of P_R ≤ 0.007 at P = 0.9); report B and P_R's MC SE.
    Same share simulator per draw.
  - F: OVUS GPU port from T9, variable order fixed per orthant at evaluation (order frozen across
    the price grid in the probe).
- **Step 0 (needed: T5 stored θ̂ only, no V).** Sandwich V at the stored θ̂ (Hessian + scores at
  the fit; no optimisation). Step0_G1 timing: 72 s at N = 600. Estimate 40 × ~72 s + 40 × ~20 s
  ≈ 60 T4-min ≈ $0.60. Then a 1-dataset timing probe of R (~$0.05) before the full run.
- **Estimate (before the probe).** Build ~half a day. D/E/F searches seconds per dataset. R:
  80 × 189 lines × 2,000 draws of a 10-alternative share simulation, rough 1–3 T4-h ($0.60–1.80).
  Total paid ≈ $1.30–2.50.
- **Kill criteria (proposed, approved by user 2026-10-11).**
  - Binding gate: if the constraint binds in < 20% or > 80% of the 80 dataset searches, stop and
    ask before scoring.
  - (1) Different decisions: PASS if, among binding searches, D's line ≠ B's in ≥ 15% and
    D ≠ C in ≥ 10%. KILL if D ≠ C in < 5% (joint adds nothing over independence).
  - (2) Survival: PASS if D's realized survival ≥ (1 − α) − 2 binomial SE pooled per truth × N,
    AND D's survival exceeds B's by > 2 paired SE, AND D's profit regret ≤ C's + 2 paired SE.
    KILL if D's realized survival < (1 − α) − 2 SE at N = 600.
  - (3) Speed: PASS if the fastest of D/E/F has decision agreement with R ≥ 95% and full-search wall
    time ≤ R/10. D vs E and D vs F reported, no gate (at d ≤ 6 all three are microsecond-scale; the
    gap to R is the delta method, not the integrator).
  - Normality flag (not a kill): median |P_delta − P_R| > 0.05 in a cell → that cell's (2) is
    reported with the caveat; N = 150 expected to be the weak cell.
  - Overall: PASS = (1) and (2) PASS; (3) separate. INCONCLUSIVE if (1) passes and (2) is within 2 SE.
- **Report-only probe.** One assortment (all six kept, d = 6), A's price on a 41-point grid
  4.50–5.50: P from D, E, F; max step jump / local slope; finite differences vs the Plackett
  gradient (∂P/∂a and ∂P/∂ρ chained through ∂Σ_s/∂p), per integrator. No price search built.
- **Out of scope.** J ≥ 20, posterior draws, gradient search, new truths, real data, G0.
- **Approved as drafted (user, 2026-10-11).** Code: `analysis/T11/t11.py` (ghk_gpu.py: sandwich and
  draws factored out, fit unchanged). Local wiring check OK.
- **Pilot (2026-10-11), G1 N = 600 rep 0, one T4: 147 s wall** (V 67 s, R 62 s at B = 2,000, delta step
  1.4 s, orthant arms < 1 s). nll at T5's θ̂ reproduces T5 (Δ 2e-4); V positive definite. D/E/F agree
  to ≤ 1.3e-3; |P_R − P_E| ≤ 0.016. Projected full run ≈ 2.5 T4-h ≈ $1.50 (in the estimate).
  - **Binding gate trips by construction (stop-and-ask).** With no per-SKU fixed cost, arm A keeps all
    six SKUs, and C's share is ~1.5% (fit and truth), below every t in {2..10}%: A's line is infeasible
    at every t, so binding ≈ 100% (> 80%). Full run not launched; awaiting user.
  - **User (2026-10-11): option 1, t grid extended with 0.5%, 1%, 1.5%;** same 50% selection rule. Arm names
    in results: Unconstrained (A), Point (B), Independent (C), Joint-engine (D), Joint-GHK (E), Joint-OVUS (F),
    Nested MC (R).
  - **User (2026-10-11): binding is not a gate** (Unconstrained failing the hurdle is its expected behaviour; a
    different test). **Every t scored, no selection**; overall PASS needs (1) and (2) at a majority of the 8 t.
- **Results (2026-10-11): FAIL. (1) KILL at all 8 t: the joint rule never picks a different line from
  Independent (0 of 640 searches).** `analysis/T11/out/summary.json`, `out/G*_N*/rep*.json`, `out/price_probe.json`,
  `run.log`. GPU: 80 jobs 973 s wall, ~2.5 T4-h (~$1.50); pilot + probe ~$0.10.

| t | NMC ≠ Point | Joint ≠ Point | Joint ≠ Independent | (1) | (2) | (3): agree with NMC (D = E = F) | speed vs NMC |
|---|---|---|---|---|---|---|---|
| 0.5% | 0.07 | 0.15 | 0.00 | KILL | fail | 0.90 | 45× |
| 1% | 0.25 | 0.33 | 0.00 | KILL | fail | 0.80 | 45× |
| 1.5% | 0.64 | 0.65 | 0.00 | KILL | fail | 0.78 | 45× |
| 2% | 0.36 | 0.36 | 0.00 | KILL | fail | 0.88 | 45× |
| 3% | 0.33 | 0.34 | 0.00 | KILL | fail | 0.79 | 45× |
| 5% | 0.81 | 0.80 | 0.00 | KILL | PASS | 0.76 | 45× |
| 8% | 0.44 | 0.44 | 0.00 | KILL | fail | 0.95 | 45× |
| 10% | 0.69 | 0.70 | 0.00 | KILL | PASS | 0.89 | 45× |

  - **Why joint = independent.** Near the 90% boundary, risk sits in one SKU: at t = 5%, 350 of 399 borderline
    lines have exactly one SKU with z < 2.5. With one risky SKU, the joint probability equals the product
    whatever the correlations (here −0.95 to 0.91; 68% of Σ_s mixed-sign). Joint vs Independent feasibility
    differs on ≤ 3 of 9,520 lines per t (max |ΔP| 0.01–0.11).
  - **Integrators.** Joint-engine, Joint-GHK and Joint-OVUS make identical decisions at every t;
    |P − P_NMC| medians identical to 4 digits by d.
  - **Survival (2).** The uncertainty-aware rules (Independent = Joint) keep realized survival 0.80–1.00
    (floor 0.77) in every cell; Point 0.30–1.00. Gain over Point > 2 SE at 5% and 10% in all cells, and
    in some cells at 2% and 8%. Price: regret 0.2–11% vs Point's −0.5 to 6% (Point lines often break the
    hurdle in truth); conservative at N = 150 (survival 0.95–1.00, regret up to 11%).
  - **Speed (3).** Full search ~1.4 s (delta step 1.35 s, orthants ≤ 0.06 s) vs NMC 62 s at B = 2,000: 45×,
    from the delta method, not the integrator (Independent costs the same). Agreement with NMC 0.76–0.95;
    passes only at 8%.
  - **Normality.** Median |P_delta − P_NMC| ≤ 0.03 (N = 150, t 1–2%), ~0 at N = 600 for t ≥ 3%; max 0.10–0.34
    in every cell. Joint max = marginal max: the gap is non-normal marginal shares near zero, not dependence.
  - **Price probe (report only)**, all six kept, G1 N = 600 rep 0, 41 prices: informative only at t ≤ 2%
    (C at ~1.5% puts P ≈ 0 above). Max step jump vs E's local slope: D 0.03 / 0.10 / 0.05 / 0.20 at
    0.5 / 1 / 1.5 / 2%, E and F ≤ 0.15 (curvature, not roughness). FD vs Plackett (max |Δ| / max |g|):
    E, F ≤ 2.3e-5; D 3e-6 / 2e-4 / 1.7e-3 / 1.5e-2, growing as P falls.
  - **Reading.** The hypothesis fails at its core: on this DGP joint and independent chance constraints
    make the same decisions. What helps is the uncertainty-aware constraint itself (vs Point), and any
    product-of-marginals version gets that.
