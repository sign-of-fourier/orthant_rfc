# T15 · Engine pipeline on the shelf, J = 20–60 (partial 2026-10-16: engine vs OVUS done; GPU pipeline vs logit IIA not run)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

- **Design:** [analysis/T15/DESIGN.md](../../analysis/T15/DESIGN.md). This is T4's decision test, reframed as a
  search-throughput test. T4 stays paused.
- **Question (Q3).** At the same wall-clock budget, does the engine pipeline find better shelf decisions than a
  search that uses only GHK?
  - Stage 1: the engine scores about 100× more candidates.
  - Stage 2: GHK-QMC rescores the top K, then runs a local search.
- **Model.** T4's J = 20 gate fit (MVP-G 512 × 256). Population shares are orthants of dim = shelf size, with
  tastes folded into the covariance, as in `t13.shares()`; confirmed for T4's model.
- **Decisions (proposed):**
  - D1: assortment, k of 20.
  - D2: joint 6-SKU price grid, 9⁶ candidates.
  - D3: line extensions, 6⁶.
  - D4: joint line + extension + price, 6¹² ≈ 2.2 × 10⁹.
  - All use a cost/margin-weighted objective.
- **Arms:**
  - (a) HB-MNL + RFC-S with ASM-style search.
  - (b) Engine pipeline.
  - (c) GHK only, at the same budget.
  - (d) Shared-draw first-choice simulator in stage 1, then the same stage 2.
  - (e) Low-M GHK in stage 1, then the same stage 2 (user, 2026-10-16).
- **Reference.** High-M GHK on the union of shortlists. Assumption: the true best is in the union.
- **Kill criteria (user, 2026-10-16).** Full text and resolutions in the design.
  - **Materiality:** 0.5% of max profit.
  - **Step 0 gate:**
    - Each stage-1 scorer is sized so that its K = 100 shortlist contains a candidate within materiality of the
      reference best in ≥ 95% of streams (20 per instance, pooled per decision).
    - The simulator's N is the larger of that and correct ordering of D2's adjacent price levels.
    - **KILL** if the engine is < 3× faster in candidates/s than the best of (d) and (e). Then ask about a d = 40
      probe.
  - **Primary: (b) vs (c) on D4 and D1, scored against the reference.**
    - PASS: (b) lower by > 2 paired SE and ≥ materiality at ≥ 3 budgets; or (c) needs ≥ 5× (b)'s budget
      (log-interpolated), or never matches.
    - KILL: a tie or a loss at every budget.
    - INCONCLUSIVE: a win at 1–2 budgets, or a multiple of 2–5×.
    - The verdict must hold on both D1 and D4; otherwise INCONCLUSIVE, then ask about refits.
  - **Attribution, (b) vs (d) and (e):**
    - *Validated:* (b) beats both with ≥ 2× time-to-match.
    - *Engine not the reason:* (b) ties the better of the two.
    - *Other scorer better:* (b) loses to one of them.
  - **Guards (record only, scored against the truth):**
    - S0: one refit; MVP no worse than the incumbent by more than materiality.
    - S1: MVP beats HB-MNL + RFC by > 2 SE and materially on most of D1–D4; otherwise flag the contradiction with
      T5.
  - **Escalation to J = 40 / 60** if (c) never matches (b) on D4, or if the engine's speed ratio rises from
    d = 20 to d = 40.
  - **Runs:**
    - 5 seeds.
    - Three budgets set from Step 0's H100 throughput (about 10⁴, 10⁵ and 10⁶ candidates for the fastest
      adequate scorer). A budget is a cap: a run stops early once its search is exhausted and converged.
    - A fourth budget (about 3.3× the largest) for arm (c) only, where (c) hasn't matched (b). It resumes from
      the largest budget and is quoted first.
    - H100 for the main grid (user, 2026-10-16).
- **Step 0.** Adequacy sizing, the speed gate, and throughput on both a T4 and an H100: about $2.50–3.50. Stop if
  the projection exceeds $5. Then stop and report.
- **Cost (revised).** About $10–15 in total:
  - Step 0: $2.50–3.50.
  - Main grid on the H100: about $4, if its 12.5× speedup holds for these scorers.
  - Fourth budget: at most $4.
  - S0 guard: about $2.
  - CPU: under $1.50.
  - J = 40 / 60 only through the escalation rule.
- **Stated risk.** From T7 and T10 numbers, the engine may be only about 2–8× faster than GHK per candidate at
  d = 20, and slower than the simulator. Step 0 settles this before any spend.

- **Step 0 results (2026-10-16), first gate.** By the criteria as written, the engine is slower than the best
  competing stage-1 scorer at adequacy on every decision (gate: ≥ 3× faster).
  - **User (2026-10-16):** the gate compared the wrong things. The engine's competitor is OVUS only; the GPU
    pipeline's competitor is logit IIA (HB-MNL + RFC). The KILL label is withdrawn; the revised criteria are
    pending.
  - **Files:** `analysis/T15/t15.py`, `out/probe.json` (phase A), `out/step0b.json` (phase B).
  - **Runs:** phase A on one T4 and one H100; phase B on one H100, 809 s. About $1.15 in all, including a failed
    first launch (under $0.05).
  - **Model checks:**
    - Orthant shares match the simulator at 2^20 draws (largest z 2.3–3.3 per decision).
    - Engine vs GHK: median |ΔP| 1.1–2.5e-3, within its known envelope.
    - Fixed cost F = 0.0049 per SKU (0.5 × the median brand-0 SKU contribution). The firm is all of brand 0: 7
      shelved SKUs + 5 extension candidates.

| decision | candidates in sample (within materiality) | engine cand/s (adequate) | low-M GHK at adequacy | simulator at adequacy | engine ÷ best |
|---|---|---|---|---|---|
| D4 | 20,000 × 3 (1–3) | 32,800 | M = 128: 61,300 | N = 2,048: about 800,000 (probe) | **0.04** (0.53 vs GHK) |
| D1 | 38,760–184,756, exhaustive (1–10) | 697,000 | M = 64: 1,325,000 | N = 2,048: about 3–4 million (probe) | **0.2** (0.53 vs GHK) |
| D2 | 20,000 × 3 (1–2) | 225,000 | M = 64: 484,000 | N = 2,048: 520,000 | **0.43** |
| D3 | 7,776 × 3, exhaustive (2–10) | 80,000 | M = 256: 50,000 | N = 2,048: about 1 million (probe) | **0.08** (1.6 vs GHK) |

  - **Simulator N = 2,048, from the D2 adjacent-level ordering rule.** Adequacy alone was met at N = 1,024
    everywhere except D2, which needed 2,048. Ordering: 0.94 at N = 1,024, 0.97 at 2,048.
  - **Where a phase B simulator rate is missing**, the phase A H100 rate at N = 2,048 is used, marked "probe". The
    gate holds with or without it: low-M GHK alone already beats the engine on D4, D1 and D2.
  - **Why the cheap scorers suffice.** Only 1–10 candidates per instance are within materiality of the best, yet
    the top 100 of each sample contains one at GHK M = 64–256 and simulator N = 1,024. Decisions at this
    materiality need ranking, not accuracy.
    - The engine's objective error is large: 95th percentile 1–4% of max profit, several times the 0.5% floor.
      But it is systematic, so the engine's shortlists are adequate too (rate 1.0).
  - **M_acc** under my default rule (95th-percentile error ≤ 0.1 × materiality): D1 M = 2,048; D2 M = 16,384; D3
    and D4 not reached at M = 16,384 (error 1.1–1.4e-3 of max profit). Moot after the kill.
  - **Per the criteria:** stop and ask about a d = 40 timing probe. The main grid, the S0 guard and the incumbent
    were not run.

- **Engine vs OVUS as the stage-1 screener (2026-10-16).**
  - **Setup.** The same model and 12 instances, plus D2 and D4 on T4's J = 40 shelf (d = 40–43). One H100 and one T4.
    About $0.60 in all.
  - **Engine.** Run in screening settings chosen per hardware, with a logged trace.
  - **Reference.** High-M GHK, as in Step 0.
  - **Result.** Every arm met the adequacy rule on all 14 instances.

| | D1 (d 6–10) | D2 (d 20) | D3 (d 25) | D4 (d 25) | D2 J = 40 | D4 J = 40 (d 43) |
|---|---|---|---|---|---|---|
| engine ÷ OVUS, H100 | 4.6–7.0× | 9.1–10.4× | 11.4–13.2× | 13.8–14.6× | 10.2× | 20.5× |
| engine ÷ OVUS, T4 | 3.6–4.7× | 5.0× | 7.2–7.9× | 8.7–9.3× | 5.4× | 11.9× |

  - **Side result.** The test found an engine speed-up, fixed in the engine repository (not deployed).
  - **Private.** Screening settings, timings and the rules for choosing them are in private notes, not in this
    repository.

- **Status (2026-10-16): partial, no verdict.**
  - **Done (Q3, engine vs OVUS as the screener).** On this model at J = 20 / 40, the engine's screen is 3.6–20×
    faster than OVUS at equal adequacy, on both an H100 and a T4. The gap grows with d. That clears the proposed 3×
    bar on every instance.
  - **Also found.**
    - Screening at a 0.5% materiality needs a correct ranking, not accurate probabilities. Every cheap scorer
      (low-M GHK, the simulator, the engine, OVUS) is adequate.
    - The simulator and low-M GHK screen faster than the engine. That is not the engine's claim (user: its
      competitor is OVUS).
  - **Not run.**
    - The main question: the GPU M*P pipeline vs HB-MNL + RFC (logit IIA), regret against the truth.
    - (b) vs (c), pipeline value.
    - The S0 guard.
  - **Open.** Revised kill criteria for those (blank, user). The screening rules are provisional (private notes).
  - **Cost so far:** about $1.70 of GPU.

## Main comparison: GPU M*P pipeline vs HB-MNL + RFC-S (logit IIA) (criteria set by the user, 2026-10-16)

**Arms.**
- **Incumbent.** HB-MNL (bayesm, 20k iterations) + RFC-S tuned on the 6 holdout tasks (T4's `tune_rfc`), RFC draws
  R = 200 per respondent. Scenario search on CPU in the style of Sawtooth ASM: exhaustive if the space is ≤ 50,000,
  otherwise best-improvement hill-climbing from 20 random starts. Same neighbourhoods as below.
- **M*P pipeline.** The fitted probit (MVP-G 512 × 256) on an H100.
  - Naming (2026-10-16): the model is a mixed multinomial probit (MNP); "MVP-G" is T4's name for it. The
    pipeline is called M*P because the same rules serve MNP and MVP.
  - Stage 1: the engine screen to a top 100, over D1 and D3 exhaustive and 10⁶ random candidates for D2 and D4.
  - Stage 2: GHK M = 4,096 rescoring, then best-improvement local search.
- **Accurate-only (secondary, report only).** GHK M = 4,096 for 80% of the pipeline's wall time, then the same
  local search.

**Truth and regret.**
- **True profit:** each pick is scored at the true parameters, with GHK M = 65,536 (S1). S0's truth is a mixed
  logit, scored by simulation.
- **True best:**
  - D1 and D3: exhaustive at the truth.
  - D2 and D4: the best of a 10⁷-candidate screen plus local search at the truth, together with every arm's pick.
    Assumption: the true best is in that set.
- **Regret** = (true best − true profit of the pick) / true best, in % of best profit.
- **Also reported:** each arm's regret under its own model (its search error), separately from regret against the
  truth.

**Replication.**
- 5 S1 datasets (T4's DGP, reps 0–4, N = 1,000, 12 tasks). Rep 0 is the T4 gate fit; reps 1–4 are H100 refits.
- One S0 dataset (logit truth) for the guard.
- Pilot: rep 0 end to end, then stop and report.

**Verdict (graded; materiality m = 0.5% of best profit; user 2026-10-16).**
- **The statistic.** Per decision: Δ = incumbent regret − MVP regret, in points (positive = MVP better), with a 95%
  CI. The CI is a t-interval over the 5 per-dataset means (each the mean of the 3 instances).

| label | condition |
|---|---|
| PASS | lo > 0 and Δ ≥ m |
| SOFT FAIL (small lift) | lo > 0 and Δ < m |
| INCONCLUSIVE | lo ≤ 0 and hi ≥ m |
| HARD FAIL (no value) | hi < m and lo ≤ 0 |
| HARD FAIL + harm | hi < 0 |

- **Primary:** D1 and D4.
  - Overall PASS needs PASS on both; overall HARD FAIL needs HARD FAIL on both.
  - Otherwise the overall label is the better of the two, with the other stated.
- **Controls:** D2 and D3 are reported with the same labels and don't count.
- **Reporting rule (2026-10-13):** points with the CI first, then relative lift with its CI, then the Holm-adjusted
  p over D1–D4.
- **Wall time** to decision: report only.
- **S0 guard:** record only; MVP's regret should be no worse than the incumbent's by more than m.
- **Cost estimate:** about $12–14 for 5 datasets plus S0.

### Results, S1 (2026-10-16): D1 PASS, D4 SOFT FAIL; controls D2 SOFT FAIL, D3 INCONCLUSIVE

**Files.** `main.py`; `out/main/S1_rep{0..4}.json`, `truth_best_S1.json`, `summary.json`.

**Statistics.** 5 datasets × 12 instances. Δ = incumbent regret − M*P regret, in points of % best profit (positive
= M*P better). The 95% CI is a t-interval over the 5 per-dataset means; Holm is over D1–D4.

| decision | incumbent regret | Δ [95% CI] | relative lift | Holm p | label |
|---|---|---|---|---|---|
| D1 (primary) | 1.85% | +1.85 [+1.30, +2.40] | 100% | 0.003 | **PASS** |
| D4 (primary) | 0.25% | +0.20 [+0.13, +0.27] | 82% | 0.004 | **SOFT FAIL** |
| D2 (control) | 0.14% | +0.11 [+0.03, +0.19] | 79% | 0.03 | SOFT FAIL |
| D3 (control) | 0.37% | +0.32 [−0.19, +0.83] | 86% | 0.16 | INCONCLUSIVE |

- **Per-dataset Δ.** D1 1.20 / 1.79 / 2.40 / 2.07 / 1.79. D4 0.20 / 0.15 / 0.21 / 0.29 / 0.16.
- **Overall, by the pre-set rule.** PASS (D1), with D4 SOFT FAIL stated.
- **Model vs search error.**
  - The incumbent's D1 k = 6 and D3 searches were exhaustive under its own model, and it still lost up to 2.9%
    (D1 k = 6). That is model error.
  - M*P's regret is about 0 apart from a few instances (at most 0.41%).
- **Pipeline vs accurate-only (b vs c): tie.**
  - Identical picks on reps 1–4. The one difference in 60 instances: rep 0, D4 cost+20, where the pipeline's local
    search stopped 0.35% short under its own model (0.41% against the truth) and accurate-only did not.
  - The escalation rule (accurate-only fails to match on D4) is not triggered.
- **Incumbent fit.**
  - HB converged on every rep (drift ≤ 0.0007).
  - RFC-S holdout MAE 0.46–0.52 pts.
  - Rep 3: RFC-S σ_l at the lower bound (0.01, plain logit), flagged.
- **Time and cost.**
  - M*P fits on H100: 19–27 min (45–53 iterations; rep 0 is T4's L4 fit, 4.5 h).
  - HB-MNL: 3–4 min on CPU.
  - Searches: seconds per instance (M*P, H100) vs 2–12 min (incumbent, 4-core CPU). That is different hardware,
    so not comparable as speed.
  - Cost per dataset: about $1.50–2 for M*P (mostly the fit) vs about $0.30 for the incumbent.
  - Total for S1: about $8.50.

### Results, S0 guard (2026-10-16): pass

- **Setup.** Logit truth (mixed logit, iid Gumbel), one dataset; M*P is the misspecified model here.
  `out/main/S0_rep0.json`, `truth_best_S0.json`.
- **Regret, M*P − incumbent, in points.** D1 +0.00, D2 −0.06, D3 −0.11, D4 −0.03. M*P is never worse by more than
  m = 0.5 (it ties or is slightly better), so the guard passes.
- **No invented similarity.** The M*P fit put 0.88 of the error variance on the SKU's own term (brand 0.05,
  flavor 0.07, pack 0.003): the probit did not invent substitution structure under a logit truth.
- **Time.** Fit 27 min on H100 (61 iterations); HB 4 min on CPU.
- **Cost.** About $11 for T15's main comparison in all (S1 + S0).
