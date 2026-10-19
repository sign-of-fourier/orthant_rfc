# T20 · Sample-size curves and equivalent sample size (design, 2026-10-19)

Status: **done 2026-10-19. RFC-S has a bias floor of about 5% cost regret at every N from 50 to 1,500 on both
correlated truths. MVP at N = 50 is already below it. No crossing, so the equivalent-sample-size ratio is
unbounded: > 30× at N_MVP = 50 (model-free, grid-limited). The logit control shows no MVP advantage.**

## Question

T5 compared two points. MVP at N = 150 had lower decision error than HB-MNL + RFC-S at N = 1,200, and RFC-S
did not improve with N. The "N ≥ 720" in ADVANTAGES.md was T5's pass threshold, not a measurement. T20
replaces that with:

- error-vs-N curves for both methods;
- a fitted bias floor for each method;
- an equivalent-sample-size ratio N_RFC-S(e) / N_MVP(e) wherever the two curves reach the same error e.

## Reused from T5, unchanged

- **Generator:** `analysis/T1/run.py` `make_data`, `holdout_tasks`, `simulate`. Market, T = 12 tasks, 3
  products + none, 6 holdout tasks.
- **Truths:**
  - G1 and G2 (correlated errors);
  - G0 (logit truth), as the control.
- **MVP:** `analysis/T1/ghk_gpu.py` `fit`. Exact panel likelihood by GHK, M = 4,096, total error variance
  fixed (`--scale total`), no SE pass (T5's `--no-se`), T4 GPU.
- **Incumbent:** `run_hb` (bayesm, 20,000 iterations), then `RFC` + `tune_rfc`. RFC-S has 1,000 level draws and
  is tuned on holdout share MAE, exactly as in T5.
- **Decisions and metric:** `decisions` + `score` against the truth simulator. The decisions are D-price,
  D-ext (placeholder hurdle) and D-cost.

## What changes

- **Grid:** N = 50, 75, 100, 150, 200, 300, 450, 600, 800, 1,000, 1,500 × G1, G2, G0 × R = 10 (planned 20; halved
  for cost).
- **Seeds:** independent per (truth, N, replicate), from `SeedSequence([20261019, truth, N, rep])`.
  - In T5, every N for a given replicate shared the same seed, so the first 150 respondents' tastes were
    identical at N = 150 and N = 1,200.
  - T20 cells are therefore independent, and T5's N = 150 / 300 / 600 numbers will not reproduce exactly.
- **Where things run:**
  - Both arms run on Modal (`modal_t20.py`). Each container regenerates its own dataset from the seeds.
  - `run.py assemble` checks that both arms saw identical choices (by hash) and scores the decisions locally.
  - The probit truth's shares use the local orthant binary, as in T5.

## Metrics and analysis (`summarize.py`)

- **Primary:** cost regret % (T5's regret metric).
- **Secondary:** flips per replicate (price + ext + cost, 0–3), T5's other metric.
- **Per method, truth and N:** mean and 95% bootstrap CI over the replicates.
- **Curve:** err(N) = a + b·N^(−c), fitted by bounded least squares on the cell means. Bootstrap CIs resample
  replicates within each N, with the same draw for both methods (the two methods are paired on the same
  dataset).
- **Equivalent sample size:** take 10 target errors e, log-spaced, from MVP's fitted error at N = 1,500 to its
  fitted error at N = 50.
  - For each e, N_m(e) = (b / (e − a))^(1/c), and the result is the ratio N_RFC-S / N_MVP with a bootstrap CI.
  - If e ≤ RFC-S's floor a, the result is "no crossing" (the ratio is infinite). P(no crossing) is reported.
  - Any N outside 50–1,500 is marked as an extrapolation.
- **Headline:**
  - (a) Is RFC-S's floor a above MVP's mean error at N = 150, 200 and 300? Reported as a paired bootstrap CI
    of the difference.
  - (b) The multiplier curve, where a crossing exists. If the multiplier is below 2 anywhere in the
    practical range (N_MVP 150–300), say so.
- **Control (G0):** MVP should show no advantage. Flag it if RFC-S − MVP has a CI excluding 0 in MVP's favour.
- **Plots:** `out/err_vs_N_<metric>.png` (error vs N, log x, both methods, fitted curves) and
  `out/multiplier_<metric>.png` (ratio vs target error; no-crossing targets shaded).
- **Wall time** per fit is recorded: MVP on a T4, HB and RFC-S tuning on one Modal CPU core.

## Expected problems (from T5 and a synthetic check of the summarizer)

- **RFC-S's floor may sit above every MVP target.** In T5 it was about 4% cost regret at all N, against MVP's
  0–0.3%. If that holds, most targets will show "no crossing". Headline (a) is then the result, and the
  multiplier curve will be short or empty.
- **The metric is coarse.** Cost regret takes a few discrete values per replicate and is often exactly 0.
  - The power-law b and c will be weakly identified, and the summarizer test hit the c bounds.
  - The floor a and the cell means are the robust outputs.
- **N = 50:** 21 MVP parameters from 600 choices. Non-convergence is recorded, not hidden.

## Budget (estimated from T5's measured times, before any T20 run)

| | fits | basis | estimate |
|---|---|---|---|
| MVP, T4 GPU | 660 | T5 median 0.12–0.145 s per respondent; ΣN = 5,225 per truth × replicate → 313,500 respondent-fits; T5 billed $3.40 for 17,379 fit-s | 40–46k fit-s → **$8–9** (worst case with T5's max times, ~$12) |
| HB + RFC-S, Modal CPU (1 core, 3 GiB) | 660 | T5 local: HB 0.16 s + RFC-S tune 0.09 s per respondent | ~78k core-s ≈ 22 core-h → **~$5** (±50%: Modal core speed vs the local box) |
| **Total** | 1,320 | | **~$13 (range $10–18)**; wall 1–2 h, set by how many T4s Modal runs at once |
| Smoke: G1, N = 50 and 1,500, 1 replicate | 4 | | under $0.10, ~10 min including image builds |

**Revised after the smoke test, R = 10:**
- Smoke times: MVP 0.149 s per respondent at N = 1,500; HB 0.16 s + RFC-S tuning 0.066 s per respondent.
- Full run: 330 fits per method, about **$7**.

## What ran (user's cuts, 2026-10-19)

- **Replicates halved:** R = 10, not 20.
- **"Binary search" on the N grid:**
  - MVP ran only at N = 50, 100 and 200 (the cheap end, where the question is).
  - HB-MNL + RFC-S ran at all 11 N.
  - Two RFC-S cells are partial because a stopped run had not reached them: G0 N = 600 (5 of 10) and G2
    N = 1,000 (6 of 10). G0 N = 1,000 is missing.
- **Failure:** 1 MVP fit failed. G0 N = 100, replicate 9: Cholesky not positive definite, deterministic on retry,
  under the misspecified logit truth. That cell has 9 replicates.
- **Smoke fit not counted:** the G1 N = 1,500 smoke MVP fit is kept in out/ but excluded as a cell.
- **Compute:** 90 MVP fits, 1,913 T4 fit-seconds (~$0.4); 311 HB + RFC-S fits, 7.8 CPU-hours (~$1.6). Plus
  whatever GPU time the stopped first launch used before any MVP fit returned (see the Modal dashboard).

## Results

Full tables: [out/RESULTS.md](out/RESULTS.md). Plots: [err_vs_N_cost_regret.png](out/err_vs_N_cost_regret.png),
[multiplier_cost_regret.png](out/multiplier_cost_regret.png), and the `_flips` versions.

### Cost regret %, mean [95% bootstrap CI]

| truth | MVP N = 50 | MVP N = 100 | MVP N = 200 | RFC-S range of cell means, N = 50–1,500 | RFC-S floor a [95% CI] |
|---|---|---|---|---|---|
| G1 | 1.56 [0.66, 2.73] | 0.17 [0.00, 0.50] | 0.61 [0.00, 1.51] | 2.76–7.37, no trend | 4.95 [3.24, 5.71] |
| G2 | 0.63 [0.00, 1.68] | 0.42 [0.00, 1.27] | 0.42 [0.00, 1.27] | 1.78–7.43, no trend | 5.02 [3.89, 5.81] |
| G0 (logit) | 1.68 [0.00, 4.11] | 0.83 [0.00, 2.49] | 0.00 | 2.43 at N = 50, 0.75 at 150, 0 from 200 | ≈ 0 |

- **(a) The floor sits above MVP at practical N.** RFC-S floor minus MVP's mean:
  - G1: +3.39 [+1.37, +4.67] at N = 50, +4.78 [+2.97, +5.61] at 100, +4.33 [+2.27, +5.38] at 200.
  - G2: +4.39 to +4.60, every CI lower bound ≥ +2.9.
  - Paired by dataset, RFC-S − MVP is +3.1 to +5.1 points with every CI above 0 (G1, G2; N = 50, 100, 200).
- **(b) There's no multiplier curve.** RFC-S's error does not fall with N: the fitted curve is flat and b, c are
  unidentified, as expected. So no target error in MVP's range is ever reached by RFC-S, and P(no crossing) is
  0.98–1.00 across the bootstrap.
  - The defensible statement is model-free: RFC-S at N = 1,500 is still worse than MVP at N = 50, so the
    ratio is > 30 at N_MVP = 50, > 15 at 100 and > 7.5 at 200. These are lower bounds set by the grid's top, not
    estimates.
  - The multiplier never falls below 2 in the practical range. That isn't a speed-up claim: it's because the
    incumbent doesn't converge to MVP's error at any N.
- **Flips per replicate (secondary)** agree on G2, and at N = 100–200 on G1, where RFC-S never reaches MVP's
  rate up to N = 1,500. On G1 at N = 50, MVP's 0.90 flips per replicate is matched by RFC-S at N = 450 (ratio
  ≈ 9, one grid cell, noisy). On G2 at N = 50 (0.40), it's matched at N = 300 (≈ 6).
- **Control (G0):** no MVP advantage, nothing to flag.
  - Cost regret RFC-S − MVP: +0.75 [0.00, +2.24] at N = 50, −0.83 [−2.49, 0.00] at 100, 0 at 200.
  - Flips: +0.40 [−0.20, +1.00], 0.00, 0.00.
  - Both methods reach 0 by N = 200, and MVP is slightly worse at N = 100 (as T5 found at N = 150).
- **Time per fit, median:**

  | | N = 50 | N = 200 | N = 1,500 |
  |---|---|---|---|
  | MVP (T4) | 10–11 s | 23–29 s | 223 s (smoke fit) |
  | HB + RFC-S tune (1 CPU) | 6–7 s + 2 s | 25–28 s + 9–18 s | 182–204 s + 67–121 s |

### Caveats

- **RFC-S tuning lands on a bound in about 40% of fits** (T20 G1/G2: 43 of 110 and 48 of 106; T5 had the same
  rate). Checked with `floor.py` (local CPU, no Modal, 2026-10-19): G1 and G2 at N = 200, replicates 0–4, same
  datasets and HB seeds as the Modal arm. HB + T5 tuning reproduces the Modal RFC-S parameters exactly.

  | RFC variant, cost regret % (mean of 5) | G1 | G2 |
  |---|---|---|
  | RFC-S, T5 box (as run) | 7.37 | 3.97 |
  | RFC-S, wide box [1e-3, 50] | 7.37 | 3.97 |
  | RFC-G (σ per attribute), wide box, observed holdout shares | 6.19 | 0.85 |
  | RFC-G, tuned to the true population holdout shares | 9.21 | 0.40 |
  | RFC-G, tuned to the true shares of the decision scenarios (oracle share fit) | 3.69 | 3.97 |
  | MVP, same datasets | 0.00 | 0.00 |

  - **The bound is not the cause.** The bound being hit is mostly σ_l's lower limit. Tuning to holdout shares
    drives the attribute variability to zero, so RFC-S collapses to plain HB logit. A wider box lets it go further
    toward zero, with identical decisions.
  - **Per-attribute σ (RFC-G, not Sawtooth practice) closes most of the gap on G2 but not G1.**
    - On G2 (strong similarity, pair correlation .97) the holdout pairs carry enough signal.
    - On G1 (.90) even noise-free holdout shares don't identify σ.
    - T1's RFC-G at N = 600 (20 replicates) was 3.67% on G2, so the 0.85 here is 5 noisy replicates.
  - **Even the oracle stays at 3.7–4.0%,** with MVP at 0 on the same datasets. The oracle is RFC-G fitted to the
    true shares of the decision scenarios themselves. Share MAE is a poor proxy for these decisions, which turn on
    small profit differences, so this is not a hard lower bound on RFC. But no practical tuning of RFC on these
    HB estimates reached MVP.
  - **Conclusion:** the floor is robust for faithful RFC-S; it is not a tuning-box artefact. A generous RFC-G can
    narrow it when the similarity is strong.
- **Not tested: a nested logit estimated from the choices.** The similarity here is on two crossed attributes
  (flavor, pack), so one nesting tree can capture at most one of them. A cross-nested logit is the fair
  structural incumbent. Earlier oracle-utility work (`tests/rfc_vs_probit`) had a tuned nested-logit simulator
  worse than RFC on sibling source of volume (7.3–10.3 vs 5.6–6.4 points).
- **Cost regret is coarse:** a replicate's value is one of a few levels (0, 2.76, 4.61, 9.21, ...). The cell means
  are noisy at R = 10 (CIs ±2.5–3 points), but the floor–MVP gap is about 4 points with CIs clear of 0.
- **One simulated market (T1's), J ≤ 4 per task.** The floor is RFC-S's form failing on this market's
  near-duplicate structure (T1, T5). Its size elsewhere is untested.
