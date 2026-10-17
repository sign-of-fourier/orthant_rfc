# T16 · T6 against a practitioner incumbent: per-household independence

Design and kill criteria approved by the user 2026-10-17. Follow-up to [T6](../T6/TEST.md).

- **Question.** T6's MVP beat the pooled independence formula 1 − ∏(1 − p_j) by 49% on held-out reach. The
  correlation MVP estimated mixes two sources: household heterogeneity (heavy or loyal households buy several
  items) and dependence within a household given its tastes. Practitioner TURF (Sawtooth) computes reach per
  respondent from respondent-level estimates, 1 − ∏(1 − p_hj), averaged over respondents, which captures the
  first source. Does MVP still beat that incumbent?
- **Data.** As T6: dunnhumby Complete Journey, SOFT DRINKS, the same 30 products, the same 2,338 households
  active in both windows (weeks 1–27, 28–53), window-2 household reach as the target.
- **Repeat unit.** Window 1 split into 6 blocks of about 4.5 weeks (weeks 1–27 split evenly). Observation =
  household × block × product, bought at least once.
- **Arms.**
  - **IND:** T6's pooled independence (unchanged).
  - **MVP:** T6's IFM probit on window-1 vectors, reach by GHK on a T4 (unchanged).
  - **HB-IND (the incumbent, arm (a)):** `bayesm::rhierMnlRwMixture` as a binary logit per (block, product),
    one intercept per product per household, β_h ~ N(Δ, V_β) with full 30 × 30 covariance, one normal
    component, the Sawtooth-like prior used in T1 (`analysis/T1/hb.R`). Independent across products and blocks
    given the household. Window probability q_hj = 1 − (1 − σ(β_hj))^6, computed per posterior draw; reach of S
    = mean over draws and households of 1 − ∏_{j∈S}(1 − q_hj).
  - **BB-IND (sanity arm (b)):** per product, beta-binomial MLE on household block counts (0–6); q_hj = the
    posterior-predictive probability of at least one purchase in 6 new blocks. No cross-product covariance.
  - **Marginal match (T6's design rule: only the dependence differs).** For HB-IND and BB-IND, each product's
    q_hj is shifted by one constant on the logit scale so its mean over households equals the window-1 rate
    p_j that IND and MVP use. k = 1 reach is then identical across all arms (sanity check).
- **Assumptions (state in the write-up).** Blocks are independent given the household (no within-household
  clustering in time); window 2 (26 weeks) is treated as 6 blocks like window 1 (27). Removing an item does not
  change the others' utilities (as T6).
- **SEs.** Household bootstrap, B = 200. IND and MVP refit IFM per resample, as T6. HB-IND and BB-IND keep the
  fitted per-household q (their fit uncertainty enters through the posterior draws, not refits); predicted
  reach is averaged over the resampled households. Holm over k = 2–6 for the per-k HB-IND − MVP differences.

## Kill criteria (user, 2026-10-17)

- **Control (gate, run first; failure = stop and report).** Truth: one posterior draw of β_h per household from
  the real HB-IND fit; window-1 blocks and window-2 blocks simulated independently given the household (real
  heterogeneity, no within-household dependence). All arms refit on the simulated window 1.
  - MVP's edge over HB-IND (HB-IND error − MVP error, pooled k = 2–6) ≤ max(0.2 pts, 5% of HB-IND's error).
  - Recovery: HB-IND's predicted reach within 0.2 pts of an oracle's (mean absolute difference, pooled k = 2–6).
    The oracle is the true household q_hj, marginal-matched to the same simulated window-1 rates, so both sides
    carry the same sampling noise in those rates. Reported, not gated: HB-IND and oracle distance to the true
    expected reach, and the true-vs-fitted q_hj correlation.
  - History (user, 2026-10-17): first "q_hj correlation ≥ 0.9" (smoke test 0.79; 6 blocks at 1–7% block rates
    can't reach it), then "within 0.2 pts of the true expected reach" (an oracle with the true q misses by
    0.41–0.69 pts at full size because of the marginal match), then the oracle version above.
- **Prediction (primary).** Mean absolute window-2 reach error, pooled k = 2–6.
  - **PASS:** MVP's error ≥ 20% lower than HB-IND's.
  - **FAIL (heterogeneity explains it):** HB-IND within 0.5 pts of MVP, or better.
  - Otherwise **INCONCLUSIVE**.
  - **"Mostly heterogeneity" label:** HB-IND closes ≥ 50% of T6's gap, (IND err − HB-IND err) / (IND err − MVP
    err) ≥ 0.5.
- **Decision.** Greedy portfolios at k = 3–6 per arm (MVP vs HB-IND), selected on window 1, scored by window-2
  reach. A flip counts if |reach difference| ≥ max(0.5 pts, 2 SE). PASS: ≥ 2 of k ∈ {3, 4, 5, 6} have counted
  flips, MVP wins the majority, none lost.
- **Out of scope.** MVP conditioned on heterogeneity (a fourth arm), other categories, full-ML MVP.

## Order

1. Smoke test (a few hundred households, short chain): wiring, k = 1 sanity, timing. Quote the full cost.
2. Real HB-IND fit (it supplies the control's truth). No real scoring yet.
3. Control gate. Stop and report.
4. Real stage (only after the gate passes).

## Results (2026-10-17): prediction FAIL ("mostly heterogeneity"), decision FAIL (no flips)

**Files.** `t16.py`, `hb_bin.R`; `out/fit.json`, `out/control.json`, `out/real.json`, `*.stdout`. HB posterior draws
(`out/hb_real_beta.npy`) are not committed.

**Fits.** HB-IND on Modal CPU (1 core): real 49 min, control 44 min, log-likelihood drift Q3→Q4 0.08–0.09%.
About $0.15 in all, plus about $0.01 of T4 GHK for MVP.

**Control gate: PASS.** MVP edge over HB-IND −0.05 pts (limit 0.2); HB-IND vs oracle 0.19 pts (limit 0.2,
narrow). Reported: pooled errors IND 2.62, MVP 0.58, HB-IND 0.53, BB-IND 1.66 pts; HB-IND and the oracle both
0.32 pts from the true reach; q correlation 0.82 (BB 0.76). Under a heterogeneity-only truth, MVP's pooled
correlation absorbs the heterogeneity about as well as HB-IND.

**Prediction (real window 2), mean absolute reach error, pts.** HB-IND − MVP with household-bootstrap SE (B = 200),
Holm over k = 2–6.

| k | IND | MVP | HB-IND | BB-IND | HB-IND − MVP (SE) | Holm p |
|---|---|---|---|---|---|---|
| 1 | 1.78 | 1.78 | 1.78 | 1.78 | — (sanity) | — |
| 2 | 2.59 | 2.36 | 2.38 | 2.52 | +0.03 (0.08) | 1.0 |
| 3 | 2.92 | 2.35 | 2.37 | 2.65 | +0.02 (0.09) | 1.0 |
| 4 | 4.13 | 2.19 | 2.28 | 3.28 | +0.08 (0.10) | 1.0 |
| 5 | 5.85 | 2.17 | 2.41 | 4.42 | +0.24 (0.12) | 0.20 |
| 6 | 7.22 | 2.18 | 2.46 | 5.37 | +0.28 (0.13) | 0.20 |
| pooled 2–6 | 4.54 | 2.25 | 2.38 | 3.65 | +0.13 (0.10), 95% CI [−0.06, +0.32] | — |

- MVP is 5% better than HB-IND pooled (PASS needs ≥ 20%); HB-IND is within 0.5 pts. **FAIL** as labelled in
  advance.
- HB-IND closes **94%** of T6's gap ((4.54 − 2.38) / (4.54 − 2.25)): **mostly heterogeneity**.
- A small, non-significant MVP edge appears at k = 5–6 (+0.24, +0.28 pts; Holm 0.20), below materiality.
- BB-IND (no cross-product covariance in the heterogeneity) closes only 39%: the covariance of household tastes
  across products is what matters, not per-household rates alone.

**Decision: FAIL (no flips).** MVP and HB-IND greedy paths are identical at every k (1–6), including T6's
mechanism: HB-IND also takes the 2 L format (844165) rather than stacking the second 12-pack.

**Conclusion.** T6's win over pooled independence was almost entirely household heterogeneity, which practitioner
TURF (respondent-level independence) already captures. Against that incumbent MVP gives no different portfolio
and no material reach gain on this category. T6's PASS stands as stated (against pooled independence), but it is
not an advantage over Sawtooth-style TURF.
