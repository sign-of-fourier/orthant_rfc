# T17 · Within-person substitution on a real shelf: dunnhumby soft drinks

Design and kill criteria approved by the user 2026-10-17. Open question C4 of the CBC map: given a household's
tastes, does a near-duplicate take share mainly from its twin (which IIA can't represent), or from everyone in
proportion? First real-data test of PLAN item 1. The basket counterpart was [T16](../T16/TEST.md).

## Data

- dunnhumby Complete Journey, the 30 SOFT DRINKS SKUs of [T6](../T6/TEST.md). "Twin" = same `manufacturer_id`
  (1208: 10 SKUs, 103: 9, 2224: 8, 69 private label: 3).
- Choice occasion = a trip with exactly one of the 30 SKUs (78% of the 19,425 trips with any). Conditional on a
  category purchase (no "none").
- Choice set = the SKUs sold in that store within ±4 weeks (user 2026-10-17; the same-week proxy first planned gave
  a median shelf of 4 SKUs). 15,147 occasions, 1,875 households, 112 stores, median shelf 16 (2–27).
- Price of an SKU in a store-week = median unit price paid (`sales_value / quantity`) there, positive-price lines
  only; an SKU on the shelf but unsold that week takes the nearest sold week's price in that store. 5 occasions
  dropped (chosen SKU with no positive price within ±4 weeks).
- Time split: weeks 1–40 train (a validation slice for RFC tuning), weeks 41–53 holdout.
- All households (HB shrinkage handles the thin panel: median 4 trips per household).

## Arms

- **HB-MNL + RFC (incumbent):** bayesm `rhierMnlRwMixture`, SKU constants + log price, RFC tuned on the
  validation slice.
- **Pooled nested logit by manufacturer (second incumbent, user 2026-10-17):** Biogeme 3.3.5 on Modal (`mlogit`
  would not build locally; nothing installed in the local environment), one λ per nest. The
  textbook IIA fix; closed form, seconds to fit.
- **MNP:** structured Σ (manufacturer, format, flavour variance shares, the T4/T15 form) plus random tastes, GPU fit.
- **Orientation only:** pooled MNL.

## Kill criteria (user, 2026-10-17)

**Definitions.**
- **Twin episode:** a holdout store-week where SKU A is missing (A carried there, i.e. sold in ≥ 50% of that
  store's training weeks, but not sold that week) while at least one twin is present, or A's price is ≥ 15% below
  its regular price (median in that store over the training weeks).
- **Twin error:** |predicted − observed share of A's twins| within the episode, averaged over episodes weighted by
  choices.
- **Δ_twin** = incumbent twin error − MNP twin error, points (positive = MNP better). Reported for each incumbent
  (HB-MNL + RFC; nested logit).

**Step 0 (power gate, free).** Proceed only if ≥ 3 distinct focal SKU–twin groups, ≥ 30 episodes with ≥ 300
holdout choices in them, and the projected 95% CI half-width for Δ_twin ≤ m_twin.

**Materiality.** m_twin = max(1.0 pt, 15% of the incumbent's twin error). m_share = max(0.2 pts, 10% of the
incumbent's error).

**Primary: twin test,** graded on Δ_twin with its 95% CI [lo, hi] (household bootstrap):
- PASS: lo > 0 and Δ_twin ≥ m_twin.
- SOFT FAIL: lo > 0 and Δ_twin < m_twin.
- INCONCLUSIVE: lo ≤ 0 and hi ≥ m_twin.
- HARD FAIL: lo ≤ 0 and hi < m_twin.
- HARD FAIL + harm: hi < 0.
Per-group results with Holm-adjusted p; the verdict is on the pooled number.

**Guard 1: overall share error.** Mean absolute share error by chain × holdout week over all 30 SKUs; MNP not
worse than the incumbent by more than m_share.

**Guard 2: no invented similarity.** Logit truth simulated from the fitted HB-MNL; MNP's Δ_twin within m_twin of 0.

**Reading.** PASS with both guards = real evidence for item 1; SOFT FAIL = real but small; HARD FAIL = the incumbent
is adequate on this shelf. A PASS over HB-MNL + RFC but not over the nested logit = an advantage over Sawtooth
defaults only, not over a careful analyst.

## Known weaknesses

- "Not sold" ≠ "not on the shelf": low-volume SKUs look unavailable by chance.
- Price endogeneity: promotions are retailer-chosen; both arms share the bias.
- Multi-SKU trips (22%) dropped: the households most likely to buy both twins.
- Thin panel: heavy shrinkage for any heterogeneity model.

## Order

0. Step 0 episode count (free, local). Stop and report.
1. Smoke test of all arms on a subset; quote the full cost.
2. Guard 2 and the real fits; stop and report before reading the verdict if anything fails.

## Revisions after Step 0 (user, 2026-10-17)

- **Step 0 (same-week shelf): FAIL on power.** 379 episodes (323 missing-SKU, 56 price-cut), 1,432 choices, 20
  groups, 4 manufacturers, but the projected CI half-width was 2.56 pts against m_twin = 1.0 (`out/step0.json`).
- **Metric:** episodes pooled per focal SKU before taking |predicted − observed| (the per-episode version was
  mostly sampling noise); rolling time folds (weeks 1–13, 14–26, 27–40, 41–53), each predicted from models fitted
  on the other weeks.
- **Shelf:** ±4 weeks (above). This removes nearly all missing-SKU episodes (64 carried-but-absent pairs left), so
  episodes are **price cuts only**: the focal SKU sold that week ≥ 15% below its regular price, a twin on the shelf.
- **Simulated power check (gate):** truth = the nested logit fitted to the real choices; MNL and NL refitted per
  fold on simulated choices; the design passes if the correct model (NL) beats MNL on Δ_twin by the PASS rule
  (lo > 0 and Δ ≥ m_twin). If it fails, stop.

## Results (2026-10-17): power check FAIL, T17 stopped

**Files.** `step0.py`, `t17.py`; `out/step0.json`, `out/power.json`, `out/power_smoke.json`. `out/occasions.npz`
(derived data) is not committed. Modal CPU: smoke 1 min, full run under 2 min, about $0.01 in all.

**Real-data fits.** Biogeme's log-likelihoods match the probability formulas used for prediction to 1e-11.

| | log-likelihood | price coefficient (log price) | λ by manufacturer (103, 1208, 2224, 69) |
|---|---|---|---|
| MNL | −40,158.7 | +0.152 | — |
| Nested logit | −40,152.5 | +0.158 | 0.90, 1.00 (bound), 0.82, 1.00 (bound) |

**Power check.** 2,425 price-cut (occasion, focal SKU) pairs, 14 focal SKUs. Twin error MNL 2.52 pts, NL 2.50 pts:
Δ = +0.02 pts, 95% CI [−0.07, +0.10] (household bootstrap, B = 200), m_twin = 1.0. **FAIL**: even with the nested
logit as the truth and as a fitted arm, the twin test can't separate it from MNL.

**Reading.**
- Pooled over households, substitution within manufacturer is weak on this shelf (λ 0.82–1.0; two nests at the IIA
  bound). The departure from IIA that the data support is about 0.02 pts of twin share, 50 times below
  materiality, so no arm can show a material twin advantage with this design.
- The price coefficient is positive in both models. Price is not exogenous here (promotions, multi-buy unit
  prices, unit price not volume-adjusted), so "price cut" episodes are weak shocks. This also weakens any
  price-based version of the test.
- Not tested: within-person dependence given heterogeneous tastes (HB-MNL + RFC vs MNP). The power check concerns
  the pooled structure only; the real-data question stays open, but this shelf and price signal can't answer it.
