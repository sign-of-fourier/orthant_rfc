# T33 · MBC at k = 20 / 30: MVP vs Sawtooth-style serial cross-effects (2026-10-24): KILL

## Why

The MBC advantage (T2, T14) was measured against coded subsets (exhaustive alternatives within small blocks,
independence between them). Sawtooth's documentation says exhaustive coding is for small menus or "logical
sections", and recommends the serial cross-effects model for larger menus (Orme 2010; MBC v1.1 manual, 2016). On
real takeaway baskets, a serial conditional logit already tied MVP on decisions and beat it on joint log-likelihood
(T19's M1). This test adds that incumbent to T14's simulated menus.

## Design

- **Data, truths, arms reused from T14 Part A** (`analysis/T14/out/A_full`): truths L0 (null), L1 (one factor),
  L2 (two clusters), LC (logit complementarity) at k = 20 and 30, M = k, 3 price levels, own-price effects only,
  N = 1,000 × 10 menus + 4 holdout menus, 20 reps per cell; T14's truth decisions and its CS-F / MVP-IFM scores.
- **New arm, SCX (serial cross-effects logit).** One logit per item in order of train frequency, on all M log
  prices (own and cross-price effects) and the 0/1 picks of the items earlier in the order (conditional
  dependencies). The chain rule makes the joint exact. Weak L2 (C = 100) only against separation, as T19's M1.
  Pooled (no HB), as T14's truths have no heterogeneity. Decisions by 1M ancestral draws (common random numbers);
  holdout pattern log-likelihood exact.
- **New truth, CL (home advantage for SCX).** A serial logit in index order: intercepts set to T2's take rates,
  own log-price slope 1.6 × B_PRICE, conditional dependencies 0.9 within each half of the linked set and 0.3
  across. Truth decisions by 2M draws. MVP-IFM and CS-F are fitted on it here (MVP decisions by GHK at 4,096
  points).
- **Decisions** (as T14): best pair, best triple, TURF-4 at base prices, scored against the truth.

## Kill criteria (proposed by Claude, run approved by the user 2026-10-24)

- **Kill pool** L1 + L2 at k = 20, 30 (80 datasets × 3 decisions = 240), SCX against MVP-IFM. PASS (MVP's MBC
  advantage holds against serial cross-effects) if SCX flips at least 48 more decisions than MVP-IFM and its mean
  regret exceeds MVP-IFM's by more than 2 paired SE. KILL if SCX is within 24 flips of MVP-IFM. Else INCONCLUSIVE.
- **Guard** (CL truth): MVP-IFM flips no more than 24 more decisions than SCX.

## Results (2026-10-24): KILL

**Files.** `t33.py`; `out/` (per-dataset scores, `run.log`). Local CPU, 2 workers, about 4 h wall; no Modal cost.

| truth | k | flips b2 / b3 / TURF-4 of 20, SCX | MVP-IFM | coded subsets | holdout LL per menu, SCX / IFM / CS-F |
|---|---|---|---|---|---|
| L1 | 20 | 0 / 0 / 0 | 1 / 3 / 0 | 16 / 20 / 20 | −9.202 / −9.176 / −10.093 |
| L1 | 30 | 8 / 0 / 16 | 9 / 0 / 14 | 15 / 20 / 20 | −13.214 / −13.174 / −14.554 |
| L2 | 20 | 0 / 7 / 0 | 0 / 5 / 0 | 0 / 20 / 20 | −8.599 / −8.573 / −9.318 |
| L2 | 30 | 5 / 1 / 12 | 5 / 2 / 11 | 16 / 20 / 20 | −12.278 / −12.237 / −13.563 |
| LC | 20 | 0 / 2 / 4 | 0 / 3 / 4 | 0 / 20 / 19 | −12.055 / −12.055 / −12.037 |
| LC | 30 | 0 / 1 / 16 | 0 / 2 / 16 | 0 / 0 / 10 | −17.187 / −17.189 / −17.141 |
| L0 | 20 | 4 / 2 / 0 | 4 / 2 / 0 | 1 / 1 / 0 | −11.028 / −11.028 / −10.997 |
| L0 | 30 | 1 / 11 / 16 | 1 / 11 / 16 | 0 / 3 / 14 | −15.660 / −15.661 / −15.603 |
| CL | 20 | 0 / 1 / 4 | 1 / 13 / 7 | 0 / 6 / 20 | −8.837 / −8.836 / −10.218 |
| CL | 30 | 1 / 1 / 19 | 0 / 1 / 20 | 13 / 20 / 20 | −9.552 / −9.799 / −12.560 |

Many TURF-4 flips are immaterial (≤ 0.5% regret); material counts are in `t33.py summary`.

- **Kill pool (L1 + L2, k = 20 / 30, 240 decisions): KILL.** Flips SCX 49, MVP-IFM 50 (gap −1, kill line 24).
  Mean regret gap SCX − IFM −0.001 points per decision (paired SE 0.046). An exact tie. The partial read (4 reps)
  had SCX 4 flips behind, mostly L2 triples at k = 20; over 20 reps that is 7 vs 5 and reverses at k = 30.
- **Guard (CL truth, SCX's home model): holds.** MVP-IFM 42 flips vs SCX 26 (gap 16, limit 24), regret gap
  +0.005 points (SE 0.044); material flips (> 0.5% regret) IFM 4, SCX 2.
- **Holdout fit.** On the probit truths MVP-IFM fits the holdout slightly better, by 0.02-0.04 nats per menu; on
  CL, SCX is better by 0.25 at k = 30. Neither difference moves decisions.
- **Coded subsets** reproduce T14 (207 of 240 kill-pool flips), so the T2 / T14 wins stand against coded subsets
  and only against them.
- **Reading.** Sawtooth's recommended large-menu model, serial cross-effects logit, fitted pooled with plain
  logistic regressions in about 1-3 s, matches MVP's decisions on T14's menus. Same as T19 on real baskets. The MBC
  advantage is against the exhaustive-alternatives model, which Sawtooth itself says not to use at this size.
- **Not covered.** Respondent heterogeneity (T14's truths have none; Sawtooth fits serial cross-effects by HB) and
  an item order chosen by the analyst rather than by train frequency.
