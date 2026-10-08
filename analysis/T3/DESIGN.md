# T3. Real basket data, a linked set larger than the cap: design (approved 2026-10-08)

Items 2 and 4. Code: `t3.py`, `run.py`, `summarize.py`.

## Question

T2 showed that coded subsets flip decisions exactly when a linked set exceeds the 5-item
cap, on simulated data. T3 asks the same on revealed data, where nobody knows the truth:

- **Q1.** On a natural cluster of 8+ complements, do decisions from coded subsets differ
  from those of a multivariate probit (MVP) over the whole cluster? Where they differ,
  which one does the holdout support?
- **Q2.** IFM vs full ML on real data (misspecified margins, heterogeneity): same
  decisions?
- **Q3.** Time-to-decision per method (analyst specification plus compute).

## Non-goals

- Price decisions. Prices are weekly, promotion-driven and not randomized
  (`analysis/rho_test/SUMMARY.md`); price coefficients are covariates, not levers.
- Respondent heterogeneity in the model (tracked follow-up from T2); handled here by a
  fixed covariate shared by every arm (below).
- SEs (tracked follow-up, after T4). Gibbs (T2: IFM = ML = Gibbs; too slow at this N).
- MACML (backlog).

## Data

dunnhumby Complete Journey, the `rho_test` loader and weekly price construction reused
unchanged. Household = respondent, trip = task, item = "any product of the category on the
trip".

- **Households and trips.** All households with ≥ 20 trips (1,869; ~150k trips), not the 800
  sample: item rates are 1–8%, so pair counts need the full panel. Holdout: last 2 trips per
  household (as `rho_test`).
- **Cluster: taco night (9 items).** Raw basket rates and lifts from a quick count over all
  155,691 baskets:

  | item | mapping (to confirm in the build) | rate |
  |---|---|---|
  | shells / tortillas | MEXICAN TACO TOSTADO SHELLS; MEXICAN SOFT TORTILLAS AND WRA | 1.3% |
  | taco seasoning | MEXICAN SEASONING MIXES | 1.2% |
  | salsa | MEXICAN SAUCESSALSAPICANTEE; SAL:SALSA/DPS-PRPCK | 2.4% |
  | beans | MEXICAN BEANS REFRIED; VARIETY BEANS - KIDNEY PINTO | 2.4% |
  | sour cream | SOUR CREAMS | 3.2% |
  | shredded cheese | SHREDDED CHEESE | 7.7% |
  | tortilla chips | TORTILLA/NACHO CHIPS | 4.6% |
  | lettuce | HEAD LETTUCE; VARIETY LETTUCE | 3.7% |
  | ground beef | BEEF (ground types; the name match found almost none, mapping to fix) | ? |

  Raw lifts: shells × seasoning 28, seasoning × salsa 16, other pairs 3–12. Raw lifts are
  inflated by trip size (in `rho_test`, controlling for it cut median |ρ| from 0.30 to
  0.08), so the cluster is confirmed by the gate below, not by these numbers.
- **Fallback cluster: cookout (8–10 items).** Hot dogs, dinner sausage, hot dog buns,
  hamburger buns, mustard, pickles/relish, baked beans, charcoal (+ ketchup, ground beef
  once mapped). Raw lifts 4–12, weaker than taco.
- **Covariates, identical for every arm.** Own log weekly price (descriptive); trip size
  (`rho_test` model B: log 1 + count of the 20 most frequent other categories, excluding
  cluster categories); household propensity per item (log of the household's
  training-period rate, shrunk to the population rate, computed without the scored trip) as
  a fixed stand-in for random intercepts.

**Gate (before any comparison).** Fit MVP-IFM on the training data. Proceed if at least 6
items are connected by within-trip ρ ≥ 0.2 (beyond trip size and propensity), i.e. the
linked set exceeds the cap. Otherwise try cookout; if neither passes, stop and report
(that is itself a finding: on this data real linked sets fit inside the cap).

## Methods (as T2, minus Gibbs)

- **CS-F.** T2's analyst rule on the training data (lift > 1.2 or < 0.8, cap 5, greedy
  merge), each subset an MNL over its patterns with pattern constants, item covariates and
  the cross-price screen (p < 0.20). Lift computed after trip size (lift of residualised
  counts), since a raw-lift analyst would link everything; both lifts reported.
- **CS-O.** Coded subsets with blocks chosen from the MVP R (best ≤ 5 blocks by within-block
  |ρ|): the cap's cost given the best possible grouping.
- **IL.** Independent binary logits.
- **MVP-IFM.** `multivariate_probit` 0.2.3, linear margins in the shared covariates.
- **MVP-ML.** `mvp_fit` (GPU service), `se=False`.

Probit decisions scored by GHK-QMC, the engine as a second column (as T2).

## Decisions (no truth, so: agreement, then adjudication by holdout)

- **D-bundle.** Best pair and best triple by joint take (co-promotion / taco kit).
- **D-TURF.** Best 4 items by reach (trips buying ≥ 1), for a display.
- **D-recommend.** For each item A, the item B with the highest P(B | A) ("bought shells,
  suggest ..."): 9 decisions, the most joint-dependent of the set.

**Adjudication.** Where CS and MVP disagree, compare each model's predicted value of the
disputed quantity (e.g. P(both) for its pair vs the other's pair; P(B | A)) with the
holdout frequency, with a household cluster bootstrap CI (200 resamples of the holdout
only; no refits). A disagreement counts for the method the holdout supports at 95%;
otherwise it is "unresolved".

**Fit metrics.** Holdout pattern log-likelihood (2^9 = 512 patterns per trip); pair- and
triple-joint calibration on holdout trips; basket-size distribution.

## Fairness controls (absorbed from `joint_vs_proxy`)

Same data, covariates and screen for every arm. Two semi-synthetic truths built on the real
covariates, 10 replicates each:
- **S-MVP:** simulate trips from the MVP-ML fit. MVP should win, CS should flip where its
  cap cuts.
- **S-CS:** simulate from the CS-F fit (complementarity patterns). CS should win; MVP's
  flips here bound its misspecification cost.

If either control fails (the "right" method does not win under its own truth), the
real-data comparison is reported as inconclusive.

## Verdict rule

- **PASS:** the gate passes, controls pass, and on real data ≥ 1 decision differs with the
  holdout supporting MVP, and none with the holdout supporting CS; or the same decisions
  with MVP faster to specify plus fit.
- **KILL:** gate passes and controls pass, but decisions agree and CS is no slower; or the
  holdout supports CS on the disagreements.
- **INCONCLUSIVE:** otherwise (all disagreements unresolved, or a control fails).
- Blanks for you: the minimum holdout support level (default 95%) and whether one
  supported disagreement is enough (default yes, given 2 + 1 + 9 = 12 decisions).

## Run time (estimates, all fits on Modal)

| step | where | estimate |
|---|---|---|
| data build + gate (IFM, 9 items, ~150k trips) | local + 1 CPU container | 10 min |
| CS-F / CS-O / IL / IFM, real data | CPU containers | 5 min |
| MVP-ML, real data (150k × 9, 512 draws) | GPU service | 10–20 min |
| semi-synthetic: 20 datasets × 5 methods | CPU + GPU, parallel | 20–30 min |
| scoring (QMC over 512 patterns × ~3.7k holdout trips) + bootstrap | CPU containers | 10 min |

About 1 h wall in total, a few dollars of GPU. Same structure as T2 (`--eval-only`,
per-phase caches). The gate runs first and stops there if it fails.

## Open points

- A. **Cluster.** Taco (recommended) with cookout as fallback, or both as two clusters in
  one menu (17+ items; closer to a real category plan, longer run).
- B. **Heterogeneity proxy.** The household propensity covariate (recommended) vs none (the
  pooled model T2 used; ρ then absorbs household tastes).
- C. **Analyst lift.** Residualised after trip size (recommended) vs raw.
- D. **Kill blanks** in the verdict rule.

## Revisions during the build

- **Ground beef dropped (8 items).** Fresh ground beef is not coded separately in this panel
  (BEEF GRND/PATTY types are on 0.04% of baskets; all BEEF, 13%, is too generic to be a taco
  item). The gate still needs 6 of 8.
- **Holdout widened** from the last 2 trips to the last 25% of each household's trips (at
  least 2): 37,470 holdout trips instead of ~3,700. With 2 trips, pair joints near 0.3% would
  rest on ~11 holdout trips; now the rarest pair has 69.
- **Adjudication by realised value.** Where two methods pick different options, the holdout
  value of each pick is compared (e.g. observed P(both) for each method's pair), with a 95%
  household cluster bootstrap CI of the difference; the method whose pick is better on the
  holdout is supported.
- **Same covariates for every item in every arm.** Each item's index (probit margin or MNL
  item utility) is linear in all of X (7 log prices, trip size, 8 propensities); no cross-price
  screen. Lettuce's weekly price is flat (sd < 0.01) and is dropped.
- **Decision quantities from the averaged pattern distribution.** Each model's 2^8 pattern
  distribution averaged over the holdout trips (exact for coded subsets; 4,096 scrambled-Sobol
  latent draws per trip for probits); all decisions are functions of it.
