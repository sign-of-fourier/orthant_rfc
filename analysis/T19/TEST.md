# T19 · MVP on real takeaway baskets: joint basket probabilities and bundle cannibalization (2026-10-19)

Status: **done. M3 (MVP) beats near-independence (M0) on every metric, by a wide margin. M3 does not beat the
MBC-style cross-effects logit (M1).** On all orders, the two models tie on the decision test. On small orders and with the
size covariate, M1 is better on pair and bundle error, and it has the higher joint log-likelihood everywhere.

## Question

On real revealed restaurant baskets, does a multivariate probit (correlated pick-any) estimate joint basket
probabilities, and the "already buys the whole bundle" rate that drives bundle-discount cannibalization, better
than models that treat items as near-independent? This is the data half of a QSR menu-bundle demo for restaurant
pricing consultancies.

## Data

- **Source.** Kaggle `henslersoftware/19560-indian-takeaway-orders`: two UK Indian takeaways, order lines with
  timestamp, item, quantity and unit price. Unzipped into `data/` (not committed; `data/` is gitignored).
- **Unit.** One order. Each item group is binarized to "ordered at least once".
- **Window.** Restaurant 1 from 2016-03-01 and restaurant 2 from 2016-05-01, both to 2019-08-03. This drops the
  sparse ramp-up months before regular trading: 29 orders in 2015 for restaurant 1 (with a 4-month gap after
  them) and 1 order for restaurant 2.
- **Split.** By date: the first 75% of trading dates train, the last 25% are holdout. The cut is 2018-10-07 for
  restaurant 1 and 2018-10-27 for restaurant 2 (all orders; the small-order cuts fall a month later).

## Step 0 · Data checks (all pass; nothing to stop on)

**1. Plausibly real POS data: yes.**

| | Restaurant 1 | Restaurant 2 |
|---|---|---|
| line rows / orders | 74,818 / 13,397 | 119,183 / 19,658 |
| first / last order | 2015-09-01 / 2019-08-03 | 2015-12-16 / 2019-08-03 |
| orders per trading day (from start): mean, p10 / p50 / p90 | 10.7: 3 / 9 / 21 | 16.5: 4 / 13 / 36 |
| days with zero orders (from start) | 51 of 1,251 | 71 of 1,190 |
| orders by year 2016 / 17 / 18 / 19 (to Aug) | 1,872 / 3,673 / 4,553 / 3,270 | 2,536 / 5,857 / 6,848 / 4,416 |
| share of orders 17:00–20:59 | 81% | 84% |
| share Fri + Sat (Mon–Thu each) | 46% (9–11%) | 53% (7–10%) |
| prices on a 5p grid | 100% | 100% |
| line price = menu-file price | 98.7% | 98.8% |
| orders with more than one timestamp | 0 | 0 |

There is an evening peak (18:00 is the busiest hour), a Friday/Saturday peak and steady volume growth, with
prices at UK takeaway levels (£0.50 condiments, £2.50–2.95 breads, £7.95–12.95 mains, £17.95 wine). The order
value floor sits near £15 (p5 = £15.85 and £16.35), which looks like a delivery minimum. In April–May 2018 the
menu was restructured: generic curry names ("Korma") were retired and protein variants appeared ("Korma -
Chicken"). That is 23 and 35 items last seen before 2018-06, and 64 and 136 items first seen after 2018-03. The
group mapping below absorbs the restructure, because both name forms map to the same curry type.

**2. Multi-item orders: yes.** 97.5% (restaurant 1) and 97.6% (restaurant 2) of orders have 2 or more distinct
items. The modal order has 4–5 distinct items, and the mean is 4.85 and 5.27 *groups* per order.

**3. Price variation: none over time.** No item's modal price differs between the first and the last 6 months
(155 and 161 items compared). 13 items (restaurant 1) and 14 (restaurant 2) carry more than one recorded price.
Every one of them is a generic pre-restructure curry name ("Bhuna", "Korma", "Madras", "Curry", …). Its price
points (£7.95/8.95 vs £8.95/9.95 vs £11.95/12.95) are protein variants sold at the same time: the price switches
back and forth hundreds of times across the same dates, until the name was retired in April 2018. The one
exception is restaurant 1's "Chicken Tikka": 2 lines at £3.95 and £7.95 on 2015-09-29, before the analysis
window. **There are no natural experiments.** Also, prices that never move in 3.4 years, together with a 98.7%
match to a static menu file, suggest that the price column may be today's menu price joined onto old lines
rather than the price charged. Either way, price response cannot be estimated from this data. The full list,
with every price point and its date span, is in `out/step0.json` (`multi_price_items`).

**4. Restaurants.** There are two independent restaurants with similar menus. Each is analysed separately with
the same group rules.

## Step 1 · Prep

### Item groups (21; rule-based, `classify()` in `t19.py`)

The 248 and 337 item names collapse into 6 entrée groups, 13 bundle-side groups and 2 add-on groups. The rules
use the name and, where the name is ambiguous, the price. For example, "Chicken Tikka" at £4.95 is a starter and
"Chicken Tikka (Main)" at £8.95 is a main; "Bombay Aloo" at £5.95 is a side and "Bombay" at £8.95 is a curry. The
full item → group tables are in `out/mapping_r1.csv` and `out/mapping_r2.csv`.

| group | role | what goes in (examples) | orders r1 / r2 |
|---|---|---|---|
| MILD_CURRY | entrée | tikka masala, korma, butter chicken, pasanda, masala dishes | 6,410 / 9,707 |
| MEDIUM_CURRY | entrée | curry, bhuna, dhansak, dupiaza, rogon, pathia, saag, methi (all proteins) | 4,545 / 6,883 |
| HOT_CURRY | entrée | madras, vindaloo, jalfrezi, chilli masala, hari / lal mirch | 3,768 / 6,467 |
| HOUSE_SPECIAL | entrée | balti, karahi, named house dishes (rezala, mysore, achar, royal, garlic) | 2,331 / 3,720 |
| BIRYANI | entrée | all biryanis | 2,411 / 3,106 |
| TANDOORI_MAIN | entrée | tikka / tandoori (main), mixed grill, shashlik, sizzlers | 4,213 / 5,935 |
| VEG_SIDE | side | £5.95 vegetable sides (bombay aloo, saag aloo, tarka dall, …), £6.95 veg curries, salads | 7,659 / 12,664 |
| RICE_PILAU | side | pilau rice | 4,721 / 8,370 |
| RICE_PLAIN | side | plain rice | 2,369 / 2,799 |
| RICE_SPECIAL | side | mushroom, special fried, keema, egg, garlic, lemon … rice | 4,047 / 6,647 |
| NAAN_PLAIN | bread | plain naan | 3,752 / 6,679 |
| NAAN_FLAV | bread | garlic, peshwari, keema, kulcha, onion naan | 5,873 / 8,965 |
| BREAD_OTHER | bread | chapati, roti, paratha, puree | 2,364 / 3,001 |
| PAPADUM | side | plain and spicy papadum | 4,052 / 7,022 |
| FRIES | side | french fries | 524 / 854 |
| STARTER_FRIED | side | onion bhaji, samosa, pakora, roll, butterfly | 4,297 / 8,027 |
| STARTER_GRILL | side | tikka / kebab / tandoori starters, mixed starter, lamb chops | 2,384 / 3,638 |
| STARTER_OTHER | side | puree starters, chaat, garlic mushroom | 1,174 / 2,067 |
| DRINK | drink | soft drinks, Cobra, wine, water | 1,017 / 1,343 |
| CONDIMENT | add-on | chutneys, sauces at £0.50, pickle, raita | 5,862 / 9,931 |
| SAUCE | add-on | £3.95–4.95 side curry sauces | 834 / 1,076 |

**Why this grouping.**
- **Entrées by heat and style.** Mild, medium and hot is how these menus are laid out and how people substitute
  (one curry per person, chosen by heat). Biryani is separate because it is a one-dish meal that replaces rice.
  Tandoori mains are separate because they are dry dishes that pull in a sauce.
- **Rice and naan split by type.** Pilau vs plain vs special rice, and plain vs flavoured naan, are the
  substitution pairs that matter for a "curry + rice + naan" deal.
- **Starters split into fried, grill and other.** These have distinct price points and pairings.
- **What the groups do not carry.** Protein (chicken, lamb, prawn) is not a bundle dimension here. Splitting by
  it would also bring back the 2018 menu restructure as a time break.
- **Add-ons.** CONDIMENT and SAUCE are modelled, because they carry dependence (papadum + chutney), but are left
  out of the bundle components: no one builds a meal deal around a 50p chutney.
- **No desserts.** Neither menu has any.

### Order-size handling

- **(a) all.** All orders.
- **(b) small.** Order value ≤ £20: one person's meal (main £8.95 + rice £2.95 + naan £2.60–2.95 + papadum and
  chutney ≈ £16, with room for a second side but not a second main). This is 18.5% and 14.8% of orders. "Total
  items ≤ 3" was the other candidate, but it keeps only 10–15% and would drop a single meal with a papadum and
  chutney.
- **(c) all_size.** All orders plus a standardized log(total quantity) covariate in every model.

### Covariates (all models)

- Hour dummies: before 17:00, 17:00, 19:00 and 20:00 or later, with 18:00 as the reference.
- Day dummies: Friday, Saturday and Sunday, with Monday–Thursday as the reference.
- Variant (c) only: log quantity.

There is no time trend, so the train → holdout drift in the marginals hits every model equally.

## Step 2 · Models (fit on train)

- **M0, independent item logits.** One logit per group on the covariates. Every joint probability is a product of
  marginals.
- **M1, MBC-style cross-effects logits, fit as sequential conditionals.** Group order[k] is a logit on the
  covariates plus the 0/1 presence of the groups earlier in the order. The order is train frequency, descending
  (VEG_SIDE, MILD_CURRY, NAAN_FLAV, RICE_PILAU, … for restaurant 1). This is the chain rule, so the joint is
  proper: P(y) = ∏ P(y_k | y_<k, x) holds exactly, and holdout baskets are scored exactly.
  - **Why this is not a straw man.** It has the same 210 pairwise dependence parameters as M3's correlation
    matrix, a logit link with weak L2 (C = 100, only to stop separation) and no restriction on the sign or size
    of the cross effects.
  - **Order sensitivity.** A reversed-order fit (M1 rev) is reported throughout. Marginal and joint event rates
    come from 400 ancestral draws per holdout order.
  - **What differs from Sawtooth MBC.** MBC's cross-*price* effects are absent because prices never vary (Step 0).
- **M2, coded-subset multinomial.** A multinomial logit over all 2⁵ = 32 combinations of 5 groups, with
  covariate effects specific to each combination (31 × 8 parameters).
  - **Which 5 groups.** The 5 most frequent groups overall include MILD_CURRY but also MEDIUM_CURRY, so I used
    the most frequent entrée plus the 4 most frequent bundle sides. That makes 10 of the 546 bundles scoreable;
    the per-cell 5-group lists are in the sub-basket table below.
  - **Scope.** M2 covers 5 of the 21 groups. 21 groups would need 2²¹ ≈ 2.1 M cells, which is why the
    coded-subset approach stops at about 5.
- **M3, MVP over all 21 groups.**
  - **Fit.** `multivariate_probit` 0.2.3, `MultivariateProbit(inner="linear", dependence="pairwise", cv=5)`.
    This is IFM: probit margins on the same covariates, then the correlation by composite pairwise likelihood on
    cross-fitted indices and the library's projection to positive-definite.
  - **Shrinkage (s = 0.05).** The projected matrix lands on the PD boundary (minimum eigenvalue 1e-8 in every
    cell, along a curry-vs-rice direction). That makes the 21-dimensional GHK weights degenerate: the four
    scrambles' summed holdout log-likelihood ranged from −45k to −637k on restaurant 1. M3 therefore uses
    R ← 0.95 R + 0.05 I. With it, the four scrambles agree to within 9 nats out of 45,000.
  - **How s was chosen.** On GHK stability alone, using 300 holdout rows of restaurant 1. It was not tuned on
    any accuracy metric.
  - **Unshrunk version.** It is reported as M3 raw. It changes pair and bundle errors by ≤ 0.008 pts.
  - **Scoring.** Joint basket probabilities use GHK with 4 × 1,024 scrambled Sobol points (my own code in
    `ghk_logp`). Pair, triple and bundle probabilities use the library's closed-form bivariate CDF and
    Gauss-Legendre quadrature (d ≤ 3). **The proprietary engine is not used anywhere.**

## Step 3 · Evaluation (holdout)

**Uncertainty.** The bootstrap resamples holdout orders (B = 1,000; Spearman uses every 5th replicate) with the
models held fixed, so the CIs cover holdout sampling only, not estimation. Paired differences between models
are the inferential quantities. A per-model MAE "CI" is biased upward (|error| on resampled data is inflated), so
those are given as points only. All errors are absolute, in percentage points of orders.

### Samples

| restaurant | variant | train orders | holdout orders | groups / order (train) |
|---|---|---|---|---|
| 1 | all, all_size | 8,829 | 4,539 | 4.85 |
| 1 | small | 1,654 | 799 | 3.01 |
| 2 | all, all_size | 13,915 | 5,742 | 5.27 |
| 2 | small | 1,987 | 913 | 2.95 |

### Pairwise and triple co-occurrence

Mean absolute error in pts. For differences, positive means M3 is better.

| r | variant | pairs (210): M0 / M1 / M3 | M0 − M3 | M1 − M3 | triples (top 50): M0 / M1 / M3 | M0 − M3 | M1 − M3 |
|---|---|---|---|---|---|---|---|
| 1 | all | 0.94 / 0.54 / 0.55 | +0.39 [+0.31, +0.43] | −0.010 [−0.019, −0.002] | 1.79 / 0.64 / 0.58 | +1.21 [+0.80, +1.47] | +0.06 [−0.02, +0.10] |
| 1 | small | 0.98 / 0.56 / 0.66 | +0.31 [+0.25, +0.36] | −0.106 [−0.118, −0.070] | 1.32 / 0.60 / 0.62 | +0.70 [+0.41, +0.82] | −0.02 [−0.05, +0.03] |
| 1 | all_size | 0.93 / 0.54 / 0.59 | +0.34 [+0.28, +0.39] | −0.048 [−0.066, −0.025] | 1.04 / 0.54 / 0.65 | +0.39 [+0.26, +0.48] | −0.11 [−0.15, −0.03] |
| 2 | all | 1.01 / 0.64 / 0.66 | +0.36 [+0.29, +0.43] | −0.012 [−0.016, +0.001] | 2.40 / 0.72 / 0.65 | +1.75 [+1.27, +2.10] | +0.06 [−0.01, +0.11] |
| 2 | small | 1.09 / 0.66 / 0.77 | +0.32 [+0.25, +0.35] | −0.109 [−0.134, −0.095] | 1.15 / 0.79 / 0.81 | +0.34 [+0.14, +0.44] | −0.02 [−0.04, +0.02] |
| 2 | all_size | 1.07 / 0.62 / 0.67 | +0.39 [+0.34, +0.43] | −0.052 [−0.066, −0.029] | 1.29 / 0.65 / 0.74 | +0.55 [+0.40, +0.69] | −0.09 [−0.13, −0.02] |

- **Triple selection.** Triples are the top 50 by *train* frequency, so the holdout does not choose its own test
  set.
- **M0 vs the rest.** Independence underpredicts the frequent complements: mild curry + pilau, and papadum +
  chutney (ρ = +0.74/+0.79). Its triple error is 2–3× that of M1 and M3.
- **M1 vs M3.** They are close on all orders, and M1 is ahead on small orders and with the size covariate.

### Joint basket log-likelihood (nats per order, 21 groups)

| r | variant | M0 | M1 | M1 rev | M3 | M3 − M0 | M3 − M1 |
|---|---|---|---|---|---|---|---|
| 1 | all | −10.483 | −9.929 | −9.932 | −9.977 | +0.505 [+0.475, +0.536] | −0.049 [−0.058, −0.041] |
| 1 | small | −8.525 | −7.485 | −7.482 | −7.907 | +0.617 [+0.531, +0.702] | −0.422 [−0.479, −0.370] |
| 1 | all_size | −9.925 | −9.290 | −9.268 | −9.398 | +0.527 [+0.487, +0.560] | −0.109 [−0.128, −0.091] |
| 2 | all | −10.653 | −10.033 | −10.033 | −10.078 | +0.574 [+0.544, +0.603] | −0.045 [−0.052, −0.037] |
| 2 | small | −7.828 | −6.883 | −6.858 | −7.282 | +0.545 [+0.463, +0.626] | −0.399 [−0.454, −0.344] |
| 2 | all_size | −10.066 | −9.357 | −9.336 | −9.451 | +0.614 [+0.586, +0.640] | −0.095 [−0.106, −0.082] |

- **M3 vs M0.** M3 gains 0.5–0.6 nats per order (a likelihood ratio of about 1.7× per basket).
- **M1 vs M3.** M1 gains another 0.05 nats on all orders and 0.4 on small orders.
- **Unshrunk M3.** Its GHK scrambles disagree by up to 10× (Step 2), so its log-likelihood is not usable.

**5-group sub-basket (M2's subset, all four models).**

| r | variant | M2 groups | M0 | M1 | M2 | M3 | M3 − M2 |
|---|---|---|---|---|---|---|---|
| 1 | all | mild, veg side, naan flav, pilau, naan plain | −3.259 | −3.231 | −3.191 | −3.191 | −0.001 [−0.007, +0.006] |
| 1 | small | mild, naan flav, papadum, starter fried, pilau | −2.583 | −2.626 | −2.704 | −2.585 | +0.119 [+0.071, +0.172] |
| 2 | all | mild, veg side, pilau, naan flav, naan plain | −3.306 | −3.245 | −3.198 | −3.209 | −0.011 [−0.018, −0.005] |
| 2 | small | hot, naan plain, starter fried, pilau, veg side | −2.562 | −2.568 | −2.699 | −2.530 | +0.169 [+0.095, +0.246] |

- **M3 vs M2.** On the 5 groups M2 can see, M3 matches M2 when train is large (all orders) and beats it on small
  orders (2,000 train baskets for 248 parameters).
- **M1's figures here are approximate.** M1 needs the other 16 groups marginalized out. That is done with 400
  draws per order, smoothed by 0.5 counts, which biases its value down.
- **all_size.** Same pattern as all (`out/report.md`).

### Decision test: the "already buys the whole bundle" rate

- **Candidates.** 546 bundles: each of the 6 entrée groups with every 1 or 2 of the 13 side, bread and drink
  groups.
- **The rate.** For each bundle, it is the share of holdout orders containing every group in it.
- **Model predictions.** The mean over holdout orders of the model's P(all groups | covariates).

| r | variant | MAE, all 546: M0 / M1 / M1 rev / M3 | M0 − M3 | M1 − M3 | MAE on observed top 10: M0 / M1 / M3 | top-10 overlap: M0 / M1 / M3 | Spearman: M0 / M1 / M3 |
|---|---|---|---|---|---|---|---|
| 1 | all | 0.44 / 0.30 / 0.30 / 0.29 | +0.150 [+0.102, +0.175] | +0.007 [+0.001, +0.012] | 1.93 / 0.65 / 0.62 | 8 / 9 / 9 | .955 / .968 / .968 |
| 1 | small | 0.35 / 0.23 / 0.23 / 0.27 | +0.084 [+0.062, +0.099] | −0.036 [−0.045, −0.029] | 2.76 / 1.30 / 1.40 | 7 / 8 / 7 | .710 / .811 / .794 |
| 1 | all_size | 0.46 / 0.29 / 0.29 / 0.31 | +0.153 [+0.120, +0.171] | −0.026 [−0.041, −0.006] | 1.24 / 0.68 / 0.63 | 8 / 9 / 9 | .962 / .968 / .969 |
| 2 | all | 0.46 / 0.33 / 0.33 / 0.32 | +0.145 [+0.095, +0.186] | +0.008 [+0.003, +0.014] | 1.73 / 1.24 / 1.20 | 9 / 9 / 9 | .967 / .976 / .976 |
| 2 | small | 0.40 / 0.22 / 0.21 / 0.28 | +0.122 [+0.093, +0.128] | −0.056 [−0.066, −0.050] | 2.93 / 1.41 / 1.29 | 4 / 7 / 6 | .606 / .733 / .690 |
| 2 | all_size | 0.54 / 0.31 / 0.31 / 0.35 | +0.193 [+0.163, +0.209] | −0.037 [−0.050, −0.016] | 1.53 / 1.17 / 1.18 | 9 / 9 / 9 | .970 / .975 / .977 |

**Reading the decision test.**
- **M3 vs M0: M3 wins, and the gain is material.**
  - The MAE gain is 0.08–0.19 pts over all 546 bundles, where most bundles are rare and every model is near 0.
    It is 0.35–1.6 pts on the 10 most common bundles, which are the ones a consultant would actually price.
  - On small orders, ranking also improves: top-10 overlap 7 vs 4 for restaurant 2, and Spearman 0.69–0.79 vs
    0.61–0.71.
  - On all orders the ranking is easy (8–9 of 10 for every model), because bundle popularity is driven by the
    marginals.
- **M3 vs M1: no material difference, and M1 is better on small orders.**
  - **All orders.** M3 is ahead by 0.007–0.008 pts. That is statistically significant and immaterial: less than
    one order in 10,000.
  - **Ranking.** Top-10 overlap and Spearman are identical.
  - **Small orders and the size covariate.** M1 is better by 0.03–0.06 pts. M1 rev is no worse than M1, so this
    is not an artefact of the order.
- **M2.** It is competitive on the 10 bundles inside its subset, where all models have MAE 0.5–1.8 pts
  (`out/report.md`), but it says nothing about the other 536.
- **Unshrunk M3 (M3 raw).** Bundle MAE within 0.008 pts of M3.

**Why M1 catches up.** The two models have the same number of dependence parameters. A probit with a single
correlation matrix forces every pairwise dependence through one Gaussian copula. The chained logit is more
flexible about higher-order structure, such as "exactly one rice", or the budget constraint that the ≤ £20
filter imposes. That filter makes small baskets strongly negatively dependent in a way a Gaussian latent
structure fits poorly: the small-order log-likelihood gap is 0.4 nats. With the size covariate, part of the
dependence moves into the margins, and the remaining residual structure suits M1 better too.

## Step 4 · Outputs

### Correlation heatmaps (M3, train, all orders)

- `out/corr_r1_all.png` and `out/corr_r2_all.png`. Blue means substitutes and red means complements. The lines
  separate entrées, sides and add-ons.
- **Structure.**
  - Entrées are mutually substitutive (−0.01 to −0.25), and biryani is negative with every rice group.
  - Rice types substitute for each other, and so do naan types.
  - Mild curry is the dish that pulls rice and naan.

**Top 10 complements and substitutes (M3, ρ).**

| r1 complement | ρ | r1 substitute | ρ | r2 complement | ρ | r2 substitute | ρ |
|---|---|---|---|---|---|---|---|
| papadum + condiment | +0.74 | naan plain + naan flav | −0.34 | papadum + condiment | +0.79 | naan plain + naan flav | −0.42 |
| fries + sauce | +0.34 | rice pilau + rice special | −0.34 | mild curry + rice pilau | +0.34 | rice pilau + rice special | −0.32 |
| mild curry + rice pilau | +0.29 | rice pilau + rice plain | −0.29 | rice special + naan flav | +0.30 | biryani + rice special | −0.31 |
| rice pilau + naan plain | +0.26 | biryani + rice pilau | −0.27 | mild curry + naan flav | +0.26 | rice pilau + rice plain | −0.30 |
| rice pilau + papadum | +0.25 | rice plain + rice special | −0.26 | rice plain + naan plain | +0.22 | biryani + rice pilau | −0.25 |
| mild curry + naan plain | +0.24 | mild curry + biryani | −0.25 | rice pilau + naan plain | +0.21 | naan flav + bread other | −0.23 |
| mild curry + naan flav | +0.23 | biryani + rice plain | −0.22 | fries + starter grill | +0.21 | biryani + veg side | −0.23 |
| tandoori main + sauce | +0.22 | biryani + rice special | −0.22 | tandoori main + sauce | +0.21 | hot curry + biryani | −0.22 |
| naan plain + fries | +0.21 | naan flav + bread other | −0.21 | papadum + fries | +0.20 | medium curry + house special | −0.21 |
| rice plain + naan plain | +0.20 | mild curry + tandoori main | −0.21 | medium curry + rice special | +0.20 | rice special + naan plain | −0.20 |

The two restaurants show the same structure, which suggests the correlations are menu-generic rather than
noise.

### Bundle table: top 10 by observed holdout full-bundle rate (%, all orders)

Error = model − observed, in pts. All 546 bundles per cell are in `out/bundles_r<r>_<variant>.csv`.

**Restaurant 1**

| bundle | observed | M0 | M1 | M3 | M0 err | M1 err | M3 err |
|---|---|---|---|---|---|---|---|
| mild curry + naan flav | 19.1 | 16.7 | 20.6 | 20.2 | −2.4 | +1.5 | +1.1 |
| mild curry + rice pilau | 18.5 | 15.1 | 20.2 | 19.4 | −3.4 | +1.6 | +0.9 |
| mild curry + veg side | 16.7 | 18.2 | 16.7 | 16.9 | +1.4 | −0.1 | +0.1 |
| mild curry + papadum | 14.3 | 11.7 | 14.3 | 14.1 | −2.6 | −0.0 | −0.2 |
| mild curry + naan plain | 13.9 | 12.0 | 15.4 | 15.2 | −1.9 | +1.6 | +1.3 |
| medium curry + rice pilau | 13.7 | 10.8 | 13.0 | 12.6 | −2.9 | −0.7 | −1.2 |
| medium curry + naan flav | 13.6 | 11.9 | 13.4 | 13.2 | −1.7 | −0.2 | −0.4 |
| medium curry + veg side | 13.1 | 13.0 | 13.3 | 13.3 | −0.1 | +0.2 | +0.2 |
| mild curry + rice special | 12.9 | 11.6 | 13.1 | 12.7 | −1.3 | +0.2 | −0.2 |
| hot curry + naan flav | 11.6 | 10.0 | 11.2 | 11.1 | −1.6 | −0.4 | −0.5 |

**Restaurant 2**

| bundle | observed | M0 | M1 | M3 | M0 err | M1 err | M3 err |
|---|---|---|---|---|---|---|---|
| mild curry + rice pilau | 21.7 | 18.0 | 23.9 | 23.2 | −3.7 | +2.2 | +1.5 |
| mild curry + naan flav | 19.8 | 17.7 | 22.2 | 21.7 | −2.1 | +2.3 | +1.9 |
| mild curry + veg side | 19.6 | 21.4 | 20.8 | 21.0 | +1.8 | +1.2 | +1.4 |
| hot curry + veg side | 15.9 | 15.3 | 16.0 | 16.0 | −0.6 | +0.1 | +0.1 |
| medium curry + rice pilau | 15.9 | 12.5 | 13.8 | 13.4 | −3.4 | −2.1 | −2.5 |
| mild curry + rice special | 15.6 | 13.2 | 15.4 | 14.9 | −2.4 | −0.2 | −0.7 |
| medium curry + veg side | 15.2 | 14.9 | 14.7 | 14.8 | −0.3 | −0.5 | −0.4 |
| hot curry + rice pilau | 15.1 | 12.9 | 15.0 | 14.6 | −2.2 | −0.1 | −0.4 |
| mild curry + papadum | 14.9 | 14.4 | 15.7 | 15.6 | −0.5 | +0.9 | +0.7 |
| mild curry + naan plain | 14.9 | 14.7 | 17.6 | 17.3 | −0.2 | +2.7 | +2.5 |

- **M0's error pattern.** It underpredicts the curry + rice and curry + naan bundles by 2–4 pts, because it
  cannot see the complementarity.
- **M1 and M3.** They overpredict the mild-curry bundles by 1–2.5 pts. Dependence in the holdout year is a
  little weaker than in train, so this is a drift that neither model can see.

### Account-lead illustration (current prices only; no price response is estimated)

- **The candidate.** In both restaurants it is **mild curry + pilau rice + flavoured naan**, the most common
  entrée + 2-side combination in the holdout. List price at current menu prices is £8.95 + £2.95 + £2.95 =
  £14.85, and a 15% bundle discount is £2.23.
- **Would-be bundle buyers.** Taken as the orders that already contain a mild curry: 39.5% of restaurant 1's
  holdout orders and 39.0% of restaurant 2's.

| | already buy all three, given a mild curry: r1 | discount given to them per 100 mild-curry orders: r1 | already buy all three: r2 | discount per 100: r2 |
|---|---|---|---|---|
| observed (holdout) | 23.3% | £52.00 | 29.5% | £65.62 |
| M0 (independent) | 15.7% | £35.07 | 19.5% | £43.51 |
| M1 (cross-effects) | 28.1% | £62.62 | 33.3% | £74.08 |
| M3 (MVP) | 26.8% | £59.60 | 31.9% | £71.01 |

> **Account lead (restaurant 1).** Roughly one in four people who order a mild curry today already buy pilau rice
> and a flavoured naan with it (23% in the holdout year). An independent-items model puts that at 16%. Planning
> on that number would understate by a third the margin a 15% "curry + rice + naan" deal gives away to customers
> who would have paid full price: £35 rather than about £52 per 100 mild-curry orders at current prices. The MVP
> says 27% (£60 per 100), which errs on the cautious side by 3.5 pts. The cross-effects logit (M1) says 28%. So
> the case for modelling dependence is clear, and the case for MVP over a well-specified cross-effects logit is
> not, on this data. All of these figures are at current prices. Nothing here estimates how many *new* bundle
> buyers the discount would recruit, or how it would shift item choice. That needs price variation this data
> does not have.

## Limitations

- **No price variation.** No item's price moved over 3.4 years (Step 0), and the price column may be a menu join.
  Cannibalization is computed at current prices only. Bundle uptake, incremental volume and cross-price effects
  are not identified, so this is not yet a bundle-pricing model.
- **No non-buyers and no no-purchase option.** Every observation is an order, so P(entrée) is a share of orders,
  not of visitors or occasions. Penetration and the decision to order at all are invisible.
- **Group orders.** Multi-person orders inflate co-occurrence: the all-orders basket averages about 5 groups. The
  small-order filter (≤ £20) removes most of them but selects on the outcome, since value is a function of the
  basket. That filter imposes a budget constraint that suits M1's structure. The size covariate is also
  mechanically tied to the basket. Neither variant is a clean "one diner" sample, and the data has no party-size
  field.
- **Two restaurants with near-identical menus,** from one platform, both in the UK. The structure replicates
  across the two, but they are not independent tests of the method in other cuisines or QSR formats.
- **Drift.** The holdout year (Oct 2018 onward) is entirely after the April 2018 menu restructure, and in it the
  dependence is slightly weaker than in train. There is no trend term in any model.
- **Inference scope.** The bootstrap covers holdout sampling with the models held fixed, not estimation
  uncertainty. M1's event rates and its sub-basket log-likelihood are Monte Carlo (400 draws per order).
- **M3 specifics.**
  - The pairwise-IFM correlation needed shrinkage (s = 0.05) to make 21-dimensional GHK stable.
  - Full-information ML (the library's `fitter="modal"`) was not tried and might change the picture.
  - Some probit margins are weakly identified by the 7 time dummies: calibration slopes are 0.15–0.33 for
    house special, hot curry, biryani and bread other in restaurant 1. The other groups are at 0.56–0.97.

## Files and reproduction

- **`t19.py`**, the single entry point, seed 19.
  - `python t19.py` runs Step 0, then all 6 restaurant × variant cells, then `report`.
  - `python t19.py step0` runs the data checks only.
  - `python t19.py run R V` runs one cell.
  - `python t19.py report` writes the tables to `out/report.md`.
- **Outputs (`out/`).**
  - `step0.json`: data checks, including every multi-price item with its date spans.
  - `mapping_r1.csv` and `mapping_r2.csv`: item → group.
  - `results_r<r>_<variant>.json`: all metrics with CIs, the M3 correlation, M1 cross-effects, GHK per-scramble
    sums and calibration slopes.
  - `bundles_r<r>_<variant>.csv`: all 546 bundles, observed and per model.
  - `corr_r<r>_all.png`: correlation heatmaps.
  - `report.md`: every table, including the M2-subset comparisons and the shrinkage check.
- **Runtime and cost.** About 13 min on the local CPU box (2 cores, peak memory about 650 MB). $0.
