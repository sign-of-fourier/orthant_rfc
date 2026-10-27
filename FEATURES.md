# Features

Last updated 2026-10-27 (through T37). What the product does, in the buyer's terms. A feature is a thing in itself;
the evidence behind it is the test list on each one, and a test can back more than one feature.
[ADVANTAGES.md](ADVANTAGES.md) holds the technical comparisons; [USE_CASES.md](USE_CASES.md) holds the decisions
and the two tests a case must pass against the incumbent.

Scope: stated-preference conjoint (CBC, shelf CBC, menu-based choice). "Sawtooth" means Sawtooth and Conjointly as
they ship today, run their recommended way (HB, randomized first choice, simulation over posterior draws where
documented). Every result below is simulation against a known truth unless tagged otherwise; no real stated-shelf
study with a holdout has been run yet.

**Status.** **Shown**: ahead of Sawtooth's best practice in our tests. **Partial**: ahead in some tests, tied or
untested in others. **Same as Sawtooth**: works, but Sawtooth's best practice gets the same answer. **Untested**: we
expect it, nothing measures it yet.

## At a glance

| feature | what you get | status |
|---|---|---|
| [Learns how shoppers switch](#learns-how-shoppers-switch) | switching measured from the survey, no decision tree drawn up front | **shown** |
| [Better at near-duplicates](#better-at-near-duplicates) | variants of the same product split share instead of stacking it | **shown** |
| [Forecasts you can take to a hurdle](#forecasts-you-can-take-to-a-hurdle) | the volume or profit level is right, not just the ranking | **shown** on forecast error; hurdle flips untested |
| [Full-shelf studies](#full-shelf-studies) | 20–60 SKUs at once, estimates in minutes | **shown** at 20, partial at 60, untested at 100+ |
| [Fewer respondents for small studies](#fewer-respondents-for-small-studies) | right answer at 50 respondents where the alternative plateaus | **partial** |
| [Large add-on menus](#large-add-on-menus) | 20–30 linked menu items, no cap on combinations | **same as Sawtooth** |
| [Range and placement search](#range-and-placement-search) | thousands of candidate lines a minute, across store clusters | **partial** (borderline) |

## Learns how shoppers switch

**What you get.** When a shopper's product is missing, the model knows where they go: to the same flavor, the same
pack, the same brand, or anywhere. It learns this from respondents' choices. You don't draw a consumer decision tree,
declare nests, or pick which attributes count for similarity before the study.

**Sawtooth today.** HB estimates what each respondent likes. How they switch comes from a rule: randomized first
choice treats products as similar if they share attribute levels, with the strength tuned to holdout shares. The rule
can't learn that shoppers stay within flavor but not within pack.

**Evidence.** Range of k of 20: 1.85 pts lower regret [1.30, 2.40] against the documented readout (T15); still
1.74 pts [1.09, 2.38] against simulation over posterior draws, with the k = 6 pick wrong on every dataset (T34).
Two logit models fitted to the choices also learn it and come within 0.27–0.35 pts (T37); neither ships in
Sawtooth or Conjointly.

## Better at near-duplicates

**What you get.** Line extensions, flavor variants and pack sizes that shoppers see as near-substitutes split share
between them, so adding a variant doesn't inflate the line's total.

**Sawtooth today.** Exact clones are handled by randomized first choice. Products that are close but not identical,
and close on some attributes but not others, get the rule's similarity, not the shoppers'.

**Evidence.** Same tests as above, on a truth where 80% of the error variance is shared by brand, flavor and pack
(T15, T34). At 60 SKUs the documented readout misplaces 1.6–2.2× more share than we do; simulation over posterior
draws closes that on shares (T31). Not claimed for exact clones.

## Forecasts you can take to a hurdle

**What you get.** Go / no-go and ROI decisions depend on the forecast's level, not only which option ranks first.
Our profit and volume forecasts land within about 3% of the truth.

**Sawtooth today.** Simulation over posterior draws forecasts 11–20% low on the range test (T34) and 7–9% low under
correlated switching at 60 SKUs (T31). The usual remedy is calibrating to sales, which a new launch doesn't have.

**Evidence.** T31, T34; the fitted logits that tie us on picks miss the level (−8 to +18%, T37). Open: no test yet
shows a hurdle decision flipping.

## Full-shelf studies

**What you get.** Test the shelf as shoppers see it, 20–60 SKUs at once, with switching learned across all of them.
Estimates come back in minutes at cents per run: 0.35–2.2 min at 20 SKUs, 3.7 min at 60.

**Sawtooth today.** HB itself handles large shelves in minutes. Learning switching at that size did not exist as a
practical option: the same estimate took about 4.5 h at 20 SKUs before (T18). That is why the industry uses a rule.

**Evidence.** T18 (20 SKUs), T31 (60 SKUs). At 60 SKUs the picks tie simulation over posterior draws; the gain there
is the forecast level. Beyond 60 SKUs untested.

## Fewer respondents for small studies

**What you get.** For small choice sets (up to 6 options, e.g. price or cost trade-offs), the right answer from
about 50 respondents.

**Sawtooth today.** HB with randomized first choice has a regret floor of about 5 pts that more respondents don't
lower; over posterior draws it halves to about 3.5 pts and stays flat.

**Evidence.** T20; T35 (ahead on one of two truths, +4.5 pts, p 0.008; not significant on the other). On real
stated data with no error correlation (Swissmetro, T22) the gain is negligible. Not tested against a structural
logit fitted to the choices.

## Large add-on menus

**What you get.** Menu-based choice with 20–30 linked items (options, add-ons, bundles), with every pair's
dependence estimated, and no cap on the number of combinations.

**Sawtooth today.** Coded subsets cap out near 5 items. For large menus Sawtooth recommends serial cross-effects
models, which tie ours (49 vs 50 flips of 240, T33).

**Evidence.** Ahead of coded subsets by 118–157 flips of 240 (T2, T14); tie against the recommended large-menu
model (T33).

## Range and placement search

**What you get.** An optimizer that scores thousands of candidate lines a minute, across store clusters, on the
switching-aware model.

**Sawtooth today.** Search on logit share of preference or randomized first choice.

**Evidence.** 0.35 / 0.75% regret against 0.91 / 2.32% (logit) and 2.49 / 3.60% (randomized first choice) at a
fixed 300 s (T25). Every gap is under 3 $ per 1,000 trips, two seeds, and the truth was estimated from purchase
data [RP]. Right direction; materiality unshown.
