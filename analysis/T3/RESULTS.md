# T3 results: real basket data, taco-night cluster (2026-10-08)

Run: `python analysis/T3/run.py` (dunnhumby Complete Journey, 1,869 households with ≥ 20 trips,
149,789 trips; the last 25% of each household's trips, 37,470, held out). 8 items, all fitted on
Modal: CS/IL/IFM on CPU containers, MVP-ML on the deployed `mvp_fit` GPU service (no SEs);
10.4 min wall. Raw: `out/full_taco/` (`results.json`, `log.txt`). Summary:
`python analysis/T3/summarize.py`. No errors; MVP-ML converged on the real data and 20/20
controls. Design and its build revisions: DESIGN.md.

## Verdict

**PASS (MVP better).** The gate passed, both controls passed, and on the real data the
coded subsets and the MVP disagree on 6 of 11 decisions; the holdout supports MVP on 4 and the
coded subsets on none (2 unresolved). Same verdict for IFM and for full ML. No time saving:
CS-F 29 s, MVP-IFM 36 s, MVP-ML 105 s (fit + decisions).

| | decisions = holdout's own pick (of 11) | holdout LL / trip | fit + decide |
|---|---|---|---|
| CS-F (analyst subsets) | 5 | −0.8569 | 29 s |
| CS-O (subsets from the MVP correlations) | 5 | −0.8519 | 33 s |
| IL (independent logits) | 7 | −0.8845 | 3 s |
| MVP-IFM | 9 | −0.8341 | 36 s |
| MVP-ML (GPU service) | 10 | −0.8336 | 105 s |

## Gate: the linked set is the whole cluster

After trip size, prices and household propensity, every pair is still linked: residual lifts
1.3–8.2, IFM correlations 0.13–0.76 (median 0.37; ML the same to 0.01). All 8 items are
connected at ρ ≥ 0.2, so the 5-item cap must cut real dependence. The analyst rule groups
{shells, seasoning, salsa, beans, sour cream} + {cheese, chips, lettuce}; raw lifts give the
same split.

## Q1: real-data decisions and the holdout

MVP-ML against CS-F. Holdout value of each pick; 95% household cluster bootstrap CI of the
difference (200 resamples):

| decision | MVP-ML pick | holdout | CS-F pick | holdout | verdict |
|---|---|---|---|---|---|
| best pair | sour cream + cheese | 1.49% | cheese + chips | 1.46% | unresolved |
| best triple | salsa + cheese + chips | 0.39% | seasoning + salsa + sour cream | 0.23% | MVP |
| shells → ? | cheese | 52.6% | seasoning | 41.9% | MVP |
| seasoning → ? | cheese | 59.3% | sour cream | 39.4% | MVP |
| salsa → ? | chips | 43.0% | sour cream | 24.8% | MVP |
| cheese → ? | sour cream | 18.9% | chips | 18.5% | unresolved |

(Recommendation = the item B with the highest P(B | A).) TURF-4 and 5 of 8 recommendations
agree across all methods.

- **The cap's signature.** Every CS-F error recommends or bundles inside its own subset:
  shells → seasoning, seasoning → sour cream, salsa → sour cream, cheese → chips. Cheese is
  the most common complement of shells and seasoning, but it sits in the other subset, which
  CS treats as independent, so CS can't see it. That is the T2 failure on real data.
- **Knowing the structure doesn't fix it.** CS-O (blocks chosen from the MVP's own
  correlations) disagrees with MVP on 5 decisions and the holdout sides with MVP on all 5
  (e.g. best pair: cheese + lettuce at 0.92% vs 1.49%).
- **Coded subsets do worse than independence here.** IL matches the holdout's own pick on 7 of
  11, CS-F on 5: tying items into a subset inflates within-subset joints relative to every
  cross-subset joint.
- Holdout log-likelihood: MVP −0.834 vs CS-F −0.857 vs IL −0.885 per trip.

## Controls (10 replicates per truth, simulated on the real covariates)

Flips of the truth's decisions (of 110), mean regret over all decisions:

| truth | CS-F | CS-O | IL | MVP-IFM | MVP-ML |
|---|---|---|---|---|---|
| S-MVP (from the MVP-ML fit) | 60 (15.0%) | 53 (13.5%) | 40 (5.1%) | 2 (0.06%) | 2 (0.06%) |
| S-CS (from the CS-F fit) | 2 (0.03%) | 2 (0.03%) | 40 (9.4%) | 9 (0.19%) | 10 (0.21%) |

- Both controls pass: each method wins under its own truth.
- Under S-MVP, CS-F flips the same 6 decisions in 10 of 10 replicates (pair, triple, and
  the recommendations for shells, seasoning, salsa and cheese). Those are exactly the 6 it
  disagrees with MVP on in the real data, and 4 of those 6 are the real-data disagreements
  where the holdout sides with MVP.
- Under S-CS, MVP's misspecification cost is concentrated in the best triple (flipped in 9 of
  10, regret 1.7% each); every other decision holds. This is the same pattern as T2's LC guard.
  The holdout prefers MVP's triple on the real data (0.39% vs 0.23%), which points to the
  real dependence being closer to the probit than to the CS pattern constants.

## Q2: IFM vs full ML

Same decisions except salsa → ? (IFM: cheese; ML: chips), a near tie on the holdout (CI of
the difference spans 0). The same flips in the controls. IFM is 3× faster.

## Q3: time

No time saving at 8 items, as in T2: CS-F needs 29 s (IL + lifts + subset MNLs), MVP-IFM
36 s, MVP-ML 105 s on the hosted service. Analyst time isn't timed. CS needs a lift table and
a grouping rule; MVP needs neither.

**Engine column.** Computing the probit decisions per trip with the engine took 44 s against
5–7 s for the QMC pattern simulation, with the same decisions except IFM's salsa near tie.
Here the engine is slower: decisions averaged over 37k heterogeneous trips need
5.7M low-dimensional orthants, whereas one simulated pattern distribution answers every
decision at once. The engine's advantage (T2, ~500×) is for a few high-dimensional queries,
not many trips.

## Caveats

1. Revealed data: prices are weekly and promotion-driven, so no price decisions; the
   decisions are co-promotion, display (TURF) and recommendation.
2. "Supported by the holdout" means a later period of the same households. It is not a field
   test. 2 of 6 disagreements are unresolved.
3. Heterogeneity enters only as a fixed household propensity covariate (all arms); no random
   effects (tracked follow-up).
4. The analyst rule (lift > 1.2 or < 0.8, cap 5) is ours, as in T2; lift computed beyond the
   covariates (raw lift gives the same subsets).
5. Build revisions (DESIGN.md): 8 items (ground beef not coded), 25% holdout, adjudication by
   realised value, all covariates in every item's index.
