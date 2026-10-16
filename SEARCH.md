# Searching huge decision spaces: speed, error, and what "best" means

As of 2026-10-14. How we search assortment, bundle, and TURF spaces too large to enumerate, and how we judge the
methods that score them. Model catalog: [MODELS.md](MODELS.md). Test results: [PLAN.md](PLAN.md).

## The problem

Choosing 10 items out of 1,000 gives C(1000, 10) ≈ 2.6 × 10^23 candidate sets. No method scores all of them. Every
practical search samples the space, scores the sample, and keeps the winners.

Scoring a set means computing a probability from the fitted model, for example reach = 1 − P(buys none of the set).
Under a correlated (probit) model, that is one multivariate normal orthant probability per candidate.

## What "best" means here

- **The guaranteed best is not on offer.** Even a perfect scorer only finds the best of its sample. With n random
  candidates, the best of the sample sits around the top 1/n of the space: the top 10^-9 at a billion candidates,
  10^-10 at ten billion.
- **Ties within the materiality floor are ties.** We treat differences under 0.5 points of share or reach as
  immaterial. That floor is fixed before each test. Telling a client that basket A beats basket B by 10^-4 (0.01
  points) would be a mistake: it is below the floor and below the noise in any real data.
- **So the target is a probability.** The question is: how likely is the final pick to land within the floor of
  the best candidate in the sample? We don't ask whether a method finds the exact top 10.

## Absolute error, not ratios

Approximations are compared by absolute error |ΔP| against a high-accuracy reference, set against the floor (0.5
points = 5 × 10^-3).

Ratios mislead. An error of 10^-3 is "10× worse" than 10^-4, but both are far below the floor, and neither can
change a decision the floor protects. If the cheaper method is 10× faster, a ratio of errors hides the trade that
matters.

## At a fixed budget, speed is sample size

With a fixed compute budget, a method that scores candidates k× faster searches a sample k× larger. A larger sample
moves the best candidate deeper into the tail of the space. For search, scoring speed usually buys more than the
last digits of accuracy.

## The cascade

Spend the cheap method where the volume is, and the accurate method where the decision is:

| Stage | Scorer | Candidates | Job |
|---|---|---|---|
| 1 | Fastest deterministic approximation | 10^9 to 10^10 sampled sets | Keep the top 1% |
| 2 | A slower, more accurate approximation | 10^7 to 10^8 | Keep the top 1% |
| 3 | Simulation (quasi-Monte Carlo) | 10^5 to 10^6 | Final ranking |

- Only stage 1 grows with the sample. Going from 10^9 to 10^10 candidates multiplies stage-1 time by ten and leaves
  stages 2 and 3 alone, so stage 1 is where scoring speed pays.
- A stage only has to keep good candidates above its cut, not order them. An error of a few 10^-4 can only reorder
  candidates that are already tied within the floor.
- Each stage can be checked directly. Score a small sample exhaustively by simulation, then measure, for each
  stage, how often a candidate within the floor of the best survives its cut. The product across stages is the
  chance the final pick is within the floor.

## Evidence so far

- **TURF search (T7).** The GPU engine's TURF picks had zero regret, apart from 0.06 points once, in greedy search
  at k = 10. A shared-draw simulator was still faster at small k, so the engine's speed case rests on larger
  spaces like the ones above.
- **Screen then verify (ORTHANT_PLAN S1).** Screening with the engine and rescoring the top 1%, 5% or 10% by
  simulation had zero regret on all five test truths, for both assortment and price decisions. The engine's error
  was mostly systematic, so it barely moved the ranking: rank correlation with the reference was 0.999–1.000. Its
  one miss on its own was a 0.5% regret, at the reference's own noise level, on a near-clone (two items correlated at 0.98): it picked the clone. Rescoring
  the top 1% fixed it.
- **Cost per orthant (T10).** On one T4 GPU at batch 10^5, µs per orthant at d = 6 / 20 / 40: engine 1.9 / 25 / 32,
  OVUS 0.72 / 4.3 / 28. Faster engine settings exist for screening. Their cost and accuracy on real basket
  correlations are not published yet.

## Not yet measured

- Per-stage survival rates for the cascade at d = 10 (10-item sets), on a fitted basket model.
- End-to-end search quality at 10^9 and 10^10 candidates against a simulation-only search with the same budget.
