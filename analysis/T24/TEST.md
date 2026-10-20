# T24 · Trivago hotel shelves, a real panel with rich properties: MVP vs mixed logit; engine vs OVUS (2026-10-20)

Status: **done. B: on 24-dimensional shelf orthants from the fitted model, the engine at "high" is 2.4× faster
than batched OVUS, with a lower median |ΔP| (0.0011 vs 0.0036). "Highest" is the only method whose maximum share
error stays under the 0.005 floor. A: there is no error correlation beyond tastes. MVP gives 0.999 of the error
variance to the iid component, and MVP and the mixed logit differ by under 0.01 nats per choice, in opposite
directions on the two scores.**

## Why

T23 (Expedia) found no error correlation, but it could not separate a lasting taste from a shared error: one
choice per search, and only 7 hotel characteristics. Trivago fixes both. Users repeat across sessions, so a taste
that persists across a user's choices is told apart from an error component redrawn at each choice. Hotels carry
about 150 properties, so near-identical hotels can be recognised. If correlation beyond tastes exists in real
shelf choice, this design should find it.

Part B asks the engine's question on the same data: a new user's click probabilities on a shelf are 24-dimensional
orthants with a full covariance, at P of a few percent. That is the shelf-share use, against OVUS.

## Data and split

- **Source.** RecSys Challenge 2019 (trivago), the Kaggle mirror `pranavmahajan725/trivagorecsyschallengedata2019`
  (the original portal is gone). `train.csv`: session events over 1–6 Nov 2018, 1.59M of them clickouts.
  `item_metadata.csv`: 927k hotels, 157 properties. In `data/` (not committed); archived on S3.
- **Choices.** One per session: the session's last clickout (the competition's target). The shelf is the displayed
  list (25 hotels in almost all), the outcome the clicked hotel. 99.96% of displayed hotels have metadata.
- **Panel.** Users with at least 2 such sessions: 78,135 users. The last 8 sessions per user are kept. A random
  24,879 of them: 59,176 choices.
- **Split by user.** 19,903 users (47,368 choices) fit the models. 4,976 users (11,808 choices) are held out and
  scored with population probabilities.

## Variables

- **Shelf position** (no taste sd): log position and a slot-1 dummy. Trivago's order is not randomized, so position
  is a control; 35% of clicks are on slot 1.
- **Price**: log price minus the shelf's median log price, clipped at ±2.
- **Stars** (exact, else the "From n Stars" lower bound; 0 = unknown, with a flag), **rating** (ordinal 0–4 from
  the nested Satisfactory ⊂ Good ⊂ Very Good ⊂ Excellent flags, with a flag for none), **type** (Hotel, House /
  Apartment, other), and a **thin-listing** flag (3 or fewer properties: 25% of hotels list that few).
- **54 amenities**: every property on at least 10% of hotels, minus the star, rating and type flags and two flags
  that are unions of others.
- 64 coefficients; 62 with a taste sd.

## Models

All three are scored on the same held-out users.

| model | what it is | fit |
|---|---|---|
| MNL | pooled logit | scipy, local, 29 s |
| MXL | panel mixed logit: normal tastes with a diagonal sd, one draw per user shared by all the user's choices | simulated ML, R = 512 scrambled-Sobol draws per user, 11 min on an L4 |
| MVP | panel mixed MNP: the same tastes and the same draws; normal error = a component per band shared by the shelf's hotels at the same level + iid, total variance π²/6, shares free | GHK on each 24-dimensional choice orthant given the taste draw, 48 min on an A100 |

- **MVP bands.** Hotel type, star level, rating level, price tertile within the shelf, and an amenity cluster
  (k-means, k = 8, on the 54 amenity flags: hotels with near-identical property profiles).
- **Why simulated tastes, not one stacked orthant.** A user's 8 choices stacked are a 192-dimensional orthant with
  P around 1e-3 to 1e-9. Given a taste draw the choices are independent, so P = mean over draws of the product of
  24-dimensional GHK probabilities. That is an unbiased simulator of the same panel probability, at about 8× less
  cost than GHK on the stacked orthant. The engine is not used for fitting (T18, T21: it fails at this size and P).
- **Scores.** Per choice: log P(click) for a new user, tastes integrated (the market-simulator prediction). Per user:
  log P of the user's whole sequence, tastes shared (how well the taste structure fits). 95% CIs from 2,000
  bootstrap resamples of held-out users.

## A · Results (held-out users; positive = first model better)

| score (nats per choice) | MNL | MXL | MVP | MVP − MXL | MVP − MNL | MXL − MNL |
|---|---|---|---|---|---|---|
| per choice, tastes integrated | −2.4595 | −2.4601 | **−2.4549** | +0.0052 [+0.0037, +0.0066] | +0.0046 [+0.0001, +0.0095] | −0.0005 [−0.0055, +0.0046] |
| per user (sequence), per choice | −2.4595 | **−2.3722** | −2.3795 | −0.0073 [−0.0099, −0.0046] | +0.0800 [+0.0713, +0.0888] | +0.0874 [+0.0777, +0.0970] |

A uniform guess scores −3.10. For scale, 0.01 nats per choice is 0.024 nats per held-out user.

**MVP error shares** (of π²/6): type 0.001, star 0.000, rating 0.000, price tertile 0.000, amenity cluster 0.000,
iid 0.999.

**Largest taste sds** (MXL / MVP): relative price 1.36 / 0.99; House / Apartment 0.83 / 0.53; no rating 0.64 /
0.69; other type 0.69 / 0.45; no stars 0.60 / 0.44; Airport Shuttle 0.66 / 0.43.

- **No error correlation beyond tastes.** Every shared component is 0 at full N. A smoke run on 1,000 users gave
  the amenity cluster 0.088; at 19,903 users it is gone, so it was small-sample noise.
- **MVP vs MXL is a tie in substance.** The two significant gaps point in opposite directions and are both under
  0.01 nats per choice. With no shared components, MVP is a mixed probit with iid errors: the gap is the error
  shape (normal vs Gumbel), not correlation.
- **Tastes are what matter, and only for a known user.** Scoring a user's sequence, the mixed models beat MNL by
  0.08–0.09 nats per choice. For a new user (tastes integrated) they gain nothing over MNL: the taste spread
  averages out over the shelf.

## B · Engine vs OVUS on the fitted model (`engine.py`)

**Setup.**
- MVP's full-fit parameters. Held-out shelves with all 25 hotels, so every orthant is 24-dimensional: upper D V,
  covariance D (X W X' + Σ_err) D' (tastes integrated).
- Two sets of 200 orthants: *clicked* (the clicked hotel on 200 random shelves) and *shares* (every hotel on 8
  random shelves; each shelf's shares should sum to 1).
- Reference: GHK with 2^15 scrambled-Sobol points × 8 scrambles (T10's `ghk_ref`). Median SE of log P 3e-5.
- T10's harness: time per orthant is the best of 3 after a warm-up, on the local 2-core CPU. Engine: the paid local
  binary, including resolution "highest".

**Time per orthant (µs)**

| method | clicked | shares |
|---|---|---|
| engine highest | 305 | 300 |
| engine high | 95 | 92 |
| engine low | 31 | 29 |
| OVUS (torch batch) | 225 | 231 |
| OVUS (pybhatlib row loop) | 461 | 460 |

**Error** (median |ΔP| (max); shares also the largest |sum of a shelf's shares − 1|; the reference's own is 3e-5)

| method | clicked (median P 0.096) | shares (median P 0.020) | shares: max \|sum − 1\| |
|---|---|---|---|
| engine highest | 0.00024 (0.010) | 0.00008 (0.0016) | 0.004 |
| engine high | 0.0011 (0.019) | 0.00029 (0.0065) | 0.010 |
| engine low | 0.0031 (0.040) | 0.0013 (0.015) | 0.052 |
| OVUS | 0.0036 (0.021) | 0.0015 (0.016) | 0.016 |

- **Speed.** "High" is 2.4× faster than batched OVUS and 4.9× faster than the published row loop; "low" is 7×
  faster than batched OVUS.
- **Error.** Every method's median is under the 0.005 decision floor. "High" has a lower median |ΔP| than OVUS on
  both sets. Only "highest" keeps the maximum share error (0.0016) under the floor.
- **No zeros.** At these P (a few percent) no method returns P = 0, unlike T21's d = 21 basket patterns at P ≈ 5e-5.
- The smoke-fit parameters gave the same picture (high 90 µs, median |ΔP| 0.0013; OVUS 220 µs, 0.0029).

## Reading

- **A closes the question T23 left open.** With a real panel to separate tastes from errors and properties rich
  enough to mark near-identical hotels, real shelf choice shows no error correlation beyond tastes. That is the
  same answer as Swissmetro (T22) and Expedia (T23). The large MVP gaps of T1, T5 and T20 come from simulated
  truths with error correlation of 0.90–0.97 that is separate from tastes; this data has none.
- **B is the engine's clearest real-data result.** On 24-dimensional shelf-share orthants, it is faster than OVUS
  and more accurate at the same time. It is used here as the engine is meant to be used: population shares and
  decision queries, not likelihoods.
- **The ADVANTAGES shelf-CBC row rests on simulation.** On real shelves the model gain over a mixed logit is nil;
  what remains is the engine's speed in share scoring and screening.

## Files

- `summary.py`: step-0 profile of the clickouts (streamed; 9 s, 1.1 GB peak).
- `t24.py`: `prep` builds `out/arrays.npz` from `out/clickouts.parquet` and the metadata (62 s, 1.6 GB peak);
  `local` fits MNL and writes `out/local.json`.
- `t24_modal.py`: `smoke` (1,000 / 500 users, L4) and `full` (MXL on an L4, MVP on an A100). Writes
  `out/{mxl,mvp}_{smoke,full}.json`.
- `t24_score.py`: `python analysis/T24/t24_score.py full` prints the tables in A.
- `engine.py` (B): `PYTHONPATH=<multivariate_probit src whose binary has resolution "highest"> python analysis/T24/engine.py`,
  about 2 minutes. Writes `out/engine.json`.
- Raw data, `clickouts.parquet` and `arrays.npz` are not committed (S3 archive).
- Cost: about $2 of Modal time (full: L4 11 min + A100 48 min; smoke: two L4 runs under 4 min).
