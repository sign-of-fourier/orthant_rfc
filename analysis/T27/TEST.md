# T27 · The engine as an MNP likelihood: fitting, same-hardware speed, near-substitute nests

Status 2026-10-21: exploratory (no kill criteria; user-directed quick checks). All runs local, one CPU core, $0.

## Question

Can the engine replace simulation inside a multinomial probit (MNP) fit? Three parts:

- **Speed and scale:** on the same hardware, what does one MNP likelihood evaluation cost against OVUS (the
  approximation inside MACML), GHK and SciPy as the choice set grows?
- **Accuracy guard:** how far is an engine fit from the GHK maximum-likelihood estimate (MLE)?
- **Near-substitutes:** what happens as two or three alternatives become near-perfect substitutes (nest
  correlation near 1), and can that case be handled without asking the analyst to drop alternatives?

## Setup

- Cross-sectional MNP: U_ij = x_ij'β + e_ij, one choice per row, differenced against the chosen alternative, so
  each row is one orthant of dimension J − 1. Three linear coefficients, error covariance by Cholesky factor
  (first variance fixed for scale): 9 covariance parameters at J = 5.
- Error structures:
  - **iid**: differenced correlations all 0.5.
  - **random Ω**: Ω = AA'/J + 0.5I from a random A; differenced correlations 0–0.75.
  - **nested pairs**: two 2-alternative nests with within-nest correlation r, unequal scales.
  - **3-alternative nest**: J = 6, nests {2, 3, 4} and {5, 6} at r.
- Reference: GHK with 2^14–2^16 scrambled-Sobol draws.
- Engine resolutions: "low", "high", "highest" (the public names). A prototype setting with smooth near-duplicate
  handling, and an exact rewrite of analyst-declared nests, were also tested (both private, not in any release).
- Bias metric: the score-implied first-order bias of θ̂, −I⁻¹ (mean engine score − mean GHK score) at the true θ,
  in units of the GHK SE at N = 1,000. This bias does not shrink with N, so 0.3 SE at N = 1,000 is about 1 SE at
  N = 10,000.

## Results

### Speed and scale: one likelihood evaluation, same CPU core (J − 1 = d)

Value only, 200 MNP choice rows per d, random Ω (correlations concentrate near 0.5 as d grows, a favourable case
for the engine). Median |ΔP| vs GHK 2^16.

| d | engine "highest" | OVUS | GHK 256 | GHK 1,024 | GHK 4,096 | SciPy |
|---|---|---|---|---|---|---|
| 4 | 49 µs, 2.0e-4 | 104 µs, 5.0e-4 | 48 µs, 5.5e-5 | 208 µs, 2.7e-5 | 0.87 ms, 1.6e-5 | 3.3 ms, 1.2e-6 |
| 8 | 81 µs, 2.5e-4 | 174 µs, 1.1e-3 | 103 µs, 2.7e-4 | 434 µs, 6.1e-5 | 1.9 ms, 1.4e-5 | 24 ms, 1.8e-6 |
| 16 | 191 µs, 2.6e-4 | 326 µs, 1.9e-3 | 232 µs, 8.3e-4 | 1.1 ms, 2.2e-4 | 4.4 ms, 4.6e-5 | 108 ms, 6.0e-6 |
| 32 | 437 µs, 7.4e-5 | 630 µs, 2.5e-3 | 508 µs, 1.7e-3 | 2.4 ms, 3.2e-4 | 10 ms, 1.0e-4 | 506 ms, 1.1e-5 |
| 64 | 1.4 ms, 3.9e-5 | 1.4 ms, 2.4e-3 | 1.5 ms, 1.9e-3 | 5.7 ms, 6.7e-4 | 24 ms, 1.2e-4 | 1.9 s, 3.3e-5 |

- **vs OVUS:** the engine is 1.5–2.1× faster up to d = 32, equal at d = 64, and more accurate at every d (OVUS's
  error grows from 5e-4 to 2.4e-3; its 90th-percentile log error reaches 0.23 at d = 64).
- **vs GHK at matched accuracy:** GHK 256 wins at d = 4, ties at d = 8; from d = 16 GHK needs 6–23× the engine's
  time, and at d = 64 GHK 4,096 (17× the time) is still 3× less accurate.
- **vs SciPy:** 3.3 ms to 1.9 s per row (a 1,000-row likelihood at d = 64 is about 31 min per evaluation).
- "high" is 2.5–4× faster than "highest" and 5–8× less accurate.

### Accuracy guard: score bias and an actual fit (J = 5, N = 1,000)

| truth | "low" | "high" | "highest" |
|---|---|---|---|
| iid | 0.59 | 0.00 | 0.00 |
| random Ω | 2.53 | 0.82 | 0.11 |
| nested pairs, r = 0.8 | 8.32 | 1.86 | 0.51 |

- "low" is wrong even when its probabilities are right (iid: |Δlog P| about 1e-6): its gradient ignores
  departures from the mean correlation.
- **Actual fit, "highest":** L-BFGS from β = 0 and iid errors converged in 35–36 iterations (46–52 s with
  finite-difference gradients, numpy). The engine estimate is within **0.33 SE** (random Ω) and **0.28 SE**
  (nested, a third of the rows with mixed-sign correlations) of the GHK MLE; three BHHH steps of GHK from the
  engine estimate moved no parameter more than 0.07.

### Near-substitutes (nest correlation near 1)

- **The release engine's near-duplicate step has a fixed cutoff, and the likelihood jumps there.** On a smooth
  scan of one correlation the jump is 4e-4 in P at the cutoff; in a finite-difference score it showed up as a
  77 SE artefact. Moving the cutoff does not remove it: below the cutoff the bias is 0.47 SE (r = 0.95), and
  raising it above 0.97 makes rows past it 10× worse (0.5–0.8 SE).
- **3-alternative nests at r = 0.97 are the worst case: 5.04 SE**, all from near-1 rows.
- **Prototype + exact nest rewrite** (nests declared by the analyst; smooth, no cutoff):

| case | engine "highest" | prototype | prototype + nest rewrite | OVUS |
|---|---|---|---|---|
| random Ω | 0.09 | 0.07 | — | 0.21 |
| nested pairs, r = 0.9 | 0.18 | 0.19 | 0.21 | 0.42 |
| nested pairs, r = 0.97 | 0.09 | 0.68 | **0.06** | 0.45 |
| nested pairs, r = 0.99 | 0.09 | 0.85 | **0.11** | 0.40 |
| 3-alternative nest, r = 0.9 | 0.58 | 0.28 | **0.17** | 5.35 |
| 3-alternative nest, r = 0.97 | 5.04 | 1.07 | **0.14** | 0.96 |

  - Median |ΔP| with the rewrite: 7e-6 to 1.2e-4, against 4e-4 to 3e-3 for the release engine.
  - The rewrite is exact (it matches SciPy to 1e-8 with SciPy as the inner integrator). It costs one engine call
    per combination of nest winners (the product of nest sizes: 2–6 here), and each term is the probability that
    a given nest member wins against the rest of the market, so within-nest shares come out of the fit.
  - Cost of the prototype alone vs "highest": 0.88× at d = 6, 0.95× at 12, 1.24× at 32, 1.46× at 64.

### Ornery matrices (values, n = 6 and 12, 150 rows per family)

- Families: a near-duplicate pair on a weak background, two pairs, a 3-cluster near 1 and in the middle band, AR(1)
  chains, low-rank with mixed signs, frustrated sign blocks, one tiny eigenvalue, tail limits.
- **The engine fails a likelihood guard on sign-frustrated and low-rank small-P families:** P ≤ 0 on 11–97 of 150
  rows (every engine setting, prototype included), where OVUS and GHK never return 0 and are 10–100× more
  accurate. These are MVP-like matrices; MNP choice orthants are mostly positive.
- **OVUS breaks on AR(1) chains:** median |ΔP| 4–6e-3, 10–20× the engine.
- **Smoothness:** the prototype (no cutoffs), OVUS and fixed-draw GHK are smooth on both scans; the release engine
  jumps at its cutoffs.

## Open items

- **Warning:** at r = 0.99 the prototype raised an exp overflow in an intermediate that the sum does not use;
  values and scores were all finite. Needs a guard.
- **Unexplained:** the nest rewrite fed into the release engine is clearly worse than into the prototype (0.54 and
  1.35 SE at r = 0.97 / 0.99), though no near-1 pairs remain after the rewrite.
- **Near-1 tracking:** every bias above is split by rows with max |corr| ≥ 0.97 at the truth; the large results for
  the prototype alone (0.68, 0.85, 1.07) and for "highest" on the 3-alternative nest (5.04) come entirely from
  those rows.
- **Middle band:** nested pairs at r = 0.9 remain at about 0.2 SE in every engine setting.
- **Scale of the rewrite:** the product of nest sizes grows multiplicatively; fine for a few declared nests.
- The P ≤ 0 failure on frustrated / low-rank matrices needs its own fix.

## Caveats

- One dataset per cell, J = 5–6, N = 1,000; scores by finite differences; the bias is a linear (first-order)
  estimate except in the two actual fits.
- Cross-sectional MNP only. Panel likelihoods with respondent random effects (T18's 240-dimensional orthant)
  stay with GHK.
- The nested BHHH polish was not fully converged (last step up to 0.1 SE).
- CPU timings are numpy on one core; no GPU comparison was run.

## Code

The scripts call engine internals and are kept out of the public repo.
