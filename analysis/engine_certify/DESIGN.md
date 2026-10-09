# Engine-until-it-gets-bad: design (2026-10-09)

Follow-up to T4's gate. Accuracy only, local CPU, independent of T4.

## Question

Fit a probit with the engine (fast, fixed approximation error) while the covariance implied by
the current parameters is in the range where the engine is accurate; once a cheap indicator
computed from the parameters alone says it is leaving that range, finish with QMC (GHK with
scrambled Sobol, accurate given enough points). Does that give estimates that match a QMC-only
fit, and where along the correlation axis does the handoff happen?

- No per-fit QMC check: the indicator is a function of theta only (the J orthant covariances are
  the same for every respondent here), so it costs microseconds per iteration.
- The 0.97 merge is off (`dup_corr=0`, exact duplicates only); near-duplicates go to QMC.

## Model

T4's shelf structure shrunk so that only the orthant evaluator differs: J = 8 (2 brands x 2
flavors x 2 packs) and J = 12 (3 x 2 x 2) plus "none", no random tastes (the taste integral is
the same for both methods). Dummy-coded brand, flavor, pack, price, none constant (5 or 6) plus 3
error-share parameters (T4's softmax, total error variance fixed). N = 300, T = 8.

Truths (variance shares brand / flavor / pack / product):

| truth | shares | same flavor+pack corr |
|---|---|---|
| weak | .05 / .10 / .10 / .75 | .20 |
| medium | .10 / .25 / .25 / .40 | .50 |
| shelf (T4 S1) | .10 / .35 / .35 / .20 | .70 |
| near-dup | .05 / .45 / .45 / .05 | .90 |

4 truths x 2 J x 3 replicates = 24 datasets.

## Per dataset

1. Engine-only fit (local binary, `dup_corr=0`, central finite-difference gradients, L-BFGS),
   starting from weak correlation; the path (theta and the indicator per iteration) is kept.
2. QMC reference fit (GHK, R = 1,024 Sobol points, torch CPU) and sandwich SEs.
3. Engine error D_LR = sqrt(2 (nll_Q(theta_E) - nll_Q(theta_Q))), the QMC log-likelihood lost at the
   engine estimate in SE units (>= every per-parameter |z|; per-parameter z breaks when a share goes
   to 0, where both fits send its log-ratio to -inf). Also per-parameter z for b and share differences.
4. Calibration at the truth: per-choice log-probability error of the engine vs GHK at R = 8,192.
5. Indicator at theta: rho_max, the largest correlation in the J difference covariances.

## The rule and its check

rho* = the largest indicator value such that every engine-only fit ending at or below it has
D_LR <= 0.25. Then the hybrid (engine until rho_max > rho*, then QMC from that point) is run on all
24 datasets: it passes if every hybrid estimate is within 0.25 SE of the QMC reference. Also
reported: QMC iterations of the hybrid vs QMC from a cold start, and which truths stay entirely
on the engine.

If no useful rho* exists (the engine is off by > 0.25 SE even at weak correlation), that is the
answer: the indicator would have to be finer than rho_max (an error table in dimension and tail
depth), or the engine is not usable for estimation.

## Footprint

Local CPU, under 1 GB, chunked GHK; under an hour expected. No Modal.
