# T36 · Near-clone shelves: engine vs OVUS as the MNP likelihood (2026-10-24): KILL as specified; revised engine passes Stage A

## Why

Near-clone line extensions are the cannibalization question CBC simulators get. RFC handles exact clones, so this is
not a claim against the incumbent; it is an engine claim against OVUS (the other analytic orthant approximation)
on shelves with near-clone families. T27 found the engine's bias at r ≈ 0.97 depended on the release's
near-duplicate setting (5 SE at the default, 0.14 with it off); correlations above 0.97, families above 3 and
decisions were untested.

## Design

- **Data:** T27's J = 10 MNP design (pooled tastes, free error covariance) with one clone family of size 2, 3 or 5
  at r = 0.97, 0.99, 0.995 (9 cells). N = 20,000 per cell, seeded.
- **Metric (Stage A):** max bias/SE of the mean score at the true parameters on one large sample (no fits); noise
  about 0.22 SE. Speed on one CPU core, same sample.
- **Arms:** engine order 2 with the near-duplicate setting off, engine order 2 at the release default, OVUS (the
  MACML approximation, `pybhatlib`).
- **Stage B** (screening on a J = 20 shelf with a clone family) runs only if Stage A does not KILL.

## Kill criteria (proposed by Claude, accepted by the user 2026-10-24)

Stage A:
- **PASS:** engine (setting off) within OVUS + 0.3 SE in every cell and no non-finite rows.
- **KILL:** in any cell at r ≥ 0.99, worse than OVUS by more than 0.5 SE, or a non-finite value.
- **Else INCONCLUSIVE.**

## Results (2026-10-24): KILL as specified; a post-hoc engine revision passes all 9 cells

Runs live in the private engine repo (the test harness calls the engine directly). $0, local CPU.

Max bias/SE against the truth (noise about 0.22 SE):

| r | family | engine, setting off | engine, release default | OVUS | revised engine (post hoc) |
|---|---|---|---|---|---|
| 0.97 | 2 | 0.42 | 0.45 | 0.56 | 0.42 |
| 0.97 | 3 | 0.72 | 0.92 | 0.54 | 0.72 |
| 0.97 | 5 | 0.75 | **2.24** | 1.09 | 0.75 |
| 0.99 | 2 | 0.43, **1 non-finite** | 0.45, 1 non-finite | 0.56 | 0.44, 0 non-finite |
| 0.99 | 3 | 0.64 | 0.83 | 0.54 | 0.62 |
| 0.99 | 5 | 0.69 | 0.82 | 0.73 | 0.69 |
| 0.995 | 2 | 0.46 | 0.46 | 0.59 | 0.45 |
| 0.995 | 3 | 0.63 | 0.63 | 0.63 | 0.60 |
| 0.995 | 5 | 1.03 | 0.99 | 1.08 | 1.03 |

- **Verdict: KILL as specified.** One row in 20,000 (r 0.99, family 2) returned a non-finite score for both engine
  settings. Accuracy otherwise passes: the engine is within OVUS + 0.3 SE everywhere (worst +0.18, r 0.97 family 3)
  and better than OVUS in 6 of 9 cells.
- **The failing row is not a near-clone problem.** True P 1.627e-4 (scipy at full precision); OVUS 1.654e-4; engine
  order 2 returns 0. The matrix is well conditioned (smallest eigenvalue 0.29): 8 coordinates correlated at about
  0.6 and a ninth nearly independent of them (0.05) with a far bound. A block plus a near-independent part is where
  the expansion around one common correlation breaks; the same shape appears in MACML's pair orthants (T38).
- **Speed:** engine about 190 s per cell against OVUS's 365 s (1.9×), same core, same sample.
- **The release default is the wrong setting on near-clones:** 2.24 SE at r 0.97 family 5 against 0.75 off.
- **Post-hoc revision.** A private engine change aimed at block-structured matrices (designed after seeing this
  failure; not in any release) removes the non-finite row and passes all 9 cells at 1.7× OVUS's speed. On 200
  sampled rows of the affected kind (r 0.99 family 2) its median |ΔP| against scipy is 1.9e-5, against 1.5e-4 for
  order 2 and 1.5e-3 for OVUS. Because it was designed on this data, it is not a PASS of T36; a fresh run (Stage B
  or new seeds) would confirm it. Whether it fits cleanly (it uses a per-row structure that must be held fixed
  during a fit) is being tested.
- **Stage B not run** (Stage A killed).

## Follow-up: the revised engine on fresh seeds (2026-10-24)

Same 9 cells, same rule, Modal CPU (under $3 in all). The revision has one tuning parameter; we have no opinion
on its value yet.

| seed | setting | non-finite rows | vs OVUS (max bias/SE) | time per cell |
|---|---|---|---|---|
| 101 | as on seed 1 | 1 (r 0.97, family 2) | within OVUS + 0.3 in all 9; worst +0.11 | 122–190 s vs OVUS 231–485 s |
| 101 | wider (chosen after seeing that row) | 0 | within in all 9; worst +0.09 | 125–181 s vs 216–477 s |
| 102 | wider | 0 | within in all 9; worst +0.09 | 119–180 s vs 218–432 s |

- **Seed 101's row** has the same shape as seed 1's: a well-conditioned block plus a near-independent coordinate
  with a far bound (true P 3.7e-5; OVUS 2.25e-5; the engine 0). The revision as first set missed it; the wider
  setting computes it at 3.6864e-5 against scipy's 3.6867e-5.
- **Seed 102 cannot confirm robustness:** the unrevised engine had no non-finite row on it either.
- **No setting guarantees a positive P.** That needs a positivity guard: OVUS on rows where the engine returns P ≤ 0 (OVUS is
  positive by construction).

### Confirmation with the guard (2026-10-25): revised engine PASSES Stage A

Revised engine at the default setting (user, 2026-10-24; a midpoint, not a tuned value) plus the guard (OVUS on
any row with P ≤ 0).

| seed | cells | non-finite rows (revised / unrevised) | rows guarded | vs OVUS (max bias/SE) | time per cell |
|---|---|---|---|---|---|
| 103 (fresh) | all 9 | 0 / 1 (r 0.97, family 2) | 0 | within OVUS + 0.3 in all 9; worst +0.16 (r 0.995, family 3) | 162–204 s vs OVUS 364–441 s |
| 101, first setting (forces the guard) | r 0.97, family 2 | 0 / 1 | the 1 bad row, in 94 of 95 passes | 0.58 vs OVUS 0.46 (+0.12); unrevised with the row dropped 0.57 | 167 s vs 416 s |

- **Verdict for the revised engine: PASS on Stage A** (fresh seed, the user's rule). T36 as specified stays KILL.
- **On seed 103 the revision alone handled the hard row**; the guard did not fire. On seed 101 the guard
  carried the row it exists for, at no measurable accuracy cost.
- **Families of 5:** OVUS 0.85–1.03 SE against the engine's 0.61–0.83.
- **Stage B** (screening on a J = 20 shelf) is not run; the claim covers the MNP likelihood, not the decision.
- **Accuracy and speed hold.** On families of 5, OVUS is at 1.04–1.27 SE against the engine's 0.48–0.69, and the
  engine is about 2× faster in every cell.
