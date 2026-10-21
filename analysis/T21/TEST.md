# T21 · T19's real takeaway baskets, redone against the MBC incumbent and the engine's competitor (2026-10-19)

Status: **done. A: MVP beats coded subsets over all 21 groups in all 6 cells, and the gap is in events that span
blocks. Inside a block, coded subsets match or beat MVP. The cross-effects logit (M1) still ties MVP. B: on real
21-dimensional basket patterns, the engine is 3–4× faster than OVUS, but its error is about 10× larger, and at
resolution "high" it returns P = 0 for 4 of 200 patterns. It fails the accuracy guard for likelihood scoring.**

## Why

T19 compared MVP (M3) with independence (M0), a cross-effects logit (M1), and a coded-subset MNL over only 5
groups (M2). It never ran MBC's actual incumbent at k = 21: coded subsets that cover every group, capped at about
5 per block, with independence across blocks. Two ADVANTAGES rows rest on synthetic data or on dunnhumby (T3):

- "MBC beyond the coded-subset cap" and "Dependence across subsets" (A);
- "Scoring pattern probabilities on a near-singular Σ" (B). T19's pairwise-IFM R has a minimum eigenvalue of
  1e-8, so it is a real near-singular instance at d = 21, outside the benchmark's d = 8–16.

## A · Coded subsets over all 21 groups (`t21.py`)

**Setup.** T19's data, split, covariates, M0 / M1 / M3 fits (same seed; T19's numbers are reproduced exactly),
holdout and bundle list. The coded subsets are T3's code (`t3.CovMNL`): each block is an MNL over its 2^s
patterns, with item indices linear in the covariates and one constant per multi-item pattern.

- **CS-A, the analyst rule** (T3's CS-F): lift beyond the covariates outside [0.8, 1.2], greedy merge, cap 5.
- **CS-R, blocks from MVP's fitted R** (T3's CS-O): the best grouping the cap allows, given MVP's correlations.
  This is a generous incumbent, since it borrows MVP's estimate.

Both produce 4–6 blocks of 4–5 groups. The block lists are in `out/report_cs.md`.

**Metrics.**
- Joint holdout log-likelihood: exact for coded subsets, GHK for M3, as in T19.
- Co-occurrence MAE (pts of holdout orders) on 210 pairs, the top-50 triples and the 546 bundles. Each event set
  is split into events **inside one block** and events that **span blocks**.
- Bootstrap over holdout orders (B = 1,000), with paired differences against M3.

### Joint log-likelihood (nats per order; positive = M3 better)

| r | variant | CS-A | CS-R | M1 | M3 | M3 − CS-A | M3 − CS-R | M3 − M1 |
|---|---|---|---|---|---|---|---|---|
| 1 | all | −10.156 | −10.126 | −9.929 | −9.977 | +0.18 [+0.15, +0.20] | +0.15 [+0.13, +0.17] | −0.05 [−0.06, −0.04] |
| 1 | small | −8.015 | −8.133 | −7.485 | −7.907 | +0.11 [+0.04, +0.17] | +0.23 [+0.17, +0.28] | −0.42 [−0.48, −0.37] |
| 1 | all_size | −9.568 | −9.580 | −9.290 | −9.398 | +0.17 [+0.14, +0.20] | +0.18 [+0.15, +0.21] | −0.11 [−0.13, −0.09] |
| 2 | all | −10.340 | −10.254 | −10.033 | −10.078 | +0.26 [+0.24, +0.28] | +0.18 [+0.16, +0.19] | −0.05 [−0.05, −0.04] |
| 2 | small | −7.536 | −7.639 | −6.883 | −7.282 | +0.25 [+0.18, +0.33] | +0.36 [+0.30, +0.42] | −0.40 [−0.46, −0.34] |
| 2 | all_size | −9.660 | −9.684 | −9.357 | −9.451 | +0.21 [+0.19, +0.23] | +0.23 [+0.21, +0.25] | −0.10 [−0.11, −0.08] |

### Where the gap is: pairs inside a block vs pairs spanning blocks (CS-A blocks; MAE pts, positive = M3 better)

| r | variant | span: n | CS-A / M3 | CS-A − M3 | within: n | CS-A / M3 | CS-A − M3 |
|---|---|---|---|---|---|---|---|
| 1 | all | 174 | 0.76 / 0.53 | +0.23 [+0.15, +0.30] | 36 | 0.58 / 0.65 | −0.07 [−0.10, −0.06] |
| 1 | small | 173 | 0.84 / 0.65 | +0.19 [+0.11, +0.25] | 37 | 0.52 / 0.75 | −0.23 [−0.27, −0.13] |
| 1 | all_size | 174 | 0.77 / 0.58 | +0.19 [+0.13, +0.23] | 36 | 0.47 / 0.63 | −0.16 [−0.21, −0.07] |
| 2 | all | 174 | 0.89 / 0.67 | +0.22 [+0.15, +0.28] | 36 | 0.53 / 0.59 | −0.05 [−0.07, −0.00] |
| 2 | small | 173 | 1.02 / 0.79 | +0.23 [+0.15, +0.27] | 37 | 0.46 / 0.70 | −0.24 [−0.32, −0.20] |
| 2 | all_size | 174 | 0.92 / 0.68 | +0.24 [+0.19, +0.29] | 36 | 0.54 / 0.65 | −0.12 [−0.16, −0.05] |

- **Spanning pairs.** Coded subsets set the dependence to zero across blocks, so their error there equals
  independence (CS-A's spanning MAE equals M0's to 3 decimals). MVP is 0.19–0.24 pts better, every CI above 0.
- **Pairs inside a block.** The block's saturated pattern MNL beats MVP's single Gaussian copula by 0.05–0.24 pts.
  This is the same flexibility that lets M1 tie MVP.
- **CS-R.** Same pattern with its own blocks: +0.13 to +0.27 pts on spanning pairs, −0.03 to −0.13 inside. Using
  MVP's R to choose the blocks does not rescue coded subsets, because about 80% of pairs still span blocks.

### Decision test: bundles

| r | variant | all 546: CS-A − M3 | CS-R − M3 | observed top 10 (spanning CS-A / CS-R) | top 10: CS-A / CS-R / M3 | CS-A − M3 | CS-R − M3 |
|---|---|---|---|---|---|---|---|
| 1 | all | +0.09 [+0.04, +0.12] | +0.08 [+0.04, +0.10] | 10 / 8 | 1.92 / 1.64 / 0.62 | +1.31 [+0.64, +1.67] | +1.02 [+0.57, +1.26] |
| 1 | small | +0.02 [−0.00, +0.03] | +0.02 [+0.00, +0.04] | 6 / 7 | 2.19 / 1.70 / 1.40 | +0.79 [+0.43, +0.87] | +0.30 [−0.06, +0.51] |
| 1 | all_size | +0.06 [+0.03, +0.08] | +0.06 [+0.03, +0.08] | 9 / 9 | 1.22 / 1.22 / 0.63 | +0.59 [+0.12, +0.96] | +0.60 [+0.12, +0.96] |
| 2 | all | +0.11 [+0.07, +0.15] | +0.09 [+0.04, +0.12] | 10 / 8 | 1.72 / 1.36 / 1.20 | +0.53 [+0.02, +1.03] | +0.16 [−0.15, +0.53] |
| 2 | small | +0.06 [+0.03, +0.07] | +0.07 [+0.05, +0.08] | 9 / 7 | 2.56 / 2.08 / 1.29 | +1.27 [+0.71, +1.45] | +0.79 [+0.24, +1.01] |
| 2 | all_size | +0.12 [+0.09, +0.14] | +0.12 [+0.09, +0.14] | 9 / 8 | 1.54 / 1.41 / 1.18 | +0.36 [+0.02, +0.72] | +0.23 [+0.02, +0.47] |

- **The bundles a consultant would price.** The 10 most common entrée + side bundles mostly span blocks (6–10 of
  10). Examples are mild curry + pilau rice, and mild curry + flavoured naan. On these, coded subsets miss by
  1.2–2.6 pts against MVP's 0.6–1.4: worse by 0.36–1.31 pts (CS-A), with every CI above 0. CS-R is worse by
  0.16–1.02 pts, with 2 of 6 CIs touching 0.
- **Over all 546 bundles,** most of them rare, the gap is 0.02–0.12 pts.
- **M1 vs M3 is unchanged from T19:** a tie on all orders, and M1 ahead on small orders. See `out/report_cs.md`.

**Reading.** On real baskets, MBC's incumbent loses to MVP, and it loses exactly where the cap forces
independence: dependence across blocks. This holds for both the analyst's blocks and the blocks chosen from MVP's
own R. That is the "beyond the coded-subset cap" and "across subsets" claim, now on a second real dataset after
T3 (dunnhumby). It does not change T19's other finding: a cross-effects logit that estimates every pair, with no
blocks, ties MVP.

## B · Engine vs OVUS on T19's fitted 21-dimensional model (`engine.py`)

**Setup.**
- Per restaurant (all orders): 100 random holdout orders, each scored as P(Y = y), a 21-dimensional orthant.
- Raw R (min eigenvalue 1e-8) and T19's shrunk R (0.95 R + 0.05 I).
- Reference: GHK with 2^15 scrambled-Sobol points × 8 scrambles. Median SE of log P is 2.5e-4 to 1.1e-3.
- T10's harness: time per orthant is the best of 3 after a warm-up, on the local 2-core CPU.

| set | median P | engine high: µs, \|ΔP\| median (max) | engine low | OVUS (torch batch) | OVUS (pybhatlib loop) |
|---|---|---|---|---|---|
| r1 raw | 8.4e-5 | 105, 1.5e-5 (5.9e-4) | 72, 3.3e-5 (1.6e-3) | 312, 1.5e-6 (7.7e-5) | 396, same |
| r1 shrunk | 9.2e-5 | 101, 1.4e-5 (5.0e-4) | 72, 3.0e-5 (1.5e-3) | 324, 1.1e-6 (6.4e-5) | 406, same |
| r2 raw | 4.0e-5 | 102, 1.4e-5 (3.5e-4) | 68, 1.7e-5 (1.1e-3) | 289, 4.7e-7 (2.9e-5) | 398, same |
| r2 shrunk | 4.4e-5 | 101, 1.2e-5 (3.0e-4) | 69, 1.6e-5 (9.7e-4) | 288, 3.9e-7 (2.1e-5) | 393, same |

- **Speed.** The engine is 3× faster than batched OVUS at "high", and 4× faster at "low".
- **Error.**
  - The engine's median |ΔP| is 1.2–1.5e-5, against OVUS's 0.4–1.5e-6: about 10× larger.
  - The patterns are small (median P about 5e-5), so in log terms the engine's median |Δlog P| is 0.24–0.30 against
    OVUS's 0.02.
- **Zeros.** At "high", the engine returns P = 0 for 3 of 100 patterns (restaurant 1) and 1 of 100 (restaurant 2),
  on both R versions. The reference P for those patterns is 1e-11 to 7e-6, on baskets of 4–6 groups. "Low"
  returns no zeros. A zero is fatal in a log-likelihood.
- **Near-singularity is not the problem.** Raw and shrunk R give the same engine error, so the failure here is
  dimension and small P, not the conditioning of R.

**Reading.** The speed claim holds at d = 21 on real data, though at 3× rather than the benchmark's 5–6×. The
accuracy guard fails for joint basket likelihoods: the error is 10× OVUS's, and some patterns come back as exact
zeros. For this use, OVUS (or GHK) is the right scorer. The engine's T2 / T15 use (decision queries and screening)
is not tested here. The zeros are an engine bug to fix in the private repo.

## Files

- `t21.py` (A): `python analysis/T21/t21.py` takes about 7 minutes locally. It writes `out/cs_r<r>_<variant>.json`,
  `out/report_cs.md` and `out/m3_r<r>_<variant>.npz`. The npz files hold M3's holdout indices and R for B; they
  are not committed.
- `engine.py` (B): `PYTHONPATH=<dir with pybhatlib> python analysis/T21/engine.py`, about 3 minutes. It writes
  `out/engine.json`. It needs the paid local orthant binary (`ORTHANT_KEY` in `.env`).
- Cost: $0 (local CPU only).
