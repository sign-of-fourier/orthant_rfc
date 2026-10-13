# Model catalog

As of 2026-10-13. Every tool tested or built in this repo, what it's for, and how it did. Test details and numbers
live in [PLAN.md](PLAN.md); open items in [BACKLOG.md](BACKLOG.md).

## Tasks

Every model below feeds one of these decisions. Tests score models on decisions, not on fit statistics.

| Task | Data | Question it answers | Tests |
|---|---|---|---|
| CBC share simulation | Single choice per task (shelf, J = 3–60) | Share after a price change or a line extension; source of volume | T1, T4, T5, T8, T9, T10 |
| Cost-weighted decision | CBC fit | Best option once margins or costs weight the shares | T1, T5, D1 (backlog) |
| MBC / pick-any | Several items per trip or menu | Joint purchase probabilities across k items | T2, T3, T14 |
| Bundles | MBC fit | Best pair, best triple | T2, T3, T14 |
| Follow-on recommendation | MBC fit | Given A in the basket, which B (P(B given A)) | T3, T14 |
| TURF reach | MBC or basket fit | Best k-item set by reach, 1 − P(buys none of S) | T6, T7, T14 |
| Product line / assortment | CBC fit + parameter uncertainty | Which SKUs to keep; joint survival P(every kept share ≥ t) | T11, T13 |
| Substitution structure (σ) | CBC fit | Which items are near-duplicates; correct explanation, not only shares | T1, T8 |
| Standard errors / decision uncertainty | Any fit | SEs, coverage, P(decision wrong) | SE test (backlog), T13 delta method |
| Holdout likelihood | Any fit | Out-of-sample fit as a tie-breaker | T3, T14 |

## Multivariate probit (MVP): models and estimators

The MVP family fits correlated latent utilities. The question is which estimator, and which structure on Σ.

| Tool | What it's for | Pros | Cons | Where it stands |
|---|---|---|---|---|
| IFM (`multivariate_probit`, cross-fit pairwise) | MBC / basket data: margins first, then pairwise ρ | CPU, minutes; sklearn-style; no GPU needed; best MBC arm so far | Pairwise R can be non-PSD and needs a nearest-correlation repair; no single-choice CBC; SEs by bootstrap only | T14 A: 50 flips vs CS-F 207 over 240 decisions; T14 B: holdout backs it 3–0 over CS-F |
| Full ML (`mvp_fit`, GHK + Sobol on GPU) | Same data as IFM, all parameters jointly | Proper MLE; sandwich SEs; H100 5–8 s per simulated fit, ~2 min on T14 real data | Linear index only; SE pass can dominate time (~18 of ~20 min at d = 12); GPU cost | Ties IFM on decisions (T14 A 115/120 same; T14 B 26/27) |
| Panel MVP-G (`mvp_gpu.py`, T4) | Shelf CBC with taste heterogeneity, J = 20+ | Only probit that ran at J = 20 | One-off code; 34 min fit + 29 min SEs at 256×128 draws; pack0 at about −3 SE in every fit | T4 paused on cost |
| Variance-share Σ | CBC: Σ built from per-attribute similarity | Identifiable; explains near-duplicates directly | Needs the analyst's attribute list | T1 PASS; T5 PASS (matches RFC-S at N = 150 vs ≥ 720) |
| GPR kernel Σ | CBC: Σ from a kernel over attributes | Handles continuous attributes; recovers weights within 2 SE | No gain over variance shares | T8 FAIL (ties variance shares) |
| MACML (Bhat) | Probit MLE via analytic MVN approximation (OVUS inside) | No simulation noise; analytic gradients | Known bias at large J and strong ρ; CPU reference is slow, so it was ported to torch | T9 FAIL (ties GHK, both 0 regret at N = 600); T10 A inconclusive |
| Gibbs (bayesm `rmvpGibbs`, `rmnpGibbs`) | Bayesian probit, posterior SDs | Standard reference; full posterior | Slow; R only | Baseline for the SE test (backlog) |
| MNP (multinomial probit, GHK-MSL) | Multinomial probit for single choice | Runs on GPU and locally in torch | Not tested in this series yet | Built 2026-10-07 |

## Probability engines and simulators

These compute MVN orthant probabilities ("P(this purchase pattern)") from a fitted model; they don't fit anything.
Errors are absolute |ΔP| against a GHK reference.

| Tool | What it's for | Pros | Cons | Where it stands |
|---|---|---|---|---|
| The engine (`orthant_cdf` service, local `.so` in `multivariate_probit.orthant`) | Deterministic orthant probabilities, d = 4 to ~128 | 70–150× faster than QMC Sobol; deterministic, so no noise between scenarios; smooth | Median \|ΔP\| 3e-3 to 7e-3; returns 0 for 12-dim patterns near 1e-4; weaker near ρ ≈ 0.8; not accurate enough for fitting yet | T7 FAIL: 12× slower than the bitset simulator for TURF at k = 6 |
| OVUS (Bhat; torch port of pybhatlib) | Analytic orthant approximation; MACML's inner integral | Median \|ΔP\| 5e-4 to 2e-3, about 5× below the engine on mixed-sign Σ; faster on GPU at d ≤ 20 (0.7 / 4.3 µs vs 1.9 / 25 µs at d = 6 / 20) | Depends on variable order (ties); speed relies on large batches | T10 side study only; not in any decision test |
| QMC (scrambled Sobol, Genz) | General-purpose orthant probabilities; the speed baseline the engine is quoted against | Error shrinks predictably with more points; error bars from scrambling; no approximation bias | 70–150× slower than the engine at matched use; noisy at small P | Baseline in the engine benchmarks |
| GHK-QMC (GHK on scrambled Sobol) | Reference truth for scoring; the likelihood inside Full ML and MNP | Accuracy set by the draw count; smooth in parameters, so it can be fitted; standard | Simulation noise; cost grows with draws × d | Truth scorer for T2, T3, T14 |
| Shared-draw (bitset) simulator | Score many scenarios or portfolios from one draw set | One set of draws scores every candidate; 9–16× faster than the engine for TURF search | Monte Carlo noise: needs N ≥ 4,096 to stay within 0.5 pts | T7 winner for TURF; ~25× faster than the engine for scenario scoring |

## Incumbents, baselines and decision-layer tools

The incumbents stand in for Sawtooth-style practice; the decision-layer tools sit on top of a fitted model. IIA
(independence of irrelevant alternatives) is the logit assumption behind most incumbents: a new item draws share
from every item in proportion. RFC exists to patch it; probit doesn't assume it.

| Tool | What it's for | Pros | Cons | Where it stands |
|---|---|---|---|---|
| HB-MNL (proxy: bayesm `rhierMnlRwMixture`) | CBC utilities per respondent; the Sawtooth CBC/HB estimator | Industry standard; cheap; heterogeneity built in | Logit, so IIA within a respondent | Always paired with RFC in our tests |
| RFC (randomized first choice; tuned RFC-S / RFC-G) | Share simulator on top of HB-MNL: adds error draws to soften IIA | Restores some near-duplicate substitution; what clients see | Gets substitution by inflating σ, so shares can be right for the wrong reason; tuning is ad hoc | Lost T1 and T5 (needs ≥ 720 completes where MVP needs 150); 4.5% regret vs 0.5% in T8 |
| Coded subsets, CS-F (analyst lift rule) / CS-O (oracle blocks) | MBC: exact joint patterns within blocks of ≤ 5 items, independence between blocks | Exact within a block; explainable; what MBC analysts do | Capped at about 5 items per block; blind to dependence across blocks | Lost T2 (146 vs 28 flips), T3 (holdout 4–0 for MVP), T14 A (207 vs 50 flips), T14 B (holdout 3–0 for MVP) |
| Independent logits (IL) | Baseline: one logit per item | Fastest; trivial | Ignores all dependence | Worst MBC arm (T14 A: 235 material flips) |
| MaxDiff | Best–worst scaling designs | Familiar to clients; strong individual-level signal | Not built or tested here | Backlog: designs that identify error correlation |
| Joint chance constraint (T11) | Product line kept only if P(every kept share ≥ t) ≥ 1 − α, as one orthant | One orthant per candidate line | Picked the same line as the independent rule in 0 of 640 searches | T11 FAIL |
| Cardinality line with joint survival (T13) | Best 6 of 20 SKUs (38,760 lines) with joint survival | Exact share orthants; delta-method uncertainty | Joint = Independent on all 18 decisions | T13 KILL |
| TURF search (exhaustive, greedy) | Best k-set by reach | Reach is one orthant per portfolio | The bitset simulator beats the engine on time | T6 PASS (MVP reach beats independence); T7 FAIL on engine speed |

## What to use, by task

On the evidence so far, MVP wins on the model question, IFM is enough as the estimator, and the engine has not won
on speed.

| Task | Current best | Runner-up | Evidence | Open |
|---|---|---|---|---|
| CBC share simulation, J ≤ 6 | MVP with variance-share Σ, GHK ML | MACML (ties) | T1, T5, T8, T9 | Larger J (T4 paused) |
| CBC at J = 20+ | Panel MVP-G | HB-MNL + RFC | T4 gate only | Cost; pack0 at −3 SE |
| Cost-weighted decision | MVP | HB-MNL + RFC-S | T1, T5 | D1 demo |
| MBC bundles and follow-ons | MVP-IFM | Full ML (ties) | T2, T3, T14 | IFM holdout LL under a near-singular R |
| TURF reach | MVP reach + bitset simulator | MVP + engine | T6, T7 | Engine speed-up at d ≤ 32 |
| Product line with joint survival | Independent rule (joint adds nothing) | — | T11, T13 | None |
| Substitution structure (σ) | MVP, variance shares | GPR kernel (ties) | T1, T8 | Second dataset |
| SEs / decision uncertainty | Full ML sandwich SEs | IFM bootstrap, Gibbs | Not tested | SE value-proposition test |
