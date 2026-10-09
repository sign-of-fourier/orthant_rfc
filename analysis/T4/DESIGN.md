# T4. Shelf scaling, simulated CBC: design (draft 2026-10-08)

Item 1. No analysis code has been written for this test.

## Question

On a shelf CBC (every task shows all J SKUs plus "none", J = 20–60), with errors that share
attribute-level components:

- **Q1.** Does a probit fitted by full ML change decisions relative to HB-MNL plus tuned RFC
  (the Sawtooth proxy), and does it recover the similarity structure?
- **Q2.** Which probit estimator scales in J: conditioning on the attribute factors (K shared
  components, K fixed as J grows) vs per-task GHK orthants (J dims) vs Gibbs?
- **Q3.** Time to fit (and to SEs) against J, per method.

## Claim (reframed 2026-10-09, while paused)

- **Not the claim:** a full joint correlation matrix at J = 60. An unrestricted Σ at J = 60 has
  1,770 correlations; N = 600 × 12 tasks cannot identify them on any hardware.
- **The claim:** with the substitution structure practitioners already assume (variance shares
  on brand / flavor / pack, CDT-shaped, a few parameters), probit recovers that structure at
  J = 20–60 in practical time and changes delist, line and price decisions relative to
  HB-MNL + RFC.
- **Compute, kept separate:** structure reduces the number of parameters, not the integral. Each
  choice is still a (J − 1)-dimensional orthant unless conditioned on the factors, and the
  factor-conditioned fit (MVP-F) failed recovery at affordable draws (gate 2026-10-09, PLAN.md).
  The compute burden at J = 60 is real.
- **Positioning:** incumbents use logit because it is cheap and avoid large-J non-IIA models
  because they are expensive. That is a compute constraint, and GPU probit removes it.

## Non-goals

- Engine-based MLE (the engine is still in progress; it has no gradients). The engine
  appears only as a second decision-layer column, as in T2.
- MACML (backlog). The full SE comparison (tracked follow-up). SE time is recorded here only
  because it comes with our fit.
- Real shelf data.

## DGP (T1's, scaled to a shelf)

- **SKUs.** Brand (5) × flavor (4) × pack (3) = 60. J = 20 and 40 are fixed nested subsets
  with every level present; J = 60 is the full grid. Effects-coded part-worths
  (4 + 3 + 2), a price slope and a "none" constant: 11 parameters.
- **Tasks.** T = 12 training tasks per respondent, all J SKUs plus "none", each SKU's price
  drawn from 5 levels ($3.50–$5.50) independently; 6 fixed holdout tasks (for RFC tuning).
- **Respondents.** N = 600, β_n ~ N(b, diag(ω²)) on all 11 coefficients (T1's b and ω, extended
  to the extra levels; normal price slope).
- **Errors.** U_njt = x'β_n + Σ_a σ_a ζ_{n,t,a,level_a(j)} + ν_njt: SKUs sharing a level of an
  attribute in a task share that ζ draw (K = 5 + 4 + 3 = 12 components). Total error variance
  fixed at π²/6 (T1's normalisation).
- **Truths.**
  - **S0 null:** σ_a = 0, ν Gumbel (logit truth). Does the probit invent similarity?
  - **S1 shelf:** variance shares brand / flavor / pack / product = 0.10 / 0.35 / 0.35 / 0.20.
    Same flavor and pack, different brand: error correlation 0.70; the same flavor only: 0.35.
    T1's G1 had a product share of only 0.02, which makes every factor-conditional choice almost
    deterministic. 0.20 is closer to a shelf (open point A).
- **Replicates.** R = 10 per (truth, J): 2 × 3 × 10 = 60 datasets.

## Methods

**Incumbent (T1's code).**
- **HB-MNL** (`bayesm::rhierMnlRwMixture`, T1's prior), point estimates = posterior means.
- **RFC-S** (one σ + Gumbel + exponent) and **RFC-G** (one σ per attribute), tuned to holdout
  share MAE.

**Probit (same model as the DGP, normal ν), three likelihood evaluators.**
- **MVP-F (factor-conditioned MSL, the main fit).** Per respondent, Q joint scrambled-Sobol
  draws of the tastes and every task's ζ. Given those, the ν are iid, so a task's choice
  probability is a 1-D integral, ∫ φ(z) Π_{k≠j} Φ((V_j − V_k)/σ_ν + z) dz, done by 20-node
  Gauss-Hermite. The cost is linear in J and K is fixed. On Modal GPUs (torch, fixed draws,
  L-BFGS, autograd Hessian, sandwich SEs), as T1's `ghk_gpu.py`. A one-off fitter here; moving
  it into the library or API is the tracked follow-up.
- **MVP-G (per-task GHK).** Tastes by the same Q draws, each task's J-dim orthant by GHK.
  The cost grows as J². Timed per likelihood evaluation at J = 20, 40, 60. Fitted in full only
  at J = 20, where it serves as the cross-check of MVP-F.
- **Gibbs** (`bayesm::rmnpGibbs`, unstructured (J × J) Σ, no random tastes, which is what
  exists). Capped at 1 h per fit at J = 20; time per iteration recorded at 40 and 60. This is
  the incumbent probit; we expect it not to scale.

**Recovery gate (before the full run).** J = 20, S1, N = 1,000, one dataset: MVP-F with
Q = 2,048 and 8,192. Every σ_a and the mean of b must be within 2 SEs of the truth, and the two
Q values must agree within 1 SE. If Q = 8,192 is needed, the time estimates double.

## Metrics (as T1, plus shelf decisions)

- **σ recovery.** S1: the error correlation of a near-duplicate pair (same flavor and pack)
  against the truth. S2: per-attribute variance shares (MVP, RFC-G). S4: 95% coverage of σ_a
  (MVP).
- **Decisions** (scored against the truth simulator, 10⁶ simulated choices per scenario):
  - **D-delist:** which of the firm's 6 SKUs to drop to lose the least portfolio
    contribution. This depends directly on where the dropped volume goes.
  - **D-ext:** the incrementality of a near-duplicate line extension, and launch vs hurdle.
  - **D-price:** the price of one SKU from 5 levels, for margin (T1).
  - Reported as flips and regret %, with a flip material when regret > 0.5%. Holdout share
    MAE is descriptive.
- **Time.** Fit, SEs and decisions, wall-clock time per method against J. Also the time per
  likelihood evaluation, MVP-F vs MVP-G, against J (Q2).

## Verdict rule (from PLAN)

- **KILL** if the incumbents scale adequately and the decisions match HB-MNL + RFC.
- **PASS** if decisions differ materially in MVP's favour (S1) with no invented similarity
  (S0), and MVP-F fits at J = 60 in practical time.
- Blanks for you: what "practical" means (default: under 30 min per fit), and the flip
  threshold (default: MVP has fewer material flips than the best RFC variant at every J).

## Run time (estimates)

| step | where | estimate |
|---|---|---|
| recovery gate (J = 20, two Q) | 2 GPU | 15 min |
| MVP-F fits, 60 datasets (~5–20 min each, growing with J) | 60 GPU, parallel | 25 min wall |
| MVP-G full fits at J = 20 (20) + timing at 40/60 | GPU | 20 min wall |
| HB-MNL (20k iterations, J = 60 is the long pole) | 60 CPU containers | 60–80 min wall |
| RFC-S / RFC-G tuning | CPU containers | 20 min |
| Gibbs (J = 20 capped, timing at 40/60) | CPU containers | 60 min wall |
| decisions (simulation) | CPU containers | 10 min |

About 2 h wall in total, HB being the long pole; about $10 of GPU. The recovery gate runs
first, and I report back before the full run.

## Open points

- A. **Truth shares** for S1 (0.10 / 0.35 / 0.35 / 0.20 recommended) vs T1's G1
  (0.08 / 0.45 / 0.45 / 0.02).
- B. **Estimators.** MVP-F as the main fit, MVP-G at J = 20 plus timing, Gibbs capped
  (recommended); or full MVP-G fits at every J (several hours at J = 60).
- C. **HB iterations.** 20k as in T1 (recommended, comparable) vs 10k (halves the long pole).
- D. **Kill blanks** above.
