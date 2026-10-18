# Where the advantage is: CBC, MBC and TURF

Last updated 2026-10-18 (through T18). Maintained alongside [PLAN.md](PLAN.md); each test's full write-up is in
`analysis/T<n>/TEST.md`.

The project is about scale: unlocking problems that are too big or too slow for the incumbents, (1) with the
engine and (2) with the GPU. Speed and scale lead each case; accuracy is a guard (the faster method must not lose
the decision). Speed claims are against the right incumbent: the GPU probit pipeline against HB-MNL + RFC (logit IIA),
the engine against OVUS (the other analytic orthant approximation).

## Speed and scale at a glance

| problem | incumbent: time | ours: time | scaling | status |
|---|---|---|---|---|
| Shelf CBC fit, J = 20, N = 1,000, 12 tasks | HB-MNL (bayesm, 20k iter), CPU: 3–4 min | MNP, one orthant per respondent (T18), H100: 0.35–2.2 min ($0.02–0.15) | — | measured; 1.5–10× faster wall time than HB |
| Shelf CBC fit, J = 60 | HB-MNL: not measured (est. about 10 min; MNL cost grows linearly in J) | MNP (T18), H100: 14 min ($0.90) | ours grows about 6× from J = 20 to 60 | ours measured; HB estimated; about parity |
| Earlier GPU MNP (nested simulation), J = 20 | — | 19–27 min H100 (T15); 4.5 h L4 (T4) | — | superseded by T18 (10–80× faster) |
| Orthant screening in decision search | OVUS | engine: 3.6–20× faster (T15) | gap grows with dimension | measured, H100 and T4 |
| MBC, k = 6–30 linked items | coded subsets: combinations grow as 2^k, capped near k = 5 | MVP by IFM, CPU: seconds | linear-ish in k | measured (T2, T3, T14) |
| Few high-dimensional decision queries (MBC) | QMC | engine: about 500× faster (T2) | — | measured |
| TURF search | shared-draw simulator | engine 9–16× slower (T7) | — | not the engine's competitor (OVUS is) |

## How to read this

Each case lists the tests behind it, whether the evidence is **sim** (simulated, so the truth's structure is our
assumption) or **real** (observed choices), the result, and where any advantage comes from:

- **model**: correlated errors instead of IIA or independence;
- **GPU**: makes a large probit fit feasible;
- **engine**: fast orthant probabilities for screening and search.

The last section says which advantages hold up and which don't.

## CBC (single choice from a set)

### C1. Classic small CBC: J ≈ 3–5 alternatives, ≤ 6 attributes

- **Tests**
  - T1 (sim): MVP 0 flips of 120 vs 29 for HB-MNL + RFC.
  - T5 (sim): MVP at N = 150 matched HB + RFC at N ≥ 720.
  - X1 (real, camera): no method more share-accurate. HB beat latent class by a wide margin; at equal heterogeneity
    the probit tied the logit.
- **Speed/scale.** Small problems: no scale story. T1's probit panel fit took about 4 min on a T4 GPU (2.6 h on
  CPU); HB-MNL is comparable or faster at this size.
- **Where it stands.** We win only when the truth contains within-task similarity (products sharing a level share an
  error shock). T1 and T5 built that in; on real data it didn't show up. Sawtooth's design advice (independent
  attributes, few levels) probably designs much of it away.
- **Advantage:** probably none in practice. The exception is explanation: RFC's σ is wrong (T1), so its "why" is
  wrong even when its shares are right (model).

### C2. Shelf CBC: J = 20 SKUs with near-duplicates

- **Tests**
  - T4: the MNP fit was feasible but expensive (4.5 h per fit on an L4); paused on cost.
  - T9, T10 Part A: MACML and GHK give the same decisions.
  - T8: a GPR kernel covariance didn't beat variance shares.
  - T15 (sim): assortment PASS (+1.85 pts regret); joint line + extension + price SOFT FAIL (+0.20, real but below
    the 0.5 floor); the logit-truth guard passed (no invented similarity).
  - T18 (sim): one orthant per respondent (Genz-ordered GHK) replaces T4's nested likelihood. It is more accurate
    and smooth, and the J = 20 fit takes 0.3–2.2 min on an H100 instead of 4.5 h. With non-normal tastes, a
    Gaussian copula keeps it to one orthant with no measurable loss on shares or decisions.
- **Speed/scale (J = 20, N = 1,000, 12 tasks).** HB-MNL takes 3–4 min on CPU. The MNP fit took 19–27 min on an
  H100 before T18 and takes 0.35 min (1k draws) to 2.2 min (16k draws) after it: 1.5–10× faster than HB in wall
  time, at $0.02–0.15 of GPU per fit against fractions of a cent of CPU. The engine wasn't needed at this size: the
  pipeline tied accurate-only search.
- **Where it stands.** The model advantage is real in simulation, and since T18 it comes at HB-like or better wall
  time.
- **Advantage:** model, enabled by GPU, now without a speed penalty.

### C3. Shelf CBC: J = 40–60+, or larger decision spaces (more slots, more price levels)

- **Tests**
  - T18 cost check (sim, one dataset): a J = 60 fit (720-dimensional orthant per respondent) takes 14 min on an
    H100. It did not recover the true variance shares on that dataset (shares Wald p 7e-9). A higher-draw check
    says more draws would barely move the estimate, which points to the dataset rather than the method; a second
    dataset has not been checked.
- **Speed/scale.** MNP fit time measured at 2.2 / 6.5 / 14 min for J = 20 / 40 / 60 on an H100 (J = 40 from timed
  evaluations): about 6× from J = 20 to 60, against a 3× larger orthant (240 → 720 dimensions per respondent).
  HB-MNL at J = 60 is not measured; MNL's cost grows linearly in J, so about 10 min on CPU is the estimate, which
  puts the two at about parity in wall time with MNP carrying the correlated errors HB can't. The full T4 run
  (2 truths × J = 20 / 40 / 60 × 10 replicates) now costs about $35 of GPU time instead of about $600, if the J = 60
  miss is a dataset effect.
- **Where it stands.** Expected, not yet shown:
  - the model advantage grows, since more SKUs mean more near-duplicates for IIA to mishandle;
  - fit cost grows with J (measured: about 6× from J = 20 to 60);
  - the engine screen becomes more valuable as the search space explodes. The engine was already 3.6–20× faster
    than OVUS, rising with orthant size (T15).
- **Advantage:** this is where all three sources would stack, if they stack anywhere. Untested at decision level.

### C4. Real shelf CBC data

- **Tests**
  - T17 (real, dunnhumby soft drinks): stopped at the power check. Even with a nested-logit truth, the twin test
    could not separate NL from MNL on this shelf.
- **Where it stands.** Still the biggest open question for CBC. All the shelf wins rest on simulated truths with
  within-task similarity. Real data might show that most cannibalisation is taste heterogeneity, which HB already
  captures. X1 is a weak signal in that direction (4 alternatives, not a shelf).
- **Advantage:** unknown.

**Heterogeneity in CBC.** Both sides model it: our MNP has random tastes, and HB-MNL is the incumbent's continuous
heterogeneity model (stronger than latent class). Heterogeneity isn't where our CBC edge comes from.

## MBC (pick-any menus, linked items)

### M1. k ≤ 5 linked items (inside the coded-subset cap)

- **Tests:** T2 at k = 5. Coded subsets are fine here, and no faster or slower than MVP.
- **Advantage:** none.

### M2. k = 6–12

- **Tests**
  - T2 (sim): 28 vs 146 decision flips of 240.
  - T3 (real dunnhumby, 8 items): the holdout backed MVP on 4 disagreements, coded subsets on 0.
- **Where it stands.** Strong. Coded subsets fail exactly where the linked set exceeds the cap; knowing the true
  linked set doesn't help them, because the cap is the limit.
- **Speed/scale.** MVP by IFM runs on CPU in seconds at k = 6–12. The incumbent's coded subsets grow as 2^k
  combinations and are capped near k = 5, so the scale limit is the incumbent's, not ours. No GPU or engine needed.
- **Advantage:** model, made possible by a method that scales past the coded-subset cap.

### M3. k > 12 (20–30 items)

- **Tests**
  - T14 Part A (sim): 50 vs 207 flips of 240.
  - T14 Part B (real 24-item baking cluster): 3 to 0 at the threshold.
- **Speed/scale.** Same as M2: IFM stays in seconds to minutes at k = 20–30, where coded subsets cannot run at all.
- **Advantage:** model, and the gap grows with k.

**Heterogeneity exposure (MBC).** The T2 and T14 Part A truths had no heterogeneity. The real tests gave every arm a
household propensity covariate, but the coded-subset incumbent had no HB random intercepts. Sawtooth fits MBC by HB,
and correlated intercepts would pick up part of what MVP's ρ captures. Untested: the same comparison against an HB
coded-subset incumbent (HB-MVP for stated MBC is on the backlog). T16 showed in TURF how much a per-respondent
incumbent can close; the same risk applies here.

### M4. Joint constraints on menu or line decisions

- **Tests**
  - T11: a joint chance constraint never chose a different line (0 of 640).
  - T13: joint survival picked the same line on all 18 decisions.
- **Advantage:** none. Correlation didn't change these decisions.

**Speed in MBC.** Decisions by the engine were about 500× faster than QMC for a few high-dimensional queries (T2); the
engine was slower when decisions average over 37k low-dimensional trips (T3). At k ≤ 12 there's no time saving either
way.

## TURF (reach)

### T-a. Basket TURF, real data

- **Tests**
  - T6 (real, dunnhumby soft drinks, 30 items): against pooled independence, MVP's reach error was 49% lower and
    greedy portfolios flipped 4 of 4 in MVP's favour.
  - T16 (real, same data): against per-household independence (the practitioner incumbent), FAIL. Pooled reach
    error MVP 2.25 vs 2.38 pts (Δ +0.13, 95% CI [−0.06, +0.32]); greedy portfolios identical at every k. Per-household
    independence closes 94% of T6's gap.
- **Advantage:** none against practice. T6's win was household heterogeneity, which the practitioner incumbent already
  handles.

### T-b. TURF search speed

- **Tests:** T7. A shared-draw simulator was 9–16× faster than the engine at matched regret.
- **Speed/scale.** Against the engine's actual competitor (OVUS) this wasn't run. Against simulators the engine
  loses at this size (30 items, one orthant per portfolio).
- **Advantage:** none against simulators.

### T-c. MaxDiff TURF (Conjointly demo study, 14 colours, N = 202)

- **Tests:** experiment 7. We reproduced their TURF exactly. Their winning triple is uncertain (P(win) 0.53–0.82
  depending on the threshold; no lead over #2 is significant). Blue and Navy behave like a correlated pair. The
  split-half MVP reach test was inconclusive.
- **Advantage:** uncertainty reporting (not specific to MVP), plus some structure insight. Not a decision advantage.

### T-d. Large-item or large-k TURF

- **Tests:** none.
- **Where it stands.** Dependence matters more as k grows (the independence error grew with k in T6), but T16 shows a
  per-household incumbent absorbs most of it at the k tested. Expect the same unless within-household dependence is
  strong.
- **Advantage:** doubtful after T16.

## Cross-cutting comparisons

- **IIA.** Incumbents patch it two ways: heterogeneity (HB), which handles similarity between people, and RFC, which
  patches similarity within a choice. Our wins appear where within-occasion dependence exists beyond heterogeneity:
  confirmed in simulation (CBC) and in real MBC; not in real baskets once heterogeneity is modelled (T16).
- **RFC.** Its S1 is capped at 2/3 by functional form, and it is tuned on share fit, which doesn't identify the error
  structure (T1). Advantage: explanation, and downstream use of the correlation.
- **Estimators.** GHK and MACML tie (T9, T10 Part A). IFM is good and cheap for MBC. Gibbs is slow. Full MLE vs IFM
  barely changes decisions. With Genz ordering, one GHK orthant per respondent beats nested simulation on accuracy
  and speed (T18). These are implementation choices, not advantages over the incumbents.
- **GPU.** The enabler for CBC probit at J ≥ 20. Since T18 a J = 20 fit takes 0.35–2.2 min on an H100, 1.5–10×
  faster than HB-MNL on CPU (3–4 min); J = 60 takes 14 min (HB at J = 60 not measured, est. about 10 min). Per fit
  the GPU costs more dollars ($0.02–0.90) than HB's CPU time.
- **Engine.**
  - It beats OVUS 3.6–20× on speed (T15). On accuracy OVUS is about 5× better on mixed-sign matrices (T10 side
    study); the engine's new highest-accuracy mode (2026-10-18) closes that gap at 6 dimensions but not at 20–40.
  - It loses to simulators and low-draw GHK (T7, T15 Step 0, T3).
  - It cannot replace GHK inside a high-dimensional panel likelihood: on T18's 240-dimensional orthants it is fast
    and smooth but about 5 nats per respondent off, and its approximate likelihood has no usable optimum.
  - Its niche is many high-dimensional queries in a huge search space. We've only seen a tie at J = 20, so its
    decision value at scale is unproven.
- **Sample size.** T5 says MVP needs about 5× fewer respondents under similarity truths. Sim only.

## Summary: where the advantage probably is

1. **MBC beyond the cap (k ≥ 6).** Strongest: simulated and real evidence, cheap, model-driven. Main gap: no HB-fair
   incumbent yet, and T16 shows how much a per-respondent incumbent can close.
2. **Shelf CBC assortment with near-duplicates.** Shown in simulation at J = 20, where the MNP fit is now 1.5–10×
   faster than HB-MNL in wall time (T18). Real-data support is missing (T17 could not test it).
3. **Explanation.** The right σ and correlation vs RFC's distorted ones. A model advantage, solid in T1.
4. **Large shelves, J = 40+ and big decision spaces.** Where model, GPU and engine would stack. The J = 60 fit takes
   14 min on an H100 (about parity with an estimated HB time), and the engine screens 3.6–20× faster than OVUS. Now
   affordable to test (about $35 for T4's full run); one J = 60 recovery miss to resolve first.

**Not advantages so far:** classic small CBC; basket TURF against per-household independence (T16); speed at small
sizes; engine vs simulators; the engine inside a panel likelihood; joint chance or survival constraints; kernel
covariances; MACML vs GHK.

**The common thread.** Every surviving advantage is in a case practitioners can't keep small, either because the
coded-subset cap binds or because the decision space is too big. Where good practice keeps the problem small, or
models heterogeneity per respondent, the incumbents hold up.
