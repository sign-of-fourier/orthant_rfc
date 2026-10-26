# Business use cases

Last updated 2026-10-25 (through T37). The layer over [ADVANTAGES.md](ADVANTAGES.md): that page lists technical
advantages; this one lists the business decisions they change, and keeps only those that pass two tests.

1. **Speed enables it.** The GPU fit or the engine makes a method practical that was too slow, too costly, or
   impossible for the incumbent's workflow. GPU-over-CPU speed itself needs no test.
2. **No workaround is essentially as good.** Sawtooth and Conjointly, as they ship today (HB-MNL, SoP / RFC,
   coded subsets, serial cross-effects, simulation over posterior draws), cannot reach an equally good decision.
   Methods that exist only in research code (Apollo, Biogeme, Bhat's GAUSS) are noted but do not count as the
   incumbent's workaround.

**From advantage to use case.** [ADVANTAGES.md](ADVANTAGES.md) lists technical advantages, each against its own
comparator. A row there is not a use case until it passes both tests above. Inferences that do not follow:

- **Faster than a method the incumbent does not use.** The engine is 3.6–20× faster than OVUS at screening
  candidates (T15) and faster than OVUS as the probit likelihood (T27, T36). Sawtooth and Conjointly score
  candidates with logit share of preference or RFC, which compute no orthants, so the engine has no incumbent step
  to be faster than. Engine speed counts only through a case whose model needs orthants.
- **A component the incumbent already has.** A shared-draw simulator over the fitted probit works, and at times
  beats the engine as a screen (T26). RFC is also a shared-draw simulator. What the incumbent lacks is the
  error covariance estimated from the choices, which is the fit, and that is counted in the cases below.
- **Cheap anywhere.** If the incumbent could run it today at no real cost, speed does not enable it (test 1).
  TURF is solved exactly by an ILP in about 1 s (T28).
- **The cost of delivering a case.** Which scorer computes a passing case (engine, simulator, GHK), or whether a
  faster fit exists (MACML, T38), changes its cost, not its verdict.
- **Beats the incumbent's documented default only.** A case must also beat the best practice the incumbent
  ships: HB over posterior draws (T31 tie, T34 holds), serial cross-effects for large menus (T33 tie).

**Focus: stated preference.** These cases are stated preference (SP: designed conjoint surveys) against SP
incumbents, because that is where the product fits; revealed preference (RP) isn't the fit, so it isn't the focus.
Each claim, test and incumbent is labelled SP or RP and only like is compared with like. RP demand tools
(scanner-data transference, loyalty panels, consumer decision trees from sales) answer some of the same decisions;
here they appear in "who buys": a buyer with rich sales history answers ranging from RP, and the SP opening is
where RP has no history (new products, ranges outside historical variation). RP results are kept in
[ADVANTAGES.md](ADVANTAGES.md#revealed-preference-results-not-the-focus). An SP + RP fusion study would be a
separate, labelled case.

Decisions are named in the client's terms. Regret is in points of the best achievable outcome against a known truth
(simulation) unless a row says otherwise.

## At a glance

| business decision | 1. speed enables | 2. incumbent's best workaround today | verdict |
|---|---|---|---|
| **Which products to range when shoppers see some as near-substitutes on some attributes but not others** (assortment, k of 20) | yes: the probit fit went from 4.5 h to 0.35–2.2 min on an H100 (T18) | HB-MNL + RFC over posterior draws: 1.74 pts behind [1.09, 2.38], wrong pick at k = 6 on every dataset (T34) | **passes** |
| **Go / no-go, hurdle and ROI on the forecast itself** (the profit or volume level, not the ranking) | yes (same fit) | none closes it: HB over draws forecasts 11–20% low (T34), −7 to −9% under correlated substitution (T31); MNP within 3% | **passes on forecast error**; a decision that flips is untested |
| **How many respondents to buy, small CBC** (J ≤ 6, cost-driven choices) | partly: at this size the GPU fit is 23–29 s, about HB's time (T20) | RFC over draws: the regret floor halves (≈ 5.4 → 3.5 pts) and stays flat in N (T35) | **partial**: holds on one of two correlated truths |
| **Brand-side line and placement across store clusters** (GA over lines of up to 6, 5 clusters) | yes: the engine runs about 18k shelf evaluations a minute where GHK on 8 cores runs 11k (T25) | logit share of preference 0.91 / 2.32% regret; RFC 2.49 / 3.60%; MNP + engine 0.35 / 0.75% (T25) | **borderline**: right direction, every gap under 3 $ per 1,000 trips, 2 seeds |
| Large-menu MBC (k = 20–30 linked items) | yes | serial cross-effects logit (Sawtooth's recommendation): tie, 49 vs 50 flips of 240 (T33) | fails |
| Line + price at J = 60 | yes | HB over posterior draws: tie on shares and decisions (T31) | fails (forecast level moves to the go / no-go row) |
| TURF reach | no | stepwise or an exact ILP solves it in about 1 s; the T6 win was heterogeneity, which HB covers (T16, T28) | fails |
| Exact-clone line extension | — | RFC handles exact clones by construction | fails (not tested; not claimed) |

## Passing

### Ranging when near-substitution differs by attribute

- **Decision.** Which k of 20 SKUs to stock (T15 D1, k = 6, 8, 10), when shoppers who lose their flavor switch
  within flavor, those who lose their pack switch within pack, and so on.
- **What speed enables.** A probit fit that estimates the substitution from the choices at shelf size: 0.35–2.2
  min on an H100, $0.02–0.15 (T18), against 4.5 h on an L4 before.
- **Why the workaround falls short.** RFC adds one declared error per attribute and tunes its spread on holdout
  shares; it cannot learn which attribute shoppers substitute on. Over posterior draws it is still 1.74 pts behind
  (T34) and misses the k = 6 pick on every dataset. MNP's regret is 0.00 on all 15 instances.
- **What it is not.** Not a probit advantage. An error-component mixed logit or a cross-nested logit fitted to the
  choices gets within 0.27–0.35 pts (T37). Neither ships in Sawtooth or Conjointly, so the claim stands against
  them, but it is "estimates near-substitution from the choices, not from the attribute list", not "probit".
- **Evidence.** Simulation; the truth is heavy in shared errors (80% of error variance by brand / flavor / pack),
  drawn fresh per choice occasion (not per respondent, so HB's random part-worths cannot absorb it). The evidence
  it still needs is SP: a stated-shelf CBC with occasion variation in the design and a holdout. Scanner data can
  motivate that occasion-level substitution is plausible in a category; it cannot stand in for that test.

### Forecast level for go / no-go, hurdle and ROI

- **Decision.** Whether a launch or range change clears a hurdle: the forecast's level matters, not only the
  ranking of options.
- **What speed enables.** Same fit as above.
- **Why the workaround falls short.** HB over posterior draws fixes the readout's overconfidence but adds no
  substitution structure: profit forecasts 11–20% low on T15's assortment (T34) and −7 to −9% under the J = 60
  stress truth (T31), where MNP is within 3% and about +3%. The two logits that tie MNP on picks miss the level
  (EC −8 to +1%, CNL +11 to +18%, T37). Calibrating to sales is the practitioner's remedy, and needs sales history
  for the products in question; a new launch has none.
- **Open.** No test yet shows a hurdle decision flipping. The proposed check: on T34's instances, the share of
  hurdle levels between the incumbent's forecast and the truth (about $1).

### Respondent budget for small CBC (partial)

- **Decision.** How many respondents a small study (J ≤ 6, e.g. cost- or price-driven choices among a few options)
  needs to pick correctly.
- **What speed enables.** Weak at this size: the MVP fit takes 23–29 s on a T4 against HB's 25–28 s plus RFC tuning
  9–18 s (T20). The GPU makes it routine, not newly possible.
- **Why the workaround falls short.** HB + RFC has a regret floor (about 5 pts) that more respondents do not lower;
  MVP is at 0.2–1.6% from N = 50 (T20). Over posterior draws the floor halves to about 3.5 and stays flat; it is
  still above MVP at N = 50 on one truth (+4.5 pts, p 0.008), not significantly on the other (+2.3, p 0.14) (T35).
- **What it is not yet.** Tested against no structural logit fitted to the choices; T37 suggests one would close
  much of it, though none ships in the incumbents' tools.

## Borderline

### Brand-side line and placement search (T25)

- Engine as the evaluator inside a GA over lines and store-cluster placement (orthants of d = 18–31). At a fixed
  300 s: 0.35 / 0.75% regret against logit 0.91 / 2.32% and RFC 2.49 / 3.60% (estimated / stress truth). Speed
  matters here: at 60 s GHK on 8 cores is at 2.08 / 4.50%.
- Every gap is under 3 $ per 1,000 trips, on two seeds, and the estimated truth has no error correlation. It passes
  both tests in direction; materiality is unshown.

## Failing (a workaround is essentially as good)

- **Large-menu MBC.** Beats coded subsets by 118–157 fewer flips of 240 (T2, T14), but Sawtooth recommends serial
  cross-effects for large menus, and that ties MVP (T33).
- **Line + price at J = 60.** Beats the documented point-estimate readout; ties HB over posterior draws on shares
  and decisions (T31). Its forecast-level gap is counted under go / no-go.
- **TURF.** No speed gate (exact ILP in about 1 s, T28), and the T6 win was respondent heterogeneity, which HB
  per-household independence closes 94% of (T16).
- **Exact-clone line extension.** RFC handles exact clones. Near-clones (r 0.97–0.995) are an engine-vs-OVUS claim
  (T36), not a business case against the incumbent, until a decision test on a near-clone shelf says otherwise.

## Pending

- **MACML at shelf size (T38).** If a composite-likelihood probit fit on a GPU matches MNP, the ranging and
  forecast cases above become cheaper to deliver; it does not change their verdicts against the incumbents.
- **Hurdle flips** (see the forecast case).
- **Real SP data with strong near-substitution** (a stated-shelf CBC with a holdout). All passing cases are simulation.
