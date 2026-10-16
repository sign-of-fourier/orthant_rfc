# T4 · Shelf scaling (item 1). Paused 2026-10-09 on cost

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

Reframe (2026-10-09): not a full joint correlation at J = 60 (1,770 correlations, not
identifiable from N = 600 × 12 tasks on any hardware). The claim is structured Σ (brand / flavor /
pack variance shares, CDT-shaped) recovered at J = 20–60 in practical time, changing delist, line
and price decisions vs HB-MNL + RFC; the (J − 1)-dim integral and its compute stay. See
[T4 design, Claim](../../analysis/T4/DESIGN.md#claim-reframed-2026-10-09-while-paused). Follow-ups are
in [BACKLOG.md](../../BACKLOG.md#t4-follow-ups).

**T4. Shelf scaling (item 1).** Simulate CBC at J = 20–60 with an
attribute-structured Σ. Fit the probit by engine MLE against GHK, Gibbs and
MACML; compare against logit plus tuned RFC. Record the number of shared
components K against J−1 (orthants vs conditioning on factors). Measure
time to fit against J and the decision difference against logit plus RFC.
Kill: the incumbents scale adequately, or decisions match logit plus RFC.

Status at pause:

| Test | Items | Status | Decision | Write-up |
|---|---|---|---|---|
| T4 | 1 | design approved 2026-10-08 (A–C as recommended); core (t4.py), MVP-F/G fitter (mvp_gpu.py, nested taste × inner-draw simulation) and recovery gate (gate.py) run 2026-10-09: FAIL as specified (128×128 collapses to product-only; 512×256 recovers shares but 29 min fit + 31 min SE at J = 20, b: pack0 z −3.2). Gauss-Hermite for the chosen nu is worse at equal compute (noise is in the 12 shared terms); per-task GHK at J = 20 is ~20× less noisy per unit time, so the gate reran with MVP-G as the main fit at J = 20 (user 2026-10-09: decisions first, scaling later, ~1 h per fit acceptable). GHK gate on L4 (256×128, 512×256): 512×256 hit the 3 h Modal limit at 04:37 UTC; the script saved nothing, so the finished 256×128 result was lost too (fixed: gate.py now saves each fit; fit timeout now 24 h, L4). Rerun 04:50–13:10 UTC on L4: 256×128 fit 34 min + SE 29 min, shares .11/.30/.29/.31 (u z −0.8/−2.4/−2.5); 512×256 fit 4.5 h + SE 3.6 h, shares .13/.35/.29/.23 (u z +0.2/−0.6/−1.2), pack0 −3.4, none −2.2, two taste SDs −2.7/−2.2; sizes agree to 1.8 SE. Strictly FAIL (2-SE and 1-SE rules); shares recovered at 512×256; pack0 ≈ −3 in every fit so far (likely this dataset). T4 PAUSED 2026-10-09 on cost (user): full run est. J = 20 only ~$45 at 512×256 (~$6 at 256×128), all J ~$600; SEs not needed for decisions (move S4 to the SE test) | — | [design](../../analysis/T4/DESIGN.md) |
