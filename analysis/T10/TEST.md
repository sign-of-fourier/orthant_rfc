# T10 · MACML vs GHK follow-up: smaller N, and J = 20 (Part A done, inconclusive; Part B PAUSED 2026-10-10)

Moved verbatim from PLAN.md on 2026-10-16 (links re-rooted). Summary and current status: [PLAN.md](../../PLAN.md).

User, 2026-10-10: follow up T9 (decisions saturated at N = 600, J = 3) with smaller N, and harder at
J = 20. Kill criteria: BLANKS for the user. Nothing runs until they are filled in.

- **Part A: smaller N, T1 DGP (J = 3).** T9's code unchanged; GHK arm reused from T5's N = 150 / 300
  cells (same seeds). Power warning: T5's GHK made only 1–2 flips per 20 replicates at N = 150–300
  (cost regret 0.10–0.37%, SE 0.1–0.4), so a > 2 paired-SE regret gap needs MACML to flip often.
  Estimate: no build; MACML 60 fits per N ≈ 30 T4-min ≈ $0.30 per N.
- **Part B: J = 20, T4 shelf DGP** (`analysis/T4/t4.py`: brand 5 × flavor 4 × pack 3 grid, J = 20
  subset, 12 shared error components; 13 decisions per dataset: 6 price, delist, 6 extensions;
  decisions scored by 10⁶-draw simulation, flip material at 0.5% regret).
  - MACML: T9's pair likelihood at 2J = 40 dims per pair (66 pairs × N).
  - GHK: T1's exact panel GHK at T·J = 240 dims (T4's MVP-G nested simulation was the costly path).
    Not yet validated at this size; GHK noise at 240 dims with M = 4,096 is a risk.
  - Step 0, timing probe: one likelihood + gradient per arm at J = 20, N = 600 (build ~2 h, ~$0.05).
  - Rough estimate before the probe (d² scaling from T1, unverified): GHK ~45–90 T4-min per fit,
    MACML ~5–20; S1 truth × 10 reps ≈ $8–15; adding S0 doubles it.
- **Kill criteria (user, 2026-10-10).**
  - Step 0 likelihood (user): the 240-dim panel orthant arises only from the person-level tastes
    β_n (ζ and ν are fresh per task), so GHK uses the conditioned likelihood
    P(y_n) = ∫ Π_t P(y_nt | β_n) φ(β_n) dβ_n: QMC over β_n × per-task 20-dim GHK (T4's MVP-G).
    Full 240-dim GHK only for across-task occasion errors (not this model). The 240-dim probe was
    stopped unfinished.
  - Recovery gate (J = 20, one shelf-truth dataset, each arm vs the TRUTH): coefficients and
    variance shares within 2 SE (own SEs; Godambe for MACML); implied correlation max |error| ≤ 0.1.
    MACML failure: record, drop the arm, report. GHK failure: stop.
  - Win: mean regret over the 13 decisions per dataset; GHK < MACML by > 2 paired-SE. Reported, not
    gating: by decision type and flips at 0.5%; delist is underpowered (1 per dataset).
  - Truths / replicates: shelf (S1) and null (S0), 10 each. Pre-registered extension: paired gap
    between 1 and 2 SE → extend to 20; otherwise stop at 10.
  - Time caps: GHK ≤ 90 T4-min per fit, feasibility kill if the probe projects > 3 h; MACML ≤ 30
    min per fit, kill if > 2 h. On a kill: stop and ask.
  - Part A: N = 150 and 300, T9's criteria unchanged. Labelled in advance as expected inconclusive
    (it tests sample efficiency, not correlation strength).

- **Part A results (2026-10-10): inconclusive, as labelled.** `analysis/T9/out/full_G*_N{150,300}/`,
  `summary_N150.json`, `summary_N300.json`, `analysis/T9/run_G*_N*.log`. ~33 T4-min (~$0.33).

| N | truth | GHK regret / flips | MACML regret / flips | MACML − GHK (paired SE) |
|---|---|---|---|---|
| 150 | G0 | 0.37% / 2 | 0.37% / 3 | 0.00 (0.00) |
| 150 | G1 | 0.17% / 2 | 0.25% / 3 | +0.08 (0.15), no flag |
| 150 | G2 | 0.10% / 1 | 0.65% / 3 | +0.55 (0.45), no win (1.2 SE) |
| 300 | G0 | 0.00% / 1 | 0.00% / 1 | 0.00 (0.00) |
| 300 | G1 | 0.31% / 2 | 0.31% / 2 | 0.00 (0.00), no flag |
| 300 | G2 | 0.31% / 2 | 0.31% / 2 | 0.00 (0.00), no win |

- **Step 0 probe (2026-10-10), one T4, J = 20, S1, N = 600.** `analysis/T10/probe.py`,
  `out/probe.json`, `out/probe_macml_chunks.json`. Evaluations per fit from T4's gate fits.
  - GHK conditioned (MVP-G) 256 × 128: 40 s / eval × ~55 evals ≈ 37 min fit (+ ~31 min SEs).
    T4's gate (N = 1,000) at this size under-recovered shares (u z −2.4 / −2.5).
  - GHK conditioned 512 × 256: 160 s / eval (chunk 4; chunk 8 OOM on 16 GB) × ~78 ≈ 3.5 h fit:
    over the 3 h feasibility kill. This is the size at which T4's gate recovered the shares.
  - MACML (40-dim pairs): 9 s / eval at chunk 60 (1.5 GiB) → ~15–25 min per fit with order rounds.
  - Stopped and asked (kill rule) 2026-10-10. Options put to the user: (1) gate GHK at 256 × 128
    (in the cap, known share under-recovery); (2) an intermediate 320 × 160; (3) stop Part B.
  - **PAUSED (user, 2026-10-10: "Lets pause on this") with that choice open.** No Part B fits run.
  - **Bigger-GPU probe (user, 2026-10-12, cap $1).** `probe_gpu.py`, `out/probe_gpu.json`, `probe_gpu.log`. Same
    512 × 256 evaluation as above, chunk 32 (peak 65 GiB): A100-80GB 22.4 s / eval (7.2× the T4), H100 12.8 s (12.5×).
    At ~78 evaluations: A100 ~29 min, H100 ~17 min per fit (SEs not timed), under the 90-min cap. At list prices
    (T4 $0.59/h, A100-80GB $2.50/h, H100 $3.95/h) about $2.10 / $1.20 / $1.10 per fit. Probe ~2 min wall, ~$0.15.

- **Side study: OVUS vs the engine (user, 2026-10-10, while Part B is paused).** Not a kill-criteria
  test; measurements only. Accuracy by |ΔP| (absolute, user: ratios alone mislead), by dimension.
  - Accuracy on T9/T10's own orthants (`ovus_vs_engine.py`, `out/ovus_vs_engine.json`; 100 orthants
    per set; reference GHK scrambled Sobol 2^15 × 8). |ΔP| median (max):

| set | d | median P | engine high | engine low | OVUS |
|---|---|---|---|---|---|
| T9 pair, G2 | 6 | 0.087 | 6.8e-3 (3.1e-2) | 2.0e-2 (1.5e-1) | 5.2e-4 (3.5e-3) |
| T9 pair, G1 | 6 | 0.084 | 6.9e-3 (3.1e-2) | 1.9e-2 (1.5e-1) | 5.0e-4 (3.4e-3) |
| J = 20 task given tastes | 20 | 0.19 | 2.7e-3 (1.3e-2) | 9.9e-3 (4.0e-2) | 2.1e-3 (1.9e-2) |
| J = 20 MACML pair | 40 | 0.0085 | 3.6e-3 (1.4e-2) | 2.9e-3 (1.7e-2) | 8.0e-4 (3.1e-3) |

  - Speed, CPU, 2 threads (`speed_sweep.py`, `out/speed_sweep.json`), µs per orthant at the largest
    batch run: engine (compiled) 15 / 103 / 424 at d = 6 / 20 / 40; OVUS torch 6.3 / 38 / 163. The
    engine is flat in batch; OVUS falls with batch (fixed per-call overhead amortised). The engine's
    torch version matches the compiled one to ≤ 2.2e-16.
  - Speed, GPU, one T4, numpy in / out (`gpu_sweep.py`), µs per orthant at batch 10⁵: first run
    (`out/gpu_sweep_before_fix.json`) engine 1.9 / 25 / 1,456 vs OVUS 0.70 / 4.3 / 26. The d = 40
    engine figure was a GPU batched-kernel limit above d = 32 in one preprocessing step; fixed in the
    engine (results unchanged to ≤ 7e-18 on T10's orthants). Rerun (`out/gpu_sweep.json`): engine
    1.9 / 25 / 32 vs OVUS 0.72 / 4.3 / 28. GPU equals CPU to ≤ 3.3e-16 for both methods.
  - On generated mixed-sign two-factor correlation matrices (500 per d, GHK reference SE ≤ 2.2e-5),
    |ΔP| median (max): OVUS 8.7e-4 (1.6e-2) / 8.2e-4 (6.7e-3) / 4.5e-5 (2.1e-3) at d = 6 / 20 / 40;
    engine high 5.6e-3 (4.9e-2) / 4.1e-3 (3.6e-2) / 2.3e-4 (7.0e-3). Cost at 10⁵ about equal at d = 40,
    OVUS cheaper at d ≤ 20. On T10's positive-leaning orthants the gap is smaller (d = 20 about equal).
  - GPU ~25 T4-min in all (~$0.25).
