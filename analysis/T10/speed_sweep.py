"""Speed sweep, CPU (2 cores): orthant engine vs OVUS at d = 6 / 20 / 40, batch 100 / 1,000 / 10,000.

Engine: the local compiled binary and the GPU service's code (private sibling checkout), both run on CPU.
OVUS: T9's batched torch port (analysis/T9/macml.ovus_log). Orthants from ovus_vs_engine.py's sets,
tiled to the batch size. Torch on 2 threads; best of 3 after a warm-up. Also: torch engine vs
compiled engine agreement (max |dP|).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))   # private engine source, sibling checkout
import modal_gp_api as gp  # noqa: E402
import ovus_vs_engine as B  # noqa: E402

torch.set_num_threads(2)
cpu = torch.device("cpu")


def best(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


d4, th4 = B.t4_data()
sets = {6: B.set_a("G2"), 20: B.set_b(d4), 40: B.set_c(d4, th4)}
out = []
print("| d | batch | engine compiled µs | engine torch-CPU µs | OVUS torch-CPU µs | engine torch vs compiled max |dP| |")
print("|---|---|---|---|---|---|")
for d, (a0, S0) in sets.items():
    for nb in (100, 1000, 10000):
        if d == 40 and nb == 10000:
            continue                       # 10k x 40 x 40 doubles per OVUS step: keep the 4 GB box safe
        k = -(-nb // len(a0))
        a, S = np.tile(a0, (k, 1))[:nb], np.tile(S0, (k, 1, 1))[:nb]
        tc = best(lambda: B.rfc._LOCAL.cdf(a, S, resolution="high"))
        tt = best(lambda: gp.orthant_cdf_batch(a, S, cpu, resolution="high"))
        to = best(lambda: B.ovus_log(torch.tensor(a), torch.tensor(S)))
        agree = float(np.abs(gp.orthant_cdf_batch(a[:100], S[:100], cpu) - B.rfc._LOCAL.cdf(a[:100], S[:100])).max())
        row = dict(d=d, batch=nb, engine_compiled_us=1e6 * tc / nb, engine_torch_us=1e6 * tt / nb,
                   ovus_torch_us=1e6 * to / nb, engine_torch_vs_compiled=agree)
        out.append(row)
        print(f"| {d} | {nb:,} | {row['engine_compiled_us']:.1f} | {row['engine_torch_us']:.1f} | "
              f"{row['ovus_torch_us']:.1f} | {agree:.1e} |", flush=True)
(HERE / "out" / "speed_sweep.json").write_text(json.dumps(out, indent=1))
