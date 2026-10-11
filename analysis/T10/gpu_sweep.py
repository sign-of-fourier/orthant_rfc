"""GPU speed sweep on one T4: orthant engine (the GPU service's code, private
sibling checkout, resolution high) vs OVUS (T9's batched torch port,
analysis/T9/macml.ovus_log, chunked at OVUS_CHUNK rows). d = 6 / 20 / 40, batch 10^3 .. 10^6.
Same orthants as ovus_vs_engine.py's sets, tiled. Timed from numpy in to numpy out (both include the
host <-> device copies), best of 3 after a warm-up. Also: max |dP| GPU vs the CPU values.

    PYTHONPATH=<dir containing pybhatlib> python analysis/T10/gpu_sweep.py
"""
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "T9"))
sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))   # private engine source, sibling checkout

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy", "pybhatlib==0.4.0", "pydantic", "fastapi")
         .add_local_python_source("modal_gp_api", "macml"))
app = modal.App("orthant-rfc-t10-gpu-sweep")
BATCHES = {6: (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6), 20: (10 ** 3, 10 ** 4, 10 ** 5), 40: (10 ** 3, 10 ** 4, 10 ** 5)}
OVUS_ROWS = {6: 250_000, 20: 25_000, 40: 6_000}   # rows per OVUS chunk (peak memory ~ rows * d^2)


@app.function(image=image, gpu="T4", timeout=3600)
def sweep(sets, ref_cpu):
    import torch

    import macml
    import modal_gp_api as gp

    dev = torch.device("cuda")

    def ovus(a, S):
        out = np.empty(len(a))
        r = OVUS_ROWS[a.shape[1]]
        for lo in range(0, len(a), r):
            at = torch.as_tensor(a[lo:lo + r], device=dev)
            St = torch.as_tensor(S[lo:lo + r], device=dev)
            out[lo:lo + r] = macml.ovus_log(at, St).exp().cpu().numpy()
        return out

    def engine(a, S):
        return gp.orthant_cdf_batch(a, S, dev, resolution="high")

    def best(fn, n=3):
        fn()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    rows = []
    with torch.no_grad():
        for d, (a0, S0) in sets.items():
            d = int(d)
            ce, co = ref_cpu[str(d)]
            agree = dict(engine=float(np.abs(engine(a0, S0) - ce).max()), ovus=float(np.abs(ovus(a0, S0) - co).max()))
            for nb in BATCHES[d]:
                k = -(-nb // len(a0))
                a, S = np.tile(a0, (k, 1))[:nb], np.tile(S0, (k, 1, 1))[:nb]
                te, to = best(lambda: engine(a, S)), best(lambda: ovus(a, S))
                rows.append(dict(d=d, batch=nb, engine_us=1e6 * te / nb, ovus_us=1e6 * to / nb,
                                 gpu_vs_cpu_max_dP=agree))
                del a, S
                torch.cuda.empty_cache()
    return dict(rows=rows, gpu=torch.cuda.get_device_name())


if __name__ == "__main__":
    import torch

    import ovus_vs_engine as B
    from macml import ovus_log

    d4, th4 = B.t4_data()
    sets = {6: B.set_a("G2"), 20: B.set_b(d4), 40: B.set_c(d4, th4)}
    ref = {str(d): (B.rfc._LOCAL.cdf(a, S, resolution="high"), ovus_log(torch.tensor(a), torch.tensor(S)).exp().numpy())
           for d, (a, S) in sets.items()}
    t0 = time.time()
    with modal.enable_output(), app.run():
        r = sweep.remote(sets, ref)
    r["wall_s"] = time.time() - t0
    print(f"\n{r['gpu']}, wall {r['wall_s']:.0f} s")
    print("| d | batch | engine µs / orthant | OVUS µs / orthant | engine ÷ OVUS |")
    print("|---|---|---|---|---|")
    for x in r["rows"]:
        print(f"| {x['d']} | {x['batch']:,} | {x['engine_us']:.2f} | {x['ovus_us']:.2f} | {x['engine_us'] / x['ovus_us']:.1f}× |")
    print("GPU vs CPU agreement (max |dP|):", {x["d"]: x["gpu_vs_cpu_max_dP"] for x in r["rows"]})
    (HERE / "out" / "gpu_sweep.json").write_text(json.dumps(r, indent=1))
