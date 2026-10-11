"""T10 step 0: timing probe at J = 20 (T4 shelf DGP, truth S1, N = 600, T = 12), on one T4.

Likelihood + gradient at the true theta, per arm:
  - GHK, conditioned on the person-level tastes (user 2026-10-10: the 240-dim panel orthant comes only
    from integrating beta_n out; zeta and nu are per task): T4's MVP-G (analysis/T4/mvp_gpu.py),
    Q taste draws x per-task 20-dim GHK with R draws, full N = 600, at Q x R = 256 x 128 and 512 x 256.
  - MACML: T9's pairwise composite likelihood (analysis/T9/macml.cl_rows), 66 pairs x 2J = 40 dims,
    on the first N_PROBE respondents (extrapolated linearly to N = 600).

    python analysis/T10/probe.py
"""
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
for d in ("T1", "T4", "T9"):
    sys.path.insert(0, str(A / d))
import ghk_gpu  # noqa: E402
import macml  # noqa: E402
import mvp_gpu  # noqa: E402
import t4  # noqa: E402

J, N, T, N_PROBE = 20, 600, 12, 60
image = macml.image.add_local_python_source("ghk_gpu", "macml", "mvp_gpu", "t4")
app = modal.App("orthant-rfc-t10-probe")


def arrays():
    d = t4.make_data("S1", J, N, T, 0)
    lv = d["levels"]
    prods = np.concatenate([np.broadcast_to(lv, (N, T, J, 3)), d["prices"][..., None]], -1)
    Xd, G = ghk_gpu.panel_arrays(prods, d["y"], lambda pr: t4.task_X(lv, pr[..., 3]), levels=t4.LEVELS)
    sh = t4.TRUTHS["S1"]
    th = np.r_[t4.b_true(J), np.log(t4.W_TRUE), np.log(sh[:3] / sh[3])]
    X = t4.task_X(lv, d["prices"]).astype(np.float32)
    A = t4.member(lv).astype(np.float32)
    return Xd[:N_PROBE], G[:, :N_PROBE], th, X, d["y"], A


@app.function(image=image, gpu="T4", timeout=3600)
def probe(Xd, G, th, tvar, X, y, A, macml_only=False, chunks=(2, 5)):
    import torch

    import macml
    import mvp_gpu

    dev, f64 = torch.device("cuda"), torch.float64
    Xd = torch.as_tensor(Xd, dtype=f64, device=dev)
    G = torch.as_tensor(G, dtype=f64, device=dev)
    n, T, K, _ = Xd.shape
    D = T * K
    out = dict(n=n, dims_ghk=D, dims_pair=2 * K)

    def timed(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        v = fn()
        torch.cuda.synchronize()
        return v, time.time() - t0, torch.cuda.max_memory_allocated() / 2 ** 30

    for Q, R in (() if macml_only else ((256, 128), (512, 256))):
        for chunk in (8, 4, 2, 1):                  # T4 has 16 GB (T4 was tuned on a 24 GB L4)
            t0 = time.time()
            try:
                out[f"mvpg_{Q}x{R}"] = dict(mvp_gpu._time_eval(X, y, A, Q, R, "G", th, 0, chunk), wall=time.time() - t0)
                break
            except torch.OutOfMemoryError:
                out[f"mvpg_{Q}x{R}_chunk{chunk}"] = "OOM"
                torch.cuda.empty_cache()

    IDX = torch.as_tensor(macml.pair_index(T, K), device=dev)
    for chunk in chunks:
        def evm():
            thr = torch.as_tensor(th, dtype=f64, device=dev).clone().requires_grad_(True)
            tot = 0.0
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                r = macml.cl_rows(thr.expand(hi - lo, -1), Xd[lo:hi], G[:, lo:hi], tvar, IDX)
                r.sum().backward()
                tot += float(r.sum())
            return tot

        try:
            v, sec, mem = timed(evm)
            out[f"macml_chunk{chunk}"] = dict(seconds=sec, peak_GiB=mem, cl=v)
        except torch.OutOfMemoryError:
            out[f"macml_chunk{chunk}"] = "OOM"
            torch.cuda.empty_cache()
    out["gpu"] = torch.cuda.get_device_name()
    return out


if __name__ == "__main__":
    Xd, G, th, X, y, A = arrays()
    print(f"arrays: Xd {Xd.shape}, G {G.shape}, theta {len(th)}", flush=True)
    t0 = time.time()
    with modal.enable_output(), app.run():
        r = probe.remote(Xd, G, th, float(t4.T_VAR), X, y, A, *((True, (20, 60)) if '--macml-chunks' in sys.argv else ()))
    r["wall_s"] = time.time() - t0
    scale = N / N_PROBE
    for k, v in r.items():
        if isinstance(v, dict) and "seconds_per_eval" in v:
            print(f"{k:22s} {v['seconds_per_eval']:8.1f} s per eval at N = {N} (full sample)")
        elif isinstance(v, dict) and "seconds" in v:
            print(f"{k:22s} {v['seconds']:8.1f} s on {N_PROBE} resp -> {v['seconds'] * scale:8.0f} s per eval at N = {N}; "
                  f"peak {v['peak_GiB']:.2f} GiB")
        else:
            print(f"{k:22s} {v}")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / ("probe_macml_chunks.json" if "--macml-chunks" in sys.argv else "probe.json")).write_text(json.dumps(r, indent=1))
