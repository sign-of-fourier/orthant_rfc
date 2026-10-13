"""T10 step 0 follow-up: the 512 x 256 MVP-G timing probe (probe.py) on bigger GPUs (user 2026-10-12, cap $1).

Same data, truth and theta as probe.py; one likelihood + gradient evaluation at N = 600, best of 2 after a warm-up.

    python analysis/T10/probe_gpu.py
"""
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if len(HERE.parents) > 2:                                      # host only (not in the Modal container)
    for d in ("T1", "T4", "T9"):
        sys.path.insert(0, str(HERE.parent / d))
import macml  # noqa: E402

image = macml.image.add_local_python_source("ghk_gpu", "macml", "mvp_gpu", "t4")
app = modal.App("orthant-rfc-t10-probe-gpu")
Q, R, TIMEOUT = 512, 256, 480


def _probe(X, y, A, th, chunks):
    import torch

    import mvp_gpu
    out = {}
    for chunk in chunks:
        t0 = time.time()
        try:
            out["result"] = dict(mvp_gpu._time_eval(X, y, A, Q, R, "G", th, 0, chunk), wall=time.time() - t0,
                                 peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
            break
        except torch.OutOfMemoryError:
            out[f"chunk{chunk}"] = "OOM"
            torch.cuda.empty_cache()
    out["gpu"] = torch.cuda.get_device_name()
    return out


@app.function(image=image, gpu="A100-80GB", timeout=TIMEOUT)
def probe_a100(X, y, A, th):
    return _probe(X, y, A, th, (32, 16, 8))


@app.function(image=image, gpu="H100", timeout=TIMEOUT)
def probe_h100(X, y, A, th):
    return _probe(X, y, A, th, (32, 16, 8))


if __name__ == "__main__":
    import probe
    _, _, th, X, y, A = probe.arrays()
    t0 = time.time()
    with modal.enable_output(), app.run():
        calls = {name: f.spawn(X, y, A, th) for name, f in (("A100-80GB", probe_a100), ("H100", probe_h100))}
        res = {}
        for name, c in calls.items():
            try:
                res[name] = c.get()
            except Exception as e:
                res[name] = f"{type(e).__name__}: {e}"[:300]
    res["wall_s"] = time.time() - t0
    for k, v in res.items():
        print(k, v)
    (HERE / "out" / "probe_gpu.json").write_text(json.dumps(res, indent=1))
