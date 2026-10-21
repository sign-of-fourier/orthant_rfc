"""T22 MVP fits on Modal: one T4 per training subset, T1's ghk_gpu.fit (GHK panel likelihood, M = 4,096 scrambled
Sobol points, total error variance fixed at pi^2 / 6, sandwich SEs). Mixed MNP: normal tastes with a diagonal sd on
all 5 coefficients, error = rail component (train, SM shared) + iid. Each respondent's 7 fit tasks are one
14-dimensional orthant. Start: MNL coefficients, taste sd 0.5, equal error shares."""
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
T1 = HERE.parent / "T1"
GHK_M = 4096

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy")
         .add_local_file(T1 / "ghk_gpu.py", "/root/t1/ghk_gpu.py"))
app = modal.App("orthant-rfc-t22")


@app.function(image=image, gpu="T4", timeout=3600, retries=1)
def mvp(tag, Xd, G, th0, seed, tvar):
    sys.path.insert(0, "/root/t1")
    import ghk_gpu
    t0 = time.time()
    r = ghk_gpu.fit.local(Xd, G, 0.0, th0, th0, GHK_M, seed, tvar=tvar, se=True)
    r = {k: v for k, v in r.items() if k != "cov_sandwich"}
    return dict(tag=tag, wall=time.time() - t0, **r)


def launch(tags=None):
    import t22
    X, y, ids = t22.load()
    tr, _ = t22.split(len(ids))
    out = t22.OUT
    jobs = []
    for i, (tag, idx) in enumerate(t22.subsets(tr).items()):
        if tags and tag not in tags or (out / f"mvp_{tag}.json").exists():
            continue
        loc = json.loads((out / f"local_{tag}.json").read_text())
        Xd, G = t22.diff_arrays(X[idx, :t22.FIT_T], y[idx, :t22.FIT_T])
        th0 = np.r_[loc["mnl"], np.full(t22.P, np.log(.5)), 0.0]
        jobs.append((tag, Xd, G, th0, 2200 + i, float(t22.T_VAR)))
    print(f"{len(jobs)} MVP fits", flush=True)
    with app.run():
        for r in mvp.starmap(jobs, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
            if isinstance(r, Exception):
                print(f"FAILED: {r!r}"[:500], flush=True)
                continue
            (out / f"mvp_{r['tag']}.json").write_text(json.dumps(r, indent=1))
            print(f"{r['tag']}: fit {r['seconds']['fit']:.0f} s + se {r['seconds']['se']:.0f} s, {r['n_iter']} iters, "
                  f"grad {r['grad_max']:.1e}, {r['gpu']}", flush=True)


if __name__ == "__main__":
    launch(sys.argv[1:] or None)
