"""T20 fits on Modal: one container per (truth, N, replicate) and arm.

  mvp(truth, N, rep)        T4 GPU: MVP by GHK, T1's ghk_gpu.fit exactly as T5 ran it (M = 4,096,
                            total error variance fixed, no SE pass)
  incumbent(truth, N, rep)  1 CPU core: HB-MNL (bayesm, 20,000 iterations) + RFC-S tuned on the 6 holdout
                            tasks, T1's run_hb / RFC / tune_rfc exactly as T5 ran them

Both arms regenerate the replicate's data from cell_data() (same seeds, same T1 generator) and return a
hash of the choices so run.py can check they saw the same dataset. Decisions are scored locally
(run.py assemble), where the truth's shares use the local orthant binary.
"""
import hashlib
import sys
import tempfile
import time
import types
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
T1 = HERE.parent / "T1"
T1_REMOTE = "/root/analysis/T1"

SEED = 20261019
TIDX = {"G0": 0, "G1": 1, "G2": 2}
T = 12  # tasks per respondent (T1/T5)
HB_ITERS = 20000  # T1/T5 full run
R_L = 1000  # RFC level draws (T1/T5 full run)
GHK_M = 4096  # GHK Sobol points per respondent (T5)

gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.5.1", "numpy", "scipy")
             .add_local_file(T1 / "run.py", f"{T1_REMOTE}/run.py")
             .add_local_file(T1 / "ghk_gpu.py", f"{T1_REMOTE}/ghk_gpu.py"))
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("r-base-core", "r-cran-bayesm")
             .pip_install("numpy==2.5.1", "scipy==1.16.0")
             .add_local_file(T1 / "run.py", f"{T1_REMOTE}/run.py")
             .add_local_file(T1 / "hb.R", f"{T1_REMOTE}/hb.R"))
app = modal.App("orthant-rfc-t20")


def t1_module():
    """analysis/T1/run.py. In a container the orthant backend is stubbed: neither arm's fit calls it."""
    if not modal.is_local() and "rfc" not in sys.modules:
        sys.modules["rfc"] = types.SimpleNamespace(BACKEND="none", orthant_cdf=None)
    t1_dir = str(T1) if modal.is_local() else T1_REMOTE
    if t1_dir not in sys.path:
        sys.path.insert(0, t1_dir)
    import run
    return run


def cell_seeds(truth, N, rep):
    """Independent streams per (truth, N, replicate): data, holdout, RFC draws; plus HB and GHK seeds."""
    ss = np.random.SeedSequence([SEED, TIDX[truth], N, rep])
    r_data, r_hold, r_rfc = [np.random.default_rng(s) for s in ss.spawn(3)]
    hb_seed, ghk_seed = (int(x) % 2 ** 31 for x in np.random.SeedSequence([SEED, TIDX[truth], N, rep, 1]).generate_state(2))
    return r_data, r_hold, r_rfc, hb_seed, ghk_seed


def cell_data(t1, truth, N, rep):
    """T5's replicate() data step with T20 seeds: training data, holdout tasks and observed holdout shares."""
    sig = np.sqrt(t1.T_VAR * t1.TRUTHS[truth])
    logit = truth in t1.LOGIT_TRUTHS
    r_data, r_hold, r_rfc, hb_seed, ghk_seed = cell_seeds(truth, N, rep)
    d = t1.make_data(r_data, N, T, sig, logit)
    H = t1.holdout_tasks(r_hold)
    yh = t1.simulate(r_hold, np.broadcast_to(H, (N,) + H.shape), d["beta"], sig, logit)
    obs = np.stack([np.bincount(yh[:, i], minlength=4) / N for i in range(6)])
    y_sha = hashlib.sha256(np.ascontiguousarray(d["y"], dtype=np.int64).tobytes()).hexdigest()[:16]
    return dict(d=d, H=H, obs=obs, r_rfc=r_rfc, hb_seed=hb_seed, ghk_seed=ghk_seed, sig=sig, y_sha=y_sha)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


@app.function(image=gpu_image, gpu="T4", timeout=3600, retries=1)
def mvp(truth, N, rep):
    t1 = t1_module()
    import ghk_gpu

    c = cell_data(t1, truth, N, rep)
    P = t1.P
    th0 = np.r_[np.zeros(P), np.full(P, np.log(.5)), np.zeros(3)]  # T5 start (--scale total)
    th_true = t1.theta_true(c["sig"], "total")
    Xd, G = ghk_gpu.panel_arrays(c["d"]["prods"], c["d"]["y"], t1.task_X)
    t0 = time.time()
    r = ghk_gpu.fit.local(Xd, G, 0.0, th0, th_true, GHK_M, c["ghk_seed"], tvar=float(t1.T_VAR), se=False)
    r = {k: v for k, v in r.items() if k != "cov_sandwich"}
    return _jsonable(dict(truth=truth, N=N, rep=rep, y_sha=c["y_sha"], wall=time.time() - t0, max_iter=500, **r))


@app.function(image=cpu_image, cpu=1.0, memory=3072, timeout=3600, retries=1)
def incumbent(truth, N, rep):
    t1 = t1_module()
    c = cell_data(t1, truth, N, rep)
    work = Path(tempfile.mkdtemp())
    B, llh, t_hb = t1.run_hb(work, c["d"]["prods"], c["d"]["y"], HB_ITERS, c["hb_seed"])
    q = len(llh) // 4
    drift = abs(llh[3 * q:].mean() - llh[2 * q:3 * q].mean()) / abs(llh[3 * q:].mean())
    t0 = time.time()
    R = t1.RFC(B, R_L, c["r_rfc"])
    (sl, g), mae, flags = t1.tune_rfc(R, c["H"], c["obs"])
    t_tune = time.time() - t0
    dec = t1.decisions(lambda pr: R.shares(pr, sl, g))
    return _jsonable(dict(truth=truth, N=N, rep=rep, y_sha=c["y_sha"], sigma_l=sl, g=g, holdout_mae=mae,
                          flags=flags, hb_loglike_drift=drift, seconds=dict(hb=t_hb, tune=t_tune),
                          decisions=dec))
