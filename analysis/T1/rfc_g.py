"""T1 RFC-G arm, added to finished runs: one sigma per attribute (brand, flavor, pack) plus the
Gumbel product error g, tuned to holdout share MAE like RFC-S. A deviation from Sawtooth
practice (DESIGN.md, "RFC-G (generous)").

    python analysis/T1/rfc_g.py G0 G1 G2   # adds rec["rfc_g"] to out/full_<truth>/results.json

Regenerates each replicate's data, holdouts and RFC level draws from the same seed streams as
run.replicate and reruns HB with the same seed (deterministic), checked by reproducing RFC-S's
stored holdout MAE. HB point estimates are cached in out/full_<truth>/hb/beta_rep<k>.npy.

HB runs locally (bayesm, 2 workers); each replicate's tuning (~5 min of numpy) runs in its own
Modal CPU container as soon as its HB finishes (ephemeral app; nothing deployed).
"""
import json
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.1", "scipy==1.16.0")
         .add_local_file(HERE / "run.py", "/root/analysis/T1/run.py"))
app = modal.App("orthant-rfc-t1-rfcg")


def _run():
    """run.py without the orthant engine (RFC-G needs none of it)."""
    try:
        import run
    except ImportError:
        import types
        sys.modules["rfc"] = types.SimpleNamespace(BACKEND="none")
        sys.path.insert(0, "/root/analysis/T1")
        import run
    return run

LO, HI = np.log([0.01, 0.01, 0.01, 0.05]), np.log([5.0, 5.0, 5.0, 5.0])


def tune(R, H, obs, x_s):
    """Minimise holdout share MAE over log (sigma_b, sigma_f, sigma_p, g): Nelder-Mead from the
    RFC-S optimum and from the best point of a 3^4 log grid; the better result is kept."""
    run = _run()

    def mae(x):
        x = np.clip(x, LO, HI)
        p = np.exp(x)
        return np.mean([np.abs(R.shares(H[i], p[:3], p[3]) - obs[i]).mean() for i in range(len(H))])

    axes = [np.linspace(LO[i], HI[i], 3) for i in range(4)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(4, -1).T
    x_g = grid[int(np.argmin([mae(x) for x in grid]))]
    best = None
    for x0 in (np.log(x_s), x_g):
        res = minimize(mae, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=600))
        if best is None or res.fun < best.fun:
            best = res
    x = np.clip(best.x, LO, HI)
    flags = [f"param {i} at bound" for i in range(4) if x[i] - LO[i] < .05 or HI[i] - x[i] < .05]
    return np.exp(x), float(best.fun), flags


def replicate_data(run, truth, rep, N, T):
    sig = np.sqrt(run.T_VAR * run.TRUTHS[truth])
    logit = truth in run.LOGIT_TRUTHS
    r_data, r_hold, r_rfc = [np.random.default_rng(s) for s in np.random.SeedSequence([run.SEED, rep]).spawn(3)]
    d = run.make_data(r_data, N, T, sig, logit)
    H = run.holdout_tasks(r_hold)
    yh = run.simulate(r_hold, np.broadcast_to(H, (N,) + H.shape), d["beta"], sig, logit)
    obs = np.stack([np.bincount(yh[:, i], minlength=4) / N for i in range(6)])
    return d, H, obs, r_rfc


@app.function(image=image, cpu=2.0, timeout=3600)
def tune_remote(truth, rep, N, T, B, rec_truth, rfc_s, check_only=False):
    run = _run()
    _, H, obs, r_rfc = replicate_data(run, truth, rep, N, T)
    R = run.RFC(B, 1000, r_rfc)
    mae_s = np.mean([np.abs(R.shares(H[i], rfc_s["sigma_l"], rfc_s["g"]) - obs[i]).mean() for i in range(6)])
    if not np.isclose(mae_s, rfc_s["holdout_mae"], rtol=1e-6, atol=1e-9):
        raise RuntimeError(f"{truth} rep {rep}: RFC-S MAE not reproduced ({mae_s} vs {rfc_s['holdout_mae']})")
    if check_only:
        return float(mae_s)
    t0 = time.time()
    p, mae, flags = tune(R, H, obs, [rfc_s["sigma_l"]] * 3 + [rfc_s["g"]])
    t_tune = time.time() - t0
    var = np.r_[p[:3] ** 2, np.pi ** 2 * p[3] ** 2 / 6]
    S1 = float((var[1] + var[2]) / var.sum())
    rdec = run.decisions(lambda pr: R.shares(pr, p[:3], p[3]))
    return dict(sigma=p[:3].tolist(), g=float(p[3]), holdout_mae=mae, flags=flags, S1=S1,
                S2=(var / var.sum()).tolist(), S1_err=S1 - rec_truth["S1"],
                S3_ratio=S1 / rec_truth["S1"] if rec_truth["S1"] else None,
                seconds=dict(tune=t_tune), **run.score(rdec, rec_truth))


def main(truths):
    import run

    run.LOGF = open(run.HERE / "out" / "log_rfc_g.txt", "a")
    recs = {t: json.loads((run.HERE / "out" / f"full_{t}" / "results.json").read_text()) for t in truths}
    todo = [(t, r) for t in truths for r in recs[t] if "rfc_g" not in r]
    run.log(f"RFC-G: {len(todo)} replicates (HB local x2, tuning on Modal CPU)")

    def hb(t, rec):
        out = run.HERE / "out" / f"full_{t}" / "hb"
        cache = out / f"beta_rep{rec['rep']}.npy"
        if cache.exists():
            return np.load(cache), 0.0
        d, *_ = replicate_data(run, t, rec["rep"], rec["N"], rec["T"])
        work = out / f"w{rec['rep']}"
        work.mkdir(parents=True, exist_ok=True)
        B, _, t_hb = run.run_hb(work, d["prods"], d["y"], 20000, 1000 + rec["rep"])
        np.save(cache, B)
        return B, t_hb

    calls = []
    with app.run(), ThreadPoolExecutor(2) as ex:
        futs = {ex.submit(hb, t, rec): (t, rec) for t, rec in todo}
        for f in as_completed(futs):
            t, rec = futs[f]
            B, t_hb = f.result()
            run.log(f"{t} rep {rec['rep']}: HB {t_hb:.0f} s; tuning submitted")
            calls.append((t, rec, t_hb, tune_remote.spawn(t, rec["rep"], rec["N"], rec["T"], B,
                                                         rec["truth"], rec["rfc_s"])))
        for t, rec, t_hb, c in calls:
            g = c.get()
            g["seconds"]["hb"] = t_hb
            rec["rfc_g"] = g
            run.log(f"{t} rep {rec['rep']}: tune {g['seconds']['tune']:.0f} s; sigma {np.round(g['sigma'], 3)}, "
                    f"g {g['g']:.3f}, S1 {g['S1']:.3f} (RFC-S {rec['rfc_s']['S1']:.3f}), "
                    f"MAE {g['holdout_mae']:.4f} (RFC-S {rec['rfc_s']['holdout_mae']:.4f}) {g['flags']}")
            (run.HERE / "out" / f"full_{t}" / "results.json").write_text(json.dumps(recs[t], indent=1))
    run.log("done")


if __name__ == "__main__":
    main(sys.argv[1:])
