"""T2. Linked-set scaling, simulated MBC: coded subsets vs multivariate probit (DESIGN.md).

    python analysis/T2/run.py --selftest     # L1, k = 8, 1 rep, N = 300, all methods
    python analysis/T2/run.py                # 4 truths x k in {5, 8, 12} x 20 reps (ask first)

Everything is fitted in parallel off this machine:
  - truths (decision quantities, SciPy Genz at tight tolerance): Modal CPU, one per (truth, k);
  - CS-F, CS-O, IL and MVP-IFM (multivariate-probit 0.2.3, pairwise): Modal CPU, one per dataset;
  - MVP-Gibbs (bayesm::rmvpGibbs, gibbs.R): Modal CPU with R, one per dataset;
  - MVP-ML: the deployed mvp_fit service (full ML by GHK on a GPU), one call per dataset.
Scoring (Modal CPU, one per dataset): fitted probits' decisions by GHK with 16k Sobol points
(t2.orth_qmc, accurate to ~1e-5) so Q1 is not mixed with evaluator error; coded subsets by
pattern enumeration. The orthant engine (local binary, rfc.orthant_cdf) recomputes the probit
decisions as a second column (engine flips and decision time, for Q3). Fits are saved to
fits.json; `--eval-only` rescores them. Writes analysis/T2/out/<mode>/results.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2  # noqa: E402

KS = (5, 8, 12)
py_image = (modal.Image.debian_slim(python_version="3.12")
            .pip_install("numpy==2.5.1", "scipy==1.16.0", "multivariate-probit==0.2.3")
            .add_local_file(HERE / "t2.py", "/root/t2.py"))
r_image = (modal.Image.debian_slim(python_version="3.12")
           .apt_install("r-base-core", "r-cran-bayesm")
           .pip_install("numpy==2.5.1", "scipy==1.16.0")
           .add_local_file(HERE / "t2.py", "/root/t2.py")
           .add_local_file(HERE / "gibbs.R", "/root/gibbs.R"))
app = modal.App("orthant-rfc-t2")
LOGF = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


# ---------------------------------------------------------------------------
# Remote jobs
# ---------------------------------------------------------------------------
@app.function(image=py_image, cpu=1.0, timeout=3600)
def truth_job(truth, k):
    import t2
    t0 = time.time()
    tm = t2.truth_model(t2.truth_params(truth, k))
    return dict(dec=t2.decisions(tm, k), hold=t2.holdout_joints(tm, k), seconds=time.time() - t0)


@app.function(image=py_image, cpu=2.0, memory=3072, timeout=3600)
def fit_cpu(X, Y, truth, k):
    """CS-F, CS-O, IL and MVP-IFM, each timed."""
    import t2
    from multivariate_probit import MultivariateProbit

    out = {}
    t0 = time.time()
    sub = t2.analyst_subsets(Y)
    out["cs_f"] = dict(blocks=t2.fit_cs(X, Y, sub), subsets=sub, seconds=time.time() - t0)
    t0 = time.time()
    sub = t2.oracle_subsets(t2.truth_params(truth, k))
    out["cs_o"] = dict(blocks=t2.fit_cs(X, Y, sub), subsets=sub, seconds=time.time() - t0)
    t0 = time.time()
    out["il"] = dict(blocks=t2.fit_cs(X, Y, [[m] for m in range(t2.M)]), seconds=time.time() - t0)
    t0 = time.time()
    mp = MultivariateProbit(inner="linear", dependence="pairwise", cv=5, random_state=0).fit(X, Y.astype(int))
    E = mp.decision_function(np.vstack([np.zeros(t2.M), np.eye(t2.M)]))
    out["ifm"] = dict(c=E[0].tolist(), B=(E[1:] - E[0]).T.tolist(), R=mp.correlation_.tolist(),
                      seconds=time.time() - t0)
    return out


@app.function(image=r_image, cpu=1.0, memory=4096, timeout=3600)
def fit_gibbs(X, Y, seed, iters, keep):
    import subprocess
    import tempfile
    d = Path(tempfile.mkdtemp())
    np.savetxt(d / "X.csv", X, delimiter=",")
    np.savetxt(d / "Y.csv", Y, delimiter=",", fmt="%d")
    t0 = time.time()
    r = subprocess.run(["Rscript", "/root/gibbs.R", str(d / "X.csv"), str(d / "Y.csv"), str(d / "g"),
                        str(iters), str(keep), str(seed)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    rd = lambda f: np.loadtxt(d / f, delimiter=",", skiprows=1)  # noqa: E731
    tr = rd("g_trace.csv")
    q = len(tr) // 4
    drift = float(abs(tr[3 * q:].mean() - tr[2 * q:3 * q].mean()))
    return dict(c=rd("g_c.csv").tolist(), B=np.diag(rd("g_b.csv")).tolist(), R=rd("g_R.csv").tolist(),
                corr_drift=drift, seconds=time.time() - t0)


def _remote_client():
    """multivariate_probit's fitter="modal" client (repo source; the installed 0.2.3 predates it)."""
    spec = importlib.util.spec_from_file_location(
        "mvp_remote", Path.home() / "projects" / "multivariate_probit" / "src" / "multivariate_probit" / "_remote.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_ml(client, X, Y, seed):
    t0 = time.time()
    # no SEs: the decisions don't use them, and the 222-parameter Hessian took ~18 of ~20 min
    out = client.fit_remote(X, Y.astype(float), n_draws=512, seed=seed, alpha=1e-6, se=False)
    d, p = Y.shape[1], X.shape[1]
    return dict(c=np.asarray(out["intercept"], float).tolist(),
                B=np.asarray(out["coef"], float).reshape(d, p).tolist(),
                R=np.asarray(out["corr"], float).reshape(d, d).tolist(),
                n_iter=int(out["n_iter"]), converged=bool(out["converged"]), seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# Local: decisions and scoring
# ---------------------------------------------------------------------------
def engine():
    sys.path.insert(0, str(HERE.parents[1] / "research" / "rfc_vs_exact"))
    import rfc
    return rfc


def build(fit, orth):
    if "blocks" in fit:
        return t2.cs_model(fit["blocks"])
    R = np.array(fit["R"])
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return t2.Probit(fit["c"], fit["B"], R, orth)


def evaluate(name, fit, truth_res, k, Xh, Yh, orth):
    t0 = time.time()
    model = build(fit, orth)
    dec = t2.decisions(model, k)
    hj = t2.holdout_joints(model, k)
    t_dec = time.time() - t0
    rec = dict(t2.score(dec, truth_res["dec"], hj, truth_res["hold"]),
               holdout_ll=float(model.pattern_logprob(Xh, Yh).mean()),
               seconds_fit=fit["seconds"], seconds_decide=t_dec, seconds_total=fit["seconds"] + t_dec)
    if "blocks" not in fit:
        iu = np.triu_indices(t2.M, 1)
        rec["corr_offdiag_abs_median"] = float(np.median(np.abs(np.array(fit["R"])[iu])))
    for key in ("subsets", "converged", "n_iter", "corr_drift"):
        if key in fit:
            rec[key] = fit[key]
    return rec


@app.function(image=py_image, cpu=1.0, memory=2048, timeout=3600)
def score_job(fits, truth_res, k, Xh, Yh):
    import t2  # noqa: F401
    out = {}
    for name, fit in fits.items():
        out[name] = fit if "error" in fit else evaluate(name, fit, truth_res, k, Xh, Yh, t2.orth_qmc)
    return out


def engine_column(fit, truth_res, k, orth):
    """The probit decisions recomputed with the engine: flips vs truth, time."""
    t0 = time.time()
    dec = t2.decisions(build(fit, orth), k)
    t = time.time() - t0
    sc = t2.score(dec, truth_res["dec"], truth_res["hold"], truth_res["hold"])
    return dict({f"{d}_{f}": sc[f"{d}_{f}"] for d in ("bundle2", "bundle3", "turf4") for f in ("flip", "regret_pct")},
                seconds_decide=t)


def _cached(path, fn):
    """Phase results survive a stopped run: keyed by 'truth|k|rep'."""
    if path.exists():
        return {tuple(int(x) if x.isdigit() else x for x in key.split("|")): v
                for key, v in json.loads(path.read_text()).items()}
    res = fn()
    path.write_text(json.dumps({"|".join(map(str, j)): v for j, v in res.items()}))
    return res


def fit_all(jobs, data, truths, ks, args, t0, out):
    client = _remote_client()
    def collect(calls, label):
        res = {}
        for j, c in calls.items():
            try:
                res[j] = c.get() if hasattr(c, "get") else c.result()
            except Exception as e:  # recorded, not fatal
                res[j] = dict(error=repr(e)[:500])
        log(f"{label} done ({time.time() - t0:.0f} s)")
        return res

    have = lambda name: (out / f"phase_{name}.json").exists()  # noqa: E731
    truth_calls = {} if have("truths") else {(tr, k, 0): truth_job.spawn(tr, k) for tr in truths for k in ks}
    cpu_calls = {} if have("cpu") else {j: fit_cpu.spawn(data[j][0], data[j][1], j[0], j[1]) for j in jobs}
    gibbs_calls = {} if have("gibbs") else {
        j: fit_gibbs.spawn(data[j][0], data[j][1].astype(int), 1000 + i, args.gibbs_iters, 4)
        for i, j in enumerate(jobs)}
    with ThreadPoolExecutor(len(jobs)) as ex:
        ml_futs = {} if have("ml") else {j: ex.submit(fit_ml, client, data[j][0], data[j][1], i)
                                         for i, j in enumerate(jobs)}
        tr3 = _cached(out / "phase_truths.json", lambda: collect(truth_calls, "truths"))
        truths_res = {(a, b): v for (a, b, _), v in tr3.items()}
        cpu = _cached(out / "phase_cpu.json", lambda: collect(cpu_calls, "CS / IL / IFM fits"))
        gibbs = _cached(out / "phase_gibbs.json", lambda: collect(gibbs_calls, "MVP-Gibbs fits"))
        ml = _cached(out / "phase_ml.json", lambda: collect(ml_futs, "MVP-ML fits"))
    return truths_res, {j: dict(cs_f=cpu[j]["cs_f"], cs_o=cpu[j]["cs_o"], il=cpu[j]["il"], ifm=cpu[j]["ifm"],
                                ml=ml[j], gibbs=gibbs[j]) for j in jobs}


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--truths", default=",".join(t2.TRUTHS))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--gibbs-iters", type=int, default=4000)
    ap.add_argument("--eval-only", action="store_true", help="rescore saved fits.json")
    args = ap.parse_args()
    truths, ks = args.truths.split(","), [int(k) for k in args.ks.split(",")]
    n_resp, reps = t2.N_RESP, range(args.reps)
    if args.selftest:
        truths, ks, n_resp, reps = ["L1"], [8], 300, range(1)
    out = HERE / "out" / ("selftest" if args.selftest else "full")
    out.mkdir(parents=True, exist_ok=True)
    LOGF = open(out / "log.txt", "w")
    rfc = engine()
    log(f"T2 {'selftest' if args.selftest else 'full'}: truths {truths}, k {ks}, reps {len(reps)}, "
        f"N {n_resp}, engine backend {rfc.BACKEND}")
    jobs = [(tr, k, rep) for tr in truths for k in ks for rep in reps]
    data = {j: t2.make_data(*j, n_resp=n_resp) for j in jobs}
    for (tr, k, rep), (X, Y, _, _) in list(data.items())[:len(truths) * len(ks)]:
        log(f"  {tr} k={k}: take rates {Y.mean(0).round(2)}")
    t0 = time.time()
    fits_path = out / "fits.json"
    with app.run():
        if args.eval_only:
            saved = json.loads(fits_path.read_text())
            truths_res = {tuple(key.split("|")[:1]) + (int(key.split("|")[1]),): v for key, v in saved["truths"].items()}
            fits = {(f["truth"], f["k"], f["rep"]): f["fits"] for f in saved["fits"]}
        else:
            truths_res, fits = fit_all(jobs, data, truths, ks, args, t0, out)
            fits_path.write_text(json.dumps(dict(
                truths={f"{a}|{b}": v for (a, b), v in truths_res.items()},
                fits=[dict(truth=j[0], k=j[1], rep=j[2], fits=f) for j, f in fits.items()])))
            log(f"fits saved to {fits_path}")
        calls = {j: score_job.spawn(fits[j], truths_res[(j[0], j[1])], j[1], data[j][2], data[j][3]) for j in jobs}
        scored = {j: c.get() for j, c in calls.items()}
        log(f"scoring done ({time.time() - t0:.0f} s)")
    results = []
    orth = rfc.orthant_cdf
    for j in jobs:
        tr, k, rep = j
        rec = dict(truth=tr, k=k, rep=rep, N=n_resp, methods=scored[j])
        for name in ("ifm", "ml", "gibbs"):
            if "error" not in fits[j][name]:
                rec["methods"][name]["engine"] = engine_column(fits[j][name], truths_res[(tr, k)], k, orth)
        results.append(rec)
        m = rec["methods"]
        for n, v in m.items():
            if "error" in v:
                log(f"{tr} k={k} rep {rep} {n}: ERROR {v['error'][:200]}")
        log(f"{tr} k={k} rep {rep}: " + "; ".join(
            f"{n} turf {'F' if v.get('turf4_flip') else '-'}{v.get('turf4_regret_pct', np.nan):.2f}% "
            f"j2 {v.get('joint2_mae_pts', np.nan):.2f} t {v.get('seconds_total', np.nan):.0f}s"
            for n, v in m.items() if "error" not in v))
    (out / "results.json").write_text(json.dumps(dict(truths={f"{a}_{b}": v for (a, b), v in truths_res.items()},
                                                      results=results), indent=1))
    if args.selftest:
        bad = []
        for rec in results:
            for n, v in rec["methods"].items():
                if "error" in v:
                    bad.append(f"{n} failed")
            g = rec["methods"].get("gibbs", {})
            if g.get("corr_drift", 0) > 0.01:
                bad.append(f"Gibbs mean-correlation drift {g['corr_drift']:.3f} > 0.01")
        log("selftest " + ("OK" if not bad else "FAILED: " + "; ".join(bad)))
    log(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
