"""T3. Real basket data: coded subsets vs multivariate probit on a linked set larger than the
cap (DESIGN.md).

    python analysis/T3/run.py --selftest           # 300 households, 1 control rep per truth
    python analysis/T3/run.py [--cluster taco]     # all households, 10 control reps per truth

Data build local (t3.load). Fits off this machine:
  - CS-F, CS-O, IL and MVP-IFM (multivariate-probit 0.2.3, pairwise): Modal CPU, one per dataset;
    the gate (largest set linked at rho >= 0.2 >= 6 items) is checked on the real IFM fit first
    and the run stops if it fails;
  - MVP-ML: the deployed mvp_fit service (full ML by GHK on a GPU, no SEs), one call per dataset;
  - controls: S-CS (simulated from the real CS-F fit) start as soon as the CPU phase is back,
    S-MVP (from the real MVP-ML fit) once that fit is back.
Scoring (Modal CPU, one per model): the 2^M pattern distribution averaged over holdout trips
(exact for coded subsets, 4096 scrambled-Sobol draws per trip for probits) and the holdout
pattern log-likelihood (GHK, 2048 points). Real-data disagreements are adjudicated on the holdout
by a household cluster bootstrap; the engine (rfc.orthant_cdf) recomputes the real-data probit
decisions as a second column. Phases are cached in out/<mode>/phase_*.json; results.json at the end.
"""
import argparse
import importlib.util
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE   # /root inside containers
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "T2"))
import t3  # noqa: E402

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.5.1", "scipy==1.16.0", "multivariate-probit==0.2.3")
         .add_local_file(HERE / "t3.py", "/root/t3.py")
         .add_local_file(HERE.parent / "T2" / "t2.py", "/root/t2.py"))
app = modal.App("orthant-rfc-t3")
LOGF = None
METHODS = ["cs_f", "cs_o", "il", "ifm", "ml"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


# ---------------------------------------------------------------------------
# Remote jobs
# ---------------------------------------------------------------------------
@app.function(image=image, cpu=2.0, memory=6144, timeout=3600)
def fit_cpu(X, Y):
    """IL, lifts, CS-F, MVP-IFM, CS-O (blocks from the IFM correlation), each timed."""
    import t3
    from multivariate_probit import MultivariateProbit

    M, p = Y.shape[1], X.shape[1]
    out = {}
    t0 = time.time()
    il = t3.fit_blocks(X, Y, [[m] for m in range(M)])
    out["il"] = dict(blocks=il, seconds=time.time() - t0)
    t0 = time.time()
    raw, res = t3.lifts(X, Y, il)
    sub = t3.analyst_subsets(res)
    out["cs_f"] = dict(blocks=t3.fit_blocks(X, Y, sub), subsets=sub, seconds=time.time() - t0 + out["il"]["seconds"],
                       lift_raw=np.nan_to_num(raw).tolist(), lift_resid=np.nan_to_num(res).tolist(),
                       subsets_raw_lift=t3.analyst_subsets(raw))
    t0 = time.time()
    mp = MultivariateProbit(inner="linear", dependence="pairwise", cv=5, random_state=0).fit(X, Y.astype(int))
    E = mp.decision_function(np.vstack([np.zeros(p), np.eye(p)]))
    out["ifm"] = dict(c=E[0].tolist(), B=(E[1:] - E[0]).T.tolist(), R=mp.correlation_.tolist(),
                      seconds=time.time() - t0)
    t0 = time.time()
    sub = t3.oracle_subsets(np.array(out["ifm"]["R"]))
    out["cs_o"] = dict(blocks=t3.fit_blocks(X, Y, sub), subsets=sub, seconds=time.time() - t0)
    return out


@app.function(image=image, cpu=1.0, memory=6144, timeout=3600)
def score_job(fit, Xh, Yh, m_sim):
    """Averaged pattern distribution over the holdout trips and the holdout log-likelihood."""
    import t2
    import t3

    M = Yh.shape[1]
    t0 = time.time()
    if "blocks" in fit:
        q = t3.cs_avg_dist(fit["blocks"], Xh, M)
        t_dec = time.time() - t0
        ll = t3.cs_holdout_ll(fit["blocks"], Xh, Yh)
    else:
        q = t3.probit_avg_dist(fit, Xh, m=m_sim)
        t_dec = time.time() - t0
        ll = t3.probit_holdout_ll(fit, Xh, Yh, lambda u, c: t2.orth_qmc(u, c, m=2048))
    return dict(q=q.tolist(), holdout_ll=float(ll.mean()), seconds_decide=t_dec)


def _remote_client():
    """multivariate_probit's fitter="modal" client (repo source; the installed 0.2.3 predates it)."""
    spec = importlib.util.spec_from_file_location(
        "mvp_remote", Path.home() / "projects" / "multivariate_probit" / "src" / "multivariate_probit" / "_remote.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_ml(client, X, Y, seed):
    t0 = time.time()
    out = client.fit_remote(X, Y.astype(float), n_draws=512, seed=seed, alpha=1e-6, se=False)
    d, p = Y.shape[1], X.shape[1]
    return dict(c=np.asarray(out["intercept"], float).tolist(),
                B=np.asarray(out["coef"], float).reshape(d, p).tolist(),
                R=np.asarray(out["corr"], float).reshape(d, d).tolist(),
                n_iter=int(out["n_iter"]), converged=bool(out["converged"]), seconds=time.time() - t0)


def _try(fn):
    try:
        return fn()
    except Exception as e:  # recorded, not fatal
        return dict(error=repr(e)[:500])


def _cached(path, fn):
    if path.exists():
        return json.loads(path.read_text())
    res = fn()
    path.write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------------------
# Engine column (real data): per-trip orthants for the decision quantities, averaged
# ---------------------------------------------------------------------------
def engine_quantities(fit, Xh, M, orth):
    import itertools
    eta = t3.probit_eta(fit, Xh)
    R = t3.clean_R(fit["R"])
    n = len(Xh)

    def avg(S, sign):
        S = list(S)
        cov = np.broadcast_to(R[np.ix_(S, S)], (n, len(S), len(S))).copy()
        return float(np.mean(orth(np.ascontiguousarray(sign * eta[:, S]), cov)))

    from scipy.stats import norm
    marg = norm.cdf(eta).mean(0)
    pairs = list(itertools.combinations(range(M), 2))
    J2 = np.zeros((M, M))
    for i, j in pairs:
        J2[i, j] = J2[j, i] = avg((i, j), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = J2 / marg[:, None]
    np.fill_diagonal(cond, -np.inf)
    Q = dict(bundle2=np.array([J2[i, j] for i, j in pairs]),
             bundle3=np.array([avg(t, 1.0) for t in itertools.combinations(range(M), 3)]),
             turf4=np.array([1 - avg(S, -1.0) for S in itertools.combinations(range(M), 4)]),
             **{f"rec{a}": cond[a] for a in range(M)})
    return {k: int(np.argmax(Q[k])) for k in t3.decision_keys(M)}


# ---------------------------------------------------------------------------
def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="taco", choices=list(t3.CLUSTERS))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--controls", type=int, default=10, help="replicates per control truth")
    ap.add_argument("--m-sim", type=int, default=4096)
    args = ap.parse_args()
    n_hh, n_ctrl = (300, 1) if args.selftest else (None, args.controls)
    mode = ("selftest_" if args.selftest else "full_") + args.cluster
    out = HERE / "out" / mode
    out.mkdir(parents=True, exist_ok=True)
    LOGF = open(out / "log.txt", "a")
    t_start = time.time()

    dpath = out / "data.pkl"
    if dpath.exists():
        D = pickle.loads(dpath.read_bytes())
    else:
        D = t3.load(args.cluster, ROOT / "data" / "completejourney", n_hh=n_hh)
        dpath.write_bytes(pickle.dumps(D))
    X, Y, tr, names = D["X"], D["Y"], D["train"], D["names"]
    M = len(names)
    log(f"T3 {mode}: {D['n_hh']} households, {len(Y)} trips ({tr.sum()} training), M = {M}, "
        f"X {X.shape[1]} columns; rates {dict(zip(names, D['rate'].round(4)))}")

    client = _remote_client()
    ctrl_jobs = [("S-MVP", r) for r in range(n_ctrl)] + [("S-CS", r) for r in range(n_ctrl)]
    with modal.enable_output(), app.run(), ThreadPoolExecutor(2 * n_ctrl + 2) as ex:
        # real data: CPU phase, gate, then ML
        real_cpu = _cached(out / "phase_real_cpu.json", lambda: fit_cpu.remote(X[tr], Y[tr]))
        g = t3.gate(np.array(real_cpu["ifm"]["R"]), names)
        (out / "gate.json").write_text(json.dumps(g, indent=1))
        log(f"CPU phase done ({time.time() - t_start:.0f} s); CS-F subsets "
            f"{[[names[m] for m in s] for s in real_cpu['cs_f']['subsets']]}; gate {g}")
        if not g["passed"]:
            log("gate FAILED: the linked set fits inside the cap; stopping (DESIGN.md, Gate)")
            return
        ml_path = out / "phase_real_ml.json"
        ml_fut = None if ml_path.exists() else ex.submit(lambda: _try(lambda: fit_ml(client, X[tr], Y[tr], 0)))

        # controls: simulate on every trip, fit on training trips
        def ctrl_data(truth, r, truth_fit):
            rng = np.random.default_rng(t3.SEED + (100 if truth == "S-MVP" else 200) + r)
            return (t3.simulate_probit(truth_fit, X, rng) if truth == "S-MVP"
                    else t3.simulate_cs(truth_fit["blocks"], X, M, rng))

        ctrl_cpu_path, ctrl_ml_path = out / "phase_ctrl_cpu.json", out / "phase_ctrl_ml.json"
        need_ctrl = not (ctrl_cpu_path.exists() and ctrl_ml_path.exists())
        cdata, cpu_calls, ml_futs = {}, {}, {}

        def launch(truth, truth_fit):
            for r in range(n_ctrl):
                j = (truth, r)
                cdata[j] = ctrl_data(truth, r, truth_fit)
                if need_ctrl:
                    cpu_calls[j] = fit_cpu.spawn(X[tr], cdata[j][tr])
                    ml_futs[j] = ex.submit(lambda j=j, r=r: _try(lambda: fit_ml(client, X[tr], cdata[j][tr], 1 + r)))

        launch("S-CS", real_cpu["cs_f"])
        real_ml = _cached(ml_path, lambda: ml_fut.result())
        if "error" in real_ml:
            log(f"real MVP-ML failed: {real_ml['error']}")
            return
        log(f"real MVP-ML done ({time.time() - t_start:.0f} s, fit {real_ml['seconds']:.0f} s, "
            f"converged {real_ml['converged']})")
        launch("S-MVP", real_ml)
        ctrl_cpu = _cached(ctrl_cpu_path, lambda: {f"{a}|{r}": _try(c.get) for (a, r), c in cpu_calls.items()})
        log(f"control CPU fits done ({time.time() - t_start:.0f} s)")
        ctrl_ml = _cached(ctrl_ml_path, lambda: {f"{a}|{r}": f.result() for (a, r), f in ml_futs.items()})
        log(f"control MVP-ML fits done ({time.time() - t_start:.0f} s)")

        # scoring
        Xh, hh_h = X[~tr], D["hh"][~tr]
        fits = {("real", 0): dict(real_cpu, ml=real_ml)}
        for (a, r) in ctrl_jobs:
            c = ctrl_cpu[f"{a}|{r}"]
            fits[(a, r)] = dict({m: c.get(m, c) for m in METHODS[:4]}, ml=ctrl_ml[f"{a}|{r}"])
        Yh = {("real", 0): Y[~tr], **{j: cdata[j][~tr] for j in ctrl_jobs}}
        score_path = out / "phase_score.json"
        if score_path.exists():
            scored = json.loads(score_path.read_text())
        else:
            calls = {}
            for j, f in fits.items():
                for m in METHODS:
                    if "error" not in f[m]:
                        calls[f"{j[0]}|{j[1]}|{m}"] = score_job.spawn(f[m], Xh, Yh[j], args.m_sim)
            calls["truth|S-MVP"] = score_job.spawn(real_ml, Xh, Y[~tr], 4 * args.m_sim)
            calls["truth|S-CS"] = score_job.spawn(real_cpu["cs_f"], Xh, Y[~tr], 0)
            scored = {k: _try(c.get) for k, c in calls.items()}
            score_path.write_text(json.dumps(scored))
        log(f"scoring done ({time.time() - t_start:.0f} s)")

    # real data: picks, adjudication, engine column
    sys.path.insert(0, str(ROOT / "research" / "rfc_vs_exact"))
    import rfc
    C = t3.hh_pattern_counts(hh_h, Y[~tr], D["n_hh"])
    q_obs = C.sum(0) / C.sum()
    real = dict(picks={}, holdout_ll={}, seconds_fit={}, seconds_decide={}, adjudication={}, engine={})
    for m in METHODS:
        s = scored.get(f"real|0|{m}", dict(error="not scored"))
        if "error" in s:
            real["picks"][m] = s
            continue
        real["picks"][m] = t3.picks(np.array(s["q"]), M)
        real["holdout_ll"][m] = s["holdout_ll"]
        real["seconds_fit"][m] = fits[("real", 0)][m]["seconds"]
        real["seconds_decide"][m] = s["seconds_decide"]
    real["picks"]["observed"] = t3.picks(q_obs, M)
    for a, b in (("ml", "cs_f"), ("ifm", "cs_f"), ("ml", "cs_o"), ("ml", "il"), ("ml", "ifm")):
        if "error" not in real["picks"][a] and "error" not in real["picks"][b]:
            real["adjudication"][f"{a}_vs_{b}"] = t3.adjudicate(real["picks"][a], real["picks"][b], C, M)
    for m in ("ifm", "ml"):
        t0 = time.time()
        p = engine_quantities(fits[("real", 0)][m], Xh, M, rfc.orthant_cdf)
        real["engine"][m] = dict(picks=p, seconds_decide=time.time() - t0,
                                 differs_from_qmc=[k for k in p if p[k] != real["picks"][m][k]])
    log(f"real data: picks {json.dumps(real['picks'])}")

    # controls: flips vs truth
    controls = {}
    for a in ("S-MVP", "S-CS"):
        qt = np.array(scored[f"truth|{a}"]["q"])
        for r in range(n_ctrl):
            rec = {}
            for m in METHODS:
                s = scored.get(f"{a}|{r}|{m}", dict(error="not scored"))
                rec[m] = s if "error" in s else dict(score=t3.score_vs_truth(np.array(s["q"]), qt, M),
                                                    holdout_ll=s["holdout_ll"])
            controls[f"{a}|{r}"] = rec
    res = dict(cluster=args.cluster, names=names, cols=D["cols"], n_hh=D["n_hh"], n_trips=len(Y),
               n_train=int(tr.sum()), rate=D["rate"].tolist(), carried=D["carried"], lp_dropped=D["lp_dropped"],
               trip_size_categories=D["H"], gate=g, subsets=dict(cs_f=real_cpu["cs_f"]["subsets"],
                                                                 cs_f_raw_lift=real_cpu["cs_f"]["subsets_raw_lift"],
                                                                 cs_o=real_cpu["cs_o"]["subsets"]),
               lift_resid=real_cpu["cs_f"]["lift_resid"], lift_raw=real_cpu["cs_f"]["lift_raw"],
               R_ifm=real_cpu["ifm"]["R"], R_ml=real_ml["R"], ml_converged=real_ml["converged"],
               real=real, controls=controls,
               ctrl_ml_converged=[ctrl_ml[k].get("converged") for k in ctrl_ml], seconds_total=time.time() - t_start)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    log(f"done ({time.time() - t_start:.0f} s); summary: python analysis/T3/summarize.py {out / 'results.json'}")


if __name__ == "__main__":
    main()
