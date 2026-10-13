"""T14 Part A: T2's simulated MBC at k = 20 / 30 with the menu M = k (PLAN.md, T14).

    python analysis/T14/run_a.py selftest     # L1 and LC at k = 30, rep 0, all arms
    python analysis/T14/run_a.py full         # 4 truths x k {20, 30} x 20 reps; ML on L1/L2, reps 0-9
    python analysis/T14/run_a.py summary [selftest|full]

Arms: CS-F (analyst lift rule, cap 5), CS-O (oracle blocks), IL, MVP-IFM (multivariate-probit 0.2.3, pairwise,
unrepaired R kept and repaired here with its nearest_correlation), MVP-ML (mvp_fit's own code on one H100).
Truth decisions by GHK-QMC (2^16 scrambled Sobol, CPU); fitted probits' decisions by GHK-QMC (2^14), as T2, on a T4.
LC: T2's pairs (2m, 2m + 1) and triple (0, 2, 4); its pattern law factors into the 6-item block {0..5} and
the remaining pairs, so it is simulated and queried block by block (2^k enumeration is out at k = 20 / 30).
Every job's result is saved as it arrives (out/A_<mode>/<phase>/<truth>_<k>_<rep>.json); a rerun skips them.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
if len(HERE.parents) > 2:                                      # host only (not in the Modal container)
    sys.path.insert(0, str(HERE.parent / "T2"))
    sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))  # mvp_fit's fitting code (private sibling checkout)
    T2_FILE = HERE.parent / "T2" / "t2.py"
else:
    T2_FILE = Path("/root/t2.py")
import t2  # noqa: E402

KS = (20, 30)
ML_TRUTHS, ML_REPS = ("L1", "L2"), 10
ML_TIMEOUT = 600           # per H100 fit (step 0: <= ~1 min projected at k = 30)
py_image = (modal.Image.debian_slim(python_version="3.12")
            .pip_install("numpy==2.5.1", "scipy==1.16.0", "multivariate-probit==0.2.3")
            .add_local_file(T2_FILE, "/root/t2.py"))
fit_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]", "torch==2.5.1", "numpy", "scipy", "pydantic")
             .add_local_python_source("modal_gp_api")
             .add_local_file(T2_FILE, "/root/t2.py"))
app = modal.App("orthant-rfc-t14a")


def set_menu(k):
    """T2 with M = k: the module's menu size and its decision sets."""
    import itertools
    t2.M = k
    t2.PAIRS = list(itertools.combinations(range(k), 2))
    t2.TRIPLES = list(itertools.combinations(range(k), 3))
    t2.SETS4 = list(itertools.combinations(range(k), 4))


# ---------------------------------------------------------------------------
# LC by blocks
# ---------------------------------------------------------------------------
def lc_blocks(k):
    return [list(range(6))] + [[2 * m, 2 * m + 1] for m in range(3, k // 2)]


def lc_block_logits(tp, items, X):
    """Pattern log-weights over patterns(len(items)) for rows of X (n, M) -> (n, 2^s)."""
    P = t2.patterns(len(items))
    pos = {m: q for q, m in enumerate(items)}
    g = sum(t2.GAMMA2 * P[:, pos[i]] * P[:, pos[j]] for i, j in tp["pairs"] if i in pos and j in pos)
    if all(m in pos for m in tp["triple"]):
        g = g + t2.GAMMA3 * np.prod(P[:, [pos[m] for m in tp["triple"]]], 1)
    it = np.array(items)
    return (tp["c"][it] + tp["bl"][it] * X[:, it]) @ P.T + g


def lc_simulate(tp, X, rng):
    from scipy.special import logsumexp
    Y = np.zeros(X.shape, dtype=np.int8)
    for items in lc_blocks(tp["k"]):
        P = t2.patterns(len(items))
        lw = lc_block_logits(tp, items, X)
        pr = np.exp(lw - logsumexp(lw, 1, keepdims=True)).cumsum(1)
        idx = (pr < rng.random((len(X), 1))).sum(1).clip(max=len(P) - 1)
        Y[:, items] = P[idx]
    return Y


def simulate(tp, X, rng):
    return lc_simulate(tp, X, rng) if tp["truth"] == "LC" else t2.simulate(tp, X, rng)


def make_data(truth, k, rep):
    """t2.make_data with M = k and LC by blocks."""
    set_menu(k)
    tp = t2.truth_params(truth, k)
    rng = np.random.default_rng([t2.SEED, k, t2.TRUTHS.index(truth), rep])
    X = t2.LOGP[rng.integers(0, 3, (t2.N_RESP * t2.T_TRAIN, k))]
    Y = simulate(tp, X, rng)
    Xh = np.repeat(t2.holdout_menus(k), t2.N_RESP, 0)
    Yh = simulate(tp, Xh, rng)
    return X, Y, Xh, Yh


def truth_model(tp):
    from scipy.special import logsumexp
    if tp["truth"] != "LC":
        return t2.Probit(tp["a"], np.diag(tp["b"]), tp["R"], lambda u, c: t2.orth_qmc(u, c, m=2 ** 16, seed=1))
    blocks = []
    for items in lc_blocks(tp["k"]):
        def fn(x, items=items):
            lw = lc_block_logits(tp, items, x[None])[0]
            return np.exp(lw - logsumexp(lw))
        blocks.append((items, fn))
    return t2.Blocks(blocks)


# ---------------------------------------------------------------------------
# Remote jobs
# ---------------------------------------------------------------------------
@app.function(image=py_image, cpu=2.0, memory=4096, timeout=3600)
def truth_job(truth, k):
    set_menu(k)
    t0 = time.time()
    tm = truth_model(t2.truth_params(truth, k))
    return dict(dec=t2.decisions(tm, k), hold=t2.holdout_joints(tm, k), seconds=time.time() - t0)


@app.function(image=py_image, cpu=2.0, memory=4096, timeout=3600)
def fit_cpu(X, Y, truth, k):
    """CS-F, CS-O, IL and MVP-IFM, each timed; IFM's unrepaired R and the repair are recorded."""
    from multivariate_probit import MultivariateProbit
    from multivariate_probit._corr import nearest_correlation
    set_menu(k)
    out = {}
    t0 = time.time()
    sub = t2.analyst_subsets(Y)
    out["cs_f"] = dict(blocks=t2.fit_cs(X, Y, sub), subsets=sub, seconds=time.time() - t0)
    t0 = time.time()
    sub = t2.oracle_subsets(t2.truth_params(truth, k))
    out["cs_o"] = dict(blocks=t2.fit_cs(X, Y, sub), subsets=sub, seconds=time.time() - t0)
    t0 = time.time()
    out["il"] = dict(blocks=t2.fit_cs(X, Y, [[m] for m in range(k)]), seconds=time.time() - t0)
    t0 = time.time()
    mp = MultivariateProbit(inner="linear", dependence="pairwise", cv=5, random_state=0,
                            project_correlation=False).fit(X, Y.astype(int))
    raw = mp.correlation_
    R = nearest_correlation(raw)
    E = mp.decision_function(np.vstack([np.zeros(k), np.eye(k)]))
    ev = float(np.linalg.eigvalsh((raw + raw.T) / 2).min())
    out["ifm"] = dict(c=E[0].tolist(), B=(E[1:] - E[0]).T.tolist(), R=R.tolist(), seconds=time.time() - t0,
                      raw_min_eig=ev, psd_repaired=bool(ev < 1e-8), repair_fro=float(np.linalg.norm(raw - R)),
                      raw_offdiag_abs_max=float(np.abs(raw[np.triu_indices(k, 1)]).max()))
    return out


@app.function(image=fit_image, gpu="H100", timeout=ML_TIMEOUT)
def fit_ml(X, Y, seed):
    """As T2's MVP-ML (mvp_fit: 512 draws, float32 GHK, no SEs), on one H100."""
    import torch

    import modal_gp_api as g
    t0 = time.time()
    out = g.mvp_fit_core(X, Y.astype(float), None, g.MvpFitParams(n_draws=512, seed=seed, alpha=1e-6, se=False),
                         torch.device("cuda"))
    d, p = Y.shape[1], X.shape[1]
    return dict(c=np.asarray(out["intercept"], float).tolist(), B=np.asarray(out["coef"], float).reshape(d, p).tolist(),
                R=np.asarray(out["corr"], float).reshape(d, d).tolist(), n_iter=int(out["n_iter"]),
                converged=bool(out["converged"]), grad_max=float(out["grad_max"]), seconds=time.time() - t0,
                gpu=torch.cuda.get_device_name())


def orth_qmc_torch(upper, cov, m=16384, seed=0):
    """t2.orth_qmc on a GPU: P(Z <= upper), Z ~ N(0, cov), by GHK with m scrambled Sobol points shared across
    rows, float64."""
    import torch
    from torch.special import ndtr, ndtri
    dev, f64 = torch.device("cuda"), torch.float64
    n, d = upper.shape
    w = torch.quasirandom.SobolEngine(max(d - 1, 1), scramble=True, seed=seed).draw(m, dtype=f64).to(dev)
    out = np.empty(n)
    chunk = max(1, int(2 ** 26 // (m * d)))
    for s in range(0, n, chunk):
        L = torch.linalg.cholesky(torch.as_tensor(cov[s:s + chunk], dtype=f64, device=dev))
        u = torch.as_tensor(upper[s:s + chunk], dtype=f64, device=dev)
        logp = torch.zeros(len(u), m, dtype=f64, device=dev)
        z = torch.zeros(len(u), m, d, dtype=f64, device=dev)
        for j in range(d):
            c = (u[:, j, None] - torch.einsum("nk,nmk->nm", L[:, j, :j], z[:, :, :j])) / L[:, j, j, None]
            p = ndtr(c)
            logp += p.clamp_min(1e-300).log()
            if j < d - 1:
                z[:, :, j] = ndtri((w[None, :, j] * p).clamp(1e-300, 1 - 1e-16))
        out[s:s + chunk] = logp.exp().mean(1).cpu().numpy()
    return out


def build(fit):
    if "blocks" in fit:
        return t2.cs_model(fit["blocks"])
    R = np.array(fit["R"])
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return t2.Probit(fit["c"], fit["B"], R, t2.orth_qmc)


@app.function(image=fit_image, gpu="T4", timeout=1800)
def score_job(fits, truth_res, k, Xh, Yh):
    """As T2's scoring; the probits' GHK-QMC (decisions 2^14 points, holdout patterns 2^11) runs on the GPU."""
    set_menu(k)
    t2.orth_qmc = orth_qmc_torch
    out = {}
    for name, fit in fits.items():
        if "error" in fit:
            out[name] = fit
            continue
        t0 = time.time()
        model = build(fit)
        dec = t2.decisions(model, k)
        hj = t2.holdout_joints(model, k)
        t_dec = time.time() - t0
        rec = dict(t2.score(dec, truth_res["dec"], hj, truth_res["hold"]),
                   picks={d: int(np.argmax(dec[key])) for d, key in (("bundle2", "joint2"), ("bundle3", "joint3"),
                                                                   ("turf4", "reach4"))},
                   holdout_ll=float(model.pattern_logprob(Xh, Yh).mean()),
                   seconds_fit=fit["seconds"], seconds_decide=t_dec, seconds_total=fit["seconds"] + t_dec)
        if "blocks" not in fit:
            iu = np.triu_indices(k, 1)
            rec["corr_offdiag_abs_median"] = float(np.median(np.abs(np.array(fit["R"])[iu])))
            rec["corr_offdiag_abs_max"] = float(np.abs(np.array(fit["R"])[iu]).max())
        for key in ("subsets", "converged", "n_iter", "grad_max", "raw_min_eig", "psd_repaired", "repair_fro",
                    "raw_offdiag_abs_max"):
            if key in fit:
                rec[key] = fit[key]
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def tag(j):
    return "_".join(map(str, j))


def run(mode):
    out = HERE / "out" / f"A_{mode}"
    for ph in ("truths", "fits", "ml", "scores"):
        (out / ph).mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def save(ph, j, obj):
        (out / ph / f"{tag(j)}.json").write_text(json.dumps(obj))

    def load(ph, j):
        p = out / ph / f"{tag(j)}.json"
        return json.loads(p.read_text()) if p.exists() else None

    if mode == "selftest":
        jobs = [("L1", 30, 0), ("LC", 30, 0)]
        ml_jobs = [("L1", 30, 0)]
    else:
        jobs = [(tr, k, rep) for tr in t2.TRUTHS for k in KS for rep in range(20)]
        ml_jobs = [(tr, k, rep) for tr in ML_TRUTHS for k in KS for rep in range(ML_REPS)]
    cells = sorted({(tr, k) for tr, k, _ in jobs})
    data = {}
    for j in jobs:
        data[j] = make_data(*j)
    log(f"T14 A {mode}: {len(jobs)} datasets, {len(ml_jobs)} ML fits; data built")
    t0 = time.time()

    def guarded(ph, j, fn):
        r = load(ph, j)
        if r is not None:
            return r
        try:
            r = fn()
        except Exception as e:
            r = dict(error=f"{type(e).__name__}: {e}"[:500])
        save(ph, j, r)
        log(f"{ph} {tag(j)} {'ERROR ' + r['error'][:200] if 'error' in r else 'ok'} ({time.time() - t0:.0f} s)")
        return r

    with modal.enable_output(), app.run():
        with ThreadPoolExecutor(64) as ex:
            ft = {c: ex.submit(guarded, "truths", c + (0,), lambda c=c: truth_job.remote(*c)) for c in cells}
            ff = {j: ex.submit(guarded, "fits", j, lambda j=j: fit_cpu.remote(data[j][0], data[j][1], j[0], j[1]))
                  for j in jobs}
            fm = {j: ex.submit(guarded, "ml", j, lambda j=j: fit_ml.remote(data[j][0], data[j][1], j[2]))
                  for j in ml_jobs}
            truths = {c: f.result() for c, f in ft.items()}
            fits = {j: f.result() for j, f in ff.items()}
            ml = {j: f.result() for j, f in fm.items()}

            def score(j):
                tr = truths[(j[0], j[1])]
                if "error" in tr or "error" in fits[j]:
                    return dict(error="missing truth or CPU fits")
                fs = {n: fits[j][n] for n in ("cs_f", "cs_o", "il", "ifm")}
                if j in ml:
                    fs["ml"] = ml[j]
                return score_job.remote(fs, tr, j[1], data[j][2], data[j][3])

            fsc = {j: ex.submit(guarded, "scores", j, lambda j=j: score(j)) for j in jobs}
            for f in fsc.values():
                f.result()
    log(f"done ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
METHODS = ["cs_f", "cs_o", "il", "ifm", "ml"]
DECS = ["bundle2", "bundle3", "turf4"]


def summary(mode):
    out = HERE / "out" / f"A_{mode}"
    rows = []
    for p in sorted((out / "scores").glob("*.json")):
        tr, k, rep = p.stem.split("_")
        rows.append(dict(truth=tr, k=int(k), rep=int(rep), m=json.loads(p.read_text())))
    ok = lambda r, m: m in r["m"] and "error" not in r["m"][m]  # noqa: E731
    lines = ["truth k method n | flips b2/b3/turf4 (material >0.5%) | mean regret % b2/b3/turf4 | dMarg MAE pts | "
             "dAny |err| pts | holdout LL | t_med s"]
    for tr, k in sorted({(r["truth"], r["k"]) for r in rows}):
        cell = [r for r in rows if r["truth"] == tr and r["k"] == k]
        for m in METHODS:
            rs = [r["m"][m] for r in cell if ok(r, m)]
            if not rs:
                continue
            fl = "/".join(f"{sum(x[f'{d}_flip'] for x in rs)}({sum(x[f'{d}_flip'] and x[f'{d}_regret_pct'] > .5 for x in rs)})"
                          for d in DECS)
            rg = "/".join(f"{np.mean([x[f'{d}_regret_pct'] for x in rs]):.2f}" for d in DECS)
            extra = ""
            if "corr_offdiag_abs_median" in rs[0]:
                extra = (f" | |rho| med {np.median([x['corr_offdiag_abs_median'] for x in rs]):.3f} "
                         f"max {max(x['corr_offdiag_abs_max'] for x in rs):.3f}")
            if "psd_repaired" in rs[0]:
                extra += (f" | PSD repair {sum(x['psd_repaired'] for x in rs)}/{len(rs)}, "
                          f"|R - R~|_F max {max(x['repair_fro'] for x in rs):.3f}")
            if "converged" in rs[0]:
                extra += f" | conv {sum(x['converged'] for x in rs)}/{len(rs)}"
            lines.append(f"{tr} {k} {m:5s} {len(rs):2d} | {fl:24s} | {rg:16s} | "
                         f"{np.mean([x['dmarg_mae_pts'] for x in rs]):.2f} | "
                         f"{np.mean([abs(x['d_any_linked_err_pts']) for x in rs]):.2f} | "
                         f"{np.mean([x['holdout_ll'] for x in rs]):.3f} | "
                         f"{np.median([x['seconds_total'] for x in rs]):.0f}{extra}")
        lines.append("")
    pool = [r for r in rows if r["truth"] in ("L1", "L2") and ok(r, "cs_f") and ok(r, "ifm")]
    res = {}
    if pool:
        fl = {m: sum(r["m"][m][f"{d}_flip"] for r in pool for d in DECS) for m in ("cs_f", "cs_o", "ifm")}
        diff = np.array([np.mean([r["m"]["cs_f"][f"{d}_regret_pct"] - r["m"]["ifm"][f"{d}_regret_pct"] for d in DECS])
                         for r in pool])
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        gap = fl["cs_f"] - fl["ifm"]
        verdict = ("PASS" if gap >= 48 and diff.mean() > 2 * se else "KILL" if gap <= 24 else "INCONCLUSIVE")
        res = dict(datasets=len(pool), decisions=3 * len(pool), flips=fl, flip_gap=gap,
                   regret_gap_mean=float(diff.mean()), regret_gap_se=float(se), verdict=verdict)
        lines.append(f"Kill pool (L1 + L2, k 20/30): {len(pool)} datasets x 3 = {3 * len(pool)} decisions. Flips CS-F "
                     f"{fl['cs_f']}, CS-O {fl['cs_o']}, MVP-IFM {fl['ifm']}; gap {gap} (PASS >= 48, KILL <= 24). "
                     f"Mean regret gap CS-F - IFM {diff.mean():.3f} pts per decision (paired SE over datasets "
                     f"{se:.3f}). Verdict: {verdict}.")
    l0 = [r["m"]["ifm"] for r in rows if r["truth"] == "L0" and ok(r, "ifm")]
    if l0:
        med = float(np.median([x["corr_offdiag_abs_median"] for x in l0]))
        lines.append(f"L0 guard: IFM |rho| median {med:.3f} (<= 0.05), max {max(x['corr_offdiag_abs_max'] for x in l0):.3f}; "
                     f"PSD repair triggered {sum(x['psd_repaired'] for x in l0)}/{len(l0)}.")
        res["L0"] = dict(rho_median=med, psd_repaired=sum(x["psd_repaired"] for x in l0), n=len(l0))
    ml_rows = [r for r in rows if ok(r, "ml") and ok(r, "ifm")]
    if ml_rows:
        agree = sum(r["m"]["ml"]["picks"][d] == r["m"]["ifm"]["picks"][d] for r in ml_rows for d in DECS)
        mlf = sum(r["m"]["ml"][f"{d}_flip"] for r in ml_rows for d in DECS)
        iff = sum(r["m"]["ifm"][f"{d}_flip"] for r in ml_rows for d in DECS)
        lines.append(f"Q2: ML on {len(ml_rows)} datasets; IFM and ML pick the same set on {agree}/{3 * len(ml_rows)} "
                     f"decisions (flips IFM {iff}, ML {mlf}); "
                     f"decisions; ML fit seconds median {np.median([r['m']['ml']['seconds_fit'] for r in ml_rows]):.0f}, "
                     f"max {max(r['m']['ml']['seconds_fit'] for r in ml_rows):.0f}.")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else "full")
    else:
        run(cmd)
