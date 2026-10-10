"""T6. Correlated-reach TURF on household-level basket data (design: PLAN.md T6).

    python analysis/T6/t6.py control     # GATE: simulated independent truth; run first
    python analysis/T6/t6.py real        # only after the control passes
    python analysis/T6/t6.py control --B 5 --sets 10   # quick wiring check

MVP reach by GHK on one Modal T4 (ephemeral app), checked against SciPy on the main fit.

Data: dunnhumby Complete Journey, SOFT DRINKS, the 30 product IDs fixed in PLAN.md. Observation =
household x window binary vector (bought the item at least once in the window); windows are weeks
1-27 and 28-53; households active in both windows. Marginals and Sigma by IFM on window 1
(intercept-only probit margins, pairwise correlation from each 2x2 table, nearest valid correlation
matrix if needed); scored against observed window-2 reach = share of households buying >= 1 item.

Reach estimators for a set S (both from the same fitted marginals mu = Phi^-1(p)):
  independence: 1 - prod_j (1 - Phi(mu_j))
  MVP:          1 - Phi_k(-mu_S; Sigma_S)          (SciPy MVN CDF, k <= 6)

SEs: household bootstrap (B resamples; prediction refits and rescores; decision rescoring of the
fixed portfolios). Kill rules: PLAN.md T6. Writes analysis/T6/out/<stage>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import multivariate_normal, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the Modal container the file is /root/t6.py
DATA = ROOT / "data" / "completejourney"
SEED = 20261009
ITEMS = ["5569230", "1053690", "8090521", "8090537", "844165", "5569471", "1092026", "1085604",
         "5569845", "1110572", "916381", "879755", "1132770", "6534480", "13511722", "8090509",
         "893501", "6534035", "8090532", "1138189", "5569374", "6534077", "868764", "1037894",
         "1076875", "882441", "1107553", "1074524", "947798", "991951"]
K_PRED = (1, 2, 3, 4, 5, 6)
M_GHK = 4096          # scrambled Sobol points per orthant (GHK, GPU)
CHECK_TOL = 1e-4      # GPU vs SciPy reach agreement required on the main fit's sets (0.01 pts)

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy", "scipy")  # t6.py imports scipy at module level
app = modal.App("orthant-rfc-t6-ghk")


@app.function(image=image, gpu="T4", timeout=1800)
def orthants(groups, M=M_GHK, seed=SEED, chunk=2000):
    """P(L z <= eta) for each row by GHK with fixed scrambled Sobol draws (same draws for every row:
    common random numbers). groups: {k: (eta (n, k), L (n, k, k))}. Returns {k: (n,) array}."""
    import torch
    from torch.special import log_ndtr, ndtri

    dev, f64 = torch.device("cuda"), torch.float64
    tiny, one = torch.finfo(f64).tiny, 1.0 - torch.finfo(f64).eps
    out, t0 = {}, time.time()
    for k, (eta, L) in groups.items():
        sob = torch.quasirandom.SobolEngine(k, scramble=True, seed=seed + k).draw(M, dtype=f64).to(dev)
        res = []
        for lo in range(0, len(eta), chunk):
            e = torch.as_tensor(eta[lo:lo + chunk], dtype=f64, device=dev)
            Lc = torch.as_tensor(L[lo:lo + chunk], dtype=f64, device=dev)
            n = len(e)
            logp = e.new_zeros(n, M)
            us = []
            for j in range(k):
                c = e[:, j, None]
                for i in range(j):
                    c = c + Lc[:, j, i, None] * us[i]
                lp = log_ndtr(c / Lc[:, j, j, None])
                logp = logp + lp
                if j < k - 1:
                    us.append(-ndtri((sob[None, :, j] * lp.exp()).clamp(tiny, one)))
            res.append(logp.exp().mean(1).cpu().numpy())
        out[k] = np.concatenate(res)
    torch.cuda.synchronize()
    return dict(p=out, seconds=time.time() - t0, gpu=torch.cuda.get_device_name())


GPU_SECONDS = [0.0]


def fill(pairs):
    """GPU MVP reach for (fit, S) pairs not yet cached; k = 1 is analytic."""
    groups, keys = {}, {}
    for fit, S in pairs:
        S = tuple(S)
        c = fit.setdefault("cache", {})
        if S in c or len(S) == 1:
            continue
        c[S] = None
        k = len(S)
        mu, R = fit["mu"][list(S)], fit["R"][np.ix_(S, S)]
        g = groups.setdefault(k, ([], []))
        g[0].append(-mu)
        g[1].append(np.linalg.cholesky(R))
        keys.setdefault(k, []).append((fit, S))
    if not groups:
        return
    r = orthants.remote({k: (np.array(e), np.array(L)) for k, (e, L) in groups.items()})
    GPU_SECONDS[0] += r["seconds"]
    for k, ks in keys.items():
        for (fit, S), p0 in zip(ks, r["p"][k]):
            fit["cache"][S] = 1 - float(p0)
K_DEC = (3, 4, 5, 6)
P_CLIP = 1e-4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    import pandas as pd
    import pyreadr

    t = pyreadr.read_r(str(DATA / "transactions.rds"))[None][["household_id", "basket_id", "product_id", "week"]]
    t["win"] = np.where(t.week <= 27, 1, 2)
    act = t.groupby(["household_id", "win"]).basket_id.nunique().unstack()
    hh = act.dropna().index
    s = t[t.product_id.isin(ITEMS) & t.household_id.isin(hh)].drop_duplicates(["household_id", "product_id", "win"])
    Y = {}
    for w in (1, 2):
        m = pd.crosstab(s[s.win == w].household_id, s[s.win == w].product_id)
        Y[w] = (m.reindex(index=hh, columns=ITEMS, fill_value=0).to_numpy() > 0).astype(np.int8)
    return Y[1], Y[2]


# ---------------------------------------------------------------------------
# Bivariate normal, copied from analysis/rho_test/run.py (Genz, "Numerical computation of rectangular bivariate and
# trivariate normal and t probabilities", 2004), vectorized over h, k; scalar r.
# ---------------------------------------------------------------------------
_GL = {
    6: ([0.1713244923791705, 0.3607615730481384, 0.4679139345726904],
        [0.9324695142031522, 0.6612093864662647, 0.2386191860831970]),
    12: ([.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659,
          0.2334925365383547, 0.2491470458134029],
         [0.9815606342467191, 0.9041172563704750, 0.7699026741943050, 0.5873179542866171,
          0.3678314989981802, 0.1252334085114692]),
    20: ([.01761400713915212, .04060142980038694, .06267204833410906, .08327674157670475,
          0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
          0.1491729864726037, 0.1527533871307259],
         [0.9931285991850949, 0.9639719272779138, 0.9122344282513259, 0.8391169718222188,
          0.7463319064601508, 0.6360536807265150, 0.5108670019508271, 0.3737060887154196,
          0.2277858511416451, 0.07652652113349733]),
}
TP = 2 * np.pi


def bvnu(h, k, r):
    """P(X > h, Y > k), standard bivariate normal with correlation r (scalar)."""
    h, k = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float))
    if r == 0:
        return ndtr(-h) * ndtr(-k)
    w, x = _GL[6 if abs(r) < .3 else 12 if abs(r) < .75 else 20]
    w, x = np.r_[w, w], np.r_[1 - np.array(x), 1 + np.array(x)]
    hk = h * k
    if abs(r) < .925:
        hs = (h * h + k * k) / 2
        asr = np.arcsin(r) / 2
        sn = np.sin(asr * x)
        bvn = np.zeros_like(h)
        for wl, sl in zip(w, sn):
            bvn += wl * np.exp((sl * hk - hs) / (1 - sl * sl))
        return np.clip(bvn * asr / TP + ndtr(-h) * ndtr(-k), 0, 1)
    if r < 0:
        k, hk = -k, -hk
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        as_ = 1 - r * r
        a = np.sqrt(as_)
        bs = (h - k) ** 2
        asr = -(bs / as_ + hk) / 2
        c = (4 - hk) / 8
        d = (12 - hk) / 80
        bvn = np.where(asr > -100, a * np.exp(asr) * (1 - c * (bs - as_) * (1 - d * bs) / 3 + c * d * as_ ** 2), 0.)
        b = np.sqrt(bs)
        sp_ = np.sqrt(TP) * ndtr(-b / a)
        bvn = np.where(hk > -100, bvn - np.exp(-hk / 2) * sp_ * b * (1 - c * bs * (1 - d * bs) / 3), bvn)
        a2 = a / 2
        term = np.zeros_like(h)
        for wl, xl in zip(w, x):
            xs = (a2 * xl) ** 2
            asl = -(bs / xs + hk) / 2
            spl = 1 + c * xs * (1 + 5 * d * xs)
            rs = np.sqrt(1 - xs)
            ep = np.exp(-(hk / 2) * xs / (1 + rs) ** 2) / rs
            term += np.where(asl > -100, wl * np.exp(asl) * (spl - ep), 0.)
        bvn = (a2 * term - bvn) / TP
    if r > 0:
        bvn = bvn + ndtr(-np.maximum(h, k))
    else:
        L = np.where(h < 0, ndtr(k) - ndtr(h), ndtr(-h) - ndtr(-k))
        bvn = np.where(h >= k, -bvn, L - bvn)
    return np.clip(bvn, 0, 1)


def bvn11(a, b, r):
    """P(Z1 < a, Z2 < b), corr r."""
    return float(bvnu(-a, -b, r))


def nearest_corr(R):
    w, V = np.linalg.eigh(R)
    if w.min() > 1e-6:
        return R, 0.0
    A = (V * np.maximum(w, 1e-6)) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / d[:, None] / d[None, :]
    return A, float(np.abs(A - R).max())


def fit_ifm(Y):
    """Intercept-only IFM: mu_j = Phi^-1(p_j); rho_jk solves Phi2(mu_j, mu_k; rho) = p11 (the 2x2 MLE
    with margins fixed)."""
    p = np.clip(Y.mean(0), P_CLIP, 1 - P_CLIP)
    mu = norm.ppf(p)
    J = Y.shape[1]
    Yf = Y.astype(float)
    P11 = (Yf.T @ Yf) / len(Y)
    R = np.eye(J)
    lo, hi = -0.99, 0.99
    for j in range(J):
        for k in range(j + 1, J):
            f = lambda r: bvn11(mu[j], mu[k], r) - P11[j, k]  # noqa: E731
            fl, fh = f(lo), f(hi)
            r = lo if fl >= 0 else hi if fh <= 0 else brentq(f, lo, hi, xtol=1e-5)
            R[j, k] = R[k, j] = r
    R2, moved = nearest_corr(R)
    return dict(mu=mu, p=p, R=R2, R_raw=R, psd_moved=moved)


def reach_ind(fit, S):
    return 1 - np.prod(1 - fit["p"][list(S)])


def reach_mvp(fit, S):
    if len(S) == 1:
        return fit["p"][list(S)[0]]
    return fit["cache"][tuple(S)]  # filled on GPU by fill()


def reach_mvp_scipy(fit, S):
    """SciPy (Genz) reference, for the GPU agreement check only."""
    S = list(S)
    mu, R = fit["mu"][S], fit["R"][np.ix_(S, S)]
    return 1 - multivariate_normal.cdf(-mu, mean=np.zeros(len(S)), cov=R, abseps=1e-5, releps=0)


def reach_obs(Y, S, w=None):
    hit = Y[:, list(S)].any(1).astype(float)
    return float(hit.mean()) if w is None else float((hit * w).sum() / w.sum())


def greedy(fit, fn, kmax=6):
    S, path = [], []
    for _ in range(kmax):
        if fn is reach_mvp:
            fill([(fit, S + [j]) for j in range(len(fit["p"])) if j not in S])
        best = max((j for j in range(len(fit["p"])) if j not in S), key=lambda j: fn(fit, S + [j]))
        S = S + [best]
        path.append(list(S))
    return path


def random_sets(J, n, rng):
    return {k: [sorted(rng.choice(J, k, replace=False).tolist()) for _ in range(n)] for k in K_PRED}


def pred_errors(fit, Y2, sets):
    """Mean |predicted - observed window-2 reach| (pts) per k and method; k = 1 sanity diff."""
    out = {}
    for k, ss in sets.items():
        ei = [abs(reach_ind(fit, S) - reach_obs(Y2, S)) for S in ss]
        em = [abs(reach_mvp(fit, S) - reach_obs(Y2, S)) for S in ss]
        out[k] = dict(ind=100 * float(np.mean(ei)), mvp=100 * float(np.mean(em)))
        if k == 1:
            out[k]["sanity_max_diff"] = 100 * max(abs(reach_ind(fit, S) - reach_mvp(fit, S)) for S in ss)
    return out


def pooled(e, ks, m):
    return float(np.mean([e[k][m] for k in ks]))


def run(stage, B, nsets):
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    Y1, Y2 = load()
    n, J = Y1.shape
    log(f"data: {n} households x {J} items; window-1 rates {100 * Y1.mean(0).min():.1f}-{100 * Y1.mean(0).max():.1f}%")
    if stage == "control":
        p1 = Y1.mean(0)
        rs = np.random.default_rng(SEED + 1)
        Y1 = (rs.random((n, J)) < p1).astype(np.int8)
        Y2 = (rs.random((n, J)) < p1).astype(np.int8)
        log("control: independent household vectors with the window-1 marginals, same household count")
    fit = fit_ifm(Y1)
    off = fit["R"][np.triu_indices(J, 1)]
    log(f"IFM fit: rho off-diagonal median {np.median(off):+.3f}, range {off.min():+.3f}..{off.max():+.3f}, "
        f"max|rho| {np.abs(off).max():.3f}; PSD projection moved {fit['psd_moved']:.4f}")

    sets = random_sets(J, nsets, rng)
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    t1 = time.time()
    bfits = [fit_ifm(Y1[idx]) for idx in idxs]
    log(f"{B} bootstrap IFM fits: {time.time() - t1:.0f} s (local CPU)")
    allsets = [S for k in sets for S in sets[k]]
    fill([(f, S) for f in [fit] + bfits for S in allsets])
    log(f"GPU reach: {sum(len(f['cache']) for f in [fit] + bfits)} orthants, {GPU_SECONDS[0]:.1f} GPU-s")
    chk = [S for S in allsets if len(S) > 1]
    gap = max(abs(reach_mvp(fit, S) - reach_mvp_scipy(fit, S)) for S in chk)
    log(f"GPU vs SciPy on the main fit's {len(chk)} sets: max |diff| {100 * gap:.4f} pts (tolerance {100 * CHECK_TOL:.2f})")
    if gap > CHECK_TOL:
        sys.exit("GPU reach disagrees with SciPy beyond tolerance: stop")
    err = pred_errors(fit, Y2, sets)
    log("prediction (pts): " + ", ".join(f"k={k} ind {e['ind']:.2f} mvp {e['mvp']:.2f}" for k, e in err.items()))

    # decision: greedy on the window-1 fit, nested paths
    gi, gm = greedy(fit, reach_ind), greedy(fit, reach_mvp)
    first_div = next((i + 1 for i in range(6) if sorted(gi[i]) != sorted(gm[i])), None)

    # household bootstrap
    boot_pred, boot_dec = [], []
    for b, (idx, fb) in enumerate(zip(idxs, bfits)):
        boot_pred.append(pred_errors(fb, Y2[idx], sets))
        w = np.bincount(idx, minlength=n).astype(float)
        boot_dec.append({k: 100 * (reach_obs(Y2, gm[k - 1], w) - reach_obs(Y2, gi[k - 1], w)) for k in K_DEC})
        if (b + 1) % max(1, B // 10) == 0:
            log(f"bootstrap {b + 1}/{B}")

    def se(f):
        return float(np.std([f(x) for x in boot_pred], ddof=1)) if B > 1 else float("nan")

    ks2 = (2, 3, 4, 5, 6)
    pred = dict(per_k={k: dict(**err[k], se_ind=se(lambda e: e[k]["ind"]), se_mvp=se(lambda e: e[k]["mvp"]),
                               se_diff=se(lambda e: e[k]["mvp"] - e[k]["ind"])) for k in err},
                pooled_2_6=dict(ind=pooled(err, ks2, "ind"), mvp=pooled(err, ks2, "mvp"),
                                se_diff=se(lambda e: pooled(e, ks2, "mvp") - pooled(e, ks2, "ind"))))
    dec = {}
    for k in K_DEC:
        d = 100 * (reach_obs(Y2, gm[k - 1]) - reach_obs(Y2, gi[k - 1]))
        sd = float(np.std([x[k] for x in boot_dec], ddof=1)) if B > 1 else float("nan")
        differ = sorted(gm[k - 1]) != sorted(gi[k - 1])
        counted = bool(differ and abs(d) >= max(0.5, 2 * sd))
        dec[k] = dict(mvp=[ITEMS[j] for j in gm[k - 1]], ind=[ITEMS[j] for j in gi[k - 1]], differ=differ,
                      reach_mvp=100 * reach_obs(Y2, gm[k - 1]), reach_ind=100 * reach_obs(Y2, gi[k - 1]),
                      diff=d, se=sd, counted_flip=counted, winner=("MVP" if d > 0 else "IND") if counted else None)

    # verdicts (PLAN.md T6)
    P, D = pred["pooled_2_6"], dec
    flips = [k for k in K_DEC if D[k]["counted_flip"]]
    v = {}
    if stage == "control":
        added = P["mvp"] - P["ind"]
        v["counted_flips"] = len(flips)
        v["added_error_pts"] = added
        v["added_error_limit"] = max(0.2, 0.05 * P["ind"])
        v["max_abs_rho"] = float(np.abs(off).max())
        v["PASS"] = bool(len(flips) == 0 and added <= v["added_error_limit"])
    else:
        rel = 1 - P["mvp"] / P["ind"]
        rel56 = {k: 1 - err[k]["mvp"] / err[k]["ind"] for k in (5, 6)}
        worse = [k for k in ks2 if err[k]["mvp"] - err[k]["ind"] > 2 * pred["per_k"][k]["se_diff"]]
        v["prediction"] = dict(rel_improvement_pooled=rel, rel_improvement_k5_k6=rel56, worse_by_2se_at=worse,
                               PASS=bool(rel >= 0.2 and all(r >= 0.2 for r in rel56.values()) and not worse))
        wins = sum(D[k]["winner"] == "MVP" for k in flips)
        v["decision"] = dict(counted_flips_at=flips, mvp_wins=wins,
                             PASS=bool(len(flips) >= 2 and wins > len(flips) / 2 and wins == len(flips)))
    v["k1_sanity_max_diff_pts"] = err[1]["sanity_max_diff"]
    v["first_divergence_k"] = first_div

    out = dict(stage=stage, integrator=f"GHK, scrambled Sobol M={M_GHK}, T4", gpu_seconds=GPU_SECONDS[0],
               gpu_vs_scipy_max_pts=100 * gap, households=n, J=J, B=B, sets_per_k=nsets, seed=SEED, seconds=time.time() - t0,
               rho_offdiag=dict(median=float(np.median(off)), min=float(off.min()), max=float(off.max())),
               psd_moved=fit["psd_moved"], p_window1=fit["p"].tolist(), prediction=pred, decision=dec,
               greedy_paths=dict(mvp=[[ITEMS[j] for j in s] for s in gm], ind=[[ITEMS[j] for j in s] for s in gi]),
               verdict=v)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"{stage}_gpu{'' if B >= 100 else '_check'}.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"verdict: {json.dumps(v, default=str)}")
    log(f"done in {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["control", "real"])
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--sets", type=int, default=200, help="random candidate sets per k")
    ap.add_argument("--control-waived", action="store_true",
                    help="user waived the failed control gate (2026-10-10, PLAN.md T6)")
    a = ap.parse_args()
    gate = HERE / "out" / "control_gpu.json"
    if a.control_waived and not gate.exists():
        gate = HERE / "out" / "control.json"  # the SciPy control run (failed, waived)
    if a.stage == "real" and not gate.exists():
        sys.exit("run the control gate first (PLAN.md T6)")
    if a.stage == "real" and not json.loads(gate.read_text())["verdict"]["PASS"] and not a.control_waived:
        sys.exit("control gate failed: stop and report (PLAN.md T6)")
    with modal.enable_output(), app.run():
        run(a.stage, a.B, a.sets)
