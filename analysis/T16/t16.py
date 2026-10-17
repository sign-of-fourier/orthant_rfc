"""T16. T6 against a practitioner incumbent: per-household independence (design: analysis/T16/TEST.md).

    python analysis/T16/t16.py smoke      # local: a few hundred households, short chain, SciPy MVP
    python analysis/T16/t16.py fit        # Modal CPU: HB-IND on the real window-1 blocks (no scoring)
    python analysis/T16/t16.py control    # GATE: simulated heterogeneity-only truth; run first
    python analysis/T16/t16.py real       # only after the control passes

Arms: IND and MVP are T6's (t6.fit_ifm; MVP reach by GHK on a Modal T4, t6.orthants). HB-IND: household-level
product intercepts by bayesm::rhierMnlRwMixture (hb_bin.R) on household x block x product purchases, independent
across products given the household. BB-IND: per-product beta-binomial on block counts. HB-IND and BB-IND are
marginal-matched to the window-1 rates, so only the dependence differs between arms.
Writes analysis/T16/out/<stage>.json.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import betaln, expit, logit
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "T6"))
import t6  # noqa: E402

OUT = HERE / "out"
SEED = 20261017
NB = 6                                     # blocks in window 1
BLOCKS = np.array_split(np.arange(1, 28), NB)  # weeks 1-27
HB_R, HB_KEEP, HB_ND = 20000, 50, 200      # iterations, thinning, posterior draws used (last kept)
K2 = (2, 3, 4, 5, 6)

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("r-base-core", "r-cran-bayesm")
         .pip_install("numpy", "scipy")
         .add_local_file(HERE / "hb_bin.R", "/root/hb_bin.R")
         .add_local_file(HERE.parent / "T6" / "t6.py", "/root/t6.py"))
app = modal.App("orthant-rfc-t16-hb", image=image)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hb_local(Yb, R, keep, nd, seed, script=HERE / "hb_bin.R"):
    """Yb (n, NB, J) int8 -> beta draws (nd, n, J), log-likelihood trace, seconds."""
    n, nb, J = Yb.shape
    with tempfile.TemporaryDirectory() as d:
        fi, fo = Path(d) / "y.bin", Path(d) / "beta.bin"
        np.ascontiguousarray(Yb, dtype=np.int8).tofile(fi)
        t0 = time.time()
        r = subprocess.run(["Rscript", str(script), str(fi), str(n), str(nb), str(J), str(fo), str(R), str(keep),
                            str(nd), str(seed)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"hb_bin.R failed:\n{r.stderr[-2000:]}")
        beta = np.fromfile(fo, dtype=np.float64).reshape(nd, n, J)
        ll = np.fromfile(str(fo) + ".loglike", dtype=np.float64)
        return beta, ll, time.time() - t0


@app.function(cpu=1.0, memory=6144, timeout=6 * 3600)
def hb_remote(Yb, R, keep, nd, seed):
    beta, ll, sec = hb_local(Yb, R, keep, nd, seed, script=Path("/root/hb_bin.R"))
    return beta.astype(np.float32), ll, sec


def load():
    """T6's households and products; window-1 vectors, window-2 vectors, window-1 block array (n, NB, J)."""
    import pandas as pd
    import pyreadr

    t = pyreadr.read_r(str(t6.DATA / "transactions.rds"))[None][["household_id", "basket_id", "product_id", "week"]]
    t["win"] = np.where(t.week <= 27, 1, 2)
    act = t.groupby(["household_id", "win"]).basket_id.nunique().unstack()
    hh = act.dropna().index
    s = t[t.product_id.isin(t6.ITEMS) & t.household_id.isin(hh)]
    Y = {}
    for w in (1, 2):
        sw = s[s.win == w].drop_duplicates(["household_id", "product_id"])
        m = pd.crosstab(sw.household_id, sw.product_id)
        Y[w] = (m.reindex(index=hh, columns=t6.ITEMS, fill_value=0).to_numpy() > 0).astype(np.int8)
    Yb = np.zeros((len(hh), NB, len(t6.ITEMS)), np.int8)
    for b, wk in enumerate(BLOCKS):
        sb = s[s.week.isin(wk)].drop_duplicates(["household_id", "product_id"])
        m = pd.crosstab(sb.household_id, sb.product_id)
        Yb[:, b] = (m.reindex(index=hh, columns=t6.ITEMS, fill_value=0).to_numpy() > 0)
    assert (Yb.any(1) == Y[1]).all()
    return Y[1], Y[2], Yb


def match(Q, p):
    """Shift each product's q on the logit scale so its mean over draws and households equals p_j."""
    L = logit(np.clip(Q, 1e-9, 1 - 1e-9))
    out = np.empty_like(Q)
    for j in range(Q.shape[-1]):
        c = brentq(lambda c: expit(L[..., j] + c).mean() - p[j], -20, 20, xtol=1e-10)
        out[..., j] = expit(L[..., j] + c)
    return out


def q_hb(beta, p):
    """(D, n, J) window probabilities from HB beta draws, marginal-matched."""
    return match(1 - (1 - expit(beta.astype(np.float64))) ** NB, p).astype(np.float32)


def q_bb(Yb, p):
    """(1, n, J) posterior-predictive P(>= 1 purchase in NB new blocks) under a per-product beta-binomial."""
    c = Yb.sum(1).astype(float)
    Q = np.empty(c.shape)
    for j in range(c.shape[1]):
        cj = c[:, j]

        def nll(t):
            a, b = np.exp(t)
            return -(betaln(a + cj, b + NB - cj) - betaln(a, b)).sum()

        a, b = np.exp(minimize(nll, [0.0, 2.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8)).x)
        Q[:, j] = 1 - np.exp(betaln(a + cj, b + 2 * NB - cj) - betaln(a + cj, b + NB - cj))
    return match(Q[None], p).astype(np.float32)


def hh_reach(Q, S):
    """Per-household reach of set S, averaged over draws: (n,)."""
    return 1 - np.prod(1 - Q[:, :, list(S)].astype(np.float64), axis=2).mean(0)


def greedy_q(Q, kmax=6):
    none, S, path = np.ones(Q.shape[:2]), [], []
    for _ in range(kmax):
        cand = [j for j in range(Q.shape[2]) if j not in S]
        best = max(cand, key=lambda j: 1 - (none * (1 - Q[:, :, j])).mean())
        none = none * (1 - Q[:, :, best])
        S = S + [best]
        path.append(list(S))
    return path


def fill_scipy(pairs):
    for fit, S in pairs:
        c = fit.setdefault("cache", {})
        if len(S) > 1 and tuple(S) not in c:
            c[tuple(S)] = float(t6.reach_mvp_scipy(fit, S))


def holm(p):
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def score(stage, Y1, Y2, Yb, beta, B, nsets, fill, truth_q=None):
    """Prediction and decision for all four arms; verdicts per analysis/T16/TEST.md."""
    rng = np.random.default_rng(SEED)
    n, J = Y1.shape
    fit = t6.fit_ifm(Y1)
    p = fit["p"]
    Qh, Qb = q_hb(beta, p), q_bb(Yb, p)
    sets = t6.random_sets(J, nsets, rng)
    allsets = [S for k in sets for S in sets[k]]
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    bfits = [t6.fit_ifm(Y1[idx]) for idx in idxs]
    fill([(f, S) for f in [fit] + bfits for S in allsets])
    rh = {tuple(S): hh_reach(Qh, S) for S in allsets}
    rb = {tuple(S): hh_reach(Qb, S) for S in allsets}

    def errs(f, Yo, idx):
        e = {}
        for k, ss in sets.items():
            obs = [t6.reach_obs(Yo, S) for S in ss]
            sel = slice(None) if idx is None else idx
            pr = dict(ind=[t6.reach_ind(f, S) for S in ss], mvp=[t6.reach_mvp(f, S) for S in ss],
                      hb=[rh[tuple(S)][sel].mean() for S in ss], bb=[rb[tuple(S)][sel].mean() for S in ss])
            e[k] = {m: 100 * float(np.mean(np.abs(np.array(v) - obs))) for m, v in pr.items()}
        return e

    err = errs(fit, Y2, None)
    boot = [errs(fb, Y2[idx], idx) for idx, fb in zip(idxs, bfits)]
    pool = lambda e, m: float(np.mean([e[k][m] for k in K2]))  # noqa: E731
    sd = lambda f: float(np.std([f(e) for e in boot], ddof=1)) if B > 1 else float("nan")  # noqa: E731
    per_k = {k: dict(**err[k], se_hb_minus_mvp=sd(lambda e: e[k]["hb"] - e[k]["mvp"])) for k in err}
    z = [per_k[k]["hb"] - per_k[k]["mvp"] for k in K2]
    pk = [2 * norm.sf(abs(d) / per_k[k]["se_hb_minus_mvp"]) if B > 1 else float("nan") for d, k in zip(z, K2)]
    for k, h in zip(K2, holm(np.array(pk)) if B > 1 else [float("nan")] * len(K2)):
        per_k[k]["holm_p"] = float(h)
    P = {m: pool(err, m) for m in ("ind", "mvp", "hb", "bb")}
    P["se_hb_minus_mvp"] = sd(lambda e: pool(e, "hb") - pool(e, "mvp"))
    k1 = max(abs(t6.reach_ind(fit, S) - rh[tuple(S)].mean()) for S in sets[1])
    k1 = 100 * max(k1, max(abs(t6.reach_ind(fit, S) - rb[tuple(S)].mean()) for S in sets[1]))

    gm, gh = t6.greedy(fit, t6.reach_mvp) if fill is not fill_scipy else greedy_mvp_scipy(fit), greedy_q(Qh)
    dec = {}
    for k in (3, 4, 5, 6):
        d = 100 * (t6.reach_obs(Y2, gm[k - 1]) - t6.reach_obs(Y2, gh[k - 1]))
        ws = [np.bincount(idx, minlength=n).astype(float) for idx in idxs]
        bs = [100 * (t6.reach_obs(Y2, gm[k - 1], w) - t6.reach_obs(Y2, gh[k - 1], w)) for w in ws]
        s = float(np.std(bs, ddof=1)) if B > 1 else float("nan")
        differ = sorted(gm[k - 1]) != sorted(gh[k - 1])
        counted = bool(differ and abs(d) >= max(0.5, 2 * s))
        dec[k] = dict(mvp=[t6.ITEMS[j] for j in gm[k - 1]], hb=[t6.ITEMS[j] for j in gh[k - 1]], differ=differ,
                      reach_mvp=100 * t6.reach_obs(Y2, gm[k - 1]), reach_hb=100 * t6.reach_obs(Y2, gh[k - 1]),
                      diff=d, se=s, counted_flip=counted, winner=("MVP" if d > 0 else "HB") if counted else None)

    v = dict(k1_sanity_max_diff_pts=k1)
    if stage == "control":
        edge = P["hb"] - P["mvp"]
        qt = truth_q.ravel()
        qo = match(truth_q[None], p)[0]  # oracle: true q, marginal-matched to the same window-1 rates
        tr = lambda q, S: 1 - np.prod(1 - q[:, list(S)], axis=1).mean()  # noqa: E731
        pooled_k = lambda f: 100 * float(np.mean([np.mean([f(S) for S in sets[k]]) for k in K2]))  # noqa: E731
        rec = pooled_k(lambda S: abs(rh[tuple(S)].mean() - tr(qo, S)))
        v.update(mvp_edge_pts=edge, edge_limit=max(0.2, 0.05 * P["hb"]), recovery_vs_oracle_pts=rec,
                 recovery_limit=0.2, raw_vs_true_reach_pts=pooled_k(lambda S: abs(rh[tuple(S)].mean() - tr(truth_q, S))),
                 oracle_vs_true_reach_pts=pooled_k(lambda S: abs(tr(qo, S) - tr(truth_q, S))),
                 recovery_corr_q=float(np.corrcoef(qt, Qh.mean(0).ravel())[0, 1]),
                 recovery_corr_q_bb=float(np.corrcoef(qt, Qb[0].ravel())[0, 1]))
        v["PASS"] = bool(edge <= v["edge_limit"] and rec <= 0.2)
    else:
        rel = 1 - P["mvp"] / P["hb"]
        closes = (P["ind"] - P["hb"]) / (P["ind"] - P["mvp"])
        label = "PASS" if rel >= 0.2 else "FAIL" if P["hb"] - P["mvp"] <= 0.5 else "INCONCLUSIVE"
        flips = [k for k in dec if dec[k]["counted_flip"]]
        wins = sum(dec[k]["winner"] == "MVP" for k in flips)
        v.update(prediction=dict(rel_improvement=rel, label=label, gap_closed_by_hb=closes,
                                 mostly_heterogeneity=bool(closes >= 0.5)),
                 decision=dict(counted_flips_at=flips, mvp_wins=wins,
                               PASS=bool(len(flips) >= 2 and wins == len(flips))))
    return dict(prediction=dict(per_k=per_k, pooled_2_6=P), decision=dec, verdict=v,
                greedy_paths=dict(mvp=[[t6.ITEMS[j] for j in s] for s in gm],
                                  hb=[[t6.ITEMS[j] for j in s] for s in gh]))


def greedy_mvp_scipy(fit, kmax=6):
    S, path = [], []
    for _ in range(kmax):
        cand = [j for j in range(len(fit["p"])) if j not in S]
        fill_scipy([(fit, S + [j]) for j in cand])
        best = max(cand, key=lambda j: t6.reach_mvp(fit, S + [j]))
        S = S + [best]
        path.append(list(S))
    return path


def hb_summary(ll, sec):
    q = len(ll) // 4
    return dict(seconds=sec, loglike_last=float(ll[-1]),
                drift_q3_q4=float(abs(ll[-q:].mean() - ll[-2 * q:-q].mean()) / abs(ll[-q:].mean())))


def simulate(beta, rng):
    """Heterogeneity-only truth: one posterior draw per household; blocks independent given the household."""
    pt = expit(beta[-1].astype(np.float64))
    n, J = pt.shape
    Yb = (rng.random((n, NB, J)) < pt[:, None]).astype(np.int8)
    Y2 = (rng.random((n, NB, J)) < pt[:, None]).any(1).astype(np.int8)
    return Yb.any(1).astype(np.int8), Y2, Yb, 1 - (1 - pt) ** NB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "fit", "control", "real"])
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--sets", type=int, default=200)
    ap.add_argument("--households", type=int, default=300, help="smoke only")
    ap.add_argument("--R", type=int, default=HB_R)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    Y1, Y2, Yb = load()
    log(f"data: {Y1.shape[0]} households x {Y1.shape[1]} products; block rates "
        f"{100 * Yb.mean((0, 1)).min():.1f}-{100 * Yb.mean((0, 1)).max():.1f}%, window-1 "
        f"{100 * Y1.mean(0).min():.1f}-{100 * Y1.mean(0).max():.1f}%")
    real_beta = OUT / "hb_real_beta.npy"

    if a.stage == "smoke":
        rng = np.random.default_rng(SEED)
        h = rng.choice(len(Y1), a.households, replace=False)
        Y1, Y2, Yb = Y1[h], Y2[h], Yb[h]
        R = min(a.R, 2000)
        beta, ll, sec = hb_local(Yb, R, 10, 50, SEED)
        log(f"HB local: {len(h)} households, R = {R}: {sec:.0f} s; {hb_summary(ll, sec)}")
        r = score("smoke", Y1, Y2, Yb, beta, min(a.B, 5), min(a.sets, 10), fill_scipy)
        rng2 = np.random.default_rng(SEED + 1)
        s1, s2, sb, qt = simulate(beta, rng2)
        b2, _, _ = hb_local(sb, R, 10, 50, SEED + 1)
        rc = score("control", s1, s2, sb, b2, min(a.B, 5), min(a.sets, 10), fill_scipy, truth_q=qt)
        out = dict(stage="smoke", households=len(h), R=R, hb_seconds=sec, real=r, control=rc,
                   seconds=time.time() - t0)
        (OUT / "smoke.json").write_text(json.dumps(out, indent=1, default=str))
        log("smoke real pooled: " + json.dumps(r["prediction"]["pooled_2_6"]))
        log("smoke real verdict: " + json.dumps(r["verdict"], default=str))
        log("smoke control verdict: " + json.dumps(rc["verdict"], default=str))
        log(f"done in {time.time() - t0:.0f} s")
        return

    with modal.enable_output(), app.run():
        if a.stage == "fit":
            beta, ll, sec = hb_remote.remote(Yb, a.R, HB_KEEP, HB_ND, SEED)
            np.save(real_beta, beta)
            s = hb_summary(ll, sec)
            (OUT / "fit.json").write_text(json.dumps(dict(R=a.R, keep=HB_KEEP, ndraws=HB_ND, **s), indent=1))
            log(f"HB real fit: {s}")
            return
        if not real_beta.exists():
            sys.exit("run `t16.py fit` first")
        beta = np.load(real_beta)
        gate = OUT / "control.json"
        if a.stage == "real" and not (gate.exists() and json.loads(gate.read_text())["verdict"]["PASS"]):
            sys.exit("control gate missing or failed: stop and report (analysis/T16/TEST.md)")
        truth_q = None
        if a.stage == "control":
            Y1, Y2, Yb, truth_q = simulate(beta, np.random.default_rng(SEED + 1))
            beta, ll, sec = hb_remote.remote(Yb, a.R, HB_KEEP, HB_ND, SEED + 1)
            log(f"HB control fit: {hb_summary(ll, sec)}")
        with t6.app.run():
            r = score(a.stage, Y1, Y2, Yb, beta, a.B, a.sets, t6.fill, truth_q=truth_q)
        r.update(stage=a.stage, B=a.B, sets_per_k=a.sets, seed=SEED, gpu_seconds=t6.GPU_SECONDS[0],
                 seconds=time.time() - t0)
        (OUT / f"{a.stage}.json").write_text(json.dumps(r, indent=1, default=str))
        log("pooled 2-6: " + json.dumps(r["prediction"]["pooled_2_6"]))
        log("verdict: " + json.dumps(r["verdict"], default=str))
        log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
