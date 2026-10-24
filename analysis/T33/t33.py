"""T33. MBC at k = 20 / 30: MVP against a Sawtooth-style serial cross-effects logit (analysis/T33/TEST.md).

    python analysis/T33/t33.py run [--truths L0,L1,L2,LC,CL] [--ks 20,30] [--reps 0-19] [--workers 2]
    python analysis/T33/t33.py summary

T14 Part A's datasets, truths and arms are reused as they are (out/A_full in analysis/T14): the T14 truth decisions,
and CS-F / CS-O / IL / MVP-IFM scores. New here:
  SCX  serial cross-effects logit (Sawtooth MBC's recommended model for large menus): one logit per item, in
       order of train frequency, on all M log prices (own + cross-price effects) and the 0/1 picks of the items
       earlier in the order (conditional dependencies). The chain rule makes the joint exact; L2 penalty C = 100
       only to stop separation (as T19's M1). Decisions by ancestral simulation (common random numbers), holdout
       pattern log-likelihood exact.
  CL   a new truth that is itself a serial logit (home advantage for SCX): items in index order, intercepts set to
       T2's take rates, own log-price slope 1.6 x B_PRICE (T2's LC logit scale), conditional dependencies 0.9
       within each half of the linked set and 0.3 across. Truth decisions by a 2M-draw simulation. MVP-IFM and
       CS-F are fitted on it here (MVP decisions by GHK, 4,096 Sobol points).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T14 = HERE.parent / "T14"
sys.path[:0] = [str(HERE.parent / "T2"), str(T14)]
import t2  # noqa: E402

OUT = HERE / "out"
S_FIT = 1_000_000          # ancestral draws for a fitted SCX's decisions
S_TRUTH = 2_000_000        # draws for the CL truth's decisions
C_L2 = 100.0
G_IN, G_OUT = 0.9, 0.3


def set_menu(k):
    import itertools
    t2.M = k
    t2.PAIRS = list(itertools.combinations(range(k), 2))
    t2.TRIPLES = list(itertools.combinations(range(k), 3))
    t2.SETS4 = list(itertools.combinations(range(k), 4))


# ---------------------------------------------------------------------------
# Serial logit (fitted SCX and the CL truth share it)
# ---------------------------------------------------------------------------
class Serial:
    """Item order[q] is a logit on [1, x (M log prices), y of order[:q]]: W[q] = (intercept, price coefs (M,),
    dependence coefs (q,))."""

    def __init__(self, order, W):
        self.order, self.W = list(order), W

    @classmethod
    def fit(cls, X, Y):
        from sklearn.linear_model import LogisticRegression
        order = list(np.argsort(-Y.mean(0), kind="stable"))
        W = []
        for q, j in enumerate(order):
            F = np.hstack([X, Y[:, order[:q]]])
            lr = LogisticRegression(C=C_L2, max_iter=5000).fit(F, Y[:, j])
            W.append(np.r_[lr.intercept_, lr.coef_[0]])
        return cls(order, W)

    def logits(self, q, x, Yp):
        w = self.W[q]
        M = len(x) if x.ndim == 1 else x.shape[1]
        lin = w[0] + (x @ w[1:M + 1] if x.ndim == 1 else x @ w[1:M + 1])
        return lin + Yp @ w[M + 1:]

    def sample(self, x, U):
        """Draws (S, M) at one price vector x (M,), from uniforms U (S, M) (common random numbers)."""
        S, M = U.shape
        Y = np.zeros((S, M), dtype=np.int8)
        for q, j in enumerate(self.order):
            p = 1 / (1 + np.exp(-self.logits(q, x, Y[:, self.order[:q]].astype(np.float32))))
            Y[:, j] = U[:, q] < p
        return Y

    def pattern_logprob(self, X, Y):
        lp = np.zeros(len(X))
        for q, j in enumerate(self.order):
            z = self.logits(q, X, Y[:, self.order[:q]].astype(float))
            lp += np.where(Y[:, j] == 1, -np.logaddexp(0, -z), -np.logaddexp(0, z))
        return lp


def cl_truth(k):
    rng = np.random.default_rng([t2.SEED, k])
    take = rng.uniform(.10, .40, k)                                   # T2's take rates (same draw order)
    h = (k + 1) // 2
    G = np.full((k, k), G_OUT)
    G[:h, :h] = G[h:, h:] = G_IN
    W = []
    for q in range(k):
        g = G[q, :q]
        a = np.log(take[q] / (1 - take[q])) - g @ take[:q]            # centre so margins stay near take
        price = np.zeros(k)
        price[q] = 1.6 * t2.B_PRICE
        W.append(np.r_[a, price, g])
    return Serial(range(k), W)


def sample_decisions(model, k, S, seed):
    """t2.decisions and t2.holdout_joints by simulation; draws at x0 and x1 share uniforms."""
    rng = np.random.default_rng(seed)
    U = rng.random((S, k), dtype=np.float32)
    x0 = np.zeros(k)
    x1 = x0.copy()
    x1[0] = np.log(1.2)
    Y0 = model.sample(x0, U)
    Y1 = model.sample(x1, U)
    bits = 1 << np.arange(k, dtype=np.int64)
    u, cnt = np.unique(Y0.astype(np.int64) @ bits, return_counts=True)
    w = cnt / S
    Yf = Y0.astype(np.float32)
    m0 = Yf.mean(0)
    C2 = (Yf.T @ Yf) / S
    j2 = np.array([C2[i, j] for i, j in t2.PAIRS])
    T3 = {}
    for i in range(k):
        T3[i] = ((Yf * Yf[:, i:i + 1]).T @ Yf) / S
    j3 = np.array([T3[i][j, l] for i, j, l in t2.TRIPLES])
    masks = np.array([bits[list(s)].sum() for s in t2.SETS4])
    reach4 = np.array([w[(u & mk) != 0].sum() for mk in masks])
    lift = j2 / np.array([m0[i] * m0[j] for i, j in t2.PAIRS])
    any0 = 1 - (Y0.sum(1) == 0).mean()
    any1 = 1 - (Y1.sum(1) == 0).mean()
    dec = dict(marg=m0.tolist(), joint2=j2.tolist(), joint3=j3.tolist(), lift2=lift.tolist(), reach4=reach4.tolist(),
               dmarg=(Y1.mean(0) - m0).tolist(), d_any_linked=float(any1 - any0))
    hj = []
    for x in t2.holdout_menus(k):
        Yh = model.sample(x, U).astype(np.float32)
        Ch = (Yh.T @ Yh) / S
        hj.append([float(Ch[i, j]) for i, j in t2.PAIRS])
    return dec, hj


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def cl_data(k, rep):
    set_menu(k)
    tm = cl_truth(k)
    rng = np.random.default_rng([t2.SEED, k, 99, rep])
    X = t2.LOGP[rng.integers(0, 3, (t2.N_RESP * t2.T_TRAIN, k))]
    Y = serial_rows(tm, X, rng)
    Xh = np.repeat(t2.holdout_menus(k), t2.N_RESP, 0)
    Yh = serial_rows(tm, Xh, rng)
    return X, Y, Xh, Yh


def serial_rows(model, X, rng):
    """One draw per row of X (n, M)."""
    n, M = X.shape
    Y = np.zeros((n, M), dtype=np.int8)
    U = rng.random((n, M))
    for q, j in enumerate(model.order):
        z = model.logits(q, X, Y[:, model.order[:q]].astype(float))
        Y[:, j] = U[:, q] < 1 / (1 + np.exp(-z))
    return Y


def truth_cl(k):
    f = OUT / "truths" / f"CL_{k}.json"
    if f.exists():
        return json.loads(f.read_text())
    set_menu(k)
    t0 = time.time()
    dec, hj = sample_decisions(cl_truth(k), k, S_TRUTH, [t2.SEED, k, 777])
    r = dict(dec=dec, hold=hj, seconds=time.time() - t0)
    f.write_text(json.dumps(r))
    return r


def job(truth, k, rep):
    try:
        return _job(truth, k, rep)
    except Exception as e:                                                   # noqa: BLE001
        return truth, k, rep, f"ERROR {type(e).__name__}: {str(e)[:200]}"


def _job(truth, k, rep):
    f = OUT / "scores" / f"{truth}_{k}_{rep}.json"
    if f.exists():
        return truth, k, rep, "cached"
    set_menu(k)
    import run_a
    run_a.set_menu(k)
    if truth == "CL":
        X, Y, Xh, Yh = cl_data(k, rep)
        tr = truth_cl(k)
    else:
        X, Y, Xh, Yh = run_a.make_data(truth, k, rep)
        tr = json.loads((T14 / "out" / "A_full" / "truths" / f"{truth}_{k}_0.json").read_text())
    set_menu(k)
    out = {}
    t0 = time.time()
    m = Serial.fit(X, Y)
    t_fit = time.time() - t0
    dec, hj = sample_decisions(m, k, S_FIT, [t2.SEED, k, rep, 5])
    out["scx"] = dict(t2.score(dec, tr["dec"], hj, tr["hold"]), holdout_ll=float(m.pattern_logprob(Xh, Yh).mean()),
                      seconds_fit=t_fit, seconds_total=time.time() - t0,
                      picks={d: int(np.argmax(dec[key])) for d, key in (("bundle2", "joint2"), ("bundle3", "joint3"),
                                                                      ("turf4", "reach4"))})
    if truth == "CL":                                   # MVP-IFM and CS-F fitted here
        from multivariate_probit import MultivariateProbit
        from multivariate_probit._corr import nearest_correlation
        t0 = time.time()
        mp = MultivariateProbit(inner="linear", dependence="pairwise", cv=5, random_state=0,
                                project_correlation=True).fit(X, Y.astype(int))   # nearest valid R, as T14's repair
        R = nearest_correlation(mp.correlation_)
        E = mp.decision_function(np.vstack([np.zeros(k), np.eye(k)]))
        pm = t2.Probit(E[0], (E[1:] - E[0]).T, (R + R.T) / 2, lambda u, c: t2.orth_qmc(u, c, m=4096))
        np.fill_diagonal(pm.R, 1.0)
        dec, hj = t2.decisions(pm, k), t2.holdout_joints(pm, k)
        out["ifm"] = dict(t2.score(dec, tr["dec"], hj, tr["hold"]), holdout_ll=float(pm.pattern_logprob(Xh, Yh).mean()),
                          seconds_total=time.time() - t0)
        t0 = time.time()
        sub = t2.analyst_subsets(Y)
        cm = t2.cs_model(t2.fit_cs(X, Y, sub))
        dec, hj = t2.decisions(cm, k), t2.holdout_joints(cm, k)
        out["cs_f"] = dict(t2.score(dec, tr["dec"], hj, tr["hold"]), holdout_ll=float(cm.pattern_logprob(Xh, Yh).mean()),
                           seconds_total=time.time() - t0)
    f.write_text(json.dumps(out))
    return truth, k, rep, f"scx fit {t_fit:.1f} s, total {out['scx']['seconds_total']:.0f} s"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
DECS = ["bundle2", "bundle3", "turf4"]


def summary():
    rows = []
    for p in sorted((OUT / "scores").glob("*.json")):
        tr, k, rep = p.stem.split("_")
        m = json.loads(p.read_text())
        if tr != "CL":
            t14 = json.loads((T14 / "out" / "A_full" / "scores" / p.name).read_text())
            m = {**{n: t14[n] for n in ("cs_f", "ifm") if n in t14 and "error" not in t14[n]}, **m}
        rows.append(dict(truth=tr, k=int(k), rep=int(rep), m=m))
    lines = ["truth k method n | flips b2/b3/turf4 (material > 0.5%) | mean regret % b2/b3/turf4 | holdout LL"]
    for tr, k in sorted({(r["truth"], r["k"]) for r in rows}):
        cell = [r for r in rows if (r["truth"], r["k"]) == (tr, k)]
        for name in ("cs_f", "ifm", "scx"):
            rs = [r["m"][name] for r in cell if name in r["m"]]
            if not rs:
                continue
            fl = "/".join(f"{sum(x[f'{d}_flip'] for x in rs)}({sum(x[f'{d}_flip'] and x[f'{d}_regret_pct'] > .5 for x in rs)})"
                          for d in DECS)
            rg = "/".join(f"{np.mean([x[f'{d}_regret_pct'] for x in rs]):.2f}" for d in DECS)
            lines.append(f"{tr} {k} {name:4s} {len(rs):2d} | {fl:24s} | {rg:16s} | "
                         f"{np.mean([x['holdout_ll'] for x in rs]):.3f}")
        lines.append("")

    def pool_verdict(truths, a, b, name):
        pool = [r for r in rows if r["truth"] in truths and a in r["m"] and b in r["m"]]
        if not pool:
            return None
        fa = sum(r["m"][a][f"{d}_flip"] for r in pool for d in DECS)
        fb = sum(r["m"][b][f"{d}_flip"] for r in pool for d in DECS)
        diff = np.array([np.mean([r["m"][a][f"{d}_regret_pct"] - r["m"][b][f"{d}_regret_pct"] for d in DECS])
                         for r in pool])
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        lines.append(f"{name}: {len(pool)} datasets x 3 = {3 * len(pool)} decisions. Flips {a} {fa}, {b} {fb}; gap "
                     f"{fa - fb}. Mean regret gap {a} - {b} {diff.mean():.3f} pts per decision (paired SE {se:.3f}).")
        return fa - fb, diff.mean(), se

    r = pool_verdict(("L1", "L2"), "scx", "ifm", "Kill pool (L1 + L2, k 20/30), SCX vs MVP-IFM")
    if r:
        gap, d, se = r
        v = "PASS" if gap >= 48 and d > 2 * se else "KILL" if gap <= 24 else "INCONCLUSIVE"
        lines.append(f"  Verdict (PASS: gap >= 48 and regret gap > 2 SE; KILL: gap <= 24): {v}")
    r = pool_verdict(("CL",), "ifm", "scx", "Guard (CL truth), MVP-IFM vs SCX")
    if r:
        lines.append(f"  Guard (MVP-IFM flips no more than 24 above SCX's): {'holds' if r[0] <= 24 else 'FAILS'}")
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--truths", default="L1,L2,L0,LC,CL")
    ap.add_argument("--ks", default="20,30")
    ap.add_argument("--reps", default="0-19")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for d in ("truths", "scores"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    if a.cmd == "summary":
        return summary()
    lo, hi = (int(x) for x in a.reps.split("-")) if "-" in a.reps else (int(a.reps), int(a.reps))
    jobs = [(tr, int(k), rep) for rep in range(lo, hi + 1) for tr in a.truths.split(",") for k in a.ks.split(",")]
    for k in a.ks.split(","):
        if "CL" in a.truths:
            truth_cl(int(k))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for tr, k, rep, msg in ex.map(job, *zip(*jobs)):
            print(f"[{time.strftime('%H:%M:%S')}] {tr} k={k} r{rep}: {msg} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
