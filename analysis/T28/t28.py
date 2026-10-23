"""T28. TURF reach line at scale: stepwise vs exhaustive / wide search (design: analysis/T28/TEST.md).

    python analysis/T28/t28.py prep      # household x product acceptable sets, top 100 SOFT DRINKS
    python analysis/T28/t28.py gate      # Gate 0: ILP optimum vs stepwise (+ swap), CPU

Data: dunnhumby Complete Journey, SOFT DRINKS; products ranked by window-1 household rate (weeks 1-27), the top J
for J in {30, 50, 70, 100} (nested). Households active in both windows (as T6). A household accepts an item if it
bought it at least once in the window. Reach of a line S = share of households accepting >= 1 item of S; the
search runs on window 1, window 2 (weeks 28-53) is the holdout.

Arms: forward stepwise (strict), stepwise + best-improvement 1-swap (strong incumbent), max-coverage ILP (SciPy
milp / HiGHS; the proven optimum). Writes analysis/T28/out/<stage>.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "completejourney"
OUT = HERE / "out"
CATEGORY = "SOFT DRINKS"
JMAX = 100
JS = (30, 50, 70, 100)
MS = (4, 6, 8, 10, 12, 15)
ILP_LIMIT = 600.0     # seconds per cell


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prep():
    import pandas as pd
    import pyreadr

    t = pyreadr.read_r(str(DATA / "transactions.rds"))[None][["household_id", "basket_id", "product_id", "week"]]
    p = list(pyreadr.read_r(str(DATA / "products.rda")).values())[0][["product_id", "product_category"]]
    t["win"] = np.where(t.week <= 27, 1, 2)
    act = t.groupby(["household_id", "win"]).basket_id.nunique().unstack()
    hh = act.dropna().index
    cat = set(p.product_id[p.product_category == CATEGORY])
    s = t[t.product_id.isin(cat) & t.household_id.isin(hh)].drop_duplicates(["household_id", "product_id", "win"])
    rate = s[s.win == 1].groupby("product_id").household_id.nunique().sort_values(ascending=False, kind="stable")
    items = rate.index[:JMAX].tolist()
    Y = {}
    for w in (1, 2):
        m = pd.crosstab(s[s.win == w].household_id, s[s.win == w].product_id)
        Y[w] = (m.reindex(index=hh, columns=items, fill_value=0).to_numpy() > 0)
    np.savez_compressed(OUT / "sets.npz", Y1=Y[1], Y2=Y[2], items=np.array(items, dtype=str))
    log(f"{len(hh)} households, {len(items)} items, window-1 rate {rate.iloc[0] / len(hh):.3f}..."
        f"{rate.iloc[JMAX - 1] / len(hh):.3f}; category reach w1 {Y[1].any(1).mean():.3f} w2 {Y[2].any(1).mean():.3f}")


def load():
    z = np.load(OUT / "sets.npz")
    return z["Y1"], z["Y2"], z["items"]


def reach(Y, S):
    return float(Y[:, list(S)].any(1).mean())


def stepwise(Y, m):
    S, cov = [], np.zeros(len(Y), bool)
    for _ in range(m):
        gain = (Y & ~cov[:, None]).sum(0)
        gain[S] = -1
        j = int(np.argmax(gain))
        S.append(j)
        cov |= Y[:, j]
    return sorted(S)


def swap(Y, S):
    S = list(S)
    best = (Y[:, S].any(1)).sum()
    while True:
        cnt = Y[:, S].sum(1)
        move = None
        for i, a in enumerate(S):
            base = cnt - Y[:, a]
            covered = base > 0
            gain = (Y & ~covered[:, None]).sum(0) + covered.sum()
            gain[S] = -1
            j = int(np.argmax(gain))
            if gain[j] > best and (move is None or gain[j] > move[2]):
                move = (i, j, gain[j])
        if move is None:
            return sorted(S)
        S[move[0]] = move[1]
        best = move[2]


def ilp(Y, m, limit=ILP_LIMIT):
    # max sum_h w_h y_h  s.t.  y_h <= sum_{j in A_h} x_j,  sum_j x_j = m;  x binary, y in [0, 1]
    # households with identical acceptable sets are merged (weight w_h); empty sets drop out
    rows, w = np.unique(Y[Y.any(1)], axis=0, return_counts=True)
    H, J = rows.shape
    c = np.concatenate([np.zeros(J), -w.astype(float)])
    A = hstack([-csr_matrix(rows.astype(float)), identity(H, format="csr")])
    cons = [LinearConstraint(A, -np.inf, 0.0),
            LinearConstraint(np.concatenate([np.ones(J), np.zeros(H)])[None, :], m, m)]
    integ = np.concatenate([np.ones(J), np.zeros(H)])
    t0 = time.perf_counter()
    r = milp(c, constraints=cons, integrality=integ, bounds=Bounds(0, 1),
             options=dict(time_limit=limit, mip_rel_gap=0.0))
    dt = time.perf_counter() - t0
    S = sorted(np.flatnonzero(r.x[:J] > 0.5).tolist()) if r.x is not None else None
    bound = -r.mip_dual_bound / len(Y) if getattr(r, "mip_dual_bound", None) is not None else None
    return S, dt, r.status, bound, H


def cell(Y1, Y2, J, m):
    Y = Y1[:, :J]
    t0 = time.perf_counter()
    sw = stepwise(Y, m)
    t_sw = time.perf_counter() - t0
    t0 = time.perf_counter()
    sws = swap(Y, sw)
    t_sws = time.perf_counter() - t0
    opt, t_ilp, status, bound, H = ilp(Y, m)
    res = dict(J=J, m=m, H_unique=H, ilp_status=int(status), ilp_s=round(t_ilp, 3), ilp_bound=bound,
               stepwise_s=round(t_sw, 4), swap_s=round(t_sws, 4))
    for name, S in (("stepwise", sw), ("swap", sws), ("ilp", opt)):
        res[name] = dict(S=S, w1=reach(Y, S) if S else None, w2=reach(Y2[:, :J], S) if S else None)
    o = res["ilp"]["w1"]
    log(f"J={J:3d} m={m:2d}  ILP {o:.4f} ({t_ilp:6.2f}s, status {status}, {H} unique rows)  "
        f"gap pts: stepwise {100 * (o - res['stepwise']['w1']):.2f}  swap {100 * (o - res['swap']['w1']):.2f}  "
        f"| w2 opt-step {100 * (res['ilp']['w2'] - res['stepwise']['w2']):+.2f}")
    return res


def gate(full):
    Y1, Y2, items = load()
    log(f"{len(Y1)} households, J max {Y1.shape[1]}")
    cells = [(J, m) for J in JS for m in MS] if full else [(100, 15)]
    out = [cell(Y1, Y2, J, m) for J, m in cells]
    (OUT / ("grid_cpu.json" if full else "gate.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "gate"])
    ap.add_argument("--full", action="store_true", help="gate arms on every (J, m) cell")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    prep() if a.stage == "prep" else gate(a.full)
