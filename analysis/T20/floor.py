"""T20 floor check (local CPU, no Modal): is HB-MNL + RFC's ~5% cost-regret floor a tuning artefact?

    python analysis/T20/floor.py --truth G1 --N 200 --reps 10

Per replicate: the T20 dataset (cell_data, same seeds), HB-MNL by bayesm with the same seed as the Modal arm, then
RFC on the HB point estimates, tuned five ways and scored against the truth with T1's score():
  T5      RFC-S, T5's box (sigma_l in [0.01, 5], g in [0.05, 5]), 6 observed holdout shares; T5/T20 exactly
  wide    RFC-S, wide box (both in [1e-3, 50]), observed holdout shares
  G       RFC-G (one sigma per attribute + g), wide box, observed holdout shares
  G-pop   RFC-G, wide box, tuned to the TRUE population shares of the 6 holdout tasks (no sampling noise)
  G-dec   RFC-G, wide box, tuned to the true shares of the decision scenarios themselves (oracle share fit; it
          minimises share MAE, not regret, so it is not the best possible decision setting)
Writes out/floor/<truth>_N<N>_r<rep>.json.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modal_t20 as m  # noqa: E402

t1 = m.t1_module()
OUT = HERE / "out" / "floor"
WIDE = ((1e-3, 1e-3), (50.0, 50.0))


def scenarios():
    """The decision scenarios of t1.decisions, in its order."""
    S = []
    for p in t1.PRICES:
        mk = t1.BASE.copy()
        mk[0, 3] = p
        S.append(mk)
    S.append(t1.BASE)
    S.append(np.vstack([t1.BASE, t1.CAND["A''"]]))
    for c in t1.CAND.values():
        S.append(np.vstack([t1.BASE, c]))
    return S


def tune(R, scen, target, k, lo, hi, grid_n):
    """Minimise mean share MAE over scenarios; x = log(sigma_1..k, g). Log grid, then Nelder-Mead."""
    llo = np.log(np.r_[[lo[0]] * k, lo[1]])
    lhi = np.log(np.r_[[hi[0]] * k, hi[1]])

    def mae(x):
        x = np.clip(x, llo, lhi)
        e = np.exp(x)
        sl, g = (e[0] if k == 1 else e[:k]), e[k]
        return np.mean([np.abs(R.shares(sc, sl, g) - tg).mean() for sc, tg in zip(scen, target)])

    axes = [np.linspace(llo[i], lhi[i], grid_n) for i in range(k + 1)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(k + 1, -1).T
    vals = [mae(x) for x in grid]
    res = minimize(mae, grid[int(np.argmin(vals))], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-6, maxiter=200 * (k + 1)))
    x = np.clip(res.x, llo, lhi)
    flags = [f"param {i} at bound" for i in range(k + 1) if x[i] - llo[i] < .05 or lhi[i] - x[i] < .05]
    return np.exp(x), float(res.fun), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="G1")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sig = np.sqrt(t1.T_VAR * t1.TRUTHS[a.truth])
    truth_fn = (t1.mixed_logit_shares if a.truth in t1.LOGIT_TRUTHS
                else (lambda pr: t1.pop_shares(pr, t1.B_TRUE, t1.W_TRUE, sig)))
    tdec = t1.decisions(truth_fn)
    scen = scenarios()
    scen_true = [truth_fn(s) for s in scen]
    for rep in range(a.reps):
        path = OUT / f"{a.truth}_N{a.N}_r{rep:02d}.json"
        if path.exists():
            continue
        c = m.cell_data(t1, a.truth, a.N, rep)
        t0 = time.time()
        B, _, t_hb = t1.run_hb(Path(tempfile.mkdtemp()), c["d"]["prods"], c["d"]["y"], m.HB_ITERS, c["hb_seed"])
        R = t1.RFC(B, m.R_L, c["r_rfc"])
        H, obs = c["H"], c["obs"]
        H_true = [truth_fn(h) for h in H]
        rec = dict(truth=a.truth, N=a.N, rep=rep, y_sha=c["y_sha"], seconds_hb=t_hb, arms={})
        arms = {
            "T5": lambda: (lambda r: (r[0], r[1], r[2]))(t1.tune_rfc(R, H, obs)),
            "wide": lambda: tune(R, H, obs, 1, *WIDE, 6),
            "G": lambda: tune(R, H, obs, 3, *WIDE, 4),
            "G-pop": lambda: tune(R, H, H_true, 3, *WIDE, 4),
            "G-dec": lambda: tune(R, scen, scen_true, 3, *WIDE, 4),
        }
        for name, fn in arms.items():
            t = time.time()
            x, mae, flags = fn()
            x = np.atleast_1d(np.asarray(x, float))
            sl, g = (x[0] if len(x) == 2 else x[:3]), x[-1]
            dec = t1.decisions(lambda pr: R.shares(pr, sl, g))
            rec["arms"][name] = dict(params=x.tolist(), mae=mae, flags=flags, seconds=time.time() - t,
                                     **t1.score(dec, tdec))
        path.write_text(json.dumps(m._jsonable(rec)))
        print(f"{a.truth} N={a.N} rep {rep}: {time.time() - t0:.0f} s; cost regret "
              + ", ".join(f"{k} {v['cost_regret_pct']:.2f}{'*' if v['flags'] else ''}" for k, v in rec["arms"].items()),
              flush=True)


if __name__ == "__main__":
    main()
