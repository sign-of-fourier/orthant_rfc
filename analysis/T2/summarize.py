"""T2 summary: python analysis/T2/summarize.py [out/full/results.json]

Per truth x k x method: flips (of R) and mean regret for D-bundle (pair, triple), lift pick and
D-TURF (best 4 items by reach); a flip is "material" when its regret exceeds 0.5%. Joint MAE,
D-price errors, holdout log-likelihood, time to decision. Then the kill-rule pool (L1 + L2,
k >= 8) and the guards (L0: MVP |rho| median; LC: flips).
"""
import json
import sys
from pathlib import Path

import numpy as np

METHODS = ["cs_f", "cs_o", "il", "ifm", "ml", "gibbs"]
DECS = ["bundle2", "bundle3", "turf4"]
MATERIAL = 0.5


def load(path):
    return json.loads(Path(path).read_text())["results"]


def stats(recs, m):
    rows = [r["methods"][m] for r in recs if "error" not in r["methods"].get(m, {"error": 1})]
    if not rows:
        return None
    s = dict(n=len(rows))
    for d in DECS + ["lift2"]:
        s[f"{d}_flips"] = sum(r[f"{d}_flip"] for r in rows)
        s[f"{d}_material"] = sum(r[f"{d}_flip"] and r[f"{d}_regret_pct"] > MATERIAL for r in rows)
        s[f"{d}_regret"] = float(np.mean([r[f"{d}_regret_pct"] for r in rows]))
    for key in ("joint2_mae_pts", "joint3_mae_pts", "holdout_joint2_mae_pts", "dmarg_mae_pts", "holdout_ll"):
        s[key] = float(np.mean([r[key] for r in rows]))
    s["d_any_abs_err_pts"] = float(np.mean([abs(r["d_any_linked_err_pts"]) for r in rows]))
    s["t_median"] = float(np.median([r["seconds_total"] for r in rows]))
    if "corr_offdiag_abs_median" in rows[0]:
        s["corr_abs_median"] = float(np.median([r["corr_offdiag_abs_median"] for r in rows]))
    return s


def main(path):
    res = load(path)
    cells = sorted({(r["truth"], r["k"]) for r in res})
    print("truth k  method  n | flips b2/b3/turf4 (material) | regret% b2/b3/turf4 | j2 j3 hold MAE pts | "
          "dAny dMarg pts | holdLL | t_med s")
    for tr, k in cells:
        recs = [r for r in res if r["truth"] == tr and r["k"] == k]
        for m in METHODS:
            s = stats(recs, m)
            if s is None:
                print(f"{tr} {k:2d} {m:6s} all failed")
                continue
            fl = "/".join(f"{s[f'{d}_flips']}({s[f'{d}_material']})" for d in DECS)
            rg = "/".join(f"{s[f'{d}_regret']:.2f}" for d in DECS)
            extra = f" |rho| {s['corr_abs_median']:.3f}" if "corr_abs_median" in s else ""
            print(f"{tr} {k:2d} {m:6s} {s['n']:2d} | {fl:22s} | {rg:16s} | {s['joint2_mae_pts']:.2f} "
                  f"{s['joint3_mae_pts']:.2f} {s['holdout_joint2_mae_pts']:.2f} | {s['d_any_abs_err_pts']:.2f} "
                  f"{s['dmarg_mae_pts']:.2f} | {s['holdout_ll']:.3f} | {s['t_median']:.0f}{extra}")
        print()
    pool = [r for r in res if r["truth"] in ("L1", "L2") and r["k"] >= 8]
    print(f"Kill-rule pool (L1+L2, k>=8, {len(pool)} datasets x {len(DECS)} decisions):")
    for m in METHODS:
        s = stats(pool, m)
        if s:
            print(f"  {m:6s} flips {sum(s[f'{d}_flips'] for d in DECS):3d} (material "
                  f"{sum(s[f'{d}_material'] for d in DECS):3d}); mean regret turf4 {s['turf4_regret']:.2f}% "
                  f"bundle3 {s['bundle3_regret']:.2f}%; t_med {s['t_median']:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out" / "full" / "results.json")
