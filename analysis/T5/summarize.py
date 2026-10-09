"""T5 N-sweep summary: python analysis/T5/summarize.py

Reads analysis/T1/out/T5_<truth>_N<N>/results.json (run.py --no-se). Per truth x N x method:
flips (price / ext at the placeholder hurdle / cost, of R) and mean regret with replicate SE.
Kill rules (PLAN.md T5, user 2026-10-09):
  N_match: smallest N at which MVP's mean (flips per replicate, or cost regret) is within one
           replicate-SE of RFC-S's at N = 1,200 (SE of RFC-S's mean at 1,200).
  G0 guard: at every N, MVP's cost regret no worse than RFC-S's by more than one SE (SE of the
           paired replicate difference).
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "T1" / "out"
NS = (150, 300, 600, 1200)
FLIPS = ("price_flip", "ext_flip_at_placeholder_h", "cost_flip")


def load(truth, n):
    p = OUT / f"T5_{truth}_N{n}" / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def arr(r, m, k):
    return np.array([float(bool(x[m][k])) if k in FLIPS else float(x[m][k]) for x in r])


def mse(v):
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def main():
    for truth in ("G0", "G1", "G2"):
        rs = {n: load(truth, n) for n in NS}
        if not any(rs.values()):
            continue
        print(f"\n## {truth}\n")
        print("| N | method | R | flips price / ext / cost | flips per rep (SE) | cost regret % (SE) | price regret % (SE) |")
        print("|---|---|---|---|---|---|---|")
        stats = {}
        for n in NS:
            r = rs[n]
            if not r:
                continue
            for m, lab in (("mvp", "MVP"), ("rfc_s", "RFC-S")):
                fl = [int(arr(r, m, k).sum()) for k in FLIPS]
                tot = sum(arr(r, m, k) for k in FLIPS)
                f, fse = mse(tot)
                c, cse = mse(arr(r, m, "cost_regret_pct"))
                p, pse = mse(arr(r, m, "price_regret_pct"))
                stats[(n, m)] = dict(f=f, fse=fse, c=c, cse=cse)
                print(f"| {n} | {lab} | {len(r)} | {fl[0]} / {fl[1]} / {fl[2]} | {f:.2f} ({fse:.2f}) | {c:.2f} ({cse:.2f}) | {p:.2f} ({pse:.2f}) |")
        if (1200, "rfc_s") in stats:
            ref = stats[(1200, "rfc_s")]
            for key, se, lab in (("f", "fse", "flips per rep"), ("c", "cse", "cost regret")):
                bar = ref[key] + ref[se]
                hit = [n for n in NS if (n, "mvp") in stats and stats[(n, "mvp")][key] <= bar]
                print(f"\nN_match ({lab}): {min(hit) if hit else 'none'}  (RFC-S at 1,200: {ref[key]:.2f} + 1 SE {ref[se]:.2f} = {bar:.2f})")
        if truth == "G0":
            print("\nG0 guard (MVP cost regret - RFC-S, paired; fails if > 1 SE):")
            for n in NS:
                r = rs[n]
                if not r:
                    continue
                d = arr(r, "mvp", "cost_regret_pct") - arr(r, "rfc_s", "cost_regret_pct")
                m, se = mse(d)
                print(f"  N {n}: diff {m:+.2f} (SE {se:.2f}) -> {'FAIL' if m > se else 'ok'}")


if __name__ == "__main__":
    main()
