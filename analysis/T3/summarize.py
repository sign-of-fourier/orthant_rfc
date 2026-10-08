"""T3 summary: python analysis/T3/summarize.py [out/full_taco/results.json]

Gate and subsets; real-data picks per method against the holdout's own picks; adjudication of
every disagreement (household cluster bootstrap, 95%); engine column; controls (flips of the
truth's decisions, material = regret > 0.5%); holdout log-likelihood; times; the verdict rule.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

METHODS = ["cs_f", "cs_o", "il", "ifm", "ml"]
MATERIAL = 0.5


def label(key, idx, names):
    M = len(names)
    short = [n.split("/")[0].split(" ")[-1] if " " in n else n.split("/")[0] for n in names]
    if key.startswith("rec"):
        return f"{short[int(key[3:])]}->{short[idx]}"
    k = {"bundle2": 2, "bundle3": 3, "turf4": 4}[key]
    return "+".join(short[m] for m in list(itertools.combinations(range(M), k))[idx])


def main(path):
    r = json.loads(Path(path).read_text())
    names = r["names"]
    sn = lambda s: [names[m] for m in s]  # noqa: E731
    print(f"{r['cluster']}: {r['n_hh']} households, {r['n_trips']} trips ({r['n_train']} training)")
    print(f"gate: {r['gate']}")
    print(f"CS-F subsets {[sn(s) for s in r['subsets']['cs_f']]}; CS-O {[sn(s) for s in r['subsets']['cs_o']]}")
    iu = np.triu_indices(len(names), 1)
    for m in ("R_ifm", "R_ml"):
        R = np.array(r[m])
        print(f"{m}: offdiag min {R[iu].min():.2f} median {np.median(R[iu]):.2f} max {R[iu].max():.2f}")
    real = r["real"]
    keys = list(real["picks"]["observed"])
    print("\nreal-data picks (observed = holdout's own pick)")
    print(f"{'decision':10s} " + " ".join(f"{m:>22s}" for m in METHODS + ["observed"]))
    for k in keys:
        print(f"{k:10s} " + " ".join(f"{label(k, real['picks'][m][k], names) if 'error' not in real['picks'][m] else 'ERR':>22s}"
                                     for m in METHODS + ["observed"]))
    print("\nadjudication (verdict a / b / unresolved; holdout values)")
    for pair, adj in real["adjudication"].items():
        a, b = pair.split("_vs_")
        diff = {k: v for k, v in adj.items() if v["differ"]}
        tally = {v: sum(x["verdict"] == v for x in diff.values()) for v in ("a", "b", "unresolved")}
        print(f"  {pair}: {len(diff)} of {len(adj)} differ; {a} supported {tally['a']}, {b} supported {tally['b']}, "
              f"unresolved {tally['unresolved']}")
        for k, x in diff.items():
            print(f"     {k:8s} {label(k, real['picks'][a][k], names):>22s} {x['hold_a']:.4f} vs "
                  f"{label(k, real['picks'][b][k], names):>22s} {x['hold_b']:.4f}  CI [{x['diff_lo']:+.4f}, "
                  f"{x['diff_hi']:+.4f}] -> {x['verdict']}")
    print("\nholdout log-likelihood per trip / fit s / decide s")
    for m in METHODS:
        if m in real["holdout_ll"]:
            print(f"  {m:6s} {real['holdout_ll'][m]:.5f}  {real['seconds_fit'][m]:7.1f}  {real['seconds_decide'][m]:6.1f}")
    for m, e in real["engine"].items():
        print(f"  engine {m}: decide {e['seconds_decide']:.1f} s; differs from QMC on {e['differs_from_qmc']}")

    print("\ncontrols: flips (material) of the truth's decisions, summed over replicates")
    for a in ("S-MVP", "S-CS"):
        recs = [v for k, v in r["controls"].items() if k.startswith(a)]
        row = []
        for m in METHODS:
            sc = [x[m]["score"] for x in recs if "error" not in x[m]]
            fl = sum(d["flip"] for s in sc for d in s.values())
            mat = sum(d["flip"] and d["regret_pct"] > MATERIAL for s in sc for d in s.values())
            reg = np.mean([d["regret_pct"] for s in sc for d in s.values()]) if sc else np.nan
            row.append(f"{m} {fl}({mat}) {reg:.2f}%")
        print(f"  {a} (n={len(recs)}): " + "; ".join(row))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out" / "full_taco" / "results.json")
