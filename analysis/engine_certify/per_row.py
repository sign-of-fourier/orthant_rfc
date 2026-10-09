"""Per-row engine error vs the chosen alternative's probability, at the truth (J = 8, rep 0).
Reference: GHK with R = 8,192. Value error and the error in d log p / d u (share parameters),
binned by the reference probability. Signed gradient-error sums show where the bias comes from."""
import json
import numpy as np
import ec

BINS = [0, .01, .03, .1, .3, 1]
H = 1e-4
out = {}
for truth in ec.TRUTHS:
    sh = ec.Shelf(8)
    d = sh.simulate(truth, 0)
    th = d["th_true"]
    ghk = ec.GHK(sh, d, 8192, seed=7)
    lq, le = ghk.ll_rows_nograd(th), ec.ll_engine_rows(sh, d, th)
    gq, ge = [], []
    for i in range(sh.P, len(th)):
        e = np.zeros_like(th); e[i] = H
        gq.append((ghk.ll_rows_nograd(th + e) - ghk.ll_rows_nograd(th - e)) / (2 * H))
        ge.append((ec.ll_engine_rows(sh, d, th + e) - ec.ll_engine_rows(sh, d, th - e)) / (2 * H))
    gq, ge = np.array(gq).T, np.array(ge).T                      # (M, 3)
    pq, pe = np.exp(lq), np.exp(le)
    gerr = ge - gq
    print(f"\n{truth}: {len(pq)} choices; total grad error (brand, flavor, pack) {np.round(gerr.sum(0), 1)}"
          f"  vs total |grad| {np.round(np.abs(gq).sum(0), 0)}")
    print("  p bin        n   abs err p  rel err p  log err  | share of grad error (signed sum per share param)")
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (pq >= lo) & (pq < hi)
        if not m.any():
            continue
        r = dict(lo=lo, hi=hi, n=int(m.sum()), abs_p=float(np.abs(pe - pq)[m].mean()),
                 rel_p=float(np.abs(pe / pq - 1)[m].mean()), log_err=float((le - lq)[m].mean()),
                 grad_err=gerr[m].sum(0).tolist())
        rows.append(r)
        print(f"  [{lo:.2f},{hi:.2f})  {r['n']:5d}  {r['abs_p']:.5f}    {r['rel_p']:.4f}    {r['log_err']:+.4f}  | "
              f"{np.round(r['grad_err'], 1)}")
    out[truth] = rows
json.dump(out, open(ec.OUT / "per_row.json", "w"), indent=1)
