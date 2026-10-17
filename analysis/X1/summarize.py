"""X1 pooled-phase summary: prediction metrics on test tasks, paired comparisons, novelty penalty.

    python summarize.py [--B 1000]

Share errors are absolute values, so their bootstrap distribution is skewed and a percentile CI need not
contain the point estimate; relative lift is reported only for log-likelihood and hit rate.
Respondent bootstrap: one resample of the 332 respondents is applied to every split and every arm
(paired). CIs are percentile 95%; p-values are two-sided bootstrap; Holm across the comparisons within
each (family, metric). Writes out/summary.json and out/summary.txt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import OUT, SEED  # noqa: E402

ARMS = ["P-XGB", "P-XGB-I", "P-XGB-1", "P-LIN", "P-LIN-I", "L-RFC", "HB-RFC"]
COMPARISONS = [  # (a, b): reported as metric(a) - metric(b), oriented so positive = a better
    ("P-LIN", "P-LIN-I", "Sigma, linear"),
    ("P-LIN-I", "L-RFC", "probit vs logit link (no correlation)"),
    ("P-XGB", "P-XGB-I", "Sigma, trees"),
    ("P-XGB-1", "P-LIN", "additive trees vs linear"),
    ("P-XGB", "P-XGB-1", "interactions vs none"),
    ("P-XGB", "P-LIN", "our method vs linear probit"),
    ("P-XGB", "L-RFC", "our method vs pooled logit + RFC"),
]
METRICS = {  # name: (higher is better, unit scale, label)
    "ll": (True, 1.0, "log-likelihood per task"),
    "hit": (True, 100.0, "hit rate, %"),
    "pair_err": (False, 100.0, "held-out-combination share error, pts"),
    "none_err": (False, 100.0, "none share error, pts"),
}
N_RESP = 332
LC_ARMS = ["LC-P-XGB", "LC-P-LIN", "LC-L-RFC", "HB-RFC"]
LC_COMPARISONS = [
    ("LC-P-XGB", "HB-RFC", "our method vs incumbent (headline)"),
    ("LC-L-RFC", "HB-RFC", "discrete vs continuous heterogeneity"),
    ("LC-P-LIN", "LC-L-RFC", "probit vs logit + RFC, same classes"),
    ("LC-P-XGB", "LC-P-LIN", "trees vs linear, same classes"),
]


def load(sub="splits"):
    S = []
    for p in sorted((OUT / sub).glob("*.npz")):
        meta = json.loads(p.with_suffix(".json").read_text())
        z = np.load(p)
        S.append(dict(meta=meta, **{k: z[k] for k in z.files}))
    return S


def split_stats(s, arm, w):
    """Metric values for one split with respondent weights w (N_RESP,)."""
    P = s[f"P_{arm}"]
    y = s["y"]
    wt = w[s["resp"]]
    n = wt.sum()
    py = np.maximum(P[np.arange(len(y)), y], 1e-300)
    pair_pred = (P * s["pair_alt"]).sum(1)
    pair_obs = s["pair_alt"][np.arange(len(y)), y]
    return dict(
        ll_sum=(wt * np.log(py)).sum(), hit_sum=(wt * (P.argmax(1) == y)).sum(), n=n,
        pair_err=abs((wt * pair_pred).sum() - (wt * pair_obs).sum()) / n,
        none_err=abs((wt * P[:, 4]).sum() - (wt * (y == 4)).sum()) / n,
    )


def pooled(splits, arm, w):
    st = [split_stats(s, arm, w) for s in splits]
    n = sum(x["n"] for x in st)
    return dict(ll=sum(x["ll_sum"] for x in st) / n, hit=sum(x["hit_sum"] for x in st) / n,
                pair_err=np.mean([x["pair_err"] for x in st]), none_err=np.mean([x["none_err"] for x in st]))


def holm(ps):
    ps = np.asarray(ps, float)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def share_section(groups, B, rng, out, res):
    """Held-out-combination share error, the primary measure. One value per held-out combination (level-pair
    split): |predicted - observed| share of test choices going to alternatives with that combination, in
    points. Uncertainty is across combinations (bootstrap over level-pair splits), since which combinations
    are held out dominates the noise; respondents are not resampled here."""
    out("=== PRIMARY: held-out-combination share error, points (|predicted - observed|), level-pair splits ===")
    out("    mean over combinations; 95% CI from a bootstrap over combinations; Holm within each family.\n")
    for fam in ("brand_price", "brand_feature"):
        ss = groups.get((fam, "pair"))
        if not ss:
            continue
        w = np.ones(N_RESP)
        E = {arm: np.array([100 * split_stats(s, arm, w)["pair_err"] for s in ss]) for arm in ARMS}
        S = {arm: np.array([100 * ((s[f"P_{arm}"] * s["pair_alt"]).sum(1).mean()
                                   - s["pair_alt"][np.arange(len(s["y"])), s["y"]].mean()) for s in ss])
             for arm in ARMS}
        idx = rng.integers(0, len(ss), (B, len(ss)))
        r = res.setdefault("share", {}).setdefault(fam, dict(n=len(ss), arms={}, comparisons={}))
        out(f"  {fam}: {len(ss)} held-out combinations")
        out(f"    {'arm':<8} {'mean |err|':>10} {'95% CI':>16} {'median':>7} {'mean signed':>12} {'best on':>8}")
        best = np.argmin(np.stack([E[a] for a in ARMS]), 0)
        for j, arm in enumerate(ARMS):
            m = E[arm][idx].mean(1)
            lo, hi = np.percentile(m, [2.5, 97.5])
            r["arms"][arm] = dict(est=E[arm].mean(), lo=lo, hi=hi, median=np.median(E[arm]),
                                  signed=S[arm].mean(), wins=int((best == j).sum()), per_split=E[arm].tolist())
            out(f"    {arm:<8} {E[arm].mean():>10.2f} {f'[{lo:.2f}, {hi:.2f}]':>16} {np.median(E[arm]):>7.2f} "
                f"{S[arm].mean():>+12.2f} {int((best == j).sum()):>5}/{len(ss)}")
        out("    paired differences, error(second) - error(first): positive = first arm more accurate")
        ds, ps = [], []
        for x, y, _ in COMPARISONS:
            d = E[y] - E[x]
            bm = d[idx].mean(1)
            ds.append((d, bm))
            ps.append(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean())))
        for (x, y, what), (d, bm), ph in zip(COMPARISONS, ds, holm(ps)):
            lo, hi = np.percentile(bm, [2.5, 97.5])
            r["comparisons"][f"{x} vs {y}"] = dict(est=d.mean(), lo=lo, hi=hi, p_holm=ph,
                                                   first_better=int((d > 0).sum()))
            out(f"     {x:>8} vs {y:<8} {d.mean():+6.2f} [{lo:+.2f}, {hi:+.2f}]  first better on "
                f"{int((d > 0).sum()):>2}/{len(d)}  Holm p {ph:.2g}   ({what})")
        out("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--phase", choices=["pooled", "lc"], default="pooled")
    a = ap.parse_args()
    global ARMS, COMPARISONS
    if a.phase == "lc":
        ARMS, COMPARISONS = LC_ARMS, LC_COMPARISONS
    S = load("splits" if a.phase == "pooled" else "lc_splits")
    rng = np.random.default_rng(SEED)
    W = [np.ones(N_RESP)] + [np.bincount(rng.integers(0, N_RESP, N_RESP), minlength=N_RESP).astype(float)
                             for _ in range(a.B)]
    groups = {}
    for s in S:
        groups.setdefault((s["meta"]["family"], s["meta"]["kind"]), []).append(s)
    res, lines = {}, []
    out = lines.append
    out(f"X1 {a.phase} phase: {len(S)} splits; respondent bootstrap B = {a.B}; 95% percentile CIs; Holm within "
        f"each (family, metric).\n")
    if a.phase == "pooled":
        share_section(groups, 10000, np.random.default_rng(SEED + 1), out, res)
        out("=== Secondary measures (respondent bootstrap, combinations fixed) ===\n")
    for (fam, kind), ss in sorted(groups.items()):
        vals = {arm: [pooled(ss, arm, w) for w in W] for arm in ARMS}
        key = f"{fam}/{kind}"
        res[key] = dict(n_splits=len(ss), n_test_tasks=int(sum(len(s["y"]) for s in ss)), arms={}, comparisons={})
        out(f"=== {fam}, {kind} splits ({len(ss)} splits, {res[key]['n_test_tasks']} test tasks) ===")
        out(f"  {'arm':<8} " + "  ".join(f"{METRICS[m][2]:>44}" for m in METRICS))
        for arm in ARMS:
            row = []
            res[key]["arms"][arm] = {}
            for m, (hb, sc, _) in METRICS.items():
                v = np.array([x[m] for x in vals[arm]]) * sc
                lo, hi = np.percentile(v[1:], [2.5, 97.5])
                res[key]["arms"][arm][m] = dict(est=v[0], lo=lo, hi=hi)
                row.append(f"{v[0]:>10.4f} [{lo:.4f}, {hi:.4f}]".rjust(44))
            out(f"  {arm:<8} " + "  ".join(row))
        for m, (hb, sc, lab) in METRICS.items():
            out(f"  -- {lab}: comparisons (positive = first arm better)")
            ds, ps, rels = [], [], []
            for x, y, what in COMPARISONS:
                vx = np.array([v[m] for v in vals[x]]) * sc
                vy = np.array([v[m] for v in vals[y]]) * sc
                d = (vx - vy) if hb else (vy - vx)
                p = min(1.0, 2 * min((d[1:] <= 0).mean(), (d[1:] >= 0).mean()))
                rel = 100 * d / np.abs(vy)
                ds.append(d), ps.append(p), rels.append(rel)
            for (x, y, what), d, p, rel, ph in zip(COMPARISONS, ds, ps, rels, holm(ps)):
                lo, hi = np.percentile(d[1:], [2.5, 97.5])
                rlo, rhi = np.percentile(rel[1:], [2.5, 97.5])
                res[key]["comparisons"].setdefault(m, {})[f"{x} vs {y}"] = dict(
                    est=d[0], lo=lo, hi=hi, rel=rel[0], rel_lo=rlo, rel_hi=rhi, p=p, p_holm=ph)
                relpart = (f"rel {rel[0]:+6.1f}% [{rlo:+.1f}, {rhi:+.1f}]" if m in ("ll", "hit")
                           else "rel n/a (baseline near 0)".ljust(28))
                out(f"     {x:>8} vs {y:<8} {d[0]:+9.4f} [{lo:+.4f}, {hi:+.4f}]  {relpart}  Holm p {ph:.3g}   ({what})")
        out("")
    # novelty penalty: pair metric minus matched random metric, per arm (positive = worse on new combinations)
    out("=== Novelty penalty: level-pair splits minus matched random splits (positive = worse on unseen pairs) ===")
    for fam in ("brand_price", "brand_feature"):
        if (fam, "pair") not in groups or (fam, "random") not in groups:
            continue
        out(f"  {fam}")
        for arm in ARMS:
            parts = []
            for m, (hb, sc, _) in METRICS.items():
                vp = np.array([pooled(groups[(fam, 'pair')], arm, w)[m] for w in W]) * sc
                vr = np.array([pooled(groups[(fam, 'random')], arm, w)[m] for w in W]) * sc
                d = (vr - vp) if hb else (vp - vr)
                lo, hi = np.percentile(d[1:], [2.5, 97.5])
                res.setdefault("novelty", {}).setdefault(fam, {}).setdefault(arm, {})[m] = dict(est=d[0], lo=lo, hi=hi)
                parts.append(f"{m} {d[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            out(f"    {arm:<8} " + "   ".join(parts))
    txt = "\n".join(lines)
    print(txt)
    tag = "" if a.phase == "pooled" else "_lc"
    (OUT / f"summary{tag}.txt").write_text(txt + "\n")
    (OUT / f"summary{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
