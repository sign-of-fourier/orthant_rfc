"""X1 explanation: part-worths, importance and WTP from full-data fits; SHAP for the tree arms.

    python explain.py [--B 30]

Arms: P-LIN, P-LIN-I, MNL (L-RFC's utility), HB-MNL population mean, P-XGB-1 and P-XGB (TreeSHAP via
xgboost pred_contribs / pred_interactions). Scale-free quantities only are compared across arms:
importance shares, level order, WTP = part-worth / |price slope|. CIs: respondent bootstrap with B refits
(rounds and Sigma for the tree arms fixed at the full-data values inside the bootstrap).
Writes out/explain.json and out/explain.txt.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (BRANDS, COLS, FEATS, OUT, PRICES, SEED, Data, cam_matrix, fit_linear, fit_mnl,  # noqa: E402
                  fit_shares, fit_xgb, shares_from_logits, xgb_V)

ATTRS = ["brand"] + FEATS + ["price"]


def linear_pw(beta):
    """Part-worths from a linear utility: brand levels (centered), feature on-off, price per level."""
    b = np.asarray(beta)
    brand = b[:4] - b[:4].mean()
    price = b[9] * PRICES
    return dict(brand=brand, feats=b[4:9], price_levels=price - price.mean(), slope=b[9])


def shap_pw(bst, rounds, data):
    """Part-worths from TreeSHAP contributions of camera rows."""
    import xgboost as xgb
    Xc = cam_matrix(data)
    C = bst.predict(xgb.DMatrix(Xc), pred_contribs=True, iteration_range=(0, rounds))
    brand_id = Xc[:, :4].argmax(1)
    bc = C[:, :4].sum(1)
    brand = np.array([bc[brand_id == j].mean() for j in range(4)])
    feats = np.array([C[Xc[:, j] == 1, j].mean() - C[Xc[:, j] == 0, j].mean() for j in range(4, 9)])
    plev = np.abs(Xc[:, 9:10] - PRICES).argmin(1)
    price = np.array([C[plev == k, 9].mean() for k in range(5)])
    slope = np.polyfit(PRICES, price, 1)[0]
    return dict(brand=brand - brand.mean(), feats=feats, price_levels=price - price.mean(), slope=slope)


def shap_interactions(bst, rounds, data):
    """Mean SHAP interaction (utility units) for each brand x feature and brand x price level: for cameras
    of brand b, mean interaction(brand dummies, attribute) with the level present minus absent."""
    import xgboost as xgb
    Xc = cam_matrix(data)
    I = bst.predict(xgb.DMatrix(Xc), pred_interactions=True, iteration_range=(0, rounds))   # (n, 11, 11)
    brand_id = Xc[:, :4].argmax(1)
    plev = np.abs(Xc[:, 9:10] - PRICES).argmin(1)
    out = {}
    for j, b in enumerate(BRANDS):
        m = brand_id == j
        for k, f in enumerate(FEATS):
            col = 4 + k
            v = 2 * I[m][:, :4, col].sum(1)          # both off-diagonal halves
            on = Xc[m, col] == 1
            out[f"{b}+{f}"] = float(v[on].mean() - v[~on].mean())
        v = 2 * I[m][:, :4, 9].sum(1)
        pv = np.array([v[plev[m] == k].mean() for k in range(5)])
        for k in range(5):
            out[f"{b}@{PRICES[k]:.2f}"] = float(pv[k] - pv.mean())
    return out


def derived(pw):
    """Importance shares (range share) and WTP (per $ of price) from part-worths."""
    rng = [np.ptp(pw["brand"])] + list(np.abs(pw["feats"])) + [np.ptp(pw["price_levels"])]
    imp = np.array(rng) / np.sum(rng)
    wtp = np.r_[pw["brand"], pw["feats"]] / abs(pw["slope"])
    return dict(importance=imp, wtp=wtp)


def fit_tree(data, val_idx, s0, depth):
    tr = data.sub(np.setdiff1d(np.arange(data.T), val_idx))
    va = data.sub(val_idx)
    bst, r = fit_xgb(tr, va, s0, depth)
    s1 = shares_from_logits(__import__("torch").as_tensor(fit_shares(va, xgb_V(bst, va, r)))).numpy()
    final, _ = fit_xgb(data, None, s1, depth, rounds=r)
    return final, r, s1


def all_arms(data, hb_mean, tree=None):
    """Part-worths per arm. tree: dict arm -> (depth, rounds, shares) to refit with fixed settings."""
    lin = fit_linear(data, True)
    linI = fit_linear(data, False)
    pw = dict(**{"P-LIN": linear_pw(lin["beta"]), "P-LIN-I": linear_pw(linI["beta"]),
                 "MNL": linear_pw(fit_mnl(data))})
    if hb_mean is not None:
        pw["HB-mean"] = linear_pw(hb_mean)
    models = {}
    for arm, (depth, r, s) in (tree or {}).items():
        bst, _ = fit_xgb(data, None, s, depth, rounds=r)
        pw[arm] = shap_pw(bst, r, data)
        models[arm] = bst
    return pw, models, lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=30)
    a = ap.parse_args()
    data = Data()
    rng = np.random.default_rng(SEED)
    val = np.sort(rng.choice(data.T, size=data.T // 5, replace=False))
    import torch
    lin0 = fit_linear(data, True)
    s0 = shares_from_logits(torch.as_tensor(lin0["w"])).numpy()
    tree = {}
    for arm, depth in (("P-XGB-1", 1), ("P-XGB", 3)):
        _, r, s1 = fit_tree(data, val, s0, depth)
        tree[arm] = (depth, r, s1)
    hb = np.loadtxt(OUT / "hb" / "explain_fit_beta.csv", delimiter=",", skiprows=1).mean(0) \
        if (OUT / "hb" / "explain_fit_beta.csv").exists() else None
    if hb is None:
        from x1 import run_hb
        hb = np.mean(list(run_hb(data, "explain_fit", SEED).values()), 0)
    pw, models, _ = all_arms(data, hb, tree)
    inter = shap_interactions(models["P-XGB"], tree["P-XGB"][1], data)
    # bootstrap: resample respondents, refit everything (HB mean excluded: one MCMC run per replicate is
    # too slow; its CI is not reported)
    boot = []
    resp = np.unique(data.resp)
    for b in range(a.B):
        pick = rng.choice(resp, size=len(resp), replace=True)
        idx = np.concatenate([np.flatnonzero(data.resp == r) for r in pick])
        db = data.sub(idx)
        pwb, mb, _ = all_arms(db, None, tree)
        boot.append(dict(pw=pwb, inter=shap_interactions(mb["P-XGB"], tree["P-XGB"][1], db)))
        print(f"bootstrap {b + 1}/{a.B}", flush=True)

    lines = []
    out = lines.append
    out(f"X1 explanation (full data; bootstrap B = {a.B} respondent resamples, tree rounds and Sigma fixed)")
    out(f"tree arms: " + ", ".join(f"{k}: depth {v[0]}, {v[1]} rounds" for k, v in tree.items()))
    labels = BRANDS + FEATS
    res = dict(tree={k: dict(depth=v[0], rounds=v[1], shares=v[2].tolist()) for k, v in tree.items()}, arms={})
    for arm, p in pw.items():
        d = derived(p)
        bs = [derived(x["pw"][arm]) for x in boot if arm in x["pw"]]
        res["arms"][arm] = dict(brand=p["brand"].tolist(), feats=p["feats"].tolist(),
                                price_levels=p["price_levels"].tolist(), slope=float(p["slope"]),
                                importance=d["importance"].tolist(), wtp=d["wtp"].tolist())
        out(f"\n{arm}: price slope {p['slope']:+.3f} per $")
        out("  importance share (range share), 95% CI:")
        for k, at in enumerate(ATTRS):
            ci = np.percentile([x["importance"][k] for x in bs], [2.5, 97.5]) if bs else (np.nan, np.nan)
            out(f"    {at:<8} {100 * d['importance'][k]:5.1f}%  [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
        out("  WTP ($ per unit of utility / |slope|; brands centered), 95% CI:")
        for k, lab in enumerate(labels):
            ci = np.percentile([x["wtp"][k] for x in bs], [2.5, 97.5]) if bs else (np.nan, np.nan)
            out(f"    {lab:<10} {d['wtp'][k]:+6.2f}  [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        if bs:
            res["arms"][arm]["importance_ci"] = np.percentile([x["importance"] for x in bs], [2.5, 97.5], 0).tolist()
            res["arms"][arm]["wtp_ci"] = np.percentile([x["wtp"] for x in bs], [2.5, 97.5], 0).tolist()
    out("\nP-XGB SHAP interactions (utility units; brand x feature: on minus off; brand x price: level minus "
        "mean), 95% CI, sorted by |estimate|:")
    keys = sorted(inter, key=lambda k: -abs(inter[k]))
    res["interactions"] = {}
    for k in keys:
        ci = np.percentile([x["inter"][k] for x in boot], [2.5, 97.5]) if boot else (np.nan, np.nan)
        res["interactions"][k] = dict(est=inter[k], lo=float(ci[0]), hi=float(ci[1]))
        flag = "  *" if ci[0] > 0 or ci[1] < 0 else ""
        out(f"    {k:<16} {inter[k]:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]{flag}")
    out("  * = CI excludes 0 (no multiplicity adjustment; 40 interactions)")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "explain.txt").write_text(txt + "\n")
    (OUT / "explain.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
