"""X1 pooled phase runner (design: analysis/X1/DESIGN.md).

    python x1.py wiring               # full-data fits of every arm + the three wiring checks
    python x1.py pilot                # 2 level-pair splits per family + their random references, timed
    python x1.py run [--only i,j,..]  # all 80 splits (skips splits already saved)

Per split and arm, saves per-test-task outputs to out/splits/<i>.npz (log-likelihood, hit, choice
probabilities, pair membership, respondent). Summaries are computed by summarize.py.
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from core import (COLS, OUT, SEED, Data, RFC, fit_linear, fit_mnl, fit_xgb, make_splits,  # noqa: E402
                  pair_mask_any, probit_choice_probs, shares_from_logits, timer, tune_rfc, xgb_V)

HB_R = 20000
HB_SCRIPT = OUT.parent.parent / "T1" / "hb.R"
XGB_DEPTH = 3
ARMS = ["P-XGB", "P-XGB-I", "P-XGB-1", "P-LIN", "P-LIN-I", "L-RFC", "HB-RFC"]
LOG = OUT / "x1.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_hb(data, tag, seed):
    path = OUT / "hb" / f"{tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.c_[np.repeat(data.resp, 5), np.repeat(np.arange(data.T), 5), np.tile(np.arange(5), data.T),
                 (np.arange(5)[None, :] == data.y[:, None]).reshape(-1), data.X.reshape(-1, len(COLS))]
    np.savetxt(path, rows, delimiter=",", comments="", fmt="%.6g",
               header="resp,task,alt,chosen," + ",".join(f"x{i}" for i in range(len(COLS))))
    r = subprocess.run(["Rscript", str(HB_SCRIPT), str(path), str(path.with_suffix("")), str(HB_R), "10",
                        str(seed)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    B = np.loadtxt(str(path.with_suffix("")) + "_beta.csv", delimiter=",", skiprows=1)
    resp_ids = np.unique(data.resp)       # hb.R writes one row per respondent, in sorted resp order
    return dict(zip(resp_ids, B))


def task_B(Bmap, data, default):
    return np.stack([Bmap.get(r, default) for r in data.resp])


def fit_all(train, val, test, tag, seed=SEED):
    """Fit every arm on `train` (tuning on `val` where needed, val is a subset of train's complement for
    tuning fits only) and return test choice probabilities (T_test, 5) per arm, plus fit info."""
    fit_tr = train                                    # final fits use all training tasks (val included)
    tune_tr = train.sub(np.flatnonzero(~np.isin(np.arange(train.T), val)))
    tune_val = train.sub(val)
    P, info, tm = {}, {}, {}

    t = timer()
    linI = fit_linear(fit_tr, sigma_free=False)
    P["P-LIN-I"] = probit_choice_probs(test, test.X @ linI["beta"], None)
    tm["P-LIN-I"] = t()
    t = timer()
    lin = fit_linear(fit_tr, sigma_free=True, beta0=linI["beta"])
    P["P-LIN"] = probit_choice_probs(test, test.X @ lin["beta"], lin["shares"])
    tm["P-LIN"] = t()
    info["P-LIN"] = dict(beta=lin["beta"].tolist(), shares=lin["shares"].tolist())
    info["P-LIN-I"] = dict(beta=linI["beta"].tolist())

    P, info, tm = fit_trees(fit_tr, tune_tr, tune_val, test, tag, lin["shares"], P, info, tm)

    # L-RFC: aggregate MNL; RFC tuned on val with the MNL fitted without val; applied to the full-train MNL
    t = timer()
    b_tune = fit_mnl(tune_tr)
    (sl, g), flags = tune_rfc(RFC(tune_val, seed=seed), np.tile(b_tune, (tune_val.T, 1)))
    b = fit_mnl(fit_tr)
    P["L-RFC"] = RFC(test, seed=seed + 1).probs(np.tile(b, (test.T, 1)), sl, g)
    tm["L-RFC"] = t()
    info["L-RFC"] = dict(beta=b.tolist(), sl=sl, g=g, at_bound=flags)

    # HB-RFC: same tuning scheme with HB point estimates
    t = timer()
    Bt = run_hb(tune_tr, f"{tag}_tune", seed)
    mean_t = np.mean(list(Bt.values()), 0)
    (sl, g), flags = tune_rfc(RFC(tune_val, seed=seed), task_B(Bt, tune_val, mean_t))
    Bf = run_hb(fit_tr, f"{tag}_fit", seed)
    mean_f = np.mean(list(Bf.values()), 0)
    P["HB-RFC"] = RFC(test, seed=seed + 1).probs(task_B(Bf, test, mean_f), sl, g)
    tm["HB-RFC"] = t()
    info["HB-RFC"] = dict(sl=sl, g=g, at_bound=flags, beta_mean=mean_f.tolist())
    return P, info, tm


TREE_ARMS = (("P-XGB", XGB_DEPTH, True), ("P-XGB-1", 1, True), ("P-XGB-I", XGB_DEPTH, False))


def fit_trees(fit_tr, tune_tr, tune_val, test, tag, lin_shares, P, info, tm, arms=None):
    """Tree arms: early stopping on the validation tasks, then refit on all training tasks with the chosen
    rounds. Sigma for the Sigma arms is the linear probit's (fitted on all training tasks) and held fixed:
    re-estimating it on the validation tasks was unstable (split 68: own share -> 0, near-singular)."""
    for arm, depth, sig in TREE_ARMS:
        if arms is not None and arm not in arms:
            continue
        t = timer()
        s = np.asarray(lin_shares) if sig else None
        bst, rounds = fit_xgb(tune_tr, tune_val, s, depth)
        final, _ = fit_xgb(fit_tr, None, s, depth, rounds=rounds)
        P[arm] = probit_choice_probs(test, xgb_V(final, test, rounds), s)
        tm[arm] = t()
        info[arm] = dict(rounds=rounds, shares=None if s is None else s.tolist(), sigma="from P-LIN")
        final.save_model(str(OUT / "models" / f"{tag}_{arm}.json"))
    return P, info, tm


def refit_trees(data, i, sp, arms):
    """Refit only the given tree arms on a saved split (uses the saved P-LIN shares)."""
    path = OUT / "splits" / f"{i:02d}.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    if all(meta["info"][a].get("sigma") == "from P-LIN" for a in arms):
        log(f"split {i} trees already refitted, skipped")
        return
    z = dict(np.load(path))
    train, test = data.sub(sp["train"]), data.sub(sp["test"])
    vpos = np.flatnonzero(np.isin(sp["train"], sp["val"]))
    tune_tr = train.sub(np.flatnonzero(~np.isin(np.arange(train.T), vpos)))
    P, info, tm = fit_trees(train, tune_tr, train.sub(vpos), test, f"s{i:02d}", meta["info"]["P-LIN"]["shares"],
                            {}, {}, {}, arms)
    for a in arms:
        z[f"P_{a}"] = P[a]
        meta["info"][a] = info[a]
        meta["seconds"][a] = tm[a]
    np.savez(path, **z)
    path.with_suffix(".json").write_text(json.dumps(meta, default=float, indent=1))
    ll = {a: float(np.mean(np.log(np.maximum(P[a][np.arange(test.T), test.y], 1e-300)))) for a in arms}
    log(f"split {i} {sp['name']}: refitted {', '.join(f'{a} {v:.4f}' for a, v in ll.items())}")


def run_split(data, i, sp):
    path = OUT / "splits" / f"{i:02d}.npz"
    if path.exists():
        log(f"split {i} exists, skipped")
        return
    t = timer()
    train, test = data.sub(sp["train"]), data.sub(sp["test"])
    val = np.flatnonzero(np.isin(sp["train"], sp["val"]))      # positions of val tasks inside train
    P, info, tm = fit_all(train, val, test, f"s{i:02d}", seed=SEED + i)
    pair_alt = np.c_[pair_mask_any(test, sp["pair"]), np.zeros(test.T, bool)]
    out = dict(resp=test.resp, y=test.y, pair_alt=pair_alt)
    for a in ARMS:
        out[f"P_{a}"] = P[a]
    np.savez(path, **out)
    meta = dict(i=i, family=sp["family"], kind=sp["kind"], name=sp["name"], n_train=len(sp["train"]),
                n_test=len(sp["test"]), n_val=len(sp["val"]), info=info, seconds=tm, total=t())
    (OUT / "splits" / f"{i:02d}.json").write_text(json.dumps(meta, default=float, indent=1))
    ll = {a: float(np.mean(np.log(np.maximum(P[a][np.arange(test.T), test.y], 1e-300)))) for a in ARMS}
    log(f"split {i} {sp['name']}: test {test.T} tasks, {t():.0f} s; LL/task "
        + ", ".join(f"{a} {v:.4f}" for a, v in ll.items()))


def wiring(data):
    """Full-data fits + wiring checks (no holdout)."""
    log("WIRING: full-data fits")
    rng = np.random.default_rng(SEED)
    val = np.sort(rng.choice(data.T, size=data.T // 5, replace=False))     # tuning only
    res = {}
    t = timer()
    linI = fit_linear(data, sigma_free=False)
    lin = fit_linear(data, sigma_free=True, beta0=linI["beta"])
    b = fit_mnl(data)
    log(f"linear fits {t():.0f} s")
    # check 1: MNL matches bayesm-based R optimum of the same likelihood
    path = OUT / "data" / "hb_full.csv"
    r = subprocess.run(["Rscript", "-e", f"""
suppressMessages(library(bayesm)); d <- read.csv('{path}'); X <- as.matrix(d[, grep('^x', names(d))])
y <- d$alt[d$chosen == 1] + 1
f <- function(b) -llmnl(b, y, X)
o <- optim(rep(0, ncol(X)), f, method = 'BFGS', control = list(reltol = 1e-12, maxit = 1000))
cat(o$par, sep = ','); cat('\\n'); cat(-o$value)"""], capture_output=True, text=True)
    rb = np.array(r.stdout.splitlines()[0].split(","), float)
    c1 = float(np.abs(rb - b).max())
    log(f"check 1 MNL vs bayesm llmnl optimum: max |beta diff| {c1:.2e} -> {'PASS' if c1 < 1e-3 else 'FAIL'}")
    # check 2: P-LIN-I ~ MNL x scale
    ratio = b / linI["beta"]
    big = np.abs(linI["beta"]) > 0.1
    c2 = float(np.std(ratio[big]) / np.mean(ratio[big]))
    log(f"check 2 MNL / P-LIN-I ratios {np.round(ratio, 2).tolist()} (|beta| > 0.1: CV {c2:.3f}) "
        f"-> {'PASS' if c2 < 0.1 else 'FAIL'}")
    # check 3: P-XGB-1 SHAP part-worths vs P-LIN betas
    import torch
    import xgboost as xgb
    s0 = shares_from_logits(torch.as_tensor(lin["w"])).numpy()
    tr = data.sub(np.setdiff1d(np.arange(data.T), val))
    bst1, r1 = fit_xgb(tr, data.sub(val), s0, 1)
    final1, _ = fit_xgb(data, None, s0, 1, rounds=r1)
    from core import cam_matrix
    Xc = cam_matrix(data)
    C = final1.predict(xgb.DMatrix(Xc), pred_contribs=True, iteration_range=(0, r1))   # (n, 11) incl. bias
    # brand part-worth = mean total contribution of the 4 brand dummies for cameras of that brand
    brand = Xc[:, :4].argmax(1)
    pw = [C[brand == j, :4].sum(1).mean() for j in range(4)]
    pw += [C[Xc[:, j] == 1, j].mean() - C[Xc[:, j] == 0, j].mean() for j in range(4, 9)]
    pw = np.array(pw)
    slope = np.polyfit(Xc[:, 9], C[:, 9], 1)[0]
    shap_feat = np.r_[pw[4:9], slope]
    lin_feat = lin["beta"][4:10]
    brand_shap = pw[:4] - pw[0]
    brand_lin = lin["beta"][:4] - lin["beta"][0]
    corr = float(np.corrcoef(np.r_[shap_feat, brand_shap[1:]], np.r_[lin_feat, brand_lin[1:]])[0, 1])
    log(f"check 3 P-XGB-1 ({r1} rounds) SHAP part-worths vs P-LIN betas (features + price slope + brand diffs): "
        f"corr {corr:.3f}; SHAP {np.round(np.r_[shap_feat, brand_shap[1:]], 3).tolist()} vs "
        f"P-LIN {np.round(np.r_[lin_feat, brand_lin[1:]], 3).tolist()} -> {'PASS' if corr > 0.9 else 'FAIL'}")
    res = dict(check1=c1, check2_cv=c2, ratio=ratio.tolist(), check3_corr=corr, rounds_xgb1=r1,
               lin=dict(beta=lin["beta"].tolist(), shares=lin["shares"].tolist(), ll=lin["ll"]),
               linI=dict(beta=linI["beta"].tolist(), ll=linI["ll"]), mnl=b.tolist(), seconds=t())
    (OUT / "wiring.json").write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["wiring", "pilot", "run"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--refit-trees", action="store_true", help="refit P-XGB and P-XGB-1 on saved splits")
    a = ap.parse_args()
    for d in ("splits", "models", "hb"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    data = Data()
    if a.stage == "wiring":
        wiring(data)
        return
    splits = make_splits(data)
    if a.stage == "pilot":
        idx = [i for i, s in enumerate(splits) if s["family"] == "brand_price"][:4] + \
              [i for i, s in enumerate(splits) if s["family"] == "brand_feature"][:4]
    else:
        idx = range(len(splits)) if a.only is None else [int(x) for x in a.only.split(",")]
    for i in idx:
        if a.refit_trees:
            if (OUT / "splits" / f"{i:02d}.npz").exists():
                refit_trees(data, i, splits[i], ("P-XGB", "P-XGB-1"))
        else:
            run_split(data, i, splits[i])


if __name__ == "__main__":
    main()
