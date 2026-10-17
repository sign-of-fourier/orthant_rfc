"""X1 heterogeneity phase runner (design: DESIGN.md, "Heterogeneity phase").

    python lc_run.py wiring            # guard checks: K = 1 reproduces pooled fits; timing of K = 3 EM
    python lc_run.py run --only i,j    # level-pair (and reference) splits by index, as in x1.py

K is chosen per split by LC-L (latent-class logit) validation log-likelihood over K in {2, 3, 4} and used
for all three latent-class arms (same K). LC-P-LIN starts from LC-L's class posteriors; LC-P-XGB starts
from LC-P-LIN's. HB-RFC is read from the pooled split file (out/splits/<i>.npz).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (OUT, SEED, Data, Probit, fit_linear, fit_xgb, make_splits, pair_mask_any,  # noqa: E402
                  shares_from_logits, timer, xgb_V)
from lc import (em_mnl, em_probit_lin, em_probit_xgb, mnl_task_ll, predict_mnl_rfc,  # noqa: E402
                predict_probit_lin, predict_probit_xgb, tau_for, tune_lc_rfc, val_ll)

KS = (2, 3, 4)
STARTS = 3
LIN_SCALE = 1.45      # MNL / probit coefficient ratio (wiring check 2)
LOG = OUT / "lc.log"
ARMS = ["LC-L-RFC", "LC-P-LIN", "LC-P-XGB", "HB-RFC"]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def best_em_mnl(data, K, seed):
    fits = [em_mnl(data, K, np.random.default_rng(seed + 100 * s)) for s in range(STARTS)]
    return max(fits, key=lambda f: f["ll"])


def mixture_mnl_val(fit, val):
    tau = tau_for(fit, val)
    P = sum(tau[:, k] * np.exp(mnl_task_ll(val, b)) for k, b in enumerate(fit["B"]))
    return float(np.mean(np.log(P)))


def class_agreement(fit_l, fit_p):
    """Agreement of LC-L and LC-P-LIN classes: correlation of class sizes and of class part-worths after
    matching classes by posterior overlap."""
    ids = sorted(fit_l["tau"])
    Tl = np.stack([fit_l["tau"][i] for i in ids])
    Tp = np.stack([fit_p["tau"][i] for i in ids])
    match = (Tl.T @ Tp).argmax(1)
    bl = np.concatenate([fit_l["B"][k] / LIN_SCALE for k in range(len(match))])
    bp = np.concatenate([fit_p["B"][match[k]] for k in range(len(match))])
    same = float((match[Tl.argmax(1)] == Tp.argmax(1)).mean())
    return dict(match=match.tolist(), pi_l=fit_l["pi"].tolist(), pi_p=fit_p["pi"][match].tolist(),
                beta_corr=float(np.corrcoef(bl, bp)[0, 1]), same_assign=same)


def run_split(data, i, sp):
    path = OUT / "lc_splits" / f"{i:02d}.npz"
    if path.exists():
        log(f"lc split {i} exists, skipped")
        return
    pooled = OUT / "splits" / f"{i:02d}.npz"
    meta_p = json.loads(pooled.with_suffix(".json").read_text())
    t_all = timer()
    train, test = data.sub(sp["train"]), data.sub(sp["test"])
    vpos = np.flatnonzero(np.isin(sp["train"], sp["val"]))
    tune_tr = train.sub(np.setdiff1d(np.arange(train.T), vpos))
    tune_val = train.sub(vpos)
    seed = SEED + i
    tm = {}

    # K by LC-L on validation
    t = timer()
    vll = {K: mixture_mnl_val(best_em_mnl(tune_tr, K, seed), tune_val) for K in KS}
    K = max(vll, key=vll.get)
    fit_lt = best_em_mnl(tune_tr, K, seed)
    sl, g = tune_lc_rfc(fit_lt, tune_val, seed)
    fit_l = best_em_mnl(train, K, seed)
    P_l = predict_mnl_rfc(fit_l, test, sl, g, seed + 1)
    tm["LC-L-RFC"] = t()

    # LC-P-LIN from LC-L posteriors
    t = timer()
    lin0 = meta_p["info"]["P-LIN"]
    w0 = np.log(np.array(lin0["shares"][:-1]) / lin0["shares"][-1])
    b0 = np.array(lin0["beta"])
    fit_p = em_probit_lin_from(train, K, fit_l, w0, b0, seed)
    P_p = predict_probit_lin(fit_p, test)
    tm["LC-P-LIN"] = t()
    agree = class_agreement(fit_l, fit_p)

    # LC-P-XGB from LC-P-LIN posteriors, rounds from the pooled P-XGB fit on this split
    t = timer()
    rounds = meta_p["info"]["P-XGB"]["rounds"]
    fit_x = em_probit_xgb(train, K, None, fit_p["w"], fit_p["tau"], rounds)
    P_x = predict_probit_xgb(fit_x, test, rounds)
    tm["LC-P-XGB"] = t()

    z = np.load(pooled)
    np.savez(path, resp=test.resp, y=test.y, pair_alt=np.c_[pair_mask_any(test, sp["pair"]), np.zeros(test.T, bool)],
             **{"P_LC-L-RFC": P_l, "P_LC-P-LIN": P_p, "P_LC-P-XGB": P_x, "P_HB-RFC": z["P_HB-RFC"]})
    meta = dict(i=i, family=sp["family"], kind=sp["kind"], name=sp["name"], K=K, val_ll_by_K=vll,
                rfc=dict(sl=sl, g=g), pi=dict(l=fit_l["pi"].tolist(), p=fit_p["pi"].tolist(), x=fit_x["pi"].tolist()),
                em_iters=dict(l=fit_l["iters"], p=fit_p["iters"], x=fit_x["iters"]), agreement=agree,
                seconds=tm, total=t_all())
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    ll = {a: val_ll(P, test) for a, P in (("LC-L-RFC", P_l), ("LC-P-LIN", P_p), ("LC-P-XGB", P_x),
                                          ("HB-RFC", z["P_HB-RFC"]))}
    log(f"lc split {i} {sp['name']}: K={K}, {t_all():.0f} s ({', '.join(f'{k} {v:.0f}' for k, v in tm.items())}); "
        f"LL/task " + ", ".join(f"{a} {v:.4f}" for a, v in ll.items())
        + f"; LC-L vs LC-P-LIN beta corr {agree['beta_corr']:.2f}, same class {agree['same_assign']:.2f}")


def em_probit_lin_from(data, K, fit_l, w0, b0, seed):
    """LC-P-LIN started from LC-L's posteriors and class betas (rescaled to the probit scale)."""
    from lc import probit_lin_weighted, resp_index, shares_weighted, estep
    ids, inv = resp_index(data)
    pr = Probit(data, 128)
    X = torch.as_tensor(data.X)
    tau = np.stack([fit_l["tau"][r] for r in ids])
    B = [b / LIN_SCALE for b in fit_l["B"]]
    w = np.asarray(w0, float)
    prev = -np.inf
    for it in range(40):
        s = shares_from_logits(torch.as_tensor(w)).numpy()
        B = [probit_lin_weighted(pr, X, tau[inv, k], s, B[k]) for k in range(K)]
        Vs = [data.X @ b for b in B]
        w = shares_weighted(pr, Vs, tau[inv], w)
        s = shares_from_logits(torch.as_tensor(w)).numpy()
        with torch.no_grad():
            tll = np.stack([pr.loglik(torch.as_tensor(v), torch.as_tensor(s)).numpy() for v in Vs], 1)
        log_pi = np.log(tau.mean(0))
        tau, ll = estep(tll, inv, len(ids), log_pi)
        if ll - prev < 1e-5 * abs(ll):
            break
        prev = ll
    return dict(B=B, w=w, shares=s, pi=np.exp(log_pi), tau=dict(zip(ids, tau)), ll=ll, iters=it + 1)


def wiring(data):
    """K = 1 reproduces pooled fits; timing of one K = 3 EM per arm on full data."""
    rng = np.random.default_rng(SEED)
    lin = fit_linear(data, True, M=128)
    one = em_probit_lin(data, 1, rng, lin["w"], lin["beta"], M=128, iters=3)
    d1 = abs(one["ll"] - lin["ll"])
    log(f"guard 1a: LC-P-LIN K=1 ll {one['ll']:.2f} vs P-LIN {lin['ll']:.2f} (|diff| {d1:.2f}) -> "
        f"{'PASS' if d1 < 1.0 else 'FAIL'}")
    s = shares_from_logits(torch.as_tensor(lin["w"])).numpy()
    bst, _ = fit_xgb(data, None, s, 3, rounds=60)
    pr = Probit(data, 128)
    with torch.no_grad():
        ll_pooled = pr.loglik(torch.as_tensor(xgb_V(bst, data, 60)), torch.as_tensor(s)).sum().item()
    onex = em_probit_xgb(data, 1, None, lin["w"], {r: np.ones(1) for r in np.unique(data.resp)}, 60, iters=1)
    d2 = abs(onex["ll"] - ll_pooled)
    log(f"guard 1b: LC-P-XGB K=1 ll {onex['ll']:.2f} vs pooled P-XGB {ll_pooled:.2f} (|diff| {d2:.2f}; the K=1 "
        f"run also refits the shares) -> {'PASS' if d2 < 5.0 else 'FAIL'}")
    t = timer()
    fl = best_em_mnl(data, 3, SEED)
    log(f"timing: LC-L K=3 x {STARTS} starts {t():.0f} s, pi {np.round(fl['pi'], 3).tolist()}, ll {fl['ll']:.1f}")
    t = timer()
    fp = em_probit_lin_from(data, 3, fl, lin["w"], lin["beta"], SEED)
    ag = class_agreement(fl, fp)
    log(f"timing: LC-P-LIN K=3 {t():.0f} s ({fp['iters']} EM iters), pi {np.round(fp['pi'], 3).tolist()}, ll {fp['ll']:.1f}")
    log(f"guard 2: LC-L vs LC-P-LIN classes: beta corr {ag['beta_corr']:.3f}, same class {ag['same_assign']:.2f}, "
        f"pi {np.round(ag['pi_l'], 3).tolist()} vs {np.round(ag['pi_p'], 3).tolist()} -> "
        f"{'PASS' if ag['beta_corr'] > 0.9 and ag['same_assign'] > 0.8 else 'FAIL'}")
    t = timer()
    fx = em_probit_xgb(data, 3, None, fp["w"], fp["tau"], 60, iters=2)
    log(f"timing: LC-P-XGB K=3, 60 rounds, 2 EM iters {t():.0f} s, ll {fx['ll']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["wiring", "run"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    (OUT / "lc_splits").mkdir(parents=True, exist_ok=True)
    data = Data()
    if a.stage == "wiring":
        wiring(data)
        return
    splits = make_splits(data)
    for i in [int(x) for x in a.only.split(",")]:
        run_split(data, i, splits[i])


if __name__ == "__main__":
    main()
