"""Split-half test: MVP reach vs counting on Conjointly experiment 7 (design: CONJOINTLY_EXPERIMENT_7_TURF.md).

Usage: python mvp_reach.py [path/to/conjointly_experiment_7.json] [--R 200]

Top-k membership (k = 2, 3, 4) as 14 binary indicators per respondent. R random 101/101 splits; on each
training half: Count reach (Conjointly's method), MVP reach (T6 IFM fit, 3-dim orthants by GHK with
scrambled Sobol, CPU), independence reach (reference). Scored against held-out observed reach.
CIs: corrected resampled t-test (Nadeau & Bengio 2003); Holm across k.
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as tdist
from torch.special import log_ndtr, ndtri

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "T6"))
from t6 import fit_ifm  # noqa: E402

SEED = 20261015
KS = (2, 3, 4)
M_GHK = 4096
MATERIAL = 1.0  # reach points (~1 respondent of 101)
ALPHA = 0.05

NAMES = ["Red", "Black", "Orange", "Yellow", "Green", "Blue", "Purple", "Violet", "Maroon", "Navy",
         "Turquoise", "Lime", "White", "Silver"]
IDS = [str(7955433 + j) for j in range(14)]


def ghk_orthant(eta, L, M=M_GHK, seed=SEED, chunk=256):
    """P(L z <= eta) per row by GHK with one fixed scrambled Sobol set (common random numbers)."""
    f64 = torch.float64
    tiny, one = torch.finfo(f64).tiny, 1.0 - torch.finfo(f64).eps
    n, k = eta.shape
    sob = torch.quasirandom.SobolEngine(k, scramble=True, seed=seed + k).draw(M, dtype=f64)
    out = np.empty(n)
    for lo in range(0, n, chunk):
        e = torch.as_tensor(eta[lo:lo + chunk], dtype=f64)
        Lc = torch.as_tensor(L[lo:lo + chunk], dtype=f64)
        logp = e.new_zeros(len(e), M)
        us = []
        for j in range(k):
            c = e[:, j, None]
            for i in range(j):
                c = c + Lc[:, j, i, None] * us[i]
            lp = log_ndtr(c / Lc[:, j, j, None])
            logp = logp + lp
            if j < k - 1:
                us.append(-ndtri((sob[None, :, j] * lp.exp()).clamp(tiny, one)))
        out[lo:lo + chunk] = logp.exp().mean(1).numpy()
    return out


def corrected_ci(d, n_train, n_test):
    """Mean, 95% CI and two-sided p for per-split differences d (Nadeau & Bengio correction)."""
    R = len(d)
    m, v = d.mean(), d.var(ddof=1)
    se = np.sqrt(v * (1 / R + n_test / n_train))
    if se == 0:
        return m, m, m, 1.0 if m == 0 else 0.0
    tq = tdist.ppf(0.975, R - 1)
    p = 2 * tdist.sf(abs(m / se), R - 1)
    return m, m - tq * se, m + tq * se, p


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(ROOT / "conjointly_experiment_7.json"))
    ap.add_argument("--R", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(2)

    raw = json.loads(Path(a.path).read_text())
    U = np.array([[r[i] for i in IDS] for r in raw], float)
    N, J = U.shape
    rk = np.argsort(np.argsort(-U, axis=1, kind="stable"), axis=1) + 1
    triples = np.array(list(combinations(range(J), 3)))
    T = len(triples)
    # item x triple membership, to count reach for all triples at once
    Mm = np.zeros((J, T))
    for t, tr in enumerate(triples):
        Mm[tr, t] = 1
    n_tr = N // 2
    n_te = N - n_tr
    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(N) for _ in range(a.R)]

    print(f"Split-half MVP reach vs counting. N = {N}, R = {a.R} splits of {n_tr}/{n_te}, "
          f"GHK M = {M_GHK}, seed {SEED}")
    print("CIs: corrected resampled t-test (Nadeau & Bengio 2003), Holm across k. "
          f"Materiality {MATERIAL:.1f} pt.\n")

    res = {}
    for k in KS:
        Y = (rk <= k).astype(np.int8)
        hitsY = (Y @ Mm) > 0                      # N x T: respondent reached by triple
        mae = {"count": [], "mvp": [], "ind": []}
        pick_reach = {"count": [], "mvp": []}
        differ = 0
        psd_moved, n_bound = [], 0
        full_fit = None
        for s, perm in enumerate(splits):
            tr, te = perm[:n_tr], perm[n_tr:]
            obs = hitsY[te].mean(0)
            est_c = hitsY[tr].mean(0)
            fit = fit_ifm(Y[tr])
            psd_moved.append(fit["psd_moved"])
            n_bound += int(np.sum(np.abs(fit["R_raw"][np.triu_indices(J, 1)]) >= 0.99 - 1e-9))
            mu, R = fit["mu"], fit["R"]
            eta = -mu[triples]
            L = np.stack([np.linalg.cholesky(R[np.ix_(t3, t3)]) for t3 in triples])
            est_m = 1 - ghk_orthant(eta, L)
            est_i = 1 - np.prod(1 - fit["p"][triples], axis=1)
            for nm, est in (("count", est_c), ("mvp", est_m), ("ind", est_i)):
                mae[nm].append(100 * np.abs(est - obs).mean())
            # decision: top triple on training; count ties broken by MVP is not allowed, use lowest index
            pc, pm = int(np.argmax(est_c)), int(np.argmax(est_m))
            pick_reach["count"].append(100 * obs[pc])
            pick_reach["mvp"].append(100 * obs[pm])
            differ += pc != pm
            if s == 0:
                full_fit = fit
        mae = {kk: np.array(v) for kk, v in mae.items()}
        pick_reach = {kk: np.array(v) for kk, v in pick_reach.items()}
        res[k] = dict(mae=mae, pick=pick_reach, differ=differ, psd=np.array(psd_moved), n_bound=n_bound,
                      fit0=full_fit)
        print(f"k = {k} done ({time.time() - t0:.0f} s)")

    # Holm across k, separately for the prediction and the decision families
    p_pred = [corrected_ci(res[k]["mae"]["count"] - res[k]["mae"]["mvp"], n_tr, n_te)[3] for k in KS]
    p_dec = [corrected_ci(res[k]["pick"]["mvp"] - res[k]["pick"]["count"], n_tr, n_te)[3] for k in KS]
    h_pred, h_dec = holm(p_pred), holm(p_dec)

    print("\nPrediction: mean absolute error vs held-out reach over all 364 triples, reach points")
    print(f"  {'k':>2} {'Count':>7} {'MVP':>7} {'Indep.':>7}   {'Count - MVP':>11} {'95% CI':>18} "
          f"{'rel. lift':>9} {'95% CI':>18} {'Holm p':>8}")
    verdict_pred = []
    for i, k in enumerate(KS):
        m = res[k]["mae"]
        d, lo, hi, _ = corrected_ci(m["count"] - m["mvp"], n_tr, n_te)
        rel = 100 * (m["count"] - m["mvp"]) / m["count"]
        r_, rlo, rhi, _ = corrected_ci(rel, n_tr, n_te)
        print(f"  {k:>2} {m['count'].mean():>7.2f} {m['mvp'].mean():>7.2f} {m['ind'].mean():>7.2f}   "
              f"{d:>+11.2f} [{lo:+6.2f}, {hi:+6.2f}] {r_:>+8.1f}% [{rlo:+6.1f}, {rhi:+6.1f}]% {h_pred[i]:>8.3g}")
        verdict_pred.append("mvp" if (h_pred[i] < ALPHA and d > 0) else "count" if (h_pred[i] < ALPHA and d < 0)
                            else "tie")

    print("\nDecision: each arm's top triple on the training half, scored by held-out reach (points)")
    print(f"  {'k':>2} {'picks differ':>12} {'Count pick':>10} {'MVP pick':>9}   {'MVP - Count':>11} "
          f"{'95% CI':>18} {'Holm p':>8}")
    verdict_dec = []
    for i, k in enumerate(KS):
        pr = res[k]["pick"]
        d, lo, hi, _ = corrected_ci(pr["mvp"] - pr["count"], n_tr, n_te)
        print(f"  {k:>2} {res[k]['differ']:>5}/{len(splits):<6} {pr['count'].mean():>10.2f} {pr['mvp'].mean():>9.2f}   "
              f"{d:>+11.2f} [{lo:+6.2f}, {hi:+6.2f}] {h_dec[i]:>8.3g}")
        worse = h_dec[i] < ALPHA and d < 0
        verdict_dec.append("mvp_worse_material" if worse and abs(d) >= MATERIAL else
                           "mvp_worse" if worse else "ok")

    print("\nFit diagnostics (IFM on training halves)")
    for k in KS:
        f0 = res[k]["fit0"]
        iu = np.triu_indices(J, 1)
        print(f"  k = {k}: PSD repair max move median {np.median(res[k]['psd']):.3f} "
              f"(max {res[k]['psd'].max():.3f}); raw pairwise rho at the +-0.99 bound: "
              f"{res[k]['n_bound']} pairs over {len(splits)} fits; split 0 rho median "
              f"{np.median(f0['R'][iu]):+.2f}, range {f0['R'][iu].min():+.2f}..{f0['R'][iu].max():+.2f}")

    n_mvp = sum(v == "mvp" for v in verdict_pred)
    any_count = any(v == "count" for v in verdict_pred)
    any_worse = any(v != "ok" for v in verdict_dec)
    any_worse_mat = any(v == "mvp_worse_material" for v in verdict_dec)
    if n_mvp >= 2 and not any_count and not any_worse:
        verdict = "PASS"
    elif any_count or any_worse_mat:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\nPrediction outcome per k: {dict(zip(KS, verdict_pred))}")
    print(f"Decision outcome per k:   {dict(zip(KS, verdict_dec))}")
    print(f"VERDICT (pass rule fixed in the design): {verdict}")
    print(f"\nwall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
