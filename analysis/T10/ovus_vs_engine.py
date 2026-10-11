"""OVUS vs the orthant engine on CPU, on the orthants T9 / T10 actually evaluate (user, 2026-10-10).

Extends multivariate_probit docs/benchmarks.md (d = 3-16, synthetic Sigma) to choice-model orthants:
  A6   T9 MACML task-pair orthants (6-dim), T1 DGP truth G2 (pair corr .97), at the T5 GHK fit, rep 0
  A6g1 the same at truth G1 (pair corr .90)
  B20  T4 shelf, J = 20: one task's choice probability given the respondent's true tastes (20-dim),
       the per-task orthant of the conditioned GHK likelihood (MVP-G)
  C40  T4 shelf, J = 20: MACML task-pair orthants (40-dim) at the true theta
100 orthants per set. Reference: GHK with scrambled Sobol, M = 2^15 per scramble, 8 scrambles (mean
and SE). Methods, each scoring the same orthants: engine (paid local binary, resolution high and low,
default dup_corr = 0.03); OVUS by pybhatlib 0.4.0 (`_mvncd_ovus`, one row at a time, as published)
and by T9's batched torch port (CPU). Time per orthant: best of 3 after a warm-up.

    PYTHONPATH=<dir containing pybhatlib> python analysis/T10/ovus_vs_engine.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parent
for d in ("T1", "T4", "T9"):
    sys.path.insert(0, str(A / d))
sys.path.insert(0, str(ROOT / "research" / "rfc_vs_exact"))
import rfc  # noqa: E402
import t4  # noqa: E402
import t9  # noqa: E402
from ghk_gpu import panel_arrays  # noqa: E402
from macml import ovus_log, pair_index, pair_orthants  # noqa: E402
from pybhatlib.gradmvn._mvncd import _mvncd_ovus  # noqa: E402

torch.set_num_threads(2)
NPROB = 100


def set_a(truth):
    Xd, G = t9.data(truth, 0)
    th = np.array(json.loads((t9.T1 / "out" / f"T5_{truth}_N600" / "results.json").read_text())[0]["mvp"]["theta"])
    n = 2                                                       # 2 respondents x 66 pairs, first 100
    a, S = pair_orthants(torch.tensor(th).expand(n, -1), torch.tensor(Xd[:n]), torch.tensor(G[:, :n]),
                         float(t9.run.T_VAR), torch.as_tensor(pair_index(12, 3)))
    return a.numpy()[:NPROB], S.numpy()[:NPROB]


def t4_data():
    d = t4.make_data("S1", 20, 600, 12, 0)
    sh = t4.TRUTHS["S1"]
    th = np.r_[t4.b_true(20), np.log(t4.W_TRUE), np.log(sh[:3] / sh[3])]
    return d, th


def set_b(d):
    lv, sig = d["levels"], t4.sig_of("S1")
    Am = t4.member(lv)
    s = np.repeat(sig[:3], t4.LEVELS)
    E = Am @ np.diag(s ** 2) @ Am.T
    K = len(lv)
    E[np.arange(K), np.arange(K)] += sig[3] ** 2
    E[K, K] = t4.T_VAR                                          # none: own error of the total variance
    a_out, S_out = [], []
    for i in range(NPROB):
        n, t = divmod(i, 12)
        X = t4.task_X(lv, d["prices"][n, t])                       # (K+1, P)
        mu = X @ d["beta"][n]
        y = d["y"][n, t]
        oth = [k for k in range(K + 1) if k != y]
        M = np.zeros((K, K + 1))
        M[np.arange(K), oth] = 1
        M[:, y] = -1
        a_out.append(-(M @ mu))
        S_out.append(M @ E @ M.T)
    return np.array(a_out), np.array(S_out)


def set_c(d, th):
    lv = d["levels"]
    prods = np.concatenate([np.broadcast_to(lv, (2, 12, 20, 3)), d["prices"][:2, ..., None]], -1)
    Xd, G = panel_arrays(prods, d["y"][:2], lambda pr: t4.task_X(lv, pr[..., 3]), levels=t4.LEVELS)
    a, S = pair_orthants(torch.tensor(th).expand(2, -1), torch.tensor(Xd), torch.tensor(G), float(t4.T_VAR),
                         torch.as_tensor(pair_index(12, 20)))
    return a.numpy()[:NPROB], S.numpy()[:NPROB]


def ghk_ref(a, S, M=2 ** 15, reps=8, chunk=5):
    """log P by GHK, scrambled Sobol; mean over reps of P, and the SE of log P (delta method)."""
    from torch.special import log_ndtr, ndtri
    B, d = a.shape
    P = np.zeros((reps, B))
    L = np.linalg.cholesky(S)
    for r in range(reps):
        u = torch.quasirandom.SobolEngine(d - 1, scramble=True, seed=100 + r).draw(M, dtype=torch.float64)
        for lo in range(0, B, chunk):
            at, Lt = torch.tensor(a[lo:lo + chunk]), torch.tensor(L[lo:lo + chunk])
            n = len(at)
            logp = torch.zeros(n, M, dtype=torch.float64)
            zs = []
            for j in range(d):
                c = at[:, j, None] - sum(Lt[:, j, k, None] * zs[k] for k in range(j)) if j else at[:, j, None].expand(n, M)
                lp = log_ndtr(c / Lt[:, j, j, None])
                logp = logp + lp
                if j < d - 1:
                    zs.append(ndtri((u[None, :, j] * lp.exp()).clamp(1e-300, 1 - 1e-16)))
            P[r, lo:lo + n] = logp.exp().mean(1).numpy()
    m = P.mean(0)
    return np.log(m), P.std(0, ddof=1) / np.sqrt(reps) / m


def timed(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        v = fn()
        ts.append(time.perf_counter() - t0)
    return v, min(ts)


def main():
    d4, th4 = t4_data()
    sets = {"A6 (T9 pair, G2)": set_a("G2"), "A6g1 (T9 pair, G1)": set_a("G1"),
            "B20 (J=20 task | tastes)": set_b(d4), "C40 (J=20 MACML pair)": set_c(d4, th4)}
    out = {}
    for name, (a, S) in sets.items():
        t0 = time.time()
        ref, ref_se = ghk_ref(a, S)
        B = len(a)
        res = dict(n=B, dim=a.shape[1], median_P=float(np.exp(np.median(ref))), ref_seconds=time.time() - t0,
                   ref_se_logP_median=float(np.median(ref_se)), ref_se_logP_max=float(ref_se.max()))
        meth = {
            "engine high": lambda: np.log(np.maximum(rfc._LOCAL.cdf(a, S, resolution="high"), 1e-300)),
            "engine low": lambda: np.log(np.maximum(rfc._LOCAL.cdf(a, S, resolution="low"), 1e-300)),
            "OVUS pybhatlib (row loop)": lambda: np.log([_mvncd_ovus(a[i], S[i]) for i in range(B)]),
            "OVUS torch batch (T9 port)": lambda: ovus_log(torch.tensor(a), torch.tensor(S)).numpy(),
        }
        for m, fn in meth.items():
            v, sec = timed(fn)
            e = np.abs(v - ref)
            pe = np.abs(np.exp(v) - np.exp(ref))
            res[m] = dict(us_per_orthant=1e6 * sec / B, dlogP_median=float(np.median(e)), dlogP_max=float(e.max()),
                          dP_median=float(np.median(pe)), dP_max=float(pe.max()))
        out[name] = res
        print(f"\n## {name}: d = {res['dim']}, {B} orthants, median P {res['median_P']:.3g}; reference SE(log P) "
              f"median {res['ref_se_logP_median']:.1e}, max {res['ref_se_logP_max']:.1e} ({res['ref_seconds']:.0f} s)")
        print("| method | µs / orthant | |Δlog P| median (max) | |ΔP| median (max) |")
        print("|---|---|---|---|")
        for m in meth:
            r = res[m]
            print(f"| {m} | {r['us_per_orthant']:.1f} | {r['dlogP_median']:.1e} ({r['dlogP_max']:.1e}) | "
                  f"{r['dP_median']:.1e} ({r['dP_max']:.1e}) |")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "ovus_vs_engine.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
