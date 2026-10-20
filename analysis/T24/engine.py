"""T24 B · population click probabilities on held-out Trivago shelves from the fitted MVP: engine vs OVUS.

    PYTHONPATH=<multivariate_probit src whose binary has resolution "highest">:<dir with pybhatlib> python analysis/T24/engine.py

A new user's probability of clicking hotel y on a shelf, averaged over tastes, is one (J - 1)-dimensional orthant:
upper D V, covariance D (X W X' + Sigma_err) D', with D the differences y - other. Shelves with J = 25 only, so
every orthant is 24-dimensional. Parameters: out/mvp_full.json (else mvp_smoke.json).
  clicked  200 random holdout shelves, the clicked hotel
  shares   8 random holdout shelves, every hotel (200 orthants); a shelf's shares should sum to 1
Reference: GHK with scrambled Sobol, 2^15 points x 8 scrambles (T10's ghk_ref). Methods: engine (paid local
binary) at resolution highest, high and low; OVUS by T9's batched torch port and by pybhatlib's row loop. Time per
orthant: best of 3 after a warm-up (T10's timed). Writes out/engine.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "T10"))
sys.path.insert(0, str(HERE))
import ovus_vs_engine as T10  # noqa: E402  (puts T1, T4, T9 and rfc on the path)
import t24  # noqa: E402

rfc, ovus_log, _mvncd_ovus = T10.rfc, T10.ovus_log, T10._mvncd_ovus
T_VAR = math.pi ** 2 / 6
NSHELF, NSHARE = 200, 8


def params():
    f = t24.OUT / "mvp_full.json"
    f = f if f.exists() else t24.OUT / "mvp_smoke.json"
    th = np.array(json.loads(f.read_text())["theta"])
    return th, f.name


def orthant(X, band, y, th, rnd):
    """(upper, cov) of P(click y) on one full shelf, tastes integrated."""
    P, nr, A = X.shape[1], len(rnd), band.shape[1]
    b, sd = th[:P], np.exp(th[P:P + nr])
    u = np.r_[th[P + nr:], 0.0]
    s2 = np.exp(u - u.max())
    s2 = T_VAR * (s2 / s2.sum() * (1 - 1e-5) + 1e-5 / len(s2))
    Xs = X[:, rnd] * sd
    Sig = Xs @ Xs.T + sum(s2[a] * (band[:, a, None] == band[None, :, a]) for a in range(A)) + s2[A] * np.eye(len(X))
    D = -np.eye(len(X))[np.arange(len(X)) != y]
    D[:, y] = 1.0
    return D @ X @ b, D @ Sig @ D.T


def main():
    A = t24.load()
    th, src = params()
    feat = A["feat"].tolist()
    rnd = np.array([k for k, f in enumerate(feat) if f not in t24.FIXED])
    full = np.flatnonzero((A["split"] == 1) & (A["mask"].sum(1) == t24.JMAX))
    rng = np.random.default_rng(24)
    X, band = A["X"], A["band"]
    sets = {"clicked": [(c, int(A["y"][c])) for c in rng.choice(full, NSHELF, replace=False)],
            "shares": [(c, j) for c in rng.choice(full, NSHARE, replace=False) for j in range(t24.JMAX)]}
    out = {"params": src}
    for name, rows in sets.items():
        a, S = map(np.array, zip(*[orthant(X[c].astype(float), band[c], j, th, rnd) for c, j in rows]))
        ref, ref_se = T10.ghk_ref(a, S)
        n = len(rows)
        res = dict(dim=a.shape[1], median_P=float(np.exp(np.median(ref))), ref_se_logP_median=float(np.median(ref_se)))
        meth = {
            "engine highest": lambda: np.log(np.maximum(rfc._LOCAL.cdf(a, S, resolution="highest"), 1e-300)),
            "engine high": lambda: np.log(np.maximum(rfc._LOCAL.cdf(a, S, resolution="high"), 1e-300)),
            "engine low": lambda: np.log(np.maximum(rfc._LOCAL.cdf(a, S, resolution="low"), 1e-300)),
            "OVUS torch batch": lambda: ovus_log(torch.tensor(a), torch.tensor(S)).numpy(),
            "OVUS pybhatlib (row loop)": lambda: np.log([_mvncd_ovus(a[i], S[i]) for i in range(n)]),
        }
        if name == "shares":
            res["ref_sum_dev_max"] = float(np.abs(np.exp(ref).reshape(NSHARE, -1).sum(1) - 1).max())
        for m, fn in meth.items():
            try:
                v, sec = T10.timed(fn)
            except Exception as e:                       # report, don't hide
                res[m] = dict(error=repr(e)[:200])
                continue
            pe = np.abs(np.exp(v) - np.exp(ref))
            res[m] = dict(us_per_orthant=1e6 * sec / n, dP_median=float(np.median(pe)), dP_max=float(pe.max()),
                          n_nonfinite=int((~np.isfinite(v)).sum()), n_zero=int((v <= -690).sum()))
            if name == "shares":
                res[m]["sum_dev_max"] = float(np.abs(np.exp(v).reshape(NSHARE, -1).sum(1) - 1).max())
        out[name] = res
        print(f"\n## {name} ({src}): d = {res['dim']}, median P {res['median_P']:.3f}, "
              f"reference SE(log P) median {res['ref_se_logP_median']:.1e}"
              + (f", reference |sum - 1| max {res['ref_sum_dev_max']:.1e}" if name == "shares" else ""))
        print("| method | µs / orthant | |ΔP| median (max) | zeros |" + (" |sum − 1| max |" if name == "shares" else "")
              + "\n|---|---|---|---|" + ("---|" if name == "shares" else ""))
        for m in meth:
            q = res[m]
            print(f"| {m} | {q['error']} |" if "error" in q else
                  f"| {m} | {q['us_per_orthant']:.0f} | {q['dP_median']:.1e} ({q['dP_max']:.1e}) | {q['n_zero']} |"
                  + (f" {q['sum_dev_max']:.1e} |" if name == "shares" else ""), flush=True)
    (t24.OUT / "engine.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
