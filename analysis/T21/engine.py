"""T21 B · scoring real basket patterns on T19's fitted 21-dimensional MVP: engine vs OVUS.

    python analysis/T21/engine.py     # after t21.py (needs out/m3_r<r>_all.npz)

Per restaurant (all orders): 100 random holdout orders, each scored as P(Y = y | eta, R), the 21-dimensional
orthant with upper s * eta and covariance diag(s) R diag(s), s = 2y - 1. Two correlations: M3's raw pairwise-IFM
R (minimum eigenvalue about 1e-8, the near-singular case) and T19's shrunk R (0.95 R + 0.05 I).
Reference: GHK with scrambled Sobol, 2^15 points x 8 scrambles (T10's ghk_ref). Methods: engine (paid local
binary) at resolution high and low; OVUS by T9's batched torch port and by pybhatlib's row loop. Time per
orthant: best of 3 after a warm-up (T10's timed). Writes out/engine.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "T10"))
import ovus_vs_engine as T10  # noqa: E402  (puts T1, T4, T9 and rfc on the path)

rfc, ovus_log, _mvncd_ovus = T10.rfc, T10.ovus_log, T10._mvncd_ovus
NPROB = 100


def orthants(eta, Y, R):
    s = 2.0 * Y - 1.0
    return s * eta, s[:, :, None] * R[None] * s[:, None, :]


def main():
    out = {}
    for r in (1, 2):
        z = np.load(HERE / "out" / f"m3_r{r}_all.npz")
        idx = np.random.default_rng(21).choice(len(z["Y"]), NPROB, replace=False)
        for tag, R in (("raw", z["Rraw"]), ("shrunk", z["Rm"])):
            a, S = orthants(z["eta"][idx], z["Y"][idx].astype(float), R)
            ref, ref_se = T10.ghk_ref(a, S)
            name = f"r{r} {tag}"
            res = dict(dim=a.shape[1], min_eig=float(np.linalg.eigvalsh(R)[0]), median_P=float(np.exp(np.median(ref))),
                       ref_se_logP_median=float(np.median(ref_se)), ref_se_logP_max=float(ref_se.max()))
            meth = {
                "engine high": lambda: np.log(np.maximum(rfc._LOCAL.cdf(a, S, resolution="high"), 1e-300)),
                "engine low": lambda: np.log(np.maximum(rfc._LOCAL.cdf(a, S, resolution="low"), 1e-300)),
                "OVUS torch batch": lambda: ovus_log(torch.tensor(a), torch.tensor(S)).numpy(),
                "OVUS pybhatlib (row loop)": lambda: np.log([_mvncd_ovus(a[i], S[i]) for i in range(NPROB)]),
            }
            for m, fn in meth.items():
                try:
                    v, sec = T10.timed(fn)
                except Exception as e:                       # report, don't hide
                    res[m] = dict(error=repr(e)[:200])
                    continue
                e = np.abs(v - ref)
                pe = np.abs(np.exp(v) - np.exp(ref))
                res[m] = dict(us_per_orthant=1e6 * sec / NPROB, dP_median=float(np.median(pe)), dP_max=float(pe.max()),
                              dlogP_median=float(np.median(e)), dlogP_max=float(e.max()),
                              n_nonfinite=int((~np.isfinite(v)).sum()))
            out[name] = res
            print(f"\n## {name}: d = {res['dim']}, min eig {res['min_eig']:.1e}, median P {res['median_P']:.2e}; "
                  f"reference SE(log P) median {res['ref_se_logP_median']:.1e}, max {res['ref_se_logP_max']:.1e}")
            print("| method | µs / orthant | |ΔP| median (max) | |Δlog P| median (max) |\n|---|---|---|---|")
            for m in meth:
                q = res[m]
                print(f"| {m} | {q['error']} |" if "error" in q else
                      f"| {m} | {q['us_per_orthant']:.0f} | {q['dP_median']:.1e} ({q['dP_max']:.1e}) | "
                      f"{q['dlogP_median']:.2f} ({q['dlogP_max']:.2f}) |", flush=True)
    (HERE / "out" / "engine.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
