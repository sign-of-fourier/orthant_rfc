"""T1 estimator diagnostics on the recovery-check dataset (N = 1,000, seed [SEED, 999]).

Why the selftest recovery fit leaned low (all taste spreads below the truth, S1 0.80 vs 0.90):
objective bias (finite MSL draws or engine 3-dim values) or the optimizer stopping early.

    python analysis/T1/diag.py nll            # nll at the truth for R = 200, 500
    python analysis/T1/diag.py from-truth     # refit at R = 200 starting at the truth

Writes analysis/T1/out/selftest/diag_<mode>.json and log_diag_<mode>.txt.
"""
import json
import sys
import time

import numpy as np

import run

N, T = 1000, 12


def setup(R):
    sig = np.sqrt(run.T_VAR * run.TRUTHS["G1"])
    d = run.make_data(np.random.default_rng([run.SEED, 999]), N, T, sig)
    return sig, run.MSL(d["prods"], d["y"], R, seed=run.SEED + 999)


def main():
    mode = sys.argv[1]
    out = run.HERE / "out" / "selftest"
    run.LOGF = open(out / f"log_diag_{mode}.txt", "w")
    th_true = None
    res = {}
    if mode == "nll":
        for R in (200, 500):
            sig, m = setup(R)
            th_true = np.r_[run.B_TRUE, np.log(run.W_TRUE), np.log(sig[:3])]
            t0 = time.time()
            res[f"nll_truth_R{R}"] = float(-m.ll_vec(*run.unpack(th_true, sig[3])).sum())
            run.log(f"R={R}: nll at truth {res[f'nll_truth_R{R}']:.3f} ({time.time() - t0:.0f} s)")
    elif mode == "from-truth":
        sig, m = setup(200)
        th_true = np.r_[run.B_TRUE, np.log(run.W_TRUE), np.log(sig[:3])]
        fit, t = run.fit_probit(m, sig[3], th_true.copy(), "MVP from truth")
        s_hat = np.exp(fit.x[2 * run.P:])
        res = dict(theta=fit.x.tolist(), theta_true=th_true.tolist(), nll=float(fit.fun),
                   nll_truth=float(-m.ll_vec(*run.unpack(th_true, sig[3])).sum()),
                   sigma_hat=s_hat.tolist(), sigma_true=sig[:3].tolist(),
                   S1=float(run.pair_corr_probit(np.r_[s_hat, sig[3]])),
                   w_hat=np.exp(fit.x[run.P:2 * run.P]).tolist(), seconds=t)
        run.log(f"from truth: sigma {s_hat.round(3)}, S1 {res['S1']:.3f}, nll {fit.fun:.3f} "
                f"(truth {res['nll_truth']:.3f})")
    (out / f"diag_{mode}.json").write_text(json.dumps(res, indent=1))
    run.log("done")


if __name__ == "__main__":
    main()
