"""T1 MVP refit with the total error variance fixed (2026-10-08).

The original fits pin the scale by sigma_nu alone (2% of the error variance in G1/G2), which
leaves the overall scale weakly identified: raw sigma ran ~1 SE high and one G2 replicate ran
away (x18). Here sigma_b^2 + sigma_f^2 + sigma_p^2 + sigma_nu^2 = pi^2/6 (true in every truth)
and the fit estimates the split, u_a = log(sigma_a^2 / sigma_nu^2) (ghk_gpu.sig2_of).

    python analysis/T1/mvp_refit.py G0 G1 G2

All replicates of all truths fit in parallel on Modal T4s. Writes
out/full_<truth>/results_mvp_total.json (one record per replicate, same fields as
results.json's "mvp" plus delta-method SEs of log sigma_a and S4 coverage flags). HB/RFC
results are untouched.
"""
import json
import sys
import time

import numpy as np

import ghk_gpu
import run

P = run.P


def softmax0(u):
    z = np.r_[u, 0.0]
    e = np.exp(z - z.max())
    return e / e.sum()


def main(truths):
    th0 = np.r_[np.zeros(P), np.full(P, np.log(.5)), np.zeros(3)]
    jobs, keys, data = [], [], {}
    for truth in truths:
        sh = run.TRUTHS[truth]
        sig = np.sqrt(run.T_VAR * sh)
        logit = truth in run.LOGIT_TRUTHS
        u_true = np.log(np.maximum(sh[:3], 1e-6) / sh[3])
        th_true = np.r_[run.B_TRUE, np.log(run.W_TRUE), u_true]
        recs = json.loads((run.HERE / "out" / f"full_{truth}" / "results_pre_rfc_g.json").read_text())
        for rec in recs:
            rep = rec["rep"]
            r_data, r_hold, _ = [np.random.default_rng(s) for s in np.random.SeedSequence([run.SEED, rep]).spawn(3)]
            d = run.make_data(r_data, rec["N"], rec["T"], sig, logit)
            H = run.holdout_tasks(r_hold)
            yh = run.simulate(r_hold, np.broadcast_to(H, (rec["N"],) + H.shape), d["beta"], sig, logit)
            obs = np.stack([np.bincount(yh[:, i], minlength=4) / rec["N"] for i in range(6)])
            Xd, G = ghk_gpu.panel_arrays(d["prods"], d["y"], run.task_X)
            jobs.append((Xd, G, 0.0, th0, th_true, 4096, run.SEED + rep))
            keys.append((truth, rep))
            data[(truth, rep)] = dict(H=H, obs=obs, rec=rec, th_true=th_true, sig=sig)
    run.log(f"MVP refit (total variance fixed): {len(jobs)} GHK fits on GPU, in parallel")
    t0 = time.time()
    with ghk_gpu.app.run():
        res = list(ghk_gpu.fit.starmap(jobs, kwargs=dict(tvar=float(run.T_VAR)), return_exceptions=True))
    run.log(f"GPU fits done: {time.time() - t0:.0f} s wall")
    out = {t: [] for t in truths}
    for (truth, rep), r in zip(keys, res):
        if isinstance(r, BaseException):
            run.log(f"{truth} rep {rep}: FAILED {r!r}"[:300])
            out[truth].append(dict(rep=rep, truth=truth, mvp=None, error=repr(r)[:2000]))
            continue
        x = data[(truth, rep)]
        th = np.array(r["theta"])
        V = np.array(r["cov_sandwich"])
        u = th[2 * P:]
        s2 = run.T_VAR * (softmax0(u) * (1 - 4e-6) + 1e-6)  # as ghk_gpu.sig2_of
        s_hat = np.sqrt(s2)
        b, w = th[:P], np.exp(th[P:2 * P])
        # delta method: log sigma_a = (log T_VAR + u_a - logsumexp(u, 0)) / 2
        J = 0.5 * (np.eye(3) - softmax0(u)[None, :3])  # floor ignored (1e-6)
        se_log_sig = np.sqrt(np.diag(J @ V[2 * P:, 2 * P:] @ J.T))
        sig = x["sig"]
        z_sig = (np.log(s_hat[:3]) - np.log(np.maximum(sig[:3], 1e-300))) / se_log_sig
        fn = lambda pr: run.pop_shares(pr, b, w, s_hat)  # noqa: E731
        mh = np.stack([fn(x["H"][i]) for i in range(6)])
        S1 = float(run.pair_corr_probit(s_hat))
        truth_rec = x["rec"]["truth"]
        m = dict(theta=th.tolist(), theta_true=x["th_true"].tolist(), se_sandwich=r["se_sandwich"],
                 se_hessian=r["se_hessian"], sigma=s_hat.tolist(), se_log_sigma=se_log_sig.tolist(),
                 z_sigma=z_sig.tolist(), S4_cover=(np.abs(z_sig) <= 1.96).tolist() if sig[0] > 0 else None,
                 ll=-r["nll"], ll_true=-r["nll_true"], iters=r["n_iter"], grad_max=r["grad_max"],
                 converged=r["n_iter"] < 500, normalisation="total error variance = pi^2/6",
                 S1=S1, S2=(s2 / s2.sum()).tolist(), S1_err=S1 - truth_rec["S1"],
                 S3_ratio=S1 / truth_rec["S1"] if truth_rec["S1"] else None,
                 holdout_mae=float(np.abs(mh - x["obs"]).mean()), seconds=r["seconds"],
                 **run.score(run.decisions(fn), truth_rec))
        out[truth].append(dict(rep=rep, truth=truth, mvp=m))
        run.log(f"{truth} rep {rep}: {r['n_iter']} iters, sigma {s_hat.round(3)} vs {sig.round(3)}, "
                f"S1 {S1:.3f}, ll {-r['nll']:.2f} vs truth {-r['nll_true']:.2f}")
    for truth in truths:
        (run.HERE / "out" / f"full_{truth}" / "results_mvp_total.json").write_text(json.dumps(out[truth], indent=1))
    run.log("done")


if __name__ == "__main__":
    run.LOGF = open(run.HERE / "out" / "log_mvp_refit.txt", "a")
    main(sys.argv[1:])
