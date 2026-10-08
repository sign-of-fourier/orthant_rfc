"""T1 MVP fit on GPU: the exact 36-dim panel likelihood by GHK with fixed scrambled Sobol draws.

Replaces the MSL arm of run.py (taste draws x 3-dim engine orthants), which is biased at
R = 200-500 and takes ~2 h per fit on CPU. Here the taste integral is exact: a respondent's
12 choices are one orthant of dimension T(J-1) = 36 with covariance
    Delta (X diag(w^2) X' + blockdiag_t sum_a sigma_a^2 S_a(t)) Delta',
evaluated by GHK. The draws are fixed for the whole fit, so the simulated likelihood is smooth
and L-BFGS with exact (autograd) gradients and Hessian applies. GHK is not the orthant engine:
the engine's 36-dim values were too rough to optimise (DESIGN.md, option 1). Population shares
and decisions still use the engine (pop_shares, 3-4 dims).

    python analysis/T1/ghk_gpu.py recovery          # selftest recovery dataset, M = 1024 and 4096

Runs as an ephemeral Modal app (nothing deployed); one T4 per fit, fits in parallel.
"""
import json
import sys
import time

import modal
import numpy as np

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")
app = modal.App("orthant-rfc-t1-ghk")


def panel_arrays(prods, y, task_X, levels=(4, 3, 3)):
    """Per respondent, per task: Xd (N, T, K, P) and S (4, N, T, K, K), differences chosen - other.
    S: brand, flavor, pack shared-level indicators and the product error; "none" is in all four
    (its error variance is the total, as run.err_cov)."""
    N, T, K, _ = prods.shape
    K1 = K + 1
    X = task_X(prods)                                                     # (N, T, K1, P)
    oth = np.array([[k for k in range(K1) if k != c] for c in range(K1)])[y]
    M = np.zeros((N, T, K, K1))
    np.put_along_axis(M, oth[..., None], -1.0, axis=3)
    np.put_along_axis(M, np.broadcast_to(y[:, :, None, None], (N, T, K, 1)), 1.0, axis=3)
    S = np.zeros((4, N, T, K1, K1))
    for a in range(3):
        lv = prods[..., a]
        S[a, :, :, :K, :K] = lv[..., :, None] == lv[..., None, :]
    S[3, :, :, :K, :K] = np.eye(K)
    S[:, :, :, K, K] = 1
    Xd = M @ X
    G = np.einsum("ntik,antkl,ntjl->antij", M, S, M)                       # (4, N, T, K, K)
    return Xd, G


def _ghk(eta, L, w):
    import math

    import torch
    from torch.special import log_ndtr, ndtri

    n, d = eta.shape
    tiny = torch.finfo(eta.dtype).tiny
    one = 1.0 - torch.finfo(eta.dtype).eps
    logp = eta.new_zeros(n, w.shape[1])
    us = []
    for j in range(d):
        c = eta[:, j, None]
        for k in range(j):
            c = c + L[:, j, k, None] * us[k]
        lp = log_ndtr(c / L[:, j, j, None])
        logp = logp + lp
        if j < d - 1:
            us.append(-ndtri((w[:, :, j] * lp.exp()).clamp(tiny, one)))
    return torch.logsumexp(logp, 1) - math.log(w.shape[1])


def sig2_of(u, snu=None, tvar=None):
    """Error variances (brand, flavor, pack, product) from the last 3 parameters u.
    snu: u = log sigma_a, sigma_nu fixed (scale pinned by sigma_nu alone).
    tvar: u_a = log(sigma_a^2 / sigma_nu^2), total error variance fixed at tvar."""
    import torch

    if tvar is not None:
        # floor each share at 1e-6 (total unchanged): a line-search step that sends a share to 0
        # would otherwise make a near-duplicate pair's covariance singular
        p = torch.softmax(torch.cat([u, torch.zeros_like(u[..., :1])], -1), -1)
        return tvar * (p * (1 - 4e-6) + 1e-6)
    return torch.cat([(2 * u).exp(), torch.full_like(u[..., :1], snu**2)], -1)


def _ll_rows(th, Xd, G, snu, w, tvar=None):
    """Per-respondent log-likelihood. th (n, 2P+3) per row: b, log w, u (see sig2_of).
    Xd (n, T, K, P), G (4, n, T, K, K), w (n, M, T*K - 1)."""
    import torch

    n, T, K, P = Xd.shape
    b, sd = th[:, :P], th[:, P:2 * P].exp()
    sig2 = sig2_of(th[:, 2 * P:], snu, tvar)                                # (n, 4)
    Z = Xd.reshape(n, T * K, P)
    eta = (Z * b[:, None, :]).sum(-1)
    cov = (Z * sd[:, None, :] ** 2) @ Z.transpose(1, 2)
    blk = torch.einsum("na,antij->ntij", sig2, G)                         # (n, T, K, K)
    for t in range(T):
        cov[:, t * K:(t + 1) * K, t * K:(t + 1) * K] += blk[:, t]
    return _ghk(eta, torch.linalg.cholesky(cov), w)


@app.function(image=image, gpu="T4", timeout=3600)
def fit(Xd, G, snu, th0, th_true, M, seed, chunk=100, max_iter=500, tvar=None):
    """L-BFGS on the summed GHK log-likelihood; Hessian and sandwich SEs at the optimum."""
    import torch

    dev = torch.device("cuda")
    f64 = torch.float64
    Xd = torch.as_tensor(Xd, dtype=f64, device=dev)
    G = torch.as_tensor(G, dtype=f64, device=dev)
    N, T, K, P = Xd.shape
    D = T * K
    sob = torch.quasirandom.SobolEngine(D - 1, scramble=True, seed=seed).draw(M, dtype=f64).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    shift = torch.rand(N, 1, D - 1, generator=gen, dtype=f64, device=dev)

    def w(lo, hi):
        return torch.remainder(sob[None] + shift[lo:hi], 1.0)

    def nll(th, lo, hi):
        return -_ll_rows(th.expand(hi - lo, -1), Xd[lo:hi], G[:, lo:hi], snu, w(lo, hi), tvar).sum()

    theta = torch.as_tensor(th0, dtype=f64, device=dev).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-7,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    nev = [0]

    def closure():
        opt.zero_grad()
        tot = 0.0
        for lo in range(0, N, chunk):
            l = nll(theta, lo, min(lo + chunk, N)) / N
            l.backward()
            tot += l.item()
        nev[0] += 1
        return torch.tensor(tot)

    torch.cuda.synchronize()
    t0 = time.time()
    opt.step(closure)
    final = closure().item() * N
    torch.cuda.synchronize()
    t_fit = time.time() - t0
    grad_max = float(theta.grad.abs().max() * N)
    n_iter = int(opt.state[opt._params[0]]["n_iter"])

    # Hessian (double backward) and per-respondent scores (theta replicated per row).
    t0 = time.time()
    th = theta.detach()
    nth = th.numel()
    H = torch.zeros(nth, nth, dtype=f64, device=dev)
    Sc = torch.zeros(N, nth, dtype=f64, device=dev)
    with torch.no_grad():
        tt = torch.as_tensor(th_true, dtype=f64, device=dev)
        nll_true = sum(float(nll(tt, lo, min(lo + chunk, N))) for lo in range(0, N, chunk))
    for lo in range(0, N, chunk // 2):
        hi = min(lo + chunk // 2, N)
        thr = th.expand(hi - lo, -1).clone().requires_grad_(True)
        ll = _ll_rows(thr, Xd[lo:hi], G[:, lo:hi], snu, w(lo, hi), tvar)
        g = torch.autograd.grad(ll.sum(), thr, create_graph=True)[0]          # (n, nth)
        Sc[lo:hi] = g.detach()
        gs = g.sum(0)
        for i in range(nth):
            H[i] += torch.autograd.grad(gs[i], thr, retain_graph=i < nth - 1)[0].sum(0).detach()
    Hi = torch.linalg.inv(-H)
    V = Hi @ (Sc.T @ Sc) @ Hi
    torch.cuda.synchronize()
    return dict(theta=th.cpu().numpy().tolist(), nll=final, nll_true=nll_true, n_iter=n_iter, n_evals=nev[0],
                grad_max=grad_max, se_sandwich=V.diagonal().clamp_min(0).sqrt().cpu().numpy().tolist(),
                se_hessian=Hi.diagonal().clamp_min(0).sqrt().cpu().numpy().tolist(),
                cov_sandwich=V.cpu().numpy().tolist(), tvar=tvar,
                M=M, seed=seed, seconds=dict(fit=t_fit, se=time.time() - t0), gpu=torch.cuda.get_device_name())


def recovery(Ms=(1024, 4096)):
    """The selftest recovery dataset (run.recovery_check: N = 1,000, seed [SEED, 999])."""
    import run

    N, T = 1000, 12
    sig = np.sqrt(run.T_VAR * run.TRUTHS["G1"])
    d = run.make_data(np.random.default_rng([run.SEED, 999]), N, T, sig)
    Xd, G = panel_arrays(d["prods"], d["y"], run.task_X)
    P = run.P
    th0 = np.r_[np.zeros(P), np.full(P, np.log(.5)), np.full(3, np.log(.5))]
    th_true = np.r_[run.B_TRUE, np.log(run.W_TRUE), np.log(sig[:3])]
    out = run.HERE / "out" / "selftest"
    run.LOGF = open(out / "log_recovery_ghk.txt", "w")
    run.log(f"recovery (GHK on GPU): N={N}, T={T}, D={T * 3}, M in {Ms}")
    t0 = time.time()
    with app.run():
        res = list(fit.starmap([(Xd, G, float(sig[3]), th0, th_true, M, run.SEED + 999) for M in Ms]))
    run.log(f"wall {time.time() - t0:.0f} s (fits in parallel, incl. container start)")
    recs = []
    for r in res:
        th, se = np.array(r["theta"]), np.array(r["se_sandwich"])
        s_hat = np.exp(th[2 * P:])
        z = (th - th_true) / se
        S1 = run.pair_corr_probit(np.r_[s_hat, sig[3]])
        rec = dict(r, sigma_hat=s_hat.tolist(), sigma_true=sig[:3].tolist(), z_all=z.tolist(),
                   z_sigma=z[2 * P:].tolist(), w_hat=np.exp(th[P:2 * P]).tolist(), S1=float(S1),
                   S1_true=float(run.pair_corr_probit(sig)), ok=bool(np.all(np.abs(z[2 * P:]) <= 2)))
        recs.append(rec)
        run.log(f"M={r['M']}: {r['n_iter']} iters, fit {r['seconds']['fit']:.0f} s, SEs {r['seconds']['se']:.0f} s, "
                f"max|grad| {r['grad_max']:.2e}, nll {r['nll']:.2f} vs truth {r['nll_true']:.2f}")
        run.log(f"  sigma {s_hat.round(3)} vs true {sig[:3].round(3)}; z_sigma {z[2 * P:].round(2)}; "
                f"S1 {S1:.3f} vs {rec['S1_true']:.3f}")
        run.log(f"  w z {z[P:2 * P].round(2)}; b z {z[:P].round(2)}")
    (out / "recovery_ghk.json").write_text(json.dumps(recs, indent=1))
    run.log("done")


if __name__ == "__main__":
    if sys.argv[1:] == ["recovery"]:
        recovery()
    else:
        sys.exit(__doc__)
