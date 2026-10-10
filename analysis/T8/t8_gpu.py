"""T8 (GPR) probit fit on GPU: T1's exact panel likelihood by GHK, with the within-task error
covariance generalised to  sum_c w_c S_c  (+ w_rbf exp(-D2 / 2 l^2))  (+ nugget, the last S_c).

Weights: total error variance fixed at tvar, w = tvar * softmax(u, 0) over the components (the
last fixed component, the product nugget, is the reference). theta = (b (P), log omega (P),
u (n_comp - 1), [log l]). "none" carries the total variance in every component, as T1.
Ephemeral Modal app; one T4 per fit.
"""
import time

import modal
import numpy as np

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")
app = modal.App("orthant-rfc-t8-ghk")


def panel_arrays(prods, y, task_X, S_fn, sweet=None):
    """Differenced panel arrays. prods (N, T, K, d); S_fn(prods) -> (C, N, T, K1, K1) fixed components
    (last = nugget; "none" = 1 in each). Returns Xd (N, T, K, P), G (C, N, T, K, K), and for the RBF
    M (N, T, K, K1) and D2 (N, T, K1, K1) (none row/col = 0)."""
    N, T, K, _ = prods.shape
    K1 = K + 1
    X = task_X(prods)
    oth = np.array([[k for k in range(K1) if k != c] for c in range(K1)])[y]
    M = np.zeros((N, T, K, K1))
    np.put_along_axis(M, oth[..., None], -1.0, axis=3)
    np.put_along_axis(M, np.broadcast_to(y[:, :, None, None], (N, T, K, 1)), 1.0, axis=3)
    S = S_fn(prods)
    G = np.einsum("ntik,antkl,ntjl->antij", M, S, M)
    D2 = None
    if sweet is not None:
        s = sweet(prods)                                                   # (N, T, K)
        D2 = np.zeros((N, T, K1, K1))
        D2[:, :, :K, :K] = (s[..., :, None] - s[..., None, :]) ** 2
    return M @ X, G, M, D2


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


def weights_of(u, tvar):
    """Component variances from u (..., C - 1) (+ the rbf weight if present): tvar * softmax(u, 0),
    floored at 1e-6 of tvar (a weight at 0 would make a pair's covariance singular)."""
    import torch
    p = torch.softmax(torch.cat([u, torch.zeros_like(u[..., :1])], -1), -1)
    return tvar * (p * (1 - 1e-6 * p.shape[-1]) + 1e-6)


def _ll_rows(th, Xd, G, M, D2, w, P, C, rbf, tvar):
    """Per-respondent log-likelihood. Component order: G's fixed components except the nugget,
    then rbf (if any), then the nugget last."""
    import torch

    n, T, K, _ = Xd.shape
    b, sd = th[:, :P], th[:, P:2 * P].exp()
    nu = C - 1 + (1 if rbf else 0)                       # free u's
    wts = weights_of(th[:, 2 * P:2 * P + nu], tvar)       # (n, C + rbf)
    Z = Xd.reshape(n, T * K, P)
    eta = (Z * b[:, None, :]).sum(-1)
    cov = (Z * sd[:, None, :] ** 2) @ Z.transpose(1, 2)
    fixed = torch.cat([wts[:, :C - 1], wts[:, -1:]], 1)  # fixed comps incl. nugget, in G's order
    blk = torch.einsum("na,antij->ntij", fixed, G)
    if rbf:
        ell = th[:, -1].exp()
        K1 = D2.shape[-1]
        prod = torch.ones(K1, K1, dtype=D2.dtype, device=D2.device)
        prod[K1 - 1, :] = 0
        prod[:, K1 - 1] = 0
        none = torch.zeros_like(prod)
        none[K1 - 1, K1 - 1] = 1
        S = torch.exp(-D2 / (2 * ell[:, None, None, None] ** 2)) * prod + none  # no in-place on autograd output
        blk = blk + wts[:, C - 1, None, None, None] * (M @ S @ M.transpose(-1, -2))
    for t in range(T):
        cov[:, t * K:(t + 1) * K, t * K:(t + 1) * K] += blk[:, t]
    return _ghk(eta, torch.linalg.cholesky(cov), w)


@app.function(image=image, gpu="T4", timeout=7200)
def fit(Xd, G, M, D2, P, rbf, th0, th_true, Msob, seed, tvar, se=False, chunk=100, max_iter=500):
    import torch

    dev, f64 = torch.device("cuda"), torch.float64
    Xd = torch.as_tensor(Xd, dtype=f64, device=dev)
    G = torch.as_tensor(G, dtype=f64, device=dev)
    Mt = torch.as_tensor(M, dtype=f64, device=dev) if rbf else None
    D2t = torch.as_tensor(D2, dtype=f64, device=dev) if rbf else None
    C = G.shape[0]
    N, T, K, _ = Xd.shape
    D = T * K
    sob = torch.quasirandom.SobolEngine(D - 1, scramble=True, seed=seed).draw(Msob, dtype=f64).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    shift = torch.rand(N, 1, D - 1, generator=gen, dtype=f64, device=dev)

    def w(lo, hi):
        return torch.remainder(sob[None] + shift[lo:hi], 1.0)

    def rows(th, lo, hi):
        return _ll_rows(th, Xd[lo:hi], G[:, lo:hi], Mt[lo:hi] if rbf else None, D2t[lo:hi] if rbf else None,
                        w(lo, hi), P, C, rbf, tvar)

    def nll(th, lo, hi):
        return -rows(th.expand(hi - lo, -1), lo, hi).sum()

    theta = torch.as_tensor(th0, dtype=f64, device=dev).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-7,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        tot = 0.0
        for lo in range(0, N, chunk):
            l = nll(theta, lo, min(lo + chunk, N)) / N
            l.backward()
            tot += l.item()
        return torch.tensor(tot)

    torch.cuda.synchronize()
    t0 = time.time()
    opt.step(closure)
    final = closure().item() * N
    torch.cuda.synchronize()
    t_fit = time.time() - t0
    grad_max = float(theta.grad.abs().max() * N)
    n_iter = int(opt.state[opt._params[0]]["n_iter"])
    th = theta.detach()
    with torch.no_grad():
        tt = torch.as_tensor(th_true, dtype=f64, device=dev)
        nll_true = sum(float(nll(tt, lo, min(lo + chunk, N))) for lo in range(0, N, chunk)) if len(th_true) == len(th) else None
    out = dict(theta=th.cpu().numpy().tolist(), nll=final, nll_true=nll_true, n_iter=n_iter, grad_max=grad_max,
               seconds=dict(fit=t_fit, se=0.0), gpu=torch.cuda.get_device_name(), cov_sandwich=None)
    if se:
        t0 = time.time()
        nth = th.numel()
        H = torch.zeros(nth, nth, dtype=f64, device=dev)
        Sc = torch.zeros(N, nth, dtype=f64, device=dev)
        for lo in range(0, N, chunk // 2):
            hi = min(lo + chunk // 2, N)
            thr = th.expand(hi - lo, -1).clone().requires_grad_(True)
            ll = rows(thr, lo, hi)
            g = torch.autograd.grad(ll.sum(), thr, create_graph=True)[0]
            Sc[lo:hi] = g.detach()
            gs = g.sum(0)
            for i in range(nth):
                H[i] += torch.autograd.grad(gs[i], thr, retain_graph=i < nth - 1)[0].sum(0).detach()
        Hi = torch.linalg.inv(-H)
        V = Hi @ (Sc.T @ Sc) @ Hi
        out["cov_sandwich"] = V.cpu().numpy().tolist()
        out["seconds"]["se"] = time.time() - t0
    return out
