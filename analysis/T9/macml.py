"""T9 MACML (Bhat): pairwise composite marginal likelihood for T1's panel probit, on GPU.

Per respondent, every pair of choice tasks (t < s) contributes log P(both choices): a 2(J-1) = 6-dim
orthant with the cross-task covariance from the random coefficients,
    a = (eta_t, eta_s),  Sigma = Z Omega Z' + blockdiag(G_t, G_s)   (rows of tasks t and s),
as T1's full likelihood (analysis/T1/ghk_gpu.py) restricted to the pair. Each orthant is Bhat's
(2018) OVUS approximation: `ovus_log` is a batched torch port of pybhatlib 0.4.0's
`gradmvn._mvncd._mvncd_ovus` (same ordering, conditioning, floors and clips), with pybhatlib's own
torch Genz BVN (`gradmvn._mvncd_torch.bvn_cdf_torch`). The LDLT rank-1 update of the CPU code is
written here in covariance form (the same matrices). Gradients by autograd. SEs: Godambe sandwich
H^-1 J H^-1, J from per-respondent composite scores.

Ephemeral Modal app; one T4 per fit.
"""
import time

import modal
import numpy as np

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy", "pybhatlib==0.4.0"))
app = modal.App("orthant-rfc-t9-macml")


def _bvn_high_branch_safe(dh, dk, r, gl20_w, gl20_x, signs):
    """pybhatlib 0.4.0 `_mvncd_torch._bvn_high_branch`, verbatim except sqrt(bs) floored at 1e-300:
    at dh == k (exact |a/sd| ties give equal limits) sqrt'(0) = inf made the gradient NaN even when
    torch.where discards this branch. The value changes by < 1e-150."""
    import torch
    from pybhatlib.gradmvn._mvncd_torch import _SQRT_TWOPI, _TWOPI, _ndtr_torch

    neg_r = r < 0
    k = torch.where(neg_r, -dk, dk)
    hk = dh * k
    ass1 = (1.0 - r) * (1.0 + r)
    a_val = torch.sqrt(ass1.clamp(min=1e-30))
    bs = (dh - k) ** 2
    c = (4.0 - hk) / 8.0
    d = (12.0 - hk) / 16.0
    asr_h = torch.where(ass1 > 1e-30, -(bs / ass1.clamp(min=1e-30) + hk) / 2.0, torch.full_like(ass1, -200.0))
    term1 = torch.where(
        asr_h > -100,
        a_val * torch.exp(asr_h.clamp(min=-200.0)) * (
            1.0 - c * (bs - ass1) * (1.0 - d * bs / 5.0) / 3.0 + c * d * ass1 * ass1 / 5.0),
        torch.zeros_like(a_val))
    b = torch.sqrt(bs.clamp(min=1e-300))                                   # pybhatlib: clamp(min=0.0)
    second_valid = (-hk < 100) & (a_val > 1e-30)
    term2 = torch.where(
        second_valid,
        torch.exp((-hk / 2.0).clamp(min=-200.0)) * _SQRT_TWOPI * _ndtr_torch(-b / a_val.clamp(min=1e-30))
        * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0),
        torch.zeros_like(a_val))
    bvn = term1 - term2
    a_half = a_val / 2.0
    xs = (a_half[:, None, None] * (signs[None, None, :] * gl20_x[None, :, None] + 1.0)) ** 2
    rs = torch.sqrt((1.0 - xs).clamp(min=0.0))
    valid = (xs > 1e-30) & (rs > 1e-30)
    asr2 = -(bs[:, None, None] / xs.clamp(min=1e-30) + hk[:, None, None]) / 2.0
    exp_valid = valid & (asr2 > -100)
    exp_term = torch.where(
        exp_valid,
        torch.exp((-hk[:, None, None] * (1.0 - rs) / (2.0 * (1.0 + rs).clamp(min=1e-30))).clamp(min=-200.0))
        / rs.clamp(min=1e-30) - (1.0 + c[:, None, None] * xs * (1.0 + d[:, None, None] * xs)),
        torch.zeros_like(xs))
    gl_contrib = torch.where(
        exp_valid, a_half[:, None, None] * gl20_w[None, :, None] * torch.exp(asr2.clamp(min=-200.0)) * exp_term,
        torch.zeros_like(xs))
    bvn = bvn + gl_contrib.sum(dim=(1, 2))
    bvn = -bvn / _TWOPI
    return torch.where(
        neg_r,
        -bvn + torch.where(k > dh, _ndtr_torch(k) - _ndtr_torch(dh), torch.zeros_like(bvn)),
        bvn + _ndtr_torch(-torch.maximum(dh, k)))


def _patch_bvn():
    from pybhatlib.gradmvn import _mvncd_torch
    _mvncd_torch._bvn_high_branch = _bvn_high_branch_safe   # bvn_cdf_torch looks it up at call time


def ovus_order(a, S):
    """OVUS variable order: |a_k / sd_k| ascending (pybhatlib _reorder_by_limits; np.argsort is stable at
    K < 16 on older numpy, so ties break stably here)."""
    import torch
    with torch.no_grad():
        sd0 = S.diagonal(dim1=1, dim2=2).clamp_min(1e-30).sqrt()
        return (a / sd0).abs().sort(dim=1, stable=True).indices


def ovus_log(a, S, order=None):
    """log P(X <= a), X ~ N(0, S), by OVUS. a (B, K), S (B, K, K), K >= 3. order (B, K): a frozen
    variable order (the fit's smooth objective); None = OVUS's own order at these a, S."""
    import torch
    from pybhatlib.gradmvn._mvncd_torch import bvn_cdf_torch
    from torch.special import log_ndtr

    _patch_bvn()
    B, K = a.shape
    eps, tiny = 1e-15, 1e-300
    if order is None:
        order = ovus_order(a, S)
    a = a.gather(1, order)
    S = S.gather(1, order[:, :, None].expand(B, K, K)).gather(2, order[:, None, :].expand(B, K, K))
    m = torch.zeros_like(a)

    def bvn_step(C, m, i):
        """BVN of the first two current variables, limits a[:, i], a[:, i+1] (pybhatlib _get_bvn_params)."""
        s0 = C[:, 0, 0].clamp_min(eps).sqrt()
        s1 = C[:, 1, 1].clamp_min(eps).sqrt()
        rho = (C[:, 0, 1] / (s0 * s1)).clamp(-0.9999, 0.9999)
        w0 = (a[:, i] - m[:, 0]) / s0
        return bvn_cdf_torch(w0, (a[:, i + 1] - m[:, 1]) / s1, rho), w0

    p1, _ = bvn_step(S, m, 0)
    logp = p1.clamp_min(tiny).log()
    C = S
    for h in range(K - 2):
        sig_h = C[:, 0, 0].clamp_min(eps)
        sd_h = sig_h.sqrt()
        w = (a[:, h] - m[:, 0]) / sd_h
        lam = -torch.exp(-0.5 * w * w - 0.5 * np.log(2 * np.pi) - log_ndtr(w))   # E[Z | Z <= w]
        var_z = (1.0 + lam * (w - lam)).clamp_min(eps)
        v = C[:, 1:, 0] / C[:, :1, 0]                         # L[1:, 0]
        m = m[:, 1:] + v * (sd_h * lam)[:, None]
        C = C[:, 1:, 1:] + ((sig_h * var_z - sig_h)[:, None, None]) * v[:, :, None] * v[:, None, :]
        bvn, wn = bvn_step(C, m, h + 1)
        # pybhatlib: p_h = bvn / Phi(w_next_0); prob *= max(1e-300, p_h)
        logp = logp + (bvn.clamp_min(tiny).log() - log_ndtr(wn)).clamp_min(np.log(tiny))
    return logp


def pair_index(T, K):
    import itertools
    return np.array([list(range(t * K, t * K + K)) + list(range(s * K, s * K + K))
                     for t, s in itertools.combinations(range(T), 2)])


def pair_orthants(th, Xd, G, tvar, IDX):
    """Pair orthants (n * npair, 2K) limits and (n * npair, 2K, 2K) covariances. th (n, 2P+3)."""
    import torch

    n, T, K, P = Xd.shape
    b, sd = th[:, :P], th[:, P:2 * P].exp()
    u = th[:, 2 * P:]                                   # as T1 ghk_gpu.sig2_of(u, tvar=tvar)
    p = torch.softmax(torch.cat([u, torch.zeros_like(u[..., :1])], -1), -1)
    sig2 = tvar * (p * (1 - 4e-6) + 1e-6)
    Z = Xd.reshape(n, T * K, P)
    eta = (Z * b[:, None, :]).sum(-1)
    cov = (Z * sd[:, None, :] ** 2) @ Z.transpose(1, 2)
    blk = torch.einsum("na,antij->ntij", sig2, G)
    for t in range(T):
        cov[:, t * K:(t + 1) * K, t * K:(t + 1) * K] += blk[:, t]
    a = eta[:, IDX]                                                   # (n, npair, 2K)
    S = cov[:, IDX[:, :, None], IDX[:, None, :]]                      # (n, npair, 2K, 2K)
    d = IDX.shape[1]
    return a.reshape(-1, d), S.reshape(-1, d, d)


def cl_rows(th, Xd, G, tvar, IDX, order=None):
    """Per-respondent composite log-likelihood (sum over task pairs). order (n * npair, 2K) or None."""
    a, S = pair_orthants(th, Xd, G, tvar, IDX)
    return ovus_log(a, S, order).reshape(Xd.shape[0], IDX.shape[0]).sum(1)


@app.function(image=image, gpu="T4", timeout=7200)
def fit(Xd, G, th0, th_true, tvar, se=False, chunk=200, max_iter=500, max_rounds=8):
    """MACML fit. OVUS re-sorts variables at every theta, so its composite likelihood jumps where an
    order flips and the line search stalls (T9 gate diagnosis, 2026-10-10). Rounds: freeze every
    orthant's order at the current theta, L-BFGS on the now smooth objective, re-sort at the new theta;
    stop when no order changes, so at the fit every orthant is pybhatlib's OVUS value."""
    import torch

    dev, f64 = torch.device("cuda"), torch.float64
    Xd = torch.as_tensor(Xd, dtype=f64, device=dev)
    G = torch.as_tensor(G, dtype=f64, device=dev)
    N, T, K, _ = Xd.shape
    IDX = torch.as_tensor(pair_index(T, K), device=dev)

    npair = IDX.shape[0]
    ORD = None

    def orders(th):
        with torch.no_grad():
            return torch.cat([ovus_order(*pair_orthants(th.expand(min(lo + chunk, N) - lo, -1), Xd[lo:min(lo + chunk, N)],
                                                        G[:, lo:min(lo + chunk, N)], tvar, IDX))
                              for lo in range(0, N, chunk)])

    def rows(th, lo, hi):
        return cl_rows(th, Xd[lo:hi], G[:, lo:hi], tvar, IDX, ORD[lo * npair:hi * npair])

    def nll(th, lo, hi):
        return -rows(th.expand(hi - lo, -1), lo, hi).sum()

    theta = torch.as_tensor(th0, dtype=f64, device=dev).clone().requires_grad_(True)

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
    ORD = orders(theta.detach())
    changes, iters = [], []
    for _ in range(max_rounds):
        opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-7,
                                tolerance_change=1e-12, line_search_fn="strong_wolfe")
        opt.step(closure)
        iters.append(int(opt.state[opt._params[0]]["n_iter"]))
        new = orders(theta.detach())
        changes.append(int((new != ORD).any(1).sum()))
        ORD = new
        if changes[-1] == 0:
            break
    final = closure().item() * N
    torch.cuda.synchronize()
    t_fit = time.time() - t0
    grad_max = float(theta.grad.abs().max() * N)
    if not np.isfinite(grad_max) or not np.isfinite(final):
        raise FloatingPointError(f"non-finite composite ll / gradient at the fit: {final}, {grad_max}")
    n_iter = sum(iters)
    th = theta.detach()
    with torch.no_grad():
        tt = torch.as_tensor(th_true, dtype=f64, device=dev)
        ORD_fit, ORD = ORD, orders(tt)                      # OVUS's own order at th_true
        ncl_true = sum(float(nll(tt, lo, min(lo + chunk, N))) for lo in range(0, N, chunk))
        ORD = ORD_fit
    out = dict(theta=th.cpu().numpy().tolist(), ncl=final, ncl_true=ncl_true, n_iter=n_iter, max_iter=max_iter,
               order_rounds=len(iters), order_changes=changes, iters_per_round=iters,
               grad_max=grad_max, seconds=dict(fit=t_fit, se=0.0), gpu=torch.cuda.get_device_name(),
               cov_sandwich=None)
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
        out["cov_sandwich"] = (Hi @ (Sc.T @ Sc) @ Hi).cpu().numpy().tolist()   # Godambe
        out["seconds"]["se"] = time.time() - t0
    return out
