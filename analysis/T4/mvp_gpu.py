"""T4 probit fits on Modal GPUs (torch, float32 kernels, float64 parameters and sums).

Model (= the S1 DGP): U_ntk = x_ntk'beta_n + sum_a sigma_a zeta_{nta,level_a(k)} + s_k nu_ntk,
beta_n ~ N(b, diag w^2), s_k = sigma_nu for SKUs and sqrt(T_VAR) for "none"; the error shares
(brand, flavor, pack, product) are a softmax of u (3) with the total fixed at T_VAR (T1's tvar
parameterisation). theta = (b (11), log w (11), u (3)).

Panel likelihood, nested simulation (DESIGN.md revision: a joint draw of tastes and all 12 tasks'
components makes the 12-task product far too noisy):
    L_n = mean_q prod_t P_nt(beta_nq),   beta_nq from Q scrambled-Sobol taste draws per respondent,
and P_nt(beta) by one of
  - MVP-F (factor-conditioned): mean over R Sobol draws of the task's 12 components and the chosen
    alternative's own nu; given those, P = prod_{k != chosen} Phi((U_chosen - V_k - f_k) / s_k).
    Cost linear in J.
  - MVP-G (per-task GHK): the J-dim orthant of the differences against the chosen alternative,
    by GHK with R draws. Cost ~ J^2.
Draws are fixed for the whole fit (smooth simulated likelihood); L-BFGS with autograd gradients;
Hessian by double backward and sandwich SEs at the optimum.
"""
import math
import time

import modal
import numpy as np

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")
app = modal.App("orthant-rfc-t4-mvp")
# progress checkpoints (key -> dict), readable from anywhere: modal.Dict.from_name(...)[key]
PROGRESS = modal.Dict.from_name("orthant-rfc-t4-progress", create_if_missing=True)
T_VAR = math.pi ** 2 / 6
NLEV = 12
LEVELS = (5, 4, 3)


def sig2_of(u):
    import torch

    p = torch.softmax(torch.cat([u, torch.zeros_like(u[..., :1])], -1), -1)
    return T_VAR * (p * (1 - 4e-6) + 1e-6)


def _draws(N, T, P, Q, R, d_inner, seed, dev):
    """Taste normals (N, Q, P) and inner draws (N, T, R, d_inner) [uniforms for GHK, normals for F]:
    one scrambled Sobol set each, randomly shifted per respondent (tastes) and per task (inner)."""
    import torch

    f32 = torch.float32
    g = torch.Generator(device="cpu").manual_seed(seed)
    sb = torch.quasirandom.SobolEngine(P, scramble=True, seed=seed).draw(Q, dtype=torch.float64)
    sz = torch.quasirandom.SobolEngine(d_inner, scramble=True, seed=seed + 1).draw(R, dtype=torch.float64)
    ub = torch.remainder(sb[None] + torch.rand(N, 1, P, generator=g, dtype=torch.float64), 1.0)
    uz = torch.remainder(sz[None, None] + torch.rand(N, T, 1, d_inner, generator=g, dtype=torch.float64), 1.0)
    eps = 1e-7
    zb = torch.special.ndtri(ub.clamp(eps, 1 - eps)).to(f32).to(dev)
    return zb, uz.clamp(eps, 1 - eps).to(f32).to(dev)


class Lik:
    def __init__(self, X, y, A, Q, R, method, seed, dev, gh=0):
        import torch

        self.dev = dev
        self.X = torch.as_tensor(X, dtype=torch.float32, device=dev)          # (N, T, K1, P)
        self.y = torch.as_tensor(y, dtype=torch.long, device=dev)             # (N, T)
        self.A = torch.as_tensor(A, dtype=torch.float32, device=dev)          # (K1, NLEV)
        N, T, K1, P = self.X.shape
        self.N, self.T, self.K1, self.P = N, T, K1, P
        self.method = method
        self.gh = gh if method == "F" else 0
        d_inner = (NLEV if self.gh else NLEV + 1) if method == "F" else K1 - 2
        if self.gh:   # chosen alternative's own nu by gh-node Gauss-Hermite (probabilists')
            x, w = np.polynomial.hermite_e.hermegauss(gh)
            self.gh_x = torch.tensor(x, dtype=torch.float32, device=dev)
            self.gh_lw = torch.tensor(np.log(w / w.sum()), dtype=torch.float32, device=dev)
        self.zb, uz = _draws(N, T, P, Q, R, d_inner, seed, dev)
        self.zz = torch.special.ndtri(uz) if method == "F" else uz         # F: normals; G: uniforms
        self.Q, self.R = Q, R

    def _sig(self, th):
        import torch

        P = self.P
        b, w = th[:P].float(), th[P:2 * P].exp().float()
        s2 = sig2_of(th[2 * P:]).float()
        sa = torch.repeat_interleave(s2[:3].sqrt(), torch.tensor(LEVELS, device=self.dev))   # (NLEV,)
        s = torch.full((self.K1,), float(math.sqrt(T_VAR)), device=self.dev)
        s = torch.cat([s2[3].sqrt().expand(self.K1 - 1), s[-1:]])                          # SKUs, none
        return b, w, s2, sa, s

    def ll_rows(self, th, lo, hi):
        """Per-respondent simulated log-likelihood (hi - lo,), float64. th may be (nth,) or
        (hi - lo, nth) (per-row copies, for scores)."""
        import torch

        if th.dim() == 1:
            b, w, s2, sa, s = self._sig(th)
            beta = b + w * self.zb[lo:hi]                                    # (n, Q, P)
        else:
            parts = [self._sig(t) for t in th]
            beta = torch.stack([p[0] + p[1] * self.zb[lo + i] for i, p in enumerate(parts)])
            s2 = torch.stack([p[2] for p in parts])
            sa = torch.stack([p[3] for p in parts])
            s = torch.stack([p[4] for p in parts])
        X, y = self.X[lo:hi], self.y[lo:hi]
        V = torch.einsum("ntkp,nqp->nqtk", X, beta)                          # (n, Q, T, K1)
        lp = self._task_logp(V, y, sa, s, s2, lo, hi)                        # (n, Q, T)
        return (torch.logsumexp(lp.double().sum(2), 1) - math.log(self.Q))

    def _task_logp(self, V, y, sa, s, s2, lo, hi):
        import torch

        n = V.shape[0]
        per_row = sa.dim() == 2
        if self.method == "F":
            zz = self.zz[lo:hi]                                              # (n, T, R, NLEV+1)
            fz = zz[..., :NLEV] * (sa[:, None, None, :] if per_row else sa)
            f = fz @ self.A.T                                                # (n, T, R, K1)
            sv = s[:, None, None, :] if per_row else s                       # (.., K1)
            idx = y[:, :, None, None].expand(n, self.T, self.R, 1)
            fj = f.gather(3, idx)[..., 0]                                    # (n, T, R)
            sj = (s.gather(1, y) if per_row else s[y])                       # (n, T)
            Vj = V.gather(3, y[:, None, :, None].expand(n, self.Q, self.T, 1))[..., 0]   # (n, Q, T)
            sv5 = sv[:, None] if per_row else sv
            mask = torch.nn.functional.one_hot(y, self.K1).bool()[:, None, :, None, :]
            if self.gh:
                own = sj[..., None, None] * self.gh_x                                      # (n, T, 1, G)
                Uj = Vj[..., None, None] + (fj[..., None] + own)[:, None]                  # (n, Q, T, R, G)
                sv6 = sv5[..., None, :] if per_row else sv5
                arg = (Uj[..., None] - V[:, :, :, None, None, :] - f[:, None, :, :, None]) / sv6
                lk = torch.special.log_ndtr(arg).masked_fill(mask[..., None, :], 0.0)     # (n,Q,T,R,G,K1)
                lp = torch.logsumexp(lk.sum(5) + self.gh_lw, 4)                           # (n, Q, T, R)
                return torch.logsumexp(lp, 3) - math.log(self.R)
            Uj = Vj[..., None] + (fj + sj[..., None] * zz[..., NLEV])[:, None]           # (n, Q, T, R)
            arg = (Uj[..., None] - V[:, :, :, None, :] - f[:, None]) / sv5                # (n,Q,T,R,K1)
            lk = torch.special.log_ndtr(arg).masked_fill(mask, 0.0)
            return torch.logsumexp(lk.sum(4), 3) - math.log(self.R)
        # G: per-task GHK on the differences U_k - U_chosen < 0
        L = self._chol(sa, s, per_row)                                       # (n?, K1, J, J) by chosen
        if per_row:
            Ly = torch.stack([L[i][y[i]] for i in range(n)])                 # (n, T, J, J)
        else:
            Ly = L[y]                                                        # (n, T, J, J)
        K1 = self.K1
        oth = self._oth[y]                                                   # (n, T, J)
        Vo = V.gather(3, oth[:, None].expand(n, self.Q, self.T, K1 - 1))
        Vj = V.gather(3, y[:, None, :, None].expand(n, self.Q, self.T, 1))
        ub = -(Vo - Vj)                                                      # upper bounds (n, Q, T, J)
        w = self.zz[lo:hi]                                                   # (n, T, R, J-1)
        return _ghk(ub, Ly, w) - math.log(self.R)

    def _chol(self, sa, s, per_row):
        import torch

        K1 = self.K1
        if not hasattr(self, "_D"):
            D = torch.zeros(K1, K1 - 1, K1, device=self.dev)
            oth = []
            for j in range(K1):
                o = [k for k in range(K1) if k != j]
                oth.append(o)
                D[j, torch.arange(K1 - 1), torch.tensor(o)] = 1
                D[j, :, j] = -1
            self._D, self._oth = D, torch.tensor(oth, device=self.dev)

        def one(sa_, s_):
            Sig = (self.A * sa_ ** 2) @ self.A.T + torch.diag(s_ ** 2)
            return torch.linalg.cholesky(self._D @ Sig @ self._D.transpose(1, 2))

        if per_row:
            return torch.stack([one(a, b) for a, b in zip(sa, s)])
        return one(sa, s)


def _ghk(ub, L, w):
    """log GHK orthant probabilities summed over R draws (logsumexp): ub (n, Q, T, J) upper bounds,
    L (n, T, J, J) Cholesky, w (n, T, R, J-1) uniforms -> (n, Q, T)."""
    import torch
    from torch.special import log_ndtr, ndtri

    n, Q, T, J = ub.shape
    R = w.shape[2]
    tiny, one = 1e-30, 1 - 1e-7
    logp = ub.new_zeros(n, Q, T, R)
    etas = []
    for i in range(J):
        c = ub[..., i, None].expand(n, Q, T, R)
        if i:
            E = torch.stack(etas, -1)                                       # (n, Q, T, R, i)
            c = c - (E * L[:, None, :, None, i, :i]).sum(-1)
        lp = log_ndtr(c / L[:, None, :, None, i, i])
        logp = logp + lp
        if i < J - 1:
            etas.append(ndtri((w[:, None, :, :, i] * lp.exp()).clamp(tiny, one)))
    return torch.logsumexp(logp, 3)


def _fit(X, y, A, Q, R, method, th0, th_true, seed, chunk, se, max_iter, gh=0, progress=None):
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f64 = torch.float64
    t_setup = time.time()
    lik = Lik(X, y, A, Q, R, method, seed, dev, gh)
    N = lik.N
    t_setup = time.time() - t_setup

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    theta = torch.as_tensor(th0, dtype=f64, device=dev).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-6,
                            tolerance_change=1e-10, line_search_fn="strong_wolfe")
    nev = [0]

    def closure():
        opt.zero_grad()
        tot = 0.0
        for lo in range(0, N, chunk):
            l = -lik.ll_rows(theta, lo, min(lo + chunk, N)).sum() / N
            l.backward()
            tot += l.item()
        nev[0] += 1
        if progress is not None:
            try:
                PROGRESS[progress] = dict(stage="fit", evals=nev[0], nll=tot * N, elapsed=time.time() - t0,
                                          theta=theta.detach().cpu().numpy().tolist())
            except Exception:
                pass
        return torch.tensor(tot, dtype=f64)

    sync()
    t0 = time.time()
    opt.step(closure)
    final = closure().item() * N
    sync()
    t_fit = time.time() - t0
    grad_max = float(theta.grad.abs().max() * N)
    n_iter = int(opt.state[opt._params[0]]["n_iter"])
    th = theta.detach()
    with torch.no_grad():
        tt = torch.as_tensor(th_true, dtype=f64, device=dev)
        nll_true = -sum(float(lik.ll_rows(tt, lo, min(lo + chunk, N)).sum()) for lo in range(0, N, chunk))
    out = dict(theta=th.cpu().numpy().tolist(), nll=final, nll_true=nll_true, n_iter=n_iter, n_evals=nev[0],
               grad_max=grad_max, method=method, Q=Q, R=R, gh=gh, seed=seed, max_iter=max_iter,
               seconds=dict(setup=t_setup, fit=t_fit, per_eval=t_fit / max(nev[0], 1)),
               device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu")
    if progress is not None:
        PROGRESS[progress + ":fit"] = dict(out)
        PROGRESS[progress] = dict(stage="se", elapsed=time.time() - t0)
    if not se:
        return out
    t0 = time.time()
    nth = th.numel()
    H = torch.zeros(nth, nth, dtype=f64, device=dev)
    Sc = torch.zeros(N, nth, dtype=f64, device=dev)
    sch = max(1, chunk // 2)
    for lo in range(0, N, sch):
        hi = min(lo + sch, N)
        thr = th.expand(hi - lo, -1).clone().requires_grad_(True)
        ll = lik.ll_rows(thr, lo, hi)
        g = torch.autograd.grad(ll.sum(), thr, create_graph=True)[0]
        Sc[lo:hi] = g.detach()
        gs = g.sum(0)
        for i in range(nth):
            H[i] += torch.autograd.grad(gs[i], thr, retain_graph=i < nth - 1)[0].sum(0).detach()
    H = (H + H.T) / 2
    Hi = torch.linalg.inv(-H)
    V = Hi @ (Sc.T @ Sc) @ Hi
    sync()
    out.update(se_sandwich=V.diagonal().clamp_min(0).sqrt().cpu().numpy().tolist(),
               se_hessian=Hi.diagonal().clamp_min(0).sqrt().cpu().numpy().tolist(),
               cov_sandwich=V.cpu().numpy().tolist())
    out["seconds"]["se"] = time.time() - t0
    if progress is not None:
        PROGRESS[progress + ":done"] = out
        PROGRESS[progress] = dict(stage="done")
    return out


def _time_eval(X, y, A, Q, R, method, th, seed, chunk, n_rep=2):
    """Seconds per likelihood + gradient evaluation (full sample), best of n_rep after a warm-up."""
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lik = Lik(X, y, A, Q, R, method, seed, dev)
    theta = torch.as_tensor(th, dtype=torch.float64, device=dev).clone().requires_grad_(True)
    ts = []
    for r in range(n_rep + 1):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for lo in range(0, lik.N, chunk):
            (-lik.ll_rows(theta, lo, min(lo + chunk, lik.N)).sum()).backward()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return dict(method=method, J=lik.K1 - 1, Q=Q, R=R, chunk=chunk, seconds_per_eval=min(ts[1:]),
                device=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu")


@app.function(image=image, gpu="L4", timeout=24 * 3600)
def fit(X, y, A, Q, R, method, th0, th_true, seed, chunk=8, se=True, max_iter=300, gh=0, progress=None):
    return _fit(X, y, A, Q, R, method, th0, th_true, seed, chunk, se, max_iter, gh, progress)


@app.function(image=image, gpu="L4", timeout=3600)
def time_eval(X, y, A, Q, R, method, th, seed, chunk=8):
    return _time_eval(X, y, A, Q, R, method, th, seed, chunk)
