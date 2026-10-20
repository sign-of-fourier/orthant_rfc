"""T24 GPU fits on Modal: panel mixed logit (MXL) and panel MVP (mixed MNP with crossed error components).

    python analysis/T24/t24_modal.py smoke      # 1,000 training users, 500 holdout users, R = 256
    python analysis/T24/t24_modal.py full       # all training users; holdout log P per choice and per user

Both models: utility = X (b + w * z), z ~ N(0, I) on every feature except the position controls. A user's taste
draw z is shared by all their choices, so persistent taste and within-choice error are separately identified (the
panel is what T23's single-search data lacked). The two models use the same R draws per user and differ only in
the error given the tastes:
  MXL: iid Gumbel (logit kernel).
  MVP: normal, a component per band (hotel type, star level, rating level, price tertile in the shelf, amenity
       cluster) shared by the shelf's hotels at the same level, plus iid; total variance pi^2 / 6, shares free.
       The click is one orthant of the 24 differences clicked - other, by GHK with one path per taste draw.
Likelihood per user: mean over r of prod over the user's choices of P(choice | z_r).
"""
import json
import math
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy", "scipy")
app = modal.App("orthant-rfc-t24")
T_VAR = math.pi ** 2 / 6
PAD_MU = 30.0


def _err_shares(u):
    import torch
    p = torch.softmax(torch.cat([u, torch.zeros_like(u[:1])]), 0)
    return T_VAR * (p * (1 - 1e-5) + 1e-5 / len(p))


def _chunks(user, size):
    """Choice ranges [lo, hi) of about `size` choices that never split a user."""
    starts = np.r_[0, np.flatnonzero(np.diff(user)) + 1, len(user)]
    out, lo = [], 0
    for s in starts[1:]:
        if s - lo >= size or s == len(user):
            out.append((lo, int(s)))
            lo = int(s)
    return out


class Data:
    def __init__(self, d, R, seed, dev):
        import torch
        self.dev, self.R, self.seed = dev, R, seed
        self.X = torch.as_tensor(d["X"].astype(np.float64), device=dev)
        self.mask = torch.as_tensor(d["mask"], device=dev)
        self.band = torch.as_tensor(d["band"].astype(np.int64), device=dev)
        self.y = torch.as_tensor(d["y"].astype(np.int64), device=dev)
        self.user = torch.as_tensor(d["user"].astype(np.int64), device=dev)
        self.rnd = torch.as_tensor(d["rnd"], device=dev)
        nu, nr = int(d["user"].max()) + 1, len(d["rnd"])
        g = torch.Generator(device=dev).manual_seed(seed)
        self.sob = torch.quasirandom.SobolEngine(nr, scramble=True, seed=seed).draw(R, dtype=torch.float64).to(dev)
        self.shift = torch.rand(nu, 1, nr, generator=g, dtype=torch.float64, device=dev)

    def z(self, lo, hi):
        """Taste normals per choice (hi - lo, R, nr): the user's draws, shared by the user's choices."""
        import torch
        from torch.special import ndtri
        u = torch.remainder(self.sob[None] + self.shift[self.user[lo:hi]], 1.0)
        return ndtri(u.clamp(1e-12, 1 - 1e-12))

    def w(self, lo, hi, K):
        import torch
        g = torch.Generator(device=self.dev).manual_seed(self.seed * 1_000_003 + lo)
        return torch.rand(hi - lo, self.R, K, generator=g, dtype=torch.float64, device=self.dev)


def choice_logp(model, th, d, lo, hi, P, A):
    """log P(choice | z_r) per choice and draw, (hi - lo, R)."""
    import torch
    from torch.special import log_ndtr, ndtri
    X, mask, y = d.X[lo:hi], d.mask[lo:hi], d.y[lo:hi]
    nr = len(d.rnd)
    b = th[:P]
    sdz = d.z(lo, hi) * th[P:P + nr].exp()                                                # (n, R, nr)
    V = (X @ b)[:, None, :] + torch.einsum("njr,nmr->nmj", X[:, :, d.rnd], sdz)          # (n, R, J)
    if model == "mxl":
        U = V.masked_fill(~mask[:, None, :], -torch.inf)
        return U.log_softmax(-1).gather(2, y[:, None, None].expand(-1, d.R, 1))[..., 0]
    n, J = mask.shape
    ar = torch.arange(J, device=th.device)
    others = torch.stack([torch.cat([ar[:c], ar[c + 1:]]) for c in y.tolist()])           # (n, J-1)
    D = torch.zeros(n, J - 1, J, dtype=th.dtype, device=th.device)
    D.scatter_(2, others[..., None], -1.0)
    D[torch.arange(n), :, y] = 1.0
    real = torch.gather(mask, 1, others)
    D = D * real[..., None]
    band = d.band[lo:hi]
    mm = mask[:, :, None] & mask[:, None, :]
    s2 = _err_shares(th[P + nr:])
    Sig = sum(s2[a] * ((band[:, :, a, None] == band[:, None, :, a]) & mm).to(th.dtype) for a in range(A))
    Sig = Sig + s2[A] * torch.diag_embed(mask.to(th.dtype))
    pad = (~real).to(th.dtype)
    C = D @ Sig @ D.transpose(1, 2) + torch.diag_embed(pad)                               # padded: N(0, 1)
    L = torch.linalg.cholesky(C + 1e-9 * torch.eye(J - 1, dtype=th.dtype, device=th.device))
    mu = torch.einsum("nkj,nmj->nmk", D, V) + PAD_MU * pad[:, None, :]                    # (n, R, K)
    K = J - 1
    w = d.w(lo, hi, K - 1)
    logp = mu.new_zeros(n, d.R)
    us = []
    tiny, one = torch.finfo(th.dtype).tiny, 1 - torch.finfo(th.dtype).eps
    for j in range(K):
        c = mu[:, :, j]
        for k in range(j):
            c = c + L[:, j, k, None] * us[k]
        lp = log_ndtr(c / L[:, j, j, None])
        logp = logp + lp
        if j < K - 1:
            us.append(-ndtri((w[:, :, j] * lp.exp()).clamp(tiny, one)))
    return logp


def user_logp(lpc, user_local, nu):
    """Panel log P per user from per-choice-per-draw log P."""
    import torch
    s = lpc.new_zeros(nu, lpc.shape[1]).index_add_(0, user_local, lpc)
    return torch.logsumexp(s, 1) - math.log(lpc.shape[1])


def _fit(model, tr, ho, th0, R, seed, chunk, max_iter=300):
    """model 'mvp' or 'mxl'. tr / ho: dicts of numpy arrays (X, mask, band, y, user, rnd). Returns theta, nll, time,
    and for the holdout users log P per choice (population, marginal over tastes) and per user (panel)."""
    import torch
    dev = torch.device("cuda")
    P, A = tr["X"].shape[2], tr["band"].shape[2]
    d = Data(tr, R, seed, dev)
    ch = _chunks(tr["user"], chunk)
    n = len(tr["y"])
    theta = torch.as_tensor(th0, dtype=torch.float64, device=dev).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-6,
                            tolerance_change=1e-10, line_search_fn="strong_wolfe")
    nev = [0]

    def closure():
        opt.zero_grad()
        tot = 0.0
        for lo, hi in ch:
            ul = d.user[lo:hi] - d.user[lo]
            l = -user_logp(choice_logp(model, theta, d, lo, hi, P, A), ul, int(ul[-1]) + 1).sum() / n
            l.backward()
            tot += l.item()
        nev[0] += 1
        return torch.tensor(tot)
    t0 = time.time()
    opt.step(closure)
    nll = closure().item() * n
    t_fit = time.time() - t0
    out = dict(model=model, theta=theta.detach().cpu().numpy().tolist(), nll=nll, n=n, R=R, n_evals=nev[0],
               n_iter=int(opt.state[opt._params[0]]["n_iter"]), grad_max=float(theta.grad.abs().max() * n),
               seconds_fit=t_fit, gpu=torch.cuda.get_device_name())
    if ho is not None:
        t0 = time.time()
        dh = Data(ho, 4 * R, seed + 7, dev)
        lc, lu = [], []
        with torch.no_grad():
            for lo, hi in _chunks(ho["user"], max(1, chunk // 4)):
                lpc = choice_logp(model, theta.detach(), dh, lo, hi, P, A)
                ul = dh.user[lo:hi] - dh.user[lo]
                lc.append(torch.logsumexp(lpc, 1) - math.log(lpc.shape[1]))
                lu.append(user_logp(lpc, ul, int(ul[-1]) + 1))
        out["holdout_logp_choice"] = torch.cat(lc).cpu().numpy().tolist()
        out["holdout_logp_user"] = torch.cat(lu).cpu().numpy().tolist()
        out["seconds_holdout"] = time.time() - t0
    return out


@app.function(image=image, gpu="L4", timeout=3 * 3600, retries=0)
def fit(*a):
    return _fit(*a)


@app.function(image=image, gpu="A100-40GB", timeout=3 * 3600, retries=0)
def fit_a100(*a):                                                 # L4 fp64 is slow: MVP full runs here
    return _fit(*a)


def arrays(split, n_users=None, seed=24):
    sys.path.insert(0, str(HERE))
    import t24
    A = t24.load()
    idx = np.where(A["split"] == split)[0]
    users = np.unique(A["user"][idx])
    if n_users is not None:
        users = np.sort(np.random.default_rng(seed).choice(users, n_users, replace=False))
        idx = idx[np.isin(A["user"][idx], users)]
    feat = A["feat"].tolist()
    rnd = np.array([k for k, f in enumerate(feat) if f not in t24.FIXED])
    out = {k: A[k][idx] for k in ("X", "mask", "band", "y")}
    out["user"] = np.unique(A["user"][idx], return_inverse=True)[1]
    return out | {"rnd": rnd}, t24


def main():
    mode = sys.argv[1]
    max_iter = 300 if mode == "smoke" else 600
    tr, t24 = arrays(0, 1000 if mode == "smoke" else None)
    ho, _ = arrays(1, 500 if mode == "smoke" else None)
    loc = json.loads((t24.OUT / "local.json").read_text())
    nr, A = len(tr["rnd"]), tr["band"].shape[2]
    th_mxl = np.r_[loc["mnl"], np.full(nr, np.log(.3))]
    th_mvp = np.r_[th_mxl, np.zeros(A)]
    R = 256 if mode == "smoke" else 512
    print(f"{mode}: train {len(tr['y'])} choices / {tr['user'].max() + 1} users, holdout {len(ho['y'])} / "
          f"{ho['user'].max() + 1}, R {R}", flush=True)
    with app.run():
        calls = [fn.spawn(m, tr, ho, th, R, 2401, c, max_iter)
                 for fn, m, th, c in ((fit, "mxl", th_mxl, 4096),
                                      (fit if mode == "smoke" else fit_a100, "mvp", th_mvp, 256))]
        for call in calls:
            try:
                r = call.get()
            except Exception as e:
                print(f"FAILED: {e!r}"[:800], flush=True)
                continue
            (t24.OUT / f"{r['model']}_{mode}.json").write_text(json.dumps(r))
            print(f"{r['model']}: nll {r['nll']:.1f} (n {r['n']}), {r['n_iter']} iters / {r['n_evals']} evals, "
                  f"fit {r['seconds_fit']:.0f} s, holdout {r.get('seconds_holdout', 0):.0f} s, "
                  f"grad {r['grad_max']:.2e}, holdout log P/choice {np.mean(r['holdout_logp_choice']):.4f}, "
                  f"{r['gpu']}", flush=True)


if __name__ == "__main__":
    main()
