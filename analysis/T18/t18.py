"""T18: one panel orthant per respondent vs the nested simulated likelihood (TEST.md).

Every arm is GHK with fixed scrambled-Sobol draws and evaluates the same integral P(D > 0),
D = stacked utility differences (chosen - other), so arm differences are simulation error only.
  panel   one orthant of dimension d = T K per respondent; tastes integrated analytically
          (T1's ghk_gpu.py likelihood); M draws, one random shift per respondent; variable order
          natural (task by task) or "genz" (most-constrained first, fixed at theta0)
  nested  Q taste draws x per-task K-dim GHK with R draws (T4's MVP-G), generic in (Xd, G)
  ref     panel at 2^18 draws per shift; nested at 4096 x 2048 as the cross-check
Theta is T4's / mvp_refit's: b, log sd, u (variance shares by softmax, total error variance fixed).

    python analysis/T18/t18.py smoke          # Step 0: 20 respondents, A-36 and A-240, one H100
    python analysis/T18/t18.py report smoke   # local summary of a saved step (numpy only)
    python analysis/T18/t18.py refprobe       # Step 0b: Genz-ordered reference convergence at d = 240
    python analysis/T18/t18.py report refprobe
    python analysis/T18/t18.py a1             # A1 on 7 H100s in parallel (4 reference, 3 arms)
    python analysis/T18/t18.py report a1
    python analysis/T18/t18.py a2             # A2: panel genz 1k and 16k fits (2 H100s); nested fit reused
    python analysis/T18/t18.py report a2

Runs as an ephemeral Modal app. Torch is imported only inside Modal functions (4 GB local box).
"""
import json
import math
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")
app = modal.App("orthant-rfc-t18")

REF_M = 2 ** 18
NESTED = (512, 256)
NESTED_REF = (4096, 2048)
BLOCK = 16


# ---------------------------------------------------------------------------
# Data (local, numpy)
# ---------------------------------------------------------------------------
def _paths():
    for d in ("T1", "T4"):
        p = str(HERE.parent / d)
        if p not in sys.path:
            sys.path.insert(0, p)


def data_a240(N=1000):
    """T4 gate dataset: S1, J = 20, rep 0 -> d = 12 x 20 = 240."""
    return data_j(20, N)


def data_j(J, N=1000):
    """T4 S1 dataset at shelf size J, rep 0 -> d = 12 x J. Cached in cache/ (J = 60 takes ~20 min to build here)."""
    f = HERE / "cache" / f"S1_J{J}_N{N}.npz"
    if f.exists():
        z = np.load(f)
        return dict(name=f"A{12 * J}", Xd=z["Xd"], G=z["G"], th_true=z["th_true"], tvar=float(z["tvar"]),
                    names=list(z["names"]))
    _paths()
    import ghk_gpu
    import t4

    T = 12
    d = t4.make_data("S1", J, N, T, 0)
    lv = d["levels"]
    prods = np.concatenate([np.broadcast_to(lv, (N, T, J, 3)), d["prices"][..., None]], -1)
    Xs, Gs = [], []
    for lo in range(0, N, 50):                                   # chunked: G is (4, N, T, J, J), large at J = 60
        Xc, Gc = ghk_gpu.panel_arrays(prods[lo:lo + 50], d["y"][lo:lo + 50], lambda pr: t4.task_X(lv, pr[..., 3]),
                                      levels=t4.LEVELS)
        Xs.append(Xc.astype(np.float32))
        Gs.append(Gc.astype(np.int8))
    Xd, G = np.concatenate(Xs), np.concatenate(Gs, 1)
    sh = t4.TRUTHS["S1"]
    th_true = np.r_[t4.b_true(J), np.log(t4.W_TRUE), np.log(sh[:3] / sh[3])]
    f.parent.mkdir(exist_ok=True)
    np.savez(f, Xd=Xd, G=G, th_true=th_true, tvar=float(t4.T_VAR), names=np.array(_names(t4.NAMES)))
    return dict(name=f"A{T * J}", Xd=Xd, G=G, th_true=th_true, tvar=float(t4.T_VAR), names=_names(t4.NAMES))


def data_a36(N=1000):
    """T1 G1 recovery dataset (run.recovery_check seed) -> d = 12 x 3 = 36."""
    _paths()
    import ghk_gpu
    import run

    sh = run.TRUTHS["G1"]
    d = run.make_data(np.random.default_rng([run.SEED, 999]), N, 12, np.sqrt(run.T_VAR * sh))
    Xd, G = ghk_gpu.panel_arrays(d["prods"], d["y"], run.task_X)
    th_true = np.r_[run.B_TRUE, np.log(run.W_TRUE), np.log(sh[:3] / sh[3])]
    return dict(name="A36", Xd=Xd, G=G, th_true=th_true, tvar=float(run.T_VAR), names=_names(run.NAMES))


def _names(b):
    return list(b) + [f"logsd_{n}" for n in b] + ["u_brand", "u_flavor", "u_pack"]


def theta0(P):
    """The T4 gate's starting value (gate.py)."""
    return np.r_[np.zeros(P), np.log(np.full(P, 0.5)), np.zeros(3)]


def pack(ds, rows=None):
    sl = slice(None) if rows is None else slice(0, rows)
    return dict(name=ds["name"], Xd=ds["Xd"][sl].astype(np.float32), G=ds["G"][:, sl].astype(np.int8),
                th_true=ds["th_true"], tvar=ds["tvar"], names=ds["names"])


# ---------------------------------------------------------------------------
# GHK (torch, inside Modal)
# ---------------------------------------------------------------------------
def ghk(c0, L, w):
    """log GHK weights for P(c0 + L z > 0), T1's recursion (u_k = -Phi^-1(w_k Phi(c_k / L_kk))).
    c0 (*S, 1, d) offsets, L (*Lpre, d, d) lower Cholesky with Lpre broadcastable to S, w (*Sw, M, d - 1)
    uniforms with Sw broadcastable to S. Returns (*S', M). Dims are done in blocks of BLOCK: inside a block
    each new u_k updates the block's remaining offsets (rank-1, shrinking); between blocks one matmul
    updates the remaining rows' offsets, so autograd keeps O(M d) per batch element instead of O(M d^2)."""
    import torch
    from torch.special import log_ndtr, ndtri

    d = c0.shape[-1]
    tiny, one = torch.finfo(w.dtype).tiny, 1.0 - torch.finfo(w.dtype).eps
    off = c0                                                    # offsets for rows b0 .. d - 1
    logp = None
    for b0 in range(0, d, BLOCK):
        b1 = min(b0 + BLOCK, d)
        us = []
        ob = off[..., :b1 - b0]                                 # offsets for rows j .. b1 - 1
        for j in range(b0, b1):
            lp = log_ndtr(ob[..., 0] / L[..., j, j].unsqueeze(-1))
            logp = lp if logp is None else logp + lp
            if j < d - 1:
                u = -ndtri((w[..., j] * lp.exp()).clamp(tiny, one))
                us.append(u)
                if j + 1 < b1:
                    ob = ob[..., 1:] + u.unsqueeze(-1) * L[..., j + 1:b1, j].unsqueeze(-2)
        if b1 < d:
            U = torch.stack(us, -1)                             # (*S, M, b1 - b0)
            off = off[..., b1 - b0:] + U @ L[..., b1:, b0:b1].transpose(-1, -2)
    return logp


def sig2_of(u, tvar):
    import torch

    p = torch.softmax(torch.cat([u, torch.zeros_like(u[..., :1])], -1), -1)
    return tvar * (p * (1 - 4e-6) + 1e-6)


def panel_moments(thr, Xd, G, tvar):
    """thr (n, 2P+3) -> eta (n, d), cov (n, d, d) of the stacked differences."""
    import torch

    n, T, K, P = Xd.shape
    b, sd = thr[:, :P], thr[:, P:2 * P].exp()
    s2 = sig2_of(thr[:, 2 * P:], tvar)
    Z = Xd.reshape(n, T * K, P)
    eta = (Z * b[:, None]).sum(-1)
    cov = (Z * sd[:, None] ** 2) @ Z.transpose(1, 2)
    blk = torch.einsum("na,antij->ntij", s2, G)                  # (n, T, K, K)
    bd = torch.diag_embed(blk.permute(0, 2, 3, 1)).permute(0, 3, 1, 4, 2).reshape(n, T * K, T * K)
    return eta, cov + bd


def permute(eta, cov, perm):
    d = eta.shape[1]
    eta = eta.gather(1, perm)
    cov = cov.gather(1, perm[:, :, None].expand(-1, -1, d)).gather(2, perm[:, None, :].expand(-1, d, -1))
    return eta, cov


def genz_order(eta, cov):
    """Most-constrained-first order (Genz-Bretz style): at each step take the remaining variable with
    the smallest conditional P(D_j > 0) given the earlier ones at their truncated means. Float64."""
    import torch

    n, d = eta.shape
    m, S = eta.clone(), cov.clone()
    used = torch.zeros(n, d, dtype=torch.bool, device=eta.device)
    perm = torch.empty(n, d, dtype=torch.long, device=eta.device)
    ar = torch.arange(n, device=eta.device)
    norm = torch.distributions.Normal(0.0, 1.0)
    for i in range(d):
        a = m / S.diagonal(dim1=1, dim2=2).clamp_min(1e-300).sqrt()
        j = a.masked_fill(used, float("inf")).argmin(1)
        perm[:, i] = j
        used[ar, j] = True
        sjj = S[ar, j, j].clamp_min(1e-300)
        aj = a[ar, j]
        zeta = torch.exp(norm.log_prob(aj) - torch.special.log_ndtr(aj))     # E[z | z > -a_j]
        col = S[ar, :, j]                                                    # (n, d)
        m = m + col / sjj.sqrt()[:, None] * zeta[:, None]
        S = S - col[:, :, None] * col[:, None, :] / sjj[:, None, None]
        S[used] = 0
        S.transpose(1, 2)[used] = 0
        S[ar, j, j] = 1.0
    return perm


def nested_moments(thr, Xd, G, tvar, zb):
    """thr (n, nth), zb (n, Q, P) normal taste draws -> eta (n, Q, T, 1, K), L_t (n, 1, T, K, K)."""
    import torch

    n, T, K, P = Xd.shape
    b, sd = thr[:, :P], thr[:, P:2 * P].exp()
    s2 = sig2_of(thr[:, 2 * P:], tvar)
    beta = b[:, None] + sd[:, None] * zb
    eta = torch.einsum("ntkp,nqp->nqtk", Xd, beta)
    Lt = torch.linalg.cholesky(torch.einsum("na,antij->ntij", s2, G))
    return eta[..., None, :], Lt[:, None]


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------
def sobol(dim, n, seed):
    import torch

    return torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed).draw(n, dtype=torch.float64)


class SobolChunks:
    """Scrambled Sobol points [c0, c0 + n) of an M-point set, generated on demand (2^22 x 239 float64 is
    8 GB) and cached when the whole set is small."""

    def __init__(self, dim, M, seed, dev, dt, cache_max=2 ** 16):
        import torch

        self.n, self.dev, self.dt = M, dev, dt
        self.eng = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
        self.full = self.eng.draw(M, dtype=torch.float64).to(dev, dt) if M <= cache_max else None
        self.pos = 0 if self.full is None else M

    def get(self, c0, n):
        import torch

        if self.full is not None:
            return self.full[c0:c0 + n]
        if c0 != self.pos:                                       # sequential use costs no fast-forward
            self.eng.reset()
            self.eng.fast_forward(c0)
        n = min(n, self.n - c0)
        self.pos = c0 + n
        return self.eng.draw(n, dtype=torch.float64).to(self.dev, self.dt)


def shifts(N, dim, seed, extra=()):
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.rand((N,) + tuple(extra) + (dim,), generator=g, dtype=torch.float64)


def arm_seed(arm, shift):
    """Distinct scramble / shift seeds per arm kind and shift; references use their own stream."""
    base = {"panel": 1_000, "nested": 2_000, "ref_panel": 3_000, "ref_nested": 4_000}[arm]
    return 20261017 + base + 37 * shift


# ---------------------------------------------------------------------------
# Evaluation (inside Modal)
# ---------------------------------------------------------------------------
def _combine(parts, grad):
    """parts: list of (lse_c, lse2_c, M_c, g_c) over draw chunks -> ll, g, ess per row."""
    import torch

    lse = torch.stack([p[0] for p in parts], 0)
    lse2 = torch.stack([p[1] for p in parts], 0)
    tot = torch.logsumexp(lse, 0)
    Mtot = sum(p[2] for p in parts)
    ll = tot - math.log(Mtot)
    ess = torch.exp(2 * tot - torch.logsumexp(lse2, 0))
    g = None
    if grad:
        wts = torch.softmax(lse, 0)                              # (C, n)
        g = (wts[..., None] * torch.stack([p[3] for p in parts], 0)).sum(0)
    return ll, g, ess


def evaluate(spec, Xd, G, tvar, th, perm=None, rows=None, grad=True):
    """Per-respondent log-likelihood (and per-respondent gradient, ESS) for one arm.
    spec: kind (panel / nested / ref_panel / ref_nested), M or (Q, R), shift, dtype, order.
    Respondents in chunks, draws in chunks; OOM halves the respondent chunk."""
    import torch

    dev = Xd.device
    dt = torch.float64 if spec.get("dtype", "f32") == "f64" else torch.float32
    kind, sh = spec["kind"], spec.get("shift", 0)
    N0 = Xd.shape[0]
    idx = torch.arange(N0, device=dev) if rows is None else torch.as_tensor(rows, device=dev)
    N, T, K, P = Xd.shape[0], Xd.shape[1], Xd.shape[2], Xd.shape[3]
    d = T * K
    seed = arm_seed(kind, sh)
    ebytes = 8 if dt == torch.float64 else 4
    budget = spec.get("budget_gb", 40) * 2 ** 30
    if kind in ("panel", "ref_panel"):
        M = spec["M"]
        dch = min(M, spec.get("draw_chunk", 16384))
        base = SobolChunks(d - 1, M, seed, dev, dt)
        shf = shifts(N0, d - 1, seed + 1).to(dev, dt)
        per = dch * d * ebytes * 12
    else:
        Q, R = spec["Q"], spec["R"]
        dch = min(Q, spec.get("draw_chunk", 64))
        zb_all = torch.special.ndtri(torch.remainder(sobol(P, Q, seed)[None] + shifts(N0, P, seed + 1)[:, None], 1.0)
                                     .clamp(1e-12, 1 - 1e-12)).to(dev, dt)          # (N0, Q, P)
        base = sobol(K - 1, R, seed + 2).to(dev, dt)
        shf = shifts(N0, K - 1, seed + 3, extra=(T,)).to(dev, dt)                      # (N0, T, K-1)
        per = dch * T * R * K * ebytes * 12
    chunk = int(max(1, min(spec.get("rows_max", 256), budget // per)))
    lls, gs, ess = [], [], []
    lo = 0
    while lo < len(idx):
        hi = min(lo + chunk, len(idx))
        r = idx[lo:hi]
        try:
            ll, g, e = _eval_rows(kind, spec, Xd[r].to(dt), G[:, r].to(dt), tvar, th, r, perm, grad, dt, dch,
                                  base, shf, zb_all if kind in ("nested", "ref_nested") else None)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            continue
        lls.append(ll.double().cpu())
        ess.append(e.double().cpu())
        if grad:
            gs.append(g.double().cpu())
        lo = hi
    out = dict(ll=torch.cat(lls).numpy(), ess=torch.cat(ess).numpy(), chunk=chunk)
    if grad:
        out["g"] = torch.cat(gs).numpy()
    return out


def _eval_rows(kind, spec, Xd, G, tvar, th, r, perm, grad, dt, dch, base, shf, zb_all):
    import torch

    n = Xd.shape[0]
    parts = []
    thr = torch.as_tensor(th, dtype=dt, device=Xd.device).expand(n, -1).clone().requires_grad_(grad)
    if kind in ("panel", "ref_panel"):
        M = base.n
        for c0 in range(0, M, dch):
            w = torch.remainder(base.get(c0, dch)[None] + shf[r][:, None], 1.0)        # (n, Mc, d-1)
            with torch.set_grad_enabled(grad):
                eta, cov = panel_moments(thr, Xd, G, tvar)
                if perm is not None:
                    eta, cov = permute(eta, cov, perm[r])
                lp = ghk(eta[:, None, :], torch.linalg.cholesky(cov), w)              # (n, Mc)
                parts.append(_part(lp, thr, grad))
    else:
        Q = zb_all.shape[1]
        w = torch.remainder(base[None, None] + shf[r][:, :, None], 1.0)[:, None]      # (n, 1, T, R, K-1)
        for c0 in range(0, Q, dch):
            with torch.set_grad_enabled(grad):
                eta, Lt = nested_moments(thr, Xd, G, tvar, zb_all[r, c0:c0 + dch])
                lpr = ghk(eta, Lt, w)                                                 # (n, Qc, T, R)
                lpt = torch.logsumexp(lpr, -1) - math.log(lpr.shape[-1])             # (n, Qc, T)
                parts.append(_part(lpt.sum(-1), thr, grad))
    return _combine(parts, grad)


def _part(lp, thr, grad):
    import torch

    lse = torch.logsumexp(lp, 1)
    lse2 = torch.logsumexp(2 * lp.detach(), 1)
    g = torch.autograd.grad(lse.sum(), thr)[0].detach() if grad else None
    return lse.detach().double(), lse2.double(), lp.shape[1], None if g is None else g.double()


def smooth(spec, Xd, G, tvar, th, perm, rows, dirs, hs):
    """Values at th +- h e for each direction e and step h (same draws), plus the gradient at th.
    Returns ll0 (n,), g (n, nth), llp / llm (n_dir, n_h, n)."""
    base = evaluate(spec, Xd, G, tvar, th, perm, rows, grad=True)
    llp = np.zeros((len(dirs), len(hs), len(rows)))
    llm = np.zeros_like(llp)
    for i, e in enumerate(dirs):
        for k, h in enumerate(hs):
            llp[i, k] = evaluate(spec, Xd, G, tvar, th + h * e, perm, rows, grad=False)["ll"]
            llm[i, k] = evaluate(spec, Xd, G, tvar, th - h * e, perm, rows, grad=False)["ll"]
    return dict(ll0=base["ll"], g=base["g"], llp=llp, llm=llm)


def timed(spec, Xd, G, tvar, th, perm, n_rep=3):
    """Seconds for likelihood + gradient over all rows: warm-up then n_rep; peak memory."""
    import torch

    evaluate(spec, Xd, G, tvar, th, perm, rows=list(range(min(4, Xd.shape[0]))))
    ts = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(n_rep):
        torch.cuda.synchronize()
        t0 = time.time()
        r = evaluate(spec, Xd, G, tvar, th, perm)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return dict(seconds=ts, chunk=r["chunk"], peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)


def fit_lbfgs(spec, Xd, G, tvar, th_true, perm, max_iter=300, start=None):
    """L-BFGS on the summed simulated log-likelihood with fixed draws (shift 0), as T4's gate fit: theta0 from
    gate.py, strong-Wolfe line search, tolerances of mvp_gpu._fit. Gradient = sum of per-respondent gradients."""
    import torch

    N, P = Xd.shape[0], Xd.shape[3]
    theta = torch.tensor(theta0(P) if start is None else np.asarray(start), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-6,
                            tolerance_change=1e-10, line_search_fn="strong_wolfe")
    nev, trace = [0], []

    def closure():
        r = evaluate(spec, Xd, G, tvar, theta.detach().numpy(), perm)
        nll = -float(np.sum(r["ll"]))
        theta.grad = torch.as_tensor(-np.sum(r["g"], 0) / N, dtype=torch.float64)
        nev[0] += 1
        trace.append(nll)
        return torch.tensor(nll / N, dtype=torch.float64)

    t0 = time.time()
    opt.step(closure)
    nll = closure().item() * N
    secs = time.time() - t0
    nll_true = -float(np.sum(evaluate(spec, Xd, G, tvar, np.asarray(th_true), perm, grad=False)["ll"]))
    return dict(theta=theta.detach().numpy().tolist(), nll=nll, nll_true=nll_true, grad_max=float(theta.grad.abs().max() * N),
                n_iter=int(opt.state[opt._params[0]]["n_iter"]), n_evals=nev[0], seconds=secs, trace=trace)


@app.function(image=image, gpu="H100", timeout=4 * 3600, memory=32768)
def step(ds, tasks):
    """Run a list of tasks on one dataset; each task: dict(op=eval|smooth|time|order, spec, ...)."""
    import torch

    dev = torch.device("cuda")
    Xd = torch.as_tensor(ds["Xd"], dtype=torch.float64, device=dev)
    G = torch.as_tensor(ds["G"], dtype=torch.float64, device=dev)
    th, tvar = np.asarray(ds["th_true"], dtype=np.float64), ds["tvar"]
    P = Xd.shape[3]
    out = dict(gpu=torch.cuda.get_device_name(), name=ds["name"], n=Xd.shape[0], d=Xd.shape[1] * Xd.shape[2])
    t0 = time.time()
    with torch.no_grad():
        eta, cov = panel_moments(torch.as_tensor(theta0(P), device=dev).expand(Xd.shape[0], -1), Xd, G, tvar)
        perm = genz_order(eta, cov)
    out["order_seconds"] = time.time() - t0
    for t in tasks:
        key, spec = t["key"], t.get("spec", {})
        pm = perm if spec.get("order") == "genz" else None
        t0 = time.time()
        try:
            if t["op"] == "eval":
                r = evaluate(spec, Xd, G, tvar, th, pm, t.get("rows"), grad=t.get("grad", True))
            elif t["op"] == "smooth":
                r = smooth(spec, Xd, G, tvar, th, pm, t["rows"], np.asarray(t["dirs"]), t["hs"])
            elif t["op"] == "time":
                r = timed(spec, Xd, G, tvar, th, pm, t.get("n_rep", 3))
            elif t["op"] == "fit":
                r = fit_lbfgs(spec, Xd, G, tvar, th, pm, t.get("max_iter", 300), t.get("start"))
            r = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        except Exception as e:                                   # record and continue (smoke)
            r = dict(error=f"{type(e).__name__}: {e}")
            torch.cuda.empty_cache()
        r["wall"] = time.time() - t0
        r["spec"] = spec
        out[key] = r
        print(f"{ds['name']} {key}: {r['wall']:.1f} s" + (f"  {r['error']}" if "error" in r else ""), flush=True)
    return out


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
def directions(nth, n_rand=8, seed=18):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_rand, nth))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    U = np.zeros((3, nth))
    U[np.arange(3), nth - 3 + np.arange(3)] = 1
    return np.vstack([D, U])


HS = [1e-4, 3e-4, 1e-3, 3e-3]


def smoke_tasks(n, nth, full_time):
    rows = list(range(n))
    tasks = [dict(op="eval", key="ref_panel", spec=dict(kind="ref_panel", M=REF_M, dtype="f64"), rows=rows),
             dict(op="eval", key="ref_nested", spec=dict(kind="ref_nested", Q=NESTED_REF[0], R=NESTED_REF[1],
                                                       dtype="f64", draw_chunk=32), rows=rows)]
    for order in ("natural", "genz"):
        for M in (1024, 4096, 16384):
            tasks.append(dict(op="eval", key=f"panel_{order}_{M}", spec=dict(kind="panel", M=M, order=order), rows=rows))
    tasks += [dict(op="eval", key="panel_natural_4096_f64", spec=dict(kind="panel", M=4096, dtype="f64"), rows=rows),
              dict(op="eval", key="nested", spec=dict(kind="nested", Q=NESTED[0], R=NESTED[1]), rows=rows),
              dict(op="eval", key="nested_f64", spec=dict(kind="nested", Q=NESTED[0], R=NESTED[1], dtype="f64"), rows=rows)]
    dirs = directions(nth).tolist()
    tasks += [dict(op="smooth", key="smooth_panel_4096", spec=dict(kind="panel", M=4096, dtype="f64"), rows=rows[:10],
                   dirs=dirs, hs=HS),
              dict(op="smooth", key="smooth_nested", spec=dict(kind="nested", Q=NESTED[0], R=NESTED[1], dtype="f64"),
                   rows=rows[:10], dirs=dirs, hs=HS)]
    timing = [dict(op="time", key="time_panel_4096", spec=dict(kind="panel", M=4096)),
              dict(op="time", key="time_nested", spec=dict(kind="nested", Q=NESTED[0], R=NESTED[1]))]
    return timing if full_time else tasks


def run_smoke(n=20):
    """Step 0. The 20-respondent tasks run on a 20-row dataset; the timing tasks on the full N = 1,000
    (A-240 only), so the A1 quote is measured, not extrapolated."""
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for build in (data_a36, data_a240):
        ds = build()
        nth = len(ds["th_true"])
        jobs.append((pack(ds, n), smoke_tasks(n, nth, False)))
        if ds["name"] == "A240":
            jobs.append((pack(ds), smoke_tasks(n, nth, True)))
    print("datasets ready:", [(j[0]["name"], j[0]["Xd"].shape, len(j[1])) for j in jobs], flush=True)
    t0 = time.time()
    with modal.enable_output(), app.run():
        for i, (ds, tasks) in enumerate(jobs):                   # one H100, sequentially; save each
            r = step.remote(ds, tasks)
            r["names"], r["th_true"] = ds["names"], list(ds["th_true"])
            tag = ds["name"] + ("_time" if i == len(jobs) - 1 else "")
            (OUT / f"smoke_{tag}.json").write_text(json.dumps(r))
            print(f"saved smoke_{tag}.json ({time.time() - t0:.0f} s wall so far)", flush=True)


REF_MS = (2 ** 18, 2 ** 20, 2 ** 22)
N_SHIFT = 8


def run_refprobe(n=50):
    """Step 0b: does a Genz-ordered panel reference converge at d = 240? Values only (no gradients):
    panel genz at 2^18 / 2^20 / 2^22 x 8 shifts, nested 4096 x 2048 x 4 shifts, on the first n respondents."""
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pack(data_a240(), n)
    tasks = [dict(op="eval", key=f"ref_genz_{M}_s{s}", grad=False,
                  spec=dict(kind="ref_panel", M=M, order="genz", dtype="f64", shift=s))
             for M in REF_MS for s in range(N_SHIFT)]
    tasks += [dict(op="eval", key=f"ref_nested_s{s}", grad=False,
                   spec=dict(kind="ref_nested", Q=NESTED_REF[0], R=NESTED_REF[1], dtype="f64", draw_chunk=32, shift=s))
              for s in range(4)]
    t0 = time.time()
    with modal.enable_output(), app.run():
        r = step.remote(ds, tasks)
    r["names"], r["th_true"] = ds["names"], list(ds["th_true"])
    (OUT / "refprobe_A240.json").write_text(json.dumps(r))
    print(f"saved refprobe_A240.json ({time.time() - t0:.0f} s wall)", flush=True)


def _pool(L):
    """Shifts (S, n) of log P estimates -> pooled log mean, its SE (delta method on the P scale), per row."""
    L = np.asarray(L)
    S = L.shape[0]
    mx = L.max(0)
    P = np.exp(L - mx)
    m = P.mean(0)
    se = P.std(0, ddof=1) / np.sqrt(S) / m
    return np.log(m) + mx, se


def report_refprobe():
    r = json.loads((OUT / "refprobe_A240.json").read_text())
    errs = [k for k, v in r.items() if isinstance(v, dict) and "error" in v]
    if errs:
        print("errors:", {k: r[k]["error"] for k in errs})
    print(f"{r['name']} d = {r['d']}, n = {r['n']}, {r['gpu']}")
    pooled = {}
    for M in REF_MS:
        L = np.array([r[f"ref_genz_{M}_s{s}"]["ll"] for s in range(N_SHIFT)])
        E = np.array([r[f"ref_genz_{M}_s{s}"]["ess"] for s in range(N_SHIFT)])
        lp, se = _pool(L)
        pooled[M] = (lp, se, L)
        wall = sum(r[f"ref_genz_{M}_s{s}"]["wall"] for s in range(N_SHIFT))
        print(f"  genz M = 2^{int(np.log2(M))}: per-shift SD of log P median {np.median(L.std(0, ddof=1)):.4f} "
              f"p90 {np.quantile(L.std(0, ddof=1), .9):.4f}; pooled ({N_SHIFT} x M) SE median {np.median(se):.4f} "
              f"p90 {np.quantile(se, .9):.4f}; ESS median {np.median(E):.0f}; {wall:.0f} s")
    Ms = list(REF_MS)
    for a, b in zip(Ms, Ms[1:]):
        dd = pooled[b][0] - pooled[a][0]
        ci = 1.96 * dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"  pooled 2^{int(np.log2(b))} - 2^{int(np.log2(a))}: mean {dd.mean():+.4f} [{dd.mean() - ci:+.4f}, "
              f"{dd.mean() + ci:+.4f}], median |.| {np.median(np.abs(dd)):.4f}, max |.| {np.abs(dd).max():.4f}")
    Ln = np.array([r[f"ref_nested_s{s}"]["ll"] for s in range(4)])
    ln, sen = _pool(Ln)
    top = pooled[Ms[-1]][0]
    dd = ln - top
    ci = 1.96 * dd.std(ddof=1) / np.sqrt(len(dd))
    print(f"  nested 4096x2048 x4: pooled SE median {np.median(sen):.4f}; nested - genz 2^22: mean {dd.mean():+.4f} "
          f"[{dd.mean() - ci:+.4f}, {dd.mean() + ci:+.4f}], max |.| {np.abs(dd).max():.4f}")
    sm = json.loads((OUT / "smoke_A240.json").read_text())
    nat = np.array(sm["ref_panel"]["ll"])
    k = min(len(nat), len(top))
    d2 = nat[:k] - top[:k]
    print(f"  smoke natural-order 2^18 reference - genz 2^22 (first {k}): mean {d2.mean():+.4f}, "
          f"median |.| {np.median(np.abs(d2)):.4f}")


# ---------------------------------------------------------------------------
# A1 (Step 0b reference: Genz order, 2^20 x 8 shifts, with gradients)
# ---------------------------------------------------------------------------
A1_REF_M = 2 ** 20
A1_ARMS = [(o, M) for o in ("natural", "genz") for M in (1024, 4096, 16384)]
M_E1, M_E3 = 0.05, 0.25           # materiality: nats per respondent; SE
SPEED_M = 2.0


def a1_jobs(nth):
    """Container task lists: 4 reference containers (2 shifts each), panel natural, panel genz, nested +
    smoothness + timing (all timings on one GPU)."""
    jobs = []
    for c in range(4):
        jobs.append((f"ref{c}", [dict(op="eval", key=f"ref_s{s}", spec=dict(kind="ref_panel", M=A1_REF_M, order="genz",
                                                                       dtype="f64", shift=s)) for s in (2 * c, 2 * c + 1)]))
    for o in ("natural", "genz"):
        jobs.append((f"panel_{o}", [dict(op="eval", key=f"panel_{o}_{M}_s{s}", spec=dict(kind="panel", M=M, order=o, shift=s))
                                     for M in (1024, 4096, 16384) for s in range(N_SHIFT)]))
    rows, dirs = list(range(100)), directions(nth).tolist()
    t = [dict(op="eval", key=f"nested_s{s}", spec=dict(kind="nested", Q=NESTED[0], R=NESTED[1], shift=s)) for s in range(N_SHIFT)]
    t += [dict(op="smooth", key=f"smooth_panel_{o}_{M}", spec=dict(kind="panel", M=M, order=o, dtype="f64"), rows=rows,
               dirs=dirs, hs=HS) for o, M in A1_ARMS]
    t += [dict(op="smooth", key="smooth_nested", spec=dict(kind="nested", Q=NESTED[0], R=NESTED[1], dtype="f64"), rows=rows,
               dirs=dirs, hs=HS)]
    t += [dict(op="time", key=f"time_panel_{o}_{M}", spec=dict(kind="panel", M=M, order=o)) for o, M in A1_ARMS]
    t += [dict(op="time", key="time_nested", spec=dict(kind="nested", Q=NESTED[0], R=NESTED[1]))]
    jobs.append(("nested", t))
    return jobs


def run_a1():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pack(data_a240())
    jobs = a1_jobs(len(ds["th_true"]))
    print("A1 containers:", [(tag, len(t)) for tag, t in jobs], flush=True)
    t0 = time.time()
    with modal.enable_output(), app.run():
        calls = [(tag, step.spawn(ds, tasks)) for tag, tasks in jobs]
        for tag, c in calls:                                      # saved one by one
            r = c.get()
            r["names"], r["th_true"] = ds["names"], list(ds["th_true"])
            (OUT / f"a1_{tag}.json").write_text(json.dumps(r))
            print(f"saved a1_{tag}.json ({time.time() - t0:.0f} s wall)", flush=True)


def _holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def _label_noninf(d, sig, m):
    if not sig:
        return "PASS (no worse)"
    if d > 0:
        return "PASS (better)"
    return "SOFT FAIL (worse, immaterial)" if abs(d) < m else "HARD FAIL (worse, material)"


def _t_ci(x, alpha=.05):
    from scipy import stats

    x = np.asarray(x, float)
    n = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    tq = stats.t.ppf(1 - alpha / 2, n - 1)
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else 0.0
    return m, m - tq * se, m + tq * se, p


def report_a1():
    from scipy import stats

    R = {}
    for f in sorted(OUT.glob("a1_*.json")):
        R.update({k: v for k, v in json.loads(f.read_text()).items()})
    errs = {k: v["error"] for k, v in R.items() if isinstance(v, dict) and "error" in v}
    if errs:
        print("ERRORS:", errs)
    names = R["names"]
    nth = len(names)
    # reference: pooled over shifts (log mean), gradient of the log mean
    Lr = np.array([R[f"ref_s{s}"]["ll"] for s in range(N_SHIFT)])
    Gr = np.array([R[f"ref_s{s}"]["g"] for s in range(N_SHIFT)])
    ll_ref, se_ref = _pool(Lr)
    w = np.exp(Lr - Lr.max(0))
    w /= w.sum(0)
    g_ref = (w[..., None] * Gr).sum(0)                                      # (N, nth)
    N = len(ll_ref)
    H = g_ref.T @ g_ref                                                     # OPG
    Hi = np.linalg.inv(H)
    se = np.sqrt(np.diag(Hi))
    out = dict(N=N, ref=dict(M=A1_REF_M, shifts=N_SHIFT, se_median=float(np.median(se_ref)),
                             se_p90=float(np.quantile(se_ref, .9)), mean_ll=float(ll_ref.mean())))
    print(f"A1: d = 240, N = {N}. Reference genz 2^20 x {N_SHIFT}: mean log L {ll_ref.mean():.3f}, pooled SE median "
          f"{np.median(se_ref):.4f} (p90 {np.quantile(se_ref, .9):.4f})")
    gate = HERE.parent / "T4" / "out" / "gate_g" / "fit_512x256.json"
    if gate.exists():
        sg = np.array(json.loads(gate.read_text())["se_sandwich"])
        print(f"  OPG SE / gate sandwich SE: median {np.median(se / sg):.2f} (range {np.min(se / sg):.2f}-{np.max(se / sg):.2f})")

    arms = {f"panel_{o}_{M}": [f"panel_{o}_{M}_s{s}" for s in range(N_SHIFT)] for o, M in A1_ARMS}
    arms["nested"] = [f"nested_s{s}" for s in range(N_SHIFT)]
    A = {}
    for a, keys in arms.items():
        L = np.array([R[k]["ll"] for k in keys])                           # (S, N)
        G = np.array([R[k]["g"] for k in keys])                            # (S, N, nth)
        E = np.array([R[k]["ess"] for k in keys])
        err = L - ll_ref
        e_abs = np.abs(err).mean(0)                                         # (N,)
        tot = err.sum(1)                                                    # (S,)
        dth = np.einsum("pq,sq->sp", Hi, (G - g_ref).sum(1)) / se           # (S, nth) in SE
        A[a] = dict(e_abs=e_abs, err=err, tot=tot, dth=dth, stat=np.abs(dth).max(1), ess=E)
    print("\nE1 per-respondent |error| (nats; mean over 8 shifts), E2 total bias over N, E3 predicted shift (SE), E4 ESS")
    print(f"  {'arm':22s} {'E1 median':>9s} {'mean':>7s} {'signed':>8s}   {'E2 total [95% CI]':>26s}   "
          f"{'E3 max|shift| mean':>18s}  {'ESS/M':>8s}")
    for a, v in A.items():
        m2, lo2, hi2, _ = _t_ci(v["tot"])
        Mx = int(a.split("_")[-1]) if a != "nested" else NESTED[0]
        print(f"  {a:22s} {np.median(v['e_abs']):9.4f} {v['e_abs'].mean():7.4f} {v['err'].mean():+8.4f}   "
              f"{m2:+8.2f} [{lo2:+8.2f}, {hi2:+8.2f}]   {v['stat'].mean():18.3f}  {np.median(v['ess']) / Mx:8.2e}")
    # E1 / E3 tests: panel arm vs nested, Holm over the 6 panel arms
    nest = A["nested"]
    rows = []
    for o, M in A1_ARMS:
        a = f"panel_{o}_{M}"
        d1 = nest["e_abs"] - A[a]["e_abs"]
        m1, lo1, hi1, p1 = _t_ci(d1)
        x, y = nest["stat"], A[a]["stat"]
        t3 = stats.ttest_ind(x, y, equal_var=False)
        m3 = x.mean() - y.mean()
        vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if vx + vy > 0 else 1
        hw = stats.t.ppf(.975, df) * np.sqrt(vx + vy)
        rows.append(dict(arm=a, o=o, M=M, d1=m1, ci1=(lo1, hi1), p1=p1, d3=m3, ci3=(m3 - hw, m3 + hw), p3=float(t3.pvalue)))
    for key in ("p1", "p3"):
        for r_, ph in zip(rows, _holm([r_[key] for r_ in rows])):
            r_[key + "h"] = ph
    print("\nNon-inferiority vs nested 512 x 256 (Δ = nested − panel error, + = panel better; Holm over 6 arms)")
    for r_ in rows:
        r_["lab1"] = _label_noninf(r_["d1"], r_["p1h"] < .05, M_E1)
        r_["lab3"] = _label_noninf(r_["d3"], r_["p3h"] < .05, M_E3)
        print(f"  {r_['arm']:22s} E1 Δ {r_['d1']:+.4f} [{r_['ci1'][0]:+.4f}, {r_['ci1'][1]:+.4f}] Holm p {r_['p1h']:.2g} "
              f"{r_['lab1']:30s} | E3 Δ {r_['d3']:+.3f} [{r_['ci3'][0]:+.3f}, {r_['ci3'][1]:+.3f}] Holm p {r_['p3h']:.2g} {r_['lab3']}")
    # E3 detail: mean shift on the variance-share parameters
    print("\nE3 mean predicted shift on the variance shares (SE units, mean over shifts [95% CI])")
    for a, v in A.items():
        parts = []
        for j in range(nth - 3, nth):
            m, lo, hi, _ = _t_ci(v["dth"][:, j])
            parts.append(f"{names[j]} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {a:22s} " + "; ".join(parts))
    # E4: bias trend
    print("\nE4 signed-error trend (bias should shrink ~4x per 4x M if 1/M holds)")
    for o in ("natural", "genz"):
        b = [A[f"panel_{o}_{M}"]["err"].mean() for M in (1024, 4096, 16384)]
        print(f"  {o:8s} mean signed error 1k / 4k / 16k: {b[0]:+.4f} / {b[1]:+.4f} / {b[2]:+.4f}")
    # E5 guard
    print("\nE5 smoothness guard (float64; slope >= 1.5 and grad gap <= 1e-3)")
    guard_ok = True
    for k, v in R.items():
        if k.startswith("smooth") and isinstance(v, dict) and "ll0" in v:
            ll0, g = np.array(v["ll0"]), np.array(v["g"])
            dirs = directions(g.shape[1])
            llp, llm = np.array(v["llp"]), np.array(v["llm"])
            res = np.abs(llp - ll0 - np.array(HS)[None, :, None] * (dirs @ g.T)[:, None, :])
            slope = np.polyfit(np.log(HS), np.log(np.median(res, axis=(0, 2)) + 1e-300), 1)[0]
            fd = (llp[:, 0] - llm[:, 0]) / (2 * HS[0])
            gap = np.abs(fd - dirs @ g.T).max() / np.abs(dirs @ g.T).max()
            ok = slope >= 1.5 and gap <= 1e-3
            guard_ok &= ok
            print(f"  {k:24s} slope {slope:.2f}, grad gap {gap:.1e}, max |dlogL| at 3e-3 {np.abs(llp[:, -1] - ll0).max():.4f} "
                  f"{'ok' if ok else 'FAIL'}")
    # E6 speed
    print("\nE6 speed, N = 1,000, likelihood + gradient, float32, one H100 (speedup = nested / panel)")
    tn = np.log(R["time_nested"]["seconds"])
    print(f"  nested 512 x 256: {np.median(R['time_nested']['seconds']):.2f} s")
    for r_ in rows:
        tp = np.log(R[f"time_{r_['arm']}"]["seconds"])
        d = tn.mean() - tp.mean()
        sd = np.sqrt(tn.var(ddof=1) / len(tn) + tp.var(ddof=1) / len(tp))
        hw = stats.t.ppf(.975, len(tn) + len(tp) - 2) * sd
        lo, hi = np.exp(d - hw), np.exp(d + hw)
        sp = np.exp(d)
        lab = ("PASS" if sp >= SPEED_M else "SOFT") if lo > 1 else ("HARD FAIL" if hi < 1 else "INCONCLUSIVE")
        r_.update(sec=float(np.median(R[f"time_{r_['arm']}"]["seconds"])), speed=sp, speed_ci=(lo, hi), lab6=lab)
        print(f"  {r_['arm']:22s} {r_['sec']:6.2f} s, speedup {sp:5.1f}x [{lo:.1f}, {hi:.1f}] {lab}")
    # choice and overall
    ok = [r_ for r_ in rows if r_["lab1"].startswith("PASS") and r_["lab3"].startswith("PASS") and r_["lab6"] != "HARD FAIL"]
    print()
    if not guard_ok:
        print("GUARD FAIL (smoothness): stop.")
    if ok:
        best = min(ok, key=lambda r_: r_["sec"])
        overall = "PASS" if best["lab6"] in ("PASS", "SOFT") else "INCONCLUSIVE"
        print(f"A1 overall: {overall}. A2 arm (cheapest passing): {best['arm']} ({best['sec']:.2f} s, speed {best['lab6']})")
    else:
        print("A1 overall: HARD FAIL (no panel arm passes E1 and E3)")
    out["rows"] = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r_.items()} for r_ in rows]
    (OUT / "a1_summary.json").write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------
# A2: one fit per panel arm (genz 1k = the rule's arm; genz 16k added by the user), nested fit reused
# ---------------------------------------------------------------------------
A2_ARMS = [("genz", 1024), ("genz", 16384)]


def run_a2():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pack(data_a240())
    t0 = time.time()
    with modal.enable_output(), app.run():
        calls = [(f"genz_{M}", step.spawn(ds, [dict(op="fit", key="fit", spec=dict(kind="panel", M=M, order=o))]))
                 for o, M in A2_ARMS]
        for tag, c in calls:
            r = c.get()
            r["names"], r["th_true"] = ds["names"], list(ds["th_true"])
            (OUT / f"a2_{tag}.json").write_text(json.dumps(r))
            print(f"saved a2_{tag}.json ({time.time() - t0:.0f} s wall)", flush=True)


def report_a2():
    from scipy import stats

    gate = json.loads((HERE.parent / "T4" / "out" / "gate_g" / "fit_512x256.json").read_text())
    thn, V = np.array(gate["theta"]), np.array(gate["cov_sandwich"])
    se = np.sqrt(np.diag(V))
    Vi = np.linalg.inv(V)
    sw = slice(len(thn) - 3, len(thn))

    def shares(th):
        z = np.exp(np.r_[th[sw], 0.0])
        return z / z.sum()

    def recovery(th, tt):
        z = (th - tt) / se
        p = 2 * stats.norm.sf(np.abs(z))
        rej = set(np.where(_holm(p) < .05)[0])
        d = th[sw] - tt[sw]
        W = d @ np.linalg.inv(V[sw, sw]) @ d
        return z, rej, W, stats.chi2.sf(W, 3)

    fits = {}
    for f in sorted(OUT.glob("a2_*.json")):
        r = json.loads(f.read_text())
        if "error" in r["fit"]:
            print(f.name, "ERROR", r["fit"]["error"])
            continue
        fits[f.stem[3:]] = r
    names, tt = next(iter(fits.values()))["names"], np.array(next(iter(fits.values()))["th_true"])
    zn, rejn, Wn, pWn = recovery(thn, tt)
    print(f"truth shares {np.round(shares(tt), 3)}")
    print(f"nested 512x256 (2026-10-09, L4): fit {gate['seconds']['fit'] / 60:.0f} min, {gate['n_evals']} evals "
          f"(H100 est. {gate['n_evals'] * 12.26 / 60:.0f} min at A1's 12.26 s/eval); shares {np.round(shares(thn), 3)}; "
          f"Holm rejections {[names[i] for i in sorted(rejn)]}; shares Wald p {pWn:.3g}")
    out = {}
    for tag, r in fits.items():
        f = r["fit"]
        th = np.array(f["theta"])
        z, rej, W, pW = recovery(th, tt)
        d = th - thn
        Wa = d @ Vi @ d
        pa = stats.chi2.sf(Wa, len(d))
        agree = "PASS" if pa >= .05 else "FAIL"
        rec = "PASS" if rej <= rejn else "FAIL"
        overall = "PASS" if agree == "PASS" and rec == "PASS" else "FAIL"
        print(f"\npanel {tag}: fit {f['seconds'] / 60:.1f} min, {f['n_iter']} iter / {f['n_evals']} evals, grad max {f['grad_max']:.2e}, "
              f"nll {f['nll']:.1f} (truth {f['nll_true']:.1f}), {r['gpu']}")
        print(f"  shares {np.round(shares(th), 3)}")
        print(f"  agreement with nested: Wald {Wa:.1f} on {len(d)} df, p {pa:.3g}, max |dθ|/SE {np.abs(d / se).max():.2f} "
              f"({names[int(np.argmax(np.abs(d / se)))]}) -> {agree}")
        print(f"  recovery: Holm rejections {[names[i] for i in sorted(rej)]} (nested: {[names[i] for i in sorted(rejn)]}) -> {rec}; "
              f"shares Wald vs truth p {pW:.3g} (nested {pWn:.3g})")
        print(f"  z vs truth, shares: {np.round(z[sw], 2)} (nested {np.round(zn[sw], 2)}); max |z| {np.abs(z).max():.2f} "
              f"({names[int(np.argmax(np.abs(z)))]})")
        print(f"  A2: {overall}; speedup vs nested on H100 ~{gate['n_evals'] * 12.26 / f['seconds']:.0f}x (wall, est.)")
        out[tag] = dict(agree_W=Wa, agree_p=pa, rej=[names[i] for i in sorted(rej)], shares=shares(th).tolist(),
                        z=z.tolist(), overall=overall, minutes=f["seconds"] / 60)
    (OUT / "a2_summary.json").write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------
# Report (local, numpy)
# ---------------------------------------------------------------------------
JS_MS = (1024, 4096, 16384)
JS_REF = 2 ** 18
JS_ROWS = 100


def run_jsmoke():
    """T4 cost check: per-evaluation time (likelihood + gradient, N = 1,000) and accuracy (ESS, error vs a 2^18
    reference on 100 respondents) for the Genz-ordered panel at J = 20 / 40 / 60 (d = 240 / 480 / 720).
    One H100 per J, in parallel."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(range(JS_ROWS))
    tasks = [dict(op="eval", key="ref", grad=False, rows=rows, spec=dict(kind="ref_panel", M=JS_REF, order="genz", dtype="f64"))]
    tasks += [dict(op="eval", key=f"acc_{M}", grad=False, rows=rows, spec=dict(kind="panel", M=M, order="genz")) for M in JS_MS]
    tasks += [dict(op="time", key=f"time_{M}", n_rep=2, spec=dict(kind="panel", M=M, order="genz")) for M in JS_MS]
    t0 = time.time()
    with modal.enable_output(), app.run():
        calls = []
        for J in (20, 40, 60):
            ds = pack(data_j(J))
            calls.append((J, step.spawn(ds, tasks)))
        for J, c in calls:
            r = c.get()
            (OUT / f"jsmoke_{J}.json").write_text(json.dumps(r))
            print(f"saved jsmoke_{J}.json ({time.time() - t0:.0f} s wall)", flush=True)


def report_jsmoke():
    fits = {20: (39, 42)}                                       # A2 evals per fit at 1k / 16k (J = 20)
    for J in (20, 40, 60):
        f = OUT / f"jsmoke_{J}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        ref = np.array(r["ref"]["ll"])
        print(f"\nJ = {J}: d = {r['d']}, {r['gpu']}, Genz order {r['order_seconds']:.1f} s; ref ESS/M median "
              f"{np.median(r['ref']['ess']) / JS_REF:.3g}")
        for M in JS_MS:
            a, t = r[f"acc_{M}"], r[f"time_{M}"]
            if "error" in a or "error" in t:
                print(f"  M {M}: ERROR {a.get('error') or t.get('error')}")
                continue
            e = np.array(a["ll"]) - ref
            sec = float(np.median(t["seconds"]))
            print(f"  M {M:6d}: {sec:6.2f} s/eval (peak {t['peak_GiB']:.1f} GiB); ESS/M median {np.median(a['ess']) / M:.3g} "
                  f"(min {np.min(a['ess']):.0f}); |err| median {np.median(np.abs(e)):.4f} p90 {np.quantile(np.abs(e), .9):.4f}, "
                  f"signed mean {e.mean():+.4f} nats/resp; 40-eval fit ~{40 * sec / 60:.1f} min")


def run_warm():
    """Two-stage fit at J = 20: the A2 1k-draw estimate as the start of the 16k-draw fit (same objective as A2's
    cold 16k fit: same draws, Genz order fixed at theta0)."""
    start = json.loads((OUT / "a2_genz_1024.json").read_text())["fit"]["theta"]
    ds = pack(data_j(20))
    with modal.enable_output(), app.run():
        r = step.remote(ds, [dict(op="fit", key="fit", start=start, spec=dict(kind="panel", M=16384, order="genz"))])
    (OUT / "warm_20.json").write_text(json.dumps(r))


def report_warm():
    w = json.loads((OUT / "warm_20.json").read_text())["fit"]
    c = json.loads((OUT / "a2_genz_16384.json").read_text())["fit"]
    s = json.loads((OUT / "a2_genz_1024.json").read_text())["fit"]
    gate = json.loads((HERE.parent / "T4" / "out" / "gate_g" / "fit_512x256.json").read_text())
    se = np.sqrt(np.diag(np.asarray(gate["cov_sandwich"])))
    gap = np.abs(np.asarray(w["theta"]) - np.asarray(c["theta"])) / se
    print(f"cold 16k        : {c['n_evals']} evals, {c['seconds'] / 60:.2f} min, nll {c['nll']:.2f}")
    print(f"1k stage (A2)   : {s['n_evals']} evals, {s['seconds'] / 60:.2f} min")
    print(f"16k from 1k     : {w['n_evals']} evals, {w['seconds'] / 60:.2f} min, nll {w['nll']:.2f}; "
          f"two-stage total {(s['seconds'] + w['seconds']) / 60:.2f} min")
    print(f"warm vs cold optimum: max |gap|/SE {gap.max():.3f}, nll diff {w['nll'] - c['nll']:+.3f}")


def run_j60():
    """J = 60 (d = 720) cold fit, Genz order, 16k draws, as A2; then per-respondent scores at the estimate (OPG SEs)."""
    ds = pack(data_j(60))
    spec = dict(kind="panel", M=16384, order="genz")
    t0 = time.time()
    with modal.enable_output(), app.run():
        r = step.remote(ds, [dict(op="fit", key="fit", spec=spec)])
        (OUT / "j60_fit.json").write_text(json.dumps(r))
        print(f"saved j60_fit.json ({time.time() - t0:.0f} s wall)", flush=True)
        if "error" in r["fit"]:
            return
        ds2 = dict(ds, th_true=np.asarray(r["fit"]["theta"]))
        s = step.remote(ds2, [dict(op="eval", key="scores", spec=spec)])
        np.save(OUT / "j60_scores.npy", np.asarray(s["scores"]["g"]))
        print(f"saved j60_scores.npy ({time.time() - t0:.0f} s wall)", flush=True)


def run_j60_shift():
    """Scores at the J = 60 16k estimate with 64k draws: where would a 64k fit move? (one-step, OPG as Hessian)."""
    r = json.loads((OUT / "j60_fit.json").read_text())
    ds = dict(pack(data_j(60)), th_true=np.asarray(r["fit"]["theta"]))
    with modal.enable_output(), app.run():
        s = step.remote(ds, [dict(op="eval", key="scores", spec=dict(kind="panel", M=65536, order="genz"))])
    np.save(OUT / "j60_scores_64k.npy", np.asarray(s["scores"]["g"]))
    print(f"saved j60_scores_64k.npy ({s['scores']['wall']:.0f} s eval)", flush=True)


def report_j60_shift():
    from scipy import stats

    r = json.loads((OUT / "j60_fit.json").read_text())
    ds = data_j(60)
    names, tt, th = ds["names"], np.asarray(ds["th_true"]), np.asarray(r["fit"]["theta"])
    g16, g64 = np.load(OUT / "j60_scores.npy"), np.load(OUT / "j60_scores_64k.npy")
    V = np.linalg.inv(g16.T @ g16)
    se = np.sqrt(np.diag(V))
    dg = (g64 - g16).sum(0)
    th64 = th + V @ dg
    sw = slice(len(th) - 3, len(th))

    def shares(t):
        e = np.exp(np.r_[t[sw], 0.0])
        return e / e.sum()

    for tag, t in (("16k fit", th), ("64k predicted", th64)):
        z = (t - tt) / se
        d = t[sw] - tt[sw]
        W = d @ np.linalg.inv(V[sw, sw]) @ d
        rej = np.where(_holm(2 * stats.norm.sf(np.abs(z))) < .05)[0]
        print(f"{tag:14s}: shares {np.round(shares(t), 3)} (truth {np.round(shares(tt), 3)}), shares Wald p "
              f"{stats.chi2.sf(W, 3):.3g}; Holm rejections {[names[i] for i in rej]}; z u_brand/u_flavor/u_pack "
              f"{np.round(z[sw], 1)}")
    sh = (th64 - th) / se
    print(f"predicted move 16k -> 64k: max |shift| {np.abs(sh).max():.2f} SE ({names[int(np.argmax(np.abs(sh)))]}); "
          f"top: " + ", ".join(f"{names[i]} {sh[i]:+.2f}" for i in np.argsort(-np.abs(sh))[:5]))


def report_j60():
    from scipy import stats

    r = json.loads((OUT / "j60_fit.json").read_text())
    f = r["fit"]
    ds = data_j(60, 1000)
    names, tt = ds["names"], np.asarray(ds["th_true"])
    th = np.asarray(f["theta"])
    g = np.load(OUT / "j60_scores.npy")
    V = np.linalg.inv(g.T @ g)
    se = np.sqrt(np.diag(V))
    z = (th - tt) / se
    p = 2 * stats.norm.sf(np.abs(z))
    rej = np.where(_holm(p) < .05)[0]
    sw = slice(len(th) - 3, len(th))
    d = th[sw] - tt[sw]
    W = d @ np.linalg.inv(V[sw, sw]) @ d
    Wall = (th - tt) @ np.linalg.inv(V) @ (th - tt)

    def shares(t):
        e = np.exp(np.r_[t[sw], 0.0])
        return e / e.sum()

    print(f"J = 60 (d = {r['d']}), {r['gpu']}: fit {f['seconds'] / 60:.1f} min, {f['n_iter']} iter / {f['n_evals']} evals, "
          f"grad max {f['grad_max']:.2e}; nll {f['nll']:.1f} (at truth {f['nll_true']:.1f})")
    print(f"  shares {np.round(shares(th), 3)} (truth {np.round(shares(tt), 3)}); shares Wald p {stats.chi2.sf(W, 3):.3g}")
    print(f"  Holm rejections vs truth (OPG SEs): {[names[i] for i in rej]}; joint Wald p {stats.chi2.sf(Wall, len(th)):.3g}")
    print(f"  max |z| {np.abs(z).max():.2f} ({names[int(np.argmax(np.abs(z)))]}); z: " +
          ", ".join(f"{n} {v:+.1f}" for n, v in zip(names, z)))


def report_smoke():
    for f in sorted(OUT.glob("smoke_*.json")):
        r = json.loads(f.read_text())
        print(f"\n== {f.name}: {r['name']} d = {r['d']}, n = {r['n']}, {r['gpu']}, order {r['order_seconds']:.1f} s")
        if "ref_panel" in r and "ll" in r["ref_panel"]:
            ref = np.array(r["ref_panel"]["ll"])
            print(f"  ref_panel: mean log L {ref.mean():.2f}, ESS/M median {np.median(r['ref_panel']['ess']) / REF_M:.3g}, "
                  f"{r['ref_panel']['wall']:.0f} s")
            if "ll" in r.get("ref_nested", {}):
                dd = np.array(r["ref_nested"]["ll"]) - ref
                print(f"  ref cross-check nested - panel: mean {dd.mean():+.4f} [{dd.mean() - 1.96 * dd.std(ddof=1) / len(dd) ** .5:+.4f},"
                      f" {dd.mean() + 1.96 * dd.std(ddof=1) / len(dd) ** .5:+.4f}], max |.| {np.abs(dd).max():.4f}, "
                      f"{r['ref_nested']['wall']:.0f} s")
            for k, v in r.items():
                if isinstance(v, dict) and v.get("spec", {}).get("kind") in ("panel", "nested") and "ll" in v:
                    e = np.array(v["ll"]) - ref
                    M = v["spec"].get("M") or v["spec"]["Q"] * v["spec"]["R"]
                    print(f"  {k:24s} |err| median {np.median(np.abs(e)):.4f} p90 {np.quantile(np.abs(e), .9):.4f} "
                          f"signed mean {e.mean():+.4f}  ESS/M {np.median(v['ess']) / (v['spec'].get('M') or v['spec']['Q']):.3g}  "
                          f"{v['wall']:.1f} s")
        for k, v in r.items():
            if k.startswith("smooth") and "ll0" in v:
                ll0, g = np.array(v["ll0"]), np.array(v["g"])
                dirs = directions(g.shape[1])
                llp, llm = np.array(v["llp"]), np.array(v["llm"])
                jump = np.abs(llp[:, -1] - ll0).max()
                res = np.abs(llp - ll0 - np.array(HS)[None, :, None] * (dirs @ g.T)[:, None, :])   # Taylor residual
                slope = np.polyfit(np.log(HS), np.log(np.median(res, axis=(0, 2)) + 1e-300), 1)[0]
                fd = (llp[:, 0] - llm[:, 0]) / (2 * HS[0])
                gap = np.abs(fd - dirs @ g.T).max() / np.abs(dirs @ g.T).max()
                print(f"  {k:24s} max |dlogL| at h=3e-3 {jump:.4f}; Taylor-residual slope {slope:.2f} (2 = smooth); "
                      f"grad vs central diff rel gap {gap:.2e}")
            if k.startswith("time") and "seconds" in v:
                print(f"  {k:24s} {np.median(v['seconds']):.1f} s per likelihood + gradient (N = {r['n']}), "
                      f"chunk {v['chunk']}, peak {v['peak_GiB']:.1f} GiB")
            if isinstance(v, dict) and "error" in v:
                print(f"  {k:24s} ERROR {v['error']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["smoke"]:
        run_smoke()
    elif a == ["report", "smoke"]:
        report_smoke()
    elif a == ["a2"]:
        run_a2()
    elif a == ["report", "a2"]:
        report_a2()
    elif a == ["a1"]:
        run_a1()
    elif a == ["report", "a1"]:
        report_a1()
    elif a == ["j60shift"]:
        run_j60_shift()
    elif a == ["report", "j60shift"]:
        report_j60_shift()
    elif a == ["j60"]:
        run_j60()
    elif a == ["report", "j60"]:
        report_j60()
    elif a == ["warm"]:
        run_warm()
    elif a == ["report", "warm"]:
        report_warm()
    elif a == ["jsmoke"]:
        run_jsmoke()
    elif a == ["report", "jsmoke"]:
        report_jsmoke()
    elif a == ["refprobe"]:
        run_refprobe()
    elif a == ["report", "refprobe"]:
        report_refprobe()
    else:
        sys.exit(__doc__)
