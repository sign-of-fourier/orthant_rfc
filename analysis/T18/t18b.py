"""T18 Part B: non-normal tastes (TEST.md, Part B). T4 S1 errors, J = 20, T = 12, N = 1,000 (d = 240).

Truths
  LN   price coefficient -exp(m + s z), mean / SD matched to T4's normal slope (-0.975, 0.41); other tastes as T4.
       theta (25) = T4's layout with the price slots read as (m, log s).
  MIX  two segments, pi = 0.4 / 0.6, means b + 0.6 delta / b - 0.4 delta (population mean = T4's b), delta on
       brand0 and price; common diagonal taste SDs. theta (28) = b (segment 2), log sd, u, delta_brand0,
       delta_price, logit pi.
Arms (all GHK with fixed draws; panel arms use the Genz order fixed at theta0)
  nested   Q x R = 512 x 256 taste draws from the true family x per-task GHK (MIX: per segment, summed)
  mixture  sum of panel orthants: MIX exact (2 components); LN by K equal-probability strata of z, each a normal
           component with the stratum's mean and variance of the lognormal (K = 3, 5)
  copula   P ~ Phi_R(k): k_i = Phi^-1(P(D_i > 0)) from each difference's true marginal, R the correlation of D
           with the true (moment-matched) taste covariance; one panel orthant
Reference (B1): MIX the exact mixture at 2^18 x 8 shifts; LN Gauss-Hermite over z_price x a panel orthant
  given z_price (price coefficient fixed at each node), exact up to quadrature and GHK.

    python analysis/T18/t18b.py check          # local: segment share gap (MIX), data shapes; numpy only
    python analysis/T18/t18b.py smoke          # B0: 20 respondents per truth, all arms, references, timing
    python analysis/T18/t18b.py report smoke
    python analysis/T18/t18b.py b2             # lean B2: copula and exact-orthant fits per truth (4 H100s)
    python analysis/T18/t18b.py report b2      # shares / decisions vs the truth (local numpy, 10^6 draws)
"""
import json
import math
import sys
import time
from pathlib import Path

import modal
import numpy as np

import t18

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
image = t18.image.add_local_python_source("t18")
app = modal.App("orthant-rfc-t18b")

J, T = 20, 12
PRICE, BR0 = 9, 0
LN_S = math.sqrt(math.log(1 + (0.41 / 0.975) ** 2))
LN_M = math.log(0.975) - LN_S ** 2 / 2
MIX_PI = 0.4
MIX_DELTA = (1.25, -0.75)                       # brand0, price
PANEL_M = 16384


# ---------------------------------------------------------------------------
# Truths and data (local, numpy)
# ---------------------------------------------------------------------------
def _t4():
    t18._paths()
    import t4
    return t4


def theta_true(family):
    t4 = _t4()
    sh = t4.TRUTHS["S1"]
    b, lsd, u = t4.b_true(J).copy(), np.log(t4.W_TRUE).copy(), np.log(sh[:3] / sh[3])
    if family == "LN":
        b[PRICE], lsd[PRICE] = LN_M, math.log(LN_S)
        return np.r_[b, lsd, u]
    d = np.zeros(t4.P)
    d[[BR0, PRICE]] = MIX_DELTA
    b2 = b - (1 - MIX_PI) * d
    return np.r_[b2, lsd, u, MIX_DELTA, math.log(MIX_PI / (1 - MIX_PI))]


def theta_start(family):
    P = _t4().P
    th = np.r_[np.zeros(P), np.log(np.full(P, 0.5)), np.zeros(3)]
    return th if family == "LN" else np.r_[th, 0.5, -0.5, 0.0]


def names(family):
    t4 = _t4()
    b = list(t4.NAMES)
    if family == "LN":
        b[PRICE] = "price_m"
    n = b + [f"logsd_{x}" for x in t4.NAMES] + ["u_brand", "u_flavor", "u_pack"]
    if family == "LN":
        n[t4.P + PRICE] = "price_logs"
        return n
    return n + ["delta_brand0", "delta_price", "logit_pi"]


def draw_beta(family, th, rng, n):
    """Taste draws from the family at theta; consumes the same streams for every theta (common random numbers)."""
    t4 = _t4()
    P = t4.P
    z = rng.standard_normal((n, P))
    uu = rng.random(n)
    b, sd = th[:P], np.exp(th[P:2 * P])
    beta = b + sd * z
    if family == "LN":
        beta[:, PRICE] = -np.exp(b[PRICE] + sd[PRICE] * z[:, PRICE])
        return beta
    d = np.zeros(P)
    d[[BR0, PRICE]] = th[2 * P + 3:2 * P + 5]
    pi = 1 / (1 + math.exp(-th[2 * P + 5]))
    return beta + (uu < pi)[:, None] * d


def sig_of_theta(th):
    t4 = _t4()
    P = t4.P
    z = np.exp(np.r_[th[2 * P:2 * P + 3], 0.0])
    return np.sqrt(t4.T_VAR * z / z.sum())


def make_data(family, N=1000, rep=0):
    """T4.make_data with this family's tastes (own seed stream)."""
    t4 = _t4()
    code = {"LN": 18, "MIX": 19}[family]
    sig = t4.sig_of("S1")
    r_data, r_hold = [np.random.default_rng(s) for s in np.random.SeedSequence([t4.SEED, J, rep, code]).spawn(2)]
    levels = t4.ALL[t4.shelf(J)]
    th = theta_true(family)
    beta = draw_beta(family, th, r_data, N)
    prices = t4.PRICES[r_data.integers(0, 5, (N, T, J))]
    y = t4.simulate(r_data, levels, prices, beta, sig, False)
    Hp = t4.PRICES[np.random.default_rng([t4.SEED, J, 7]).integers(0, 5, (6, J))]
    yh = t4.simulate(r_hold, levels, np.broadcast_to(Hp, (N, 6, J)), beta, sig, False)
    return dict(levels=levels, prices=prices, y=y, beta=beta, Hp=Hp, yh=yh)


def data_b(family, N=1000):
    t18._paths()
    import ghk_gpu
    t4 = _t4()
    d = make_data(family, N)
    lv = d["levels"]
    prods = np.concatenate([np.broadcast_to(lv, (N, T, J, 3)), d["prices"][..., None]], -1)
    Xd, G = ghk_gpu.panel_arrays(prods, d["y"], lambda pr: t4.task_X(lv, pr[..., 3]), levels=t4.LEVELS)
    return dict(name=family, family=family, Xd=Xd.astype(np.float32), G=G.astype(np.int8), th_true=theta_true(family),
                th0=theta_start(family), tvar=float(t4.T_VAR), names=names(family))


def shares_sim(family, th, scens, n_sim=10 ** 6, seed=1818, chunk=50_000):
    """Population shares (S, J+2) by first choice, T4's shares_probit with this family's tastes; the same draws for
    every theta and scenario (common random numbers)."""
    t4 = _t4()
    levels = t4.ALL[t4.shelf(J)]
    sig = sig_of_theta(th)
    s3 = np.repeat(sig[:3], t4.LEVELS)
    out = np.zeros((len(scens), J + 2))
    rng = np.random.default_rng(seed)
    for c0 in range(0, n_sim, chunk):
        n = min(chunk, n_sim - c0)
        beta = draw_beta(family, th, rng, n)
        zeta = rng.standard_normal((n, t4.NLEV)) * s3
        nu = rng.standard_normal((n, J + 2))
        for i, s in enumerate(scens):
            lv = t4.scen_levels(levels, s)
            U = beta @ t4.task_X(lv, s["prices"]).T + zeta @ t4.member(lv).T
            U[:, :J + 1] += sig[3] * nu[:, :J + 1]
            U[:, J + 1] += math.sqrt(t4.T_VAR) * nu[:, J + 1]
            U[:, :J + 1][:, ~s["present"]] = -np.inf
            out[i] += np.bincount(U.argmax(1), minlength=J + 2)
    return out / n_sim


def check():
    """Local: the MIX segment gap on the firm's 6 SKUs (base scenario) and data shapes."""
    t4 = _t4()
    scens = [s for s in t4.scenarios(J) if s["name"] == "base"]
    th = theta_true("MIX")
    P = t4.P
    seg = []
    for lpi in (30.0, -30.0):                                    # pi -> 1 (segment 1), pi -> 0 (segment 2)
        t = th.copy()
        t[-1] = lpi
        seg.append(shares_sim("MIX", t, scens, n_sim=200_000)[0, :6].sum())
    print(f"MIX firm share (6 SKUs, base): segment 1 {100 * seg[0]:.1f} pts, segment 2 {100 * seg[1]:.1f} pts, "
          f"gap {100 * (seg[0] - seg[1]):.1f} (need >= 5)")
    for fam in ("LN", "MIX"):
        d = data_b(fam, 1000)
        print(fam, "Xd", d["Xd"].shape, "theta", len(d["th_true"]), "none share",
              f"{np.mean(make_data(fam)['y'] == J):.3f}")
        print("  true theta:", dict(zip(d["names"], np.round(d["th_true"], 3))))
    print(f"LN m {LN_M:.4f} s {LN_S:.4f}; mean {-math.exp(LN_M + LN_S ** 2 / 2):.4f} "
          f"sd {math.sqrt((math.exp(LN_S ** 2) - 1) * math.exp(2 * LN_M + LN_S ** 2)):.4f}")


# ---------------------------------------------------------------------------
# Likelihood pieces (torch, inside Modal)
# ---------------------------------------------------------------------------
def _unpack(thr, P, tvar):
    b, sd2 = thr[:, :P], (2 * thr[:, P:2 * P]).exp()
    s2 = t18.sig2_of(thr[:, 2 * P:2 * P + 3], tvar)
    return b, sd2, s2


def _bd(blk):
    """Block-diagonal (n, T K, T K) from per-task blocks (n, T, K, K)."""
    import torch

    n, T_, K, _ = blk.shape
    return torch.diag_embed(blk.permute(0, 2, 3, 1)).permute(0, 3, 1, 4, 2).reshape(n, T_ * K, T_ * K)


def _delta(thr, P):
    import torch

    d = torch.zeros(thr.shape[0], P, dtype=thr.dtype, device=thr.device)
    d[:, BR0] = thr[:, 2 * P + 3]
    d[:, PRICE] = thr[:, 2 * P + 4]
    return d


def components(spec, family, thr, P):
    """[(log weight (n,), mean (n, P), variance (n, P))] of the normal components of the taste distribution
    (mixture arm and LN reference)."""
    import torch

    b, sd2 = thr[:, :P], (2 * thr[:, P:2 * P]).exp()
    n = thr.shape[0]
    out = []
    if family == "MIX":
        lpi = torch.nn.functional.logsigmoid(thr[:, 2 * P + 5])
        l1pi = torch.nn.functional.logsigmoid(-thr[:, 2 * P + 5])
        return [(lpi, b + _delta(thr, P), sd2), (l1pi, b, sd2)]
    m, s = b[:, PRICE], sd2[:, PRICE].sqrt()
    if spec.get("comp") == "gh":                                # point masses at Gauss-Hermite nodes of z
        x, w = np.polynomial.hermite_e.hermegauss(spec["n_gh"])
        for xg, lw in zip(x, np.log(w / w.sum())):
            mean = b.clone()
            mean[:, PRICE] = -(m + s * float(xg)).exp()
            var = sd2.clone()
            var[:, PRICE] = 0.0
            out.append((torch.full((n,), float(lw), dtype=thr.dtype, device=thr.device), mean, var))
        return out
    K = spec["K"]                                               # equal-probability strata, moment-matched
    from statistics import NormalDist
    edges = [-math.inf] + [NormalDist().inv_cdf(i / K) for i in range(1, K)] + [math.inf]
    nd = torch.special.ndtr
    for a, c in zip(edges[:-1], edges[1:]):
        def mass(shift):
            hi = nd(torch.as_tensor(c, dtype=thr.dtype, device=thr.device) - shift)
            lo = nd(torch.as_tensor(a, dtype=thr.dtype, device=thr.device) - shift)
            return (hi - lo) * K
        m1 = (m + s ** 2 / 2).exp() * mass(s)
        m2 = (2 * m + 2 * s ** 2).exp() * mass(2 * s)
        mean = b.clone()
        mean[:, PRICE] = -m1
        var = sd2.clone()
        var[:, PRICE] = (m2 - m1 ** 2).clamp_min(1e-10)
        out.append((torch.full((n,), -math.log(K), dtype=thr.dtype, device=thr.device), mean, var))
    return out


def panel_mc(mean, var, s2, Xd, G):
    """Stacked differences' eta (n, d) and covariance (n, d, d) for normal tastes N(mean, diag var)."""
    import torch

    n, T_, K, P = Xd.shape
    Z = Xd.reshape(n, T_ * K, P)
    eta = (Z * mean[:, None]).sum(-1)
    cov = (Z * var[:, None]) @ Z.transpose(1, 2)
    blk = torch.einsum("na,antij->ntij", s2, G)
    return eta, cov + _bd(blk)


def copula_moments(family, thr, Xd, G, tvar, n_gh=32):
    """Gaussian-copula orthant: cutoffs k (n, d) from each difference's true marginal, correlation R (n, d, d)
    of D under the moment-matched taste covariance."""
    import torch
    from torch.special import log_ndtr, ndtri

    n, T_, K, P = Xd.shape
    b, sd2, s2 = _unpack(thr, P, tvar)
    Z = Xd.reshape(n, T_ * K, P)
    bd = _bd(torch.einsum("na,antij->ntij", s2, G))
    if family == "LN":
        m, s = b[:, PRICE], sd2[:, PRICE].sqrt()
        mean = b.clone()
        mean[:, PRICE] = -(m + s ** 2 / 2).exp()
        Cb = torch.diag_embed(sd2.clone())
        Cb[:, PRICE, PRICE] = ((s ** 2).exp() - 1) * (2 * m + s ** 2).exp()
        brest, vrest = b.clone(), sd2.clone()
        brest[:, PRICE], vrest[:, PRICE] = 0.0, 0.0
        mu = (Z * brest[:, None]).sum(-1)                                            # (n, d)
        sr = ((Z ** 2 * vrest[:, None]).sum(-1) + bd.diagonal(dim1=1, dim2=2)).sqrt()
        x, w = np.polynomial.hermite_e.hermegauss(n_gh)
        xg = torch.as_tensor(x, dtype=thr.dtype, device=thr.device)
        lw = torch.as_tensor(np.log(w / w.sum()), dtype=thr.dtype, device=thr.device)
        bp = -(m[:, None] + s[:, None] * xg).exp()                                   # (n, G)
        arg = (mu[..., None] + Z[..., PRICE, None] * bp[:, None, :]) / sr[..., None]  # (n, d, G)
        logP = torch.logsumexp(log_ndtr(arg) + lw, -1)
    else:
        lpi = torch.nn.functional.logsigmoid(thr[:, 2 * P + 5])
        l1pi = torch.nn.functional.logsigmoid(-thr[:, 2 * P + 5])
        pi = lpi.exp()
        d = _delta(thr, P)
        mean = b + pi[:, None] * d
        Cb = torch.diag_embed(sd2) + (pi * (1 - pi))[:, None, None] * d[:, :, None] * d[:, None, :]
        sr = ((Z ** 2 * sd2[:, None]).sum(-1) + bd.diagonal(dim1=1, dim2=2)).sqrt()
        mu1, mu2 = (Z * (b + d)[:, None]).sum(-1), (Z * b[:, None]).sum(-1)
        logP = torch.logaddexp(lpi[:, None] + log_ndtr(mu1 / sr), l1pi[:, None] + log_ndtr(mu2 / sr))
    tiny = torch.finfo(thr.dtype).tiny
    k = ndtri(logP.exp().clamp(tiny, 1 - torch.finfo(thr.dtype).eps))
    cov = Z @ Cb @ Z.transpose(1, 2) + bd
    sdv = cov.diagonal(dim1=1, dim2=2).sqrt()
    R = cov / (sdv[:, :, None] * sdv[:, None, :])
    return k, R


def nested_beta(family, spec, thr, zb, P):
    """[(log weight (n,), beta draws (n, Q, P))] for the nested arm."""
    import torch

    b, sd = thr[:, :P], thr[:, P:2 * P].exp()
    if family == "LN":
        beta = b[:, None] + sd[:, None] * zb
        beta = torch.cat([beta[..., :PRICE], -(b[:, None, PRICE] + sd[:, None, PRICE] * zb[..., PRICE]).exp()[..., None],
                          beta[..., PRICE + 1:]], -1)
        return [(torch.zeros(thr.shape[0], dtype=thr.dtype, device=thr.device), beta)]
    lpi = torch.nn.functional.logsigmoid(thr[:, 2 * P + 5])
    l1pi = torch.nn.functional.logsigmoid(-thr[:, 2 * P + 5])
    base = b[:, None] + sd[:, None] * zb
    return [(lpi, base + _delta(thr, P)[:, None]), (l1pi, base)]


# ---------------------------------------------------------------------------
# Evaluation (inside Modal): parts over components x draw chunks, combined by one logsumexp
# ---------------------------------------------------------------------------
def evaluate(spec, family, Xd, G, tvar, th, perm=None, rows=None, grad=True):
    import torch

    dev = Xd.device
    dt = torch.float64 if spec.get("dtype", "f32") == "f64" else torch.float32
    kind, sh = spec["kind"], spec.get("shift", 0)
    N0, T_, K, P = Xd.shape
    d = T_ * K
    seed = t18.arm_seed({"nested": "nested", "ref": "ref_panel"}.get(kind, "panel"), sh) + (7 if kind == "copula" else 0)
    ebytes = 8 if dt == torch.float64 else 4
    budget = spec.get("budget_gb", 40) * 2 ** 30
    if kind == "nested":
        Q, R = spec["Q"], spec["R"]
        dch = min(Q, spec.get("draw_chunk", 64))
        zb_all = torch.special.ndtri(torch.remainder(t18.sobol(P, Q, seed)[None] + t18.shifts(N0, P, seed + 1)[:, None], 1.0)
                                     .clamp(1e-12, 1 - 1e-12)).to(dev, dt)
        base = t18.sobol(K - 1, R, seed + 2).to(dev, dt)
        shf = t18.shifts(N0, K - 1, seed + 3, extra=(T_,)).to(dev, dt)
        per = dch * T_ * R * K * ebytes * 12
    else:
        M = spec["M"]
        dch = min(M, spec.get("draw_chunk", 16384))
        base = t18.SobolChunks(d - 1, M, seed, dev, dt)
        shf = t18.shifts(N0, d - 1, seed + 1).to(dev, dt)
        zb_all = None
        per = dch * d * ebytes * 12
    chunk = int(max(1, min(spec.get("rows_max", 256), budget // per)))
    idx = torch.arange(N0, device=dev) if rows is None else torch.as_tensor(rows, device=dev)
    lls, gs, ess = [], [], []
    lo = 0
    while lo < len(idx):
        hi = min(lo + chunk, len(idx))
        r = idx[lo:hi]
        try:
            ll, g, e = _rows(spec, family, Xd[r].to(dt), G[:, r].to(dt), tvar, th, r, perm, grad, dt, dch, base, shf, zb_all)
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


def _rows(spec, family, Xd, G, tvar, th, r, perm, grad, dt, dch, base, shf, zb_all):
    import torch

    kind = spec["kind"]
    n, T_, K, P = Xd.shape
    thr = torch.as_tensor(th, dtype=dt, device=Xd.device).expand(n, -1).clone().requires_grad_(grad)
    parts = []

    def add(lp, Mc, count):
        lse, lse2, _, g = t18._part(lp, thr, grad)
        parts.append((lse, lse2, Mc if count else 0, g))

    if kind == "nested":
        Q = zb_all.shape[1]
        w = torch.remainder(base[None, None] + shf[r][:, :, None], 1.0)[:, None]
        Lt = None
        for ci in range(2 if family == "MIX" else 1):
            for c0 in range(0, Q, dch):
                with torch.set_grad_enabled(grad):
                    b, sd2, s2 = _unpack(thr, P, tvar)
                    Lt = torch.linalg.cholesky(torch.einsum("na,antij->ntij", s2, G))[:, None]
                    lw, beta = nested_beta(family, spec, thr, zb_all[r, c0:c0 + dch], P)[ci]
                    eta = torch.einsum("ntkp,nqp->nqtk", Xd, beta)[..., None, :]
                    lpr = t18.ghk(eta, Lt, w)
                    lpt = torch.logsumexp(lpr, -1) - math.log(lpr.shape[-1])
                    add(lpt.sum(-1) + lw[:, None], lpt.shape[1], ci == 0)
        return t18._combine(parts, grad)
    M = base.n
    ncomp = 1 if kind == "copula" else len(components(spec, family, thr.detach(), P))
    for ci in range(ncomp):
        for c0 in range(0, M, dch):
            w = torch.remainder(base.get(c0, dch)[None] + shf[r][:, None], 1.0)
            with torch.set_grad_enabled(grad):
                if kind == "copula":
                    eta, cov = copula_moments(family, thr, Xd, G, tvar)
                    lw = torch.zeros(n, dtype=dt, device=Xd.device)
                else:
                    b, sd2, s2 = _unpack(thr, P, tvar)
                    lw, mean, var = components(spec, family, thr, P)[ci]
                    eta, cov = panel_mc(mean, var, s2, Xd, G)
                if perm is not None:
                    eta, cov = t18.permute(eta, cov, perm[r])
                lp = t18.ghk(eta[:, None, :], torch.linalg.cholesky(cov), w)
                add(lp + lw[:, None], lp.shape[1], ci == 0)
    return t18._combine(parts, grad)


def fit_lbfgs(spec, family, Xd, G, tvar, th0, th_true, perm, max_iter=300):
    import torch

    N = Xd.shape[0]
    theta = torch.tensor(th0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-6,
                            tolerance_change=1e-10, line_search_fn="strong_wolfe")
    nev, trace = [0], []

    def closure():
        r = evaluate(spec, family, Xd, G, tvar, theta.detach().numpy(), perm)
        nll = -float(np.sum(r["ll"]))
        theta.grad = torch.as_tensor(-np.sum(r["g"], 0) / N, dtype=torch.float64)
        nev[0] += 1
        trace.append(nll)
        return torch.tensor(nll / N, dtype=torch.float64)

    t0 = time.time()
    opt.step(closure)
    nll = closure().item() * N
    secs = time.time() - t0
    nll_true = -float(np.sum(evaluate(spec, family, Xd, G, tvar, np.asarray(th_true), perm, grad=False)["ll"]))
    return dict(theta=theta.detach().numpy().tolist(), nll=nll, nll_true=nll_true, grad_max=float(theta.grad.abs().max() * N),
                n_iter=int(opt.state[opt._params[0]]["n_iter"]), n_evals=nev[0], seconds=secs, trace=trace)


def timed(spec, family, Xd, G, tvar, th, perm, n_rep=3):
    import torch

    evaluate(spec, family, Xd, G, tvar, th, perm, rows=list(range(min(4, Xd.shape[0]))))
    ts = []
    for _ in range(n_rep):
        torch.cuda.synchronize()
        t0 = time.time()
        r = evaluate(spec, family, Xd, G, tvar, th, perm)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return dict(seconds=ts, chunk=r["chunk"])


def smooth(spec, family, Xd, G, tvar, th, perm, rows, dirs, hs):
    base = evaluate(spec, family, Xd, G, tvar, th, perm, rows, grad=True)
    llp = np.zeros((len(dirs), len(hs), len(rows)))
    llm = np.zeros_like(llp)
    for i, e in enumerate(dirs):
        for k, h in enumerate(hs):
            llp[i, k] = evaluate(spec, family, Xd, G, tvar, th + h * e, perm, rows, grad=False)["ll"]
            llm[i, k] = evaluate(spec, family, Xd, G, tvar, th - h * e, perm, rows, grad=False)["ll"]
    return dict(ll0=base["ll"], g=base["g"], llp=llp, llm=llm)


@app.function(image=image, gpu="H100", timeout=4 * 3600, memory=32768)
def step(ds, tasks):
    import torch

    dev = torch.device("cuda")
    fam = ds["family"]
    Xd = torch.as_tensor(ds["Xd"], dtype=torch.float64, device=dev)
    G = torch.as_tensor(ds["G"], dtype=torch.float64, device=dev)
    th, th0, tvar = np.asarray(ds["th_true"]), np.asarray(ds["th0"]), ds["tvar"]
    P = Xd.shape[3]
    out = dict(gpu=torch.cuda.get_device_name(), name=ds["name"], n=Xd.shape[0], d=Xd.shape[1] * Xd.shape[2])
    with torch.no_grad():                               # Genz order at theta0 of the normal model (T4 gate start)
        eta, cov = t18.panel_moments(torch.as_tensor(t18.theta0(P), device=dev).expand(Xd.shape[0], -1), Xd, G, tvar)
        perm = t18.genz_order(eta, cov)
    for t in tasks:
        key, spec = t["key"], t.get("spec", {})
        pm = None if spec.get("kind") == "nested" else perm
        t0 = time.time()
        try:
            if t["op"] == "eval":
                r = evaluate(spec, fam, Xd, G, tvar, th, pm, t.get("rows"), grad=t.get("grad", True))
            elif t["op"] == "smooth":
                r = smooth(spec, fam, Xd, G, tvar, th, pm, t["rows"], np.asarray(t["dirs"]), t["hs"])
            elif t["op"] == "time":
                r = timed(spec, fam, Xd, G, tvar, th, pm)
            elif t["op"] == "fit":
                r = fit_lbfgs(spec, fam, Xd, G, tvar, th0, th, pm, t.get("max_iter", 300))
            r = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        except Exception as e:
            import traceback
            r = dict(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
            torch.cuda.empty_cache()
        r["wall"], r["spec"] = time.time() - t0, spec
        out[key] = r
        print(f"{fam} {key}: {r['wall']:.1f} s" + (f"  {r['error']}" if "error" in r else ""), flush=True)
    return out


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
def arms(family):
    a = [("nested", dict(kind="nested", Q=t18.NESTED[0], R=t18.NESTED[1])), ("copula", dict(kind="copula", M=PANEL_M))]
    if family == "MIX":
        return a + [("mixture", dict(kind="mix", M=PANEL_M))]
    return a + [(f"mixture_K{k}", dict(kind="mix", M=PANEL_M, K=k)) for k in (3, 5)]


def refs(family, shift=0):
    if family == "MIX":
        return [("ref", dict(kind="ref", M=2 ** 18, dtype="f64", shift=shift))]
    return [(f"ref_gh{n}", dict(kind="ref", comp="gh", n_gh=n, M=2 ** 14, dtype="f64", shift=shift)) for n in (32, 64)]


def run_smoke(n=20):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for fam in ("LN", "MIX"):
        ds = data_b(fam)
        small = dict(ds, Xd=ds["Xd"][:n], G=ds["G"][:, :n])
        rows = list(range(n))
        tasks = [dict(op="eval", key=k, spec=s, rows=rows) for k, s in refs(fam)]
        tasks += [dict(op="eval", key=k, spec=s, rows=rows) for k, s in arms(fam)]
        dirs = t18.directions(len(ds["th_true"])).tolist()
        tasks += [dict(op="smooth", key=f"smooth_{k}", spec=dict(s, dtype="f64"), rows=rows[:10], dirs=dirs, hs=t18.HS)
                  for k, s in arms(fam)]
        jobs.append((fam, small, tasks))
        jobs.append((fam + "_time", ds, [dict(op="time", key=f"time_{k}", spec=s) for k, s in arms(fam)]))
    print("jobs:", [(t, d["Xd"].shape, len(x)) for t, d, x in jobs], flush=True)
    t0 = time.time()
    with modal.enable_output(), app.run():
        calls = [(tag, ds, step.spawn(ds, tasks)) for tag, ds, tasks in jobs]
        for tag, ds, c in calls:
            r = c.get()
            r["names"], r["th_true"] = ds["names"], list(ds["th_true"])
            (OUT / f"b0_{tag}.json").write_text(json.dumps(r))
            print(f"saved b0_{tag}.json ({time.time() - t0:.0f} s wall)", flush=True)


def report_smoke():
    for fam in ("LN", "MIX"):
        f = OUT / f"b0_{fam}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        print(f"\n== {fam}: d = {r['d']}, n = {r['n']}, {r['gpu']}")
        for k, v in r.items():
            if isinstance(v, dict) and "error" in v:
                print(f"  {k}: ERROR {v['error']}\n{v.get('tb', '')}")
        rk = "ref" if fam == "MIX" else "ref_gh64"
        if "ll" not in r.get(rk, {}):
            continue
        ref = np.array(r[rk]["ll"])
        print(f"  {rk}: mean log L {ref.mean():.3f}, {r[rk]['wall']:.0f} s")
        if fam == "LN" and "ll" in r.get("ref_gh32", {}):
            dd = np.array(r["ref_gh32"]["ll"]) - ref
            print(f"  ref gh32 - gh64: mean {dd.mean():+.4f}, max |.| {np.abs(dd).max():.4f}")
        for k, s in arms(fam):
            v = r.get(k, {})
            if "ll" in v:
                e = np.array(v["ll"]) - ref
                print(f"  {k:14s} |err| median {np.median(np.abs(e)):.4f} p90 {np.quantile(np.abs(e), .9):.4f} "
                      f"signed mean {e.mean():+.4f}  {v['wall']:.1f} s")
        for k, v in r.items():
            if k.startswith("smooth") and "ll0" in v:
                ll0, g = np.array(v["ll0"]), np.array(v["g"])
                dirs = t18.directions(g.shape[1])
                llp, llm = np.array(v["llp"]), np.array(v["llm"])
                res = np.abs(llp - ll0 - np.array(t18.HS)[None, :, None] * (dirs @ g.T)[:, None, :])
                slope = np.polyfit(np.log(t18.HS), np.log(np.median(res, axis=(0, 2)) + 1e-300), 1)[0]
                fd = (llp[:, 0] - llm[:, 0]) / (2 * t18.HS[0])
                gap = np.abs(fd - dirs @ g.T).max() / np.abs(dirs @ g.T).max()
                print(f"  {k:22s} slope {slope:.2f}, grad gap {gap:.1e}, max |dlogL| at 3e-3 {np.abs(llp[:, -1] - ll0).max():.4f}")
        ft = OUT / f"b0_{fam}_time.json"
        if ft.exists():
            rt = json.loads(ft.read_text())
            for k, v in rt.items():
                if k.startswith("time"):
                    print(f"  {k:22s} " + (f"{np.median(v['seconds']):.2f} s (N = {rt['n']})" if "seconds" in v else
                                          f"ERROR {v.get('error')}"))


# ---------------------------------------------------------------------------
# Lean B2 (user 2026-10-17): copula vs the exact orthant arm, one fit each per truth; shares and decisions
# against the truth. No nested arm, no B1.
# ---------------------------------------------------------------------------
B2_ARMS = {"LN": [("copula", dict(kind="copula", M=PANEL_M)), ("exact", dict(kind="mix", M=PANEL_M, K=5))],
           "MIX": [("copula", dict(kind="copula", M=PANEL_M)), ("exact", dict(kind="mix", M=PANEL_M))]}
M_SHARE, M_REGRET = 0.5, 0.5


def run_b2():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for fam in ("LN", "MIX"):
        ds = data_b(fam)
        for arm, spec in B2_ARMS[fam]:
            jobs.append((f"{fam}_{arm}", ds, [dict(op="fit", key="fit", spec=spec)]))
    t0 = time.time()
    with modal.enable_output(), app.run():
        calls = [(tag, ds, step.spawn(ds, tasks)) for tag, ds, tasks in jobs]
        for tag, ds, c in calls:
            r = c.get()
            r["names"], r["th_true"] = ds["names"], list(ds["th_true"])
            (OUT / f"b2_{tag}.json").write_text(json.dumps(r))
            print(f"saved b2_{tag}.json ({time.time() - t0:.0f} s wall)", flush=True)


def report_b2(n_sim=10 ** 6):
    from scipy import stats

    t4 = _t4()
    scens = t4.scenarios(J)
    reg_keys = [f"price{k}" for k in range(6)] + ["delist"]
    rows = []
    for fam in ("LN", "MIX"):
        tt = theta_true(fam)
        nm = names(fam)
        t0 = time.time()
        sh_t = shares_sim(fam, tt, scens, n_sim)
        tv = t4.decision_values(sh_t, scens, J)
        print(f"\n== {fam} (truth shares {time.time() - t0:.0f} s)")
        res = {}
        for arm, _ in B2_ARMS[fam]:
            f = OUT / f"b2_{fam}_{arm}.json"
            r = json.loads(f.read_text())["fit"]
            if "error" in r:
                print(f"  {arm}: ERROR {r['error']}")
                continue
            th = np.array(r["theta"])
            sh = shares_sim(fam, th, scens, n_sim)
            sc = t4.score(t4.decision_values(sh, scens, J), tv)
            e_s = 100 * np.abs(sh - sh_t).mean(1)                                   # per scenario, pts
            reg = np.array([sc[k]["regret_pct"] for k in reg_keys])
            flips = sum(sc[k]["flip"] for k in sc)
            inc = np.array([abs(sc[f"ext{k}"]["err"]) for k in range(6)])
            res[arm] = dict(e_s=e_s, reg=reg)
            dth = th - tt
            show = [i for i, n in enumerate(nm) if n.startswith(("u_", "price", "delta", "logit", "logsd_price"))
                    or n in ("brand0",)]
            print(f"  {arm:7s} fit {r['seconds'] / 60:.1f} min, {r['n_iter']} iter / {r['n_evals']} evals, grad max "
                  f"{r['grad_max']:.2e}, nll {r['nll']:.1f} (truth {r['nll_true']:.1f})")
            print(f"          share error {e_s.mean():.3f} pts (max scenario {e_s.max():.3f}); regret mean {reg.mean():.3f}% "
                  f"(max {reg.max():.3f}); flips {flips}/13; ext |incr err| mean {inc.mean():.3f}")
            print("          theta - truth: " + ", ".join(f"{nm[i]} {dth[i]:+.3f}" for i in show))
            print(f"          error shares b/f/p/product {np.round(sig_of_theta(th) ** 2 / t4.T_VAR, 3)} "
                  f"(truth {np.round(sig_of_theta(tt) ** 2 / t4.T_VAR, 3)})")
        if len(res) < 2:
            continue
        for metric, key, m in (("share error", "e_s", M_SHARE), ("regret", "reg", M_REGRET)):
            d = res["exact"][key] - res["copula"][key]                              # + = copula better
            if np.allclose(d, 0):
                mu, lo, hi, p = 0.0, 0.0, 0.0, 1.0
            else:
                n = len(d)
                mu, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
                q = stats.t.ppf(.975, n - 1)
                lo, hi = mu - q * se, mu + q * se
                p = 2 * stats.t.sf(abs(mu / se), n - 1) if se > 0 else 1.0
            rows.append(dict(fam=fam, metric=metric, d=mu, lo=lo, hi=hi, p=p, m=m))
    ph = _holm([r["p"] for r in rows]) if rows else []
    print("\nCopula vs exact orthant arm (Δ = exact − copula, + = copula better; Holm over the 4 comparisons)")
    for r, q in zip(rows, ph):
        lab = _label(r["d"], q < .05, r["m"])
        print(f"  {r['fam']:3s} {r['metric']:11s} Δ {r['d']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] pts, Holm p {q:.2g} -> {lab}")
        r.update(p_holm=q, label=lab)
    (OUT / "b2_summary.json").write_text(json.dumps(rows, indent=1, default=float))


def _holm(p):
    return t18._holm(p)


def _label(d, sig, m):
    return t18._label_noninf(d, sig, m)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["check"]:
        check()
    elif a == ["smoke"]:
        run_smoke()
    elif a == ["report", "smoke"]:
        report_smoke()
    elif a == ["b2"]:
        run_b2()
    elif a == ["report", "b2"]:
        report_b2()
    else:
        sys.exit(__doc__)
