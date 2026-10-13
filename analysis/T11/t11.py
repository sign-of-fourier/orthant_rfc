"""T11: joint chance-constrained product line (PLAN.md, T11).

Per dataset (T5's MVP GHK fits, G1/G2 x N = 150/600 x 20 reps), one T4 job:
  1. sandwich V at T5's stored theta-hat (the fit's own GHK draws; no optimisation);
  2. population shares s(theta) of every line (6 owned SKUs, 3 prices on A, competitors B/D/E fixed)
     by GHK with fixed Sobol draws; Jacobian by central differences; Sigma_s = J V J';
  3. constraint probabilities P(s_j >= t for all kept j) per line and t: C (independent product),
     D (engine), E (GHK-QMC reference integrator), F (OVUS, T9 port), R (nested MC over theta ~ N(theta-hat, V));
  4. truth shares of every line (true theta, large M).
Decisions, metrics and the choice of t are made locally (`summarize`).

    python analysis/T11/t11.py check                 # local CPU wiring check, tiny sizes
    python analysis/T11/t11.py pilot                 # G1 N=600 rep 0 on one T4 (timing)
    python analysis/T11/t11.py run                   # the remaining datasets, in parallel
    python analysis/T11/t11.py summarize             # choose t, decisions, metrics
    python analysis/T11/t11.py price-probe           # report-only probe at the chosen t

PYTHONPATH must contain pybhatlib 0.4.0 for `check` (OVUS).
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
if len(HERE.parents) > 2:                                      # host only (not in the Modal container)
    sys.path.insert(0, str(HERE.parent / "T1"))
    sys.path.insert(0, str(HERE.parent / "T9"))
    sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))   # engine source, sibling checkout

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy", "pybhatlib==0.4.0", "pydantic", "fastapi")
         .add_local_python_source("modal_gp_api", "macml", "ghk_gpu"))
app = modal.App("orthant-rfc-t11")

T_GRID = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10)   # 0.5-1.5% added (user, 2026-10-11)
ALPHA = 0.10
A_PRICES = (4.50, 5.00, 5.50)
OWNED = ("A", "C", "A'", "A''", "F", "G")
COST = 1.80
FD_H = 1e-5                     # theta step for the share Jacobian (shares are smooth in theta: fixed draws)
SIZES = dict(M_V=4096, M_S=2048, M_T=2 ** 15, M_E=2 ** 14, E_REPS=4, B=2000)
PROBE = dict(prices=np.linspace(4.5, 5.5, 41).tolist(), h=1e-3, freeze_at=20)


# ---------------------------------------------------------------------------
# Host side: lines and their design arrays
# ---------------------------------------------------------------------------
def owned_rows():
    import run
    return {"A": run.BASE[0], "C": run.BASE[2], **{k: np.array(v, float) for k, v in run.CAND.items()}}


def lines():
    """Every line: (kept owned indices, A's price or None). Order: by k, then subset, then price."""
    out = []
    for k in range(1, len(OWNED) + 1):
        for sub in combinations(range(len(OWNED)), k):
            out += [(sub, p) for p in A_PRICES] if 0 in sub else [(sub, None)]
    return out


def line_prods(sub, pA):
    import run
    own = owned_rows()
    rows = [run.BASE[1], run.BASE[3], run.BASE[4]]      # competitors B, D, E
    for i in sub:
        r = np.array(own[OWNED[i]], float)
        if i == 0:
            r[3] = pA
        rows.append(r)
    return np.array(rows)


def line_prices(sub, pA):
    own = owned_rows()
    return np.array([pA if i == 0 else own[OWNED[i]][3] for i in sub])


def group_arrays(prods):
    """prods (nl, K, 4) -> X (nl, K1, P), S (4, nl, K1, K1) (as ghk_gpu.panel_arrays), Md (K1, K, K1)."""
    import run
    nl, K, _ = prods.shape
    K1 = K + 1
    X = run.task_X(prods)
    S = np.zeros((4, nl, K1, K1))
    for a in range(3):
        lv = prods[..., a]
        S[a, :, :K, :K] = lv[:, :, None] == lv[:, None, :]
    S[3, :, :K, :K] = np.eye(K)
    S[:, :, K, K] = 1
    Md = np.zeros((K1, K, K1))
    for j in range(K1):
        oth = [k for k in range(K1) if k != j]
        Md[j, np.arange(K), oth] = 1
        Md[j, :, j] = -1
    return X, S, Md


def groups(sel=None):
    """Lines grouped by number kept (same K); sel: subset of line indices (probe)."""
    L = lines()
    idx = range(len(L)) if sel is None else sel
    out = {}
    for i in idx:
        out.setdefault(len(L[i][0]), []).append(i)
    return {k: dict(idx=v, arrays=group_arrays(np.stack([line_prods(*L[i]) for i in v]))) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------
def ghk_p(a, L, u, budget=1.5e9):
    """P(X <= a), X = L Z, by GHK with fixed draws u (M, d-1); rows chunked to ~budget bytes."""
    import torch
    from torch.special import ndtr, ndtri
    n, d = a.shape
    if d == 1:
        return ndtr(a[:, 0] / L[:, 0, 0])
    M = u.shape[0]
    rows = max(1, int(budget / (M * (d + 3) * 8)))
    out = []
    for lo in range(0, n, rows):
        at, Lt = a[lo:lo + rows], L[lo:lo + rows]
        p = None
        zs = []
        for j in range(d):
            c = at[:, j, None].expand(-1, M)
            for k in range(j):
                c = c - Lt[:, j, k, None] * zs[k]
            pj = ndtr(c / Lt[:, j, j, None])
            p = pj if p is None else p * pj
            if j < d - 1:
                zs.append(ndtri((u[None, :, j] * pj).clamp(1e-300, 1 - 1e-16)))
        out.append(p.mean(1))
    return torch.cat(out)


def chol(C):
    import torch
    L, info = torch.linalg.cholesky_ex(C)
    if bool((info != 0).any()):
        eye = torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
        L = torch.linalg.cholesky(C + 1e-10 * C.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye)
    return L


def shares(TH, grp, u, tvar):
    """Population shares (nb, nl, K1) of every alternative for theta rows TH (nb, 2P+3)."""
    import torch

    import ghk_gpu
    X, S, Md = grp
    P = X.shape[-1]
    b, w2 = TH[:, :P], (2 * TH[:, P:2 * P]).exp()
    sig2 = ghk_gpu.sig2_of(TH[:, 2 * P:], tvar=tvar)
    mu = torch.einsum("lkp,bp->blk", X, b)
    Cov = torch.einsum("lkp,bp,lmp->blkm", X, w2, X) + torch.einsum("ba,alkm->blkm", sig2, S)
    a = -torch.einsum("jik,blk->blji", Md, mu)
    C = torch.einsum("jik,blkm,jnm->bljin", Md, Cov, Md)
    nb, nl, K1, K = a.shape
    p = ghk_p(a.reshape(-1, K), chol(C.reshape(-1, K, K)), u).reshape(nb, nl, K1)
    return p / p.sum(-1, keepdim=True)


def theta_chunk(nl, K1, M, budget=1.5e9):
    return max(1, int(budget / (nl * K1 * M * (K1 + 3) * 8)))


def owned_shares(TH, grp, u, tvar):
    """Owned kept shares (nb, nl, k); theta rows chunked."""
    import torch
    nl, K1 = grp[0].shape[:2]
    c = theta_chunk(nl, K1, u.shape[0])
    return torch.cat([shares(TH[i:i + c], grp, u, tvar)[..., 3:K1 - 1] for i in range(0, len(TH), c)])


def delta(th, grp, u, tvar, V):
    """s-hat (nl, k), Jacobian by central differences (nl, k, nth), Sigma_s (nl, k, k)."""
    import torch
    nth = th.numel()
    E = torch.eye(nth, dtype=th.dtype, device=th.device) * FD_H
    TH = torch.cat([th[None], th + E, th - E])
    s = owned_shares(TH, grp, u, tvar)
    J = ((s[1:nth + 1] - s[nth + 1:]) / (2 * FD_H)).permute(1, 2, 0)
    return s[0], J, J @ V @ J.transpose(1, 2)


def sync(dev):
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()


def orthant_arms(a, S, dev, u_e, gp, macml):
    """P(X <= a), X ~ N(0, S), rows (n, d): D engine, E GHK-QMC (mean, se over scrambles), F OVUS, C product.
    d = 1: exact for every arm. Returns dict of numpy arrays and per-arm seconds."""
    import torch
    from torch.special import ndtr
    n, d = a.shape
    at, St = torch.as_tensor(a, device=dev), torch.as_tensor(S, device=dev)
    sd = St.diagonal(dim1=1, dim2=2).sqrt()
    out, sec = {}, {}

    def timed(name, fn):
        sync(dev)
        t0 = time.perf_counter()
        r = fn()
        sync(dev)
        sec[name] = time.perf_counter() - t0
        return r

    out["C"] = timed("C", lambda: ndtr(at / sd).prod(1).cpu().numpy())
    if d == 1:
        for k in "DEF":
            out[k] = out["C"].copy()
            sec[k] = 0.0
        out["E_se"] = np.zeros(n)
        return out, sec
    out["D"] = timed("D", lambda: gp.orthant_cdf_batch(a, S, dev, resolution="high"))
    L = chol(St)

    def E():
        P = torch.stack([ghk_p(at, L, u) for u in u_e])
        return P.mean(0).cpu().numpy(), (P.std(0) / np.sqrt(len(u_e))).cpu().numpy()
    out["E"], out["E_se"] = timed("E", E)

    def F():
        if d == 2:
            macml._patch_bvn()
            from pybhatlib.gradmvn._mvncd_torch import bvn_cdf_torch
            rho = (St[:, 0, 1] / (sd[:, 0] * sd[:, 1])).clamp(-0.9999, 0.9999)
            return bvn_cdf_torch(at[:, 0] / sd[:, 0], at[:, 1] / sd[:, 1], rho).cpu().numpy()
        return macml.ovus_log(at, St).exp().cpu().numpy()
    out["F"] = timed("F", F)
    return out, sec


def e_draws(d, M, reps, dev):
    import torch
    return [torch.quasirandom.SobolEngine(max(d - 1, 1), scramble=True, seed=7000 + 31 * d + r)
            .draw(M, dtype=torch.float64).to(dev) for r in range(reps)]


def dataset(job, dev, sizes):
    """Everything for one dataset on device dev. job: theta, th_true, seed, tvar, Xd, G, nll_t5, grps, t_grid."""
    import torch

    import ghk_gpu
    import macml
    import modal_gp_api as gp

    f64 = torch.float64
    T = lambda x: torch.as_tensor(x, dtype=f64, device=dev)   # noqa: E731
    tv = job["tvar"]
    th = T(job["theta"])
    res = dict(rep=job["rep"], truth=job["truth"], N=job["N"], sizes=sizes, seconds={})

    # 1. sandwich V at the stored fit, with the fit's own draws
    sync(dev)
    t0 = time.time()
    Xd, G = T(job["Xd"]), T(job["G"])
    N, Tt, K, _ = Xd.shape
    w = ghk_gpu.draws(N, Tt * K, sizes["M_V"], job["seed"], dev)
    with torch.no_grad():
        nll = sum(float(-ghk_gpu._ll_rows(th.expand(min(lo + 100, N) - lo, -1), Xd[lo:lo + 100], G[:, lo:lo + 100],
                                          0.0, w(lo, min(lo + 100, N)), tv).sum()) for lo in range(0, N, 100))
    V, _ = ghk_gpu.sandwich(th, Xd, G, 0.0, w, tv, 100)
    sync(dev)
    res["seconds"]["V"] = time.time() - t0
    res["nll_at_theta"], res["nll_t5"] = nll, job["nll_t5"]
    res["V"] = V.cpu().numpy().tolist()
    del Xd, G
    ev, Q = torch.linalg.eigh(V)
    res["V_min_eig"] = float(ev.min())
    Vh = Q * ev.clamp_min(0).sqrt()                       # V = Vh Vh'

    u_s = torch.quasirandom.SobolEngine(9, scramble=True, seed=11).draw(sizes["M_S"], dtype=f64).to(dev)
    u_t = torch.quasirandom.SobolEngine(9, scramble=True, seed=12).draw(sizes["M_T"], dtype=f64).to(dev)
    gen = torch.Generator(device=dev).manual_seed(job["seed"] + 77)
    Z = torch.randn(sizes["B"], th.numel(), generator=gen, dtype=f64, device=dev)
    TH_R = th + Z @ Vh.T
    per_line = {}
    t_delta = t_R = t_truth = 0.0
    for k, g in job["grps"].items():
        grp = tuple(T(x) for x in g["arrays"])
        K1 = grp[0].shape[1]
        uk_s, uk_t = u_s[:, :K1 - 2], u_t[:, :K1 - 2]
        sync(dev)
        t0 = time.time()
        s, J, Sig = delta(th, grp, uk_s, tv, V)
        sync(dev)
        t_delta += time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            sR = owned_shares(TH_R, grp, uk_s, tv)                     # (B, nl, k)
        sync(dev)
        t_R_shares = time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            s_true = owned_shares(T(job["th_true"])[None], grp, uk_t, tv)[0]
        sync(dev)
        t_truth += time.time() - t0
        t0 = time.time()
        R = {}
        for t in job["t_grid"]:
            ex = sR >= t
            R[t] = dict(joint=ex.all(-1).double().mean(0).cpu().numpy(), marg=ex.double().mean(0).cpu().numpy())
        mR = sR.mean(0)
        cR = torch.einsum("bli,blj->lij", sR - mR, sR - mR) / (sR.shape[0] - 1)
        sync(dev)
        t_R += t_R_shares + (time.time() - t0) / len(job["t_grid"])
        for n_, li in enumerate(g["idx"]):
            per_line[li] = dict(s_hat=s[n_].cpu().numpy(), Sig=Sig[n_].cpu().numpy(), s_true=s_true[n_].cpu().numpy(),
                                R_mean=mR[n_].cpu().numpy(), R_cov=cR[n_].cpu().numpy(),
                                R={t: dict(joint=float(R[t]["joint"][n_]), marg=R[t]["marg"][n_]) for t in R})
        del sR
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    res["seconds"].update(delta=t_delta, R=t_R, truth=t_truth)

    # 3. orthant arms per d, per t
    arm_sec = {k: 0.0 for k in "CDEF"}
    by_d = {}
    for li, r in per_line.items():
        by_d.setdefault(len(r["s_hat"]), []).append(li)
    for d, idx in by_d.items():
        u_e = e_draws(d, sizes["M_E"], sizes["E_REPS"], dev)
        S = np.stack([per_line[i]["Sig"] for i in idx])
        orthant_arms(np.stack([per_line[i]["s_hat"] for i in idx]) - job["t_grid"][0], S, dev, u_e, gp, macml)  # warm-up
        for t in job["t_grid"]:
            a = np.stack([per_line[i]["s_hat"] for i in idx]) - t
            P, sec = orthant_arms(a, S, dev, u_e, gp, macml)
            for k in arm_sec:
                arm_sec[k] += sec[k] / len(job["t_grid"])
            for n_, i in enumerate(idx):
                per_line[i].setdefault("P", {})[t] = {k: float(v[n_]) for k, v in P.items()}
    res["seconds"]["orthant"] = arm_sec
    res["lines"] = per_line
    res["gpu"] = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    return res


def plackett_terms(z, Rm):
    """Conditional orthants for dP/dz_i and dP/drho_ij of P = Phi_d(z; Rm), standardised.
    Returns (singles: a (d, d-1), C (d, d-1, d-1)), (pairs: list (i, j), a (np, d-2), C)."""
    d = len(z)
    a1, C1 = [], []
    for i in range(d):
        o = [k for k in range(d) if k != i]
        r = Rm[o, i]
        a1.append(z[o] - r * z[i])
        C1.append(Rm[np.ix_(o, o)] - np.outer(r, r))
    pr, a2, C2 = [], [], []
    for i, j in combinations(range(d), 2):
        o = [k for k in range(d) if k not in (i, j)]
        Qi = np.linalg.inv(Rm[np.ix_([i, j], [i, j])])
        B = Rm[np.ix_(o, [i, j])]
        pr.append((i, j))
        a2.append(z[o] - B @ Qi @ z[[i, j]])
        C2.append(Rm[np.ix_(o, o)] - B @ Qi @ B.T)
    return (np.array(a1), np.array(C1)), (pr, np.array(a2), np.array(C2))


def phi2(x, y, r):
    q = (x * x - 2 * r * x * y + y * y) / (1 - r * r)
    return np.exp(-q / 2) / (2 * np.pi * np.sqrt(1 - r * r))


def price_probe(th, V, tv, probe, dev, sizes):
    """All six kept; A's price on a fine grid (and +-h). P by D, E, F; central FD vs the Plackett gradient
    dP/dp = sum_i phi(z_i) P_{d-1}(cond i) dz_i/dp + sum_{i<j} phi2(z_i, z_j; rho_ij) P_{d-2}(cond ij) drho_ij/dp,
    the conditional orthants by the same integrator. z, rho derivatives by central differences of the share
    model (smooth: fixed draws). F: OVUS order frozen at grid point freeze_at, per orthant."""
    import torch
    from scipy.stats import norm

    import macml
    import modal_gp_api as gp
    f64 = torch.float64
    t, h, frz = probe["t"], probe["h"], probe["freeze_at"]
    ps = np.array(probe["prices"])
    n = len(ps)
    allp = np.concatenate([ps, ps + h, ps - h])
    sub = tuple(range(len(OWNED)))
    arrays = probe.get("arrays") or group_arrays(np.stack([line_prods(sub, p) for p in allp]))   # host builds them
    grp = tuple(torch.as_tensor(x, dtype=f64, device=dev) for x in arrays)
    K1 = grp[0].shape[1]
    u_s = torch.quasirandom.SobolEngine(9, scramble=True, seed=11).draw(sizes["M_S"], dtype=f64).to(dev)
    s, _, Sig = delta(th, grp, u_s[:, :K1 - 2], tv, V)
    s, Sig = s.cpu().numpy(), Sig.cpu().numpy()
    sd = np.sqrt(np.diagonal(Sig, axis1=1, axis2=2))
    z = (s - t) / sd
    Rm = Sig / (sd[:, :, None] * sd[:, None, :])
    d = z.shape[1]
    iu = np.triu_indices(d, 1)
    dz = (z[n:2 * n] - z[2 * n:]) / (2 * h)
    dR = (Rm[n:2 * n] - Rm[2 * n:]) / (2 * h)
    terms = [plackett_terms(z[k], Rm[k]) for k in range(n)]
    pairs = terms[0][1][0]
    A1 = np.concatenate([x[0][0] for x in terms]); C1 = np.concatenate([x[0][1] for x in terms])
    A2 = np.concatenate([x[1][1] for x in terms]); C2 = np.concatenate([x[1][2] for x in terms])
    u_e = {k: e_draws(k, sizes["M_E"], sizes["E_REPS"], dev) for k in (d - 2, d - 1, d)}

    def order_at(a_, C_):
        return macml.ovus_order(torch.as_tensor(a_, device=dev), torch.as_tensor(C_, device=dev))

    o_main = order_at(z[frz:frz + 1], Rm[frz:frz + 1]).expand(len(z), -1)
    o1 = order_at(*terms[frz][0]).repeat(n, 1)
    o2 = order_at(terms[frz][1][1], terms[frz][1][2]).repeat(n, 1)

    def integ(name, a_, C_, order):
        k = a_.shape[1]
        if name == "D":
            return gp.orthant_cdf_batch(a_, C_, dev, resolution="high")
        at, Ct = torch.as_tensor(a_, device=dev), torch.as_tensor(C_, device=dev)
        if name == "E":
            L = chol(Ct)
            return torch.stack([ghk_p(at, L, u) for u in u_e[k]]).mean(0).cpu().numpy()
        return macml.ovus_log(at, Ct, order).exp().cpu().numpy()

    out = dict(t=t, prices=ps.tolist(), h=h, s=s[:n].tolist(), z=z[:n].tolist(), rho=Rm[:n][:, iu[0], iu[1]].tolist(),
               dz=dz.tolist(), drho=dR[:, iu[0], iu[1]].tolist(), pairs=pairs)
    with torch.no_grad():
        for name in "DEF":
            Pz = integ(name, z, Rm, o_main)
            P1 = integ(name, A1, C1, o1).reshape(n, d)
            P2 = integ(name, A2, C2, o2).reshape(n, len(pairs))
            g = (norm.pdf(z[:n]) * P1 * dz).sum(1)
            for q, (i, j) in enumerate(pairs):
                g += phi2(z[:n, i], z[:n, j], Rm[:n, i, j]) * P2[:, q] * dR[:, i, j]
            out[name] = dict(P=Pz[:n].tolist(), fd_h=((Pz[n:2 * n] - Pz[2 * n:]) / (2 * h)).tolist(),
                             plackett=g.tolist())
    return out


@app.function(image=image, gpu="T4", timeout=3600)
def probe_gpu(theta, V, tv, probe, sizes):
    import torch
    dev = torch.device("cuda")
    t0 = time.time()
    r = price_probe(torch.as_tensor(theta, dtype=torch.float64, device=dev),
                    torch.as_tensor(V, dtype=torch.float64, device=dev), tv, probe, dev, sizes)
    r["seconds"] = time.time() - t0
    return r


@app.function(image=image, gpu="T4", timeout=6 * 3600)
def dataset_gpu(job, sizes):
    import torch
    return dataset(job, torch.device("cuda"), sizes)


# ---------------------------------------------------------------------------
# Host: jobs
# ---------------------------------------------------------------------------
def make_job(truth, N, rep, grps):
    import ghk_gpu
    import run
    r = json.loads((HERE.parent / "T1" / "out" / f"T5_{truth}_N{N}" / "results.json").read_text())
    rec = next(x for x in r if x["rep"] == rep)["mvp"]
    sig = np.sqrt(run.T_VAR * run.TRUTHS[truth])
    r_data = np.random.default_rng(np.random.SeedSequence([run.SEED, rep]).spawn(3)[0])
    d = run.make_data(r_data, N, 12, sig)
    Xd, G = ghk_gpu.panel_arrays(d["prods"], d["y"], run.task_X)
    return dict(truth=truth, N=N, rep=rep, theta=rec["theta"], th_true=run.theta_true(sig, "total").tolist(),
                seed=run.SEED + rep, tvar=float(run.T_VAR), Xd=Xd, G=G, nll_t5=-rec["ll"], grps=grps,
                t_grid=list(T_GRID))


def out_path(truth, N, rep):
    return HERE / "out" / f"{truth}_N{N}" / f"rep{rep}.json"


def save(res):
    p = out_path(res["truth"], res["N"], res["rep"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, default=lambda o: o.tolist()))


def check():
    """Local CPU wiring check: tiny sizes, k <= 3 lines only, N = 150 rep 0 subsampled."""
    import torch
    torch.set_num_threads(2)
    sel = [i for i, (s, _) in enumerate(lines()) if len(s) <= 3][::7]
    grps = groups(sel)
    job = make_job("G1", 150, 0, grps)
    job["Xd"], job["G"] = job["Xd"][:20], job["G"][:, :20]
    sizes = dict(M_V=64, M_S=128, M_T=256, M_E=256, E_REPS=2, B=50)
    t0 = time.time()
    res = dataset(job, torch.device("cpu"), sizes)
    print(f"check: {time.time() - t0:.1f} s; nll {res['nll_at_theta']:.2f}; V min eig {res['V_min_eig']:.2e}; "
          f"seconds {res['seconds']}")
    for i in list(res["lines"])[:5]:
        r = res["lines"][i]
        print(lines()[i], "s_hat", np.round(r["s_hat"], 4), "s_true", np.round(r["s_true"], 4),
              "P@.03", {k: round(v, 4) for k, v in r["P"][0.03].items()}, "R", round(r["R"][0.03]["joint"], 3))
    th = torch.as_tensor(job["theta"], dtype=torch.float64)
    V = torch.as_tensor(res["V"], dtype=torch.float64)
    pr = price_probe(th, V, job["tvar"], dict(PROBE, prices=PROBE["prices"][::10], freeze_at=2, t=0.03),
                     torch.device("cpu"), sizes)
    for k in "DEF":
        print("probe", k, np.round(pr[k]["P"], 4), "fd", np.round(pr[k]["fd_h"], 4), "pl", np.round(pr[k]["plackett"], 4))


def launch(cells):
    grps = groups()
    jobs = [make_job(truth, N, rep, grps) for truth, N, rep in cells if not out_path(truth, N, rep).exists()]
    print(f"{len(jobs)} dataset jobs")
    t0 = time.time()
    with modal.enable_output(), app.run():
        for res in dataset_gpu.starmap([(j, SIZES) for j in jobs], order_outputs=False):
            save(res)
            print(f"[{time.time() - t0:.0f} s] {res['truth']} N={res['N']} rep {res['rep']}: {res['seconds']}", flush=True)
    print(f"wall {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------
# Host: decisions and metrics
# ---------------------------------------------------------------------------
ARMS = "ABCDEFR"


def load():
    out = {}
    for f in sorted((HERE / "out").glob("G*_N*/rep*.json")):
        r = json.loads(f.read_text())
        out[(r["truth"], r["N"], r["rep"])] = r
    return out


def search(r, t):
    """Chosen line per arm, binding flag, truth-optimal feasible line, at threshold t."""
    L = lines()
    n = len(L)
    st, tk = [r["lines"][str(i)] for i in range(n)], str(t)
    price = [line_prices(*L[i]) for i in range(n)]
    pi_hat = np.array([np.sum((price[i] - COST) * np.array(st[i]["s_hat"])) for i in range(n)])
    pi_true = np.array([np.sum((price[i] - COST) * np.array(st[i]["s_true"])) for i in range(n)])
    ok = dict(A=np.ones(n, bool), B=np.array([min(x["s_hat"]) >= t for x in st]))
    for k in "CDEF":
        ok[k] = np.array([x["P"][tk][k] >= 1 - ALPHA for x in st])
    ok["R"] = np.array([x["R"][tk]["joint"] >= 1 - ALPHA for x in st])
    true_ok = np.array([min(x["s_true"]) >= t for x in st])
    pick = lambda f, pi: int(np.argmax(np.where(f, pi, -np.inf))) if f.any() else None   # noqa: E731
    ch = {k: pick(ok[k], pi_hat) for k in ARMS}
    best = pick(true_ok, pi_true)
    return dict(choice=ch, binds=not ok["R"][ch["A"]], binds_D=not ok["D"][ch["A"]], best=best,
                survive={k: bool(true_ok[c]) if c is not None else None for k, c in ch.items()},
                regret={k: float(100 * (pi_true[best] - pi_true[c]) / pi_true[best]) if c is not None else None
                        for k, c in ch.items()})


NAMES = dict(A="Unconstrained", B="Point", C="Independent", D="Joint-engine", E="Joint-GHK", F="Joint-OVUS",
             R="Nested MC")


def summarize_t(res, t):
    """Every metric and criterion at threshold t (user, 2026-10-11: score every t, no selection)."""
    from scipy.stats import norm
    L = lines()
    S = dict(t=t, cells={})
    allse = []
    surv_floor = (1 - ALPHA) - 2 * np.sqrt(ALPHA * (1 - ALPHA) / 20)
    for cell in sorted({k[:2] for k in res}):
        rs = [res[k] for k in sorted(k for k in res if k[:2] == cell)]
        se = [search(r, t) for r in rs]
        allse += se
        c = dict(reps=len(rs), unconstrained_fails=float(np.mean([x["binds"] for x in se])))
        c["agree_R"] = {k: float(np.mean([x["choice"][k] == x["choice"]["R"] for x in se])) for k in "ABCDEF"}
        c["survival"] = {k: float(np.mean([bool(x["survive"][k]) for x in se])) for k in ARMS}
        c["regret_pct"] = {k: float(np.mean([x["regret"][k] for x in se if x["regret"][k] is not None])) for k in ARMS}
        c["no_feasible_line"] = {k: int(sum(x["choice"][k] is None for x in se)) for k in ARMS}

        def paired(k1, k2, key):
            dd = np.array([float(x[key][k1] or 0) - float(x[key][k2] or 0) for x in se])
            return float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd)))
        c["D_minus_B_survival"] = paired("D", "B", "survive")
        c["D_minus_C_regret"] = paired("D", "C", "regret")
        dB, sB = c["D_minus_B_survival"]
        dC, sC = c["D_minus_C_regret"]
        c["crit2"] = dict(survival_ok=c["survival"]["D"] >= surv_floor, beats_point=dB > 2 * sB,
                          regret_ok=dC <= 2 * sC)
        c["crit2"]["pass"] = all(c["crit2"].values())
        c["crit2"]["kill"] = cell[1] == 600 and c["survival"]["D"] < surv_floor
        err, nrm_j, nrm_m, mixed, mcse = {}, {}, {}, [], []
        for r in rs:
            for i in range(len(L)):
                x = r["lines"][str(i)]
                d = len(x["s_hat"])
                Pr = x["R"][str(t)]["joint"]
                mcse.append(np.sqrt(max(Pr * (1 - Pr), 1e-12) / r["sizes"]["B"]))
                for k in "DEF":
                    err.setdefault(d, {}).setdefault(k, []).append(abs(x["P"][str(t)][k] - Pr))
                nrm_j.setdefault(d, []).append(abs(x["P"][str(t)]["E"] - Pr))
                sd = np.sqrt(np.diag(x["Sig"]))
                nrm_m.setdefault(d, []).extend(np.abs(norm.cdf((np.array(x["s_hat"]) - t) / sd)
                                                      - np.array(x["R"][str(t)]["marg"])).tolist())
                if d >= 2:
                    off = np.array(x["Sig"])[np.triu_indices(d, 1)]
                    mixed.append(bool((off > 0).any() and (off < 0).any()))
        c["P_err_vs_R"] = {d: {k: dict(median=float(np.median(v)), max=float(np.max(v))) for k, v in e.items()}
                           for d, e in sorted(err.items())}
        c["normality_joint"] = {d: dict(median=float(np.median(v)), max=float(np.max(v))) for d, v in sorted(nrm_j.items())}
        c["normality_marginal"] = {d: dict(median=float(np.median(v)), max=float(np.max(v))) for d, v in sorted(nrm_m.items())}
        c["R_mc_se_median"] = float(np.median(mcse))
        c["mixed_sign_rate"] = float(np.mean(mixed))
        sec = [r["seconds"] for r in rs]
        med = lambda f: float(np.median([f(x) for x in sec]))   # noqa: E731
        c["seconds"] = dict(V=med(lambda x: x["V"]), delta=med(lambda x: x["delta"]), R=med(lambda x: x["R"]),
                            **{k: med(lambda x, k=k: x["delta"] + x["orthant"][k]) for k in "CDEF"})
        c["crit3"] = {k: dict(agree=c["agree_R"][k], speedup=c["seconds"]["R"] / c["seconds"][k]) for k in "DEF"}
        S["cells"][f"{cell[0]} N={cell[1]}"] = c
    ne = lambda k1, k2: float(np.mean([x["choice"][k1] != x["choice"][k2] for x in allse]))   # noqa: E731
    S["D_ne_B"], S["D_ne_C"], S["R_ne_B"] = ne("D", "B"), ne("D", "C"), ne("R", "B")
    S["crit1"] = dict(pass_=S["D_ne_B"] >= 0.15 and S["D_ne_C"] >= 0.10, kill=S["D_ne_C"] < 0.05)
    S["crit2"] = dict(pass_=all(c["crit2"]["pass"] for c in S["cells"].values()),
                      kill=any(c["crit2"]["kill"] for c in S["cells"].values()))
    agree = {k: float(np.mean([x["choice"][k] == x["choice"]["R"] for x in allse])) for k in "DEF"}
    spd = {k: float(np.median([c["crit3"][k]["speedup"] for c in S["cells"].values()])) for k in "DEF"}
    S["crit3"] = dict(agree=agree, speedup=spd, pass_=any(agree[k] >= 0.95 and spd[k] >= 10 for k in "DEF"))
    S["pass"] = S["crit1"]["pass_"] and S["crit2"]["pass_"]
    return S


def summarize():
    res = load()
    out = dict(alpha=ALPHA, n_datasets=len(res), names=NAMES, by_t=[summarize_t(res, t) for t in T_GRID])
    npass = sum(x["pass"] for x in out["by_t"])
    out["overall"] = dict(t_pass=npass, of=len(T_GRID), pass_=npass > len(T_GRID) / 2,
                          crit2_fail_t=[x["t"] for x in out["by_t"] if not x["crit2"]["pass_"]],
                          crit2_kill_t=[x["t"] for x in out["by_t"] if x["crit2"]["kill"]])
    (HERE / "out" / "summary.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()))
    A = "ABCDEFR"
    print("| t | P≠Pt | J≠Pt | J≠Ind | (1) | (2) | (3) agree D/E/F | speedup D/E/F |")
    print("|---|---|---|---|---|---|---|---|")
    for x in out["by_t"]:
        print(f"| {100 * x['t']:g}% | {x['R_ne_B']:.2f} | {x['D_ne_B']:.2f} | {x['D_ne_C']:.2f} | "
              f"{'PASS' if x['crit1']['pass_'] else 'KILL' if x['crit1']['kill'] else 'fail'} | "
              f"{'PASS' if x['crit2']['pass_'] else 'KILL' if x['crit2']['kill'] else 'fail'} | "
              + " / ".join(f"{x['crit3']['agree'][k]:.2f}" for k in "DEF") + " | "
              + " / ".join(f"{x['crit3']['speedup'][k]:.0f}x" for k in "DEF") + " |")
    print()
    print("| t | cell | survival " + " / ".join(A) + " | regret % " + " / ".join(A) + " | D−B surv (SE) | D−C regret (SE) |")
    print("|---|---|---|---|---|---|")
    for x in out["by_t"]:
        for cn, c in x["cells"].items():
            print(f"| {100 * x['t']:g}% | {cn} | " + " / ".join(f"{c['survival'][k]:.2f}" for k in A) + " | "
                  + " / ".join(f"{c['regret_pct'][k]:.1f}" for k in A)
                  + f" | {c['D_minus_B_survival'][0]:+.2f} ({c['D_minus_B_survival'][1]:.2f}) | "
                  f"{c['D_minus_C_regret'][0]:+.2f} ({c['D_minus_C_regret'][1]:.2f}) |")
    print(json.dumps(out["overall"]))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "pilot":
        launch([("G1", 600, 0)])
    elif cmd == "run":
        launch([(g, n, r) for g in ("G1", "G2") for n in (150, 600) for r in range(20)])
    elif cmd == "summarize":
        summarize()
    elif cmd == "price-probe":     # all six kept, G1 N = 600 rep 0, every t (report only)
        job = make_job("G1", 600, 0, {})
        V = json.loads(out_path("G1", 600, 0).read_text())["V"]
        ps = np.array(PROBE["prices"])
        arrays = group_arrays(np.stack([line_prods(tuple(range(len(OWNED))), p)
                                        for p in np.concatenate([ps, ps + PROBE["h"], ps - PROBE["h"]])]))
        with modal.enable_output(), app.run():
            pr = list(probe_gpu.starmap([(job["theta"], V, job["tvar"], dict(PROBE, t=t, arrays=arrays), SIZES) for t in T_GRID]))
        (HERE / "out" / "price_probe.json").write_text(json.dumps(pr, default=lambda o: o.tolist()))
