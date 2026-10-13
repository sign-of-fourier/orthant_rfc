"""T13: cardinality-constrained line (6 of 20) with joint survival (PLAN.md, T13).

Fixed model: T4's recovery-gate fit (J = 20, MVP-G 512x256, N = 1,000) and its sandwich covariance.
Every line = 6 kept SKUs + "none"; population shares are exact 6-dim orthants (normal tastes and errors),
by GHK with fixed draws. Log-share delta method: Sigma = J V J', J = dlog s / dtheta (central differences).

    python analysis/T13/t13.py step0          # one T4: delta step + constraint probabilities, all lines
    python analysis/T13/t13.py step0-report   # Step 0 metrics (local)
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
    for d in ("T4", "T9", "T11"):
        sys.path.insert(0, str(HERE.parent / d))
    sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))   # engine source, sibling checkout

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy", "pybhatlib==0.4.0", "pydantic", "fastapi")
         .add_local_python_source("modal_gp_api", "macml", "t11"))
app = modal.App("orthant-rfc-t13")

J, K = 20, 6
T_GRID = (0.065, 0.070, 0.075)
ALPHAS = (0.10, 0.05, 0.01)
FLOORS = (0.90, 0.95, 0.98)
COST = 3.00
FD_H = 1e-4
SIZES = dict(M_S=1024, M_E=2 ** 14, E_REPS=4)
FIT = HERE.parent / "T4" / "out" / "gate_g" / "fit_512x256.json"
OUT = HERE / "out"


# ---------------------------------------------------------------------------
# Host: lines and design arrays
# ---------------------------------------------------------------------------
def lines():
    return np.array(list(combinations(range(J), K)))


def design():
    """X (n, K+1, P), S (4, n, K+1, K+1): brand / flavor / pack same-level and SKU-own indicators, none last.
    none's error variance is T_VAR (fixed, not a parameter), carried separately."""
    import t4
    L = lines()
    lev = t4.ALL[t4.shelf(J)]
    pr = t4.base_prices(J)
    X = t4.task_X(lev, pr)[np.c_[L, np.full(len(L), J)]]
    n = len(L)
    S = np.zeros((4, n, K + 1, K + 1))
    for a in range(3):
        lv = lev[L, a]
        S[a, :, :K, :K] = lv[:, :, None] == lv[:, None, :]
    S[3, :, :K, :K] = np.eye(K)
    return X, S, lev, pr


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def sig2_of(u, tvar):
    import torch
    p = torch.softmax(torch.cat([u, torch.zeros_like(u[..., :1])], -1), -1)
    return tvar * (p * (1 - 4e-6) + 1e-6)          # as T4's mvp_gpu.sig2_of


def shares(TH, X, S, u, tvar):
    """Population shares (nb, n, K+1) for theta rows TH (nb, 25) over lines X (n, K+1, P)."""
    import torch

    import t11
    P = X.shape[-1]
    K1 = X.shape[1]
    b, w2 = TH[:, :P], (2 * TH[:, P:2 * P]).exp()
    s2 = sig2_of(TH[:, 2 * P:], tvar)
    mu = torch.einsum("lkp,bp->blk", X, b)
    Cov = torch.einsum("lkp,bp,lmp->blkm", X, w2, X) + torch.einsum("ba,alkm->blkm", s2, S)
    Cov[..., K1 - 1, K1 - 1] += tvar
    Md = torch.zeros(K1, K1 - 1, K1, dtype=X.dtype, device=X.device)
    for j in range(K1):
        oth = [k for k in range(K1) if k != j]
        Md[j, torch.arange(K1 - 1), oth] = 1
        Md[j, :, j] = -1
    a = -torch.einsum("jik,blk->blji", Md, mu)
    C = torch.einsum("jik,blkm,jnm->bljin", Md, Cov, Md)
    nb, n = a.shape[:2]
    p = t11.ghk_p(a.reshape(-1, K1 - 1), t11.chol(C.reshape(-1, K1 - 1, K1 - 1)), u).reshape(nb, n, K1)
    return p / p.sum(-1, keepdim=True)


def delta_all(th, V, X, S, u, tvar, chunk=1500):
    """log s-hat (n, K+1), Sigma_log (n, K, K) of the kept SKUs; central differences in theta."""
    import torch
    nth = th.numel()
    E = torch.eye(nth, dtype=th.dtype, device=th.device) * FD_H
    TH = torch.cat([th[None], th + E, th - E])
    ls, Sig = [], []
    for lo in range(0, X.shape[0], chunk):
        s = shares(TH, X[lo:lo + chunk], S[:, lo:lo + chunk], u, tvar).log()
        Jm = ((s[1:nth + 1, :, :K] - s[nth + 1:, :, :K]) / (2 * FD_H)).permute(1, 2, 0)
        ls.append(s[0])
        Sig.append(Jm @ V @ Jm.transpose(1, 2))
    return torch.cat(ls), torch.cat(Sig)


def sync(dev):
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()


def step0(th, V, X, S, tvar, dev, sizes):
    import torch

    import macml
    import modal_gp_api as gp
    import t11
    f64 = torch.float64
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=f64, device=dev)   # noqa: E731
    th, V, X, S = T(th), T(V), T(X), T(S)
    u = torch.quasirandom.SobolEngine(K - 1, scramble=True, seed=13).draw(sizes["M_S"], dtype=f64).to(dev)
    sec = {}
    sync(dev)
    t0 = time.time()
    ls, Sig = delta_all(th, V, X, S, u, tvar)
    sync(dev)
    sec["delta"] = time.time() - t0
    out = dict(log_s=ls.cpu().numpy(), Sig=Sig.cpu().numpy(), P={}, seconds=sec)
    u_e = t11.e_draws(K, sizes["M_E"], sizes["E_REPS"], dev)
    Sig_np = out["Sig"]
    sd = np.sqrt(np.diagonal(Sig_np, axis1=1, axis2=2))
    n = len(Sig_np)
    t_arm = {k: 0.0 for k in ("ind", "ghk", "ovus", "engine")}
    for t in T_GRID:
        a = out["log_s"][:, :K] - np.log(t)
        P = {"ind": None}
        sync(dev)
        t0 = time.perf_counter()
        from scipy.special import ndtr
        P["ind"] = ndtr(a / sd).prod(1)
        t_arm["ind"] += time.perf_counter() - t0
        at, St = T(a), T(Sig_np)
        with torch.no_grad():
            sync(dev)
            t0 = time.perf_counter()
            L = t11.chol(St)
            G = torch.stack([t11.ghk_p(at, L, uu) for uu in u_e])
            P["ghk"], P["ghk_se"] = G.mean(0).cpu().numpy(), (G.std(0) / np.sqrt(len(u_e))).cpu().numpy()
            sync(dev)
            t_arm["ghk"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            P["ovus"] = np.concatenate([macml.ovus_log(at[i:i + 50_000], St[i:i + 50_000]).exp().cpu().numpy()
                                        for i in range(0, n, 50_000)])
            sync(dev)
            t_arm["ovus"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        P["engine"] = gp.orthant_cdf_batch(a, Sig_np, dev, resolution="high")
        sync(dev)
        t_arm["engine"] += time.perf_counter() - t0
        out["P"][t] = P
    sec["orthant_per_t"] = {k: v / len(T_GRID) for k, v in t_arm.items()}
    return out


@app.function(image=image, gpu="T4", timeout=3 * 3600)
def step0_gpu(th, V, X, S, tvar, sizes):
    import torch
    dev = torch.device("cuda")
    # warm-up (first-call costs out of the timings)
    step_w = step0(th, V, X[:200], S[:, :200], tvar, dev, dict(sizes, M_E=1024, E_REPS=1))
    del step_w
    r = step0(th, V, X, S, tvar, dev, sizes)
    r["gpu"] = torch.cuda.get_device_name()
    return r


# ---------------------------------------------------------------------------
# Host: Step 0 report
# ---------------------------------------------------------------------------
ARMS = ("point", "ind", "ghk", "ovus", "engine")
NAMES = dict(point="Point", ind="Independent", ghk="Joint-GHK", ovus="Joint-OVUS", engine="Joint-engine")


def load_step0():
    z = np.load(OUT / "step0.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def p_of(r, t, arm):
    a = r["log_s"][:, :K] - np.log(t)
    if arm == "point":
        return (a >= 0).all(1).astype(float)
    return r[f"P_{arm}_{t}"]


def choose(r, profit, t):
    """Per arm: (1) optimum per alpha, (2) optimum per floor. (2) for Point: among lines with every s-hat >= t
    and profit >= floor, max profit; if none, max min_j s-hat_j / t."""
    pmax = profit.max()
    out = {}
    for arm in ARMS:
        P = p_of(r, t, arm)
        for al in ALPHAS:
            ok = P >= 1 - al
            out[(arm, "1", al)] = int(np.argmax(np.where(ok, profit, -np.inf))) if ok.any() else None
        for fl in FLOORS:
            ok = profit >= fl * pmax
            if arm == "point":
                feas = ok & (P > 0)
                if feas.any():
                    out[(arm, "2", fl)] = int(np.argmax(np.where(feas, profit, -np.inf)))
                else:
                    m = (r["log_s"][:, :K] - np.log(t)).min(1)
                    out[(arm, "2", fl)] = int(np.argmax(np.where(ok, m, -np.inf)))
            else:
                out[(arm, "2", fl)] = int(np.argmax(np.where(ok, P, -np.inf)))
    return out


def pair_excess(a, Sig):
    """sum_{i<j} [P(miss_i & miss_j) - p_i p_j] split by sign of rho_ij; miss_i = {eps_i > a_i}."""
    from scipy.stats import multivariate_normal as mvn
    from scipy.stats import norm
    sd = np.sqrt(np.diag(Sig))
    z = a / sd
    R = Sig / np.outer(sd, sd)
    p = norm.sf(z)
    pos = neg = 0.0
    for i, j in combinations(range(K), 2):
        both = mvn(np.zeros(2), [[1, R[i, j]], [R[i, j], 1]]).cdf([-z[i], -z[j]])
        e = both - p[i] * p[j]
        if R[i, j] >= 0:
            pos += e
        else:
            neg += e
    return dict(pos=float(pos), neg=float(neg))


def step0_report():
    from scipy.stats import norm
    import t4
    r = load_step0()
    L = lines()
    lev = t4.ALL[t4.shelf(J)]
    pr = t4.base_prices(J)
    s = np.exp(r["log_s"])
    profit = ((pr[L] - COST) * s[:, :K]).sum(1)
    rep = dict(unconstrained=dict(line=L[profit.argmax()].tolist(), profit=float(profit.max())), cells={})
    for t in T_GRID:
        ch = choose(r, profit, t)
        a_all = r["log_s"][:, :K] - np.log(t)
        sd_all = np.sqrt(np.diagonal(r["Sig"], axis1=1, axis2=2))
        pm_all = norm.sf(a_all / sd_all)
        for key, li in ch.items():
            arm, form, par = key
            if li is None:
                rep["cells"][f"t={t} ({form}) {par} {arm}"] = None
                continue
            risky = np.where(pm_all[li] > 0.01)[0]
            shared = {nm: sorted({int(lev[L[li][i], ai]) for i in risky
                                  if sum(lev[L[li][j], ai] == lev[L[li][i], ai] for j in risky) > 1})
                      for ai, nm in enumerate(("brand", "flavor", "pack"))}
            rep["cells"][f"t={t} ({form}) {par} {arm}"] = dict(
                line=L[li].tolist(), profit=float(profit[li]), profit_frac=float(profit[li] / profit.max()),
                P={k: float(p_of(r, t, k)[li]) for k in ARMS}, n_pmiss_gt_1pct=int(len(risky)),
                pmiss=pm_all[li].round(4).tolist(), shared_levels_among_risky=shared,
                pair_excess=pair_excess(a_all[li], r["Sig"][li]))
        for al in ALPHAS:
            Pi = p_of(r, t, "ind") >= 1 - al
            rep.setdefault("feas_disagree", {})[f"t={t} a={al}"] = {
                k: float(np.mean((p_of(r, t, k) >= 1 - al) != Pi)) for k in ("ghk", "ovus", "engine")}
            rep.setdefault("same_line_1", {})[f"t={t} a={al}"] = {
                k: ch[(k, "1", al)] == ch[("ind", "1", al)] for k in ("ghk", "ovus", "engine")}
        for fl in FLOORS:
            ok = np.where(profit >= fl * profit.max())[0]
            rk = {}
            for k in ("ind", "ghk", "ovus", "engine"):
                rk[k] = ok[np.argsort(-p_of(r, t, k)[ok], kind="stable")][:20]
            rep.setdefault("top20", {})[f"t={t} f={fl}"] = {
                k: dict(positions_changed=int((rk[k] != rk["ind"]).sum()),
                        not_in_ind_top20=int(len(set(rk[k]) - set(rk["ind"]))))
                for k in ("ghk", "ovus", "engine")} | {"n_lines_above_floor": int(len(ok))}
            rep.setdefault("same_line_2", {})[f"t={t} f={fl}"] = {
                k: ch[(k, "2", fl)] == ch[("ind", "2", fl)] for k in ("ghk", "ovus", "engine")}
    rep["seconds"] = r["seconds"].item()
    rep["share_check_vs_engine_cpu"] = r.get("share_check")
    (OUT / "step0_report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.item()))
    return rep


def run_step0():
    import t4  # noqa: F401
    fit = json.loads(FIT.read_text())
    X, S, _, _ = design()
    t0 = time.time()
    with modal.enable_output(), app.run():
        r = step0_gpu.remote(fit["theta"], fit["cov_sandwich"], X, S, float(np.pi ** 2 / 6), SIZES)
    OUT.mkdir(parents=True, exist_ok=True)
    flat = dict(log_s=r["log_s"], Sig=r["Sig"], seconds=np.array(r["seconds"], dtype=object), gpu=r["gpu"])
    for t, P in r["P"].items():
        for k, v in P.items():
            flat[f"P_{k}_{t}"] = v
    np.savez_compressed(OUT / "step0.npz", **flat)
    print(f"wall {time.time() - t0:.0f} s; {r['seconds']}; {r['gpu']}")


if __name__ == "__main__":
    {"step0": run_step0, "step0-report": lambda: print(json.dumps(step0_report(), indent=1, default=str))}[sys.argv[1]]()
