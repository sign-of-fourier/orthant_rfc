"""T15: engine pipeline on the shelf (TEST.md; design: DESIGN.md).

Fixed model: T4's recovery-gate fit (J = 20, MVP-G 512x256). Population shares with tastes folded into the
covariance: for alternatives A (SKUs + none) at prices p, U ~ N(X b, X diag(w^2) X' + sum_a s_a S_a + s_own I
[+ T_VAR on none]); the share of j is one orthant of the |A| - 1 differences U_k - U_j (dim = shelf size).
Each target share is its own orthant (no renormalisation).

    python t15.py probe        # Step 0 phase A: wiring + throughput on one T4 and one H100, cost projection
    python t15.py probe-local  # the same on CPU at toy sizes (wiring only)
"""
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
if len(HERE.parents) > 2:                                      # host only (not in the Modal container)
    sys.path.insert(0, str(HERE.parent / "T4"))
    sys.path.insert(0, str(HERE.parent / "T9"))
    sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))   # engine source, sibling checkout

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy", "pydantic", "fastapi", "pybhatlib==0.4.0")
         .add_local_python_source("modal_gp_api", "macml"))
app = modal.App("orthant-rfc-t15")

SEED = 20261016
FIT = HERE.parent / "T4" / "out" / "gate_g" / "fit_512x256.json"
OUT = HERE / "out"
MATERIAL = 0.005                                  # 0.5% of max profit
COST_PACK = np.array([2.60, 3.00, 3.40])
PRICES5 = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
PRICES9 = np.arange(3.5, 5.51, 0.25)
GHK_M = (64, 128, 256, 512, 1024, 2048, 4096, 16384)
SIM_N = tuple(2 ** e for e in range(10, 19))
REF_M = 2 ** 16
GPU_PRICE = {"T4": 0.59, "H100": 3.95}            # $/h, Modal list (analysis/T10/TEST.md probe)


# ---------------------------------------------------------------------------
# Host: the model and the decision instances (numpy; shipped to the container as plain arrays)
# ---------------------------------------------------------------------------
def model():
    """Grid arrays for all 60 SKUs + none (row 60), and theta-hat / theta-true (T4 parameterisation)."""
    import t4
    P = t4.P
    lev = t4.ALL                                                        # (60, 3)
    Xb = np.zeros((61, P))
    Xb[:60] = t4.code(lev, np.full(60, t4.PRICE_REF))                   # price column 0 at the reference price
    Xb[60, P - 1] = 1
    SA = np.zeros((3, 61, 61))
    for a in range(3):
        SA[a, :60, :60] = lev[:, a, None] == lev[None, :, a]
    off = np.cumsum((0,) + t4.LEVELS[:-1])
    mem = np.zeros((61, t4.NLEV))
    for a in range(3):
        mem[np.arange(60), off[a] + lev[:, a]] = 1
    s2 = t4.sig_of("S1") ** 2 / t4.T_VAR
    th_true = np.r_[t4.b_true(20), np.log(t4.W_TRUE), np.log(s2[:3] / s2[3])]
    th_hat = np.array(json.loads(FIT.read_text())["theta"])
    return dict(Xb=Xb, SA=SA, mem=mem, levels=np.array(t4.LEVELS), lev=lev, P=P, pcol=P - 2, ref=t4.PRICE_REF,
                tvar=t4.T_VAR, th_hat=th_hat, th_true=th_true,
                b_true=t4.b_true(20), w_true=t4.W_TRUE.copy())


def instances(J=20):
    """12 decision instances. The firm is brand 0: 7 SKUs on T4's J = 20 shelf (T4 counted one of them, SKU 10, as a
    competitor) + 5 unshelved brand-0 grid SKUs as extension candidates. Competitors (13) at T4's base prices."""
    import t4
    shelf = t4.shelf(J)
    bp = dict(zip(shelf.tolist(), t4.base_prices(J).tolist()))
    own = [int(i) for i in shelf if t4.ALL[i, 0] == 0]
    comp = [int(i) for i in shelf if t4.ALL[i, 0] != 0]
    ext = [i for i in range(12) if i not in own]
    cost = COST_PACK[t4.ALL[:, 2]]
    out = []
    for k in (6, 8, 10):
        out.append(dict(name=f"D1_k{k}" + ("" if J == 20 else f"_J{J}"), dec="D1", k=k, shelf=shelf.tolist(), bp=[bp[i] for i in shelf],
                        cost=cost.tolist(), F=0.0))
    for tag, cm, fm in (("base", 1.0, 1.0), ("cost+20", 1.2, 1.0), ("cost-20", 0.8, 1.0)):
        out.append(dict(name=f"D2_{tag}" + ("" if J == 20 else f"_J{J}"), dec="D2", own=own, comp=comp, bp_comp=[bp[i] for i in comp],
                        grid=PRICES9.tolist(), cost=(cost * cm).tolist(), F=0.0))
    for d in ("D3", "D4"):
        for tag, cm, fm in (("base", 1.0, 1.0), ("cost+20", 1.2, 1.0), ("F2", 1.0, 2.0)):
            out.append(dict(name=f"{d}_{tag}" + ("" if J == 20 else f"_J{J}"), dec=d, own=own, ext=ext, comp=comp, bp_own=[bp[i] for i in own],
                            bp_comp=[bp[i] for i in comp], grid=PRICES5.tolist(), cost=(cost * cm).tolist(),
                            F_mult=fm))
    return out


def space_size(inst):
    from math import comb
    d = inst["dec"]
    if d == "D1":
        return comb(len(inst["shelf"]), inst["k"])
    if d == "D2":
        return len(inst["grid"]) ** len(inst["own"])
    if d == "D3":
        return (len(inst["grid"]) + 1) ** len(inst["ext"])
    return (len(inst["grid"]) + 1) ** (len(inst["own"]) + len(inst["ext"]))


# ---------------------------------------------------------------------------
# Device: candidates -> alternative sets -> orthants / simulated first choices
# ---------------------------------------------------------------------------
class Shelf:
    """Everything that turns candidates of one instance into (alternatives, prices, targets, margins)."""

    def __init__(self, m, inst, dev):
        import torch
        self.t, self.dev, self.inst, self.dec = torch, dev, inst, inst["dec"]
        f = dict(dtype=torch.float64, device=dev)
        self.Xb, self.SA = torch.tensor(m["Xb"], **f), torch.tensor(m["SA"], **f)
        self.pcol, self.ref, self.tvar = m["pcol"], m["ref"], m["tvar"]
        self.cost = torch.tensor(inst["cost"] + [0.0], **f)               # none last (index 60)
        L = lambda k: torch.tensor(inst[k], dtype=torch.long, device=dev)  # noqa: E731
        T = lambda k: torch.tensor(inst[k], **f)                           # noqa: E731
        if self.dec == "D1":
            self.shelf, self.bp = L("shelf"), T("bp")
        else:
            self.own, self.comp, self.bpc, self.grid = L("own"), L("comp"), T("bp_comp"), T("grid")
            if self.dec != "D2":
                self.ext, self.bpo = L("ext"), T("bp_own")
        self.F = 0.0

    def sample(self, n, gen):
        t, dev = self.t, self.dev
        if self.dec == "D1":
            return t.rand(n, len(self.shelf), generator=gen, device=dev).argsort(1)[:, :self.inst["k"]].sort(1)[0]
        if self.dec == "D2":
            return t.randint(0, len(self.grid), (n, len(self.own)), generator=gen, device=dev)
        slots = len(self.ext) if self.dec == "D3" else len(self.own) + len(self.ext)
        return t.randint(0, len(self.grid) + 1, (n, slots), generator=gen, device=dev)

    def expand(self, cand):
        """-> list of groups (alts (C, n) grid index, none = 60, last; prices (C, n); m targets = positions 0..m-1;
        margin (C, m); fixed (C,) count of fixed-cost SKUs; idx (C,) rows of cand)."""
        t = self.t
        C = cand.shape[0]
        none = t.full((C, 1), 60, dtype=t.long, device=self.dev)
        if self.dec == "D1":
            alts = t.cat([self.shelf[cand], none], 1)
            pr = t.cat([self.bp[cand], t.zeros(C, 1, dtype=self.bp.dtype, device=self.dev)], 1)
            m = cand.shape[1]
            return [self._group(alts, pr, m, t.zeros(C, dtype=pr.dtype, device=self.dev), t.arange(C, device=self.dev))]
        if self.dec == "D2":
            alts = t.cat([self.own.expand(C, -1), self.comp.expand(C, -1), none], 1)
            pr = t.cat([self.grid[cand], self.bpc.expand(C, -1), t.zeros(C, 1, dtype=self.grid.dtype, device=self.dev)], 1)
            return [self._group(alts, pr, len(self.own), t.zeros(C, dtype=pr.dtype, device=self.dev),
                                t.arange(C, device=self.dev))]
        if self.dec == "D3":
            slot_sku = self.ext
            base_sku, base_pr = self.own, self.bpo
        else:
            slot_sku = t.cat([self.own, self.ext])
            base_sku, base_pr = self.own[:0], self.bpo[:0]
        on = cand > 0
        non = on.sum(1)
        groups = []
        for k in non.unique().tolist():
            idx = (non == k).nonzero()[:, 0]
            c = cand[idx]
            order = (c == 0).to(t.int8).argsort(dim=1, stable=True)[:, :k]           # on slots first
            sku = slot_sku[order]
            pr_on = self.grid[(c.gather(1, order) - 1).clamp(min=0)]
            n = len(idx)
            alts = t.cat([base_sku.expand(n, -1), sku, self.comp.expand(n, -1), none[:n]], 1)
            pr = t.cat([base_pr.expand(n, -1), pr_on, self.bpc.expand(n, -1),
                        t.zeros(n, 1, dtype=pr_on.dtype, device=self.dev)], 1)
            groups.append(self._group(alts, pr, len(base_sku) + k, t.full((n,), float(k), dtype=pr.dtype,
                                                                             device=self.dev), idx))
        return groups

    def _group(self, alts, pr, m, fixed, idx):
        margin = (pr[:, :m] - self.cost[alts[:, :m]])
        return dict(alts=alts, pr=pr, m=m, margin=margin, fixed=fixed, idx=idx)

    def moments(self, g, th):
        """mu (C, n), Cov (C, n, n) at theta th (2P + 3,)."""
        t = self.t
        P = self.Xb.shape[1]
        X = self.Xb[g["alts"]].clone()
        X[..., self.pcol] = t.where(g["alts"] == 60, 0.0, g["pr"] - self.ref)
        b, w2 = th[:P], (2 * th[P:2 * P]).exp()
        u = th[2 * P:]
        s = t.softmax(t.cat([u, u.new_zeros(1)]), 0)
        s2 = self.tvar * (s * (1 - 4e-6) + 1e-6)                              # as T4's mvp_gpu.sig2_of
        mu = X @ b
        Cov = t.einsum("cnp,p,cmp->cnm", X, w2, X)
        a = g["alts"]
        for k in range(3):
            Cov = Cov + s2[k] * self.SA[k][a[:, :, None], a[:, None, :]]
        sku = (a != 60).to(Cov.dtype)
        Cov = Cov + t.diag_embed(sku * s2[3] + (1 - sku) * self.tvar)
        return mu, Cov

    def orthants(self, g, th):
        """Upper bounds a (C*m, n-1) and covariances (C*m, n-1, n-1): share of target j = P(D <= a), D = U_k - U_j."""
        t = self.t
        mu, Cov = self.moments(g, th)
        C, n = mu.shape
        m = g["m"]
        ar = t.arange(n, device=self.dev)
        oth = t.stack([t.cat([ar[:j], ar[j + 1:]]) for j in range(m)])            # (m, n-1)
        j = ar[:m]
        mo = mu[:, oth]                                                          # (C, m, n-1)
        a = mu[:, j, None] - mo
        Coo = Cov[:, oth[:, :, None], oth[:, None, :]]                            # (C, m, n-1, n-1)
        Coj = Cov[:, oth, j[:, None]]                                             # (C, m, n-1)
        Cjj = Cov[:, j, j]                                                        # (C, m)
        S = Coo - Coj[..., :, None] - Coj[..., None, :] + Cjj[..., None, None]
        return a.reshape(C * m, n - 1), S.reshape(C * m, n - 1, n - 1)

    def objective(self, g, shares):
        """shares (C, m) -> profit (C,)."""
        return (g["margin"] * shares).sum(1) - self.F * g["fixed"]


def chol(C):
    import torch
    L, info = torch.linalg.cholesky_ex(C)
    if bool((info != 0).any()):
        eye = torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
        L = torch.linalg.cholesky(C + 1e-10 * C.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye)
    return L


def ghk_p(a, L, u, budget=1.5e9):
    """P(X <= a), X = L Z, GHK with fixed draws u (M, d-1); as analysis/T11/t11.py."""
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


class GHKScorer:
    def __init__(self, sh, th, M, stream=0):
        self.sh, self.th, self.M, self.stream, self.sob = sh, th, M, stream, {}

    def _u(self, d):
        import torch
        if d not in self.sob:
            self.sob[d] = torch.quasirandom.SobolEngine(max(d - 1, 1), scramble=True, seed=SEED + 1000 * self.stream + d) \
                .draw(self.M, dtype=torch.float64).to(self.sh.dev)
        return self.sob[d]

    def shares(self, g):
        a, S = self.sh.orthants(g, self.th)
        p = ghk_p(a, chol(S), self._u(a.shape[1]))
        return p.reshape(-1, g["m"])


class EngineScorer:
    def __init__(self, sh, th):
        import modal_gp_api as M
        self.M, self.sh, self.th = M, sh, th

    def shares(self, g):
        import torch
        a, S = self.sh.orthants(g, self.th)
        d = a.shape[1]
        out = torch.empty(a.shape[0], dtype=torch.float64, device=a.device)
        chunk = self.M.orthant_rows_per_chunk(d, 27, 5, 8, a.device)
        for lo in range(0, a.shape[0], chunk):
            hi = min(lo + chunk, a.shape[0])
            uu, ss = self.M.resolve_degenerate_coords(a[lo:hi].contiguous(), S[lo:hi].contiguous(), 0.0)
            out[lo:hi] = self.M.orthant_prob(uu, ss, d, a.device, order=1, chunk=hi - lo)[:, 0].clamp(0, 1)
        return out.reshape(-1, g["m"])


class SimScorer:
    """Common-draw first-choice simulator: N simulated people (tastes, attribute-level errors, own errors) drawn once
    for all 60 grid SKUs + none; a candidate's share of j = fraction of people whose argmax over its alternatives is j.
    Prices shift only the price term. float32 utilities."""

    def __init__(self, sh, th, N, m, stream=0):
        import torch
        dev = sh.dev
        P = sh.Xb.shape[1]
        g = torch.Generator(device=dev).manual_seed(SEED + 7919 * stream + N)
        b, w = th[:P], th[P:2 * P].exp()
        u = th[2 * P:]
        s = torch.softmax(torch.cat([u, u.new_zeros(1)]), 0)
        s2 = sh.tvar * (s * (1 - 4e-6) + 1e-6)
        beta = b + w * torch.randn(N, P, generator=g, device=dev, dtype=torch.float64)
        lv = torch.repeat_interleave(s2[:3].sqrt(), torch.tensor(m["levels"], device=dev))
        zeta = torch.randn(N, int(m["levels"].sum()), generator=g, device=dev, dtype=torch.float64) * lv
        nu = torch.randn(N, 61, generator=g, device=dev, dtype=torch.float64)
        nu[:, :60] *= s2[3].sqrt()
        nu[:, 60] *= sh.tvar ** 0.5
        mem = torch.tensor(m["mem"], dtype=torch.float64, device=dev)
        self.U0 = (beta @ sh.Xb.T + zeta @ mem.T + nu).float().T.contiguous()       # (61, N)
        self.bp = beta[:, sh.pcol].float()                                           # (N,)
        self.sh, self.N = sh, N

    def shares(self, g):
        import torch
        alts, pr, m = g["alts"], g["pr"], g["m"]
        C, n = alts.shape
        dp = torch.where(alts == 60, 0.0, pr - self.sh.ref).float()
        out = torch.empty(C, m, dtype=torch.float64, device=alts.device)
        step = max(1, 2 ** 27 // (self.N * n))
        for lo in range(0, C, step):
            hi = min(lo + step, C)
            U = self.U0[alts[lo:hi]] + dp[lo:hi, :, None] * self.bp                 # (c, n, N)
            win = U.argmax(1)                                                        # (c, N)
            cnt = torch.zeros(hi - lo, n, device=alts.device, dtype=torch.float64)
            cnt.scatter_add_(1, win, torch.ones_like(win, dtype=torch.float64))
            out[lo:hi] = cnt[:, :m] / self.N
        return out


def score(sh, scorer, cand, batch=4096):
    """Objective (C,) for candidates cand (C, s) with a scorer."""
    import torch
    out = torch.empty(cand.shape[0], dtype=torch.float64, device=cand.device)
    for lo in range(0, cand.shape[0], batch):
        c = cand[lo:lo + batch]
        for g in sh.expand(c):
            out[lo + g["idx"]] = sh.objective(g, scorer.shares(g))
    return out


def sync(dev):
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()


def throughput(sh, make, cand, min_s, dev):
    """Candidates/s of scorer make() on cand, timing growing prefixes until >= min_s (warm-up excluded)."""
    sc = make()
    score(sh, sc, cand[:64])
    sync(dev)
    n = 256
    while True:
        n = min(n, cand.shape[0])
        t0 = time.perf_counter()
        score(sh, sc, cand[:n])
        sync(dev)
        dt = time.perf_counter() - t0
        if dt >= min_s or n == cand.shape[0]:
            return dict(n=n, seconds=dt, per_s=n / dt)
        n *= 4


def calibrate_F(m, insts, dev, M=4096):
    """Fixed cost per SKU (D3/D4), from the fitted model: 0.5 x median contribution of a brand-0 SKU at base prices
    (D4 base: firm profit with all 7 shelved SKUs on minus with one dropped), so keeping everything is not optimal."""
    import torch
    inst = next(i for i in insts if i["name"] == "D4_base")
    sh = Shelf(m, inst, dev)
    th = torch.tensor(m["th_hat"], dtype=torch.float64, device=dev)
    sc = GHKScorer(sh, th, M)
    grid = np.array(inst["grid"])
    own_lv = [int(np.argmin(np.abs(grid - p))) + 1 for p in inst["bp_own"]]
    base = np.r_[own_lv, np.zeros(len(inst["ext"]), int)]
    rows = [base] + [np.where(np.arange(len(base)) == i, 0, base) for i in range(len(own_lv))]
    v = score(sh, sc, torch.tensor(np.array(rows), device=dev)).cpu().numpy()
    contrib = v[0] - v[1:]
    return float(0.5 * np.median(contrib)), contrib.tolist(), float(v[0])


def probe_impl(m, insts, cfg):
    """Step 0 phase A on one device: wiring checks, F, and candidates/s per scorer setting and decision."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"
    th = torch.tensor(m["th_hat"], dtype=torch.float64, device=dev)
    t_all = time.perf_counter()
    F, contrib, base_profit = calibrate_F(m, insts, dev, M=cfg["M_F"])
    res = dict(gpu=gpu, F=F, F_contrib=contrib, D4_base_profit=base_profit, wiring={}, tp={})
    gen = torch.Generator(device=dev).manual_seed(SEED)
    for inst in insts:
        if inst["name"] not in cfg["names"]:
            continue
        inst = dict(inst, F=F * inst.get("F_mult", 0.0)) if inst["dec"] in ("D3", "D4") else inst
        sh = Shelf(m, inst, dev)
        sh.F = inst["F"]
        cand = sh.sample(cfg["n_cand"], gen)
        # wiring: orthant shares (GHK, M_w) vs simulator at N_w, on a few candidates
        if cfg["wiring"]:
            c = cand[:cfg["n_wire"]]
            g = sh.expand(c)
            ga = [GHKScorer(sh, th, cfg["M_w"]).shares(x) for x in g]
            gs = [SimScorer(sh, th, cfg["N_w"], m).shares(x) for x in g]
            d = torch.cat([(a - b).abs().flatten() for a, b in zip(ga, gs)])
            p = torch.cat([a.flatten() for a in ga])
            se = (p * (1 - p) / cfg["N_w"]).sqrt()
            ge = [EngineScorer(sh, th).shares(x) for x in g]
            de = torch.cat([(a - e).abs().flatten() for a, e in zip(ga, ge)])
            res["wiring"][inst["name"]] = dict(max_abs=float(d.max()), max_z=float((d / se.clamp(min=1e-6)).max()),
                                               n_shares=int(d.numel()), mean_share=float(p.mean()),
                                               engine_vs_ghk_max=float(de.max()), engine_vs_ghk_median=float(de.median()))
        tp = {"engine": throughput(sh, lambda: EngineScorer(sh, th), cand, cfg["min_s"], dev)}
        for M in cfg["ghk_M"]:
            tp[f"ghk_{M}"] = throughput(sh, lambda: GHKScorer(sh, th, M), cand, cfg["min_s"], dev)
        for N in cfg["sim_N"]:
            tp[f"sim_{N}"] = throughput(sh, lambda: SimScorer(sh, th, N, m), cand, cfg["min_s"], dev)
        tp[f"ref_{cfg['ref_M']}"] = throughput(sh, lambda: GHKScorer(sh, th, cfg["ref_M"]), cand, cfg["min_s"], dev)
        res["tp"][inst["name"]] = tp
    res["seconds"] = time.perf_counter() - t_all
    return res


PROBE = dict(names=("D1_k6", "D1_k10", "D2_base", "D3_base", "D4_base"), n_cand=20000, n_wire=20, M_w=2 ** 14,
             N_w=2 ** 20, min_s=1.0, ghk_M=GHK_M, sim_N=SIM_N, ref_M=REF_M, M_F=4096, wiring=True)


@app.function(image=image, gpu="T4", timeout=3600)
def probe_t4(m, insts, cfg):
    return probe_impl(m, insts, cfg)


@app.function(image=image, gpu="H100", timeout=3600)
def probe_h100(m, insts, cfg):
    return probe_impl(m, insts, cfg)


# ---------------------------------------------------------------------------
# Step 0 phase B: adequacy sizing, D2 ordering, M_acc, speed gate (one H100, $5 cap)
# ---------------------------------------------------------------------------
TOPK = 100
STREAMS = 20
RATE = 0.95                 # adequacy / ordering threshold (user 2026-10-16; ordering threshold: my default)
MACC_FRAC = 0.1             # M_acc: 95th pct |objective error| <= 0.1 x materiality (my default)
N_SAMPLE = 20000


def samples(insts):
    """Host: per instance, the fixed candidate set (exhaustive if the space is <= 20,000 or D1; else 20,000 uniform,
    deduplicated), plus D2's adjacent-level pairs (2,000 bases x one-step neighbours)."""
    from itertools import combinations
    rng = np.random.default_rng(SEED)
    out = {}
    for inst in insts:
        d = inst["dec"]
        if d == "D1":
            c = np.array(list(combinations(range(len(inst["shelf"])), inst["k"])))
        elif d == "D3":
            g = len(inst["grid"]) + 1
            c = np.array(np.unravel_index(np.arange(g ** len(inst["ext"])), (g,) * len(inst["ext"]))).T
        elif d == "D2":
            c = np.unique(rng.integers(0, len(inst["grid"]), (N_SAMPLE + 200, len(inst["own"]))), axis=0)
            c = c[rng.permutation(len(c))[:N_SAMPLE]]
        else:
            c = np.unique(rng.integers(0, len(inst["grid"]) + 1, (N_SAMPLE + 200, len(inst["own"]) + len(inst["ext"]))), axis=0)
            c = c[rng.permutation(len(c))[:N_SAMPLE]]
        item = dict(cand=c)
        if d == "D2":
            base = c[:2000]
            nb, pairs = [], []
            for i, b in enumerate(base):
                for k in range(len(b)):
                    for st in (-1, 1):
                        v = b[k] + st
                        if 0 <= v < len(inst["grid"]):
                            x = b.copy()
                            x[k] = v
                            pairs.append((i, len(nb)))
                            nb.append(x)
            item["nb"] = np.array(nb)
            item["pairs"] = np.array(pairs)
        out[inst["name"]] = item
    return out


def step0b_impl(m, insts, smp, cfg):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    th = torch.tensor(m["th_hat"], dtype=torch.float64, device=dev)
    price = cfg["price_h"] / 3600
    spent = lambda: cfg["spent0"] + (time.perf_counter() - t0) * price      # noqa: E731
    res = dict(gpu=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", inst={}, dec={}, stopped=None)
    sh, C, R, W, best = {}, {}, {}, {}, {}
    for inst in insts:
        n = inst["name"]
        inst = dict(inst, F=cfg["F"] * inst.get("F_mult", 0.0)) if inst["dec"] in ("D3", "D4") else inst
        sh[n] = Shelf(m, inst, dev)
        sh[n].F = inst.get("F", 0.0)
        C[n] = torch.tensor(smp[n]["cand"], device=dev)
        ts = time.perf_counter()
        R[n] = score(sh[n], GHKScorer(sh[n], th, cfg["ref_M"]), C[n])
        sync(dev)
        best[n] = float(R[n].max())
        W[n] = R[n] >= best[n] * (1 - MATERIAL)
        res["inst"][n] = dict(n_cand=int(C[n].shape[0]), best=best[n], n_within=int(W[n].sum()),
                              ref_seconds=time.perf_counter() - ts)
        if n.startswith("D2"):
            Rn = score(sh[n], GHKScorer(sh[n], th, cfg["ref_M"]), torch.tensor(smp[n]["nb"], device=dev))
            pr = torch.tensor(smp[n]["pairs"], device=dev)
            dR = R[n][pr[:, 0]] - Rn[pr[:, 1]]
            keep = dR.abs() >= MATERIAL * best[n]
            smp[n]["_ord"] = (pr[keep], dR[keep].sign(), torch.tensor(smp[n]["nb"], device=dev))
            res["inst"][n]["n_pairs_material"] = int(keep.sum())
        if spent() > cfg["cap"]:
            res["stopped"] = f"cap during reference ({n})"
            return res
    decs = [d for d in ("D4", "D1", "D2", "D3") if any(i["dec"] == d for i in insts)]     # primary first

    def run(dec, make, label, streams, err=False, order=False):
        hits, tot, sec, cnt, errs, ok, ordn = 0, 0, 0.0, 0, [], 0, 0
        for n in [i["name"] for i in insts if i["dec"] == dec]:
            for s in range(streams):
                sc = make(sh[n], s)
                ts = time.perf_counter()
                S = score(sh[n], sc, C[n])
                sync(dev)
                if s == 0:
                    sec += time.perf_counter() - ts
                    cnt += C[n].shape[0]
                    if err:
                        errs.append(((S - R[n]).abs() / best[n]).cpu())
                top = S.topk(min(TOPK, S.numel())).indices
                hits += bool(W[n][top].any())
                tot += 1
                if order and n.startswith("D2"):
                    pr, sg, nb = smp[n]["_ord"]
                    Sn = score(sh[n], sc, nb)
                    ok += int(((S[pr[:, 0]] - Sn[pr[:, 1]]).sign() == sg).sum())
                    ordn += int(sg.numel())
        out = dict(setting=label, rate=hits / tot, streams=tot, streams_per_inst=streams, per_s=cnt / sec)
        if err:
            e = torch.cat(errs)
            out["err_p95_frac_best"] = float(e.quantile(0.95))
        if order and ordn:
            out["order_rate"] = ok / ordn
        return out

    for dec in decs:
        r = dict(engine=run(dec, lambda h, s: EngineScorer(h, th), "engine", 1, err=True))
        lad = []
        found = False
        for M in cfg["ghk_M"]:
            x = run(dec, lambda h, s, M=M: GHKScorer(h, th, M, stream=s), f"ghk_{M}", 1 if found else STREAMS, err=True)
            found = found or x["rate"] >= RATE
            lad.append(x)
            if spent() > cfg["cap"]:
                res["stopped"] = f"cap in GHK ladder ({dec}, M={M})"
                break
            if x["rate"] >= RATE and x["err_p95_frac_best"] <= MACC_FRAC * MATERIAL:
                break
        r["ghk"] = lad
        adq = [x for x in lad if x["rate"] >= RATE and x["streams_per_inst"] == STREAMS]
        r["ghk_adequate"] = adq[0] if adq else None
        acc = [x for x in lad if x["err_p95_frac_best"] <= MACC_FRAC * MATERIAL]
        r["M_acc"] = acc[0]["setting"] if acc else None
        if res["stopped"]:
            res["dec"][dec] = r
            return res
        lad = []
        for N in cfg["sim_N"]:
            x = run(dec, lambda h, s, N=N: SimScorer(h, th, N, m, stream=s), f"sim_{N}", STREAMS, order=(dec == "D2"))
            lad.append(x)
            if spent() > cfg["cap"]:
                res["stopped"] = f"cap in simulator ladder ({dec}, N={N})"
                break
            if x["rate"] >= RATE and (dec != "D2" or x.get("order_rate", 0) >= RATE):
                break
        r["sim"] = lad
        adq = [x for x in lad if x["rate"] >= RATE]
        r["sim_adequate"] = adq[0] if adq else None
        if dec == "D2":
            o = [x for x in lad if x.get("order_rate", 0) >= RATE]
            r["sim_order"] = o[0] if o else None
        res["dec"][dec] = r
        res["spent"] = spent()
        if res["stopped"]:
            return res
    res["seconds"] = time.perf_counter() - t0
    res["spent"] = spent()
    return res


@app.function(image=image, gpu="H100", timeout=4 * 3600)
def step0b(m, insts, smp, cfg):
    return step0b_impl(m, insts, smp, cfg)


def gate(res):
    """Speed gate per decision. Simulator N = larger of adequacy and D2 ordering (the ordering N applies to all)."""
    order_N = None
    o = res["dec"].get("D2", {}).get("sim_order")
    if o:
        order_N = int(o["setting"].split("_")[1])
    rows = {}
    for dec, r in res["dec"].items():
        e = r["engine"]["per_s"]
        g = r.get("ghk_adequate")
        sims = r.get("sim", [])
        sa = r.get("sim_adequate")
        sim_pick = None
        if sa:
            need = max(int(sa["setting"].split("_")[1]), order_N or 0)
            cand = [x for x in sims if int(x["setting"].split("_")[1]) == need]
            sim_pick = cand[0] if cand else dict(setting=f"sim_{need}", per_s=None, note="ordering N not run on this decision")
        comp = [x for x in (g, sim_pick) if x and x.get("per_s")]
        bestc = max(comp, key=lambda x: x["per_s"]) if comp else None
        rows[dec] = dict(engine_per_s=e, engine_adequate=r["engine"]["rate"] >= RATE, engine_rate=r["engine"]["rate"],
                         ghk=g and g["setting"], ghk_per_s=g and g["per_s"], sim=sim_pick and sim_pick["setting"],
                         sim_per_s=sim_pick and sim_pick.get("per_s"), best=bestc and bestc["setting"],
                         ratio=(e / bestc["per_s"]) if bestc else None, M_acc=r.get("M_acc"))
    return rows


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe-local"
    m, insts = model(), instances()
    OUT.mkdir(exist_ok=True)
    if cmd == "sizes":
        for i in insts:
            print(f"{i['name']:<12} {space_size(i):>16,}")
        return
    if cmd == "probe-local":
        cfg = dict(PROBE, n_cand=256, n_wire=4, M_w=2 ** 10, N_w=2 ** 14, min_s=0.05, ghk_M=(64,), sim_N=(1024,),
                   ref_M=256, M_F=256)
        sys.modules.setdefault("t15", sys.modules[__name__])
        r = probe_impl(m, insts, cfg)
        print(json.dumps(r, indent=1, default=float)[:4000])
        return
    if cmd == "step0b":
        F = json.loads((OUT / "probe.json").read_text())["H100"]["F"]
        smp = samples(insts)
        cfg = dict(F=F, ref_M=REF_M, ghk_M=GHK_M, sim_N=SIM_N, cap=5.0, spent0=0.2, price_h=GPU_PRICE["H100"])
        print({k: len(v["cand"]) for k, v in smp.items()})
        with modal.enable_output(), app.run():
            res = step0b.remote(m, insts, smp, cfg)
        res["gate"] = gate(res)
        (OUT / "step0b.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(dict(stopped=res["stopped"], spent=res.get("spent"), gate=res["gate"]), indent=1, default=float))
        return
    if cmd in ("screen", "screen-local", "breakdown", "breakdown-local", "tiered", "tiered-local", "trace-local"):
        sys.path.insert(0, str(HERE / "private"))
        import t15_engine                        # private (engine settings); not in the public repo
        return t15_engine.main_private(cmd, m, insts)
    if cmd == "step0b-local":
        F = 0.0049
        sub = [i for i in insts if i["name"] in ("D1_k6", "D2_base", "D4_base")]
        smp = samples(sub)
        for k in smp:
            smp[k]["cand"] = smp[k]["cand"][:300]
            if "nb" in smp[k]:
                keep = smp[k]["pairs"][:, 0] < 20
                smp[k]["pairs"], smp[k]["nb"] = smp[k]["pairs"][keep], smp[k]["nb"][:keep.sum()]
        global STREAMS
        STREAMS = 2
        cfg = dict(F=F, ref_M=1024, ghk_M=(64, 128), sim_N=(1024, 2048), cap=5.0, spent0=0.0, price_h=0.0)
        res = step0b_impl(m, sub, smp, cfg)
        res["gate"] = gate(res)
        print(json.dumps(res, indent=1, default=float)[:3500])
        return
    if cmd == "probe":
        with modal.enable_output(), app.run():
            calls = dict(T4=probe_t4.spawn(m, insts, PROBE), H100=probe_h100.spawn(m, insts, PROBE))
            res = {k: c.get() for k, c in calls.items()}
        (OUT / "probe.json").write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: dict(gpu=v["gpu"], seconds=v["seconds"], F=v["F"], wiring=v["wiring"]) for k, v in res.items()},
                         indent=1, default=float))


if __name__ == "__main__":
    main()
