"""T1. sigma recovery: probit by full MLE (orthant engine) vs HB-MNL + tuned RFC.

Design: analysis/T1/DESIGN.md. Approved scope 2026-10-06: truth G1 (D2 values, pair corr .90)
and RFC-S (strict Sawtooth). Added 2026-10-07: G0 (logit null) and G2 (pair corr .97).

    python analysis/T1/run.py --reps 20 --truth G0

    python analysis/T1/run.py --selftest     # CPU, local .so, N = 100, fail-fast checks
    python analysis/T1/run.py --reps 20      # full run (ask first)

--mvp ghk (default since 2026-10-07): the MVP arm is the exact 36-dim panel likelihood by GHK
with fixed Sobol draws, fitted on Modal GPUs (ghk_gpu.py), all replicates in parallel before
the CPU arms. --mvp msl: the earlier option-1 MSL fit on CPU.

--scale total (default since 2026-10-09): the GHK fit fixes the total error variance at T_VAR and
estimates the split, u_a = log(sigma_a^2 / sigma_nu^2), as mvp_refit.py (2026-10-08). --scale snu:
scale pinned by sigma_nu alone, as the original T1 runs (exact reproduction).

Writes analysis/T1/out/<mode>/: manifest.json (code and package versions), log.txt,
results.json (one record per replicate).
"""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "rfc_vs_exact"))
import rfc  # noqa: E402  (orthant backend: local paid .so or hosted service)

SEED = 20261006
T_VAR = np.pi ** 2 / 6  # total product error variance (logit scale, as tests/rfc_vs_probit)

# ---------------------------------------------------------------------------
# Market (tests/rfc_vs_probit, with a normal price slope so the panel is jointly normal)
# ---------------------------------------------------------------------------
LEVELS = (4, 3, 3)  # brand, flavor, pack
PRICES = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
PRICE_REF = 5.0
B_TRUE = np.array([.6, .2, -.1,  .4, 0.,  .3, 0.,  -.975,  1.0])  # brand3 flavor2 pack2 price none
W_TRUE = np.array([.8, .8, .8,   .6, .6,  .5, .5,   .41,   .5])   # sd of each (diagonal Omega)
P = len(B_TRUE)
NAMES = ["brand1", "brand2", "brand3", "flavor1", "flavor2", "pack1", "pack2", "price", "none"]
TRUTHS = {  # error variance shares (brand, flavor, pack, product) of T_VAR
    "G0": np.array([0., 0., 0., 1.]),       # null: no similarity, iid Gumbel errors (logit truth)
    "G1": np.array([.08, .45, .45, .02]),  # tests/rfc_vs_probit D2: same flavor+pack pair corr .90
    "G2": np.array([.01, .485, .485, .02]),  # strong: pair corr .97; sigma_nu share kept from G1
}
LOGIT_TRUTHS = {"G0"}  # errors are Gumbel (variance pi^2/6 = T_VAR), not normal
SIG_FLOOR = 1e-3  # log(sigma_a) "truth" for the MVP's diagnostic nll_true when sigma_a = 0
COST = 1.80  # placeholder unit cost, 40% of the mid price
HURDLE = 0.5  # placeholder line-extension hurdle (user's to set)

#                 b  f  p  price
BASE = np.array([[0, 0, 0, 5.00],   # A
                 [1, 1, 1, 4.50],   # B
                 [2, 0, 2, 5.50],   # C
                 [1, 1, 0, 4.00],   # D
                 [2, 1, 2, 3.50]])  # E
CAND = {"A'": [3, 0, 0, 5.00],    # A with another brand (same flavor, pack)
        "A''": [0, 0, 0, 4.70],   # A $0.30 cheaper: line extension
        "F": [3, 0, 1, 4.75],
        "G": [3, 2, 2, 4.75]}
OWN_COST = [0, 2]  # D-cost: the firm owns A and C


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


LOGF = None


def fail(msg):
    log(f"FAIL: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Coding
# ---------------------------------------------------------------------------
def code_rows(prod):
    """prod (..., 4) = brand, flavor, pack level indices + price -> (..., P) rows, none col 0."""
    prod = np.asarray(prod, float)
    out = np.zeros(prod.shape[:-1] + (P,))
    col = 0
    for a, L in enumerate(LEVELS):
        lv = prod[..., a].astype(int)
        for l in range(L - 1):
            out[..., col + l] = (lv == l).astype(float) - (lv == L - 1)
        col += L - 1
    out[..., 7] = prod[..., 3] - PRICE_REF
    return out


def none_row():
    r = np.zeros(P)
    r[8] = 1
    return r


def task_X(prods):
    """(..., K, 4) products -> (..., K+1, P), none last."""
    X = code_rows(prods)
    nr = np.broadcast_to(none_row(), X.shape[:-2] + (1, P))
    return np.concatenate([X, nr], -2)


def err_cov(prods, sig):
    """Within-task error covariance (K+1, K+1) for products (K, 4); sig = (sb, sf, sp, snu)."""
    prods = np.asarray(prods)
    K = len(prods)
    E = np.zeros((K + 1, K + 1))
    for a in range(3):
        E[:K, :K] += sig[a] ** 2 * (prods[:, a][:, None] == prods[:, a][None, :])
    E[:K, :K] += sig[3] ** 2 * np.eye(K)
    E[K, K] = np.sum(np.square(sig))
    return E


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def random_tasks(rng, n, k=3):
    """(n, k, 4) products with random levels; no two identical products in a task."""
    out = np.empty((n, k, 4))
    for i in range(n):
        while True:
            t = np.column_stack([rng.integers(0, L, k) for L in LEVELS] + [PRICES[rng.integers(0, 5, k)]])
            if len({tuple(r) for r in t}) == k:
                out[i] = t
                break
    return out


def holdout_tasks(rng):
    """6 fixed tasks; the first 2 contain a pair sharing flavor and pack (different brand)."""
    H = random_tasks(rng, 6)
    for i in range(2):
        H[i, 1, 1:3] = H[i, 0, 1:3]
        H[i, 1, 0] = (H[i, 0, 0] + 1 + rng.integers(0, 3)) % 4
    return H


def simulate(rng, prods, beta, sig, logit=False):
    """First choices. prods (N, T, K, 4), beta (N, P) -> (N, T) chosen index (K = none).
    logit: iid Gumbel errors (G0) instead of the normal error components."""
    N, T, K, _ = prods.shape
    V = np.einsum("ntkp,np->ntk", task_X(prods), beta)  # (N, T, K+1), none last
    if logit:
        return np.argmax(V + rng.gumbel(size=V.shape), 2)
    e = np.zeros_like(V)
    e[..., :K] = sig[3] * rng.standard_normal((N, T, K))
    for a, L in enumerate(LEVELS):
        z = sig[a] * rng.standard_normal((N, T, L))
        e[..., :K] += np.take_along_axis(z, prods[..., a].astype(int), 2)
    e[..., K] = np.sqrt(np.sum(np.square(sig))) * rng.standard_normal((N, T))
    return np.argmax(V + e, 2)


def make_data(rng, N, T, sig, logit=False):
    beta = B_TRUE + W_TRUE * rng.standard_normal((N, P))
    prods = random_tasks(rng, N * T).reshape(N, T, 3, 4)
    y = simulate(rng, prods, beta, sig, logit)
    return dict(prods=prods, y=y, beta=beta)


# ---------------------------------------------------------------------------
# Probit: shares (population, exact orthants) and simulated panel likelihood
# ---------------------------------------------------------------------------
def unpack(th, snu):
    return th[:P], np.exp(th[P:2 * P]), np.r_[np.exp(th[2 * P:2 * P + 3]), snu]


def softmax0(u):
    z = np.r_[u, 0.0]
    e = np.exp(z - z.max())
    return e / e.sum()


def unpack_total(th):
    """--scale total: u_a = log(sigma_a^2 / sigma_nu^2), total error variance T_VAR (as ghk_gpu.sig2_of)."""
    return th[:P], np.exp(th[P:2 * P]), np.sqrt(T_VAR * (softmax0(th[2 * P:]) * (1 - 4e-6) + 1e-6))


def theta_true(sig, scale):
    if scale == "total":
        sh = sig ** 2 / T_VAR
        return np.r_[B_TRUE, np.log(W_TRUE), np.log(np.maximum(sh[:3], 1e-6) / sh[3])]
    return np.r_[B_TRUE, np.log(W_TRUE), np.log(np.maximum(sig[:3], SIG_FLOOR))]


def pop_shares(prods, b, w, sig):
    """Population choice shares (K+1,) for products (K, 4): beta ~ N(b, diag w^2) integrated
    exactly (U is jointly normal). Renormalized (rule from research/rfc_vs_exact)."""
    X = task_X(np.asarray(prods, float))
    mu = X @ b
    C = X @ np.diag(w ** 2) @ X.T + err_cov(prods, sig)
    K1 = len(mu)
    ups, covs = [], []
    for j in range(K1):
        oth = [k for k in range(K1) if k != j]
        M = np.zeros((K1 - 1, K1))
        M[np.arange(K1 - 1), oth] = 1
        M[:, j] = -1
        ups.append(-(M @ mu))
        covs.append(M @ C @ M.T)
    p = rfc.orthant_cdf(np.array(ups), np.array(covs))
    return p / p.sum()


_LOGIT_Z = qmc.Sobol(P, scramble=True, seed=SEED).random(2 ** 16)


def mixed_logit_shares(prods):
    """G0 truth: population shares (K+1,) with beta ~ N(B_TRUE, diag W_TRUE^2) and iid Gumbel
    errors, integrated over 2^16 scrambled Sobol taste draws."""
    beta = B_TRUE + W_TRUE * ndtri(np.clip(_LOGIT_Z, 1e-12, 1 - 1e-12))
    V = beta @ task_X(np.asarray(prods, float)).T
    return np.exp(V - logsumexp(V, 1, keepdims=True)).mean(0)


class MSL:
    """Simulated panel probit likelihood (option 1, 2026-10-06). beta_n = b + w * z_nr over R
    scrambled-Halton draws per respondent; given beta, each task's choice probability is a 3-dim
    orthant from the engine (the within-task error covariance does not depend on beta).
    P_n = mean_r prod_t P_ntr. The exact 36-dim panel orthant was tried first and dropped:
    engine values on respondents with likelihoods near e^-19 jump with 3e-3 parameter steps."""

    def __init__(self, prods, y, R, seed, chunk=None):
        N, T, K, _ = prods.shape
        K1 = K + 1
        X = task_X(prods)  # (N, T, K1, P)
        oth = np.array([[k for k in range(K1) if k != c] for c in range(K1)])[y]  # (N, T, K)
        M = np.zeros((N, T, K, K1))
        np.put_along_axis(M, oth[..., None], 1.0, axis=3)
        np.put_along_axis(M, np.broadcast_to(y[:, :, None, None], (N, T, K, 1)), -1.0, axis=3)
        self.Xd = M @ X  # (N, T, K, P)
        S = np.zeros((4, N, T, K1, K1))  # brand, flavor, pack, product; "none" in all four
        for a in range(3):
            lv = prods[..., a]
            S[a, :, :, :K, :K] = lv[..., :, None] == lv[..., None, :]
        S[3, :, :, :K, :K] = np.eye(K)
        S[:, :, :, K, K] = 1
        self.G = np.einsum("ntik,antkl,ntjl->antij", M, S, M)  # (4, N, T, K, K)
        h = qmc.Halton(d=P, scramble=True, seed=seed).random(N * R)
        self.Z = ndtri(np.clip(h, 1e-12, 1 - 1e-12)).reshape(N, R, P)
        self.N, self.T, self.K, self.R = N, T, K, R
        self.chunk = chunk or max(1, 250_000 // (R * T))

    def _chunk(self, s, b, w, sig, grad):
        Xd, Z, G = self.Xd[s], self.Z[s], self.G[:, s]
        n, R, T, K = len(Xd), self.R, self.T, self.K
        cov = np.einsum("a,antij->ntij", sig ** 2, G)
        covb = np.broadcast_to(cov[:, None], (n, R, T, K, K)).reshape(-1, K, K)
        u = -np.einsum("ntkp,nrp->nrtk", Xd, b + w * Z)
        uf = u.reshape(-1, K)
        p = np.maximum(rfc.orthant_cdf(uf, covb), 1e-300).reshape(n, R, T)
        lr = np.log(p).sum(2)  # (n, R)
        ll = logsumexp(lr, 1) - np.log(R)
        if not grad:
            return ll, None
        # dP/du_i = phi(u_i; c_ii) * Phi_2(u_-i - c_-i,i u_i / c_ii; C_-i|i), engine at 2 dims
        dP = np.empty_like(uf)
        for i in range(K):
            o = [j for j in range(K) if j != i]
            cii = covb[:, i, i]
            ci = covb[:, o, i]
            Sc = covb[:, o][:, :, o] - ci[:, :, None] * ci[:, None, :] / cii[:, None, None]
            dens = np.exp(-uf[:, i] ** 2 / (2 * cii)) / np.sqrt(2 * np.pi * cii)
            dP[:, i] = dens * rfc.orthant_cdf(uf[:, o] - ci * (uf[:, i] / cii)[:, None], Sc)
        gu = dP.reshape(n, R, T, K) / p[..., None]
        W = np.exp(lr - logsumexp(lr, 1, keepdims=True))  # posterior weight of each draw
        gb = -np.einsum("nr,nrtk,ntkp->np", W, gu, Xd)
        gw = -np.einsum("nr,nrtk,ntkp,nrp->np", W, gu, Xd, Z) * w  # d/d log w
        return ll, np.hstack([gb, gw])

    def ll_vec(self, b, w, sig, grad=False):
        lls, gs = [], []
        for s0 in range(0, self.N, self.chunk):
            ll, g = self._chunk(slice(s0, s0 + self.chunk), b, w, sig, grad)
            lls.append(ll)
            gs.append(g)
        return (np.concatenate(lls), np.vstack(gs)) if grad else np.concatenate(lls)

    def scores(self, th, snu, h=1e-4):
        """Per-respondent log-likelihood (N,) and scores (N, 2P + 3): analytic for b and log w,
        forward differences for log sigma (brand, flavor, pack)."""
        ll, g = self.ll_vec(*unpack(th, snu), grad=True)
        gs = np.empty((self.N, 3))
        for i in range(3):
            e = th.copy()
            e[2 * P + i] += h
            gs[:, i] = (self.ll_vec(*unpack(e, snu)) - ll) / h
        return ll, np.hstack([g, gs])


def fit_probit(model, snu, th0, label, maxiter=300):
    state = dict(it=0, t0=time.time(), last=None)

    def fg(th):
        ll, S = model.scores(th, snu)
        state["last"] = -ll.sum()
        return -ll.sum(), -S.sum(0)

    def cb(th):
        state["it"] += 1
        if state["it"] % 5 == 0:
            log(f"  {label} iter {state['it']}: nll {state['last']:.3f}, {time.time() - state['t0']:.0f} s")

    res = minimize(fg, th0, jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=maxiter))
    log(f"  {label} done: nll {res.fun:.3f}, {res.nit} iters, {res.nfev} evals, "
        f"max |grad| {np.abs(res.jac).max():.3f}, {time.time() - state['t0']:.0f} s, {res.message}")
    return res, time.time() - state["t0"]


def probit_se(model, th, snu, h=1e-3):
    """Sandwich SEs (log-sd scale): per-respondent scores, and a forward-difference Hessian of
    the summed score. Returns (sandwich, Hessian-only)."""
    nth = len(th)
    _, S0 = model.scores(th, snu)
    g0 = S0.sum(0)
    H = np.empty((nth, nth))
    for i in range(nth):
        e = th.copy()
        e[i] += h
        H[:, i] = (model.scores(e, snu)[1].sum(0) - g0) / h
    H = (H + H.T) / 2
    Hi = np.linalg.inv(-H)
    V = Hi @ (S0.T @ S0) @ Hi
    return np.sqrt(np.clip(np.diag(V), 0, None)), np.sqrt(np.clip(np.diag(Hi), 0, None))


# ---------------------------------------------------------------------------
# HB-MNL (bayesm) and RFC-S
# ---------------------------------------------------------------------------
def run_hb(work, prods, y, iters, seed):
    N, T, K, _ = prods.shape
    X = task_X(prods)  # (N, T, K+1, P)
    rows = []
    for n in range(N):
        for t in range(T):
            for k in range(K + 1):
                rows.append([n, t, k, int(y[n, t] == k)] + list(X[n, t, k]))
    path = work / "hb_in.csv"
    np.savetxt(path, np.array(rows), delimiter=",", comments="",
               header="resp,task,alt,chosen," + ",".join(f"x{i}" for i in range(P)), fmt="%.6g")
    t0 = time.time()
    r = subprocess.run(["Rscript", str(HERE / "hb.R"), str(path), str(work / "hb"), str(iters), "10", str(seed)],
                       capture_output=True, text=True)
    if r.returncode:
        fail(f"hb.R: {r.stderr[-2000:]}")
    B = np.loadtxt(work / "hb_beta.csv", delimiter=",", skiprows=1)
    ll = np.loadtxt(work / "hb_loglike.csv", delimiter=",", skiprows=1)
    return B, ll, time.time() - t0


class RFC:
    """Sawtooth-style RFC on HB point estimates: normal attribute-level error sigma_l shared by
    products with the same brand / flavor / pack level (not price), Gumbel product error g, and
    an independent normal error of variance 3 sigma_l^2 on "none". The exponent is redundant
    under first choice (it rescales both errors) and is fixed at 1. Shares are RFC's expectation
    over product-error draws (Gumbel integrated analytically), over R_l fixed level draws."""

    def __init__(self, B, R_l, rng):
        self.B = B
        self.Z = [rng.standard_normal((len(B), R_l, L)) for L in LEVELS]
        self.Zn = rng.standard_normal((len(B), R_l))

    def shares(self, prods, sl, g, chunk=200):
        """sl: one sigma for all attributes (RFC-S) or (brand, flavor, pack) sigmas (RFC-G).
        "none" gets normal error of variance sum_a sl_a^2."""
        prods = np.asarray(prods, float)
        sl = np.broadcast_to(np.asarray(sl, float), (3,))
        X = task_X(prods)
        V = self.B @ X.T  # (N, K+1)
        out = np.zeros(X.shape[0])
        for s in range(0, len(self.B), chunk):
            e = sum(self.Z[a][s:s + chunk][:, :, prods[:, a].astype(int)] * sl[a] for a in range(3))
            e = np.concatenate([e, np.sqrt(np.sum(sl ** 2)) * self.Zn[s:s + chunk][:, :, None]], 2)
            u = (V[s:s + chunk, None, :] + e) / g
            u -= u.max(2, keepdims=True)
            p = np.exp(u)
            out += (p / p.sum(2, keepdims=True)).mean(1).sum(0)
        return out / len(self.B)


def tune_rfc(rfc_obj, H, obs, lo=(0.01, 0.05), hi=(5.0, 5.0)):
    """Minimize holdout share MAE over (sigma_l, g): log grid, then Nelder-Mead."""
    llo, lhi = np.log(lo), np.log(hi)

    def mae(x):
        x = np.clip(x, llo, lhi)
        sl, g = np.exp(x)
        return np.mean([np.abs(rfc_obj.shares(H[i], sl, g) - obs[i]).mean() for i in range(len(H))])

    grid = [(a, b) for a in np.linspace(llo[0], lhi[0], 6) for b in np.linspace(llo[1], lhi[1], 6)]
    vals = [mae(np.array(x)) for x in grid]
    x0 = np.array(grid[int(np.argmin(vals))])
    res = minimize(mae, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=200))
    x = np.clip(res.x, llo, lhi)
    flags = [f"param {i} at bound" for i in range(2) if x[i] - llo[i] < .05 or lhi[i] - x[i] < .05]
    return np.exp(x), res.fun, flags


# ---------------------------------------------------------------------------
# Metrics and decisions
# ---------------------------------------------------------------------------
def pair_corr_probit(sig):
    """Within-respondent error correlation of A and A' (same flavor and pack)."""
    return (sig[1] ** 2 + sig[2] ** 2) / np.sum(np.square(sig))


def pair_corr_rfc(sl, g):
    return 2 * sl ** 2 / (3 * sl ** 2 + np.pi ** 2 * g ** 2 / 6)


def decisions(share_fn):
    """share_fn(prods (K,4)) -> (K+1,) shares. Returns the decision-layer quantities."""
    out = {}
    prof = []
    for p in PRICES:
        mk = BASE.copy()
        mk[0, 3] = p
        prof.append((p - COST) * share_fn(mk)[0])
    out["price_profit"] = prof
    base = share_fn(BASE)
    ext = share_fn(np.vstack([BASE, CAND["A''"]]))
    out["ext_incr"] = (ext[0] + ext[5] - base[0]) / ext[5]
    contrib = [sum((BASE[j, 3] - COST) * base[j] for j in OWN_COST)]
    for c in CAND.values():
        s = share_fn(np.vstack([BASE, c]))
        contrib.append(sum((BASE[j, 3] - COST) * s[j] for j in OWN_COST) + (c[3] - COST) * s[5])
    out["cost_contrib"] = contrib  # [no add, A', A'', F, G]
    return out


def score(dec, truth):
    tp, pp = np.array(truth["price_profit"]), np.array(dec["price_profit"])
    tc, pc = np.array(truth["cost_contrib"]), np.array(dec["cost_contrib"])
    return {
        "price_choice": float(PRICES[pp.argmax()]),
        "price_flip": bool(pp.argmax() != tp.argmax()),
        "price_regret_pct": float(100 * (tp.max() - tp[pp.argmax()]) / tp.max()),
        "ext_incr": float(dec["ext_incr"]),
        "ext_incr_err": float(dec["ext_incr"] - truth["ext_incr"]),
        "ext_flip_at_placeholder_h": bool((dec["ext_incr"] >= HURDLE) != (truth["ext_incr"] >= HURDLE)),
        "cost_choice": int(pc.argmax()),
        "cost_flip": bool(pc.argmax() != tc.argmax()),
        "cost_regret_pct": float(100 * (tc.max() - tc[pc.argmax()]) / tc.max()),
    }


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------
def replicate(rep, truth_name, N, T, hb_iters, R_l, msl_R, work, selftest, ghk=None, scale="snu"):
    sig = np.sqrt(T_VAR * TRUTHS[truth_name])
    logit = truth_name in LOGIT_TRUTHS
    ss = np.random.SeedSequence([SEED, rep])
    r_data, r_hold, r_rfc = [np.random.default_rng(s) for s in ss.spawn(3)]
    rec = dict(rep=rep, truth=truth_name, N=N, T=T, sigma_true=sig.tolist())
    log(f"rep {rep} [{truth_name}]: simulate N={N}, T={T}")
    d = make_data(r_data, N, T, sig, logit)
    H = holdout_tasks(r_hold)
    yh = simulate(r_hold, np.broadcast_to(H, (N,) + H.shape), d["beta"], sig, logit)
    obs = np.stack([np.bincount(yh[:, i], minlength=4) / N for i in range(6)])
    none_share = float(np.mean(d["y"] == 3))
    rec["none_share_train"] = none_share
    log(f"  none share (training) {none_share:.3f}")

    # truth
    if logit:
        truth_fn = mixed_logit_shares
    else:
        truth_fn = lambda pr: pop_shares(pr, B_TRUE, W_TRUE, sig)  # noqa: E731
        sums = [rfc.orthant_cdf(*_raw(pr, B_TRUE, W_TRUE, sig)).sum() for pr in [BASE, H[0]]]
        rec["truth_raw_share_sums"] = [float(s) for s in sums]
        log(f"  truth raw share sums (before renormalization) {np.round(sums, 4)}")
        # Pipeline check, not an engine test: raw sums outside the engine's recorded range
        # (research/rfc_vs_exact: scenario sums 0.995 mean, min 0.969) point to a coding error.
        if selftest and any(abs(s - 1) > 0.035 for s in sums):
            fail(f"truth shares sum to {sums} (|sum - 1| > 0.035)")
    tdec = decisions(truth_fn)
    rec["truth"] = dict(S1=float(pair_corr_probit(sig)), **{k: (np.array(v).tolist()) for k, v in tdec.items()})

    th_true = theta_true(sig, scale)
    if ghk is not None:
        # MVP: exact panel likelihood by GHK, already fitted on GPU (ghk_gpu.fit)
        th_hat = np.array(ghk["theta"])
        ll_fit, ll_true = -ghk["nll"], -ghk["nll_true"]
        se_sand, se_hess = np.array(ghk["se_sandwich"]), np.array(ghk["se_hessian"])
        converged, iters = ghk["n_iter"] < ghk["max_iter"], ghk["n_iter"]
        t_mvp, t_se = ghk["seconds"]["fit"], ghk["seconds"]["se"]
        log(f"  MVP (GHK, M={ghk['M']}, GPU): {iters} iters, fit {t_mvp:.0f} s, max|grad| {ghk['grad_max']:.2e}")
    else:
        # MVP: probit by full MLE (simulated panel likelihood, engine at 3 dims)
        log(f"  MVP: build MSL (R={msl_R} draws)")
        panel = MSL(d["prods"], d["y"], msl_R, seed=SEED + rep)
        th0 = np.r_[np.zeros(P), np.full(P, np.log(.5)), np.full(3, np.log(.5))]
        res, t_mvp = fit_probit(panel, sig[3], th0, "MVP")
        th_hat, converged, iters = res.x, bool(res.success), int(res.nit)
        ll_fit, ll_true = -res.fun, panel.ll_vec(B_TRUE, W_TRUE, sig).sum()
        t0 = time.time()
        se_sand, se_hess = probit_se(panel, res.x, sig[3])
        t_se = time.time() - t0
        del panel
    b, w, s_hat = unpack_total(th_hat) if scale == "total" else unpack(th_hat, sig[3])
    log(f"  MVP ll fit {ll_fit:.3f} vs truth {ll_true:.3f}")
    if selftest and not logit and ll_fit < ll_true - 1e-3:
        fail("MVP log-likelihood at the fit is below the truth's")
    mvp_fn = lambda pr: pop_shares(pr, b, w, s_hat)  # noqa: E731
    mdec = decisions(mvp_fn)
    mh = np.stack([mvp_fn(H[i]) for i in range(6)])
    rec["mvp"] = dict(theta=th_hat.tolist(), theta_true=th_true.tolist(), se_sandwich=se_sand.tolist(),
                      se_hessian=se_hess.tolist(), sigma=s_hat.tolist(), ll=ll_fit, ll_true=float(ll_true),
                      converged=bool(converged), iters=int(iters),
                      likelihood="ghk" if ghk is not None else f"msl R={msl_R}",
                      ghk_M=ghk["M"] if ghk is not None else None,
                      normalisation="total error variance = pi^2/6" if scale == "total" else "sigma_nu fixed",
                      S1=float(pair_corr_probit(s_hat)),
                      S2=(s_hat ** 2 / np.sum(s_hat ** 2)).tolist(),
                      holdout_mae=float(np.abs(mh - obs).mean()),
                      seconds=dict(fit=t_mvp, se=t_se), **score(mdec, tdec))
    if scale == "total":
        # delta method, as mvp_refit.py: log sigma_a = (log T_VAR + u_a - logsumexp(u, 0)) / 2
        u = th_hat[2 * P:]
        J = 0.5 * (np.eye(3) - softmax0(u)[None, :3])
        V = np.array(ghk["cov_sandwich"])[2 * P:, 2 * P:]
        se_log_sig = np.sqrt(np.diag(J @ V @ J.T))
        z_sig = (np.log(s_hat[:3]) - np.log(np.maximum(sig[:3], 1e-300))) / se_log_sig
        rec["mvp"].update(se_log_sigma=se_log_sig.tolist(), z_sigma=z_sig.tolist(),
                          S4_cover=(np.abs(z_sig) <= 1.96).tolist() if sig[0] > 0 else None)

    # HB-MNL + RFC-S
    log(f"  HB-MNL: {hb_iters} iterations (bayesm)")
    B, llh, t_hb = run_hb(work, d["prods"], d["y"], hb_iters, 1000 + rep)
    q = len(llh) // 4
    drift = abs(llh[3 * q:].mean() - llh[2 * q:3 * q].mean()) / abs(llh[3 * q:].mean())
    log(f"  HB done {t_hb:.0f} s; loglike drift q3->q4 {drift:.4f}")
    if selftest and drift > 0.01:
        fail(f"HB log-likelihood still drifting ({drift:.4f} > 0.01)")
    t0 = time.time()
    R = RFC(B, R_l, r_rfc)
    (sl, g), mae, flags = tune_rfc(R, H, obs)
    t_rfc = time.time() - t0
    log(f"  RFC-S tuned sigma_l {sl:.3f}, g {g:.3f}, holdout MAE {mae:.4f}, {t_rfc:.0f} s {flags}")
    if selftest and flags:
        fail(f"RFC tuning at a bound: {flags}")
    rdec = decisions(lambda pr: R.shares(pr, sl, g))
    rec["rfc_s"] = dict(sigma_l=float(sl), g=float(g), holdout_mae=float(mae), flags=flags,
                        S1=float(pair_corr_rfc(sl, g)), hb_loglike_drift=float(drift),
                        seconds=dict(hb=t_hb, tune=t_rfc), **score(rdec, tdec))
    for k in ("mvp", "rfc_s"):
        rec[k]["S1_err"] = rec[k]["S1"] - rec["truth"]["S1"]
        rec[k]["S3_ratio"] = rec[k]["S1"] / rec["truth"]["S1"] if rec["truth"]["S1"] else None
    return rec


def _raw(prods, b, w, sig):
    """Unnormalized orthant problems for pop_shares (for the sum check)."""
    X = task_X(np.asarray(prods, float))
    mu = X @ b
    C = X @ np.diag(w ** 2) @ X.T + err_cov(prods, sig)
    K1 = len(mu)
    ups, covs = [], []
    for j in range(K1):
        oth = [k for k in range(K1) if k != j]
        M = np.zeros((K1 - 1, K1))
        M[np.arange(K1 - 1), oth] = 1
        M[:, j] = -1
        ups.append(-(M @ mu))
        covs.append(M @ C @ M.T)
    return np.array(ups), np.array(covs)


def recovery_check(N, T, truth_name, msl_R):
    """MVP alone on a larger sample: every sigma_a within 2 sandwich SEs of the truth."""
    sig = np.sqrt(T_VAR * TRUTHS[truth_name])
    d = make_data(np.random.default_rng([SEED, 999]), N, T, sig)
    log(f"recovery check: MVP at N={N}")
    panel = MSL(d["prods"], d["y"], msl_R, seed=SEED + 999)
    th0 = np.r_[np.zeros(P), np.full(P, np.log(.5)), np.full(3, np.log(.5))]
    res, t = fit_probit(panel, sig[3], th0, f"MVP N={N}")
    se, _ = probit_se(panel, res.x, sig[3])
    z = (res.x[2 * P:] - np.log(sig[:3])) / se[2 * P:]
    zall = (res.x - np.r_[B_TRUE, np.log(W_TRUE), np.log(sig[:3])]) / se
    log(f"  sigma hat {np.exp(res.x[2 * P:]).round(3)} vs true {sig[:3].round(3)}; z {z.round(2)}")
    return dict(N=N, theta=res.x.tolist(), se_sandwich=se.tolist(),
                sigma_hat=np.exp(res.x[2 * P:]).tolist(), sigma_true=sig[:3].tolist(),
                z_sigma=z.tolist(), z_all=zall.tolist(), seconds=t, ok=bool(np.all(np.abs(z) <= 2)))


def ghk_fits(reps, truth_name, N, T, M, scale="snu"):
    """MVP fits for every replicate, in parallel on Modal GPUs (ghk_gpu.fit). The data are the
    replicate's own: make_data on the same r_data stream replicate() uses."""
    import ghk_gpu

    sig = np.sqrt(T_VAR * TRUTHS[truth_name])
    th0 = np.r_[np.zeros(P), np.full(P, np.log(.5)), np.zeros(3) if scale == "total" else np.full(3, np.log(.5))]
    th_true = theta_true(sig, scale)
    snu = 0.0 if scale == "total" else float(sig[3])  # unused under tvar
    kw = dict(tvar=float(T_VAR)) if scale == "total" else {}
    jobs = []
    for rep in reps:
        r_data = np.random.default_rng(np.random.SeedSequence([SEED, rep]).spawn(3)[0])
        d = make_data(r_data, N, T, sig, truth_name in LOGIT_TRUTHS)
        Xd, G = ghk_gpu.panel_arrays(d["prods"], d["y"], task_X)
        jobs.append((Xd, G, snu, th0, th_true, M, SEED + rep))
    log(f"MVP: {len(jobs)} GHK fits on GPU (M={M}, scale {scale}), in parallel")
    t0 = time.time()
    with ghk_gpu.app.run():
        res = list(ghk_gpu.fit.starmap(jobs, kwargs=kw))
    log(f"MVP GPU fits done: {time.time() - t0:.0f} s wall")
    return {rep: dict(r, max_iter=500) for rep, r in zip(reps, res)}


def manifest(out, args):
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def sh(*c):
        return subprocess.run(c, capture_output=True, text=True, cwd=ROOT).stdout.strip()

    import scipy
    import multivariate_probit
    m = dict(args=vars(args), git_head=sh("git", "rev-parse", "HEAD"),
             git_status_T1=sh("git", "status", "--porcelain", "analysis/T1"),
             sha256={f: sha(HERE / f) for f in ("run.py", "hb.R")},
             sha256_rfc_py=sha(ROOT / "research" / "rfc_vs_exact" / "rfc.py"),
             orthant_backend=rfc.BACKEND, numpy=np.__version__, scipy=scipy.__version__,
             multivariate_probit=multivariate_probit.__version__,
             R=sh("Rscript", "-e", "cat(R.version.string, as.character(packageVersion('bayesm')))"),
             python=sys.version.split()[0], host=os.uname().nodename,
             started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / ("manifest_recovery.json" if args.recovery_only else "manifest.json")).write_text(json.dumps(m, indent=1))


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--truth", default="G1", choices=list(TRUTHS))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--N", type=int, default=600)
    ap.add_argument("--recovery-N", type=int, default=1000)
    ap.add_argument("--msl-R", type=int, default=0, help="MSL draws per respondent (default 200 selftest, 500 full)")
    ap.add_argument("--recovery-only", action="store_true", help="selftest: run only the recovery check")
    ap.add_argument("--no-recovery", action="store_true", help="selftest: skip the MVP recovery check")
    ap.add_argument("--mvp", default="ghk", choices=["ghk", "msl"], help="MVP likelihood (see module doc)")
    ap.add_argument("--ghk-M", type=int, default=4096, help="GHK Sobol points per respondent")
    ap.add_argument("--scale", default="total", choices=["total", "snu"],
                    help="GHK normalisation: total error variance fixed (default) or sigma_nu pinned (original T1)")
    ap.add_argument("--out-name", default="", help="output dir under out/ (default selftest or full_<truth>)")
    args = ap.parse_args()
    if args.mvp == "msl" and args.scale == "total":
        ap.error("--scale total needs --mvp ghk; use --scale snu with --mvp msl")
    if args.selftest:
        args.reps, args.N = 1, 100
    T = 12
    hb_iters = 10000 if args.selftest else 20000
    R_l = 200 if args.selftest else 1000
    msl_R = args.msl_R or (200 if args.selftest else 500)
    out = HERE / "out" / (args.out_name or ("selftest" if args.selftest else f"full_{args.truth}"))
    out.mkdir(parents=True, exist_ok=True)
    LOGF = open(out / ("log_recovery.txt" if args.recovery_only else "log.txt"), "w")
    manifest(out, args)
    log(f"T1 {'selftest' if args.selftest else 'full'}: truth {args.truth}, reps {args.reps}, N {args.N}, "
        f"backend {rfc.BACKEND}")
    if args.selftest and rfc.BACKEND != "local":
        fail("selftest needs the local orthant binary")
    results = []
    reps = range(0 if args.recovery_only else args.reps)
    ghk = {}
    if args.mvp == "ghk" and len(reps):
        ghk = ghk_fits(reps, args.truth, args.N, T, args.ghk_M, args.scale)
    for rep in reps:
        results.append(replicate(rep, args.truth, args.N, T, hb_iters, R_l, msl_R, out, args.selftest,
                                 ghk.get(rep), args.scale if args.mvp == "ghk" else "snu"))
        (out / "results.json").write_text(json.dumps(results, indent=1))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log(f"rep {rep} saved; peak RSS {rss:.0f} MB")
        if args.selftest and rss > 1024:
            fail(f"peak RSS {rss:.0f} MB > 1 GB")
    if args.selftest and not args.no_recovery:
        rc = recovery_check(args.recovery_N, T, args.truth, msl_R)
        (out / "recovery.json").write_text(json.dumps(rc, indent=1))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log(f"recovery {'OK' if rc['ok'] else 'NOT OK'}; peak RSS {rss:.0f} MB")
        if not rc["ok"]:
            fail("MVP does not recover sigma within 2 SE")
        if rss > 1024:
            fail(f"peak RSS {rss:.0f} MB > 1 GB")
    log("done")


if __name__ == "__main__":
    main()
