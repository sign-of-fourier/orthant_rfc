"""Does a similarity kernel on product error fix the near-substitute problem?

Design modeled on Huber, Orme & Miller (1999): a TV study with 6 attributes,
effects-coded. Respondent part-worths beta_n ~ N(mu, Omega). Holdouts are
4 utility-balanced alternatives plus a near-copy ("twin") of one of them.

Truths (data-generating processes), none of them the kernel fix:
  merge    soft perceptual merging: i and j are seen as one option with
           probability m(d_ij) = 1 / (1 + exp((d_ij - d0) / s)); merged
           alternatives share one product-error draw (red bus / blue bus)
  laplace  product-error correlation from a Laplace kernel on *perceptual*
           weights that differ from |mu| (wrong shape, wrong metric)
  rfc      U = x'(beta_n + e_A) + e_P, e_P iid normal (no extra similarity)
  gumbel   as rfc with Gumbel e_P at matched variance (Sawtooth's world)

Fixes, all exact probit given beta_n (Stage 1: oracle beta_n):
  none     iid probit, sigma_P only
  rfc      Sigma_A = s_A^2 I, sigma_P
  kernel   rfc + Cov(e_P) = sigma_P^2 [(1 - tau) I + tau exp(-d^2 / ell^2)]
The oracle row is the truth's own probabilities (Monte Carlo).

Choice probabilities: P(choose k) is a 4-dimensional MVN orthant of the
utility differences (rfc_vs_exact/rfc.py, local paid backend).
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rfc_vs_exact"))
from rfc import orthant_cdf, diff_ops  # noqa: E402

# ---------------------------------------------------------------------------
# Design. Part-worths per level under mu (effects coding: they sum to 0).
# Importances (range share): price .35, brand .28, screen .16, sound .10,
# pip .07, blockout .04.
# ---------------------------------------------------------------------------
ATTRS = [
    ("brand", np.array([0.9, 0.1, -1.0])),
    ("screen", np.array([-0.6, 0.1, 0.5])),
    ("sound", np.array([-0.4, 0.1, 0.3])),
    ("pip", np.array([0.25, -0.25])),
    ("blockout", np.array([0.15, -0.15])),
    ("price", np.array([1.2, 0.4, -0.4, -1.2])),
]
NA = len(ATTRS)
NLEV = [len(pw) for _, pw in ATTRS]
P = sum(n - 1 for n in NLEV)  # 11 coefficients
MU = np.concatenate([pw[:-1] for _, pw in ATTRS])
RANGE = np.array([np.ptp(pw) for _, pw in ATTRS])
LEAST = list(np.argsort(RANGE))  # blockout, pip, sound, ...
K = 5
# Heterogeneity: sd per coefficient, half the attribute's half-range plus 0.1.
OMEGA_SD = np.concatenate([np.full(n - 1, 0.25 * r + 0.1) for n, r in zip(NLEV, RANGE)])
# Perceptual weights for the laplace truth: brand looms larger, price smaller
# than their part-worth ranges say (per mismatched attribute).
PERCEPT_W = np.array([2.0, 1.0, 0.6, 0.4, 0.4, 0.8])


def encode(levels):
    """levels (..., NA) int -> effects-coded X (..., P)."""
    levels = np.asarray(levels)
    cols = []
    for a, n in enumerate(NLEV):
        l = levels[..., a]
        for c in range(n - 1):
            cols.append(np.where(l == c, 1.0, np.where(l == n - 1, -1.0, 0.0)))
    return np.stack(cols, axis=-1)


def utility_mu(levels):
    levels = np.asarray(levels)
    return sum(ATTRS[a][1][levels[..., a]] for a in range(NA))


def distance(li, lj):
    """|mu|-weighted attribute distance: sum_a |pw_a(l_i) - pw_a(l_j)|."""
    return sum(abs(ATTRS[a][1][li[a]] - ATTRS[a][1][lj[a]]) for a in range(NA))


def percept_distance(li, lj):
    return float(PERCEPT_W @ (np.asarray(li) != np.asarray(lj)))


def dist_matrix(L, f=distance):
    k = len(L)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            D[i, j] = D[j, i] = f(L[i], L[j])
    return D


# ---------------------------------------------------------------------------
# Holdouts
# ---------------------------------------------------------------------------
def balanced_four(rng, spread=0.3, offset=0.0, src=0):
    """4 level-balanced alternatives with utilities (under mu) within `spread`;
    alternative `src` sits `offset` above the mean of the other three."""
    while True:
        L = np.zeros((4, NA), int)
        for a, n in enumerate(NLEV):
            base = {2: [0, 0, 1, 1], 3: [0, 1, 2, rng.integers(3)], 4: [0, 1, 2, 3]}[n]
            L[:, a] = rng.permutation(base)
        u = utility_mu(L)
        others = np.delete(u, src)
        if np.ptp(others) < spread and abs(u[src] - others.mean() - offset) < spread / 2:
            return L


def make_twin(src, k, rng):
    """Copy of src changed on the k least important attributes, levels chosen
    to minimize the utility change. Returns levels, delta-u."""
    best, best_du = None, np.inf
    attrs = LEAST[:k]
    for combo in itertools.product(*[[l for l in range(NLEV[a]) if l != src[a]] for a in attrs]):
        t = src.copy()
        t[attrs] = combo
        du = utility_mu(t) - utility_mu(src)
        if abs(du) < abs(best_du) - 1e-12 or (abs(abs(du) - abs(best_du)) < 1e-12 and rng.random() < 0.5):
            best, best_du = t, du
    if k == 0:
        best, best_du = src.copy(), 0.0
    return best, float(best_du)


def make_holdouts(rng):
    """Per twin level k in 0..3: 2 balanced, 1 pair-above (+0.6), 1 pair-below;
    2 triplet tasks (3 near-copies at k=1); 2 repeats. Returns list of dicts."""
    tasks = []
    max_du = {0: 0.0, 1: 0.3, 2: 0.2, 3: 0.2}  # k=1 can only move by +-0.3 (blockout)
    for k in range(4):
        for i, (kind, off) in enumerate([("bal", 0.0), ("bal", 0.0), ("up", 0.6), ("down", -0.6)]):
            sign = 1 if i % 2 == 0 else -1  # alternate the twin's offset direction
            while True:
                L4 = balanced_four(rng, offset=off)
                tw, du = make_twin(L4[0].copy(), k, rng)
                if abs(du) <= max_du[k] + 1e-9 and (abs(du) < 1e-9 or np.sign(du) == sign):
                    break
            L = np.vstack([L4, tw])
            tasks.append(dict(L=L, k=k, kind=kind, twins=(0, 4), du=du))
    for _ in range(2):
        L3 = balanced_four(rng)[:3]
        t1, _ = make_twin(L3[0].copy(), 1, rng)
        t2 = t1.copy()
        a = LEAST[1]  # also vary pip relative to t1 so all three differ
        t2[a] = 1 - t2[a]
        L = np.vstack([L3, t1, t2])
        tasks.append(dict(L=L, k=1, kind="triplet", twins=(0, 3, 4), du=float(utility_mu(t1) - utility_mu(L3[0]))))
    for i in (0, 8):  # repeat a k=0 and a k=2 balanced task
        tasks.append(dict(tasks[i], kind="repeat", of=i))
    for t in tasks:
        t["X"] = encode(t["L"])
        t["D"] = dist_matrix(t["L"])
        t["Dp"] = dist_matrix(t["L"], percept_distance)
        t["d_twin"] = t["D"][t["twins"][0], t["twins"][1]]
    return tasks


def respondents(n, rng):
    return MU + OMEGA_SD * rng.standard_normal((n, P))


# ---------------------------------------------------------------------------
# Truths: simulate utilities (N, R, K) and choices
# ---------------------------------------------------------------------------
TRUTH_DEFAULTS = dict(
    merge=dict(s_a=0.3, sigma=0.6, d0=0.6, s=0.2),
    laplace=dict(s_a=0.3, sigma=0.6, tau=0.8, ell=1.0),
    rfc=dict(s_a=0.3, sigma=0.6),
    gumbel=dict(s_a=0.3, sigma=0.6),
)


def _components(adj):
    """Connected-component label (min index) for adjacency (..., K, K) bool."""
    reach = adj | np.eye(adj.shape[-1], dtype=bool)
    for _ in range(3):  # K=5: paths of length <= 4
        reach = (reach.astype(np.uint8) @ reach.astype(np.uint8)) > 0
    return np.argmax(reach, axis=-1)  # first reachable index


def draw_utilities(task, B, truth, prm, R, rng):
    X, D = task["X"], task["D"]
    N = len(B)
    V = B @ X.T                                                  # (N, K)
    SA = prm["s_a"] ** 2 * X @ X.T
    eA = rng.standard_normal((N, R, K)) @ np.linalg.cholesky(SA + 1e-12 * np.eye(K)).T
    sig = prm["sigma"]
    if truth in ("rfc", "merge"):
        eP = sig * rng.standard_normal((N, R, K))
        if truth == "merge":
            m = 1.0 / (1.0 + np.exp((D - prm["d0"]) / prm["s"]))
            iu = np.triu_indices(K, 1)
            adj = np.zeros((N, R, K, K), bool)
            hit = rng.random((N, R, len(iu[0]))) < m[iu]
            adj[..., iu[0], iu[1]] = hit
            adj |= np.swapaxes(adj, -1, -2)
            lab = _components(adj)
            eP = np.take_along_axis(eP, lab, axis=-1)
    elif truth == "gumbel":
        eP = sig * np.sqrt(6) / np.pi * rng.gumbel(size=(N, R, K))
    elif truth == "laplace":
        C = sig ** 2 * ((1 - prm["tau"]) * np.eye(K) + prm["tau"] * np.exp(-task["Dp"] / prm["ell"]))
        eP = rng.standard_normal((N, R, K)) @ np.linalg.cholesky(C).T
    else:
        raise ValueError(truth)
    return V[:, None, :] + eA + eP


def true_probs(task, B, truth, prm, R, rng, chunk=200):
    """Monte Carlo choice probabilities (N, K) under the truth."""
    out = []
    for s in range(0, len(B), chunk):
        U = draw_utilities(task, B[s:s + chunk], truth, prm, R, rng)
        out.append(np.stack([(U.argmax(-1) == k).mean(1) for k in range(K)], -1))
    return np.vstack(out)


def simulate_choices(tasks, B, truth, prm, rng):
    return np.column_stack([draw_utilities(t, B, truth, prm, 1, rng)[:, 0].argmax(-1) for t in tasks])


# ---------------------------------------------------------------------------
# Orthant engine. orthant_cdf errs by up to ~0.02 on the twin tasks (rho near
# 1), the size of the effect studied, so the default here is Genz's separation
# of variables with a fixed randomized Korobov-type lattice: common random
# numbers across calls keep the likelihood smooth in the parameters.
# ---------------------------------------------------------------------------
from scipy.special import ndtr, ndtri  # noqa: E402

_LAT = {}


def _lattice(d, m=1024, shifts=4, seed=12345):
    """Fixed scrambled Sobol point sets, (m, d) each, one per independent scramble."""
    from scipy.stats import qmc
    key = (d, m, shifts)
    if key not in _LAT:
        _LAT[key] = [qmc.Sobol(d, scramble=True, seed=seed + r).random(m) for r in range(shifts)]
    return _LAT[key]


def genz_orthant(upper, cov, m=1024, shifts=4):
    """P(Z <= upper[n]), Z ~ N(0, cov), cov (d, d) shared, upper (N, d). Returns
    (N,) estimates; shifts give a standard error if wanted (genz_orthant_se)."""
    return _genz(upper, cov, m, shifts)[0]


def _genz(upper, cov, m=1024, shifts=4):
    upper = np.atleast_2d(upper)
    d = upper.shape[1]
    # order variables by tightest standardized limit (on the mean respondent)
    sd = np.sqrt(np.diag(cov))
    order = np.argsort((upper / sd).mean(0))
    upper, cov = upper[:, order], cov[np.ix_(order, order)]
    L = np.linalg.cholesky(cov)
    ests = []
    for W in _lattice(d - 1, m, shifts):
        M = len(W)
        e = ndtr(upper[:, 0] / L[0, 0])[:, None] * np.ones((1, M))       # (N, M)
        y = np.zeros(upper.shape[:1] + (M, d))
        f = e.copy()
        for i in range(1, d):
            y[:, :, i - 1] = ndtri(np.clip(W[None, :, i - 1] * e, 1e-300, 1 - 1e-16))
            s = (y[:, :, :i] @ L[i, :i])
            e = ndtr((upper[:, i][:, None] - s) / L[i, i])
            f = f * e
        ests.append(f.mean(1))
    ests = np.array(ests)
    return ests.mean(0), ests.std(0, ddof=1) / np.sqrt(len(ests))


ENGINE = os.environ.get("SIM_ENGINE", "genz")


GENZ_M = 256  # fitting; evaluation sets GENZ_M = 1024 (max err 5e-4 vs 1.4e-4)


def orthant(upper, cov):
    return genz_orthant(upper, cov, m=GENZ_M) if ENGINE == "genz" else orthant_cdf(upper, cov)


# ---------------------------------------------------------------------------
# Fixes: exact probit probabilities
# ---------------------------------------------------------------------------
_OPS = diff_ops(K)


def fix_cov(task, model, th):
    X, D = task["X"], task["D"]
    S = th["s_a"] ** 2 * X @ X.T if model != "none" else np.zeros((K, K))
    if model == "kernel":
        S = S + th["sigma"] ** 2 * ((1 - th["tau"]) * np.eye(K) + th["tau"] * np.exp(-D ** 2 / th["ell"] ** 2))
    else:
        S = S + th["sigma"] ** 2 * np.eye(K)
    return S


def fix_probs(task, B, model, th):
    """(N, K) exact probit probabilities."""
    V = B @ task["X"].T
    S = fix_cov(task, model, th)
    p = np.column_stack([orthant(V @ A.T, A @ S @ A.T) for A in _OPS])
    return p if ENGINE == "genz" else p / p.sum(1, keepdims=True)


def fix_loglik(tasks, B, Y, model, th, floor=1e-10):
    """Sum of log P(chosen); only the chosen alternative's orthant is computed."""
    ll = 0.0
    for t, task in enumerate(tasks):
        V = B @ task["X"].T
        S = fix_cov(task, model, th)
        for k, A in enumerate(_OPS):
            sel = Y[:, t] == k
            if sel.any():
                ll += np.log(np.maximum(orthant(V[sel] @ A.T, A @ S @ A.T), floor)).sum()
    return ll


PARAMS = dict(none=["sigma"], rfc=["s_a", "sigma"], kernel=["s_a", "sigma", "tau", "ell"])
_SIG = lambda x: 1 / (1 + np.exp(-x))


def _theta(model, z):
    th = {}
    for name, v in zip(PARAMS[model], z):
        th[name] = _SIG(v) if name == "tau" else np.exp(v)
    return th


def fit(tasks, B, Y, model, start=None, maxiter=400, step=0.5):
    z0 = start if start is not None else np.array(
        [{"s_a": np.log(0.3), "sigma": np.log(0.6), "tau": 0.0, "ell": np.log(1.0)}[p] for p in PARAMS[model]])
    simplex = np.vstack([z0] + [z0 + step * np.eye(len(z0))[i] for i in range(len(z0))])
    res = minimize(lambda z: -fix_loglik(tasks, B, Y, model, _theta(model, z)), z0, method="Nelder-Mead",
                   options=dict(maxiter=maxiter, initial_simplex=simplex, xatol=1e-2, fatol=0.01))
    return _theta(model, res.x), -res.fun, res
