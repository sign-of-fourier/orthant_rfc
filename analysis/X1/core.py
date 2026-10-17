"""X1 core: camera data, level-pair splits, probit likelihood (variance-share Sigma, GHK), arms.

Design: analysis/X1/DESIGN.md. Utilities are relative to "none" (alternative 4, utility 0). Every
alternative has total error variance 1: cameras split it into attribute shares s_a (shared by cameras
with the same level of attribute a in a task) plus an own share s_nu; "none" has an independent error.
"""
import time
from pathlib import Path

import numpy as np
import torch
from torch.special import log_ndtr, ndtri

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 20261015
COLS = ["canon", "sony", "nikon", "panasonic", "pixels", "zoom", "video", "swivel", "wifi", "price"]
BRANDS = COLS[:4]
FEATS = COLS[4:9]
PRICES = np.array([0.79, 1.29, 1.79, 2.29, 2.79])
SIM_ATTRS = ["brand"] + FEATS          # attributes with an error share (price excluded, as in T1)
NA = 5                                  # alternatives per task (4 cameras + none)
NONE = 4
F64 = torch.float64

torch.set_num_threads(2)


# --------------------------------------------------------------------------- data
class Data:
    """Task-level arrays. X (T, 5, 10), y (T,), resp (T,), lev (T, 4, 7) attribute level ids of cameras."""

    def __init__(self, path=OUT / "data" / "camera_long.csv"):
        d = np.genfromtxt(path, delimiter=",", names=True)
        T = int(d["task"].size // NA)
        order = np.lexsort((d["alt"], d["task"], d["resp"]))
        d = d[order]
        self.X = np.stack([d[c] for c in COLS], 1).reshape(T, NA, len(COLS))
        self.y = d["chosen"].reshape(T, NA).argmax(1)
        self.resp = d["resp"].reshape(T, NA)[:, 0].astype(int)
        cams = self.X[:, :4]
        brand = cams[:, :, :4].argmax(2)
        feats = cams[:, :, 4:9].astype(int)
        price = np.abs(cams[:, :, 9:10] - PRICES).argmin(2)
        self.brand, self.price_lev = brand, price
        self.lev = np.concatenate([brand[:, :, None], feats], 2)        # (T, 4, 6)
        self.T = T

    def sub(self, idx):
        o = object.__new__(Data)
        for k in ("X", "y", "resp", "brand", "price_lev", "lev"):
            setattr(o, k, getattr(self, k)[idx])
        o.T = len(idx)
        return o


def sim_mats(lev):
    """(T, 6, 5, 5) indicator: alternatives i, j (cameras) share the level of attribute a. None row/col 0."""
    T = lev.shape[0]
    S = np.zeros((T, lev.shape[2], NA, NA))
    S[:, :, :4, :4] = (lev[:, :, None, :] == lev[:, None, :, :]).transpose(0, 3, 1, 2)
    return S


# --------------------------------------------------------------------------- splits
def pair_mask(data, pair):
    """Camera-level mask (T, 4) of alternatives having the level pair. pair = (brand, kind, value)."""
    b, kind, v = pair
    has_b = data.brand == BRANDS.index(b)
    if kind == "price":
        return has_b & (data.price_lev == v)
    return has_b & (data.X[:, :4, COLS.index(kind)] == 1)


def families():
    bp = [(b, "price", v) for b in BRANDS for v in range(5)]
    bf = [(b, "feat", f) for b in BRANDS for f in FEATS]
    return {"brand_price": bp, "brand_feature": [(b, f, None) for b, _, f in bf]}


def pair_mask_any(data, pair):
    b, kind, v = pair
    if kind == "price":
        return pair_mask(data, pair)
    return pair_mask(data, (b, kind, None))


def pair_name(pair):
    b, kind, v = pair
    return f"{b}@{PRICES[v]:.2f}" if kind == "price" else f"{b}+{kind}"


def make_splits(data, seed=SEED):
    """Level-pair splits + matched random-task references. Validation pair: a brand x price pair with a
    different brand from the test pair (fixed rule), removed from training for tuning only."""
    rng = np.random.default_rng(seed)
    fam = families()
    bp = fam["brand_price"]
    out = []
    for fname, pairs in fam.items():
        for i, pair in enumerate(pairs):
            test = pair_mask_any(data, pair).any(1)
            # validation pair: brand x price, next brand, same price index rotation
            vb = BRANDS[(BRANDS.index(pair[0]) + 1) % 4]
            vpair = (vb, "price", i % 5)
            val = pair_mask_any(data, vpair).any(1) & ~test
            out.append(dict(family=fname, kind="pair", pair=pair, name=pair_name(pair), vpair=vpair,
                            test=np.flatnonzero(test), val=np.flatnonzero(val),
                            train=np.flatnonzero(~test)))
            # random reference: same number of test tasks, random; validation = same size random from train
            perm = rng.permutation(data.T)
            rt = np.sort(perm[:test.sum()])
            rest = np.setdiff1d(np.arange(data.T), rt)
            rv = np.sort(rng.choice(rest, size=int(val.sum()), replace=False))
            out.append(dict(family=fname, kind="random", pair=pair, name=pair_name(pair) + " (random)",
                            vpair=None, test=rt, val=rv, train=rest))
    return out


# --------------------------------------------------------------------------- probit likelihood
_SOBOL = {}


def sobol(M, dim=4, seed=SEED):
    k = (M, dim, seed)
    if k not in _SOBOL:
        _SOBOL[k] = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed).draw(M, dtype=F64)
    return _SOBOL[k]


class Probit:
    """Choice log-likelihood for a set of tasks under variance-share Sigma.

    shares: torch (7,) = (s_brand, s_pixels, ..., s_wifi, s_nu), sum 1 (or None for Sigma = I).
    V: torch (T, 5) utilities (none = 0)."""

    def __init__(self, data, M=512):
        self.T = data.T
        self.y = torch.as_tensor(data.y)
        self.S = torch.as_tensor(sim_mats(data.lev))                 # (T, 6, 5, 5)
        self.M = M
        # difference operator per task: rows e_j - e_c for j != c
        A = np.zeros((data.T, NA - 1, NA))
        for t, c in enumerate(data.y):
            others = [j for j in range(NA) if j != c]
            A[t, np.arange(NA - 1), others] = 1
            A[t, :, c] = -1
        self.A = torch.as_tensor(A)
        self.others = torch.as_tensor(np.array([[j for j in range(NA) if j != c] for c in data.y]))

    def sigma(self, shares):
        """(T, 5, 5): sum_a s_a S_a + diag(s_nu on cameras, 1 on none). Camera diagonals sum to 1."""
        T = self.T
        if shares is None:
            return torch.eye(NA, dtype=F64).expand(T, NA, NA)
        own = torch.cat([shares[-1].expand(4), torch.ones(1, dtype=F64)])
        return torch.einsum("a,taij->tij", shares[:-1], self.S) + torch.diag_embed(own).expand(T, NA, NA)

    def loglik(self, V, shares, idx=None, M=None, chunk=2048):
        """Per-task log P(observed choice). V (T, 5)."""
        M = M or self.M
        sob = sobol(M)
        tiny, one = torch.finfo(F64).tiny, 1.0 - torch.finfo(F64).eps
        Sig = self.sigma(shares)
        out = []
        rng = range(0, self.T, chunk)
        for lo in rng:
            hi = min(lo + chunk, self.T)
            A = self.A[lo:hi]
            Om = A @ Sig[lo:hi] @ A.transpose(1, 2)
            L = torch.linalg.cholesky(Om)
            Vc = V[lo:hi].gather(1, self.y[lo:hi, None])
            eta = Vc - V[lo:hi].gather(1, self.others[lo:hi])          # P(D_j < 0) = P(L z <= V_c - V_j)
            n = hi - lo
            logp = eta.new_zeros(n, M)
            us = []
            for j in range(NA - 1):
                c = eta[:, j, None]
                for i in range(j):
                    c = c - L[:, j, i, None] * us[i]
                lp = log_ndtr(c / L[:, j, j, None])
                logp = logp + lp
                if j < NA - 2:
                    us.append(ndtri((sob[None, :, j] * lp.exp()).clamp(tiny, one)))
            out.append(torch.logsumexp(logp, 1) - np.log(M))
        return torch.cat(out)


OWN_FLOOR = 0.01


def shares_from_logits(w):
    """7 shares (6 attributes + own) from 6 free logits; own share logit fixed at 0. The own share is floored
    at OWN_FLOOR so cameras identical on every attribute keep a nonsingular error covariance."""
    s = torch.softmax(torch.cat([w, w.new_zeros(1)]), 0)
    return (1 - OWN_FLOOR) * s + OWN_FLOOR * torch.nn.functional.one_hot(torch.tensor(len(w)), len(w) + 1).to(s)


# --------------------------------------------------------------------------- linear probit
def fit_linear(data, sigma_free=True, M=512, iters=200, beta0=None, w0=None):
    """P-LIN (sigma_free) or P-LIN-I. Returns dict(beta, shares, ll)."""
    pr = Probit(data, M)
    X = torch.as_tensor(data.X)
    beta = torch.zeros(len(COLS), dtype=F64, requires_grad=True) if beta0 is None else \
        torch.tensor(beta0, dtype=F64, requires_grad=True)
    w = (torch.full((len(SIM_ATTRS),), -3.0, dtype=F64) if w0 is None else torch.tensor(w0, dtype=F64)
         ).requires_grad_(sigma_free)
    params = [beta, w] if sigma_free else [beta]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        V = X @ beta
        s = shares_from_logits(w) if sigma_free else None
        loss = -pr.loglik(V, s).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        s = shares_from_logits(w) if sigma_free else None
        ll = pr.loglik(X @ beta, s).sum().item()
    return dict(beta=beta.detach().numpy().copy(), w=w.detach().numpy().copy(),
                shares=None if s is None else s.numpy().copy(), ll=ll)


def fit_shares(data, V, M=512, w0=None, iters=100):
    """Sigma step: variance shares with utilities V (T, 5) fixed."""
    pr = Probit(data, M)
    V = torch.as_tensor(V)
    w = (torch.full((len(SIM_ATTRS),), -3.0, dtype=F64) if w0 is None else torch.tensor(w0, dtype=F64)
         ).requires_grad_(True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -pr.loglik(V, shares_from_logits(w)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy().copy()


def probit_task_ll(data, V, shares, M=4096):
    """Evaluation log-likelihood per task, more draws."""
    pr = Probit(data, M)
    with torch.no_grad():
        s = None if shares is None else torch.as_tensor(shares)
        return pr.loglik(torch.as_tensor(V), s, chunk=512).numpy()


def probit_choice_probs(data, V, shares, M=4096):
    """P(each alternative) per task: (T, 5), by evaluating every alternative as the choice."""
    P = np.zeros((data.T, NA))
    for c in range(NA):
        d = data.sub(np.arange(data.T))
        d.y = np.full(data.T, c)
        P[:, c] = np.exp(probit_task_ll(d, V, shares, M))
    return P


def timer():
    t0 = time.time()
    return lambda: time.time() - t0


# --------------------------------------------------------------------------- XGBoost with probit objective
class ProbitObjective:
    """Custom XGBoost objective: rows = cameras (4 per task, task order). Gradient of -log P(choice) w.r.t.
    each camera's utility by autodiff through GHK; Hessian = exact diagonal (double backward), recomputed
    every `hess_every` rounds and floored (a stale Hessian only rescales the Newton step)."""

    def __init__(self, data, shares, M=128, weights=None, hess_floor=1e-3, hess_every=10):
        self.pr = Probit(data, M)
        self.s = None if shares is None else torch.as_tensor(shares, dtype=F64)
        self.T = data.T
        self.w = None if weights is None else torch.as_tensor(weights, dtype=F64)
        self.floor, self.every, self.calls, self.h = hess_floor, hess_every, 0, None

    def __call__(self, preds, dtrain=None):
        Vc = torch.tensor(preds.reshape(self.T, 4), dtype=F64, requires_grad=True)
        V = torch.cat([Vc, Vc.new_zeros(self.T, 1)], 1)
        ll = self.pr.loglik(V, self.s)
        if self.w is not None:
            ll = ll * self.w
        need_h = self.h is None or self.calls % self.every == 0
        (g,) = torch.autograd.grad(-ll.sum(), Vc, create_graph=need_h)
        if need_h:
            h = torch.stack([torch.autograd.grad(g[:, j].sum(), Vc, retain_graph=True)[0][:, j]
                             for j in range(4)], 1)
            self.h = np.maximum(h.detach().numpy().reshape(-1), self.floor)
        self.calls += 1
        return g.detach().numpy().reshape(-1), self.h


def cam_matrix(data):
    """(T*4, 10) camera rows in task order."""
    return data.X[:, :4].reshape(-1, len(COLS))


def full_V(cam_utils, T):
    return np.c_[cam_utils.reshape(T, 4), np.zeros(T)]


def fit_xgb(train, val, shares, depth, M=128, eta=0.1, max_rounds=2000, patience=30, rounds=None,
            weights=None):
    """Boost with the probit objective. With `rounds` None: early stopping on validation log-likelihood
    (val tasks are not in `train`). Returns (booster, best_rounds)."""
    import xgboost as xgb
    params = dict(max_depth=depth, eta=eta, base_score=0.0, tree_method="hist", nthread=2,
                  min_child_weight=1.0, reg_lambda=1.0, disable_default_eval_metric=1)
    dtr = xgb.DMatrix(cam_matrix(train))
    obj = ProbitObjective(train, shares, M, weights=weights)
    if rounds is not None:
        return xgb.train(params, dtr, num_boost_round=rounds, obj=obj), rounds
    dval = xgb.DMatrix(cam_matrix(val))
    prv = Probit(val, M)
    sv = None if shares is None else torch.as_tensor(shares, dtype=F64)

    def metric(preds, dm):
        with torch.no_grad():
            V = torch.as_tensor(full_V(preds, val.T))
            return "nll", float(-prv.loglik(V, sv).mean())

    bst = xgb.train(params, dtr, num_boost_round=max_rounds, obj=obj, evals=[(dval, "val")],
                    custom_metric=metric, early_stopping_rounds=patience, verbose_eval=False)
    return bst, bst.best_iteration + 1


def xgb_V(bst, data, rounds):
    import xgboost as xgb
    u = bst.predict(xgb.DMatrix(cam_matrix(data)), output_margin=True, iteration_range=(0, rounds))
    return full_V(u, data.T)


# --------------------------------------------------------------------------- MNL and RFC
def fit_mnl(data, iters=200):
    X = torch.as_tensor(data.X)
    y = torch.as_tensor(data.y)
    b = torch.zeros(len(COLS), dtype=F64, requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-9)

    def cl():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ b, y, reduction="sum")
        loss.backward()
        return loss

    opt.step(cl)
    return b.detach().numpy().copy()


RFC_ATTRS = 6   # brand + 5 features (price excluded, as in T1)


class RFC:
    """Sawtooth-style RFC-S: one normal sd `sl` for attribute-level errors shared by cameras with the same
    level (brand, pixels, zoom, video, swivel, wifi), Gumbel product error of scale g (integrated
    analytically: softmax), "none" normal error of variance 6 sl^2. Shares = mean over R fixed draws.
    B: (T, 10) part-worths per task (pooled: the same row repeated; HB: the respondent's)."""

    def __init__(self, data, R=200, seed=SEED):
        rng = np.random.default_rng(seed)
        self.data = data
        nlev = [4, 2, 2, 2, 2, 2]
        Z = [rng.standard_normal((data.T, R, L)) for L in nlev]
        E = np.zeros((data.T, R, 4))
        for a in range(RFC_ATTRS):
            E += np.take_along_axis(Z[a], np.broadcast_to(data.lev[:, None, :, a], (data.T, R, 4)), 2)
        self.E = E                                          # sum over attributes of level errors, per camera
        self.En = rng.standard_normal((data.T, R))

    def probs(self, B, sl, g):
        V = np.einsum("tk,tjk->tj", B, self.data.X)          # (T, 5)
        e = np.concatenate([self.E * sl, (np.sqrt(RFC_ATTRS) * sl * self.En)[:, :, None]], 2)
        u = (V[:, None, :] + e) / g
        u -= u.max(2, keepdims=True)
        p = np.exp(u)
        return (p / p.sum(2, keepdims=True)).mean(1)


def tune_rfc(rfc, B, lo=(0.01, 0.05), hi=(3.0, 5.0)):
    """(sl, g) maximizing validation log-likelihood: log grid, then Nelder-Mead."""
    from scipy.optimize import minimize
    llo, lhi = np.log(lo), np.log(hi)
    y = rfc.data.y

    def nll(x):
        x = np.clip(x, llo, lhi)
        P = rfc.probs(B, *np.exp(x))
        return -np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)).mean()

    grid = [(a, b) for a in np.linspace(llo[0], lhi[0], 6) for b in np.linspace(llo[1], lhi[1], 6)]
    x0 = np.array(grid[int(np.argmin([nll(np.array(x)) for x in grid]))])
    r = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-7, maxiter=200))
    x = np.clip(r.x, llo, lhi)
    flags = [i for i in range(2) if x[i] - llo[i] < .05 or lhi[i] - x[i] < .05]
    return np.exp(x), flags
