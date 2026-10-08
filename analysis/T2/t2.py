"""T2 core: truths, simulation, coded-subset MNLs, probit queries, decisions.

Pure numpy/scipy so the Modal CPU containers can import it. Orthant probabilities for fitted
probits come from an evaluator passed in by the caller (the engine, in run.py); truths use
SciPy's Genz CDF at tight tolerance. Design: DESIGN.md.
"""
import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import multivariate_normal, norm, qmc

M = 12                     # menu size
N_RESP, T_TRAIN, N_HOLD = 1000, 10, 4
SEED = 20261008
LOGP = np.log([0.8, 1.0, 1.2])
B_PRICE = -1.5             # own log-price slope (probit scale)
GAMMA2, GAMMA3 = 1.0, 0.8  # LC pattern constants (logit scale)
TRUTHS = ("L0", "L1", "L2", "LC")
LIFT_LO, LIFT_HI, CAP = 0.8, 1.2, 5
SCREEN_Z = 1.2816          # two-sided p < 0.20


def patterns(s):
    return np.array(list(itertools.product([0, 1], repeat=s)), dtype=float)


# ---------------------------------------------------------------------------
# Truths and data
# ---------------------------------------------------------------------------
def truth_params(truth, k):
    rng = np.random.default_rng([SEED, k])
    take = rng.uniform(.10, .40, M)
    lam = rng.uniform(.5, .8, M)
    R = np.eye(M)
    if truth == "L1":
        R[:k, :k] = np.outer(lam[:k], lam[:k])
    elif truth == "L2":
        h, mu = (k + 1) // 2, .4
        load = np.zeros((M, 3))
        load[:h, 0], load[h:k, 1], load[:k, 2] = lam[:h], lam[h:k], mu
        R = load @ load.T
    np.fill_diagonal(R, 1.0)
    tp = dict(truth=truth, k=k, a=ndtri(take), b=np.full(M, B_PRICE), R=R, take=take)
    if truth == "LC":
        tp["pairs"] = [(2 * m, 2 * m + 1) for m in range(k // 2)]
        tp["triple"] = (0, 2, 4)
        tp["c"] = np.log(take[:k] / (1 - take[:k]))
        tp["bl"] = np.full(k, 1.6 * B_PRICE)
    return tp


def lc_logits(tp, X):
    """LC linked-set pattern log-weights (2^k,) per row of X (n, M) -> (n, 2^k)."""
    k = tp["k"]
    P = patterns(k)
    g = sum(GAMMA2 * P[:, i] * P[:, j] for i, j in tp["pairs"]) + GAMMA3 * np.prod(P[:, list(tp["triple"])], 1)
    return (tp["c"] + tp["bl"] * X[:, :k]) @ P.T + g


def simulate(tp, X, rng):
    """Picks (n, M) for log price ratios X (n, M)."""
    eta = tp["a"] + tp["b"] * X
    if tp["truth"] != "LC":
        e = rng.standard_normal(X.shape) @ np.linalg.cholesky(tp["R"]).T
        return (eta + e > 0).astype(np.int8)
    Y = (eta + rng.standard_normal(X.shape) > 0).astype(np.int8)
    P = patterns(tp["k"])
    for s in range(0, len(X), 1000):
        lw = lc_logits(tp, X[s:s + 1000])
        pr = np.exp(lw - logsumexp(lw, 1, keepdims=True)).cumsum(1)
        idx = (pr < rng.random((len(lw), 1))).sum(1).clip(max=len(P) - 1)
        Y[s:s + 1000, :tp["k"]] = P[idx]
    return Y


def holdout_menus(k):
    rng = np.random.default_rng([SEED, k, 1])
    return LOGP[rng.integers(0, 3, (N_HOLD, M))]


def make_data(truth, k, rep, n_resp=N_RESP):
    tp = truth_params(truth, k)
    rng = np.random.default_rng([SEED, k, TRUTHS.index(truth), rep])
    X = LOGP[rng.integers(0, 3, (n_resp * T_TRAIN, M))]
    Y = simulate(tp, X, rng)
    Xh = np.repeat(holdout_menus(k), n_resp, 0)
    Yh = simulate(tp, Xh, rng)
    return X, Y, Xh, Yh


# ---------------------------------------------------------------------------
# Coded subsets (Sawtooth MBC proxy)
# ---------------------------------------------------------------------------
class SubsetMNL:
    """MNL over the 2^s patterns of `items`: item constants and own log-price terms, cross
    log-price terms (y_j * x_i) for `cross` pairs, and one constant per pattern with >= 2 items.
    s = 1 is a binary logit."""

    def __init__(self, items, cross=None, theta=None):
        self.items = list(items)
        s = len(self.items)
        self.P = patterns(s)
        self.multi = np.where(self.P.sum(1) >= 2)[0]
        self.cross = [tuple(c) for c in (cross if cross is not None else
                                         [(i, j) for i in range(s) for j in range(s) if i != j])]
        self.theta = None if theta is None else np.asarray(theta, float)

    @property
    def npar(self):
        return 2 * len(self.items) + len(self.cross) + len(self.multi)

    def features(self, X):
        x = X[:, self.items]
        n, s = x.shape
        npat = len(self.P)
        F = np.zeros((n, npat, self.npar))
        F[:, :, :s] = self.P
        F[:, :, s:2 * s] = self.P[None] * x[:, None, :]
        o = 2 * s
        for c, (i, j) in enumerate(self.cross):
            F[:, :, o + c] = self.P[None, :, j] * x[:, None, i]
        o += len(self.cross)
        F[:, self.multi, o + np.arange(len(self.multi))] = 1.0
        return F

    def obs_index(self, Y):
        y = Y[:, self.items].astype(int)
        return y @ (1 << np.arange(len(self.items))[::-1])

    def fit(self, X, Y, ridge=1e-3):
        F, yi = self.features(X), self.obs_index(Y)
        n = len(X)
        Fy = F[np.arange(n), yi]

        def f(th):
            u = F @ th
            lse = logsumexp(u, 1)
            pi = np.exp(u - lse[:, None])
            g = -(Fy - np.einsum("np,npa->na", pi, F)).sum(0) + 2 * ridge * th
            return -(u[np.arange(n), yi] - lse).sum() + ridge * th @ th, g

        res = minimize(f, np.zeros(self.npar), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=2000, gtol=1e-6))
        self.theta = res.x
        u = F @ res.x
        pi = np.exp(u - logsumexp(u, 1)[:, None])
        mF = np.einsum("np,npa->na", pi, F)
        H = np.einsum("np,npa,npb->ab", pi, F, F) - mF.T @ mF + 2 * ridge * np.eye(self.npar)
        self.se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
        self.nll = float(res.fun)
        return self

    def screened(self, X, Y):
        """Sawtooth-style: fit with all within-subset cross terms, keep those with p < 0.20."""
        self.fit(X, Y)
        s = len(self.items)
        z = np.abs(self.theta[2 * s:2 * s + len(self.cross)] / self.se[2 * s:2 * s + len(self.cross)])
        keep = [c for c, zz in zip(self.cross, z) if zz > SCREEN_Z]
        return SubsetMNL(self.items, keep).fit(X, Y)

    def pattern_probs(self, X):
        u = self.features(X) @ self.theta
        return np.exp(u - logsumexp(u, 1)[:, None])

    def to_dict(self):
        return dict(items=self.items, cross=self.cross, theta=self.theta.tolist())


def analyst_subsets(Y):
    """CS-F: merge items greedily by |log lift| (lift > 1.2 or < 0.8), at most 5 per subset."""
    p = Y.mean(0)
    pij = (Y.T.astype(float) @ Y) / len(Y)
    cand = []
    for i, j in itertools.combinations(range(M), 2):
        lift = pij[i, j] / (p[i] * p[j])
        if lift > LIFT_HI or lift < LIFT_LO:
            cand.append((abs(np.log(lift)), i, j))
    return _merge(sorted(cand, reverse=True))


def oracle_subsets(tp):
    """CS-O: the true dependence, merged greedily by strength with the same cap."""
    if tp["truth"] == "LC":
        w = {}
        for i, j in tp["pairs"]:
            w[(i, j)] = GAMMA2
        for i, j in itertools.combinations(tp["triple"], 2):
            w[(i, j)] = w.get((i, j), 0) + GAMMA3
        cand = [(v, i, j) for (i, j), v in w.items()]
    else:
        R = tp["R"]
        cand = [(abs(R[i, j]), i, j) for i, j in itertools.combinations(range(M), 2) if abs(R[i, j]) > .05]
    return _merge(sorted(cand, reverse=True))


def _merge(cand):
    group = {i: {i} for i in range(M)}
    for _, i, j in cand:
        if group[i] is group[j] or len(group[i]) + len(group[j]) > CAP:
            continue
        g = group[i] | group[j]
        for m in g:
            group[m] = g
    out = []
    for g in group.values():
        if sorted(g) not in out:
            out.append(sorted(g))
    return sorted(out)


def fit_cs(X, Y, subsets):
    return [(SubsetMNL(s).screened(X, Y) if len(s) > 1 else SubsetMNL(s).fit(X, Y)).to_dict()
            for s in subsets]


# ---------------------------------------------------------------------------
# Models as query objects: marg(x), p_one(sets, x), p_none(sets, x); x is one menu (M,)
# ---------------------------------------------------------------------------
def orth_exact(upper, cov):
    """P(Z <= upper) row by row, SciPy Genz at tight tolerance (truths only)."""
    out = np.empty(len(upper))
    for r in range(len(upper)):
        if upper.shape[1] == 1:
            out[r] = norm.cdf(upper[r, 0])
        else:
            out[r] = multivariate_normal.cdf(upper[r], cov=cov[r], abseps=1e-7, releps=1e-7,
                                             maxpts=200000 * upper.shape[1], rng=np.random.default_rng(0))
    return out


def orth_qmc(upper, cov, m=16384, seed=0):
    """P(Z <= upper), Z ~ N(0, cov), row by row by GHK with m scrambled Sobol points shared
    across rows (vectorised; accurate to ~1e-4 at m = 16384). Used for the fitted probits'
    decisions so that Q1 is not mixed with evaluator error; the engine is scored separately."""
    n, d = upper.shape
    L = np.linalg.cholesky(cov)
    w = qmc.Sobol(max(d - 1, 1), scramble=True, seed=seed).random(m)
    out = np.empty(n)
    chunk = max(1, int(2.5e7 // (m * d)))
    for s in range(0, n, chunk):
        Lc, uc = L[s:s + chunk], upper[s:s + chunk]
        logp = np.zeros((len(uc), m))
        z = np.zeros((len(uc), m, d))
        for j in range(d):
            c = (uc[:, j, None] - np.einsum("nk,nmk->nm", Lc[:, j, :j], z[:, :, :j])) / Lc[:, j, j, None]
            p = norm.cdf(c)
            logp += np.log(np.clip(p, 1e-300, None))
            if j < d - 1:
                z[:, :, j] = norm.ppf(np.clip(w[None, :, j] * p, 1e-300, 1 - 1e-16))
        out[s:s + chunk] = np.exp(logp).mean(1)
    return out


class Probit:
    """Binary probit margins eta(x) = c + B x and correlation R; orthants by `orth`."""

    def __init__(self, c, B, R, orth):
        self.c, self.B, self.R, self.orth = np.asarray(c), np.asarray(B), np.asarray(R), orth

    def eta(self, x):
        return self.c + self.B @ x

    def _orth(self, sets, x, sign):
        e = sign * self.eta(x)
        out = np.empty(len(sets))
        by_d = {}
        for r, S in enumerate(sets):
            by_d.setdefault(len(S), []).append(r)
        for d, rows in by_d.items():
            S = np.array([sets[r] for r in rows])
            if d == 1:
                out[rows] = norm.cdf(e[S[:, 0]])
                continue
            up = e[S]
            cov = self.R[S[:, :, None], S[:, None, :]]
            out[rows] = self.orth(up, cov)
        return out

    def marg(self, x):
        return norm.cdf(self.eta(x))

    def p_one(self, sets, x):
        return self._orth(sets, x, 1.0)

    def p_none(self, sets, x):
        return self._orth(sets, x, -1.0)

    def pattern_logprob(self, X, Y):
        """Full-pattern log probabilities by GHK (m = 2048): the engine is not accurate for
        12-dim patterns with probabilities near 1e-4 (known small-probability envelope)."""
        XY = np.hstack([X, Y])
        u, inv = np.unique(XY, axis=0, return_inverse=True)
        Xu, Yu = u[:, :X.shape[1]], u[:, X.shape[1]:]
        s = 2.0 * Yu - 1
        up = s * (self.c + Xu @ self.B.T)
        cov = s[:, :, None] * s[:, None, :] * self.R
        return np.log(np.clip(orth_qmc(up, cov, m=2048), 1e-300, None))[inv.ravel()]


class Blocks:
    """Independent blocks, each with a pattern distribution over its items (CS models, LC
    truth). blocks: list of (items, fn) with fn(x (M,)) -> pattern probs over patterns(len)."""

    def __init__(self, blocks):
        self.blocks = [(list(it), fn, patterns(len(it))) for it, fn in blocks]

    def _block_query(self, sets, x, val):
        out = np.ones(len(sets))
        for items, fn, P in self.blocks:
            pr = fn(x)
            pos = {m: q for q, m in enumerate(items)}
            for r, S in enumerate(sets):
                cols = [pos[m] for m in S if m in pos]
                if cols:
                    out[r] *= pr[np.all(P[:, cols] == val, 1)].sum()
        return out

    def marg(self, x):
        return self.p_one([(m,) for m in range(M)], x)

    def p_one(self, sets, x):
        return self._block_query(sets, x, 1)

    def p_none(self, sets, x):
        return self._block_query(sets, x, 0)

    def pattern_logprob(self, X, Y):
        lp = np.zeros(len(X))
        xs, inv = np.unique(X, axis=0, return_inverse=True)
        for items, fn, _ in self.blocks:
            idx = Y[:, items].astype(int) @ (1 << np.arange(len(items))[::-1])
            pr = np.array([fn(x) for x in xs])
            lp += np.log(np.clip(pr[inv.ravel(), idx], 1e-300, None))
        return lp


def cs_model(dicts):
    blocks = []
    for d in dicts:
        m = SubsetMNL(d["items"], d["cross"], d["theta"])
        blocks.append((m.items, lambda x, m=m: m.pattern_probs(x[None])[0]))
    return Blocks(blocks)


def truth_model(tp):
    if tp["truth"] != "LC":
        return Probit(tp["a"], np.diag(tp["b"]), tp["R"], orth_exact)
    k = tp["k"]

    def lc(x):
        lw = lc_logits(tp, x[None])[0]
        return np.exp(lw - logsumexp(lw))

    blocks = [(list(range(k)), lc)]
    for m in range(k, M):
        blocks.append(([m], lambda x, m=m: np.array([1 - norm.cdf(tp["a"][m] + tp["b"][m] * x[m]),
                                                     norm.cdf(tp["a"][m] + tp["b"][m] * x[m])])))
    return Blocks(blocks)


# ---------------------------------------------------------------------------
# Decisions and scoring
# ---------------------------------------------------------------------------
PAIRS = list(itertools.combinations(range(M), 2))
TRIPLES = list(itertools.combinations(range(M), 3))
SETS4 = list(itertools.combinations(range(M), 4))


def decisions(model, k):
    """Quantities behind every decision, at base prices (x = 0) unless noted."""
    x0 = np.zeros(M)
    m0 = model.marg(x0)
    j2 = model.p_one(PAIRS, x0)
    j3 = model.p_one(TRIPLES, x0)
    reach4 = 1 - model.p_none(SETS4, x0)
    x1 = x0.copy()
    x1[0] = np.log(1.2)
    linked = [tuple(range(k))]
    any0, any1 = 1 - model.p_none(linked, x0)[0], 1 - model.p_none(linked, x1)[0]
    lift = j2 / np.array([m0[i] * m0[j] for i, j in PAIRS])
    return dict(marg=m0.tolist(), joint2=j2.tolist(), joint3=j3.tolist(), lift2=lift.tolist(),
                reach4=reach4.tolist(), dmarg=(model.marg(x1) - m0).tolist(), d_any_linked=any1 - any0)


def holdout_joints(model, k):
    return [model.p_one(PAIRS, x).tolist() for x in holdout_menus(k)]


def _pick(val, truth_val):
    val, tv = np.asarray(val), np.asarray(truth_val)
    c = int(np.argmax(val))
    return dict(flip=bool(c != int(np.argmax(tv))), regret_pct=float(100 * (tv.max() - tv[c]) / tv.max()))


def score(dec, tdec, hj, thj):
    out = {}
    for name, key in (("bundle2", "joint2"), ("lift2", "lift2"), ("bundle3", "joint3"), ("turf4", "reach4")):
        for f, v in _pick(dec[key], tdec[key]).items():
            out[f"{name}_{f}"] = v
    out["dmarg_mae_pts"] = float(100 * np.abs(np.array(dec["dmarg"][1:]) - np.array(tdec["dmarg"][1:])).mean())
    out["d_any_linked_err_pts"] = float(100 * (dec["d_any_linked"] - tdec["d_any_linked"]))
    out["joint2_mae_pts"] = float(100 * np.abs(np.array(dec["joint2"]) - np.array(tdec["joint2"])).mean())
    out["joint3_mae_pts"] = float(100 * np.abs(np.array(dec["joint3"]) - np.array(tdec["joint3"])).mean())
    out["holdout_joint2_mae_pts"] = float(100 * np.abs(np.array(hj) - np.array(thj)).mean())
    return out
