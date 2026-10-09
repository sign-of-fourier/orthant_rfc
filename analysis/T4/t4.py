"""T4 core (numpy only): shelf DGP, simulators, RFC, decisions, scoring. DESIGN.md.

SKUs: brand (5) x flavor (4) x pack (3) = 60, index = 12 brand + 3 flavor + pack. Every task shows
all J SKUs of the shelf plus "none"; each SKU's price is drawn per task from PRICES. Coding per
alternative: effects-coded brand (4), flavor (3), pack (2), price - PRICE_REF, none constant = 11.
Errors (probit truth S1): U = x'beta + sum_a sigma_a zeta_{a, level_a} + sigma_nu nu, a fresh zeta per
level per task, "none" with an independent normal error of the total variance T_VAR. S0: iid Gumbel.
"""
import numpy as np
from scipy.special import logsumexp

SEED = 20261009
T_VAR = np.pi ** 2 / 6
LEVELS = (5, 4, 3)
NLEV = sum(LEVELS)                      # 12 error components per task
PRICES = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
PRICE_REF = 4.5
COST = 3.00
HURDLE = 0.5                            # placeholder line-extension hurdle (as T1)
EXT_CUT = 0.30                          # the extension is a clone of an owned SKU, $0.30 cheaper
MATERIAL = 0.5                          # regret % above which a flip is material
#                  brand            flavor         pack     price   none
B_TRUE = np.array([.5, .2, -.1, .1,  .4, 0., -.2,  .3, 0.,  -.975,  0.])
W_TRUE = np.array([.8, .8, .8, .8,   .6, .6, .6,   .5, .5,   .41,   .5])
P = len(B_TRUE)
NAMES = [f"brand{i}" for i in range(4)] + [f"flavor{i}" for i in range(3)] + ["pack0", "pack1", "price", "none"]
# "none" constant per J: training none share ~ 15% under S1 (set by calibrate_none, 2026-10-08)
B_NONE = {20: 1.665, 40: 2.128, 60: 2.292}
TRUTHS = {  # variance shares of T_VAR: brand, flavor, pack, product
    "S0": np.array([0., 0., 0., 1.]),
    "S1": np.array([.10, .35, .35, .20]),
}
LOGIT_TRUTHS = {"S0"}

ALL = np.array([[b, f, p] for b in range(5) for f in range(4) for p in range(3)])   # (60, 3)
OWN_FP = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (3, 2)]   # the firm's 6 SKUs: brand 0 at these flavor, pack


def shelf(J):
    """SKU indices of the J-shelf; nested (20 in 40 in 60); the firm's 6 SKUs first. Every level
    of every attribute is present at J = 20, with at least two SKUs per competitor brand."""
    own = [12 * 0 + 3 * f + p for f, p in OWN_FP]
    rng = np.random.default_rng([SEED, 1])
    rest = [i for i in range(60) if i not in own]
    while True:
        order = list(rng.permutation(rest))
        s20 = own + order[:14]
        lv = ALL[s20]
        if all(len(set(lv[:, a])) == LEVELS[a] for a in range(3)) and \
                all((lv[:, 0] == b).sum() >= 2 for b in range(1, 5)):
            break
    s40 = s20 + order[14:34]
    return np.array({20: s20, 40: s40, 60: s40 + order[34:]}[J])


def code(levels, price):
    """levels (..., 3) int, price (...) -> (..., P) with the none column 0."""
    out = np.zeros(levels.shape[:-1] + (P,))
    col = 0
    for a, L in enumerate(LEVELS):
        lv = levels[..., a]
        for l in range(L - 1):
            out[..., col + l] = (lv == l).astype(float) - (lv == L - 1)
        col += L - 1
    out[..., P - 2] = price - PRICE_REF
    return out


def task_X(levels, prices):
    """levels (K, 3), prices (..., K) -> (..., K+1, P), none last."""
    X = code(np.broadcast_to(levels, prices.shape + (3,)), prices)
    nr = np.zeros(prices.shape[:-1] + (1, P))
    nr[..., P - 1] = 1
    return np.concatenate([X, nr], -2)


def member(levels):
    """(K, 3) levels -> (K+1, NLEV) 0/1: which error component each alternative loads on (none: none)."""
    K = len(levels)
    A = np.zeros((K + 1, NLEV))
    off = np.cumsum((0,) + LEVELS[:-1])
    for a in range(3):
        A[np.arange(K), off[a] + levels[:, a]] = 1
    return A


def b_true(J):
    b = B_TRUE.copy()
    b[-1] = B_NONE[J]
    return b


def sig_of(truth):
    return np.sqrt(T_VAR * TRUTHS[truth])


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(rng, levels, prices, beta, sig, logit):
    """First choices. levels (K, 3), prices (N, T, K), beta (N, P) -> (N, T) chosen in 0..K (K = none)."""
    N, T, K = prices.shape
    V = np.einsum("ntkp,np->ntk", task_X(levels, prices), beta)
    if logit:
        return np.argmax(V + rng.gumbel(size=V.shape), 2)
    A = member(levels)                                                    # (K+1, NLEV)
    s = np.repeat(sig[:3], LEVELS)
    z = rng.standard_normal((N, T, NLEV)) * s
    e = z @ A.T
    e[..., :K] += sig[3] * rng.standard_normal((N, T, K))
    e[..., K] = np.sqrt(T_VAR) * rng.standard_normal((N, T))
    return np.argmax(V + e, 2)


def make_data(truth, J, N, T, rep, n_hold=6):
    """One dataset. Returns levels, prices (N, T, J), y (N, T), beta, holdout prices (6, J),
    holdout choices (N, 6)."""
    sig, logit = sig_of(truth), truth in LOGIT_TRUTHS
    r_data, r_hold = [np.random.default_rng(s) for s in np.random.SeedSequence([SEED, J, rep]).spawn(2)]
    levels = ALL[shelf(J)]
    beta = b_true(J) + W_TRUE * r_data.standard_normal((N, P))
    prices = PRICES[r_data.integers(0, 5, (N, T, J))]
    y = simulate(r_data, levels, prices, beta, sig, logit)
    Hp = PRICES[np.random.default_rng([SEED, J, 7]).integers(0, 5, (n_hold, J))]   # fixed across reps
    yh = simulate(r_hold, levels, np.broadcast_to(Hp, (N, n_hold, J)), beta, sig, logit)
    return dict(levels=levels, prices=prices, y=y, beta=beta, Hp=Hp, yh=yh)


def calibrate_none(target=0.15, N=4000, T=12):
    """B_NONE[J] so that the S1 training none share is ~ target (bisection on the constant)."""
    out = {}
    for J in (20, 40, 60):
        lo, hi = -6.0, 6.0
        for _ in range(25):
            B_NONE[J] = (lo + hi) / 2
            d = make_data("S1", J, N, T, 999)
            sh = np.mean(d["y"] == J)
            lo, hi = (B_NONE[J], hi) if sh < target else (lo, B_NONE[J])
        out[J] = round(B_NONE[J], 3)
    return out


# ---------------------------------------------------------------------------
# Market scenarios and share simulators (common random numbers across scenarios and models)
# ---------------------------------------------------------------------------
def base_prices(J):
    return PRICES[np.random.default_rng([SEED, J, 3]).integers(0, 5, J)]


def scenarios(J):
    """Scenario list: dict(name, prices (J+1,), present (J+1,) bool). Slot J is the extension
    clone (absent unless launched); "none" is always present and not a slot."""
    bp = base_prices(J)
    out = []

    def add(name, pr=None, drop=None, ext=None):
        p = np.r_[bp, 0.0] if pr is None else pr
        m = np.r_[np.ones(J, bool), False]
        if drop is not None:
            m[drop] = False
        if ext is not None:
            p = p.copy()
            p[J] = max(bp[ext] - EXT_CUT, 0.0)
            m[J] = True
        out.append(dict(name=name, prices=p, present=m, ext=ext))

    add("base")
    for k in range(6):
        for lv in PRICES:
            pr = np.r_[bp, 0.0]
            pr[k] = lv
            add(f"price{k}_{lv:.1f}", pr=pr)
    for k in range(6):
        add(f"delist{k}", drop=k)
    for k in range(6):
        add(f"ext{k}", ext=k)
    return out


def scen_levels(levels, s):
    """Levels (J+1, 3): the clone slot copies the cloned SKU (slot 0 when there is no clone)."""
    return np.vstack([levels, levels[s["ext"] if s["ext"] is not None else 0]])


def shares_probit(b, w, sig, levels, scens, n_sim, seed, chunk=50_000, logit=False):
    """Population shares (S, J+2) [J SKUs, clone, none] of the probit (or S0 logit: analytic
    given beta) with beta ~ N(b, diag w^2), by first choice over n_sim draws, the same draws for
    every scenario."""
    J = len(levels)
    S = len(scens)
    out = np.zeros((S, J + 2))
    rng = np.random.default_rng(seed)
    s3 = np.repeat(np.asarray(sig[:3]), LEVELS)
    for c0 in range(0, n_sim, chunk):
        n = min(chunk, n_sim - c0)
        beta = b + w * rng.standard_normal((n, P))
        zeta = rng.standard_normal((n, NLEV)) * s3
        nu = rng.standard_normal((n, J + 2)) if not logit else None
        for i, s in enumerate(scens):
            lv = scen_levels(levels, s)
            X = task_X(lv, s["prices"])                                   # (J+2, P)
            V = beta @ X.T                                                # (n, J+2)
            if logit:
                V[:, :J + 1][:, ~s["present"]] = -np.inf
                p = np.exp(V - logsumexp(V, 1, keepdims=True))
                out[i] += p.sum(0)
                continue
            U = V + zeta @ member(lv).T
            U[:, :J + 1] += sig[3] * nu[:, :J + 1]
            U[:, J + 1] += np.sqrt(T_VAR) * nu[:, J + 1]
            U[:, :J + 1][:, ~s["present"]] = -np.inf
            out[i] += np.bincount(U.argmax(1), minlength=J + 2)
    return out / n_sim


def shares_logit_resp(B, levels, scens):
    """HB-MNL point estimates, plain logit: shares (S, J+2)."""
    J = len(levels)
    out = np.zeros((len(scens), J + 2))
    for i, s in enumerate(scens):
        V = B @ task_X(scen_levels(levels, s), s["prices"]).T
        V[:, :J + 1][:, ~s["present"]] = -np.inf
        out[i] = np.exp(V - logsumexp(V, 1, keepdims=True)).mean(0)
    return out


class RFC:
    """Sawtooth-style RFC on HB point estimates (T1's): normal attribute-level errors sigma_l
    shared by alternatives with the same brand / flavor / pack level, Gumbel product error g
    (integrated analytically), "none" with a normal error of variance sum_a sigma_a^2."""

    def __init__(self, B, R_l, rng):
        self.B = B
        self.Z = [rng.standard_normal((len(B), R_l, L)) for L in LEVELS]
        self.Zn = rng.standard_normal((len(B), R_l))

    def shares(self, levels, prices, sl, g, present=None, chunk=50):
        """levels (K, 3), prices (K,) -> (K+1,) shares. sl: one sigma (RFC-S) or three (RFC-G)."""
        sl = np.broadcast_to(np.asarray(sl, float), (3,))
        X = task_X(levels, prices)
        V = self.B @ X.T
        K = len(levels)
        if present is not None:
            V = V.copy()
            V[:, :K][:, ~present] = -np.inf
        out = np.zeros(K + 1)
        for s in range(0, len(self.B), chunk):
            e = sum(self.Z[a][s:s + chunk][:, :, levels[:, a]] * sl[a] for a in range(3))
            e = np.concatenate([e, np.sqrt(np.sum(sl ** 2)) * self.Zn[s:s + chunk][:, :, None]], 2)
            u = (V[s:s + chunk, None, :] + e) / g
            u -= u.max(2, keepdims=True)
            p = np.exp(u)
            out += (p / p.sum(2, keepdims=True)).mean(1).sum(0)
        return out / len(self.B)

    def scen_shares(self, levels, scens, sl, g):
        return np.stack([self.shares(scen_levels(levels, s), s["prices"], sl, g, s["present"]) for s in scens])


def tune_rfc(R, levels, Hp, obs, k):
    """Minimise holdout share MAE over log (sigma_l..., g): k = 1 (RFC-S) or 3 (RFC-G).
    Log grid (6 points per axis for k = 1, 3 for k = 3), then Nelder-Mead."""
    from scipy.optimize import minimize

    lo, hi = np.log([0.01] * k + [0.05]), np.log([5.0] * k + [5.0])

    def mae(x):
        x = np.clip(x, lo, hi)
        p = np.exp(x)
        return np.mean([np.abs(R.shares(levels, Hp[i], p[:k], p[k]) - obs[i]).mean() for i in range(len(Hp))])

    n = 6 if k == 1 else 3
    axes = [np.linspace(lo[i], hi[i], n) for i in range(k + 1)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(k + 1, -1).T
    x0 = grid[int(np.argmin([mae(x) for x in grid]))]
    res = minimize(mae, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-7, maxiter=300 * (k + 1)))
    x = np.clip(res.x, lo, hi)
    flags = [f"param {i} at bound" for i in range(k + 1) if x[i] - lo[i] < .05 or hi[i] - x[i] < .05]
    return np.exp(x), float(res.fun), flags


# ---------------------------------------------------------------------------
# Decisions and scoring
# ---------------------------------------------------------------------------
def decision_values(sh, scens, J):
    """Firm values from shares (S, J+2): per decision the value of each option.
    price{k}: firm margin at each of 5 levels; delist: firm margin after dropping each owned
    SKU; ext{k}: incrementality of the clone of SKU k (share of its volume new to the firm)."""
    idx = {s["name"]: i for i, s in enumerate(scens)}

    def margin(i):
        s = scens[i]
        own = list(range(6)) + ([J] if s["ext"] is not None else [])
        return sum((s["prices"][j] - COST) * sh[i, j] for j in own if s["present"][j])

    def units(i, with_clone):
        s = scens[i]
        return sum(sh[i, j] for j in range(6) if s["present"][j]) + (sh[i, J] if with_clone else 0)

    out = {}
    for k in range(6):
        out[f"price{k}"] = np.array([margin(idx[f"price{k}_{lv:.1f}"]) for lv in PRICES])
    out["delist"] = np.array([margin(idx[f"delist{k}"]) for k in range(6)])
    base_u = units(idx["base"], False)
    for k in range(6):
        i = idx[f"ext{k}"]
        out[f"ext{k}"] = np.array([(units(i, True) - base_u) / sh[i, J]])
    return out


def score(dv, tv):
    """Per decision: pick, flip vs the truth's pick, regret % (truth's value of the pick).
    ext: launch at HURDLE (flip if the side differs) and the incrementality error."""
    out = {}
    for k, v in dv.items():
        t = tv[k]
        if k.startswith("ext"):
            out[k] = dict(value=float(v[0]), err=float(v[0] - t[0]),
                          flip=bool((v[0] >= HURDLE) != (t[0] >= HURDLE)), regret_pct=0.0)
        else:
            i = int(np.argmax(v))
            reg = float(100 * (t.max() - t[i]) / t.max())
            out[k] = dict(pick=i, flip=bool(i != int(np.argmax(t))), regret_pct=reg)
    return out


def pair_corr(sig):
    """Error correlation of two SKUs with the same flavor and pack, different brand."""
    sig = np.asarray(sig)
    return float((sig[1] ** 2 + sig[2] ** 2) / np.sum(sig ** 2))


def pair_corr_rfc(sl, g):
    sl = np.broadcast_to(np.asarray(sl, float), (3,))
    return float((sl[1] ** 2 + sl[2] ** 2) / (np.sum(sl ** 2) + np.pi ** 2 * g ** 2 / 6))
