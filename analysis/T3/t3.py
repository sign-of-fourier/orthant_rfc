"""T3 core: dunnhumby basket data for one cluster, coded-subset MNLs with covariates, pattern
distributions of every model, decisions, holdout adjudication. Design: DESIGN.md.

Pure numpy/scipy (plus pandas/pyreadr for the data build) so the Modal CPU containers can import
it. Every decision quantity is a function of the 2^M pattern distribution averaged over the
holdout trips: exact for coded subsets, scrambled-Sobol simulation for probits.
"""
import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, logsumexp, ndtri
from scipy.stats import qmc

SEED = 20261008
MIN_TRIPS, HOLD_FRAC, MIN_HOLD = 20, 0.25, 2
SHRINK = 5.0               # pseudo-trips for the household propensity
MIN_PRICE_SD = 0.01        # log-price columns with less weekly variation are dropped
LIFT_LO, LIFT_HI, CAP = 0.8, 1.2, 5
RHO_GATE, GATE_SIZE = 0.2, 6
NON_MERCH = {"COUPON/MISC ITEMS"}

# item -> [(product_category, [product_type, ...])]
CLUSTERS = {
    "taco": {
        "shells/tortillas": [("HISPANIC", ["MEXICAN TACO TOSTADO SHELLS", "MEXICAN SOFT TORTILLAS AND WRA"])],
        "taco seasoning": [("HISPANIC", ["MEXICAN SEASONING MIXES"])],
        "salsa": [("HISPANIC", ["MEXICAN SAUCESSALSAPICANTEE"]), ("SALADS/DIPS", ["SAL:SALSA/DPS-PRPCK"])],
        "beans": [("HISPANIC", ["MEXICAN BEANS REFRIED"]),
                  ("BEANS - CANNED GLASS & MW", ["VARIETY BEANS - KIDNEY PINTO"])],
        "sour cream": [("MILK BY-PRODUCTS", ["SOUR CREAMS"])],
        "shredded cheese": [("CHEESE", ["SHREDDED CHEESE"])],
        "tortilla chips": [("BAG SNACKS", ["TORTILLA/NACHO CHIPS"])],
        "lettuce": [("VEGETABLES SALAD", ["HEAD LETTUCE", "VARIETY LETTUCE"])],
    },
    "cookout": {
        "hot dogs": [("HOT DOGS", None)],
        "dinner sausage": [("DINNER SAUSAGE", ["SMOKED/COOKED"])],
        "hot dog buns": [("BAKED BREAD/BUNS/ROLLS", ["HOT DOG BUNS"])],
        "hamburger buns": [("BAKED BREAD/BUNS/ROLLS", ["HAMBURGER BUNS"])],
        "mustard": [("CONDIMENTS/SAUCES", ["SALAD MUSTARD", "HOT MUSTARD/SPECIALTY MUSTAR"])],
        "ketchup": [("CONDIMENTS/SAUCES", ["CATSUP"])],
        "pickles/relish": [("PICKLE/RELISH/PKLD VEG", ["PICKLES", "RELISHES"])],
        "baked beans": [("BEANS - CANNED GLASS & MW", ["PREPARED BEANS - BAKED W/PORK"])],
        "charcoal": [("CHARCOAL AND LIGHTER FLUID", ["CHARCOAL"])],
    },
}


def patterns(s):
    return np.array(list(itertools.product([0, 1], repeat=s)), dtype=float)


def codes(Y):
    """Pattern index of each row (first item = most significant bit, as patterns())."""
    s = Y.shape[1]
    return (Y.astype(np.int64) @ (1 << np.arange(s)[::-1])).astype(np.int64)


# ---------------------------------------------------------------------------
# Data (local only: pandas + pyreadr)
# ---------------------------------------------------------------------------
def load(cluster, data_dir, n_hh=None):
    """Trips of households with >= MIN_TRIPS trips; the last HOLD_FRAC of each household's trips
    (at least MIN_HOLD) are held out. X: own weekly log prices, trip size, household propensity
    per item; standardised on training trips."""
    import pandas as pd
    import pyreadr

    items = CLUSTERS[cluster]
    names = list(items)
    M = len(names)
    t = pyreadr.read_r(str(data_dir / "transactions.rds"))[None]
    p = pyreadr.read_r(str(data_dir / "products.rda"))["products"]
    pc, pt = p.product_category.fillna(""), p.product_type.fillna("")
    item = pd.Series(-1, index=p.index)
    for m, spec in enumerate(items.values()):
        mask = pd.Series(False, index=p.index)
        for c, types in spec:
            mask |= (pc == c) & (True if types is None else pt.isin(types))
        assert mask.any(), f"no products for {names[m]}"
        assert (item[mask] == -1).all(), "overlapping item definitions"
        item[mask] = m
    p = p.assign(item=item.values)
    t = t.merge(p[["product_id", "product_category", "item"]], on="product_id", how="left")
    t["item"] = t["item"].fillna(-1).astype(int)

    # weekly item prices: median unit price paid across all households (as analysis/rho_test)
    ok = (t.item >= 0) & (t.quantity > 0) & (t.sales_value > 0)
    u = t[ok].assign(unit=lambda d: d.sales_value / d.quantity)
    weeks = np.arange(t.week.min(), t.week.max() + 1)
    LP, carried = np.zeros((len(weeks), M)), {}
    for m in range(M):
        g = u[u.item == m].groupby("week").unit.agg(["median", "size"]).reindex(weeks)
        good = g["size"].fillna(0) >= 20
        LP[:, m] = np.log(g["median"].where(good).ffill().bfill().values)
        carried[names[m]] = int((~good).sum())
    LP -= LP.mean(0)

    ntrip = t.groupby("household_id").basket_id.nunique()
    elig = np.sort(ntrip[ntrip >= MIN_TRIPS].index.astype(int))
    if n_hh:
        elig = np.sort(np.random.default_rng(SEED).choice(elig, n_hh, replace=False))
    ts = t[t.household_id.astype(int).isin(elig)]
    del t
    nb = ts.basket_id.nunique()
    freq = ts.groupby("product_category").basket_id.nunique().sort_values(ascending=False) / nb
    item_cats = set(p.product_category[p.item >= 0])
    H = [c for c in freq.index if c not in item_cats | NON_MERCH][:20]
    trips = ts.groupby("basket_id").agg(household_id=("household_id", "first"), week=("week", "first"),
                                        ts=("transaction_timestamp", "min")).reset_index()
    y = pd.crosstab(ts.basket_id[ts.item >= 0], ts.item[ts.item >= 0]).reindex(
        index=trips.basket_id, columns=range(M), fill_value=0).values > 0
    nH = ts[ts.product_category.isin(H)].groupby("basket_id").product_category.nunique().reindex(
        trips.basket_id, fill_value=0).values
    trips = trips.assign(nH=nH).sort_values(["household_id", "ts", "basket_id"])
    y = y[trips.index.values]
    trips = trips.reset_index(drop=True)
    hh = trips.household_id.astype(int).map({h: i for i, h in enumerate(elig)}).values
    T = np.bincount(hh)
    n_hold = np.maximum(MIN_HOLD, np.round(HOLD_FRAC * T).astype(int))
    train = (trips.groupby(hh).cumcount(ascending=False).values >= n_hold[hh])

    # household propensity: shrunk training rate, leave-one-out on training trips
    pbar = y[train].mean(0)
    Ttr = np.bincount(hh[train], minlength=len(elig))
    ctr = np.zeros((len(elig), M))
    np.add.at(ctr, hh[train], y[train])
    num = ctr[hh] - np.where(train[:, None], y, 0) + SHRINK * pbar
    den = (Ttr[hh] - train.astype(int))[:, None] + SHRINK
    prop = logit(num / den) - logit(pbar)

    week = (trips.week - weeks[0]).values
    lp_keep = [m for m in range(M) if LP[:, m].std() >= MIN_PRICE_SD]
    X = np.column_stack([LP[week][:, lp_keep], np.log1p(trips.nH.values), prop])
    cols = [f"log price: {names[m]}" for m in lp_keep] + ["trip size"] + [f"propensity: {n}" for n in names]
    mu, sd = X[train].mean(0), X[train].std(0)
    X = (X - mu) / sd
    return dict(X=X, Y=y.astype(np.int8), train=train, hh=hh, names=names, cols=cols, H=H, carried=carried,
                lp_dropped=[names[m] for m in range(M) if m not in lp_keep], n_hh=len(elig),
                rate=y.mean(0), weeks=week)


# ---------------------------------------------------------------------------
# Coded subsets: MNL over the 2^s patterns of `items`, every item's index linear in all of X,
# one constant per pattern with >= 2 items. s = 1 is a binary logit.
# ---------------------------------------------------------------------------
class CovMNL:
    def __init__(self, items, theta=None):
        self.items = list(items)
        self.s = len(self.items)
        self.P = patterns(self.s)
        self.multi = np.where(self.P.sum(1) >= 2)[0]
        self.theta = None if theta is None else np.asarray(theta, float)

    def _u(self, Xa, th):
        s, q = self.s, Xa.shape[1]
        eta = Xa @ th[:s * q].reshape(s, q).T
        u = eta @ self.P.T
        u[:, self.multi] += th[s * q:]
        return u

    def fit(self, X, Y, ridge=1e-4):
        Xa = np.hstack([np.ones((len(X), 1)), X])
        s, q = self.s, Xa.shape[1]
        Ys = Y[:, self.items].astype(float)
        yi = codes(Ys)
        n_multi = np.bincount(yi, minlength=len(self.P))[self.multi]
        idx = np.arange(len(X))

        def f(th):
            u = self._u(Xa, th)
            lse = logsumexp(u, 1)
            pi = np.exp(u - lse[:, None])
            gB = -((Ys - pi @ self.P).T @ Xa).ravel()
            gg = -(n_multi - pi[:, self.multi].sum(0))
            return -(u[idx, yi] - lse).sum() + ridge * th @ th, np.r_[gB, gg] + 2 * ridge * th

        res = minimize(f, np.zeros(s * q + len(self.multi)), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=5000, gtol=1e-5))
        self.theta, self.nll, self.converged = res.x, float(res.fun), bool(res.success)
        return self

    def pattern_probs(self, X):
        u = self._u(np.hstack([np.ones((len(X), 1)), X]), self.theta)
        return np.exp(u - logsumexp(u, 1)[:, None])

    def to_dict(self):
        return dict(items=self.items, theta=self.theta.tolist(), converged=self.converged)


def fit_blocks(X, Y, subsets):
    return [CovMNL(s).fit(X, Y).to_dict() for s in subsets]


def lifts(X, Y, il):
    """Raw lift and lift beyond the covariates: observed joint over the joint expected under
    conditional independence given X (from the independent logits)."""
    M = Y.shape[1]
    Yf = Y.astype(float)
    p = Yf.mean(0)
    obs = Yf.T @ Yf / len(Y)
    ph = np.column_stack([CovMNL(b["items"], b["theta"]).pattern_probs(X)[:, 1] for b in il])
    exp = ph.T @ ph / len(Y)
    raw = obs / np.outer(p, p)
    res = obs / exp
    for a in (raw, res):
        a[np.diag_indices(M)] = np.nan
    return raw, res


def merge(cand, M):
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


def analyst_subsets(lift):
    M = len(lift)
    cand = [(abs(np.log(lift[i, j])), i, j) for i, j in itertools.combinations(range(M), 2)
            if lift[i, j] > LIFT_HI or lift[i, j] < LIFT_LO]
    return merge(sorted(cand, reverse=True), M)


def oracle_subsets(R):
    M = len(R)
    cand = [(abs(R[i, j]), i, j) for i, j in itertools.combinations(range(M), 2) if abs(R[i, j]) > .05]
    return merge(sorted(cand, reverse=True), M)


def gate(R, names):
    """Largest set of items connected by rho >= RHO_GATE."""
    M = len(R)
    comp = list(range(M))
    for i, j in itertools.combinations(range(M), 2):
        if R[i, j] >= RHO_GATE:
            a, b = comp[i], comp[j]
            comp = [a if c == b else c for c in comp]
    sizes = {c: comp.count(c) for c in set(comp)}
    big = max(sizes, key=sizes.get)
    linked = [names[m] for m in range(M) if comp[m] == big]
    return dict(passed=len(linked) >= GATE_SIZE, linked=linked, size=len(linked))


# ---------------------------------------------------------------------------
# Models: per-trip pattern distributions, averaged over trips
# ---------------------------------------------------------------------------
def cs_trip_dist(blocks, X, M):
    """(n, 2^M) pattern probabilities per trip: product over independent blocks."""
    PM = patterns(M)
    out = np.ones((len(X), len(PM)))
    for b in blocks:
        m = CovMNL(b["items"], b["theta"])
        out *= m.pattern_probs(X)[:, codes(PM[:, m.items])]
    return out


def cs_avg_dist(blocks, X, M, chunk=20000):
    return sum(cs_trip_dist(blocks, X[s:s + chunk], M).sum(0) for s in range(0, len(X), chunk)) / len(X)


def cs_holdout_ll(blocks, X, Y):
    lp = np.zeros(len(X))
    for b in blocks:
        m = CovMNL(b["items"], b["theta"])
        pr = m.pattern_probs(X)
        lp += np.log(np.clip(pr[np.arange(len(X)), codes(Y[:, m.items])], 1e-300, None))
    return lp


def probit_eta(fit, X):
    return np.asarray(fit["c"]) + X @ np.asarray(fit["B"]).T


def clean_R(R):
    R = np.asarray(R, float)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return R


def probit_avg_dist(fit, X, m=4096, seed=0, chunk=64):
    """Pattern distribution averaged over trips, by m scrambled-Sobol latent draws per trip
    (shared across trips): y = 1[eta + e > 0], e ~ N(0, R)."""
    eta = probit_eta(fit, X)
    M = eta.shape[1]
    L = np.linalg.cholesky(clean_R(fit["R"]))
    w = ndtri(np.clip(qmc.Sobol(M, scramble=True, seed=seed).random(m), 1e-12, 1 - 1e-12)) @ L.T
    bits = 1 << np.arange(M)[::-1]
    hist = np.zeros(1 << M)
    for s in range(0, len(eta), chunk):
        c = ((eta[s:s + chunk, None, :] + w[None]) > 0) @ bits
        hist += np.bincount(c.ravel(), minlength=1 << M)
    return hist / (len(eta) * m)


def probit_holdout_ll(fit, X, Y, orth):
    eta = probit_eta(fit, X)
    s = 2.0 * Y - 1
    R = clean_R(fit["R"])
    return np.log(np.clip(orth(s * eta, s[:, :, None] * s[:, None, :] * R), 1e-300, None))


def simulate_probit(fit, X, rng):
    eta = probit_eta(fit, X)
    L = np.linalg.cholesky(clean_R(fit["R"]))
    return (eta + rng.standard_normal(eta.shape) @ L.T > 0).astype(np.int8)


def simulate_cs(blocks, X, M, rng):
    Y = np.zeros((len(X), M), np.int8)
    for b in blocks:
        m = CovMNL(b["items"], b["theta"])
        pr = m.pattern_probs(X)
        k = (pr.cumsum(1) < rng.random((len(X), 1))).sum(1).clip(0, len(m.P) - 1)
        Y[:, m.items] = m.P[k].astype(np.int8)
    return Y


# ---------------------------------------------------------------------------
# Decisions: functions of the averaged pattern distribution q (2^M)
# ---------------------------------------------------------------------------
def quantities(q, M):
    PM = patterns(M)
    marg = q @ PM
    J2 = PM.T @ (q[:, None] * PM)
    pairs = list(itertools.combinations(range(M), 2))
    triples = list(itertools.combinations(range(M), 3))
    sets4 = list(itertools.combinations(range(M), 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = J2 / marg[:, None]           # P(B | A), row A
    np.fill_diagonal(cond, -np.inf)
    return dict(
        marg=marg,
        bundle2=np.array([J2[i, j] for i, j in pairs]),
        bundle3=np.array([q @ PM[:, list(t)].prod(1) for t in triples]),
        turf4=np.array([1 - q @ (PM[:, list(S)].sum(1) == 0) for S in sets4]),
        **{f"rec{a}": cond[a] for a in range(M)},
        options=dict(bundle2=pairs, bundle3=triples, turf4=sets4, **{f"rec{a}": list(range(M)) for a in range(M)}))


def decision_keys(M):
    return ["bundle2", "bundle3", "turf4"] + [f"rec{a}" for a in range(M)]


def picks(q, M):
    Q = quantities(q, M)
    return {k: int(np.argmax(Q[k])) for k in decision_keys(M)}


def score_vs_truth(q, qt, M):
    """Flips and regret (% of the truth's best) against a known truth (controls)."""
    Q, Qt = quantities(q, M), quantities(qt, M)
    out = {}
    for k in decision_keys(M):
        c, tv = int(np.argmax(Q[k])), Qt[k]
        out[k] = dict(flip=bool(c != int(np.argmax(tv))), regret_pct=float(100 * (tv.max() - tv[c]) / tv.max()))
    return out


def hh_pattern_counts(hh, Y, n_hh):
    C = np.zeros((n_hh, 1 << Y.shape[1]))
    np.add.at(C, (hh, codes(Y)), 1)
    return C


def adjudicate(pick_a, pick_b, C, M, n_boot=200, seed=SEED):
    """For each decision where a and b differ: holdout value of each pick and a household
    cluster bootstrap of the difference (a - b). Verdict 'a', 'b' or 'unresolved' at 95%."""
    q = C.sum(0) / C.sum()
    Q = quantities(q, M)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(C), np.full(len(C), 1 / len(C)), size=n_boot).astype(float)
    Qb = [quantities(w @ C / (w @ C).sum(), M) for w in W]
    out = {}
    for k in decision_keys(M):
        a, b = pick_a[k], pick_b[k]
        if a == b:
            out[k] = dict(differ=False)
            continue
        d = np.array([x[k][a] - x[k][b] for x in Qb])
        lo, hi = np.quantile(d, [0.025, 0.975])
        out[k] = dict(differ=True, hold_a=float(Q[k][a]), hold_b=float(Q[k][b]), diff_lo=float(lo),
                      diff_hi=float(hi), verdict="a" if lo > 0 else "b" if hi < 0 else "unresolved")
    return out
