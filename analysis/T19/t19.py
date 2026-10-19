"""T19 · MVP on real takeaway baskets: joint basket probabilities and bundle cannibalization.

Single entry point:
    python t19.py            # step 0 checks, then every restaurant x variant
    python t19.py step0      # data checks only
    python t19.py run R V    # one restaurant (1, 2) and variant (all, small, all_size)
    python t19.py report     # markdown tables from out/results_*.json -> out/report.md

Data: Kaggle henslersoftware/19560-indian-takeaway-orders, unzipped into data/ (not committed).
The proprietary engine is not used anywhere: joint likelihoods and scoring use GHK (d > 3) or the
library's Gauss-Legendre quadrature (d <= 3).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import qmc, spearmanr
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", message="Unknown solver options")

from multivariate_probit import MultivariateProbit, bvn_cdf
from multivariate_probit._mvn import lower_orthant

SEED = 19
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
START = {1: "2016-03-01", 2: "2016-05-01"}   # drop the sparse ramp-up months before regular trading
SMALL_VALUE = 20.0                            # one person's meal: main + rice + bread + papadum ~ £16
DISCOUNT = 0.15
N_BOOT = 1000
M1_DRAWS = 400                                # ancestral draws per holdout order for M1 marginals
C_WEAK = 100.0                                # weak L2 on every logit (guards against separation only)
SHRINK = 0.05                                 # M3: R <- (1 - s) R + s I; pairwise IFM + PD projection
                                              # lands on the PD boundary, which breaks GHK weights

ENTREES = ["MILD_CURRY", "MEDIUM_CURRY", "HOT_CURRY", "HOUSE_SPECIAL", "BIRYANI", "TANDOORI_MAIN"]
SIDES = ["VEG_SIDE", "RICE_PILAU", "RICE_PLAIN", "RICE_SPECIAL", "NAAN_PLAIN", "NAAN_FLAV",
         "BREAD_OTHER", "PAPADUM", "FRIES", "STARTER_FRIED", "STARTER_GRILL", "STARTER_OTHER", "DRINK"]
ADDONS = ["CONDIMENT", "SAUCE"]               # modelled, but not bundle components
GROUPS = ENTREES + SIDES + ADDONS


def load(r):
    o = pd.read_csv(DATA / f"restaurant-{r}-orders.csv")
    o.columns = ["order", "date", "item", "qty", "price", "lines"]
    o["date"] = pd.to_datetime(o["date"], format="%d/%m/%Y %H:%M")
    o["item"] = o["item"].str.strip()
    o["value"] = o.qty * o.price
    return o


# ----------------------------------------------------------------------------------- item groups

def has(n, *keys):
    return any(k in n for k in keys)


def classify(name, price):
    """Rule-based item -> group. Order of rules matters; price separates starters/sides from mains."""
    n = name.lower()
    if has(n, "coke", "cobra", "lemonade", "water", "wine"):
        return "DRINK"
    if has(n, "chutney", "mint sauce", "red sauce", "pickle", "raita", "raitha") or n == "dahi":
        return "CONDIMENT"
    if "papadum" in n:
        return "PAPADUM"
    if n.endswith(" sauce"):
        return "SAUCE"
    if "fries" in n:
        return "FRIES"
    if "biryani" in n:
        return "BIRYANI"
    if n == "pilau rice":
        return "RICE_PILAU"
    if n == "plain rice":
        return "RICE_PLAIN"
    if "rice" in n:
        return "RICE_SPECIAL"
    if "naan" in n:
        return "NAAN_PLAIN" if n in ("naan", "plain naan") else "NAAN_FLAV"
    if n == "puree" or has(n, "chapati", "roti", "paratha"):
        return "BREAD_OTHER"
    if "(main)" in n or "mixed grill" in n or "sizzler" in n or \
            (has(n, "shashlick", "shaslick") and "curry" not in n):
        return "TANDOORI_MAIN"
    if price <= 5.95:
        if has(n, "onion bhaj", "mini bhaji", "samosa", "pakora", "roll", "butterfly", "crispy aloo"):
            return "STARTER_FRIED"
        if has(n, "tikka", "kebab", "kehab", "starter", "tandoori chicken", "tandoori fish", "lamb chops"):
            return "STARTER_GRILL"
        if has(n, "puree", "chaat", "garlic mushroom"):
            return "STARTER_OTHER"
    if price <= 6.95:                           # vegetable side dishes (5.95) and veg curries (6.95)
        return "VEG_SIDE"
    if has(n, "salad"):
        return "VEG_SIDE"
    if has(n, "madras", "vindaloo", "jalfrezi", "chilli", "mirch", "phall", "naga"):
        return "HOT_CURRY"
    if has(n, "masala", "butter chicken", "korma", "kurma", "pasanda", "malaya"):
        return "MILD_CURRY"
    if has(n, "bhuna", "dupiaza", "rogon", "dhansak", "dansak", "pathia", "curry", "saag", "methi",
           "mushroom", "bombay", "cylon") and not has(n, "shashlick", "shaslick"):
        return "MEDIUM_CURRY"
    if has(n, "balti", "karahi", "garlic"):     # tikka balti / karahi / garlic are sauced dishes
        return "HOUSE_SPECIAL"
    if has(n, "tikka", "tandoori", "kebab"):
        return "TANDOORI_MAIN"
    return "HOUSE_SPECIAL"                      # balti, karahi and named house dishes


def item_map(o):
    t = o.groupby("item").agg(orders=("order", "nunique"), price=("price", "median")).reset_index()
    t["group"] = [classify(i, p) for i, p in zip(t["item"], t["price"])]
    return t.sort_values(["group", "orders"], ascending=[True, False])


# ----------------------------------------------------------------------------------- step 0

def step0(r):
    o = load(r)
    od = o.groupby("order").agg(date=("date", "first"), value=("value", "sum"), qty=("qty", "sum"),
                                items=("item", "nunique"), stamps=("date", "nunique"))
    days = od.date.dt.normalize()
    span = pd.date_range(days.min(), days.max())
    per_day = days.value_counts().reindex(span, fill_value=0)
    reg = per_day[per_day.index >= START[r]]
    res = {
        "restaurant": r,
        "rows": len(o), "orders": len(od), "distinct_items": int(o.item.nunique()),
        "first": str(od.date.min()), "last": str(od.date.max()),
        "orders_with_multiple_timestamps": int((od.stamps > 1).sum()),
        "orders_before_start": int((od.date < START[r]).sum()),
        "from_start": {"days": len(reg), "zero_order_days": int((reg == 0).sum()),
                       "orders_per_day_mean": round(reg.mean(), 1),
                       "orders_per_day_p10_p50_p90": reg.quantile([.1, .5, .9]).tolist()},
        "orders_by_year": od.date.dt.year.value_counts().sort_index().to_dict(),
        "share_by_hour": (od.date.dt.hour.value_counts(normalize=True).sort_index().round(3)).to_dict(),
        "share_by_weekday_mon0": (od.date.dt.dayofweek.value_counts(normalize=True).sort_index()
                                  .round(3)).to_dict(),
        "prices_multiple_of_5p": float((np.round(o.price * 100) % 5 == 0).mean()),
        "price_points": sorted(o.price.unique().tolist()),
        "distinct_items_per_order": od["items"].clip(upper=10).value_counts(normalize=True)
        .sort_index().round(3).to_dict(),
        "share_2plus_items": round(float((od["items"] >= 2).mean()), 3),
        "order_value_p5_p50_p95": od.value.quantile([.05, .5, .95]).round(2).tolist(),
        "share_value_le_small": round(float((od.value <= SMALL_VALUE).mean()), 3),
    }
    menu = pd.read_csv(DATA / f"restaurant-{r}-products-price.csv", encoding="utf-8-sig")
    menu.columns = ["item", "menu_price"]
    m = o.merge(menu, on="item", how="left")
    res["line_price_equals_menu_file"] = round(float((m.price == m.menu_price).mean()), 4)
    # every item with more than one recorded price: each price point with its date span, and how
    # often consecutive sales (in time) switch price. Interleaved points = variants, not a change.
    changes = []
    for it, s in o.groupby("item"):
        if s.price.nunique() < 2:
            continue
        s = s.sort_values("date")
        pts = s.groupby("price").agg(n=("order", "size"), first=("date", "min"), last=("date", "max"))
        changes.append({"item": it, "switches": int((s.price.diff().fillna(0) != 0).sum()),
                        "points": [{"price": float(p), "n": int(v.n), "first": str(v["first"].date()),
                                    "last": str(v["last"].date())} for p, v in pts.iterrows()]})
    res["multi_price_items"] = changes
    # did any item's price move over time? compare each item's modal price, first vs last 6 months
    o2 = o[o.date >= START[r]]
    early = o2[o2.date < o2.date.min() + pd.Timedelta(days=183)].groupby("item").price.agg(
        lambda x: x.mode().iloc[0])
    late = o2[o2.date > o2.date.max() - pd.Timedelta(days=183)].groupby("item").price.agg(
        lambda x: x.mode().iloc[0])
    both = pd.concat([early, late], axis=1, keys=["early", "late"]).dropna()
    res["modal_price_changed_first_vs_last_6m"] = both[both.early != both.late].reset_index() \
        .to_dict("records")
    res["items_compared_first_vs_last_6m"] = len(both)
    # menu renames: generic curry names retired when protein variants appear (2018-04/05)
    gen = o.groupby("item").date.agg(["min", "max"])
    res["items_last_seen_before_2018_06"] = int((gen["max"] < "2018-06-01").sum())
    res["items_first_seen_after_2018_03"] = int((gen["min"] > "2018-03-01").sum())
    imap = item_map(o2)
    imap.to_csv(OUT / f"mapping_r{r}.csv", index=False)
    return res


# ----------------------------------------------------------------------------------- prep

def prep(r, variant):
    o = load(r)
    o = o[o.date >= START[r]]
    imap = item_map(o)
    o = o.merge(imap[["item", "group"]], on="item")
    od = o.groupby("order").agg(date=("date", "first"), value=("value", "sum"), qty=("qty", "sum"))
    Y = (pd.crosstab(o.order, o.group) > 0).reindex(columns=GROUPS, fill_value=False) \
        .reindex(od.index).astype(np.int8)
    if variant == "small":
        keep = od.value <= SMALL_VALUE
        od, Y = od[keep], Y[keep]
    h = od.date.dt.hour
    dow = od.date.dt.dayofweek
    X = pd.DataFrame({"h_lt17": h < 17, "h17": h == 17, "h19": h == 19, "h_ge20": h >= 20,
                      "fri": dow == 4, "sat": dow == 5, "sun": dow == 6}, index=od.index).astype(float)
    if variant == "all_size":
        ls = np.log(od.qty)
        X["log_qty"] = (ls - ls.mean()) / ls.std()
    dates = np.sort(od.date.dt.normalize().unique())
    cut = dates[int(0.75 * len(dates))]
    tr = (od.date < cut).values
    return dict(X=X.values, Y=Y.values, tr=tr, cut=str(pd.Timestamp(cut).date()), od=od,
                xcols=list(X.columns))


# ----------------------------------------------------------------------------------- GHK

def ghk_logp(eta, Y, R, m=1024, n_scr=4, chunk=128, seed=SEED):
    """log P(Y = y | eta, R) per row by GHK: n_scr independent scrambled Sobol sets of m points.
    Returns (logp over all n_scr*m weights, logp per scramble)."""
    n, d = eta.shape
    L = np.linalg.cholesky(R)
    U = [qmc.Sobol(d, scramble=True, seed=seed + s).random(m) for s in range(n_scr)]
    per = np.zeros((n, n_scr))
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        s = 2.0 * Y[sl] - 1.0
        a = s * eta[sl]
        Ls = s[:, :, None] * L[None] * s[:, None, :]
        b = a.shape[0]
        for k, u in enumerate(U):
            e = np.zeros((b, m, d))
            lw = np.zeros((b, m))
            for i in range(d):
                mu = e[:, :, :i] @ Ls[:, i, :i, None] if i else np.zeros((b, m, 1))
                t = (a[:, i, None] - mu[..., 0]) / L[i, i]
                lw += log_ndtr(t)
                p = ndtr(t)
                e[:, :, i] = ndtri(np.clip(u[None, :, i] * p, 1e-300, 1 - 1e-16))
            per[sl, k] = logsumexp(lw, axis=1) - np.log(m)
    return logsumexp(per, axis=1) - np.log(n_scr), per


# ----------------------------------------------------------------------------------- models

def logit(X, y):
    if y.min() == y.max():                       # degenerate column (never/always bought in train)
        return None
    return LogisticRegression(C=C_WEAK, max_iter=2000).fit(X, y)


def lp(model, X, y):
    """log P(y | X) per row for a fitted binary logit (None = degenerate)."""
    if model is None:
        return np.zeros(len(y))
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return np.where(y == 1, np.log(p), np.log1p(-p))


def fit_m0(X, Y):
    return [logit(X, Y[:, j]) for j in range(Y.shape[1])]


def m0_marg(ms, X, Ytr):
    return np.column_stack([m.predict_proba(X)[:, 1] if m is not None else
                            np.full(len(X), Ytr[:, j].mean()) for j, m in enumerate(ms)])


def fit_m1(X, Y, order):
    """MBC-style sequential conditionals: item order[k] ~ covariates + presence of order[:k]."""
    return [logit(np.hstack([X, Y[:, order[:k]]]), Y[:, j]) for k, j in enumerate(order)]


def m1_logp(ms, X, Y, order):
    return sum(lp(m, np.hstack([X, Y[:, order[:k]]]), Y[:, j]) for k, (m, j) in
               enumerate(zip(ms, order)))


def m1_sample(ms, X, order, d, Ytr, rng, R=M1_DRAWS):
    """Ancestral draws, (n, R, d) int8."""
    n = len(X)
    S = np.zeros((n, R, d), np.int8)
    for k, (m, j) in enumerate(zip(ms, order)):
        if m is None:
            S[:, :, j] = Ytr[0, j]
            continue
        w = m.coef_[0]
        base = m.intercept_[0] + X @ w[:X.shape[1]]
        z = base[:, None] + (S[:, :, order[:k]] @ w[X.shape[1]:]) if k else \
            np.repeat(base[:, None], R, 1)
        S[:, :, j] = rng.random((n, R)) < 1 / (1 + np.exp(-z))
    return S


def fit_m2(X, Y, top):
    code = (Y[:, top] * (1 << np.arange(len(top)))).sum(1)
    return LogisticRegression(C=C_WEAK, max_iter=5000).fit(X, code)


def m2_cellprob(m, X, k):
    P = np.zeros((len(X), 1 << k))
    P[:, m.classes_] = m.predict_proba(X)
    return P


# ----------------------------------------------------------------------------------- per-order event probabilities

def events_all(cols_list, d):
    return [tuple(c) for c in cols_list]


def m3_event_probs(eta, R, events):
    """P(all outcomes in event = 1) per row; eta rows deduplicated first."""
    uq, inv = np.unique(np.round(eta, 10), axis=0, return_inverse=True)
    out = np.zeros((len(eta), len(events)))
    for e_i, ev in enumerate(events):
        ev = list(ev)
        if len(ev) == 1:
            p = ndtr(uq[:, ev[0]])
        elif len(ev) == 2:
            p = bvn_cdf(uq[:, ev[0]], uq[:, ev[1]], np.full(len(uq), R[ev[0], ev[1]]))
        else:
            sub = R[np.ix_(ev, ev)]
            p = lower_orthant(uq[:, ev], np.repeat(sub[:, :, None], len(uq), 2))
        out[:, e_i] = np.asarray(p).reshape(-1)[inv.reshape(-1)]
    return out


def indep_event_probs(P, events):
    return np.column_stack([P[:, list(ev)].prod(1) for ev in events])


def sample_event_probs(S, events):
    return np.column_stack([S[:, :, list(ev)].all(2).mean(1) for ev in events])


def m2_event_probs(C, top, events):
    """None for events outside the coded subset."""
    pos = {g: i for i, g in enumerate(top)}
    cells = np.arange(C.shape[1])
    out = np.full((len(C), len(events)), np.nan)
    for e_i, ev in enumerate(events):
        if all(g in pos for g in ev):
            mask = np.all([(cells >> pos[g]) & 1 for g in ev], axis=0)
            out[:, e_i] = C[:, mask].sum(1)
    return out


def obs_events(Y, events):
    return np.column_stack([Y[:, list(ev)].all(1) for ev in events]).astype(float)


# ----------------------------------------------------------------------------------- bootstrap

def boot_weights(n, rng, B=N_BOOT):
    return rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float32)


def summarize(x):
    x = np.asarray(x, float)
    return {"pt": float(x[0]), "lo": float(np.nanpercentile(x[1:], 2.5)),
            "hi": float(np.nanpercentile(x[1:], 97.5))}


def boot_rates(W, M):
    """rates under the original sample (row 0) and each bootstrap weight row."""
    n = M.shape[0]
    full = np.nanmean(M, 0)[None]
    return np.vstack([full, (W @ np.nan_to_num(M)) / n])


# ----------------------------------------------------------------------------------- run

def run(r, variant):
    rng = np.random.default_rng(SEED)
    D = prep(r, variant)
    X, Y, tr = D["X"], D["Y"], D["tr"]
    d = Y.shape[1]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[~tr], Y[~tr]
    nte = len(Yte)
    freq = Ytr.mean(0)
    order = list(np.argsort(-freq, kind="stable"))
    res = {"restaurant": r, "variant": variant, "cut": D["cut"], "n_train": int(tr.sum()),
           "n_holdout": int(nte), "covariates": D["xcols"],
           "train_rate": dict(zip(GROUPS, freq.round(4).tolist())),
           "holdout_rate": dict(zip(GROUPS, Yte.mean(0).round(4).tolist())),
           "groups_per_order_train": float(Ytr.sum(1).mean())}

    # M2's coded subset: the most frequent entree plus the 4 most frequent bundle sides, so that
    # M2 can be scored on at least some bundles (the overall top 5 contains no entree).
    gi = {g: i for i, g in enumerate(GROUPS)}
    top_entree = max(ENTREES, key=lambda g: freq[gi[g]])
    top_sides = sorted(SIDES, key=lambda g: -freq[gi[g]])[:4]
    top = [gi[top_entree]] + [gi[g] for g in top_sides]
    res["m2_groups"] = [GROUPS[j] for j in top]
    res["top5_overall"] = [GROUPS[j] for j in order[:5]]
    res["m1_order"] = [GROUPS[j] for j in order]

    # ---- fit
    m0 = fit_m0(Xtr, Ytr)
    m1 = fit_m1(Xtr, Ytr, order)
    m1r = fit_m1(Xtr, Ytr, order[::-1])
    m2 = fit_m2(Xtr, Ytr, top)
    m3 = MultivariateProbit(inner="linear", dependence="pairwise", cv=5, random_state=SEED)
    m3.fit(Xtr, Ytr)
    Rraw = m3.correlation_
    Rm = (1 - SHRINK) * Rraw + SHRINK * np.eye(d)
    eta = m3.transform(Xte)
    res["m3_corr"] = Rm.round(4).tolist()
    res["m3_corr_raw_min_eig"] = float(np.linalg.eigvalsh(Rraw)[0])
    res["m3_calibration_slope"] = dict(zip(GROUPS, np.round(m3.calibration_[:, 1], 3).tolist()))
    res["m1_cross_effects"] = {GROUPS[j]: dict(zip([GROUPS[i] for i in order[:k]],
                                                   np.round(m.coef_[0][X.shape[1]:], 3).tolist()))
                               for k, (m, j) in enumerate(zip(m1, order)) if m is not None and k}

    # ---- joint basket log-likelihood (holdout)
    ll = {"M0": sum(lp(m, Xte, Yte[:, j]) for j, m in enumerate(m0)),
          "M1": m1_logp(m1, Xte, Yte, order),
          "M1_rev": m1_logp(m1r, Xte, Yte, order[::-1])}
    ll3, per3 = ghk_logp(eta, Yte.astype(float), Rm)
    ll["M3"] = ll3
    res["ghk_sum_ll_by_scramble"] = per3.sum(0).round(2).tolist()
    ll["M3_raw"], per3r = ghk_logp(eta, Yte.astype(float), Rraw + 1e-9 * np.eye(d))
    res["ghk_sum_ll_by_scramble_raw"] = per3r.sum(0).round(2).tolist()
    # top-5 coded-subset sub-basket: all four models on the same 5 outcomes
    code_te = (Yte[:, top] * (1 << np.arange(5))).sum(1)
    C2 = m2_cellprob(m2, Xte, 5)
    sub = {"M2": np.log(np.clip(C2[np.arange(nte), code_te], 1e-12, None))}
    P0 = m0_marg(m0, Xte, Ytr)
    sub["M0"] = np.where(Yte[:, top] == 1, np.log(P0[:, top]), np.log1p(-P0[:, top])).sum(1)
    sub["M3"], _ = ghk_logp(eta[:, top], Yte[:, top].astype(float), Rm[np.ix_(top, top)])
    S1 = m1_sample(m1, Xte, order, d, Ytr, rng)
    S1r = m1_sample(m1r, Xte, order[::-1], d, Ytr, rng)
    # M1's sub-basket probability needs marginalising the other groups: ancestral draws, smoothed
    # by 0.5 count over 32 cells (reported, but approximate, unlike the other three).
    cnt = np.zeros((nte, 32))
    sc = (S1[:, :, top] * (1 << np.arange(5))).sum(2)
    for c in range(32):
        cnt[:, c] = (sc == c).sum(1)
    sub["M1_mc"] = np.log((cnt[np.arange(nte), code_te] + 0.5) / (M1_DRAWS + 16))

    # ---- events: all pairs, top-50 triples (by train frequency), bundles
    pairs = list(combinations(range(d), 2))
    trip_all = list(combinations(range(d), 3))
    tf = obs_events(Ytr, trip_all).mean(0)
    triples = [trip_all[i] for i in np.argsort(-tf)[:50]]
    bundles = [(gi[e],) + tuple(gi[s] for s in c) for e in ENTREES for k in (1, 2)
               for c in combinations(SIDES, k)]
    blocks = {"pairs": pairs, "triples": triples, "bundles": bundles}
    W = boot_weights(nte, rng)
    res["ll"] = {}
    for k, v in list(ll.items()):
        res["ll"][k] = summarize(np.concatenate([[v.mean()], (W @ v) / nte]))
    for a, b in (("M3", "M0"), ("M3", "M1"), ("M3", "M1_rev"), ("M3", "M3_raw")):
        dv = ll[a] - ll[b]
        res["ll"][f"{a}-{b}"] = summarize(np.concatenate([[dv.mean()], (W @ dv) / nte]))
    res["ll_sub5"] = {}
    for k, v in sub.items():
        res["ll_sub5"][k] = summarize(np.concatenate([[v.mean()], (W @ v) / nte]))
    for b in ("M0", "M2", "M1_mc"):
        dv = sub["M3"] - sub[b]
        res["ll_sub5"][f"M3-{b}"] = summarize(np.concatenate([[dv.mean()], (W @ dv) / nte]))

    models = ("M0", "M1", "M1_rev", "M2", "M3", "M3_raw")
    bundle_rows = None
    for name, evs in blocks.items():
        obs = obs_events(Yte, evs)
        pred = {"M0": indep_event_probs(P0, evs),
                "M1": sample_event_probs(S1, evs),
                "M1_rev": sample_event_probs(S1r, evs),
                "M2": m2_event_probs(C2, top, evs),
                "M3": m3_event_probs(eta, Rm, evs),
                "M3_raw": m3_event_probs(eta, Rraw, evs)}
        Ob = boot_rates(W, obs)
        Pb = {k: boot_rates(W, v) for k, v in pred.items()}
        blk = {"n_events": len(evs)}
        for k in models:
            err = np.abs(Pb[k] - Ob) * 100                       # pts
            cov = ~np.isnan(Pb[k][0]) if k == "M2" else np.ones(len(evs), bool)
            if not cov.any():
                continue
            blk[f"{k}_mae_pts"] = summarize(np.nanmean(np.where(cov, err, np.nan), 1))
            blk[f"{k}_n_covered"] = int(cov.sum())
        for k in ("M0", "M1", "M1_rev", "M3_raw"):
            dv = np.abs(Pb[k] - Ob).mean(1) - np.abs(Pb["M3"] - Ob).mean(1)
            blk[f"{k}-M3_mae_pts"] = summarize(dv * 100)
        # M2 subset: all models on the events M2 covers
        cov = ~np.isnan(Pb["M2"][0])
        if cov.any():
            blk["on_m2_subset"] = {k: summarize(np.abs(Pb[k][:, cov] - Ob[:, cov]).mean(1) * 100)
                                   for k in models}
        if name == "bundles":
            # ranking: top-10 by observed full-bundle rate vs each model's top-10
            def overlap(p, o):
                return len(set(np.argsort(-p)[:10]) & set(np.argsort(-o)[:10]))
            for k in ("M0", "M1", "M1_rev", "M3", "M3_raw"):
                blk[f"{k}_top10_overlap"] = summarize([overlap(Pb[k][i], Ob[i])
                                                        for i in range(len(Ob))])
                blk[f"{k}_spearman"] = summarize([spearmanr(Pb[k][i], Ob[i])[0]
                                                  for i in range(0, len(Ob), 5)])
                top10 = np.argsort(-Ob[0])[:10]
                blk[f"{k}_mae_obs_top10_pts"] = summarize(
                    np.abs(Pb[k][:, top10] - Ob[:, top10]).mean(1) * 100)
            bundle_rows = pd.DataFrame({"bundle": [" + ".join(GROUPS[j] for j in ev) for ev in evs],
                                        "size": [len(ev) for ev in evs],
                                        "entree_rate": [Yte[:, ev[0]].mean() for ev in evs],
                                        "observed": Ob[0], **{k: Pb[k][0] for k in models}})
        res[name] = blk

    bundle_rows = bundle_rows.sort_values("observed", ascending=False)
    bundle_rows.to_csv(OUT / f"bundles_r{r}_{variant}.csv", index=False, float_format="%.5f")

    # ---- account lead: most popular entree + 2-side bundle in the holdout
    prices = group_prices(r)
    b3 = bundle_rows[bundle_rows["size"] == 3].iloc[0]
    parts = b3.bundle.split(" + ")
    bprice = sum(prices[g] for g in parts)
    pe = b3.entree_rate
    lead = {"bundle": b3.bundle, "list_price": round(bprice, 2), "entree_rate": pe}
    for k in ("observed", "M0", "M1", "M3"):
        share = b3[k] / pe
        lead[k] = {"already_buy_all_given_entree": share,
                   "discount_given_away_per_100_entree_orders": 100 * share * DISCOUNT * bprice}
    res["account_lead"] = lead
    res["group_price"] = prices
    tag = f"r{r}_{variant}"
    (OUT / f"results_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    if variant == "all":
        heatmap(Rm, OUT / f"corr_{tag}.png", f"Restaurant {r}: M3 latent correlations (train)")
    return res


def group_prices(r):
    """order-weighted median list price of each group (lines from START on)."""
    o = load(r)
    o = o[o.date >= START[r]]
    o["group"] = [classify(i, p) for i, p in zip(o["item"], o.groupby("item")["price"]
                                                  .transform("median"))]
    return o.groupby("group").price.median().round(2).to_dict()


def heatmap(R, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [g.lower().replace("_", " ") for g in GROUPS]
    fig, ax = plt.subplots(figsize=(9, 8))
    M = R.copy()
    np.fill_diagonal(M, np.nan)
    v = np.nanmax(np.abs(M))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(R)):
        for j in range(len(R)):
            if i != j:
                ax.text(j, i, f"{R[i, j]:.2f}".replace("0.", "."), ha="center", va="center",
                        fontsize=5, color="#222")
    for k in (len(ENTREES), len(ENTREES) + len(SIDES)):
        ax.axhline(k - .5, color="#555", lw=.6)
        ax.axvline(k - .5, color="#555", lw=.6)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.7, label="latent correlation (blue = substitutes, red = complements)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------- report tables

def fmt(c, k=2, sign=False):
    f = f"{{:{'+' if sign else ''}.{k}f}}"
    return f"{f.format(c['pt'])} [{f.format(c['lo'])}, {f.format(c['hi'])}]"


def report():
    L = []
    R = {(r, v): json.loads((OUT / f"results_r{r}_{v}.json").read_text())
         for r in (1, 2) for v in ("all", "small", "all_size")}
    L.append("### Samples\n\n| restaurant | variant | train orders | holdout orders | cut | groups/order |")
    L.append("|---|---|---|---|---|---|")
    for (r, v), x in R.items():
        L.append(f"| {r} | {v} | {x['n_train']} | {x['n_holdout']} | {x['cut']} | "
                 f"{x['groups_per_order_train']:.2f} |")
    L.append("\n### Holdout joint basket log-likelihood (nats per order, 21 groups)\n")
    L.append("| r | variant | M0 | M1 | M1 reversed | M3 | M3 − M0 | M3 − M1 | M3 − M1 rev |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (r, v), x in R.items():
        l = x["ll"]
        L.append(f"| {r} | {v} | {l['M0']['pt']:.3f} | {l['M1']['pt']:.3f} | {l['M1_rev']['pt']:.3f} | "
                 f"{l['M3']['pt']:.3f} | {fmt(l['M3-M0'], 3, True)} | {fmt(l['M3-M1'], 3, True)} | "
                 f"{fmt(l['M3-M1_rev'], 3, True)} |")
    L.append("\n### Coded-subset sub-basket log-likelihood (5 groups, nats per order)\n")
    L.append("| r | variant | M2 groups | M0 | M1 (MC) | M2 | M3 | M3 − M2 | M3 − M0 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (r, v), x in R.items():
        l = x["ll_sub5"]
        L.append(f"| {r} | {v} | {', '.join(g.lower() for g in x['m2_groups'])} | {l['M0']['pt']:.3f} | "
                 f"{l['M1_mc']['pt']:.3f} | {l['M2']['pt']:.3f} | {l['M3']['pt']:.3f} | "
                 f"{fmt(l['M3-M2'], 3, True)} | {fmt(l['M3-M0'], 3, True)} |")
    for blk, title in (("pairs", "All 210 pairs"), ("triples", "Top 50 triples (train frequency)"),
                       ("bundles", "All 546 bundles (decision test)")):
        L.append(f"\n### {title}: mean absolute error of the co-occurrence rate, pts\n")
        L.append("| r | variant | M0 | M1 | M1 rev | M3 | M0 − M3 | M1 − M3 | M1 rev − M3 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for (r, v), x in R.items():
            b = x[blk]
            L.append(f"| {r} | {v} | {b['M0_mae_pts']['pt']:.3f} | {b['M1_mae_pts']['pt']:.3f} | "
                     f"{b['M1_rev_mae_pts']['pt']:.3f} | {b['M3_mae_pts']['pt']:.3f} | "
                     f"{fmt(b['M0-M3_mae_pts'], 3, True)} | {fmt(b['M1-M3_mae_pts'], 3, True)} | "
                     f"{fmt(b['M1_rev-M3_mae_pts'], 3, True)} |")
    L.append("\n### Events inside M2's coded subset (all models on the same events), MAE pts\n")
    L.append("| r | variant | block | n | M0 | M1 | M2 | M3 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (r, v), x in R.items():
        for blk in ("pairs", "triples", "bundles"):
            s_ = x[blk]["on_m2_subset"]
            L.append(f"| {r} | {v} | {blk} | {x[blk]['M2_n_covered']} | {fmt(s_['M0'])} | {fmt(s_['M1'])} | "
                     f"{fmt(s_['M2'])} | {fmt(s_['M3'])} |")
    L.append("\n### Ranking bundles by full-bundle rate (holdout observed vs model)\n")
    L.append("| r | variant | model | top-10 overlap /10 | Spearman (546) | MAE on observed top 10, pts |")
    L.append("|---|---|---|---|---|---|")
    for (r, v), x in R.items():
        b = x["bundles"]
        for k in ("M0", "M1", "M1_rev", "M3", "M3_raw"):
            L.append(f"| {r} | {v} | {k} | {fmt(b[k + '_top10_overlap'], 0)} | {fmt(b[k + '_spearman'], 3)} | "
                     f"{fmt(b[k + '_mae_obs_top10_pts'])} |")
    L.append("\n### Shrinkage check: M3 (s = 0.05) vs unshrunk pairwise-IFM correlation\n")
    L.append("| r | variant | raw min eig | LL M3 − M3 raw | pairs M3 raw − M3 | bundles M3 raw − M3 |")
    L.append("|---|---|---|---|---|---|")
    for (r, v), x in R.items():
        L.append(f"| {r} | {v} | {x['m3_corr_raw_min_eig']:.1e} | {fmt(x['ll']['M3-M3_raw'], 3, True)} | "
                 f"{fmt(x['pairs']['M3_raw-M3_mae_pts'], 3, True)} | "
                 f"{fmt(x['bundles']['M3_raw-M3_mae_pts'], 3, True)} |")
    for r in (1, 2):
        x = R[(r, "all")]
        C = np.array(x["m3_corr"])
        iu = np.triu_indices(len(GROUPS), 1)
        vals = C[iu]
        L.append(f"\n### Restaurant {r} (all orders): M3 top complements and substitutes\n")
        L.append("| complement pair | ρ | substitute pair | ρ |\n|---|---|---|---|")
        hi, lo = np.argsort(-vals)[:10], np.argsort(vals)[:10]
        for a, b in zip(hi, lo):
            L.append(f"| {GROUPS[iu[0][a]].lower()} + {GROUPS[iu[1][a]].lower()} | {vals[a]:+.2f} | "
                     f"{GROUPS[iu[0][b]].lower()} + {GROUPS[iu[1][b]].lower()} | {vals[b]:+.2f} |")
        bt = pd.read_csv(OUT / f"bundles_r{r}_all.csv")
        L.append(f"\n### Restaurant {r} (all orders): top 15 bundles by observed holdout full-bundle rate (%)\n")
        L.append("| bundle | observed | M0 | M1 | M3 | M0 err | M1 err | M3 err |\n|---|---|---|---|---|---|---|---|")
        for _, b in bt.head(15).iterrows():
            L.append(f"| {b.bundle.lower().replace('_', ' ')} | {100 * b.observed:.1f} | {100 * b.M0:.1f} | "
                     f"{100 * b.M1:.1f} | {100 * b.M3:.1f} | {100 * (b.M0 - b.observed):+.1f} | "
                     f"{100 * (b.M1 - b.observed):+.1f} | {100 * (b.M3 - b.observed):+.1f} |")
        a = x["account_lead"]
        L.append(f"\nAccount lead, restaurant {r}: {a['bundle']} (list £{a['list_price']:.2f}), entree rate "
                 f"{100 * a['entree_rate']:.1f}%: " + "; ".join(
                     f"{k} {100 * a[k]['already_buy_all_given_entree']:.1f}% already buy all, "
                     f"£{a[k]['discount_given_away_per_100_entree_orders']:.2f} given away per 100 entree orders"
                     for k in ("observed", "M0", "M1", "M3")))
        L.append(f"\nM3 calibration slopes r{r}: {x['m3_calibration_slope']}")
        L.append(f"\nGHK summed LL by scramble r{r}: shrunk {x['ghk_sum_ll_by_scramble']}, "
                 f"raw {x['ghk_sum_ll_by_scramble_raw']}")
    (OUT / "report.md").write_text("\n".join(L) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:]
    if args[:1] == ["report"]:
        report()
        return
    if args[:1] == ["run"]:
        run(int(args[1]), args[2])
        return
    s0 = {r: step0(r) for r in (1, 2)}
    (OUT / "step0.json").write_text(json.dumps(s0, indent=1, default=str))
    if args[:1] == ["step0"]:
        return
    for r in (1, 2):
        for v in ("all", "small", "all_size"):
            print(f"== r{r} {v}", flush=True)
            run(r, v)
    report()


if __name__ == "__main__":
    main()
