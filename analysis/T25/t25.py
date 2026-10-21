"""T25: shelf-assortment search, same GA with five share evaluators, under fixed wall-clock budgets (TEST.md).

    python analysis/T25/t25.py prep          # local: category, SKUs, occasions, availability, store clusters -> out/prep.npz
    python analysis/T25/t25.py lc-diag       # Modal CPU: LC-MNL K = 10 vs 20 on 80% of households, holdout LL
    python analysis/T25/t25.py lc-fit        # Modal CPU: LC-MNL at the chosen K on all households, per cluster
    python analysis/T25/t25.py truth         # local: Sigma by switching-rate moment match, truth + arm-A model -> out/truth_c{0,1}.npz

Data: completejourney (R package, full get_transactions(), 53 weeks). Category SOFT DRINKS, top 60 SKUs by purchase
occasions (user 2026-10-20: replaces the >= 30-occasion rule, which leaves 356 SKUs). Column map: COMMODITY_DESC =
product_category, SUB_COMMODITY_DESC = product_type, BRAND = brand (National / Private), CURR_SIZE_OF_PRODUCT =
package_size, MANUFACTURER = manufacturer_id.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = HERE.parents[1] / "data" / "completejourney"
SEED = 20261020
CAT, J = "SOFT DRINKS", 60
STORE_MIN, K_CLUSTERS, OUTSIDE_MULT = 200, 2, 3
SCALE = np.sqrt(6) / np.pi                      # logit -> unit-variance probit: diff var pi^2/3 -> 2
ND_EXTRA = 0.2
PREP, PFX = {2: "prep.npz", 5: "prep_k5.npz"}, {2: "", 5: "k5_"}
CLIENT = "103"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data (local)
def size_oz(sub, size):
    """Estimated total volume per purchase unit (oz). The table gives the unit size only; pack counts come from the
    sub-commodity (12-pack cans, 20/24-pack cans, 6-pack multipack bottles)."""
    s = (size or "").upper()
    m = re.match(r"([\d.]+)\s*(OZ|L|LTR|PK)", s)
    if "2 LITER" in sub:
        return 67.6
    v = float(m.group(1)) * (33.8 if m and m.group(2) in ("L", "LTR") else 1) if m else np.nan
    if "12/18&15PK" in sub:
        return 144.0 if (m is None or m.group(2) == "PK" or v <= 16) else v
    if "20PK&24PK" in sub:
        return 288.0
    if "MLT-PK" in sub:
        return 6 * v
    return v


def band_of(oz):
    return np.digitize(oz, [24.5, 100.0, 200.0])          # 0 single, 1 2 L, 2 12-pack / 6-pack, 3 20-24 pack


def prep(k=K_CLUSTERS, store_min=STORE_MIN, fname="prep.npz"):
    """k = 2 -> prep.npz (decision 1); k = 5 -> prep_k5.npz (decision 2). Same code path, so prep.npz is unchanged."""
    import pandas as pd
    import pyreadr
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(SEED)
    t = pyreadr.read_r(str(DATA / "transactions.rds"))[None]
    p = pyreadr.read_r(str(DATA / "products.rda"))["products"].set_index("product_id")
    for c in ("household_id", "store_id", "basket_id", "product_id", "quantity", "sales_value", "week",
              "transaction_timestamp"):
        assert c in t.columns, c
    log(f"transactions {len(t):,} rows, {t.household_id.nunique()} households, {t.week.nunique()} weeks")
    cat_ids = p.index[p.product_category == CAT]
    tc = t[t.product_id.isin(cat_ids)]
    occ = tc.drop_duplicates(["basket_id", "product_id"]).groupby("product_id").size().sort_values(ascending=False)
    items = occ.index[:J].to_numpy()
    log(f"{CAT}: {len(occ)} SKUs sold, {(occ >= 30).sum()} with >= 30 occasions; top {J} kept "
        f"(occasions {occ.iloc[J - 1]}-{occ.iloc[0]})")
    pr = p.loc[items]
    sub = pr.product_type.astype(str).to_numpy()
    mfr = pr.manufacturer_id.astype(str).to_numpy()
    pl = (pr.brand == "Private").to_numpy()
    oz = np.array([size_oz(s, z) for s, z in zip(sub, pr.package_size.astype(str))])
    band = band_of(oz)
    col = {q: j for j, q in enumerate(items)}

    tc = tc.assign(month=tc.transaction_timestamp.dt.month)
    t = t.assign(month=t.transaction_timestamp.dt.month)
    lines = tc[tc.product_id.isin(items) & (tc.quantity > 0) & (tc.sales_value > 0)].copy()
    lines["j"] = lines.product_id.map(col)
    lines["unit"] = lines.sales_value / lines.quantity
    price = lines.groupby("j").unit.mean().reindex(range(J)).to_numpy()   # average unit price paid

    # store clusters on category sales-mix shares
    sb = lines.groupby("store_id").basket_id.nunique()
    stores = sb.index[sb >= store_min].to_numpy()
    mix = lines[lines.store_id.isin(stores)].pivot_table(index="store_id", columns="j", values="sales_value",
                                                         aggfunc="sum", fill_value=0.0).reindex(columns=range(J),
                                                                                                 fill_value=0.0)
    mix = mix.div(mix.sum(1), axis=0).loc[stores]
    lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(mix.to_numpy())
    log(f"stores with >= {store_min} category occasions: {len(stores)}; cluster sizes {np.bincount(lab).tolist()}")

    # availability: sold at the store that month
    av_sm = lines.groupby(["store_id", "month"]).j.apply(lambda s: np.bincount(s, minlength=J) > 0)
    pm = lines.groupby(["store_id", "month", "j"]).unit.median()           # occasion price (fit only)

    # choice occasions: baskets with >= 1 top-60 line; the highest-spend line is the choice
    ch = (lines[lines.store_id.isin(stores)].sort_values("sales_value", ascending=False)
          .drop_duplicates("basket_id"))
    any_cat = set(tc.basket_id.unique())
    out = dict(items=items, sub=sub, mfr=mfr, pl=pl, oz=oz, band=band, price=price,
               occ=occ.loc[items].to_numpy(), name=np.array([f"{m}|{'PL' if q else 'NB'}|{s[:22]}|{z}|{i}" for m, q, s, z, i in
                              zip(mfr, pl, sub, pr.package_size.astype(str), items)]))
    for c in range(k):
        st = set(stores[lab == c])
        ins = ch[ch.store_id.isin(st)]
        ins_sw = set(zip(ins.store_id, ins.week))
        bk = t[t.store_id.isin(st)].drop_duplicates("basket_id")[["basket_id", "household_id", "store_id", "week",
                                                                    "month"]]
        bk = bk[~bk.basket_id.isin(any_cat)]
        bk = bk[[sw in ins_sw for sw in zip(bk.store_id, bk.week)]]
        n_out = min(len(bk), OUTSIDE_MULT * len(ins))
        bk = bk.iloc[rng.choice(len(bk), n_out, replace=False)]
        occs = pd.concat([ins[["household_id", "store_id", "week", "month", "j"]],
                          bk.assign(j=J)[["household_id", "store_id", "week", "month", "j"]]], ignore_index=True)
        av = np.stack([av_sm.get((s, mo), np.zeros(J, bool)) for s, mo in zip(occs.store_id, occs.month)])
        y = occs.j.to_numpy()
        assert av[y < J, y[y < J]].all()
        pmat = np.tile(price, (len(occs), 1))
        key = pd.MultiIndex.from_arrays([occs.store_id.to_numpy().repeat(J), occs.month.to_numpy().repeat(J),
                                         np.tile(np.arange(J), len(occs))])
        loc = pm.reindex(key).to_numpy().reshape(len(occs), J)
        pmat = np.where(np.isnan(loc), pmat, loc)
        hh = occs.household_id.astype("category").cat.codes.to_numpy()
        sales = lines[lines.store_id.isin(st)].groupby("j").sales_value.sum().reindex(range(J), fill_value=0)
        out.update({f"c{c}_y": y, f"c{c}_av": av, f"c{c}_p": pmat.astype(np.float32), f"c{c}_hh": hh,
                    f"c{c}_week": occs.week.to_numpy(), f"c{c}_sales": sales.to_numpy(),
                    f"c{c}_stores": np.array(sorted(st)), f"c{c}_stocked": stocked(av_sm, st),
                    f"c{c}_trips": t[t.store_id.isin(st)].basket_id.nunique()})
        log(f"cluster {c}: {len(st)} stores, {len(ins)} category occasions, {n_out} outside, {hh.max() + 1} "
            f"households; choice-set size median {np.median(av.sum(1)):.0f}; top-3 by sales "
            f"{[out['name'][j] for j in np.argsort(-sales.to_numpy())[:3]]}")
    nd = near_dup(mfr, sub, band)
    iu = np.triu_indices(J, 1)
    log(f"near-duplicate pairs (same manufacturer + sub-commodity + same/adjacent size band): {nd[iu].sum()} of "
        f"{len(iu[0])}; diet/regular pairs: not identifiable (no flavor or description field in this release)")
    log(f"size bands {np.bincount(band, minlength=4).tolist()}; private label {pl.sum()}; manufacturers "
        f"{dict(zip(*np.unique(mfr, return_counts=True)))}")
    OUT.mkdir(exist_ok=True)
    out["k"] = k
    np.savez_compressed(OUT / fname, **out)


def stocked(av_sm, st, frac=0.25):
    """SKU stocked in a cluster: available (sold that month) in >= frac of the cluster's store-months with any
    category sale. 0.25, not the agreed 0.5: sales-based availability misses slow movers, and 0.5 leaves 7-10 of
    60 SKUs per cluster (client line size 2-3 with 2 forced)."""
    rows = np.stack([v for (s, _), v in av_sm.items() if s in st])
    return rows.mean(0) >= frac


def near_dup(mfr, sub, band):
    return ((mfr[:, None] == mfr[None]) & (sub[:, None] == sub[None]) & (np.abs(band[:, None] - band[None]) <= 1)
            & ~np.eye(len(mfr), dtype=bool))


# ------------------------------------------------------------------ latent-class MNL (household classes), numpy
_G = {}


def _logp_g(theta):
    """Log choice probabilities per store-month group (G, J + 1)."""
    from scipy.special import logsumexp
    g = _G
    Jn = g["Jn"]
    b = np.logaddexp(0, theta[Jn])
    V = np.where(g["avo"], np.r_[theta[:Jn], 0.0][None] - b * g["pp"], -np.inf)
    return V - logsumexp(V, 1, keepdims=True)


def _nll(theta, C):
    """C (G, J + 1): class-weighted choice counts per group."""
    g = _G
    Jn = g["Jn"]
    lp = _logp_g(theta)
    P = np.exp(lp)
    n = C.sum(1)
    r = theta[:Jn] - g["a0"] - theta[Jn + 1]                      # ridge toward pooled a0 + free segment level
    f = -np.sum(C * np.where(np.isfinite(lp), lp, 0.0)) + g["lam"] * r @ r / 2
    ga = -(C - n[:, None] * P).sum(0)[:Jn] + g["lam"] * r
    gb = ((C * g["pp"]).sum() - (n[:, None] * P * g["pp"]).sum()) / (1 + np.exp(-theta[Jn]))
    return f, np.r_[ga, gb, -g["lam"] * r.sum()]


def lc_fit(y, av, p, hh, K, seed, iters=300, lam=0.01, tol=1e-7, logf=log, a0=None):
    """EM. Class k: V_j = a_kj - b_k * price_nj on available SKUs, outside (index J) = 0, b_k = softplus(beta_k) >= 0.
    Ridge lam * |a_k - a0 - c_k|^2 / 2 per class, c_k a free segment level and a0 the pooled all-store MNL (2026-10-20
    fix: with a0 = 0, SKUs never available in a cluster sat at the outside option's level, ~60x too attractive). Occasions in the same store-month share the choice
    set and prices, so the M-step runs on class-weighted counts per group. Returns dict(a (K, J), b (K,), w (K,), ll)."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    rng = np.random.default_rng(seed)
    N, Jn = av.shape
    H = hh.max() + 1
    key, grp = np.unique(np.c_[p.astype(np.float64), av], axis=0, return_inverse=True)
    grp = grp.ravel()
    G = len(key)
    _G.update(Jn=Jn, lam=lam, a0=np.zeros(Jn) if a0 is None else np.asarray(a0), pp=np.c_[key[:, :Jn], np.zeros(G)], avo=np.c_[key[:, Jn:] > 0.5, np.ones(G, bool)])
    cell = grp * (Jn + 1) + y
    R = rng.dirichlet(np.ones(K), H)
    cnt = np.bincount(y, minlength=Jn + 1)
    base = np.log(np.maximum(cnt[:Jn], 0.5) / max(cnt[Jn], 1))
    TH = np.stack([np.r_[base + rng.normal(0, 0.5, Jn), 0.0, np.mean(base - _G["a0"])] for _ in range(K)])
    prev = -np.inf
    for it in range(iters):
        L = np.zeros((H, K))
        for k in range(K):
            C = np.bincount(cell, R[hh, k], minlength=G * (Jn + 1)).reshape(G, Jn + 1)
            TH[k] = minimize(_nll, TH[k], args=(C,), jac=True, method="L-BFGS-B", options=dict(maxiter=50)).x
            L[:, k] = np.bincount(hh, _logp_g(TH[k])[grp, y], minlength=H)
        w = R.mean(0)
        Lw = L + np.log(np.maximum(w, 1e-300))
        ll = logsumexp(Lw, 1).sum()
        R = np.exp(Lw - logsumexp(Lw, 1, keepdims=True))
        if it % 20 == 0:
            logf(f"  K={K} it {it} ll {ll:.1f} w min {w.min():.3f}")
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return dict(a=TH[:, :Jn], b=np.logaddexp(0, TH[:, Jn]), w=R.mean(0), ll=ll, iters=it + 1, groups=G)


def pooled_a0():
    """Pooled one-class MNL over all 65 clustered stores (k = 2 data, both clusters): SKU constants a0 (J,).
    Cached in out/pooled.json. Every SKU is available somewhere, so every a0_j is identified."""
    f = OUT / "pooled.json"
    if not f.exists():
        d = np.load(OUT / "prep.npz", allow_pickle=True)
        parts = [cluster_data(d, c) for c in range(int(d["k"]))]
        off = np.cumsum([0] + [p[3].max() + 1 for p in parts[:-1]])
        y, av, p = (np.concatenate([q[i] for q in parts]) for i in range(3))
        hh = np.concatenate([q[3] + o for q, o in zip(parts, off)])
        m = lc_fit(y, av, p, hh, 1, SEED)
        f.write_text(json.dumps(dict(a0=m["a"][0].tolist(), b0=float(m["b"][0]), ll=m["ll"]), indent=1))
        log(f"pooled MNL: b0 {m['b'][0]:.3f}, a0 range {m['a'][0].min():.2f} to {m['a'][0].max():.2f}")
    return np.array(json.loads(f.read_text())["a0"])


def lc_holdout_ll(m, y, av, p, hh):
    from scipy.special import logsumexp
    N, Jn = av.shape
    avo = np.c_[av, np.ones(N, bool)]
    pp = np.c_[p, np.zeros(N, np.float32)]
    H = hh.max() + 1
    L = []
    for a, b in zip(m["a"], m["b"]):
        V = np.where(avo, np.r_[a, 0.0][None] - b * pp, -np.inf)
        lp = V - logsumexp(V, 1, keepdims=True)
        L.append(np.bincount(hh, lp[np.arange(N), y], minlength=H))
    return float(logsumexp(np.stack(L, 1) + np.log(m["w"]), 1).sum())


def merge_small(m, floor=0.03):
    """Merge each segment under `floor` weight into the nearest (utility distance) larger one; weighted averages."""
    a, b, w = m["a"].copy(), m["b"].copy(), m["w"].copy()
    keep = list(range(len(w)))
    while True:
        small = [k for k in keep if w[k] < floor]
        if not small:
            break
        k = min(small, key=lambda q: w[q])
        rest = [q for q in keep if q != k]
        V = lambda q: a[q] - b[q] * 0  # noqa: E731  (constants dominate; price folded at fit prices)
        tgt = min(rest, key=lambda q: np.sum((V(q) - V(k)) ** 2))
        tw = w[tgt] + w[k]
        a[tgt] = (w[tgt] * a[tgt] + w[k] * a[k]) / tw
        b[tgt] = (w[tgt] * b[tgt] + w[k] * b[k]) / tw
        w[tgt] = tw
        keep = rest
    return dict(a=a[keep], b=b[keep], w=w[keep], merged=len(w) - len(keep))


def cluster_data(d, c):
    return d[f"c{c}_y"], d[f"c{c}_av"], d[f"c{c}_p"], d[f"c{c}_hh"]


def split_hh(hh, frac=0.2, seed=SEED):
    H = hh.max() + 1
    test = np.random.default_rng(seed).random(H) < frac
    return test[hh]


def lc_best(y, av, p, hh, K, starts=5, logf=log, a0=None):
    """Best of `starts` EM runs by log-likelihood."""
    fits = [lc_fit(y, av, p, hh, K, SEED + s, logf=lambda m: None, a0=a0) for s in range(starts)]
    logf(f"  K={K}: start ll {[round(f['ll'], 1) for f in fits]}")
    return max(fits, key=lambda f: f["ll"])


# ------------------------------------------------------------------ truth Sigma (moment match on switching)
def switching(y, hh, week, Jn):
    """Observed pair switching rates: share of a household's consecutive category purchases going i -> j or j -> i
    (i != j), over all such switches."""
    o = np.lexsort((week, hh))
    y, hh = y[o], hh[o]
    ins = y < Jn
    y, hh = y[ins], hh[ins]
    same = hh[1:] == hh[:-1]
    a, b = y[:-1][same], y[1:][same]
    sw = a != b
    M = np.zeros((Jn, Jn))
    np.add.at(M, (a[sw], b[sw]), 1)
    M = M + M.T
    return M / M[np.triu_indices(Jn, 1)].sum()


def sigma_of(wts, S, nd):
    s0, rest = 1.0, np.asarray(wts)
    raw = s0 * np.eye(S.shape[1]) + np.tensordot(rest, S, 1) + ND_EXTRA * nd
    ev, Q = np.linalg.eigh(raw)                          # near-dup term is not PSD on its own: clip to PD
    if ev.min() < 1e-3:
        raw = (Q * np.maximum(ev, 1e-3)) @ Q.T
    dg = np.sqrt(np.diag(raw))
    return raw / dg[:, None] / dg[None]


def indicator_mats(mfr, sub, pl, band):
    eq = lambda v: (v[:, None] == v[None]).astype(float)  # noqa: E731
    return np.stack([eq(sub), eq(mfr), np.outer(pl, pl).astype(float), eq(band)])   # shared factors (PSD)


def truth(c_list=(0, 1), n_draw=20000, top=200):
    from scipy.optimize import minimize
    d = np.load(OUT / "prep.npz", allow_pickle=True)
    mfr, sub, pl, band, price = d["mfr"], d["sub"], d["pl"], d["band"], d["price"]
    S = indicator_mats(mfr, sub, pl, band)
    nd = near_dup(mfr, sub, band).astype(float)
    iu = np.triu_indices(J, 1)
    for c in c_list:
        lc = json.loads((OUT / f"lc_c{c}.json").read_text())
        a, b, w = np.array(lc["a"]), np.array(lc["b"]), np.array(lc["w"])
        V = a - b[:, None] * price[None]                       # (K, J) logit scale at average prices
        mu = SCALE * V
        obs = switching(d[f"c{c}_y"], d[f"c{c}_hh"], d[f"c{c}_week"], J)
        pairs = np.argsort(-obs[iu])[:top]
        pi, pj = iu[0][pairs], iu[1][pairs]
        z = np.random.default_rng(SEED + c).standard_normal((n_draw, J + 1))

        def pred(wts, tau=0.05):
            Sg = sigma_of(np.exp(wts), S, nd)
            Lc = np.linalg.cholesky(Sg)
            e = np.c_[z[:, :J] @ Lc.T, z[:, J]]                 # outside: own unit-variance error
            P = np.zeros((len(w), J))
            for k in range(len(w)):
                U = np.r_[mu[k], 0.0][None] + e
                U = U[:, :J]                                       # conditional on buying in the category
                Q = np.exp((U - U.max(1, keepdims=True)) / tau)
                P[k] = (Q / Q.sum(1, keepdims=True)).mean(0)
            M = np.einsum("k,ki,kj->ij", w, P, P)
            return M / M[iu].sum()

        def loss(wts):
            M = pred(wts)
            return float(np.sum((M[pi, pj] - obs[pi, pj]) ** 2) / np.sum(obs[pi, pj] ** 2))

        x0 = np.log(np.full(4, 0.3))
        base = loss(np.full(4, -20.0))
        r = minimize(loss, x0, method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-3, fatol=1e-6))
        wts = np.exp(r.x)
        Sg = sigma_of(wts, S, nd)
        Sig = np.eye(J + 1)
        Sig[:J, :J] = Sg                                        # outside option: own error, uncorrelated
        off = np.abs(Sg[iu])
        rep = dict(cluster=c, weights=dict(zip(["sub", "mfr", "pl", "size"], wts.round(4).tolist())), s0=1.0,
                   near_dup_extra=ND_EXTRA, loss_identity=base, loss_fit=r.fun,
                   share_pairs_rho_gt_0p3=float((off > 0.3).mean()), n_near_dup_pairs=int(nd[iu].sum()),
                   rho_min=float(Sg[iu].min()), rho_median=float(np.median(Sg[iu])), rho_max=float(Sg[iu].max()),
                   cond=float(np.linalg.cond(Sg)), K=len(w), w=w.round(4).tolist())
        log(f"cluster {c}: {json.dumps(rep)}")
        np.savez_compressed(OUT / f"truth_c{c}.npz", mu=mu, Sigma=Sig, w=w, V_logit=V, a=a, b=b)
        (OUT / f"truth_c{c}.json").write_text(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ truth Sigma by probit likelihood (user 2026-10-20)
FACTORS = ("near_dup", "mfr", "sub")
STRESS = dict(near_dup=0.5, mfr=0.25)


def sigma3(wts, d):
    """Unit-diagonal Sigma over the J SKUs: I + w_sub S_sub + w_mfr S_mfr + w_nd ND (ND off-diagonal), clipped to
    PD, rescaled to unit diagonal. wts in FACTORS order."""
    eq = lambda v: (v[:, None] == v[None]).astype(float)  # noqa: E731
    nd = near_dup(d["mfr"], d["sub"], d["band"]).astype(float)
    raw = np.eye(J) + wts[2] * eq(d["sub"]) + wts[1] * eq(d["mfr"]) + wts[0] * nd
    ev, Q = np.linalg.eigh(raw)
    if ev.min() < 1e-3:
        raw = (Q * np.maximum(ev, 1e-3)) @ Q.T
    dg = np.sqrt(np.diag(raw))
    return raw / dg[:, None] / dg[None]


def sigma_stress(d):
    """Assumed: rho 0.25 within manufacturer, 0.5 within near-duplicate pairs (near-dups share a manufacturer)."""
    same = d["mfr"][:, None] == d["mfr"][None]
    nd = near_dup(d["mfr"], d["sub"], d["band"])
    R = np.where(nd, STRESS["near_dup"], np.where(same, STRESS["mfr"], 0.0))
    np.fill_diagonal(R, 1.0)
    ev, Q = np.linalg.eigh(R)
    if ev.min() < 1e-3:
        R = (Q * np.maximum(ev, 1e-3)) @ Q.T
        dg = np.sqrt(np.diag(R))
        R = R / dg[:, None] / dg[None]
    return R


class ProbitLik:
    """Household-panel log-likelihood of the real occasions under MNP with the LC utilities fixed (scaled to unit
    error variance) and Sigma = sigma3(wts). Identification: the choice set varies by store-month. Each distinct
    (store-month group, chosen alternative) cell is one orthant per segment, by GHK with Genz ordering and R fixed
    antithetic draws (common across evaluations, so the likelihood is smooth in wts)."""

    def __init__(self, d, c, lc, R=200):
        import ga
        self.ga, self.d = ga, d
        y, av, p, hh = cluster_data(d, c)
        a, b, self.w = np.array(lc["a"]), np.array(lc["b"]), np.array(lc["w"])
        key, grp = np.unique(np.c_[p.astype(np.float64), av], axis=0, return_inverse=True)
        grp = grp.ravel()
        cells, cell_of = np.unique(grp * (J + 1) + y, return_inverse=True)
        self.cell_of, self.hh, self.H = cell_of.ravel(), hh, hh.max() + 1
        self.cg, self.cj = cells // (J + 1), cells % (J + 1)
        self.key, self.a, self.b = key, a, b
        self.U = {}
        self.R = R
        self.byd = {}
        for i, (g, j) in enumerate(zip(self.cg, self.cj)):
            n = int((key[g, J:] > 0.5).sum())                   # alternatives = n SKUs + outside; dim n
            self.byd.setdefault(n, []).append(i)
        log(f"cluster {c}: {len(key)} store-month groups, {len(cells)} (group, choice) cells, dims "
            f"{min(self.byd)}-{max(self.byd)}, K = {len(self.w)}")

    def cell_logp(self, Sig):
        ga = self.ga
        K = len(self.w)
        Sfull = np.eye(J + 1)
        Sfull[:J, :J] = Sig
        out = np.zeros((len(self.cg), K))
        for n, idx_all in self.byd.items():
            for lo in range(0, len(idx_all), 200):
                idx = np.array(idx_all[lo:lo + 200])
                if n not in self.U:
                    self.U[n] = ga.antithetic(self.R, n, 4242 + n)
                A, Sx = [], []
                for i in idx:
                    g, j = self.cg[i], self.cj[i]
                    alts = np.r_[np.flatnonzero(self.key[g, J:] > 0.5), J]
                    mu = SCALE * np.c_[self.a - self.b[:, None] * self.key[g, :J][None], np.zeros(K)][:, alts]
                    pos = int(np.flatnonzero(alts == j)[0])
                    oth = np.r_[np.arange(pos), np.arange(pos + 1, len(alts))]
                    A.append(mu[:, pos, None] - mu[:, oth])                              # (K, n)
                    SA = Sfull[np.ix_(alts, alts)]
                    Sx.append(SA[np.ix_(oth, oth)] - SA[oth, pos][:, None] - SA[oth, pos][None] + SA[pos, pos])
                A = np.stack(A)                                                         # (c, K, n)
                Sx = np.repeat(np.stack(Sx)[:, None], K, 1)                             # (c, K, n, n)
                a2, L = ga.genz_chol_np(A.reshape(-1, n), Sx.reshape(-1, n, n))
                out[idx] = np.log(np.maximum(ga.ghk_np(a2, L, self.U[n]), 1e-300)).reshape(len(idx), K)
        return out

    def ll(self, Sig):
        from scipy.special import logsumexp
        lp = self.cell_logp(Sig)[self.cell_of]                                      # (N, K)
        L = np.stack([np.bincount(self.hh, lp[:, k], minlength=self.H) for k in range(lp.shape[1])], 1)
        return float(logsumexp(L + np.log(self.w), 1).sum())


def sigma_mle(c, k=2):
    from scipy.optimize import minimize
    d = np.load(OUT / PREP[k], allow_pickle=True)
    lc = json.loads((OUT / f"lc_{PFX[k]}c{c}.json").read_text())
    PL = ProbitLik(d, c, lc, R=100)
    t0 = time.time()
    ll_I = PL.ll(np.eye(J))
    log(f"cluster {c}: LL at Sigma = I {ll_I:.1f} ({time.time() - t0:.1f} s per evaluation)")
    hist = []

    def f(x):
        v = PL.ll(sigma3(np.exp(x), d))
        hist.append((np.exp(x).round(4).tolist(), round(v, 2)))
        return -v

    r = minimize(f, np.log([0.3, 0.3, 0.3]), method="Nelder-Mead",
                 options=dict(maxiter=120, xatol=0.02, fatol=0.5, initial_simplex=None))
    wts = np.exp(r.x)
    # curvature check: LL change for +-50% on each weight
    prof = {}
    for i, nm in enumerate(FACTORS):
        for m in (0.0, 0.5, 2.0):
            ww = wts.copy()
            ww[i] *= m
            prof[f"{nm}x{m}"] = round(PL.ll(sigma3(ww, d)) + r.fun, 2)
    rep = dict(cluster=c, factors=FACTORS, weights=wts.round(4).tolist(), ll=-r.fun, ll_identity=ll_I,
               lr_vs_identity=2 * (-r.fun - ll_I), evals=len(hist), profile_dLL=prof)
    log(f"cluster {c}: {json.dumps(rep)}")
    (OUT / f"sigma_mle_{PFX[k]}c{c}.json").write_text(json.dumps(dict(rep, history=hist), indent=1))
    return wts


def sigma_mle_modal(k, clusters):
    """sigma_mle for each cluster in parallel on 2-core Modal CPU containers (local box is busy / 2 cores)."""
    import modal
    files = [OUT / PREP[k]] + [OUT / f"lc_{PFX[k]}c{c}.json" for c in clusters]
    img = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scipy")
           .add_local_file(__file__, "/root/T25/t25.py").add_local_file(HERE / "ga.py", "/root/T25/ga.py"))
    for f in files:
        img = img.add_local_file(f, f"/root/T25/out/{f.name}")
    app = modal.App("orthant-rfc-t25-sigma", image=img)

    @app.function(cpu=2.0, memory=8192, timeout=4 * 3600, serialized=True)
    def run(c):
        import sys as _s
        _s.path.insert(0, "/root/T25")
        import t25 as T
        T.sigma_mle(c, k)
        return (T.OUT / f"sigma_mle_{T.PFX[k]}c{c}.json").read_text()

    with modal.enable_output(), app.run():
        for c, txt in zip(clusters, run.map(clusters)):
            (OUT / f"sigma_mle_{PFX[k]}c{c}.json").write_text(txt)
            log(f"k={k} cluster {c}: {json.loads(txt)['weights']} LR vs I {json.loads(txt)['lr_vs_identity']:.2f}")


def truth2(c, k=2):
    """Writes truth_{est,stress}_c{c}.npz/json: mu (K, J) unit-error-variance utilities at average prices, Sigma
    (J + 1, J + 1) with the outside option's own uncorrelated error."""
    d = np.load(OUT / PREP[k], allow_pickle=True)
    lc = json.loads((OUT / f"lc_{PFX[k]}c{c}.json").read_text())
    a, b, w = np.array(lc["a"]), np.array(lc["b"]), np.array(lc["w"])
    V = a - b[:, None] * d["price"][None]
    wts = np.array(json.loads((OUT / f"sigma_mle_{PFX[k]}c{c}.json").read_text())["weights"])
    nd = near_dup(d["mfr"], d["sub"], d["band"])
    iu = np.triu_indices(J, 1)
    for tag, Sg in (("est", sigma3(wts, d)), ("stress", sigma_stress(d))):
        Sig = np.eye(J + 1)
        Sig[:J, :J] = Sg
        off = np.abs(Sg[iu])
        rep = dict(truth=tag, cluster=c, label="estimated" if tag == "est" else "assumed",
                   weights=dict(zip(FACTORS, wts.round(4).tolist())) if tag == "est" else STRESS,
                   share_pairs_rho_gt_0p3=float((off > 0.3).mean()), n_near_dup_pairs=int(nd[iu].sum()),
                   rho_near_dup_median=float(np.median(Sg[nd])), rho_median=float(np.median(Sg[iu])),
                   rho_max=float(Sg[iu].max()), cond=float(np.linalg.cond(Sg)), K=len(w),
                   w=np.round(w, 4).tolist())
        log(json.dumps(rep))
        np.savez_compressed(OUT / f"truth_{tag}_{PFX[k]}c{c}.npz", mu=SCALE * V, Sigma=Sig, w=w, V_logit=V, a=a, b=b)
        (OUT / f"truth_{tag}_{PFX[k]}c{c}.json").write_text(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ RFC attribute-error tuning (real occasions) + bundle
RFC_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
M_SHELF, MAX_MFR, MIN_PL = 20, 6, 2
SEEDS = (1, 2)                                   # user 2026-10-20: 2 seeds so both truths fit the GPU cap


def attr_levels(d):
    ids = lambda v: np.unique(v, return_inverse=True)[1].ravel()  # noqa: E731
    return [ids(d["sub"]), ids(d["mfr"]), d["pl"].astype(np.int64), d["band"].astype(np.int64)]


def rfc_tune(d, c, lc, R=1000):
    """Attribute-error sd for arms D/E by log-likelihood of the real occasions (prior segment mixture, RFC shares
    floored at 1 / 2R), grid RFC_GRID. Same draws for every grid value; not tuned on the truth."""
    y, av, p, hh = cluster_data(d, c)
    a, b, w = np.array(lc["a"]), np.array(lc["b"]), np.array(lc["w"])
    key, grp = np.unique(np.c_[p.astype(np.float64), av], axis=0, return_inverse=True)
    grp = grp.ravel()
    G, K = len(key), len(w)
    rng = np.random.default_rng(SEED + 100 + c)
    gum = rng.gumbel(size=(K, R, J + 1)).astype(np.float32)
    lev = attr_levels(d)
    eta = [rng.standard_normal((K, R, l.max() + 1)).astype(np.float32) for l in lev]
    attr = sum(e[:, :, l] for e, l in zip(eta, lev))                   # (K, R, J) unit-sd attribute error
    cnt = np.zeros((G, J + 1))
    np.add.at(cnt, (grp, y), 1)
    res = {}
    for sd in RFC_GRID:
        ll = 0.0
        for g in range(G):
            V = a - b[:, None] * key[g, :J][None]
            U = np.concatenate([V[:, None, :] + sd * attr, np.zeros((K, R, 1), np.float32)], 2) + gum
            U[:, :, :J][:, :, key[g, J:] < 0.5] = -np.inf
            ch = U.argmax(2)
            P = np.stack([np.bincount(ch[k], minlength=J + 1) for k in range(K)]) / R
            mix = np.maximum(w @ P, 1 / (2 * R))
            ll += cnt[g] @ np.log(mix)
        res[sd] = float(ll)
    best = max(res, key=res.get)
    log(f"cluster {c}: RFC attribute sd LL {json.dumps({str(k): round(v, 1) for k, v in res.items()})} -> {best}")
    return best, res


def client_info(d):
    """Client SKUs (manufacturer 103) ordered by purchase occasions, top first."""
    cl = np.flatnonzero(d["mfr"] == CLIENT)
    return cl[np.argsort(-d["occ"][cl], kind="stable")]


def cluster_parts(d, k, c, tag):
    lc = json.loads((OUT / f"lc_{PFX[k]}c{c}.json").read_text())
    f_rfc = OUT / f"rfc_sigma_{PFX[k]}c{c}.json"
    if not f_rfc.exists():
        sd, sd_ll = rfc_tune(d, c, lc)
        f_rfc.write_text(json.dumps(dict(sd=sd, ll=sd_ll), indent=1))
    tr = np.load(OUT / f"truth_{tag}_{PFX[k]}c{c}.npz")
    st = d[f"c{c}_stocked"]
    comp = np.flatnonzero(st & (d["mfr"] != CLIENT))
    return dict(V=tr["V_logit"], mu=tr["mu"], Sigma=tr["Sigma"], w=tr["w"],
                rfc_sigma=json.loads(f_rfc.read_text())["sd"], comp=comp,
                n_client_stocked=int((st & (d["mfr"] == CLIENT)).sum()), trips=int(d[f"c{c}_trips"]))


def bundle():
    """out/bundle_d1_{truth}_c{c}.npz (decision 1, one per k = 2 cluster) and out/bundle_d2_{truth}.npz (decision 2,
    k = 5 clusters). Truth-independent parts (logit utilities, RFC sd, initial populations) are identical across the
    truths; arms A, D, E read the est bundle."""
    import ga
    for k in (2, 5):
        d = np.load(OUT / PREP[k], allow_pickle=True)
        margin = np.where(d["pl"], 0.40, 0.30) * d["price"]
        client = client_info(d)
        base = dict(margin=margin, client=client, must_idx=np.array([0, 1]), attr_levels=np.stack(attr_levels(d)),
                    name=d["name"])
        for tag in ("est", "stress"):
            if not all((OUT / f"truth_{tag}_{PFX[k]}c{c}.npz").exists() for c in range(int(d["k"]))):
                ready = [c for c in range(int(d["k"])) if (OUT / f"truth_{tag}_{PFX[k]}c{c}.npz").exists()]
                if k == 5 or not ready:
                    log(f"k={k} {tag}: truth files missing, skipped")
                    continue
            else:
                ready = list(range(int(d["k"])))
            parts = {c: cluster_parts(d, k, c, tag) for c in ready}
            if k == 2:
                specs = [(f"d1_{tag}_c{c}", [p], dict(max_build=len(client), line_min=round(p["n_client_stocked"] / 2 + 1e-9),
                                                      line_max=round(p["n_client_stocked"] / 2 + 1e-9),
                                                      must_in_lines=True)) for c, p in parts.items()]
            else:
                specs = [(f"d2_{tag}", list(parts.values()), dict(max_build=10, line_min=1, line_max=6, must_in_lines=False))]
            for name, ps, cons in specs:
                trips = np.array([p["trips"] for p in ps], float)
                b = dict(base, n_clusters=len(ps), tau=trips / trips.sum(), **cons)
                for i, p in enumerate(ps):
                    for key in ("V", "mu", "Sigma", "w", "rfc_sigma", "comp"):
                        b[f"{key}_{i}"] = p[key]
                    b[f"trips_{i}"] = p["trips"]
                prob = ga.Problem(b)
                for s in SEEDS:
                    b[f"init_{s}"] = prob.initial_population([p["V"] for p in ps], [p["w"] for p in ps], s)
                    assert all(prob.feasible(g) for g in b[f"init_{s}"]), name
                np.savez_compressed(OUT / f"bundle_{name}.npz", **b)
                log(f"{name}: clusters {len(ps)}, competitors {[len(p['comp']) for p in ps]}, client stocked "
                    f"{[p['n_client_stocked'] for p in ps]}, line {cons['line_min']}-{cons['line_max']}, build <= "
                    f"{cons['max_build']}, tau {np.round(b['tau'], 3).tolist()}, RFC sd "
                    f"{[p['rfc_sigma'] for p in ps]}; init distinct "
                    f"{[len({g.tobytes() for g in b[f'init_{s}']}) for s in SEEDS]}")

def check():
    """Sanity check before any decision run: per cluster, the segment-weighted logit utility of client SKUs that are
    rarely or never available there (< 5% of occasions) vs those that are, and the model's category and client
    shares on the stocked shelf vs the observed shares."""
    for k in (2, 5):
        d = np.load(OUT / PREP[k], allow_pickle=True)
        cl = d["mfr"] == CLIENT
        for c in range(int(d["k"])):
            lc = json.loads((OUT / f"lc_{PFX[k]}c{c}.json").read_text())
            a, b, w = np.array(lc["a"]), np.array(lc["b"]), np.array(lc["w"])
            V = w @ (a - b[:, None] * d["price"][None])
            seen = d[f"c{c}_av"].mean(0) >= 0.05
            st = d[f"c{c}_stocked"]
            e = np.exp((a - b[:, None] * d["price"][None])[:, st])
            P = e / (1 + e.sum(1, keepdims=True))
            y = d[f"c{c}_y"]
            log(f"k={k} c{c}: client V seen median {np.median(V[cl & seen]):.2f} (n {int((cl & seen).sum())}), "
                f"unseen median {np.median(V[cl & ~seen]) if (cl & ~seen).any() else float('nan'):.2f} "
                f"(n {int((cl & ~seen).sum())}); stocked shelf model inside {w @ P.sum(1):.2f} client "
                f"{w @ P[:, cl[st]].sum(1):.2f} | observed inside {(y < J).mean():.2f} client {cl[y[y < J]].sum() / len(y):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--c", type=int, default=0)
    ap.add_argument("--k", type=int, default=2, help="2: decision 1 clusters; 5: decision 2 clusters")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.cmd == "prep":
        prep()
        prep(5, STORE_MIN, "prep_k5.npz")
    elif args.cmd == "lc-diag":
        d = np.load(OUT / "prep.npz", allow_pickle=True)
        diag = []
        for c in range(K_CLUSTERS):
            y, av, p, hh = cluster_data(d, c)
            te = split_hh(hh)
            tr = ~te
            for K in (10, 20):
                m = lc_best(y[tr], av[tr], p[tr], np.unique(hh[tr], return_inverse=True)[1], K)
                hl = lc_holdout_ll(m, y[te], av[te], p[te], np.unique(hh[te], return_inverse=True)[1])
                npar = K * (J + 1) + K - 1
                diag.append(dict(cluster=c, K=K, train_ll=round(m["ll"], 1), holdout_ll=round(hl, 1),
                                 holdout_ll_per_occ=round(hl / te.sum(), 4), n_train=int(tr.sum()),
                                 bic_train=round(-2 * m["ll"] + npar * np.log(tr.sum()), 1), params=npar,
                                 w=sorted(np.round(m["w"], 3).tolist(), reverse=True)))
                log(json.dumps(diag[-1]))
        (OUT / "lc_diag.json").write_text(json.dumps(diag, indent=1))
    elif args.cmd == "lc-fit":
        a0 = pooled_a0()
        d = np.load(OUT / PREP[args.k], allow_pickle=True)
        for c in range(int(d["k"])):
            m = lc_best(*cluster_data(d, c), args.K, a0=a0)
            mm = merge_small(m)
            log(f"cluster {c}: K={args.K} ll {m['ll']:.1f}; merged {mm['merged']} segments under 3%; weights "
                f"{np.sort(mm['w'])[::-1].round(3).tolist()}; b {np.round(mm['b'], 3).tolist()}")
            (OUT / f"lc_{PFX[args.k]}c{c}.json").write_text(json.dumps(dict(
                a=mm["a"].tolist(), b=mm["b"].tolist(), w=mm["w"].tolist(), K_fit=args.K, merged=mm["merged"],
                ll=m["ll"], w_raw=m["w"].tolist()), indent=1))
    elif args.cmd == "check":
        check()
    elif args.cmd == "sigma-mle-modal":
        sigma_mle_modal(args.k, list(range(int(np.load(OUT / PREP[args.k])["k"]))))
    elif args.cmd == "sigma-mle":
        sys.path.insert(0, str(HERE))
        sigma_mle(args.c, args.k)
    elif args.cmd == "truth":
        truth2(args.c, args.k)
    elif args.cmd == "bundle":
        sys.path.insert(0, str(HERE))
        bundle()
    else:
        sys.exit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
