"""T17: within-person substitution on the dunnhumby soft-drink shelf (design: analysis/T17/TEST.md).

    python analysis/T17/t17.py prep      # local: build choice occasions -> out/occasions.npz
    python analysis/T17/t17.py power     # Modal CPU: simulated power check (nested-logit truth, MNL vs NL)

Occasion = a trip with exactly one of the 30 SKUs. Price = median unit price paid in that store-week over
positive-price lines. Choice set = SKUs sold in that store within +-4 weeks (user 2026-10-17: the same-week proxy
gave a median shelf of 4); an unsold-but-available SKU takes its price from the nearest sold week in that store.
Utility = SKU constant + beta * log price. Nested logit by manufacturer and MNL are estimated with Biogeme on
Modal (its own image; nothing installed locally). Twin predictions are computed here from the estimates (formulas
checked against Biogeme's log-likelihood). Episodes: price cuts only.
Twin test: rolling time folds; episodes pooled per focal SKU before taking |predicted - observed| (user 2026-10-17).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 20261017
FOLDS = [range(1, 14), range(14, 27), range(27, 41), range(41, 54)]
CARRIED, CUT, M_TWIN_FLOOR, WIN = 0.5, 0.15, 1.0, 4

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("biogeme==3.3.5", "numpy", "pandas"))
app = modal.App("orthant-rfc-t17", image=image)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data (local)
def prep():
    import pandas as pd
    import pyreadr
    sys.path.insert(0, str(HERE.parent / "T6"))
    import t6

    t = pyreadr.read_r(str(t6.DATA / "transactions.rds"))[None]
    t = t[t.product_id.isin(t6.ITEMS)].copy()
    prod = pyreadr.read_r(str(t6.DATA / "products.rda"))["products"].set_index("product_id")
    man = prod.loc[t6.ITEMS, "manufacturer_id"].astype(str).to_numpy()
    col = {p: j for j, p in enumerate(t6.ITEMS)}
    t["j"] = t.product_id.map(col)
    t["price"] = t.sales_value / t.quantity.clip(lower=1)
    stores = np.sort(t.store_id.unique())
    sidx = {s: i for i, s in enumerate(stores)}
    t["s"] = t.store_id.map(sidx)
    S, W, J = len(stores), 53, len(t6.ITEMS)
    price = np.full((S, W + 1, J), np.nan)                     # observed: sold that week at a positive price
    pm = t[(t.sales_value > 0) & (t.quantity > 0)].groupby(["s", "week", "j"]).price.median()
    idx = np.array(pm.index.tolist())
    price[idx[:, 0], idx[:, 1], idx[:, 2]] = pm.to_numpy()
    fill = price.copy()                                         # nearest sold week within +-WIN
    for d in range(1, WIN + 1):
        for sh in (d, -d):
            src = np.full_like(price, np.nan)
            if sh > 0:
                src[:, sh:] = price[:, :-sh]
            else:
                src[:, :sh] = price[:, -sh:]
            fill = np.where(np.isnan(fill), src, fill)
    n_sku = t.groupby("basket_id").j.transform("nunique")
    ch = t[n_sku == 1].drop_duplicates("basket_id")
    ok = ~np.isnan(fill[ch.s.to_numpy(), ch.week.to_numpy(), ch.j.to_numpy()])  # chosen SKU has a positive price
    log(f"prep: dropped {(~ok).sum()} occasions with no positive price for the chosen SKU within +-{WIN} weeks")
    ch = ch[ok]
    s, w = ch.s.to_numpy(), ch.week.to_numpy()
    av = ~np.isnan(fill[s, w])
    hh = ch.household_id.astype("category").cat.codes.to_numpy()
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "occasions.npz", y=ch.j.to_numpy(), av=av, logp=np.log(np.nan_to_num(fill[s, w], nan=1.0)),
                        store=s, week=w, hh=hh, price=price, man=man, items=np.array(t6.ITEMS))
    log(f"prep: {len(ch)} occasions, {hh.max() + 1} households, {S} stores; choice-set size median "
        f"{np.median(av.sum(1)):.0f} (range {av.sum(1).min()}-{av.sum(1).max()})")


# ------------------------------------------------------------------ models
def nests_of(man):
    names = sorted(set(man))
    return names, np.array([names.index(m) for m in man])


def probs(theta, av, logp, nest, kind):
    """P (n, J) for MNL (kind 'mnl') or nested logit with nest scale lam_m in (0, 1]."""
    J = av.shape[1]
    asc = np.r_[0.0, theta[:J - 1]]
    beta = theta[J - 1]
    v = np.where(av, asc + beta * logp, -np.inf)
    if kind == "mnl":
        v = v - v.max(1, keepdims=True)
        e = np.exp(v)
        return e / e.sum(1, keepdims=True)
    lam = theta[J:]
    lj = lam[nest]
    vl = v / lj
    M = len(lam)
    P = np.zeros_like(v)
    iv = np.full((len(v), M), -np.inf)
    for m in range(M):
        cols = nest == m
        x = vl[:, cols]
        mx = np.max(np.where(np.isfinite(x), x, -1e300), 1)
        ok = np.isfinite(x).any(1)
        lse = np.where(ok, mx + np.log(np.exp(x - mx[:, None]).sum(1)), -np.inf)
        iv[:, m] = lam[m] * lse
        within = np.where(ok[:, None], np.exp(x - lse[:, None]), 0.0)
        P[:, cols] = within
    mx = iv.max(1, keepdims=True)
    pn = np.exp(iv - mx)
    pn = pn / pn.sum(1, keepdims=True)
    return P * pn[:, nest]


def fit(y, av, logp, nest, kind):
    """Biogeme MLE; returns theta = [asc_2..asc_J, beta, (lam_1..lam_M)] and Biogeme's final log-likelihood."""
    import pandas as pd
    import biogeme.biogeme as bio
    import biogeme.database as db
    from biogeme import models
    from biogeme.expressions import Beta, Variable
    from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit

    n, J = av.shape
    df = pd.DataFrame({"choice": y + 1, **{f"av{j}": av[:, j].astype(int) for j in range(J)},
                       **{f"lp{j}": np.where(av[:, j], logp[:, j], 0.0) for j in range(J)}})
    d = db.Database("t17", df)
    asc = [None] + [Beta(f"asc{j}", 0, None, None, 0) for j in range(1, J)]
    b = Beta("beta", -1, None, None, 0)
    V = {j + 1: (asc[j] if j else 0) + b * Variable(f"lp{j}") for j in range(J)}
    avd = {j + 1: Variable(f"av{j}") for j in range(J)}
    if kind == "mnl":
        logprob = models.loglogit(V, avd, Variable("choice"))
    else:
        M = nest.max() + 1
        mus = [Beta(f"mu{m}", 2.0, 1.0, 20.0, 0) for m in range(M)]  # Biogeme's mu = 1 / lambda >= 1
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=tuple(
            OneNestForNestedLogit(nest_param=mus[m], list_of_alternatives=[j + 1 for j in range(J) if nest[j] == m],
                                  name=f"n{m}") for m in range(M)))
        logprob = models.lognested(V, avd, nests, Variable("choice"))
    B = bio.BIOGEME(d, logprob, save_iterations=False, generate_html=False, generate_yaml=False)
    B.model_name = f"t17_{kind}"
    r = B.estimate()
    est = r.get_beta_values()
    theta = [est[f"asc{j}"] for j in range(1, J)] + [est["beta"]]
    if kind == "nl":
        theta += [1 / est[f"mu{m}"] for m in range(nest.max() + 1)]
    return np.array(theta), float(r.final_log_likelihood)


def loglik(theta, y, av, logp, nest, kind):
    P = probs(theta, av, logp, nest, kind)
    return float(np.log(P[np.arange(len(y)), y]).sum())


# ------------------------------------------------------------------ twin test
def episodes(D, fold):
    """(occasion, focal sku) pairs in the fold's weeks: focal sold that week >= CUT below its regular price, a twin
    on the shelf. Carried / regular price from the other weeks."""
    price, store, week, man = D["price"], D["store"], D["week"], D["man"]
    tr = np.array([w for w in range(1, 54) if w not in fold])
    sold = ~np.isnan(price)                                   # (S, W+1, J)
    open_w = sold[:, tr].any(2)                               # store had any of the 30 that week
    frac = sold[:, tr].sum(1) / np.maximum(open_w.sum(1), 1)[:, None]
    carried = frac >= CARRIED                                 # (S, J)
    with np.errstate(all="ignore"):
        regular = np.nanmedian(price[:, tr], axis=1)          # (S, J)
    twin = (man[:, None] == man[None, :]) & ~np.eye(len(man), dtype=bool)
    occ = np.flatnonzero(np.isin(week, list(fold)))
    out = []
    for i in occ:
        s, w = store[i], week[i]
        av = D["av"][i]
        for a in np.flatnonzero(carried[s] & sold[s, w]):
            if (twin[a] & av).any() and price[s, w, a] <= (1 - CUT) * regular[s, a]:
                out.append((i, a))
    return np.array(out, dtype=int).reshape(-1, 2), twin


def twin_errors(E, twin, y, P_by_model, av, hh, boot_idx):
    """Per model: choice-weighted mean over focal SKUs of |pred - obs| twin share (pts), plus bootstrap draws."""
    i, a = E[:, 0], E[:, 1]
    tmask = twin[a] & av[i]
    obs = tmask[np.arange(len(i)), y[i]].astype(float)
    preds = {m: (P[i] * tmask).sum(1) for m, P in P_by_model.items()}
    focal = np.unique(a)

    def stat(w):
        res = {}
        for m, pr in preds.items():
            num = den = 0.0
            for f in focal:
                k = a == f
                n = w[k].sum()
                if n > 0:
                    num += abs((w[k] * pr[k]).sum() - (w[k] * obs[k]).sum())  # = n * |pred share - obs share|
                    den += n
            res[m] = 100 * num / den
        return res

    point = stat(np.ones(len(i)))
    draws = []
    for idx in boot_idx:
        cnt = np.bincount(idx, minlength=hh.max() + 1)
        draws.append(stat(cnt[hh[i]].astype(float)))
    return point, draws


@app.function(cpu=2.0, memory=4096, timeout=3 * 3600)
def power_remote(D, B=200, n_sub=0):
    D = {k: v for k, v in D.items()}
    if n_sub:  # smoke: a random subset of occasions
        keep = np.sort(np.random.default_rng(0).choice(len(D["y"]), n_sub, replace=False))
        for k in ("y", "av", "logp", "hh", "week", "store"):
            D[k] = D[k][keep]
    y, av, logp, hh, week = D["y"], D["av"], D["logp"], D["hh"], D["week"]
    names, nest = nests_of(D["man"])
    t0 = time.time()
    th_nl, ll_b = fit(y, av, logp, nest, "nl")
    ll_ours = loglik(th_nl, y, av, logp, nest, "nl")
    th_mnl, llm_b = fit(y, av, logp, nest, "mnl")
    real = dict(lam=dict(zip(names, th_nl[av.shape[1]:].tolist())), beta_nl=float(th_nl[av.shape[1] - 1]),
                beta_mnl=float(th_mnl[-1]), ll_nl=ll_b, ll_mnl=llm_b, ll_nl_ours=ll_ours,
                ll_mnl_ours=loglik(th_mnl, y, av, logp, nest, "mnl"), theta_nl=th_nl.tolist(), theta_mnl=th_mnl.tolist())
    # simulated truth: the real-data nested logit
    rng = np.random.default_rng(SEED)
    P = probs(th_nl, av, logp, nest, "nl")
    ys = (P.cumsum(1) > rng.random((len(y), 1))).argmax(1)
    Pf = {"mnl": np.zeros_like(P), "nl": np.zeros_like(P)}
    E_all = []
    for f in FOLDS:
        te = np.isin(week, list(f))
        for kind in ("mnl", "nl"):
            th, _ = fit(ys[~te], av[~te], logp[~te], nest, kind)
            Pf[kind][te] = probs(th, av[te], logp[te], nest, kind)
        E, twin = episodes(D, f)
        E_all.append(E)
    E = np.concatenate(E_all)
    rb = np.random.default_rng(SEED + 1)
    H = hh.max() + 1
    boot = [rb.integers(0, H, H) for _ in range(B)]
    point, draws = twin_errors(E, twin, ys, Pf, av, hh, boot)
    d = np.array([x["mnl"] - x["nl"] for x in draws])
    delta = point["mnl"] - point["nl"]
    m = max(M_TWIN_FLOOR, 0.15 * point["mnl"])
    lo, hi = delta - 1.96 * d.std(ddof=1), delta + 1.96 * d.std(ddof=1)
    return dict(real=real, sim=dict(episodes=int(len(E)), focal=int(len(np.unique(E[:, 1]))), twin_err=point,
                                    delta=delta, se=float(d.std(ddof=1)), ci=[lo, hi], m_twin=m,
                                    PASS=bool(lo > 0 and delta >= m)),
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "power"])
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--smoke", type=int, default=0, help="subset of occasions (wiring check)")
    a = ap.parse_args()
    if a.stage == "prep":
        return prep()
    D = dict(np.load(OUT / "occasions.npz", allow_pickle=True))
    with modal.enable_output(), app.run():
        r = power_remote.remote(D, a.B, a.smoke)
    (OUT / ("power_smoke.json" if a.smoke else "power.json")).write_text(json.dumps(r, indent=1, default=str))
    log("real fit: " + json.dumps(r["real"], default=str))
    log("power check: " + json.dumps(r["sim"], default=str))


if __name__ == "__main__":
    main()
