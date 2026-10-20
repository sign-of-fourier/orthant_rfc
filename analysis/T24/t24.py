"""T24 · Trivago hotel shelves (RecSys Challenge 2019), a real panel with ~150 hotel properties: mixed MNP with
crossed error components (MVP) vs mixed logit vs MNL. Design: TEST.md.

    python analysis/T24/t24.py prep        # out/arrays.npz from out/clickouts.parquet + data/item_metadata.zip
    python analysis/T24/t24.py local       # pooled MNL (local CPU): start values and baseline
    python analysis/T24/t24_modal.py ...    # mixed logit and MVP on Modal GPU

One choice per session: the session's last clickout (the competition's target). The shelf is the displayed list
(up to 25 hotels), the outcome is the clicked hotel. Panel: users with >= 2 such sessions (last 8 kept). Split by
user: TRAIN_USERS fit, HOLD_USERS are scored with population probabilities. Clickouts parsed in streaming batches
(the box has 4 GB; full-shelf pandas OOMs it).
"""
import collections
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 24
JMAX = 25
TMAX = 8
TRAIN_USERS, HOLD_USERS = 20_000, 5_000
STARS = {"1 Star": 1, "2 Star": 2, "3 Star": 3, "4 Star": 4, "5 Star": 5}
FROM_STARS = {"From 2 Stars": 2, "From 3 Stars": 3, "From 4 Stars": 4}
RATING = {"Satisfactory Rating": 1, "Good Rating": 2, "Very Good Rating": 3, "Excellent Rating": 4}
TYPE = {"Hotel": 0, "House / Apartment": 1}                                 # everything else: 2
UNIONS = {"Free WiFi (Combined)", "Swimming Pool (Combined Filter)"}         # unions of other flags
AMEN_MIN = 0.10                                                              # amenity flags on >= 10% of hotels
N_CLUSTER = 8
BASE = ["lp_rel", "log_pos", "pos1", "star", "star_na", "rating", "rating_na", "house_apt", "other_type", "thin"]
FIXED = ["log_pos", "pos1"]                                                  # position controls: no taste sd
BANDS = ["type", "star", "rating", "price_band", "amen_cluster"]             # shared error components (+ iid)


def _panel_rows():
    """Row numbers of each kept session's last clickout, users with >= 2 sessions, last TMAX sessions each."""
    d = pq.read_table(OUT / "clickouts.parquet", columns=["user_id", "session_id", "timestamp", "step"]).to_pandas()
    d["timestamp"] = d.timestamp.astype(np.int64)
    d["step"] = d.step.astype(np.int32)
    d["row"] = np.arange(len(d))
    last = d.sort_values(["session_id", "timestamp", "step"]).groupby("session_id").tail(1)
    last = last.sort_values(["user_id", "timestamp"]).groupby("user_id").tail(TMAX)
    n = last.groupby("user_id").size()
    return last[last.user_id.isin(n[n >= 2].index)][["row", "user_id", "timestamp"]]


def _items(needed):
    """Properties of the displayed hotels as bitmasks: dict item_id -> int, property -> bit, property -> hotel share."""
    share = collections.Counter()
    bit = {}
    props = {}
    n = 0
    with zipfile.ZipFile(HERE / "data/item_metadata.zip") as z, z.open(z.namelist()[0]) as f:
        for ch in pd.read_csv(f, chunksize=200_000, dtype=str):
            n += len(ch)
            for i, p in zip(ch.item_id, ch.properties.fillna("")):
                s = p.split("|") if p else []
                share.update(s)
                if i in needed:
                    props[i] = sum(1 << bit.setdefault(x, len(bit)) for x in s)
    return props, bit, {k: v / n for k, v in share.items()}


def prep():
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(SEED)
    keep = _panel_rows()
    users = keep.user_id.unique()
    users = rng.permutation(users)[:TRAIN_USERS + HOLD_USERS]
    keep = keep[keep.user_id.isin(users)]
    want = set(keep.row)
    rows, r0 = [], 0
    for b in pq.ParquetFile(OUT / "clickouts.parquet").iter_batches(
            batch_size=100_000, columns=["reference", "impressions", "prices"]):
        idx = [i for i in range(b.num_rows) if r0 + i in want]
        t = b.take(idx).to_pydict()
        rows += [(r0 + i, ref, imp.split("|"), [float(p) for p in pr.split("|")])
                 for i, ref, imp, pr in zip(idx, t["reference"], t["impressions"], t["prices"])]
        r0 += b.num_rows
    props, bit, share = _items({x for _, _, imp, _ in rows for x in imp})
    skip = set(STARS) | set(FROM_STARS) | set(RATING) | set(TYPE) | UNIONS
    amen = sorted(k for k, v in share.items() if v >= AMEN_MIN and k not in skip)
    feat = BASE + [f"a:{a}" for a in amen]

    # item table: features that do not depend on the shelf
    ids = sorted(props)
    pos_of = {a: k for k, a in enumerate(amen)}
    I = np.zeros((len(ids), len(amen)), np.float32)
    star, rating, typ, thin = (np.zeros(len(ids), np.int16) for _ in range(4))
    has = lambda m, names: [v for k, v in names.items() if m >> bit.get(k, 999) & 1]  # noqa: E731
    for r, i in enumerate(ids):
        m = props[i]
        for a, k in pos_of.items():
            I[r, k] = m >> bit.get(a, 999) & 1
        star[r] = max(has(m, STARS) or has(m, FROM_STARS) or [0])
        rating[r] = max(has(m, RATING) or [0])
        typ[r] = min(has(m, TYPE) or [2])
        thin[r] = bin(m).count("1") <= 3
    cluster = KMeans(N_CLUSTER, n_init=4, random_state=SEED).fit_predict(I).astype(np.int16)
    row_of = {i: r for r, i in enumerate(ids)}

    meta = keep.set_index("row")
    n, P, A = len(rows), len(feat), len(BANDS)
    X = np.zeros((n, JMAX, P), np.float16)
    mask = np.zeros((n, JMAX), bool)
    band = np.full((n, JMAX, A), -1, np.int16)
    y = np.full(n, -1, np.int16)
    uid = np.empty(n, object)
    ts = np.zeros(n, np.int64)
    for c, (row, ref, imp, pr) in enumerate(rows):
        J = len(imp)
        it = np.array([row_of.get(x, -1) for x in imp])
        ok = it >= 0
        lp = np.log(np.clip(pr, 1, None))
        lp_rel = np.clip(lp - np.median(lp), -2, 2)
        pos = np.arange(1, J + 1)
        st, rt, ty = np.where(ok, star[it], 0), np.where(ok, rating[it], 0), np.where(ok, typ[it], 2)
        X[c, :J, :len(BASE)] = np.c_[lp_rel, np.log(pos), pos == 1, np.where(st > 0, st - 3, 0), st == 0,
                                     rt, rt == 0, ty == 1, ty == 2, np.where(ok, thin[it], 1)]
        X[c, :J, len(BASE):] = np.where(ok[:, None], I[it], 0)
        mask[c, :J] = True
        pband = np.minimum((pd.Series(lp_rel).rank(pct=True).values * 3).astype(int), 2) if J > 1 else 0
        band[c, :J] = np.c_[ty, st, rt, pband, np.where(ok, cluster[it], N_CLUSTER)]
        y[c] = imp.index(ref) if ref in imp else -1
        uid[c] = meta.at[row, "user_id"]
        ts[c] = meta.at[row, "timestamp"]
    good = (y >= 0) & (mask.sum(1) >= 2)
    o = np.lexsort((ts[good], uid[good]))
    X, mask, band, y, uid = X[good][o], mask[good][o], band[good][o], y[good][o], uid[good][o]
    _, user = np.unique(uid, return_inverse=True)
    n_ch = np.bincount(user)
    multi = n_ch[user] >= 2                                    # dropping bad rows can leave a user 1 session
    X, mask, band, y, user = X[multi], mask[multi], band[multi], y[multi], np.unique(user[multi], return_inverse=True)[1]
    nu = user.max() + 1
    hold_u = rng.permutation(nu)[:round(nu * HOLD_USERS / (TRAIN_USERS + HOLD_USERS))]
    split = np.isin(user, hold_u).astype(np.int8)              # 0 train, 1 holdout
    np.savez_compressed(OUT / "arrays.npz", X=X, mask=mask, band=band, y=y, user=user, split=split,
                        feat=np.array(feat))
    print(f"{len(y)} choices, {nu} users (train {nu - len(hold_u)}, holdout {len(hold_u)}); P = {P} "
          f"({len(amen)} amenities); hotels per shelf median {np.median(mask.sum(1)):.0f}; "
          f"items without metadata {np.mean([x not in props for _, _, imp, _ in rows for x in imp]):.4f}")
    print("cluster sizes", np.bincount(cluster).tolist(), "click in slot 1", round(float(np.mean(y == 0)), 3))


def load():
    z = np.load(OUT / "arrays.npz")
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------- pooled MNL (local)

def mnl_logp(X, mask, b):
    V = np.where(mask, X @ b, -np.inf)
    return V - logsumexp(V, 1, keepdims=True)


def mnl_fit(X, mask, y):
    n, P = len(y), X.shape[2]

    def f(b):
        lp = mnl_logp(X, mask, b)
        p = np.exp(lp)
        gr = (X[np.arange(n), y] - np.einsum("nj,njp->np", p, X)).sum(0)
        return -lp[np.arange(n), y].sum(), -gr
    return minimize(f, np.zeros(P), jac=True, method="BFGS").x


def local():
    import json
    A = load()
    feat = A["feat"].tolist()
    tr, ho = A["split"] == 0, A["split"] == 1
    b = mnl_fit(A["X"][tr].astype(float), A["mask"][tr], A["y"][tr])
    lp = mnl_logp(A["X"][ho].astype(float), A["mask"][ho], b)[np.arange(ho.sum()), A["y"][ho]]
    json.dump(dict(mnl=b.tolist(), feat=feat, holdout_logp=lp.tolist()), open(OUT / "local.json", "w"))
    print(f"MNL holdout mean log P per choice {lp.mean():.4f} (n {ho.sum()})")
    print({f: round(v, 3) for f, v in zip(feat, b) if f in BASE})


if __name__ == "__main__":
    {"prep": prep, "local": local}[sys.argv[1]]()
