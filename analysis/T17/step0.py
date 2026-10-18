"""T17 Step 0: count twin episodes in the holdout weeks (power gate; analysis/T17/TEST.md).

    python analysis/T17/step0.py

Writes analysis/T17/out/step0.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "T6"))
import t6  # noqa: E402

TRAIN, HOLD = range(1, 41), range(41, 54)
CARRIED, CUT, M_TWIN = 0.5, 0.15, 1.0


def main():
    t = pyreadr.read_r(str(t6.DATA / "transactions.rds"))[None]
    t = t[t.product_id.isin(t6.ITEMS)].copy()
    prod = pyreadr.read_r(str(t6.DATA / "products.rda"))["products"].set_index("product_id")
    man = prod.loc[t6.ITEMS, "manufacturer_id"].astype(str)
    t["price"] = t.sales_value / t.quantity.clip(lower=1)
    n_sku = t.groupby("basket_id").product_id.transform("nunique")
    ch = t[n_sku == 1].drop_duplicates("basket_id")          # single-SKU trips: one choice each

    sold = t.groupby(["store_id", "week"]).product_id.agg(set)  # availability proxy
    tw = t[t.week.isin(TRAIN)]
    nweeks = tw.groupby("store_id").week.nunique()
    wk = tw.groupby(["store_id", "product_id"]).week.nunique()
    frac = wk / nweeks.reindex(wk.index.get_level_values(0)).to_numpy()
    carried = frac[frac >= CARRIED]
    regular = tw.groupby(["store_id", "product_id"]).price.median()
    price = t.groupby(["store_id", "week", "product_id"]).price.median()

    hold = ch[ch.week.isin(HOLD)]
    eps = []
    for (s, w), g in hold.groupby(["store_id", "week"]):
        avail = sold.get((s, w), set())
        for a in t6.ITEMS:
            if (s, a) not in carried.index:
                continue
            twins = [b for b in t6.ITEMS if b != a and man[b] == man[a] and b in avail]
            if not twins:
                continue
            kind = None
            if a not in avail:
                kind = "missing"
            elif (s, w, a) in price.index and price[(s, w, a)] <= (1 - CUT) * regular[(s, a)]:
                kind = "price_cut"
            if kind:
                n = len(g)
                eps.append(dict(store=int(s), week=int(w), sku=a, man=man[a], kind=kind, n=n,
                                twin_share=float(g.product_id.isin(twins).mean())))
    E = pd.DataFrame(eps)
    out = dict(single_sku_trips=int(len(ch)), holdout_choices=int(len(hold)))
    for kind in ("missing", "price_cut", "all"):
        e = E if kind == "all" else E[E.kind == kind]
        if e.empty:
            out[kind] = dict(episodes=0)
            continue
        s = np.average(e.twin_share, weights=e.n)
        N = e.n.sum()
        se = np.sqrt((e.n * s * (1 - s)).sum()) / N        # SE of the choice-weighted observed twin share
        out[kind] = dict(episodes=int(len(e)), choices=int(N), groups=int(e.sku.nunique()),
                         manufacturers=int(e.man.nunique()), mean_twin_share_pts=100 * s,
                         ci_halfwidth_pts=100 * 1.96 * se, choices_per_episode_median=float(e.n.median()))
    a = out["all"]
    out["gate"] = dict(groups_ok=a.get("groups", 0) >= 3, episodes_ok=a.get("episodes", 0) >= 30,
                       choices_ok=a.get("choices", 0) >= 300, power_ok=a.get("ci_halfwidth_pts", 99) <= M_TWIN)
    out["gate"]["PASS"] = all(out["gate"].values())
    if not E.empty:
        out["per_sku"] = E.groupby(["sku", "kind"]).n.agg(["count", "sum"]).reset_index().to_dict("records")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "step0.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "per_sku"}, indent=1, default=str))


if __name__ == "__main__":
    main()
