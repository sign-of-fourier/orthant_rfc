import io, json, urllib.request, numpy as np, pandas as pd, rfc
from quantecarlo._cdf_api import build_cdf_body, DEFAULT_CDF_URL
pd.set_option("display.width", 200); pd.set_option("display.precision", 4)

def cdf(up, cov, dup_corr):
    body = build_cdf_body(up, cov)
    z = dict(np.load(io.BytesIO(body), allow_pickle=False))
    z["params"] = json.dumps({"resolution": "high", "dup_corr": dup_corr})
    buf = io.BytesIO(); np.savez(buf, **z)
    req = urllib.request.Request(DEFAULT_CDF_URL, data=buf.getvalue(), headers={"Content-Type": "application/octet-stream"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return np.load(io.BytesIO(r.read()))["p"].astype(float)

z = np.load("orthant_accuracy_ref.npz")
rng = np.random.default_rng(31)
_, kind_s = rfc.make_sets(300, rng)                     # same draw as orthant_accuracy.ipynb
up = np.vstack([z["up_s"], z["up_w"]]); cov = np.vstack([z["cov_s"], z["cov_w"]]); ref = np.r_[z["ref_s"], z["ref_w"]]
group = np.r_[np.repeat(kind_s, 5), np.repeat("sweep", 250)]
scen = np.repeat(np.arange(350), 5)
rowcorr = rfc.max_diff_corr(cov)
scorr = pd.Series(rowcorr).groupby(scen).transform("max").values
bins = pd.cut(scorr, [0, .9, .95, .97, .99, 1.0001], right=False,
              labels=["<0.90", "0.90-0.95", "0.95-0.97", "0.97-0.99", ">=0.99"])
rows = []
for dc in (0.01, 0.03, 0.05, 0.10):
    p = cdf(up, cov, dc)
    p = p / pd.Series(p).groupby(scen).transform("sum").values      # renormalize per scenario
    e = p - ref
    for name, m in [("twin", group == "twin"), ("near", group == "near"), ("plain", group == "plain")] + \
                   [(f"max corr {b}", bins == b) for b in bins.categories]:
        rows.append(dict(fold_at=f"{1-dc:.2f}", group=name, n=m.sum(), bias=e[m].mean(), sd=e[m].std(), max_abs=np.abs(e[m]).max()))
t = pd.DataFrame(rows)
for col in ("sd", "max_abs", "bias"):
    print(col); print(t.pivot(index="group", columns="fold_at", values=col).reindex(t.group.unique()), "\n")
print(t[t.fold_at == "0.99"][["group", "n"]].to_string(index=False))
