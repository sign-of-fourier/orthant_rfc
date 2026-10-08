"""Summarize results/rep_*.json. Twin-share error is measured against the
oracle's expected share (sampling noise in actual shares is reported apart)."""
import glob
import json
import sys

import numpy as np
import pandas as pd

rows = sum((json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/rep_*.json"))), [])
df = pd.DataFrame(rows)
orc = df[df.fix == "oracle"][["seed", "truth", "task", "twin_pred"]].rename(columns={"twin_pred": "twin_oracle"})
df = df.merge(orc, on=["seed", "truth", "task"])
df["err"] = df.twin_pred - df.twin_oracle
df["k_kind"] = df.k.astype(str) + np.where(df.kind == "triplet", "T", "")

print("Twin-share error vs oracle (pred - oracle), mean over tasks and seeds, by k (T = triplet):")
print(df[df.fix != "oracle"].pivot_table(index=["truth", "fix"], columns="k_kind", values="err", aggfunc="mean").round(4))

g = df[df.fix != "oracle"].groupby(["truth", "fix"]).err.apply(lambda e: np.sqrt((e ** 2).mean())).unstack()
print("\nRMS twin-share error vs oracle; gap closed = 1 - rms(fix)/rms(none):")
print(pd.concat([g.round(4), (1 - g.div(g["none"], axis=0)).add_suffix("_gap").round(2)], axis=1))

s = df.groupby(["truth", "fix"]).agg(mae=("mae", "mean"), retest=("retest", "mean"), ll=("ll", "mean"), sec=("sec", "mean"))
s["mae/retest"] = s.mae / s.retest
print("\nMAE vs actual / test-retest MAE, out-of-sample log-lik (eval half), fit seconds:")
print(s.round(4))

th = df[df.fix.isin(["rfc", "kernel"])].drop_duplicates(["seed", "truth", "fix"])
print("\nFitted parameters (mean over seeds):")
print(pd.DataFrame([dict(truth=r.truth, fix=r.fix, **r.th) for r in th.itertuples()]).groupby(["truth", "fix"]).mean().round(3))
