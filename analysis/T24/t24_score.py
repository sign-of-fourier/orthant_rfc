"""T24 scoring: holdout log P per choice (population, marginal over tastes) and per user (panel), paired
differences with 95% CIs from a bootstrap over holdout users; MVP error shares and the taste sds.

    python analysis/T24/t24_score.py smoke|full
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t24  # noqa: E402

B = 2000


def holdout_index(mode, seed=24):
    A = t24.load()
    idx = np.where(A["split"] == 1)[0]
    if mode == "smoke":
        users = np.sort(np.random.default_rng(seed).choice(np.unique(A["user"][idx]), 500, replace=False))
        idx = idx[np.isin(A["user"][idx], users)]
    return idx, A


def boot_ci(diff_u, n_u, rng):
    """95% CI of sum(diff_u) / sum(n_u), resampling users."""
    k = rng.integers(0, len(diff_u), (B, len(diff_u)))
    s = diff_u[k].sum(1) / n_u[k].sum(1)
    return np.percentile(s, [2.5, 97.5])


def main():
    mode = sys.argv[1]
    idx, A = holdout_index(mode)
    hold = np.flatnonzero(A["split"] == 1)
    loc = json.loads((t24.OUT / "local.json").read_text())
    mnl = np.array(loc["holdout_logp"])[np.searchsorted(hold, idx)]
    r = {m: json.loads((t24.OUT / f"{m}_{mode}.json").read_text()) for m in ("mxl", "mvp")}
    user = np.unique(A["user"][idx], return_inverse=True)[1]
    n_u = np.bincount(user).astype(float)
    per_choice = {"MNL": mnl, "MXL": np.array(r["mxl"]["holdout_logp_choice"]),
                  "MVP": np.array(r["mvp"]["holdout_logp_choice"])}
    by_user = {k: np.bincount(user, v) for k, v in per_choice.items()}
    panel = {"MNL": by_user["MNL"], "MXL": np.array(r["mxl"]["holdout_logp_user"]),
             "MVP": np.array(r["mvp"]["holdout_logp_user"])}
    rng = np.random.default_rng(24)
    print(f"holdout: {len(idx)} choices, {len(n_u)} users")
    for name, tab in (("per choice (population)", by_user), ("per user (panel), per choice", panel)):
        print(f"\n{name}: " + ", ".join(f"{k} {v.sum() / n_u.sum():.4f}" for k, v in tab.items()))
        for a, b in (("MVP", "MXL"), ("MVP", "MNL"), ("MXL", "MNL")):
            d = tab[a] - tab[b]
            lo, hi = boot_ci(d, n_u, rng)
            print(f"  {a} - {b}: {d.sum() / n_u.sum():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    feat = A["feat"].tolist()
    P = len(feat)
    rnd = [f for f in feat if f not in t24.FIXED]
    nr = len(rnd)
    th = np.array(r["mvp"]["theta"])
    u = np.r_[th[P + nr:], 0.0]
    p = np.exp(u - u.max())
    p /= p.sum()
    print("\nMVP error shares:", dict(zip(t24.BANDS + ["iid"], np.round(p, 3))))
    for m in ("mxl", "mvp"):
        th = np.array(r[m]["theta"])
        sd = dict(zip(rnd, np.exp(th[P:P + nr])))
        top = sorted(sd.items(), key=lambda kv: -kv[1])[:8]
        print(f"{m} taste sd, top 8:", {k: round(v, 2) for k, v in top},
              f"| fit {r[m]['seconds_fit']:.0f} s, {r[m]['n_iter']} iters, grad {r[m]['grad_max']:.1e}")
    print(f"\nuniform guess per choice: {np.mean(-np.log(A['mask'][idx].sum(1))):.4f}; e.g. "
          f"0.01 nats/choice = {0.01 * n_u.mean():.3f} nats/user; ln 2 = {math.log(2):.3f}")


if __name__ == "__main__":
    main()
