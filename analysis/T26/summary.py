"""T26 summary: true regret of each run's pick (final and per tier) vs single-core seconds -> out/results.csv,
out/convergence.png, printed tables."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TAU = None


def main():
    rows = []
    for t in ("moderate", "high"):
        ref = json.loads((OUT / f"reference_{t}.json").read_text())
        tv = {tuple(map(int, k.split(","))): v for k, v in json.loads((OUT / f"truevals_{t}.json").read_text()).items()}
        tau = np.load(OUT / f"bundle_{t}.npz", allow_pickle=True)["tau"]
        best = ref["best"]
        for f in sorted(glob.glob(str(OUT / "runs" / f"{t}_*.json"))):
            r = json.loads(Path(f).read_text())
            for i, tr in enumerate(r["tiers"]):
                v = sum(tau[c] * tv[(c, ln)] for c, ln in enumerate(tr["pick_lines"]))
                rows.append(dict(truth=t, screen=r["screen"], seed=r["seed"], budget=r["budget"], tier=tr["tier"],
                                 tier_idx=i, final=i == len(r["tiers"]) - 1, cum_sec=tr["cum_sec"], scored=tr["scored"],
                                 of=tr["of"], stopped=tr["stopped"], true=v, best=best, regret_usd=best - v,
                                 regret_pct=100 * (best - v) / best, same_build=bool(tr["pick_build"] == ref["best_build"])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    fin = df[df.final]
    print("## Final pick: true regret % (mean [min, max] over 3 seeds), single-core seconds, picks with the optimal build set")
    for (t, s, B), g in fin.groupby(["truth", "screen", "budget"]):
        print(f"{t:8s} {s:3s} B={B:6.0f}s  regret {g.regret_pct.mean():6.3f} [{g.regret_pct.min():.3f}, "
              f"{g.regret_pct.max():.3f}]  ${g.regret_usd.mean():.3f}  sec {g.cum_sec.mean():6.1f}  optimal build {g.same_build.sum()}/3")
    print("\n## Per tier (budgets >= 80 s): regret % of the tier's pick, cumulative seconds")
    big = df[df.budget >= df.groupby("truth").budget.transform("max") / 8]
    print(big.groupby(["truth", "screen", "tier_idx", "tier"]).agg(regret=("regret_pct", "mean"), rmax=("regret_pct", "max"),
                                                              sec=("cum_sec", "mean"), scored=("scored", "mean")).round(3).to_string())
    plot(df)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"C": "#2a78d6", "Fc": "#eb6834", "Fd": "#1baf7a"}
    lab = {"C": "C engine screen (low→high→highest→QMC)", "Fc": "F simulator, collapsed", "Fd": "F simulator, dense"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, t, title in zip(axes, ("moderate", "high"), ("ρ = 0.5 (moderate)", "ρ = 0.8 (high)")):
        d = df[(df.truth == t) & df.final]
        for s in ("C", "Fd", "Fc"):
            g = d[d.screen == s].groupby("budget").agg(x=("cum_sec", "mean"), y=("regret_pct", "mean"),
                                                         lo=("regret_pct", "min"), hi=("regret_pct", "max")).sort_values("x")
            ax.errorbar(g.x, g.y + 1e-3, yerr=[g.y - g.lo, g.hi - g.y], color=col[s], lw=2, marker="o", ms=7,
                        mec="#fcfcfb", mew=1.5, capsize=3, label=lab[s])
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 40, 100, 200])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(title, loc="left", color="#0b0b0b")
        ax.set_xlabel("single-core seconds used (setup excluded)", color="#52514e")
        ax.grid(True, color="#e6e5e0", lw=0.8, which="both")
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors="#52514e")
    axes[0].set_ylabel("true regret of the pick, % of exact optimum", color="#52514e")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("T26: tiered screens on the full span (108,885 shelves), one CPU core, mean and range of 3 seeds",
                 x=0.01, ha="left", color="#0b0b0b")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)


if __name__ == "__main__":
    main()
