"""T25 summary tables, anytime curves, decision-1 time-to-optimum and decision-2 lines / client share /
cannibalization, from out/results.csv, out/runs and out/scores_*.json.

    python analysis/T25/summary.py
Writes out/anytime.png and out/summary_tables.md (the interpretation goes in SUMMARY.md by hand).
Client share and cannibalization are computed here under the truth with GHK (Genz ordering, R = 4,000) in numpy.
Cannibalization of SKU j on line L = (other client SKUs' share without j - with j) / share of j: the fraction of j's
volume taken from the client's own line.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ga

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4", "F": "#008300"}
LABEL = {"A": "A logit (CPU)", "B": "B MNP GHK (CPU, 8 cores)", "C": "C MNP engine (T4)", "D": "D RFC (T4)",
         "E": "E RFC (CPU, 8 cores)", "F": "F MNP simulator (T4)"}
PROBLEMS = {"d1_c0": "Decision 1, cluster 0", "d1_c1": "Decision 1, cluster 1", "d2": "Decision 2 (build + place)"}
TRUTHS = {"est": "estimated truth", "stress": "stress truth (assumed)"}


def mr(x, f="{:.2f}"):
    return f"{f.format(x.mean())} [{f.format(x.min())}, {f.format(x.max())}]"


def tables(df):
    fin = df[df.t == df.budget]
    rows = ["| problem | truth | budget | arm | true $/1k trips | regret % | regret $/1k | evals | gens | GPU s | $ |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (p, tr, B, a), g in fin.groupby(["problem", "truth", "budget", "arm"]):
        rows.append(f"| {p} | {tr} | {B} s | {a} | {mr(g.true)} | {mr(g.regret_pct)} | {mr(g.regret_usd, '{:.3f}')} | "
                    f"{mr(g.evals_final, '{:,.0f}')} | {mr(g.gens_final, '{:.0f}')} | {g.gpu_s.sum():.0f} | "
                    f"{g.usd.sum():.3f} |")
    return "\n".join(rows)


def run_file(tr, arm, p, B, s):
    t = "x" if arm in ("A", "D", "E") else tr
    f = OUT / "runs" / f"main_{t}_{arm}_{p}_b{B}_s{s}.json"
    return json.loads(f.read_text()) if f.exists() else None


def time_to_opt(df):
    """Decision 1: first trace time whose best-so-far genome is the truth optimum (60 s runs)."""
    rows = ["| problem | truth | arm | seeds at optimum by 60 s | time to optimum (s) per seed | final line regret % |",
            "|---|---|---|---|---|---|"]
    for p in ("d1_c0", "d1_c1"):
        for tr in TRUTHS:
            sc = json.loads((OUT / f"scores_{p}_{tr}.json").read_text())
            opt = np.array(sc["best_genome"], bool)
            for arm in COLORS:
                tt, reg = [], []
                for s in (1, 2):
                    r = run_file(tr, arm, p, 60, s)
                    if r is None:
                        continue
                    hit = [row[0] for row in r["trace"] if row[4] is not None and
                           np.array_equal(np.array(row[4], bool), opt)]
                    tt.append(f"{hit[0]:.1f}" if hit else "never")
                    g = df[(df.problem == p) & (df.truth == tr) & (df.arm == arm) & (df.budget == 60) &
                           (df.seed == s) & (df.t == 60)]
                    reg += g.regret_pct.round(3).tolist()
                rows.append(f"| {p} | {tr} | {arm} | {sum(x != 'never' for x in tt)}/{len(tt)} | {', '.join(tt)} | "
                            f"{reg} |")
    return "\n".join(rows)


def shares(b, c, line, R=4000):
    """Truth shares (m,) of the client SKUs on line in cluster c (segment-weighted)."""
    prob = ga.Problem(b)
    A = prob.alts(c, line)[None]
    m = len(line)
    up, cov = ga.diff_moments(b[f"mu_{c}"], b[f"Sigma_{c}"], A, m, np)
    _, K, _, d = up.shape
    a2, L = ga.genz_chol_np(up.reshape(-1, d), np.broadcast_to(cov[:, None], (1, K, m, d, d)).reshape(-1, d, d))
    P = ga.ghk_np(a2, L, ga.antithetic(R, d, 7)).reshape(K, m)
    return b[f"w_{c}"] @ P


def line_report(b, name, lines):
    """Per cluster: line SKUs, client share, mean cannibalization (and per SKU)."""
    out = []
    for c, line in enumerate(lines):
        if not line:
            continue
        s = shares(b, c, tuple(line))
        can = []
        for i, j in enumerate(line):
            rest = tuple(x for x in line if x != j)
            if not rest:
                continue
            s0 = shares(b, c, rest)
            other_with = s.sum() - s[i]
            can.append((s0.sum() - other_with) / s[i])
        out.append(f"  - cluster {c}: client share {100 * s.sum():.1f}% ({len(line)} SKUs); cannibalization mean "
                   f"{100 * np.mean(can):.0f}% (per SKU {', '.join(f'{100 * x:.0f}%' for x in can)}); line: "
                   + "; ".join(str(b['name'][j]) for j in line))
    return f"- {name}\n" + "\n".join(out)


def d2_lines():
    md = []
    for tr in TRUTHS:
        b = dict(np.load(OUT / f"bundle_d2_{tr}.npz", allow_pickle=True))
        sc = json.loads((OUT / f"scores_d2_{tr}.json").read_text())
        md.append(f"### {TRUTHS[tr]}")
        md.append(line_report(b, f"best-known (scored {sc['best_known']:.3f} $/1k trips)", sc["best_lines"]))
        prob = ga.Problem(b)
        for arm in ("A", "C", "F"):
            r = run_file(tr, arm, "d2", 300, 1)
            if r is None:
                continue
            g = np.array(r["checkpoints"][-1]["genome"], bool)
            md.append(line_report(b, f"arm {arm}, 300 s, seed 1", [list(k[1]) for k in prob.keys(g)]))
    return "\n".join(md)


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[(df.budget == 300) | df.problem.str.startswith("d1")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for i, tr in enumerate(TRUTHS):
        for j, p in enumerate(PROBLEMS):
            ax = axes[i, j]
            for a in COLORS:
                g = d[(d.truth == tr) & (d.problem == p) & (d.arm == a)].groupby("t").regret_pct.mean()
                if len(g):
                    ax.plot(g.index, g.values, color=COLORS[a], lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2,
                            label=LABEL[a])
            ax.set_xscale("log")
            ticks = [5, 15, 30, 60] if p.startswith("d1") else [5, 15, 30, 60, 120, 300]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(x) for x in ticks])
            ax.set_xlim(4, ticks[-1] * 1.3)
            # arms that end on the same value overlap exactly: name them at the line end
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            vals = []
            for a in COLORS:
                g = d[(d.truth == tr) & (d.problem == p) & (d.arm == a)].groupby("t").regret_pct.mean()
                if len(g):
                    vals.append((float(g.values[-1]), a))
            span = max(v for v, _ in vals) - min(v for v, _ in vals) or 1.0
            groups = []                                         # end values closer than 6% of the span share a label
            for v, a in sorted(vals):
                if groups and v - groups[-1][0] < 0.06 * span:
                    groups[-1][1].append(a)
                else:
                    groups.append([v, [a]])
            for v, arms in groups:
                ax.annotate(", ".join(sorted(arms)), (ticks[-1], v), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color="#0b0b0b")
            ax.set_title(f"{PROBLEMS[p]}\n{TRUTHS[tr]}", color="#0b0b0b", loc="left", fontsize=10)
            ax.grid(True, color="#e6e5e0", lw=0.8)
            ax.set_facecolor("#fcfcfb")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#b9b8b2")
            ax.tick_params(colors="#52514e", labelsize=8)
            if i == 1:
                ax.set_xlabel("wall time (s)", color="#52514e")
            if j == 0:
                ax.set_ylabel("regret, % of best-known", color="#52514e")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Best-so-far solution scored under the truth (decision 1: 60 s runs; decision 2: 300 s runs; "
                 "mean of 2 seeds)", x=0.01, ha="left",
                 color="#0b0b0b")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "anytime.png", dpi=150)


def main():
    df = pd.read_csv(OUT / "results.csv")
    plot(df)
    md = ["## Final checkpoint (t = budget); mean [min, max] over 2 seeds", tables(df), "",
          "## Anytime regret % (mean over seeds, 300 s runs)"]
    piv = df[(df.budget == 300) | df.problem.str.startswith("d1")].pivot_table(index=["problem", "truth", "arm"], columns="t", values="regret_pct")
    md += [piv.round(3).to_markdown(), "", "## Decision 1: time to the exact optimum", time_to_opt(df), "",
           "## Decision 2: lines, client share, cannibalization (under each truth)", d2_lines()]
    (OUT / "summary_tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
