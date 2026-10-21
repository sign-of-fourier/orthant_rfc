"""T21 A · MBC's incumbent on T19's real takeaway baskets: coded subsets over all 21 groups vs MVP.

    python analysis/T21/t21.py            # every restaurant x variant, then report
    python analysis/T21/t21.py run R V    # one cell (R = 1, 2; V = all, small, all_size)
    python analysis/T21/t21.py report     # out/report_cs.md

T19's data, split, covariates, M0 / M1 / M3 fits and bundle list (imported from analysis/T19/t19.py), plus the
MBC incumbent T19 left out: coded subsets that cover all 21 groups, as T3 built them (t3.CovMNL blocks of at
most 5 groups, independence across blocks).
  CS-A  T3's analyst rule (CS-F there): lift beyond the covariates outside [0.8, 1.2], greedy merge, cap 5
  CS-R  blocks from M3's fitted correlation (CS-O there): the best grouping the cap allows, given MVP's R
Scored on T19's holdout: joint log-likelihood (exact for CS) and co-occurrence MAE on pairs, top-50 triples and
the 546 bundles, split into events inside one block and events that span blocks. Also saves each cell's M3
holdout indices and correlations for the engine timing (engine.py).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A / "T19"), str(A / "T3")]
import t19  # noqa: E402
import t3  # noqa: E402
from multivariate_probit import MultivariateProbit  # noqa: E402

OUT = HERE / "out"
CELLS = [(r, v) for r in (1, 2) for v in ("all", "small", "all_size")]
G = t19.GROUPS


def cs_fit(X, Y, subsets):
    return [t3.CovMNL(s).fit(X, Y) for s in subsets]


def cs_event_probs(blocks, X, events):
    """P(every group in the event = 1) per row: product over blocks of the block's pattern mass."""
    out = np.ones((len(X), len(events)))
    for m in blocks:
        pr = m.pattern_probs(X)
        for e, ev in enumerate(events):
            inb = [m.items.index(g) for g in ev if g in m.items]
            if inb:
                out[:, e] *= pr[:, m.P[:, inb].all(1)].sum(1)
    return out


def spans(subsets, ev):
    blk = {g: i for i, s in enumerate(subsets) for g in s}
    return len({blk[g] for g in ev}) > 1


def run(r, variant):
    rng = np.random.default_rng(t19.SEED)
    D = t19.prep(r, variant)
    X, Y, tr = D["X"], D["Y"], D["tr"]
    d = Y.shape[1]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[~tr], Y[~tr]
    nte = len(Yte)
    order = list(np.argsort(-Ytr.mean(0), kind="stable"))

    m0 = t19.fit_m0(Xtr, Ytr)
    m1 = t19.fit_m1(Xtr, Ytr, order)
    m3 = MultivariateProbit(inner="linear", dependence="pairwise", cv=5, random_state=t19.SEED).fit(Xtr, Ytr)
    Rraw = m3.correlation_
    Rm = (1 - t19.SHRINK) * Rraw + t19.SHRINK * np.eye(d)
    eta = m3.transform(Xte)

    il = t3.fit_blocks(Xtr, Ytr, [[m] for m in range(d)])
    _, lift = t3.lifts(Xtr, Ytr, il)
    subsets = {"CS-A": t3.analyst_subsets(lift), "CS-R": t3.oracle_subsets(Rraw)}
    cs = {k: cs_fit(Xtr, Ytr, s) for k, s in subsets.items()}
    res = {"restaurant": r, "variant": variant, "n_train": int(tr.sum()), "n_holdout": nte,
           "subsets": {k: [[G[g] for g in s] for s in v] for k, v in subsets.items()},
           "cs_converged": {k: all(m.converged for m in v) for k, v in cs.items()},
           "n_params": {"M1 / M3": d * (X.shape[1] + 1) + d * (d - 1) // 2,
                        **{k: sum(m.s * (X.shape[1] + 1) + len(m.multi) for m in v) for k, v in cs.items()}}}

    # joint log-likelihood per holdout order
    ll = {"M0": sum(t19.lp(m, Xte, Yte[:, j]) for j, m in enumerate(m0)),
          "M1": t19.m1_logp(m1, Xte, Yte, order),
          "M3": t19.ghk_logp(eta, Yte.astype(float), Rm)[0],
          **{k: t3.cs_holdout_ll([m.to_dict() for m in v], Xte, Yte) for k, v in cs.items()}}
    W = t19.boot_weights(nte, rng)
    boot = lambda v: t19.summarize(np.concatenate([[v.mean()], (W @ v) / nte]))  # noqa: E731
    res["ll"] = {k: boot(v) for k, v in ll.items()}
    for k in ("M0", "M1", "CS-A", "CS-R"):
        res["ll"][f"M3-{k}"] = boot(ll["M3"] - ll[k])

    # events
    gi = {g: i for i, g in enumerate(G)}
    trip_all = list(combinations(range(d), 3))
    tf = t19.obs_events(Ytr, trip_all).mean(0)
    events = {"pairs": list(combinations(range(d), 2)),
              "triples": [trip_all[i] for i in np.argsort(-tf)[:50]],
              "bundles": [(gi[e],) + tuple(gi[s] for s in c) for e in t19.ENTREES for k in (1, 2)
                          for c in combinations(t19.SIDES, k)]}
    S1 = t19.m1_sample(m1, Xte, order, d, Ytr, rng)
    P0 = t19.m0_marg(m0, Xte, Ytr)
    models = ("M0", "M1", "M3", "CS-A", "CS-R")
    for name, evs in events.items():
        Ob = t19.boot_rates(W, t19.obs_events(Yte, evs))
        pred = {"M0": t19.indep_event_probs(P0, evs), "M1": t19.sample_event_probs(S1, evs),
                "M3": t19.m3_event_probs(eta, Rm, evs),
                **{k: cs_event_probs(v, Xte, evs) for k, v in cs.items()}}
        E = {k: np.abs(t19.boot_rates(W, v) - Ob) * 100 for k, v in pred.items()}     # pts, (1+B, n_events)
        blk = {"n": len(evs)}
        for part in ("all", "CS-A span", "CS-A within", "CS-R span", "CS-R within"):
            if part == "all":
                mask = np.ones(len(evs), bool)
            else:
                k, how = part.split()
                sp = np.array([spans(subsets[k], ev) for ev in evs])
                mask = sp if how == "span" else ~sp
            if not mask.any():
                blk[part] = {"n": 0}
                continue
            b = {"n": int(mask.sum())}
            for k in models:
                b[k] = t19.summarize(E[k][:, mask].mean(1))
            for k in ("M0", "M1", "CS-A", "CS-R"):
                b[f"{k}-M3"] = t19.summarize(E[k][:, mask].mean(1) - E["M3"][:, mask].mean(1))
            blk[part] = b
        if name == "bundles":
            top10 = np.argsort(-Ob[0])[:10]
            blk["obs_top10"] = {k: t19.summarize(E[k][:, top10].mean(1)) for k in models}
            blk["obs_top10"].update({f"{k}-M3": t19.summarize(E[k][:, top10].mean(1) - E["M3"][:, top10].mean(1))
                                     for k in ("M1", "CS-A", "CS-R")})
            blk["obs_top10_spans"] = {k: int(sum(spans(subsets[k], evs[i]) for i in top10)) for k in subsets}
        res[name] = blk

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"m3_r{r}_{variant}.npz", eta=eta, Y=Yte, Rraw=Rraw, Rm=Rm)
    (OUT / f"cs_r{r}_{variant}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"r{r} {variant}: CS-A {res['subsets']['CS-A']}\n  ll " +
          ", ".join(f"{k} {v['pt']:.3f}" for k, v in res["ll"].items() if "-" not in k or k.startswith("M3-")),
          flush=True)
    return res


def fmt(c, k=2, sign=False):
    f = f"{{:{'+' if sign else ''}.{k}f}}"
    return f"{f.format(c['pt'])} [{f.format(c['lo'])}, {f.format(c['hi'])}]"


def report():
    R = {(r, v): json.loads((OUT / f"cs_r{r}_{v}.json").read_text()) for r, v in CELLS
         if (OUT / f"cs_r{r}_{v}.json").exists()}
    L = ["# T21 A · coded subsets on T19's holdout (generated by t21.py report)\n",
         "Errors are absolute, in percentage points of holdout orders. Differences: positive = M3 better.\n",
         "## Blocks\n", "| r | variant | CS-A (analyst lift rule) | CS-R (from M3's R) |", "|---|---|---|---|"]
    for (r, v), x in R.items():
        sa = "; ".join("+".join(g.lower() for g in s) for s in x["subsets"]["CS-A"] if len(s) > 1)
        sr = "; ".join("+".join(g.lower() for g in s) for s in x["subsets"]["CS-R"] if len(s) > 1)
        L.append(f"| {r} | {v} | {sa} | {sr} |")
    L += ["\n## Joint log-likelihood (nats per order, 21 groups)\n",
          "| r | variant | M0 | M1 | CS-A | CS-R | M3 | M3 − CS-A | M3 − CS-R | M3 − M1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (r, v), x in R.items():
        l_ = x["ll"]
        L.append(f"| {r} | {v} | " + " | ".join(f"{l_[k]['pt']:.3f}" for k in ("M0", "M1", "CS-A", "CS-R", "M3"))
                 + " | " + " | ".join(fmt(l_[f"M3-{k}"], 3, True) for k in ("CS-A", "CS-R", "M1")) + " |")
    for name in ("pairs", "triples", "bundles"):
        L += [f"\n## {name}: co-occurrence MAE, pts\n",
              "| r | variant | events | n | M0 | M1 | CS-A | CS-R | M3 | CS-A − M3 | CS-R − M3 | M1 − M3 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (r, v), x in R.items():
            for part in ("all", "CS-A span", "CS-A within", "CS-R span", "CS-R within"):
                b = x[name][part]
                if not b["n"]:
                    continue
                L.append(f"| {r} | {v} | {part} | {b['n']} | " +
                         " | ".join(f"{b[k]['pt']:.3f}" for k in ("M0", "M1", "CS-A", "CS-R", "M3")) + " | " +
                         " | ".join(fmt(b[f"{k}-M3"], 3, True) for k in ("CS-A", "CS-R", "M1")) + " |")
    L += ["\n## Observed top-10 bundles: MAE, pts\n",
          "| r | variant | top 10 spanning CS-A / CS-R blocks | M1 | CS-A | CS-R | M3 | CS-A − M3 | CS-R − M3 | M1 − M3 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (r, v), x in R.items():
        b = x["bundles"]["obs_top10"]
        sp = x["bundles"]["obs_top10_spans"]
        L.append(f"| {r} | {v} | {sp['CS-A']} / {sp['CS-R']} | " +
                 " | ".join(f"{b[k]['pt']:.2f}" for k in ("M1", "CS-A", "CS-R", "M3")) + " | " +
                 " | ".join(fmt(b[f"{k}-M3"], 2, True) for k in ("CS-A", "CS-R", "M1")) + " |")
    (OUT / "report_cs.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    a = sys.argv[1:]
    if a and a[0] == "report":
        report()
    elif a and a[0] == "run":
        run(int(a[1]), a[2])
    else:
        for c in CELLS:
            run(*c)
        report()


if __name__ == "__main__":
    main()
