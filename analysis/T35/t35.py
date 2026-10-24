"""T35: T20's RFC-S bias floor rechecked against HB over posterior draws (TEST.md). Cells G1, G2 x N = 150, 600,
1,500 x reps 0-9 of T20; MVP results and T20's RFC-S (point estimates) reused.

    python t35.py fit          # Modal CPU: HB refit with 100 draws, SoP / RFC-S readouts, decisions
    python t35.py summary      # local: score against the truth, verdict
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
T20 = HERE.parent / "T20"
sys.path.insert(0, str(T20))
import modal_t20 as m  # noqa: E402

OUT = HERE / "out"
TRUTHS, NS, REPS = ("G1", "G2"), (150, 600, 1500), 10
D_DRAWS = 100
R_DRAWS = m.R_L // D_DRAWS           # RFC level draws per posterior draw: same 1,000 per respondent as T20
MATERIAL = 0.5                       # points of cost regret
ARMS = ("rfc_means", "sop_means", "sop_draws", "rfc_draws")

image = m.cpu_image.add_local_file(T20 / "modal_t20.py", "/root/modal_t20.py")
app = modal.App("orthant-rfc-t35", image=image)


@app.function(cpu=2.0, memory=8192, timeout=3 * 3600, retries=1)
def incumbent(truth, N, rep, sl20, g20):
    import subprocess
    t1 = m.t1_module()
    c = m.cell_data(t1, truth, N, rep)
    work = Path(tempfile.mkdtemp())
    prods, y = c["d"]["prods"], c["d"]["y"]
    X = t1.task_X(prods)
    n_, T, K1, P = X.shape
    rows = np.c_[np.repeat(np.arange(n_), T * K1), np.tile(np.repeat(np.arange(T), K1), n_), np.tile(np.arange(K1), n_ * T),
                 (np.arange(K1)[None, None, :] == y[:, :, None]).reshape(-1), X.reshape(-1, P)]
    path = work / "hb_in.csv"
    np.savetxt(path, rows, delimiter=",", comments="", fmt="%.6g",
               header="resp,task,alt,chosen," + ",".join(f"x{i}" for i in range(P)))
    t0 = time.time()
    hb_r = str((m.T1 if modal.is_local() else Path(m.T1_REMOTE)) / "hb.R")
    r = subprocess.run(["Rscript", hb_r, str(path), str(work / "hb"), str(m.HB_ITERS), "10",
                        str(c["hb_seed"]), str(D_DRAWS)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    B = np.loadtxt(work / "hb_beta.csv", delimiter=",", skiprows=1)
    Bd = np.fromfile(work / "hb_draws.f32", dtype=np.float32).reshape((N, P, D_DRAWS), order="F")
    Bd = Bd.transpose(0, 2, 1).reshape(-1, P).astype(float)
    t_hb = time.time() - t0

    class RFC(t1.RFC):              # rows are posterior draws: chunk so each block is ~ T20's 200 x 1,000
        def shares(self, prods, sl, g, chunk=20000):
            return super().shares(prods, sl, g, chunk)

    out = dict(truth=truth, N=N, rep=rep, y_sha=c["y_sha"], seconds=dict(hb=t_hb), arms={})
    Rm = t1.RFC(B, m.R_L, c["r_rfc"])                                    # T20's draws, T20's tuned sl, g
    out["arms"]["rfc_means"] = dict(decisions=t1.decisions(lambda pr: Rm.shares(pr, sl20, g20)), sl=sl20, g=g20)
    R1 = t1.RFC(B, 1, np.random.default_rng(0))
    out["arms"]["sop_means"] = dict(decisions=t1.decisions(lambda pr: R1.shares(pr, 0.0, 1.0)))
    Rd1 = RFC(Bd, 1, np.random.default_rng(0))
    out["arms"]["sop_draws"] = dict(decisions=t1.decisions(lambda pr: Rd1.shares(pr, 0.0, 1.0)))
    t0 = time.time()
    Rd = RFC(Bd, R_DRAWS, np.random.default_rng([m.SEED, m.TIDX[truth], N, rep, 35]))
    (sl, g), mae, flags = t1.tune_rfc(Rd, c["H"], c["obs"])
    out["arms"]["rfc_draws"] = dict(decisions=t1.decisions(lambda pr: Rd.shares(pr, sl, g)), sl=float(sl), g=float(g),
                                    holdout_mae=float(mae), flags=flags)
    out["seconds"]["tune"] = time.time() - t0
    for a in ("sop_means", "sop_draws"):
        Rx = R1 if a == "sop_means" else Rd1
        out["arms"][a]["holdout_mae"] = float(np.mean([np.abs(Rx.shares(c["H"][i], 0.0, 1.0) - c["obs"][i]).mean()
                                                       for i in range(len(c["H"]))]))
    return m._jsonable(out)


def cells():
    return [(t, n, r) for t in TRUTHS for n in NS for r in range(REPS)]


def fit():
    (OUT / "inc").mkdir(parents=True, exist_ok=True)
    todo = []
    for t, n, r in cells():
        if not (OUT / "inc" / f"{t}_N{n}_r{r:02d}.json").exists():
            ri = json.loads((T20 / "out" / "inc" / f"{t}_N{n}_r{r:02d}.json").read_text())
            todo.append((t, n, r, ri["sigma_l"], ri["g"]))
    print(f"{len(todo)} jobs", flush=True)
    t0 = time.time()
    with modal.enable_output(), app.run():
        for res in incumbent.starmap(todo, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
            if isinstance(res, Exception):
                print(f"FAILED: {res!r}"[:500], flush=True)
                continue
            (OUT / "inc" / f"{res['truth']}_N{res['N']}_r{res['rep']:02d}.json").write_text(json.dumps(res))
            print(f"[{time.time() - t0:5.0f} s] {res['truth']} N={res['N']} r{res['rep']}: hb {res['seconds']['hb']:.0f} s, "
                  f"tune {res['seconds']['tune']:.0f} s", flush=True)


def summary():
    """T20's MVP ran only at N = 50, 100, 200, so arms at N = 150-1,500 are compared with MVP's small-N cells
    (unpaired, Welch), as T20's own claim (RFC-S at N = 1,500 vs MVP at N = 50)."""
    from scipy import stats
    t1 = m.t1_module()
    res = json.loads((T20 / "out" / "results.json").read_text())
    out, mism, lines = {}, 0, []

    def ci(x):
        x = np.asarray(x)
        se = x.std(ddof=1) / np.sqrt(len(x))
        q = stats.t.ppf(0.975, len(x) - 1)
        return f"{x.mean():.2f} [{x.mean() - q * se:.2f}, {x.mean() + q * se:.2f}]"

    verdict = {}
    for truth in TRUTHS:
        sig = np.sqrt(t1.T_VAR * t1.TRUTHS[truth])
        td = t1.decisions(lambda pr, s=sig: t1.pop_shares(pr, t1.B_TRUE, t1.W_TRUE, s))
        for N in (50, 100, 200):
            out[f"{truth}_mvp_N{N}"] = [r["mvp"]["cost_regret_pct"] for r in res
                                         if r["truth"] == truth and r["N"] == N and r["mvp"]]
            lines.append(f"{truth} MVP N={N}: {ci(out[f'{truth}_mvp_N{N}'])}")
        for N in NS:
            rs = [json.loads((OUT / "inc" / f"{truth}_N{N}_r{r:02d}.json").read_text()) for r in range(REPS)]
            t20 = {r["rep"]: r["rfc_s"]["cost_regret_pct"] for r in res if r["truth"] == truth and r["N"] == N}
            for a in ARMS:
                out[f"{truth}_{a}_N{N}"] = [t1.score(x["arms"][a]["decisions"], td)["cost_regret_pct"] for x in rs]
            mism += sum(v != t20[r] for r, v in enumerate(out[f"{truth}_rfc_means_N{N}"]))
            lines.append(f"{truth} N={N}: " + "  ".join(f"{a} {ci(out[f'{truth}_{a}_N{N}'])}" for a in ARMS))
        y = np.array(out[f"{truth}_mvp_N50"])
        for N in NS:
            best = min(("sop_draws", "rfc_draws"), key=lambda a: np.mean(out[f"{truth}_{a}_N{N}"]))
            x = np.array(out[f"{truth}_{best}_N{N}"])
            d, se = x.mean() - y.mean(), np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
            verdict[(truth, N)] = (best, float(d), float(se), float(p))
    print(f"cost regret %, mean [95% CI] over 10 reps. rfc_means vs T20 mismatches: {mism}")
    print("\n".join(lines))
    for (t, N), (b, d, se, p) in verdict.items():
        print(f"  {t} N={N}: best draws arm {b} - MVP at N = 50: {d:+.2f} (SE {se:.2f}, Welch p {p:.4f})")
    surv = all(d > 2 * se and d >= MATERIAL for _, d, se, _ in verdict.values())
    readout = any(N == 1500 and d < MATERIAL for (t, N), (_, d, se, _) in verdict.items())
    v = "FLOOR SURVIVES" if surv else "FLOOR IS THE READOUT" if readout else "INCONCLUSIVE"
    print("Verdict (vs MVP at N = 50):", v)
    (OUT / "regrets.json").write_text(json.dumps(out, indent=1))
    (OUT / "summary.json").write_text(json.dumps(dict(verdict=v, cells={f"{t}_N{n}": x for (t, n), x in verdict.items()}),
                                                 indent=1))


if __name__ == "__main__":
    fit() if sys.argv[1:] == ["fit"] else summary()
