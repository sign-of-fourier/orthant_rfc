"""T34: T15's D1 assortment win (MNP vs HB-MNL + RFC-S on point estimates) rechecked against HB over posterior
draws (TEST.md). S1 truth, J = 20, N = 1,000, 12 tasks, reps 0-4; MNP picks, truth best and values reused from T15.

    python t34.py run 0,1,2,3,4     # Modal: HB refit with 100 draws (CPU), RFC tuning, D1 searches + truth scores (H100)
    python t34.py summary
"""
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
T15 = HERE.parent / "T15"
if len(HERE.parents) > 2:                                      # host only
    for d in ("T4", "T15"):
        sys.path.insert(0, str(HERE.parent / d))

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("r-base-core", "r-cran-bayesm")
         .pip_install("torch==2.5.1", "numpy", "scipy")
         .add_local_file(str(HERE.parent / "T1" / "hb.R"), "/root/hb.R")
         .add_local_python_source("t4", "t15"))
app = modal.App("orthant-rfc-t34", image=image)

OUT = HERE / "out"
N_RESP, T_TASKS, J = 1000, 12, 20
HB_ITERS, D_DRAWS = 20000, 100
R_MEANS, R_DRAWS = 200, 2               # RFC error draws per respondent (T15) / per posterior draw
M_TRUTH = 65536
MATERIAL = 0.5                          # points of % best profit
ARMS = ("sop_means", "rfc_means", "sop_draws", "rfc_draws")


def rfc_class():
    import t4

    class RFC(t4.RFC):                  # larger chunks: 100,000 pseudo-respondents for the draws arm
        def shares(self, levels, prices, sl, g, present=None, chunk=2000):
            return super().shares(levels, prices, sl, g, present, chunk)
    return RFC


# ---------------------------------------------------------------------------
# CPU: HB refit with draws, holdout tuning
# ---------------------------------------------------------------------------
def hb_impl(rep, workdir="/tmp"):
    import subprocess
    import t4
    t0 = time.time()
    d = t4.make_data("S1", J, N_RESP, T_TASKS, rep)
    X = t4.task_X(d["levels"], d["prices"])
    N, T, K1, P = X.shape
    rows = np.c_[np.repeat(np.arange(N), T * K1), np.tile(np.repeat(np.arange(T), K1), N), np.tile(np.arange(K1), N * T),
                 (np.arange(K1)[None, None, :] == d["y"][:, :, None]).reshape(-1), X.reshape(-1, P)]
    path = Path(workdir) / f"hb_S1_{rep}.csv"
    np.savetxt(path, rows, delimiter=",", comments="", fmt="%.6g",
               header="resp,task,alt,chosen," + ",".join(f"x{i}" for i in range(P)))
    hb_script = str(HERE.parent / "T1" / "hb.R") if len(HERE.parents) > 2 else "/root/hb.R"
    r = subprocess.run(["Rscript", hb_script, str(path), str(path.with_suffix("")), str(HB_ITERS), "10", str(1000 + rep),
                        str(D_DRAWS)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    pre = str(path.with_suffix(""))
    B = np.loadtxt(pre + "_beta.csv", delimiter=",", skiprows=1)
    Bd = np.fromfile(pre + "_draws.f32", dtype=np.float32).reshape((N, P, D_DRAWS), order="F").transpose(0, 2, 1)
    t_hb = time.time() - t0
    obs = np.stack([np.bincount(d["yh"][:, i], minlength=J + 1) / N for i in range(d["yh"].shape[1])])
    RFC = rfc_class()
    out = dict(rep=rep, hb_seconds=t_hb, B=B.astype(np.float32), Bd=Bd.astype(np.float32), tune={})
    for arm, Bx, R_l, seed in (("means", B, R_MEANS, 7000 + rep), ("draws", Bd.reshape(-1, P), R_DRAWS, 7100 + rep)):
        R = RFC(Bx.astype(float), R_l, np.random.default_rng(seed))
        sop = np.mean([np.abs(R.shares(d["levels"], d["Hp"][i], 0.0, 1.0) - obs[i]).mean() for i in range(len(obs))])
        t1 = time.time()
        (sl, g), mae, flags = t4.tune_rfc(R, d["levels"], d["Hp"], obs, 1)
        out["tune"][arm] = dict(sl=float(np.ravel(sl)[0]), g=float(g), rfc_mae=float(mae), sop_mae=float(sop),
                                flags=flags, seconds=time.time() - t1)
    return out


@app.function(cpu=4.0, memory=8192, timeout=3 * 3600)
def hb_fit(rep):
    return hb_impl(rep)


# ---------------------------------------------------------------------------
# GPU: exhaustive D1 search under each incumbent readout, then truth scores
# ---------------------------------------------------------------------------
def d1_values(E, mg, cand, chunk=1024):
    """D1 profit at fixed base prices. E (R, 21): exp utilities of the 20 shelf SKUs + none (col 20) per simulated
    respondent / error draw, mg (20,) margins, cand (C, k) shelf positions -> (C,) mean profit (logit within the
    candidate's alternatives)."""
    import torch
    out = torch.empty(cand.shape[0], dtype=torch.float64, device=E.device)
    En, Em = E[:, 20], E[:, :20] * mg
    for lo in range(0, cand.shape[0], chunk):
        c = cand[lo:lo + chunk]
        A = torch.zeros(len(c), 20, dtype=E.dtype, device=E.device).scatter_(1, c, 1.0)
        out[lo:lo + chunk] = ((Em @ A.T) / (E[:, :20] @ A.T + En[:, None])).mean(0)
    return out


def exp_utils(m, inst, B, R_l, sl, g, seed, dev):
    """(len(B) * R_l, 21) float64 exp utilities, RFC as t4.RFC (SoP when sl = 0, g = 1), shifted per row."""
    import torch
    import t4
    shelf = np.array(inst["shelf"])
    lev = t4.ALL[shelf]
    X = t4.task_X(lev, np.array(inst["bp"]))                                         # (21, P), none last
    rng = np.random.default_rng(seed)
    Bt = torch.as_tensor(np.asarray(B, float), device=dev)
    V = (Bt @ torch.as_tensor(X, device=dev).T)[:, None, :].expand(-1, R_l, -1)      # (N, R, 21)
    if sl > 0:
        e = torch.zeros(len(B), R_l, 21, dtype=torch.float64, device=dev)
        for a, L in enumerate(t4.LEVELS):
            Z = torch.as_tensor(rng.standard_normal((len(B), R_l, L)), device=dev)
            e[:, :, :20] += sl * Z[:, :, lev[:, a]]
        e[:, :, 20] = np.sqrt(3) * sl * torch.as_tensor(rng.standard_normal((len(B), R_l)), device=dev)
        V = V + e
    U = (V / g).reshape(-1, 21)
    return (U - U.max(1, keepdim=True)[0]).exp()


def search_impl(m, insts, hb, picks_t15):
    import torch
    import t15
    from itertools import combinations
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = m["P"]
    cfg = {"sop_means": (hb["B"], 1, 0.0, 1.0), "rfc_means": (hb["B"], R_MEANS, hb["tune"]["means"]["sl"], hb["tune"]["means"]["g"]),
           "sop_draws": (hb["Bd"].reshape(-1, P), 1, 0.0, 1.0),
           "rfc_draws": (hb["Bd"].reshape(-1, P), R_DRAWS, hb["tune"]["draws"]["sl"], hb["tune"]["draws"]["g"])}
    th = torch.as_tensor(m["th_true"], dtype=torch.float64, device=dev)
    res = {}
    for inst in insts:
        mg = torch.as_tensor(np.array(inst["bp"]) - np.array(inst["cost"])[inst["shelf"]], device=dev)
        Xs = torch.as_tensor(np.array(list(combinations(range(20), inst["k"]))), device=dev)
        r = {}
        for arm, (B, R_l, sl, g) in cfg.items():
            t0 = time.perf_counter()
            E = exp_utils(m, inst, B, R_l, sl, g, 900 + hb["rep"], dev)
            v = d1_values(E, mg, Xs)
            i = int(v.argmax())
            t15.sync(dev)
            r[arm] = dict(pick=Xs[i].tolist(), own_value=float(v[i]), seconds=time.perf_counter() - t0)
            del E
        sh = t15.Shelf(m, inst, dev)
        cand = [r[a]["pick"] for a in ARMS] + picks_t15[inst["name"]]
        tv = t15.score(sh, t15.GHKScorer(sh, th, M_TRUTH), torch.as_tensor(cand, device=dev)).cpu().numpy()
        for a, v in zip(ARMS, tv):
            r[a]["true_value"] = float(v)
        r["t15_true_values"] = tv[len(ARMS):].tolist()
        res[inst["name"]] = r
    return res


@app.function(gpu="H100", timeout=2 * 3600)
def search(m, insts, hb, picks_t15):
    return search_impl(m, insts, hb, picks_t15)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------
def setup():
    import t15
    m = t15.model()
    insts = [i for i in t15.instances() if i["dec"] == "D1"]
    return m, insts


def run(reps):
    OUT.mkdir(parents=True, exist_ok=True)
    m, insts = setup()
    with modal.enable_output(), app.run():
        calls = {rep: hb_fit.spawn(rep) for rep in reps}
        for rep in reps:
            t0 = time.time()
            hb = calls[rep].get()
            t15r = json.loads((T15 / "out" / "main" / f"S1_rep{rep}.json").read_text())
            picks = {i["name"]: [t15r["inst"][i["name"]][a]["pick"] for a in ("incumbent", "pipeline")] for i in insts}
            sr = search.remote(m, insts, hb, picks)
            res = dict(rep=rep, hb_seconds=hb["hb_seconds"], tune=hb["tune"], wall=time.time() - t0, inst={})
            for n, r in sr.items():
                best = max([t15r["inst"][n]["best_true"]] + [r[a]["true_value"] for a in ARMS])
                reg = lambda v: 100 * (best - v) / best      # noqa: E731
                res["inst"][n] = dict(best_true=best, t15_best_true=t15r["inst"][n]["best_true"],
                                      t15_rescored=dict(zip(("incumbent", "mnp"), r["t15_true_values"])),
                                      mnp=dict(pick=picks[n][1], regret=t15r["inst"][n]["pipeline"]["regret"]),
                                      t15_incumbent=dict(pick=picks[n][0], regret=t15r["inst"][n]["incumbent"]["regret"]),
                                      **{a: dict(r[a], regret=reg(r[a]["true_value"]),
                                                 forecast=100 * (r[a]["own_value"] - r[a]["true_value"]) / r[a]["true_value"])
                                         for a in ARMS})
            (OUT / f"S1_rep{rep}.json").write_text(json.dumps(res, indent=1))
            print(f"rep {rep}: HB {hb['hb_seconds']:.0f} s; tune {json.dumps(hb['tune'])}")
            for n, r in res["inst"].items():
                print(f"  {n}: MNP {r['mnp']['regret']:.2f}  T15 RFC-S {r['t15_incumbent']['regret']:.2f}  "
                      + "  ".join(f"{a} {r[a]['regret']:.2f} (fc {r[a]['forecast']:+.1f}%)" for a in ARMS), flush=True)


def summary():
    from scipy import stats
    reps = [json.loads(p.read_text()) for p in sorted(OUT.glob("S1_rep*.json"))]
    k = len(reps)
    q = stats.t.ppf(0.975, k - 1)
    print(f"{k} datasets x 3 instances (D1 k = 6, 8, 10). Regret in points of % best profit; delta = arm - MNP "
          f"(positive = MNP better), t-interval over per-dataset means.")
    rows, ps = {}, {}
    for a in ("t15_incumbent",) + ARMS:
        reg = [np.mean([r["inst"][n][a]["regret"] for n in r["inst"]]) for r in reps]
        per = [np.mean([r["inst"][n][a]["regret"] - r["inst"][n]["mnp"]["regret"] for n in r["inst"]]) for r in reps]
        d, se = float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(k))
        p = float(2 * stats.t.sf(abs(d) / se, k - 1)) if se > 0 else 0.0
        fc = [np.mean([r["inst"][n][a]["forecast"] for n in r["inst"]]) for r in reps] if a in ARMS else None
        rows[a] = dict(regret=float(np.mean(reg)), delta=d, se=se, lo=d - q * se, hi=d + q * se, p=p, per=per,
                       forecast=None if fc is None else float(np.mean(fc)))
        if a in ("sop_draws", "rfc_draws"):
            ps[a] = p
    order = sorted(ps, key=ps.get)
    run_ = 0.0
    for i, a in enumerate(order):
        run_ = max(run_, min(1.0, (len(order) - i) * ps[a]))
        rows[a]["p_holm"] = run_
    mnp = np.mean([np.mean([r["inst"][n]["mnp"]["regret"] for n in r["inst"]]) for r in reps])
    print(f"MNP regret {mnp:.2f}")
    for a, r in rows.items():
        print(f"{a:14s} regret {r['regret']:.2f}  delta {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] SE {r['se']:.2f}"
              f"  p {r['p']:.3f}" + (f" Holm {r['p_holm']:.3f}" if "p_holm" in r else "")
              + ("" if r["forecast"] is None else f"  forecast {r['forecast']:+.1f}%")
              + "  per " + " ".join(f"{x:.2f}" for x in r["per"]))
    best = min(("sop_draws", "rfc_draws"), key=lambda a: rows[a]["delta"])
    b = rows[best]
    v = ("HOLDS" if b["delta"] > 2 * b["se"] and b["delta"] >= MATERIAL else
         "KILL" if b["delta"] < MATERIAL else "INCONCLUSIVE")
    print(f"Best draws arm {best}: delta {b['delta']:+.2f} (SE {b['se']:.2f}) -> {v}")
    (OUT / "summary.json").write_text(json.dumps(dict(rows=rows, verdict=v, best=best), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "run":
        run([int(x) for x in sys.argv[2].split(",")])
    else:
        summary()
