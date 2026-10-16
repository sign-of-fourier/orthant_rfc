"""T15 main comparison (TEST.md, "Main comparison"): GPU M*P pipeline vs HB-MNL + RFC-S (logit IIA), regret against
the truth on D1-D4, 5 S1 datasets + an S0 guard.

    python main.py pilot            # S1 rep 0 end to end (T4's gate fit), then stop and report
    python main.py reps 1,2,3,4     # remaining S1 datasets (MVP refits on H100)
    python main.py summary          # graded verdict over the reps present
    python main.py local            # CPU wiring check at toy sizes
"""
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
if len(HERE.parents) > 2:                                      # host only (not in the Modal container)
    for d in ("T4", "T9"):
        sys.path.insert(0, str(HERE.parent / d))
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "private"))
    sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))
PRIVATE = (HERE / "private" / "t15_engine.py").exists()

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("r-base-core", "r-cran-bayesm")
         .pip_install("torch==2.5.1", "numpy", "scipy", "pydantic", "fastapi", "pybhatlib==0.4.0")
         .add_local_file(str(HERE.parent / "T1" / "hb.R"), "/root/hb.R")
         .add_local_python_source(*(["modal_gp_api", "macml", "t4", "mvp_gpu", "t15"]
                                    + (["t15_engine"] if PRIVATE else []))))
app = modal.App("orthant-rfc-t15-main", image=image)

OUT = HERE / "out" / "main"
N_RESP, T_TASKS, J = 1000, 12, 20
QR = (512, 256)                     # MVP-G draws (T4 gate size)
M2 = 4096                           # stage-2 GHK and local search (MVP arms)
M_TRUTH_SCREEN, M_TRUTH = 16384, 65536
N_STAGE1 = 1_000_000                # D2 / D4 stage-1 candidates (D1, D3 exhaustive)
N_TRUTH = 10_000_000                # D2 / D4 truth-side screen
HB_ITERS, RFC_R = 20000, 200
EXH_INC, STARTS = 50_000, 20        # incumbent: exhaustive if space <= 50,000, else hill-climb from 20 starts
ACC_FRAC = 0.8                      # accurate-only arm: stage 1 gets 80% of the pipeline's wall time
MATERIAL = 0.5                      # points of % best profit (user 2026-10-16)


# ---------------------------------------------------------------------------
# Shared helpers (host and container)
# ---------------------------------------------------------------------------
def setup():
    import t15
    m = t15.model()
    F = json.loads((HERE / "out" / "probe.json").read_text())["H100"]["F"] if len(HERE.parents) > 2 else None
    return m, t15.instances(), F


def prep(inst, F):
    return dict(inst, F=F * inst.get("F_mult", 0.0) if inst["dec"] in ("D3", "D4") else 0.0)


def enumerate_space(inst):
    from itertools import combinations
    if inst["dec"] == "D1":
        return np.array(list(combinations(range(len(inst["shelf"])), inst["k"])))
    if inst["dec"] == "D3":
        g = len(inst["grid"]) + 1
        return np.array(np.unravel_index(np.arange(g ** len(inst["ext"])), (g,) * len(inst["ext"]))).T
    return None


def neighbours(inst, c):
    d = inst["dec"]
    out = []
    if d == "D1":
        s = set(c.tolist())
        outs = [j for j in range(len(inst["shelf"])) if j not in s]
        for i in range(len(c)):
            for j in outs:
                x = c.copy()
                x[i] = j
                out.append(np.sort(x))
    elif d == "D2":
        for i in range(len(c)):
            for st in (-1, 1):
                v = c[i] + st
                if 0 <= v < len(inst["grid"]):
                    x = c.copy()
                    x[i] = v
                    out.append(x)
    else:
        for i in range(len(c)):
            for v in range(len(inst["grid"]) + 1):
                if v != c[i]:
                    x = c.copy()
                    x[i] = v
                    out.append(x)
    return np.array(out)


def local_search(inst, f, start, max_iter=200):
    cur, val = np.asarray(start), float(f(np.asarray(start)[None])[0])
    n = 1
    for _ in range(max_iter):
        nb = neighbours(inst, cur)
        v = f(nb)
        n += len(nb)
        i = int(np.argmax(v))
        if v[i] <= val + 1e-12:
            break
        cur, val = nb[i], float(v[i])
    return cur, val, n


def random_cands(inst, n, rng):
    d = inst["dec"]
    if d == "D1":
        return np.sort(rng.random((n, len(inst["shelf"]))).argsort(1)[:, :inst["k"]], 1)
    if d == "D2":
        return rng.integers(0, len(inst["grid"]), (n, len(inst["own"])))
    slots = len(inst["ext"]) if d == "D3" else len(inst["own"]) + len(inst["ext"])
    return rng.integers(0, len(inst["grid"]) + 1, (n, slots))


def torch_f(sh, scorer, dev):
    import torch
    import t15
    return lambda X: t15.score(sh, scorer, torch.as_tensor(np.asarray(X), device=dev)).cpu().numpy()


def stage1(sh, th, C, dev):
    """Engine screen to a top 100: private screening settings if available, else the engine's default settings."""
    import torch
    import t15
    t0 = time.perf_counter()
    try:
        import t15_engine
        sel, info = t15_engine.screen(sh, th, C, dev)
        info = dict(n=info.get("n"))
    except ImportError:
        S = t15.score(sh, t15.EngineScorer(sh, th), C)
        sel, info = S.topk(min(100, S.numel())).indices, dict(fallback="engine default")
    t15.sync(dev)
    info["seconds"] = time.perf_counter() - t0
    return sel, info


# ---------------------------------------------------------------------------
# Incumbent: HB-MNL + RFC-S (CPU)
# ---------------------------------------------------------------------------
def hb_impl(truth, rep, hb_iters=HB_ITERS, rfc_r=RFC_R, workdir="/tmp"):
    import subprocess
    import t4
    t0 = time.time()
    d = t4.make_data(truth, J, N_RESP, T_TASKS, rep)
    X = t4.task_X(d["levels"], d["prices"])                         # (N, T, J+1, P)
    N, T, K1, P = X.shape
    rows = np.c_[np.repeat(np.arange(N), T * K1), np.tile(np.repeat(np.arange(T), K1), N), np.tile(np.arange(K1), N * T),
                 (np.arange(K1)[None, None, :] == d["y"][:, :, None]).reshape(-1), X.reshape(-1, P)]
    path = Path(workdir) / f"hb_{truth}_{rep}.csv"
    np.savetxt(path, rows, delimiter=",", comments="", fmt="%.6g",
               header="resp,task,alt,chosen," + ",".join(f"x{i}" for i in range(P)))
    hb_script = "/root/hb.R" if Path("/root/hb.R").exists() else str(HERE.parent / "T1" / "hb.R")
    r = subprocess.run(["Rscript", hb_script, str(path), str(path.with_suffix("")), str(hb_iters), "10", str(1000 + rep)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    B = np.loadtxt(str(path.with_suffix("")) + "_beta.csv", delimiter=",", skiprows=1)
    llh = np.loadtxt(str(path.with_suffix("")) + "_loglike.csv", delimiter=",", skiprows=1)
    q = len(llh) // 4
    drift = float(abs(llh[3 * q:].mean() - llh[2 * q:3 * q].mean()) / abs(llh[3 * q:].mean()))
    t_hb = time.time() - t0
    obs = np.stack([np.bincount(d["yh"][:, i], minlength=J + 1) / N for i in range(d["yh"].shape[1])])
    R = t4.RFC(B, rfc_r, np.random.default_rng(7000 + rep))
    (sl, g), mae, flags = t4.tune_rfc(R, d["levels"], d["Hp"], obs, 1)
    return dict(truth=truth, rep=rep, B=B.tolist(), sl=float(np.ravel(sl)[0]), g=float(g), rfc_mae=float(mae),
                rfc_flags=flags, hb_seconds=t_hb, hb_drift=drift, rfc_seconds=time.time() - t0 - t_hb)


class RFCTorch:
    """t4.RFC (Sawtooth-style RFC-S on HB point estimates) vectorised over candidates in torch: normal attribute-level
    errors sl shared by alternatives with the same brand / flavor / pack level, Gumbel product error g integrated
    analytically, "none" with a normal error of variance 3 sl^2. Draws R_l per respondent, fixed per seed."""

    def __init__(self, m, B, R_l, seed, sl, g, dtype=None):
        import torch
        self.t = torch
        dt = dtype or torch.float32
        gen = torch.Generator().manual_seed(seed)
        self.B = torch.as_tensor(np.asarray(B), dtype=dt)
        N = self.B.shape[0]
        lv = m["levels"]
        self.Z = [torch.randn(N, R_l, int(L), generator=gen, dtype=dt) for L in lv]
        self.Zn = torch.randn(N, R_l, generator=gen, dtype=dt)
        self.lev = torch.as_tensor(np.r_[m["lev"], np.zeros((1, 3), int)], dtype=torch.long)    # none row: unused
        self.Xb = torch.as_tensor(m["Xb"], dtype=dt)
        self.pcol, self.ref, self.sl, self.g, self.dt = m["pcol"], m["ref"], float(sl), float(g), dt

    def shares(self, alts, pr, batch=8):
        """alts (C, n) grid indices with none (60) last; pr (C, n) -> (C, n) shares."""
        t = self.t
        C, n = alts.shape
        out = t.empty(C, n, dtype=self.dt)
        for lo in range(0, C, batch):
            a = alts[lo:lo + batch]
            X = self.Xb[a].clone()
            X[..., self.pcol] = t.where(a == 60, 0.0, (pr[lo:lo + batch] - self.ref).to(self.dt))
            V = t.einsum("np,ckp->cnk", self.B, X)                                   # (c, N, n)
            lv = self.lev[a[:, :-1]]                                                  # (c, n-1, 3)
            e = sum(self.Z[k][:, :, lv[:, :, k]] for k in range(3)) * self.sl         # (N, R, c, n-1)
            e = t.cat([e, (3 ** 0.5 * self.sl) * self.Zn[:, :, None, None].expand(-1, -1, a.shape[0], 1)], 3)
            u = (V.permute(1, 0, 2)[:, None] + e) / self.g                           # (N, R, c, n)
            out[lo:lo + batch] = t.softmax(u, 3).mean((0, 1))
        return out


def rfc_objective(m, inst, rfc):
    import torch
    import t15
    sh = t15.Shelf(m, inst, torch.device("cpu"))
    sh.F = inst["F"]

    def f(X):
        out = np.empty(len(X))
        for grp in sh.expand(torch.as_tensor(np.asarray(X))):
            s = rfc.shares(grp["alts"], grp["pr"]).double()[:, :grp["m"]]
            v = (grp["margin"] * s).sum(1) - inst["F"] * grp["fixed"]
            out[grp["idx"].numpy()] = v.numpy()
        return out
    return f


def incumbent_impl(m, inst, B, sl, g, seed):
    import t15
    import t4
    rfc = RFCTorch(m, B, RFC_R, seed, sl, g)
    f = rfc_objective(m, inst, rfc)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if t15.space_size(inst) <= EXH_INC:
        X = enumerate_space(inst)
        v = np.concatenate([f(X[i:i + 2000]) for i in range(0, len(X), 2000)])
        i = int(np.argmax(v))
        pick, val, n, how = X[i], float(v[i]), len(X), "exhaustive"
    else:
        best = None
        n = 0
        for s in random_cands(inst, STARTS, rng):
            c, v, k = local_search(inst, f, s)
            n += k
            if best is None or v > best[1]:
                best = (c, v)
        pick, val, how = best[0], best[1], f"hill-climb x{STARTS}"
    return dict(name=inst["name"], pick=np.asarray(pick).tolist(), own_value=val, evals=n, seconds=time.time() - t0,
                method=how)


@app.function(cpu=2.0, memory=4096, timeout=4 * 3600)
def hb_fit(truth, rep):
    return hb_impl(truth, rep)


@app.function(cpu=4.0, memory=8192, timeout=4 * 3600)
def incumbent_search(m, inst, B, sl, g, seed):
    return incumbent_impl(m, inst, B, sl, g, seed)


# ---------------------------------------------------------------------------
# MVP: pipeline and accurate-only (GPU)
# ---------------------------------------------------------------------------
def mvp_impl(m, insts, th_hat, seed, n_stage1=N_STAGE1, m2=M2):
    import torch
    import t15
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th = torch.as_tensor(np.asarray(th_hat), dtype=torch.float64, device=dev)
    out = {}
    for j, inst in enumerate(insts):
        rng = np.random.default_rng([seed, j])
        sh = t15.Shelf(m, inst, dev)
        sh.F = inst["F"]
        Xs = enumerate_space(inst)
        if Xs is None:
            Xs = random_cands(inst, n_stage1, rng)
        C = torch.as_tensor(Xs, device=dev)
        f2 = torch_f(sh, t15.GHKScorer(sh, th, m2), dev)
        # pipeline
        t0 = time.perf_counter()
        sel, info = stage1(sh, th, C, dev)
        short = C[sel].cpu().numpy()
        v = f2(short)
        pick, val, n_ls = local_search(inst, f2, short[int(np.argmax(v))])
        t_pipe = time.perf_counter() - t0
        pipe = dict(pick=np.asarray(pick).tolist(), own_value=val, seconds=t_pipe, stage1=info, ls_evals=n_ls)
        # accurate-only: GHK m2 on a random stream for ACC_FRAC of the pipeline's time, then the same local search
        t0 = time.perf_counter()
        order = rng.permutation(len(Xs))
        best, nsc = None, 0
        for lo in range(0, len(order), 2048):
            if time.perf_counter() - t0 > ACC_FRAC * t_pipe and best is not None:
                break
            X = Xs[order[lo:lo + 2048]]
            vv = f2(X)
            nsc += len(X)
            i = int(np.argmax(vv))
            if best is None or vv[i] > best[1]:
                best = (X[i], float(vv[i]))
        pick_a, val_a, n_la = local_search(inst, f2, best[0])
        acc = dict(pick=np.asarray(pick_a).tolist(), own_value=val_a, seconds=time.perf_counter() - t0, scored=nsc,
                   ls_evals=n_la)
        out[inst["name"]] = dict(pipeline=pipe, accurate=acc)
    return dict(gpu=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", inst=out)


@app.function(gpu="H100", timeout=2 * 3600)
def mvp_search(m, insts, th_hat, seed):
    return mvp_impl(m, insts, th_hat, seed)


def fit_impl(truth, rep):
    import mvp_gpu
    import t4
    d = t4.make_data(truth, J, N_RESP, T_TASKS, rep)
    X = t4.task_X(d["levels"], d["prices"]).astype(np.float32)
    A = t4.member(d["levels"]).astype(np.float32)
    s2 = t4.sig_of("S1") ** 2 / t4.T_VAR
    th_true = np.r_[t4.b_true(J), np.log(t4.W_TRUE), np.log(s2[:3] / s2[3])]
    th0 = np.r_[np.zeros(t4.P), np.log(np.full(t4.P, 0.5)), np.zeros(3)]
    r = mvp_gpu._fit(X, d["y"], A, QR[0], QR[1], "G", th0, th_true, 1, 32, False, 300)
    return dict(r, truth=truth, rep=rep)


@app.function(gpu="H100", timeout=6 * 3600)
def mvp_fit(truth, rep):
    return fit_impl(truth, rep)


class MixedLogitScorer:
    """S0 truth: beta_n ~ N(b_true, diag(W_TRUE^2)), iid Gumbel errors. Share of j = E_beta[softmax(X beta)_j] over the
    present alternatives, averaged over Nd fixed normal draws (exact logit given beta)."""

    def __init__(self, sh, b, w, Nd, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(Nd, len(b), generator=g, dtype=torch.float64)
        self.beta = (torch.as_tensor(b, dtype=torch.float64) + torch.as_tensor(w, dtype=torch.float64) * z).to(sh.dev)
        self.sh, self.Nd = sh, Nd

    def shares(self, grp):
        import torch
        sh = self.sh
        X = sh.Xb[grp["alts"]].clone()
        X[..., sh.pcol] = torch.where(grp["alts"] == 60, 0.0, grp["pr"] - sh.ref)
        C, n, _ = X.shape
        out = torch.empty(C, grp["m"], dtype=torch.float64, device=X.device)
        step = max(1, 2 ** 26 // (n * self.Nd))
        for lo in range(0, C, step):
            V = torch.einsum("cnp,dp->cnd", X[lo:lo + step], self.beta)
            out[lo:lo + step] = torch.softmax(V, 1).mean(2)[:, :grp["m"]]
        return out


def truth_scorer(m, sh, truth, M):
    """S1: GHK at theta_true with M draws; S0: mixed logit with M taste draws."""
    import torch
    import t15
    if truth == "S1":
        return t15.GHKScorer(sh, torch.as_tensor(m["th_true"], dtype=torch.float64, device=sh.dev), M)
    return MixedLogitScorer(sh, m["b_true"], m["w_true"], M)


# ---------------------------------------------------------------------------
# Truth (S1: the probit at theta_true): true best per instance, and true values of picks
# ---------------------------------------------------------------------------
def truth_best_impl(m, insts, n_truth=N_TRUTH, m_screen=M_TRUTH_SCREEN, m_final=M_TRUTH, chunk=1_000_000, truth="S1"):
    import torch
    import t15
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th = torch.as_tensor(m["th_true"], dtype=torch.float64, device=dev)
    out = {}
    for j, inst in enumerate(insts):
        t0 = time.perf_counter()
        sh = t15.Shelf(m, inst, dev)
        sh.F = inst["F"]
        fs = torch_f(sh, truth_scorer(m, sh, truth, m_screen if truth == "S1" else 4096), dev)
        ff = torch_f(sh, truth_scorer(m, sh, truth, m_final), dev)
        Xs = enumerate_space(inst)
        if Xs is not None:
            v = np.concatenate([fs(Xs[i:i + 20000]) for i in range(0, len(Xs), 20000)])
            top = Xs[np.argsort(v)[-200:]]
            how = "exhaustive"
        else:
            rng = np.random.default_rng([99, j])
            tops = []
            for lo in range(0, n_truth, chunk):
                C = torch.as_tensor(random_cands(inst, min(chunk, n_truth - lo), rng), device=dev)
                if truth == "S1":
                    sel, _ = stage1(sh, th, C, dev)
                else:                                   # S0: screen with the cheap mixed-logit truth itself
                    v = torch.as_tensor(torch_f(sh, truth_scorer(m, sh, truth, 512), dev)(C.cpu().numpy()))
                    sel = v.topk(100).indices.to(dev)
                tops.append(C[sel].cpu().numpy())
            top = np.unique(np.concatenate(tops), axis=0)
            how = f"screen {n_truth:,} + local search"
        vt = ff(top)
        start = top[int(np.argmax(vt))]
        if Xs is None:
            start, _, _ = local_search(inst, fs, start)
        best = np.asarray(start)
        out[inst["name"]] = dict(best=best.tolist(), value=float(ff(best[None])[0]), method=how,
                                 seconds=time.perf_counter() - t0)
    return out


def truth_score_impl(m, insts, picks, m_final=M_TRUTH, truth="S1"):
    """picks: {inst name: [cand, ...]} -> {inst name: [true value, ...]}."""
    import torch
    import t15
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th = torch.as_tensor(m["th_true"], dtype=torch.float64, device=dev)
    out = {}
    for inst in insts:
        if inst["name"] not in picks:
            continue
        sh = t15.Shelf(m, inst, dev)
        sh.F = inst["F"]
        out[inst["name"]] = torch_f(sh, truth_scorer(m, sh, truth, m_final), dev)(np.array(picks[inst["name"]])).tolist()
    return out


@app.function(gpu="H100", timeout=3 * 3600)
def truth_best(m, insts, truth="S1"):
    return truth_best_impl(m, insts, truth=truth)


@app.function(gpu="H100", timeout=3600)
def truth_score(m, insts, picks, truth="S1"):
    return truth_score_impl(m, insts, picks, truth=truth)


# ---------------------------------------------------------------------------
# Host: run one S1 dataset end to end
# ---------------------------------------------------------------------------
def run_rep(rep, m, insts, F, truth="S1"):
    OUT.mkdir(parents=True, exist_ok=True)
    insts = [prep(i, F) for i in insts]
    path = OUT / f"{truth}_rep{rep}.json"
    t_all = time.time()
    with modal.enable_output(), app.run():
        tb_path = OUT / f"truth_best_{truth}.json"
        tb_call = None if tb_path.exists() else truth_best.spawn(m, insts, truth)
        hb_call = hb_fit.spawn(truth, rep)
        if rep == 0 and truth == "S1":
            fit = json.loads((HERE.parent / "T4" / "out" / "gate_g" / "fit_512x256.json").read_text())
            fit = dict(theta=fit["theta"], seconds=fit["seconds"], device=fit["device"], source="T4 gate fit")
        else:
            fit = mvp_fit.remote(truth, rep)
        mvp_call = mvp_search.spawn(m, insts, fit["theta"], 500 + rep)
        hb = hb_call.get()
        inc_calls = {i["name"]: incumbent_search.spawn(m, i, hb["B"], hb["sl"], hb["g"], 900 + rep) for i in insts}
        mvp = mvp_call.get()
        inc = {k: c.get() for k, c in inc_calls.items()}
        if tb_call is not None:
            tb_path.write_text(json.dumps(tb_call.get(), indent=1))
        tb = json.loads(tb_path.read_text())
        picks = {n: [inc[n]["pick"], mvp["inst"][n]["pipeline"]["pick"], mvp["inst"][n]["accurate"]["pick"], tb[n]["best"]]
                 for n in inc}
        tv = truth_score.remote(m, insts, picks, truth)
    res = dict(rep=rep, truth=truth, fit={k: v for k, v in fit.items() if k != "cov_sandwich"},
               hb={k: v for k, v in hb.items() if k != "B"}, gpu=mvp["gpu"], inst={}, wall=time.time() - t_all)
    for n in inc:
        v_inc, v_pipe, v_acc, v_tb = tv[n]
        best = max(tv[n])
        reg = lambda v: 100 * (best - v) / best      # noqa: E731
        own = max(mvp["inst"][n]["pipeline"]["own_value"], mvp["inst"][n]["accurate"]["own_value"])
        res["inst"][n] = dict(
            best_true=best, truth_search=tb[n],
            incumbent=dict(inc[n], true_value=v_inc, regret=reg(v_inc)),
            pipeline=dict(mvp["inst"][n]["pipeline"], true_value=v_pipe, regret=reg(v_pipe),
                          own_regret=100 * (own - mvp["inst"][n]["pipeline"]["own_value"]) / own),
            accurate=dict(mvp["inst"][n]["accurate"], true_value=v_acc, regret=reg(v_acc),
                          own_regret=100 * (own - mvp["inst"][n]["accurate"]["own_value"]) / own))
    path.write_text(json.dumps(res, indent=1, default=float))
    return res


def show(res):
    print(f"{res['truth']} rep {res['rep']}: wall {res['wall'] / 60:.0f} min; HB {res['hb']['hb_seconds']:.0f} s (drift "
          f"{res['hb']['hb_drift']:.4f}), RFC-S sl {res['hb']['sl']:.3f} g {res['hb']['g']:.3f} MAE {res['hb']['rfc_mae']:.4f}")
    print(f"{'instance':<12} {'incumbent regret / s':>22} {'pipeline regret / s':>22} {'accurate regret / s':>22}")
    for n, r in res["inst"].items():
        c = lambda a: f"{r[a]['regret']:6.2f}% / {r[a]['seconds']:6.0f}"   # noqa: E731
        print(f"{n:<12} {c('incumbent'):>22} {c('pipeline'):>22} {c('accurate'):>22}")


# ---------------------------------------------------------------------------
# Summary: graded verdict
# ---------------------------------------------------------------------------
def label(d, lo, hi, m=MATERIAL):
    if hi < 0:
        return "HARD FAIL + harm"
    if lo > 0:
        return "PASS" if d >= m else "SOFT FAIL"
    return "INCONCLUSIVE" if hi >= m else "HARD FAIL"


def summary():
    from scipy import stats
    reps = [json.loads(p.read_text()) for p in sorted(OUT.glob("S1_rep*.json"))]
    rows, ps = {}, []
    for dec in ("D1", "D2", "D3", "D4"):
        per = [np.mean([r["inst"][n]["incumbent"]["regret"] - r["inst"][n]["pipeline"]["regret"]
                        for n in r["inst"] if n.startswith(dec)]) for r in reps]
        inc = [np.mean([r["inst"][n]["incumbent"]["regret"] for n in r["inst"] if n.startswith(dec)]) for r in reps]
        k = len(per)
        d = float(np.mean(per))
        if k > 1:
            se = float(np.std(per, ddof=1) / np.sqrt(k))
            q = stats.t.ppf(0.975, k - 1)
            lo, hi = d - q * se, d + q * se
            p = float(2 * stats.t.sf(abs(d) / se, k - 1)) if se > 0 else 0.0
        else:
            lo = hi = d
            p = float("nan")
        base = float(np.mean(inc))
        rows[dec] = dict(delta=d, lo=lo, hi=hi, n_datasets=k, per_dataset=per, incumbent_regret=base,
                         rel_lift=100 * d / base if base else None, p=p)
        ps.append(p)
    ok = [p for p in ps if p == p]
    if ok:
        order = np.argsort(ok)
        adj, run = np.empty(len(ok)), 0.0
        for r, i in enumerate(order):
            run = max(run, min(1.0, (len(ok) - r) * ok[i]))
            adj[i] = run
        for dec, a in zip([d for d in rows if rows[d]["p"] == rows[d]["p"]], adj):
            rows[dec]["p_holm"] = float(a)
    for dec, r in rows.items():
        r["label"] = label(r["delta"], r["lo"], r["hi"]) if r["n_datasets"] > 1 else "single dataset (no CI)"
    print(json.dumps(rows, indent=1, default=float))
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1, default=float))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "local"
    m, insts, F = setup()
    if cmd == "pilot":
        show(run_rep(0, m, insts, F))
    elif cmd == "reps":
        for rep in [int(x) for x in sys.argv[2].split(",")]:
            show(run_rep(rep, m, insts, F))
    elif cmd == "s0":
        show(run_rep(0, m, insts, F, truth="S0"))
    elif cmd == "summary":
        summary()
    elif cmd == "local":
        # wiring at toy sizes on CPU: one D1 and one D4 instance, no HB (random B), tiny budgets
        import t4
        sub = [prep(i, F) for i in insts if i["name"] in ("D1_k6", "D4_base")]
        rng = np.random.default_rng(0)
        B = t4.b_true(J) + t4.W_TRUE * rng.standard_normal((50, t4.P))
        for i in sub:
            small = dict(i)
            r = incumbent_impl(m, dict(small, name=i["name"]), B, 0.3, 1.0, 1) if i["dec"] != "D1" else None
            print("incumbent", i["name"], r and {k: r[k] for k in ("own_value", "evals", "seconds", "method")})
        global N_STAGE1
        r = mvp_impl(m, sub, m["th_hat"], 1, n_stage1=2000, m2=256)
        for n, v in r["inst"].items():
            print("mvp", n, {a: (round(v[a]["own_value"], 4), round(v[a]["seconds"], 1)) for a in v})
        tb = truth_best_impl(m, sub[1:], n_truth=4000, m_screen=256, m_final=1024, chunk=2000)
        print("truth best", {n: (round(v["value"], 4), v["method"]) for n, v in tb.items()})
        print("truth score", truth_score_impl(m, sub[1:], {sub[1]["name"]: [r["inst"][sub[1]["name"]]["pipeline"]["pick"]]},
                                              m_final=1024))


if __name__ == "__main__":
    main()
