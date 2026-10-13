"""T14 Part B: T3 on the 24-item home-baking cluster (PLAN.md, T14).

    python analysis/T14/run_b.py selftest     # 300 households, 1 control rep per truth
    python analysis/T14/run_b.py full         # all 1,869 households, 10 control reps per truth
    python analysis/T14/run_b.py summary [selftest|full]

T3's data, covariates (log prices, trip size, household propensity), time split and arms, at M = 24:
  - CS-F (residual-lift rule, cap 5), CS-O (blocks from the IFM correlation), IL, MVP-IFM (multivariate-probit
    0.2.3, pairwise; unrepaired R kept, repaired with its nearest_correlation), MVP-ML (mvp_fit's own code on one
    H100; real data only).
  - Decisions (27): best pair, best triple, TURF-4, and the follow-on A -> B for each of the 24 items, from 1- to
    4-item quantities averaged over the holdout trips (2^24 pattern matrices are out):
      probits: 4,096 scrambled-Sobol latent draws per trip (as T3) binned into the 2^24 pattern histogram on a GPU;
        P(all of S) and P(none of S) by superset / subset sums over it;
      coded subsets: per trip, the product over blocks of each block's P(all / none of S in the block).
  - Holdout pattern log-likelihood: per-trip GHK (2,048 points) on a GPU for probits, exact for coded subsets.
  - Real-data adjudication: where two arms' picks differ, the holdout difference with a household-bootstrap SE;
    resolved if |diff| >= max(0.5 pts, 2 SE) (T6's rule).
  - Gate (T3): >= 6 items connected at rho >= 0.2 in the IFM R after covariates; fail -> stop.
  - Controls (T3): S-MVP (simulated from the real ML fit) and S-CS (from the real CS-F fit), arms CS-F, CS-O, IL,
    IFM (no ML), scored against the simulating model's own quantities.
Each job's result is saved as it arrives under out/B_<mode>/; a rerun skips it.
"""
import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
if len(HERE.parents) > 2:                                      # host only (not in the Modal container)
    sys.path.insert(0, str(HERE.parent / "T3"))
    sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))  # mvp_fit's fitting code (private sibling checkout)
    T3_FILE = HERE.parent / "T3" / "t3.py"
else:
    T3_FILE = Path("/root/t3.py")
import t3  # noqa: E402

_t3_clean_R = t3.clean_R
SHRINK_EPS = 1e-6


def clean_R(R):
    """t3.clean_R, plus a 1e-6 shrink toward I when the matrix is numerically singular (Cholesky needs PD)."""
    R = _t3_clean_R(R)
    if np.linalg.eigvalsh(R).min() < 1e-8:
        R = (R + SHRINK_EPS * np.eye(len(R))) / (1 + SHRINK_EPS)
    return R


t3.clean_R = clean_R

BAKING = {
    "eggs": [("EGGS", ["EGGS - LARGE", "EGGS - X-LARGE", "EGGS - MEDIUM"])],
    "sugar": [("SUGARS/SWEETNERS", ["SUGAR"])],
    "flour": [("FLOUR & MEALS", ["FLOUR: WHITE & SELF RISING"])],
    "butter": [("BUTTER", ["BUTTER"])],
    "cream cheese": [("CHEESE", ["CREAM CHEESE"])],
    "cake mix": [("BAKING MIXES", ["LAYER CAKE MIX"])],
    "frosting": [("BAKING MIXES", ["FROSTING"])],
    "brownie mix": [("BAKING MIXES", ["BROWNIE MIX"])],
    "muffin/cornbread mix": [("BAKING MIXES", ["MUFFIN & CORN BREAD MIX"])],
    "chocolate chips": [("BAKING NEEDS", ["BITS & MORSELS"])],
    "marshmallows": [("BAKING NEEDS", ["MARSHMALLOWS"])],
    "baking nuts": [("BAKING NEEDS", ["BAKING NUTS"])],
    "oil": [("SHORTENING/OIL", ["VEGETABLE/SALAD OIL", "CANOLA OILS"])],
    "cooking spray": [("SHORTENING/OIL", ["COOKING SPRAYS"])],
    "shortening": [("SHORTENING/OIL", ["SOLID SHORTENING"])],
    "baking powder/soda": [("BAKING NEEDS", ["BAKING POWDER & SODA"])],
    "extracts": [("SPICES & EXTRACTS", ["PURE EXTRACTS", "IMITATION EXTRACTS"])],
    "pie filling": [("BAKING NEEDS", ["PIE FILLING MINCEMEAT GLAZES"])],
    "pie crust": [("BAKING MIXES", ["PIE CRUST MIXES & SHELLS"]), ("FROZEN PIE/DESSERTS", ["FRZN PIE SHELLS PASTRY SHELL"])],
    "whipped topping": [("FROZEN PIE/DESSERTS", ["FRZN WHIPPED TOPPING"])],
    "cake decor/icing": [("BAKING NEEDS", ["CAKE DECORS & ICING", "CAKE DECORS - BIRTHDAY CANDLES"])],
    "graham crackers": [("COOKIES/CONES", ["GRAHAM CRACKERS"])],
    "aluminum foil": [("FD WRAPS/BAGS/TRSH BG", ["ALUMINUM FOIL"])],
    "refrig. cookie dough": [("REFRGRATD DOUGH PRODUCTS", ["REFRIGERATED COOKIES-CHUB"])],
}
STAPLES = ("eggs", "sugar", "oil", "aluminum foil")
M = len(BAKING)
PAIRS = list(itertools.combinations(range(M), 2))
TRIPLES = list(itertools.combinations(range(M), 3))
SETS4 = list(itertools.combinations(range(M), 4))
SET_DECS = ("bundle2", "bundle3", "turf4")
DECS = list(SET_DECS) + [f"rec{a}" for a in range(M)]
OPTIONS = dict(bundle2=PAIRS, bundle3=TRIPLES, turf4=SETS4)
MATERIAL = 0.005           # 0.5 pts
M_SIM, M_SIM_TRUTH, M_LL = 4096, 16384, 2048
N_BOOT = 1000
ARMS = ["cs_f", "cs_o", "il", "ifm", "ml"]

cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("numpy==2.5.1", "scipy==1.16.0", "multivariate-probit==0.2.3")
             .add_local_file(T3_FILE, "/root/t3.py"))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]", "torch==2.5.1", "numpy", "scipy", "pydantic")
             .add_local_python_source("modal_gp_api")
             .add_local_file(T3_FILE, "/root/t3.py"))
app = modal.App("orthant-rfc-t14b")


# ---------------------------------------------------------------------------
# Remote jobs
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=3 * 3600)
def fit_cpu(X, Y):
    """IL, lifts, CS-F, MVP-IFM (unrepaired R + repair), CS-O (blocks from the IFM R), each timed (as T3)."""
    from multivariate_probit import MultivariateProbit
    from multivariate_probit.ifm import pairwise_correlation
    p = X.shape[1]
    out = {}
    t0 = time.time()
    il = t3.fit_blocks(X, Y, [[m] for m in range(M)])
    out["il"] = dict(blocks=il, seconds=time.time() - t0)
    t0 = time.time()
    raw_l, res_l = t3.lifts(X, Y, il)
    sub = t3.analyst_subsets(res_l)
    out["cs_f"] = dict(blocks=t3.fit_blocks(X, Y, sub), subsets=sub, seconds=time.time() - t0 + out["il"]["seconds"],
                       lift_raw=np.nan_to_num(raw_l).tolist(), lift_resid=np.nan_to_num(res_l).tolist(),
                       subsets_raw_lift=t3.analyst_subsets(raw_l))
    t0 = time.time()
    mp = MultivariateProbit(inner="linear", dependence="pairwise", cv=5, random_state=0).fit(X, Y.astype(int))
    R = mp.correlation_                                                       # repaired (nearest correlation)
    raw = pairwise_correlation(mp.eta_, Y.astype(float), project=False)       # the same pairwise fits, unrepaired
    E = mp.decision_function(np.vstack([np.zeros(p), np.eye(p)]))
    ev = float(np.linalg.eigvalsh((raw + raw.T) / 2).min())
    out["ifm"] = dict(c=E[0].tolist(), B=(E[1:] - E[0]).T.tolist(), R=R.tolist(), R_raw=raw.tolist(),
                      seconds=time.time() - t0, raw_min_eig=ev, psd_repaired=bool(ev < 1e-8),
                      repair_fro=float(np.linalg.norm(raw - R)))
    t0 = time.time()
    sub = t3.oracle_subsets(R)
    out["cs_o"] = dict(blocks=t3.fit_blocks(X, Y, sub), subsets=sub, seconds=time.time() - t0)
    return out


@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=3 * 3600)
def fit_ctrl(truth, r, fit, X, tr):
    """Simulate control replicate r from the truth's fit (T3 seeds), fit_cpu on its training rows; the holdout Y
    comes back bit-packed for scoring."""
    import base64
    rng = np.random.default_rng(t3.SEED + (100 if truth == "S-MVP" else 200) + r)
    Y = t3.simulate_probit(fit, X, rng) if truth == "S-MVP" else t3.simulate_cs(fit["blocks"], X, M, rng)
    out = fit_cpu.local(X[tr], Y[tr])
    out["Yh"] = base64.b64encode(np.packbits(Y[~tr].astype(np.uint8))).decode()
    return out


def unpack_yh(c, n):
    import base64
    return np.unpackbits(np.frombuffer(base64.b64decode(c["Yh"]), np.uint8), count=n * M).reshape(n, M).astype(np.int8)


@app.function(image=gpu_image, gpu="H100", timeout=1800)
def fit_ml(X, Y, seed):
    """As T3's MVP-ML (mvp_fit: 512 draws, float32 GHK, no SEs), on one H100."""
    import torch

    import modal_gp_api as g
    t0 = time.time()
    out = g.mvp_fit_core(X, Y.astype(float), None, g.MvpFitParams(n_draws=512, seed=seed, alpha=1e-6, se=False),
                         torch.device("cuda"))
    d, p = Y.shape[1], X.shape[1]
    return dict(c=np.asarray(out["intercept"], float).tolist(), B=np.asarray(out["coef"], float).reshape(d, p).tolist(),
                R=np.asarray(out["corr"], float).reshape(d, d).tolist(), n_iter=int(out["n_iter"]),
                converged=bool(out["converged"]), grad_max=float(out["grad_max"]), seconds=time.time() - t0,
                gpu=torch.cuda.get_device_name())


def set_sums(hist, d):
    """Over a pattern histogram (2^d, first item = most significant bit): sums over supersets (P(all of S))
    and over subsets (P(pattern within T))."""
    sup, sub = hist.copy().reshape([2] * d), hist.copy().reshape([2] * d)
    for ax in range(d):
        a0, a1 = [slice(None)] * d, [slice(None)] * d
        a0[ax], a1[ax] = 0, 1
        sup[tuple(a0)] += sup[tuple(a1)]
        sub[tuple(a1)] += sub[tuple(a0)]
    return sup.reshape(-1), sub.reshape(-1)


def set_codes(sets):
    bits = 1 << np.arange(M)[::-1]
    return np.array([bits[list(S)].sum() for S in sets], dtype=np.int64)


@app.function(image=gpu_image, gpu="T4", timeout=1800)
def probit_score(fit, Xh, Yh, m_sim, ll=True):
    """Probit set quantities averaged over the holdout trips (latent simulation into the 2^24 histogram, then
    superset / subset sums) and the holdout log-likelihood (per-trip GHK)."""
    import torch
    from torch.special import ndtri

    import modal_gp_api as g
    dev, f64 = torch.device("cuda"), torch.float64
    t0 = time.time()
    eta = torch.as_tensor(t3.probit_eta(fit, Xh), dtype=f64, device=dev)
    R = t3.clean_R(fit["R"])
    L = torch.as_tensor(np.linalg.cholesky(R), dtype=f64, device=dev)
    sob = torch.quasirandom.SobolEngine(M, scramble=True, seed=0).draw(m_sim, dtype=f64).to(dev)
    w = ndtri(sob.clamp(1e-12, 1 - 1e-12)) @ L.T                              # (m, M)
    bits = torch.as_tensor(1 << np.arange(M)[::-1], dtype=torch.int64, device=dev)
    hist = torch.zeros(1 << M, dtype=f64, device=dev)
    chunk = max(1, int(2 ** 25 // (m_sim * M)))
    for s in range(0, len(eta), chunk):
        c = (((eta[s:s + chunk, None, :] + w[None]) > 0).long() * bits).sum(-1)
        hist += torch.bincount(c.ravel(), minlength=1 << M).to(f64)
    hist = (hist / (len(eta) * m_sim)).cpu().numpy()
    sup, sub = set_sums(hist, M)
    full = (1 << M) - 1
    q = dict(marg=sup[set_codes([(m,) for m in range(M)])].tolist(),
             bundle2=sup[set_codes(PAIRS)].tolist(), bundle3=sup[set_codes(TRIPLES)].tolist(),
             turf4=(1 - sub[full ^ set_codes(SETS4)]).tolist())
    t_dec = time.time() - t0
    out = dict(q=q, seconds_decide=t_dec)
    if ll:
        s = torch.as_tensor(2.0 * Yh - 1, dtype=f64, device=dev)
        u = torch.quasirandom.SobolEngine(M - 1, scramble=True, seed=1).draw(M_LL, dtype=f64).to(dev)
        lls = []
        with torch.no_grad():
            for a in range(0, len(eta), 256):
                b = min(a + 256, len(eta))
                lls.append(g.mvp_ghk_loglik(eta[a:b], s[a:b], L[None], u[None].expand(b - a, -1, -1)))
        out["holdout_ll"] = float(torch.cat(lls).mean())
    return out


def cs_quantities(blocks, Xh):
    """Coded-subset set quantities averaged over trips: per trip, the product over blocks."""
    ms = [t3.CovMNL(b["items"], b["theta"]) for b in blocks]
    where = {}
    allp, nonep = [], []
    for bi, m in enumerate(ms):
        pr = m.pattern_probs(Xh)                                              # (n, 2^s)
        s = len(m.items)
        A, Nn = {}, {}
        for r in range(1, min(s, 4) + 1):
            for U in itertools.combinations(range(s), r):
                A[U] = pr[:, np.all(m.P[:, list(U)] == 1, 1)].sum(1)
                Nn[U] = pr[:, np.all(m.P[:, list(U)] == 0, 1)].sum(1)
        allp.append(A)
        nonep.append(Nn)
        for q, item in enumerate(m.items):
            where[item] = (bi, q)

    def avg(S, table):
        parts = {}
        for item in S:
            bi, q = where[item]
            parts.setdefault(bi, []).append(q)
        v = np.ones(len(Xh))
        for bi, qs in parts.items():
            v = v * table[bi][tuple(sorted(qs))]
        return float(v.mean())

    return dict(marg=[avg((m,), allp) for m in range(M)], bundle2=[avg(S, allp) for S in PAIRS],
                bundle3=[avg(S, allp) for S in TRIPLES], turf4=[1 - avg(S, nonep) for S in SETS4])


@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=3600)
def cs_score(fit, Xh, Yh):
    t0 = time.time()
    q = cs_quantities(fit["blocks"], Xh)
    t_dec = time.time() - t0
    return dict(q=q, seconds_decide=t_dec, holdout_ll=float(t3.cs_holdout_ll(fit["blocks"], Xh, Yh).mean()))


# ---------------------------------------------------------------------------
# Decisions, adjudication
# ---------------------------------------------------------------------------
def full_q(q):
    """Add the follow-on values P(B | A) = P(A, B) / P(A)."""
    q = {k: np.asarray(v, float) for k, v in q.items()}
    J2 = np.zeros((M, M))
    for (i, j), v in zip(PAIRS, q["bundle2"]):
        J2[i, j] = J2[j, i] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = J2 / q["marg"][:, None]
    np.fill_diagonal(cond, -np.inf)
    return dict(q, **{f"rec{a}": cond[a] for a in range(M)})


def picks(q):
    Q = full_q(q)
    return {k: int(np.argmax(Q[k])) for k in DECS}


def observed_q(Y):
    Yf = Y.astype(float)
    return dict(marg=Yf.mean(0).tolist(), bundle2=[float(Yf[:, list(S)].prod(1).mean()) for S in PAIRS],
                bundle3=[float(Yf[:, list(S)].prod(1).mean()) for S in TRIPLES],
                turf4=[float(Yf[:, list(S)].max(1).mean()) for S in SETS4])


def event_counts(dec, opt, Y, hh, n_hh):
    """Per-household holdout numerator and denominator of one option's value."""
    if dec.startswith("rec"):
        a = int(dec[3:])
        num, den = Y[:, a] * Y[:, opt], Y[:, a]
    else:
        S = list(OPTIONS[dec][opt])
        num = Y[:, S].max(1) if dec == "turf4" else Y[:, S].prod(1)
        den = np.ones(len(Y))
    return np.bincount(hh, num.astype(float), n_hh), np.bincount(hh, den.astype(float), n_hh)


def adjudicate(pa, pb, Y, hh, n_hh, W):
    """Where picks differ: holdout values, difference (a - b), household-bootstrap SE; verdict by T6's rule."""
    out = {}
    for k in DECS:
        if pa[k] == pb[k]:
            out[k] = dict(differ=False)
            continue
        na, da = event_counts(k, pa[k], Y, hh, n_hh)
        nb, db = event_counts(k, pb[k], Y, hh, n_hh)
        diff = na.sum() / da.sum() - nb.sum() / db.sum()
        boot = (W @ na) / (W @ da) - (W @ nb) / (W @ db)
        se = float(np.std(boot, ddof=1))
        res = abs(diff) >= max(MATERIAL, 2 * se)
        out[k] = dict(differ=True, hold_a=float(na.sum() / da.sum()), hold_b=float(nb.sum() / db.sum()),
                      diff=float(diff), se=se, verdict=("a" if diff > 0 else "b") if res else "unresolved")
    return out


def score_vs_truth(q, qt):
    Q, Qt = full_q(q), full_q(qt)
    out = {}
    for k in DECS:
        c, tv = int(np.argmax(Q[k])), Qt[k]
        out[k] = dict(flip=bool(c != int(np.argmax(tv))), regret_pts=float(100 * (tv.max() - tv[c])))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(mode):
    out = HERE / "out" / f"B_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t_start = time.time()

    def cached(name, fn):
        p = out / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text())
        try:
            r = fn()
        except Exception as e:
            r = dict(error=f"{type(e).__name__}: {e}"[:500])
        p.write_text(json.dumps(r))
        log(f"{name} {'ERROR ' + r['error'][:300] if 'error' in r else 'ok'} ({time.time() - t_start:.0f} s)")
        return r

    n_hh, n_ctrl = (300, 1) if mode == "selftest" else (None, 10)
    dpath = out / "data.pkl"
    if dpath.exists():
        D = pickle.loads(dpath.read_bytes())
    else:
        t3.CLUSTERS["baking"] = BAKING
        D = t3.load("baking", HERE.parents[1] / "data" / "completejourney", n_hh=n_hh)
        dpath.write_bytes(pickle.dumps(D))
    X, Y, tr, names, hh = D["X"], D["Y"], D["train"], D["names"], D["hh"]
    Xh = X[~tr]
    log(f"T14 B {mode}: {D['n_hh']} households, {len(Y)} trips ({tr.sum()} training), X {X.shape[1]} columns, max |z| train {np.abs(X[tr]).max():.1f} holdout {np.abs(Xh).max():.1f}")

    with modal.enable_output(), app.run(), ThreadPoolExecutor(48) as ex:
        Xtr, Ytr = X[tr], Y[tr]
        f_ml = ex.submit(cached, "real_ml", lambda: fit_ml.remote(Xtr, Ytr, 0))
        real = cached("real_cpu", lambda: fit_cpu.remote(Xtr, Ytr))
        if "error" in real:
            return
        R = np.array(real["ifm"]["R"])
        g = t3.gate(R, names)
        iu = np.triu_indices(M, 1)
        st = [names.index(s) for s in STAPLES]
        is_st = np.isin(iu[0], st) | np.isin(iu[1], st)
        g["staple_pairs"] = dict(median=float(np.median(R[iu][is_st])), max=float(R[iu][is_st].max()),
                                 frac_ge_02=float(np.mean(R[iu][is_st] >= .2)))
        g["nonstaple_pairs"] = dict(median=float(np.median(R[iu][~is_st])), max=float(R[iu][~is_st].max()),
                                    frac_ge_02=float(np.mean(R[iu][~is_st] >= .2)))
        g["linked_without_staples"] = t3.gate(np.where(np.add.outer(np.isin(range(M), st), np.isin(range(M), st)),
                                                       0.0, R), names)
        (out / "gate.json").write_text(json.dumps(g, indent=1))
        log(f"CS-F subsets {[[names[m] for m in s] for s in real['cs_f']['subsets']]}; gate {g}")
        if not g["passed"]:
            log("gate FAILED: stopping (T3 rule; stop and ask)")
            return

        fc = {("S-CS", r): ex.submit(cached, f"ctrl_cpu_S-CS_{r}", lambda r=r: fit_ctrl.remote("S-CS", r, real["cs_f"], X, tr))
              for r in range(n_ctrl)}
        real_ml = f_ml.result()
        if "error" in real_ml:
            log("real MVP-ML failed; S-MVP controls need it: stopping")
            return
        for r in range(n_ctrl):
            fc[("S-MVP", r)] = ex.submit(cached, f"ctrl_cpu_S-MVP_{r}",
                                         lambda r=r: fit_ctrl.remote("S-MVP", r, real_ml, X, tr))

        def score(name, fit, Yh, m_sim=M_SIM, ll=True):
            if "blocks" in fit:
                return cached(name, lambda: cs_score.remote(fit, Xh, Yh))
            return cached(name, lambda: probit_score.remote(fit, Xh, Yh, m_sim, ll))

        jobs = {f"score_real_{a}": ex.submit(score, f"score_real_{a}", (real_ml if a == "ml" else real[a]), Y[~tr])
                for a in ARMS}
        jobs["truth_S-MVP"] = ex.submit(score, "truth_S-MVP", real_ml, Y[~tr], M_SIM_TRUTH, False)
        jobs["truth_S-CS"] = ex.submit(score, "truth_S-CS", real["cs_f"], Y[~tr])
        ctrl = {j: f.result() for j, f in fc.items()}
        for j, c in ctrl.items():
            if "error" in c:
                continue
            for a in ARMS[:4]:
                n = f"score_ctrl_{j[0]}_{j[1]}_{a}"
                jobs[n] = ex.submit(score, n, c[a], unpack_yh(c, len(Xh)))
        for f in jobs.values():
            f.result()
    log(f"done ({time.time() - t_start:.0f} s); summary: python analysis/T14/run_b.py summary {mode}")


def summary(mode):
    out = HERE / "out" / f"B_{mode}"
    D = pickle.loads((out / "data.pkl").read_bytes())
    Y, tr, hh, names, n_hh = D["Y"], D["train"], D["hh"], D["names"], D["n_hh"]
    Yh, hhh = Y[~tr].astype(float), hh[~tr]
    ld = lambda n: json.loads((out / f"{n}.json").read_text()) if (out / f"{n}.json").exists() else {"error": "missing"}  # noqa: E731
    g = ld("gate")
    lines = [f"Gate: {g.get('size')} items linked at rho >= 0.2 ({'PASS' if g.get('passed') else 'FAIL'}): "
             f"{g.get('linked')}", f"  staple pairs (any of {STAPLES}): {g.get('staple_pairs')}",
             f"  non-staple pairs: {g.get('nonstaple_pairs')}",
             f"  linked set with staple pairs zeroed: {g.get('linked_without_staples', {}).get('size')} items"]
    real_cpu = ld("real_cpu")
    if "ifm" in real_cpu:
        f = real_cpu["ifm"]
        lines.append(f"IFM PSD repair: {f['psd_repaired']} (min eig raw {f['raw_min_eig']:.3g}, "
                     f"|R - R~|_F {f['repair_fro']:.3f}); CS-F subsets "
                     f"{[[names[m] for m in s] for s in real_cpu['cs_f']['subsets']]}")
    ml = ld("real_ml")
    if "error" not in ml:
        ev = float(np.linalg.eigvalsh(_t3_clean_R(ml["R"])).min())
        lines.append(f"ML: {ml['seconds']:.0f} s on {ml['gpu']}, n_iter {ml['n_iter']}, converged {ml['converged']}, "
                     f"R min eigenvalue {ev:.2g}{' (shrunk 1e-6 toward I for scoring)' if ev < 1e-8 else ''}")
    sc = {a: ld(f"score_real_{a}") for a in ARMS}
    pk = {a: picks(s["q"]) for a, s in sc.items() if "error" not in s}
    pk["observed"] = picks(observed_q(Y[~tr]))
    lines.append("Holdout LL per trip: " + ", ".join(f"{a} {s['holdout_ll']:.4f}" for a, s in sc.items() if "error" not in s))
    rng = np.random.default_rng(t3.SEED)
    W = rng.multinomial(n_hh, np.full(n_hh, 1 / n_hh), size=N_BOOT).astype(float)
    res = dict(picks=pk, adjudication={})
    for a, b in (("ifm", "cs_f"), ("ml", "cs_f"), ("ifm", "cs_o"), ("ifm", "il"), ("ml", "ifm")):
        if a in pk and b in pk:
            res["adjudication"][f"{a}_vs_{b}"] = adjudicate(pk[a], pk[b], Yh, hhh, n_hh, W)
    adj = res["adjudication"].get("ifm_vs_cs_f", {})
    if adj:
        by = {}
        for k, v in adj.items():
            typ = k if k in SET_DECS else "follow-on"
            d = by.setdefault(typ, dict(n=0, differ=0, mvp=0, cs=0, unresolved=0))
            d["n"] += 1
            if v["differ"]:
                d["differ"] += 1
                d[{"a": "mvp", "b": "cs", "unresolved": "unresolved"}[v["verdict"]]] += 1
        tot = {x: sum(d[x] for d in by.values()) for x in ("differ", "mvp", "cs", "unresolved")}
        set_loss = [k for k in SET_DECS if adj[k].get("verdict") == "b"]
        resolved = tot["mvp"] + tot["cs"]
        verdict = ("KILL" if tot["cs"] >= tot["mvp"] else
                   "PASS" if tot["mvp"] - tot["cs"] >= 3 and tot["mvp"] > resolved / 2 and not set_loss else "INCONCLUSIVE")
        res["by_type"], res["totals"], res["verdict"] = by, tot, verdict
        lines.append("MVP-IFM vs CS-F on the holdout, by decision type (differ / backs MVP / backs CS-F / unresolved):")
        for typ, d in by.items():
            lines.append(f"  {typ:10s} {d['differ']}/{d['n']} differ: {d['mvp']} / {d['cs']} / {d['unresolved']}")
        for k in DECS:
            v = adj[k]
            if v["differ"]:
                lines.append(f"    {k:7s} IFM {fmt(k, pk['ifm'][k], names)} {v['hold_a']:.4f} vs CS-F "
                             f"{fmt(k, pk['cs_f'][k], names)} {v['hold_b']:.4f}: diff {100 * v['diff']:+.2f} pts "
                             f"(SE {100 * v['se']:.2f}) -> {v['verdict']}")
        lines.append(f"Totals {tot}; MVP loses a set decision: {set_loss or 'none'}. Verdict: {verdict}.")
    for key in ("ml_vs_cs_f", "ml_vs_ifm"):
        a = res["adjudication"].get(key)
        if a:
            c = {x: sum(v.get("verdict") == x for v in a.values()) for x in ("a", "b", "unresolved")}
            lines.append(f"{key}: differ {sum(v['differ'] for v in a.values())}/{len(a)}; backs first {c['a']}, "
                         f"second {c['b']}, unresolved {c['unresolved']}")
    ctrl = {}
    for truth in ("S-MVP", "S-CS"):
        qt = ld(f"truth_{truth}")
        if "error" in qt:
            continue
        for p in sorted(out.glob(f"score_ctrl_{truth}_*_*.json")):
            r, a = p.stem[len(f"score_ctrl_{truth}_"):].split("_", 1)
            s = json.loads(p.read_text())
            if "error" in s:
                continue
            sv = score_vs_truth(s["q"], qt["q"])
            c = ctrl.setdefault((truth, a), dict(flips=0, material=0, n=0, ll=[]))
            c["n"] += 1
            c["flips"] += sum(v["flip"] for v in sv.values())
            c["material"] += sum(v["flip"] and v["regret_pts"] > 100 * MATERIAL for v in sv.values())
            c["ll"].append(s["holdout_ll"])
    if ctrl:
        lines.append(f"Controls (flips of {len(DECS)} decisions per rep, material > 0.5 pts; holdout LL):")
        for (truth, a), c in sorted(ctrl.items()):
            lines.append(f"  {truth} {a:5s} reps {c['n']}: flips {c['flips']} (material {c['material']}); LL {np.mean(c['ll']):.4f}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def fmt(dec, opt, names):
    if dec.startswith("rec"):
        return f"{names[int(dec[3:])]}->{names[opt]}"
    return "+".join(names[m] for m in OPTIONS[dec][opt])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else "full")
    else:
        run(cmd)
