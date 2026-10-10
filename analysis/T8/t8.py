"""T8 (GPR): kernel-parameterised error covariance vs variance shares vs HB-MNL + RFC-S.
Design and kill rules: BACKLOG.md "GPR" (user, 2026-10-10); PLAN.md T8.

    python analysis/T8/t8.py gate                 # recovery gate (truth b, N = 1,000, SEs on); run first
    python analysis/T8/t8.py run --truth b        # 20 replicates of one truth (a, b, c); gate must pass
    python analysis/T8/t8.py summary              # verdicts over a, b, c
    python analysis/T8/t8.py run --truth b --reps 1 --N 100 --check   # wiring check

DGP: T1's CBC (3 products + none per task, T = 12, normal tastes) over a fixed universe of 36 SKUs
(brand 4 x flavor 3 x pack 3), each with a fixed perceptual sweetness in [0, 1]. Sweetness does not
enter utility means; it enters only the error covariance. Truths (error variance T_VAR = pi^2/6):
  a: categorical (T1 G1): brand / flavor / pack / nugget = .08 / .45 / .45 / .02
  b: kernel: brand / flavor / pack / RBF(l = 0.3) / nugget = .05 / .20 / .20 / .40 / .15
  c: merge: brand / flavor / pack = .05 / .20 / .20; product error (.55) shared by merged products,
     P(merge i, j) = 1 / (1 + exp((d - 0.6) / 0.2)), d = |d sweetness| + categorical mismatches
Arms (same datasets): kernel (brand, flavor, pack, RBF, nugget; ARD-RBF over the one continuous
attribute) and variance shares (brand, flavor, pack, sweetness tercile, nugget), both by T1's GHK panel
fit on GPU (t8_gpu.py); HB-MNL + RFC-S (T1's code; sweetness is not a utility attribute).
Shares for decisions: GHK in numpy (M = 2^13 Sobol, exact taste integration); truth c by simulation
(10^6 draws, common random numbers). New-SKU: the held-out SKU never appears in training tasks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "T1"))
import run as T1  # noqa: E402
import t8_gpu  # noqa: E402

SEED = 20261010
T_VAR = T1.T_VAR
P = T1.P
LEVELS = T1.LEVELS
H_EXT = 0.02  # new-SKU extension threshold: net brand incremental share (user, 2026-10-10)
TRUTHS = {
    "a": dict(w=(.08, .45, .45), rbf=0.0, ell=0.3, nug=.02),
    "b": dict(w=(.05, .20, .20), rbf=.40, ell=0.3, nug=.15),
    "c": dict(w=(.05, .20, .20), merge=.55),
}
ARMS = {"kernel": dict(rbf=True), "varshare": dict(rbf=False)}

# --- SKU universe -------------------------------------------------------------------------------
COMBOS = np.array([(b, f, p) for b in range(4) for f in range(3) for p in range(3)])
SWEET = np.random.default_rng(SEED).uniform(size=len(COMBOS))
TERC = np.quantile(SWEET, [1 / 3, 2 / 3])


def sku(b, f, p):
    return int(b * 9 + f * 3 + p)


def prod(b, f, p, price):
    return [b, f, p, price, SWEET[sku(b, f, p)]]


BASE = np.array([prod(*r[:3].astype(int), r[3]) for r in T1.BASE])
CAND = {k: prod(*np.array(v[:3], int), v[3]) for k, v in T1.CAND.items()}
_used = {sku(*r[:3].astype(int)) for r in BASE} | {sku(*np.array(v[:3], int)) for v in T1.CAND.values()}
_free = [j for j in range(len(COMBOS)) if j not in _used]
_gap = [min(abs(SWEET[j] - SWEET[k]) for k in range(len(COMBOS)) if k != j) for j in _free]
HELD = _free[int(np.argmin(_gap))]  # held-out SKU: nearest another in sweetness, not in BASE / CAND
TRAIN_SKUS = np.array([j for j in range(len(COMBOS)) if j != HELD])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- error covariance ---------------------------------------------------------------------------
def tercile(s):
    return np.digitize(s, TERC)


def comps(prods, arm):
    """Fixed components (C, ..., K1, K1) for an arm: brand, flavor, pack, [tercile], nugget."""
    prods = np.asarray(prods, float)
    K = prods.shape[-2]
    K1 = K + 1
    cols = [prods[..., 0], prods[..., 1], prods[..., 2]]
    if arm == "varshare":
        cols.append(tercile(prods[..., 4]))
    out = np.zeros((len(cols) + 1,) + prods.shape[:-2] + (K1, K1))
    for c, v in enumerate(cols):
        out[c, ..., :K, :K] = v[..., :, None] == v[..., None, :]
    out[-1, ..., :K, :K] = np.eye(K)
    out[:, ..., K, K] = 1
    return out


def kernel_E(prods, w, rbf, ell, nug):
    """(K1, K1) error covariance: categorical weights w, RBF on sweetness, nugget; none = total."""
    prods = np.asarray(prods, float)
    K = len(prods)
    E = np.zeros((K + 1, K + 1))
    for a in range(3):
        E[:K, :K] += w[a] * (prods[:, a][:, None] == prods[:, a][None, :])
    s = prods[:, 4]
    E[:K, :K] += rbf * np.exp(-(s[:, None] - s[None, :]) ** 2 / (2 * ell ** 2)) + nug * np.eye(K)
    E[K, K] = sum(w) + rbf + nug
    return T_VAR * E


def model_E(prods, arm, wts, ell=None):
    """Arm's error covariance; wts in the arm's component order (see t8_gpu)."""
    prods = np.asarray(prods, float)
    if arm == "kernel":
        return kernel_E(prods, wts[:3] / T_VAR, wts[3] / T_VAR, ell, wts[4] / T_VAR)
    S = comps(prods, "varshare")
    return np.einsum("c,cij->ij", wts, S)


def merge_p(d):
    return 1 / (1 + np.exp((d - 0.6) / 0.2))


def pair_d(prods):
    prods = np.asarray(prods, float)
    mis = sum((prods[..., a][..., :, None] != prods[..., a][..., None, :]).astype(float) for a in range(3))
    return np.abs(prods[..., 4][..., :, None] - prods[..., 4][..., None, :]) + mis


# --- simulation (training data and truth c shares) ----------------------------------------------
def draw_errors(rng, prods, truth):
    """Errors (R, K1) for products prods (R, K, 5) (one task per row) under a truth."""
    R, K, _ = prods.shape
    tr = TRUTHS[truth]
    e = np.zeros((R, K + 1))
    if "merge" in tr:
        for a, L in enumerate(LEVELS):
            z = np.sqrt(T_VAR * tr["w"][a]) * rng.standard_normal((R, L))
            e[:, :K] += np.take_along_axis(z, prods[..., a].astype(int), 1)
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        dd = pair_d(prods)
        merged = np.stack([rng.random(R) < merge_p(dd[:, i, j]) for i, j in pairs], 1) if pairs else np.zeros((R, 0), bool)
        lab = np.tile(np.arange(K), (R, 1))
        for _ in range(K):
            for q, (i, j) in enumerate(pairs):
                m = merged[:, q]
                lo = np.minimum(lab[:, i], lab[:, j])
                lab[m, i] = lo[m]
                lab[m, j] = lo[m]
        zp = np.sqrt(T_VAR * tr["merge"]) * rng.standard_normal((R, K))
        e[:, :K] += np.take_along_axis(zp, lab, 1)
    else:
        Es = np.stack([kernel_E(p, tr["w"], tr["rbf"], tr["ell"], tr["nug"])[:K, :K] for p in prods])
        L = np.linalg.cholesky(Es + 1e-12 * np.eye(K))
        e[:, :K] = np.einsum("rij,rj->ri", L, rng.standard_normal((R, K)))
    e[:, K] = np.sqrt(T_VAR) * rng.standard_normal(R)
    return e


def random_tasks(rng, n, skus=TRAIN_SKUS, k=3):
    out = np.empty((n, k, 5))
    for i in range(n):
        js = rng.choice(skus, k, replace=False)
        for q, j in enumerate(js):
            out[i, q] = prod(*COMBOS[j], T1.PRICES[rng.integers(0, 5)])
    return out


def holdout_tasks(rng):
    """6 fixed RFC-tuning tasks; the first 2 contain a pair sharing flavor and pack (as T1)."""
    H = random_tasks(rng, 6)
    for i in range(2):
        b0, f0, p0 = H[i, 0, :3].astype(int)
        b1 = (b0 + 1 + rng.integers(0, 3)) % 4
        if sku(b1, f0, p0) == HELD:
            b1 = (b1 + 1) % 4 if (b1 + 1) % 4 != b0 else (b1 + 2) % 4
        H[i, 1] = prod(b1, f0, p0, H[i, 1, 3])
    return H


def simulate(rng, prods, beta, truth):
    N, T, K, _ = prods.shape
    V = np.einsum("ntkp,np->ntk", T1.task_X(prods), beta)
    e = draw_errors(rng, prods.reshape(N * T, K, 5), truth).reshape(N, T, K + 1)
    return np.argmax(V + e, 2)


def make_data(rng, N, T, truth):
    beta = T1.B_TRUE + T1.W_TRUE * rng.standard_normal((N, P))
    prods = random_tasks(rng, N * T).reshape(N, T, 3, 5)
    return dict(prods=prods, y=simulate(rng, prods, beta, truth), beta=beta)


# --- shares ------------------------------------------------------------------------------------
_SOB = {}


def orthant_np(upper, cov, M=2 ** 13):
    """P(X <= upper), X ~ N(0, cov), by GHK with fixed scrambled Sobol (numpy). upper (d,), cov (d, d)."""
    d = len(upper)
    if d not in _SOB:
        _SOB[d] = qmc.Sobol(d, scramble=True, seed=SEED).random(M)
    w = _SOB[d]
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(d))
    p = np.ones(M)
    z = np.zeros((M, d))
    for j in range(d):
        c = (upper[j] - z[:, :j] @ L[j, :j]) / L[j, j]
        pj = ndtr(c)
        p *= pj
        z[:, j] = ndtri(np.clip(w[:, j] * pj, 1e-300, 1 - 1e-16))
    return p.mean()


def pop_shares(prods, b, w, E):
    X = T1.task_X(np.asarray(prods, float))
    mu = X @ b
    C = X @ np.diag(w ** 2) @ X.T + E
    K1 = len(mu)
    p = np.empty(K1)
    for j in range(K1):
        oth = [k for k in range(K1) if k != j]
        Mx = np.zeros((K1 - 1, K1))
        Mx[np.arange(K1 - 1), oth] = 1
        Mx[:, j] = -1
        p[j] = orthant_np(-(Mx @ mu), Mx @ C @ Mx.T)
    return p / p.sum()


def sim_shares(prods, truth, R=10 ** 6, chunk=2 * 10 ** 5):
    """Truth c population shares by simulation, common random numbers (fixed seed per call)."""
    prods = np.asarray(prods, float)
    rng = np.random.default_rng(SEED + 7)
    K = len(prods)
    X = T1.task_X(prods)
    cnt = np.zeros(K + 1)
    for _ in range(R // chunk):
        beta = T1.B_TRUE + T1.W_TRUE * rng.standard_normal((chunk, P))
        e = draw_errors(rng, np.broadcast_to(prods, (chunk, K, 5)), truth)
        cnt += np.bincount(np.argmax(beta @ X.T + e, 1), minlength=K + 1)
    return cnt / cnt.sum()


def truth_share_fn(truth):
    tr = TRUTHS[truth]
    if "merge" in tr:
        return lambda pr: sim_shares(pr, truth)
    return lambda pr: pop_shares(pr, T1.B_TRUE, T1.W_TRUE, kernel_E(pr, tr["w"], tr["rbf"], tr["ell"], tr["nug"]))


def decisions(share_fn):
    """T1's decision layer on the 5-column BASE / CAND, plus the new-SKU extension."""
    out = {}
    prof = []
    for pz in T1.PRICES:
        mk = BASE.copy()
        mk[0, 3] = pz
        prof.append((pz - T1.COST) * share_fn(mk)[0])
    out["price_profit"] = prof
    base = share_fn(BASE)
    ext = share_fn(np.vstack([BASE, CAND["A''"]]))
    out["ext_incr"] = (ext[0] + ext[5] - base[0]) / ext[5]
    contrib = [sum((BASE[j, 3] - T1.COST) * base[j] for j in T1.OWN_COST)]
    for c in CAND.values():
        s = share_fn(np.vstack([BASE, c]))
        contrib.append(sum((BASE[j, 3] - T1.COST) * s[j] for j in T1.OWN_COST) + (c[3] - T1.COST) * s[5])
    out["cost_contrib"] = contrib
    hb = COMBOS[HELD]
    new = np.vstack([BASE, prod(*hb, 4.75)])
    s1 = share_fn(new)
    brand = [j for j in range(len(BASE)) if int(BASE[j, 0]) == hb[0]]
    out["newsku_net_incr"] = float(s1[len(BASE)] + sum(s1[j] for j in brand) - sum(base[j] for j in brand))
    return out


def score(dec, truth):
    s = T1.score(dec, truth)
    s["newsku_add"] = bool(dec["newsku_net_incr"] > H_EXT)
    s["newsku_match"] = bool(s["newsku_add"] == (truth["newsku_net_incr"] > H_EXT))
    s["flips"] = int(s["price_flip"]) + int(s["ext_flip_at_placeholder_h"]) + int(s["cost_flip"])
    return s


def corr_row(j, wfn):
    """Error correlations of SKU j with every other SKU, from wfn(prods (2, 5)) -> (3, 3) cov."""
    out = []
    for k in range(len(COMBOS)):
        if k == j:
            continue
        pr = np.array([prod(*COMBOS[j], 4.0), prod(*COMBOS[k], 4.0)])
        E = wfn(pr)
        out.append(E[0, 1] / np.sqrt(E[0, 0] * E[1, 1]))
    return np.array(out)


def truth_corr(j, truth):
    tr = TRUTHS[truth]
    if "merge" in tr:
        def wfn(pr):
            E = kernel_E(pr, tr["w"], 0.0, 1.0, tr["merge"])
            E[0, 1] = E[1, 0] = E[0, 1] + T_VAR * tr["merge"] * merge_p(pair_d(pr)[0, 1])
            return E
        return corr_row(j, wfn)
    return corr_row(j, lambda pr: kernel_E(pr, tr["w"], tr["rbf"], tr["ell"], tr["nug"]))


# --- fitting ------------------------------------------------------------------------------------
def unpack(th, arm):
    th = np.asarray(th)
    b, w = th[:P], np.exp(th[P:2 * P])
    if arm == "kernel":
        u, ell = th[2 * P:2 * P + 4], float(np.exp(th[-1]))
    else:
        u, ell = th[2 * P:2 * P + 4], None
    z = np.r_[u, 0.0]
    p = np.exp(z - z.max())
    p /= p.sum()
    wts = T_VAR * (p * (1 - 1e-6 * len(p)) + 1e-6)
    return b, w, wts, ell


def th_start(arm):
    return np.r_[np.zeros(P), np.full(P, np.log(.5)), np.zeros(4), [np.log(0.5)] if arm == "kernel" else []]


def th_truth(arm, truth):
    tr = TRUTHS[truth]
    if "merge" in tr or (arm == "varshare" and tr["rbf"] > 0):
        return np.array([])
    if arm == "kernel":
        v = np.array([*tr["w"], max(tr["rbf"], 1e-6), tr["nug"]])
        return np.r_[T1.B_TRUE, np.log(T1.W_TRUE), np.log(v[:4] / v[4]), np.log(tr["ell"])]
    v = np.array([*tr["w"], 1e-6, tr["nug"]])
    return np.r_[T1.B_TRUE, np.log(T1.W_TRUE), np.log(v[:4] / v[4])]


def gpu_jobs(datasets, arm, truth, M, se):
    jobs = []
    for rep, d in datasets:
        Xd, G, Mm, D2 = t8_gpu.panel_arrays(d["prods"], d["y"], T1.task_X, lambda pr: comps(pr, arm),
                                            sweet=(lambda pr: pr[..., 4]) if ARMS[arm]["rbf"] else None)
        jobs.append((Xd, G, Mm, D2, P, ARMS[arm]["rbf"], th_start(arm), th_truth(arm, truth), M, SEED + rep,
                     float(T_VAR), se))
    return jobs


def weight_se(th, V, arm):
    """Delta-method SEs of the variance weights (softmax; floor ignored) and of l."""
    u = np.asarray(th)[2 * P:2 * P + 4]
    z = np.r_[u, 0.0]
    p = np.exp(z - z.max())
    p /= p.sum()
    Jw = T_VAR * (np.diag(p) - np.outer(p, p))[:, :4]           # d w / d u  (5, 4)
    Vu = np.asarray(V)[2 * P:2 * P + 4, 2 * P:2 * P + 4]
    se_w = np.sqrt(np.diag(Jw @ Vu @ Jw.T))
    se_ell = float(np.exp(th[-1]) * np.sqrt(V[-1][-1])) if arm == "kernel" else None
    return se_w, se_ell


# --- stages -------------------------------------------------------------------------------------
def gate(args):
    import modal
    truth, N = "b", 1000
    tr = TRUTHS[truth]
    d = make_data(np.random.default_rng([SEED, 999]), N, 12, truth)
    log(f"gate: truth b, N = {N}, kernel arm, SEs on; held-out SKU {HELD} {COMBOS[HELD].tolist()} "
        f"sweet {SWEET[HELD]:.3f}")
    with modal.enable_output(), t8_gpu.app.run():
        r = t8_gpu.fit.remote(*gpu_jobs([(999, d)], "kernel", truth, 4096, True)[0])
    th = np.array(r["theta"])
    b, w, wts, ell = unpack(th, "kernel")
    se_w, se_ell = weight_se(th, r["cov_sandwich"], "kernel")
    true_w = T_VAR * np.array([*tr["w"], tr["rbf"], tr["nug"]])
    z = (wts - true_w) / se_w
    # implied correlation matrix over the SKU universe (off-diagonal), fitted vs true
    errs = [np.abs(corr_row(j, lambda pr: model_E(pr, "kernel", wts, ell)) - truth_corr(j, truth)).max()
            for j in range(len(COMBOS))]
    out = dict(theta=th.tolist(), weights=wts.tolist(), weights_true=true_w.tolist(), se_weights=se_w.tolist(),
               z_weights=z.tolist(), ell=ell, se_ell=se_ell, ell_true=tr["ell"], corr_max_abs_err=float(max(errs)),
               nll=r["nll"], nll_true=r["nll_true"], n_iter=r["n_iter"], grad_max=r["grad_max"], seconds=r["seconds"],
               PASS=bool(np.all(np.abs(z) <= 2) and max(errs) <= 0.1))
    names = ["brand", "flavor", "pack", "rbf", "nugget"]
    log(f"fit {r['seconds']['fit']:.0f} s + SE {r['seconds']['se']:.0f} s, {r['n_iter']} iters; nll {r['nll']:.1f} "
        f"vs truth {r['nll_true']:.1f}")
    for n_, a, t_, s_, z_ in zip(names, wts, true_w, se_w, z):
        log(f"  {n_:7s} {a / T_VAR:.3f} vs {t_ / T_VAR:.3f} (SE {s_ / T_VAR:.3f}, z {z_:+.2f})")
    log(f"  l {ell:.3f} (SE {se_ell:.3f}) vs {tr['ell']}; corr matrix max |err| {max(errs):.3f}")
    log("GATE " + ("PASS" if out["PASS"] else "FAIL"))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "gate.json").write_text(json.dumps(out, indent=1))


def run_truth(args):
    import modal
    truth = args.truth
    gate_p = HERE / "out" / "gate.json"
    if not args.check and not (gate_p.exists() and json.loads(gate_p.read_text())["PASS"]):
        sys.exit("recovery gate has not passed (BACKLOG GPR: run first)")
    out_dir = HERE / "out" / (f"check_{truth}" if args.check else f"full_{truth}")
    out_dir.mkdir(parents=True, exist_ok=True)
    T = 12
    hb_iters = 2000 if args.check else 20000
    R_l = 100 if args.check else 1000
    reps = range(args.reps)
    data = {}
    for rep in reps:
        r_data, r_hold, r_rfc = [np.random.default_rng(s) for s in np.random.SeedSequence([SEED, rep]).spawn(3)]
        data[rep] = make_data(r_data, args.N, T, truth)
    log(f"T8 {truth}: {len(reps)} reps, N {args.N}; held-out SKU {HELD}; GPU fits for both arms in parallel")
    t0 = time.time()
    jobs, keys = [], []
    for arm in ARMS:
        jobs += gpu_jobs([(r, data[r]) for r in reps], arm, truth, 4096, False)
        keys += [(arm, r) for r in reps]
    with modal.enable_output(), t8_gpu.app.run():
        res = list(t8_gpu.fit.starmap(jobs, return_exceptions=True))
    log(f"GPU fits done: {time.time() - t0:.0f} s wall; GPU fit-seconds {sum(r['seconds']['fit'] for r in res if isinstance(r, dict)):.0f}")
    fits = dict(zip(keys, res))
    tdec = decisions(truth_share_fn(truth))
    tcorr = truth_corr(HELD, truth)
    recs = []
    for rep in reps:
        rec = dict(rep=rep, truth=truth, N=args.N)
        r_data, r_hold, r_rfc = [np.random.default_rng(s) for s in np.random.SeedSequence([SEED, rep]).spawn(3)]
        for arm in ARMS:
            r = fits[(arm, rep)]
            if not isinstance(r, dict):
                rec[arm] = dict(error=repr(r)[:500])
                continue
            b, w, wts, ell = unpack(r["theta"], arm)
            dec = decisions(lambda pr: pop_shares(pr, b, w, model_E(pr, arm, wts, ell)))
            cm = np.abs(corr_row(HELD, lambda pr: model_E(pr, arm, wts, ell)) - tcorr).mean()
            rec[arm] = dict(theta=r["theta"], weights=(wts / T_VAR).tolist(), ell=ell, iters=r["n_iter"],
                            fit_s=r["seconds"]["fit"], newsku_corr_mae=float(cm), **score(dec, tdec))
        # HB-MNL + RFC-S (T1's code); holdout tasks drawn as T1 (fresh stream, same per replicate)
        d = data[rep]
        H = holdout_tasks(r_hold)
        bh = T1.B_TRUE + T1.W_TRUE * r_hold.standard_normal((args.N, P))
        yh = simulate(r_hold, np.broadcast_to(H, (args.N,) + H.shape).copy(), bh, truth)
        obs = np.stack([np.bincount(yh[:, i], minlength=4) / args.N for i in range(6)])
        B, llh, t_hb = T1.run_hb(out_dir, d["prods"], d["y"], hb_iters, 1000 + rep)
        Rf = T1.RFC(B, R_l, r_rfc)
        (sl, g), mae, flags = T1.tune_rfc(Rf, H, obs)
        rdec = decisions(lambda pr: Rf.shares(pr, sl, g))
        rec["rfc_s"] = dict(sigma_l=float(sl), g=float(g), holdout_mae=float(mae), flags=flags, hb_s=t_hb,
                            **score(rdec, tdec))
        recs.append(rec)
        (out_dir / "results.json").write_text(json.dumps(dict(truth_dec={k: np.asarray(v).tolist() for k, v in tdec.items()},
                                                              truth_corr_held=tcorr.tolist(), held=HELD, records=recs),
                                                         indent=1, default=str))
        log(f"rep {rep}: " + "  ".join(f"{a} flips {rec[a].get('flips')} regret {rec[a].get('cost_regret_pct', float('nan')):.2f}%"
                                       for a in ("kernel", "varshare", "rfc_s")))
    log("done")


def summary(args):
    def paired(x, y):
        d = np.asarray(x) - np.asarray(y)
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d))

    out = {}
    for truth in ("a", "b", "c"):
        p = HERE / "out" / f"full_{truth}" / "results.json"
        if not p.exists():
            continue
        R = json.loads(p.read_text())["records"]
        A = {a: dict(regret=np.array([r[a]["cost_regret_pct"] for r in R]), flips=np.array([r[a]["flips"] for r in R]))
             for a in ("kernel", "varshare", "rfc_s")}
        row = {a: dict(regret=float(v["regret"].mean()), flips=int(v["flips"].sum())) for a, v in A.items()}
        kv, kv_se = paired(A["kernel"]["regret"], A["varshare"]["regret"])
        kr, kr_se = paired(A["kernel"]["regret"], A["rfc_s"]["regret"])
        row["kernel_minus_varshare"] = (kv, kv_se)
        row["kernel_minus_rfc"] = (kr, kr_se)
        if truth == "b":
            row["PASS"] = bool(-kv > 2 * kv_se and -kr > 2 * kr_se and row["kernel"]["flips"] <= row["varshare"]["flips"])
        elif truth == "c":
            better = "varshare" if A["varshare"]["regret"].mean() <= A["rfc_s"]["regret"].mean() else "rfc_s"
            m, se = paired(A["kernel"]["regret"], A[better]["regret"])
            row["vs_better"] = (better, m, se)
            row["PASS_recorded"] = bool(m <= se)
        else:
            row["GUARD_PASS"] = bool(kv <= kv_se)
        cm = paired([r["kernel"]["newsku_corr_mae"] for r in R], [r["varshare"]["newsku_corr_mae"] for r in R])
        row["newsku"] = dict(corr_mae_kernel_minus_varshare=cm,
                             corr_better=bool(-cm[0] > 2 * cm[1]),
                             match_kernel=int(sum(r["kernel"]["newsku_match"] for r in R)),
                             match_varshare=int(sum(r["varshare"]["newsku_match"] for r in R)))
        out[truth] = row
        print(f"\n## truth {truth}  (R = {len(R)})")
        for a in ("kernel", "varshare", "rfc_s"):
            print(f"  {a:8s} mean cost regret {row[a]['regret']:.2f}%  flips {row[a]['flips']}")
        print(f"  kernel - varshare {kv:+.2f} (paired SE {kv_se:.2f}); kernel - rfc_s {kr:+.2f} (SE {kr_se:.2f})")
        print(f"  verdict: {({k: v for k, v in row.items() if k in ('PASS', 'PASS_recorded', 'GUARD_PASS', 'vs_better')})}")
        print(f"  new-SKU: {row['newsku']}")
    (HERE / "out" / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "run", "summary"])
    ap.add_argument("--truth", choices=list(TRUTHS), default="b")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--N", type=int, default=600)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    {"gate": gate, "run": run_truth, "summary": summary}[a.stage](a)
