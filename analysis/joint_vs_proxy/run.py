"""Does a structured multivariate probit beat a Sawtooth best-practice proxy on joint patterns?

Same households, trips, items, weekly prices and holdout as analysis/rho_test (its data
pipeline and cross-price screen are imported unchanged). No trip-size covariate in any model.
All four models share: random item intercepts alpha_i ~ N(mu, Sigma_alpha) integrated by
simulated ML over the same scrambled-Halton draws per household; fixed own-price terms; the
cross-price terms kept by the pooled-probit screen (p < 0.20).

  P-faithful  C1 = {pasta, pasta sauce, shredded cheese}: MNL over its 8 combinations
              (7 pattern constants + additive item intercepts + item price terms);
              C2 = {hot dogs, buns}: 4-outcome MNL, built the same way; singletons: binary
              logits. Sigma_alpha block-diagonal by sub-model (separate HB runs).
  P-generous  P-faithful with a full 8x8 Sigma_alpha.
  M-full      binary probits, full Sigma_alpha, within-trip errors
              eps_tm = l0_m f0_t + lC_m fC_t + sqrt(1 - l0_m^2 - lC_m^2) u_tm (13 loadings,
              l0^2 + lC^2 <= 0.98). Trip probability by 7^3 Gauss-Hermite over (f0, fC1, fC2),
              computed nested: given f0 the clusters are independent.
  M-noC0      M-full with l0 = 0.

Holdout (last 2 trips per household): each household's draws are weighted by its training
likelihood, and the 256 pattern probabilities are averaged over draws.

Single entry point, fixed seeds:  python analysis/joint_vs_proxy/run.py [--quick] [--go]
  stage 1  quick mode (100 households, 50 draws, all models, engine cross-check)
  gate     projected run time; stops if > 4 h (rerun with --go to proceed)
  stage 2  real-data fits (the control generators), then fairness controls
  stage 3  real-data holdout metrics (only if both controls pass)
--quick runs stage 1 only. Env: JV_CHUNK (trips per chunk), JV_KERNEL_DTYPE=float32 (probit
kernel precision; default float64). Writes results.csv, controls.csv, SUMMARY.md here.
"""
import csv
import importlib.util
import os
import sys
import time
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtri
from scipy.stats import multivariate_normal, qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("rho_test_run", ROOT / "analysis" / "rho_test" / "run.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from jax.nn import log_sigmoid, sigmoid, softmax  # noqa: E402
from jax.scipy.special import log_ndtr, logsumexp, ndtr  # noqa: E402

SEED = rt.SEED
NAMES, M = rt.NAMES, rt.M
C1, C2, SINGLE = (0, 1, 2), (3, 4), (5, 6, 7)
S1, S2, SS = slice(0, 3), slice(3, 5), slice(5, 8)
MODELS = ["P-faithful", "P-generous", "M-full", "M-noC0"]
R_DRAWS, K_NODES, K_CHECK = 200, 7, 9
QUICK_HH, QUICK_R = 100, 50
N_BOOT, N_REP, N_CROSS = 200, 5, 1000
LAM_MAX = 0.98
BUDGET_H, TARGET_H = 4.0, 2.0
GPU = jax.default_backend() == "gpu"
CHUNK = int(os.environ.get("JV_CHUNK", 1024 if GPU else 512))
# probit kernel precision: float32 on GPU (consumer/datacenter GPUs run float64 ~30-60x slower), float64 on
# CPU; likelihood sums, logsumexp over draws and the optimizer stay float64. Stage 1 logs the difference.
KDT = os.environ.get("JV_KERNEL_DTYPE", "float32" if GPU else "float64")
OPT = dict(maxiter=3000, ftol=1e-8, gtol=1e-5, maxcor=20)  # ftol 1e-8 per trip: ~1e-3 total LL per iteration at full size
HO_CHUNK = 32

BITS = (np.arange(256)[:, None] >> np.arange(M)) & 1
B1 = BITS[:8, :3].astype(float)
B2 = BITS[:4, :2].astype(float)
SG1, SG2 = 2 * B1 - 1, 2 * B2 - 1
SIZE = BITS.sum(1)


def block(m):
    return 1 if m in C1 else 2 if m in C2 else 10 + m


FULL_OFF = np.tril_indices(M, -1)
BLOCK_OFF = tuple(a[[block(i) == block(j) for i, j in zip(*FULL_OFF)]] for a in FULL_OFF)
OFF_IDX = {"P-faithful": BLOCK_OFF, "P-generous": FULL_OFF, "M-full": FULL_OFF, "M-noC0": FULL_OFF}
PAIRS = list(combinations(range(M), 2))
WITHIN = [(0, 1), (0, 2), (1, 2), (0, 1, 2), (3, 4)]
CROSS = [p for p in PAIRS if block(p[0]) != block(p[1]) or block(p[0]) >= 10]


def label(s):
    return " x ".join(NAMES[i] for i in s)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data: the rho-test pipeline, packed into whole-household chunks
# ---------------------------------------------------------------------------
def design(D, cross):
    """Item price terms: own log price (if it varies) + screened cross log prices."""
    cols = [([m] if D["LP"][:, m].std() >= rt.MIN_PRICE_SD else []) + cross[m] for m in range(M)]
    flat = [(m, k) for m in range(M) for k in cols[m]]
    A = np.zeros((len(flat), M))
    A[np.arange(len(flat)), [m for m, _ in flat]] = 1
    return dict(cols=cols, flat=flat, LPX=D["LP"][:, [k for _, k in flat]], A=A)


def subset(D, n_hh):
    keep = D["hh"] < n_hh
    return dict(hh=D["hh"][keep], week=D["week"][keep], y=D["y"][keep], train=D["train"][keep], LP=D["LP"])


def codes(y):
    return (np.asarray(y, int) * (1 << np.arange(M))).sum(1)


def make_data(D, des, R, K=K_NODES, y=None, Tc=CHUNK):
    """Training trips packed into chunks of whole households (padded), Halton draws, nodes. Trips with
    the same household, week and pattern have the same likelihood and enter once, weighted by count."""
    y = D["y"] if y is None else y
    N = int(D["hh"].max()) + 1
    code = codes(y)
    idx = np.flatnonzero(D["train"])
    key = (D["hh"][idx].astype(np.int64) * 10_000 + D["week"][idx]) * 256 + code[idx]
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    uidx = idx[first]
    hh = D["hh"][uidx]
    starts = np.flatnonzero(np.r_[True, hh[1:] != hh[:-1]])
    ends = np.r_[starts[1:], len(hh)]
    Tc = max(Tc, int((ends - starts).max()))
    groups, cur, n = [], [], 0
    for s, e in zip(starts, ends):
        if cur and n + e - s > Tc:
            groups.append(cur)
            cur, n = [], 0
        cur.append((s, e))
        n += e - s
    groups.append(cur)
    nC, Hc = len(groups), max(len(g) for g in groups)
    ch = dict(wk=np.zeros((nC, Tc), int), hl=np.zeros((nC, Tc), int), sg=np.ones((nC, Tc, M)),
              c1=np.zeros((nC, Tc), int), c2=np.zeros((nC, Tc), int), mask=np.zeros((nC, Tc)),
              hid=np.zeros((nC, Hc), int), hmask=np.zeros((nC, Hc)))
    for c, g in enumerate(groups):
        t = 0
        for h, (s, e) in enumerate(g):
            rows = uidx[s:e]
            sl = slice(t, t + e - s)
            ch["wk"][c, sl], ch["hl"][c, sl], ch["mask"][c, sl] = D["week"][rows], h, cnt[s:e]
            ch["sg"][c, sl] = 2 * y[rows] - 1
            ch["c1"][c, sl], ch["c2"][c, sl] = code[rows] & 7, (code[rows] >> 3) & 3
            ch["hid"][c, h], ch["hmask"][c, h] = hh[s], 1
            t += e - s
    U = qmc.Halton(d=M, scramble=True, seed=SEED).random(N * R)
    z, lw = rt.gh(K)
    ho = np.flatnonzero(~D["train"])
    return dict(chunks={k: jnp.asarray(v) for k, v in ch.items()}, Z=jnp.asarray(ndtri(U).reshape(N, R, M)),
                LPX=jnp.asarray(des["LPX"]), A=jnp.asarray(des["A"]), z=jnp.asarray(z), lw=jnp.asarray(lw),
                n_train=len(idx), n_unique=len(uidx), N=N, R=R, Tc=Tc, nC=nC,
                ho_hh=D["hh"][ho], ho_wk=D["week"][ho], ho_code=code[ho],
                tr_hh=D["hh"][idx], tr_wk=D["week"][idx], tr_y=y[idx])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def init_params(model, D, nb):
    tr = D["train"]
    y = D["y"][tr]
    p = dict(beta=np.zeros(nb), Ld=np.full(M, np.log(.5)), Lo=np.zeros(len(OFF_IDX[model][0])))
    if model[0] == "P":
        code = codes(y)
        n1, n2 = np.bincount(code & 7, minlength=8) + .5, np.bincount((code >> 3) & 3, minlength=4) + .5
        p.update(k1=np.log(n1[1:] / n1[0]), k2=np.log(n2[1:] / n2[0]), mu=logit(y[:, SS].mean(0)))
    else:
        p.update(mu=ndtri(y.mean(0)), aC=np.full(5, .3))
        if model == "M-full":
            p["a0"] = np.full(M, .3)
    return p


def mu8(p, model):
    return jnp.concatenate([jnp.zeros(5), p["mu"]]) if model[0] == "P" else p["mu"]


def chol(p, model):
    return jnp.diag(jnp.exp(p["Ld"])).at[OFF_IDX[model]].set(p["Lo"])


def alpha_of(p, model, Z):
    return mu8(p, model) + Z @ chol(p, model).T


def offsets(p, LPX, A):
    return (LPX * p["beta"]) @ A  # (weeks, M)


def loadings(p, model):
    """(l0, lC, 1/sd(u)); lC is the item's own-cluster loading (0 for singletons)."""
    aC = jnp.concatenate([p["aC"], jnp.zeros(3)])
    a0 = p["a0"] if model == "M-full" else jnp.zeros(M)
    d = jnp.sqrt(1 + a0 ** 2 + aC ** 2)
    l0, lC = np.sqrt(LAM_MAX) * a0 / d, np.sqrt(LAM_MAX) * aC / d
    return l0, lC, 1 / jnp.sqrt(1 - l0 ** 2 - lC ** 2)


def pconst(p):
    return jnp.concatenate([jnp.zeros(1), p["k1"]]), jnp.concatenate([jnp.zeros(1), p["k2"]])


def logp(p, model, v, sg, c1, c2, z, lw, kdt="float64"):
    """log P(observed pattern | alpha) per trip and draw. v (T, R, M) item indices, sg (T, 1, M) = 2y - 1."""
    if model[0] == "P":
        K1, K2 = pconst(p)
        U1 = K1 + v[..., S1] @ B1.T
        U2 = K2 + v[..., S2] @ B2.T
        l1 = (U1 * jax.nn.one_hot(c1, 8)[:, None]).sum(-1) - logsumexp(U1, -1)
        l2 = (U2 * jax.nn.one_hot(c2, 4)[:, None]).sum(-1) - logsumexp(U2, -1)
        return l1 + l2 + log_sigmoid(sg[..., SS] * v[..., SS]).sum(-1)
    l0, lC, s = loadings(p, model)
    x, g0, gC = [a.astype(kdt) for a in (sg * v * s, sg * l0 * s, sg * lC * s)]
    z, lw = z.astype(kdt), lw.astype(kdt)
    if model == "M-noC0":
        def clus(sl):
            e = x[:, :, None, sl] + gC[:, :, None, sl] * z[:, None]  # (T, R, Kc, n)
            return logsumexp(log_ndtr(e).sum(-1) + lw, -1)
        out = clus(S1) + clus(S2) + log_ndtr(x[..., SS]).sum(-1)
        return out.astype(jnp.float64)

    def clus(sl):
        e = x[:, :, None, None, sl] + g0[:, :, None, None, sl] * z[:, None, None] + gC[:, :, None, None, sl] * z[:, None]
        return logsumexp(log_ndtr(e).sum(-1) + lw, -1)  # (T, R, K0): fC integrated out
    sing = log_ndtr(x[:, :, None, SS] + g0[:, :, None, SS] * z[:, None]).sum(-1)
    return logsumexp(lw + clus(S1) + clus(S2) + sing, -1).astype(jnp.float64)


def pattern_probs(p, model, v, z, lw):
    """All 256 pattern probabilities per trip and draw, index sum_m y_m 2^m. v (T, R, M)."""
    if model[0] == "P":
        K1, K2 = pconst(p)
        p1 = softmax(K1 + v[..., S1] @ B1.T, -1)[..., None, :]
        p2 = softmax(K2 + v[..., S2] @ B2.T, -1)[..., None, :]
        ps = sigmoid(v[..., SS])[..., None, :]
        wa = jnp.ones(1)
    else:
        l0, lC, s = loadings(p, model)
        w = jnp.exp(lw)
        za, wa = (z, w) if model == "M-full" else (jnp.zeros(1), jnp.ones(1))

        def clus(sl, SG):
            e = v[:, :, None, None, sl] + l0[sl] * za[:, None, None] + lC[sl] * z[:, None]
            q = ndtr(SG * (e * s[sl])[..., None, :]).prod(-1)  # (T, R, Ka, Kc, patterns)
            return (q * w[:, None]).sum(-2)
        p1, p2 = clus(S1, SG1), clus(S2, SG2)
        ps = ndtr((v[:, :, None, SS] + l0[SS] * za[:, None]) * s[SS])
    q = [jnp.stack([1 - ps[..., i], ps[..., i]], -1) for i in range(3)]
    full = jnp.einsum("...ac,...ad,...ae,...af,...ag,a->...gfedc", p1, p2, q[0], q[1], q[2], wa)
    return full.reshape(full.shape[:-5] + (256,))


def chunk_S(p, model, ch, data, kdt):
    a = alpha_of(p, model, data["Z"][ch["hid"]])  # (Hc, R, M)
    v = a[ch["hl"]] + offsets(p, data["LPX"], data["A"])[ch["wk"]][:, None, :]
    lp = logp(p, model, v, ch["sg"][:, None, :], ch["c1"], ch["c2"], data["z"], data["lw"], kdt) * ch["mask"][:, None]
    return jax.ops.segment_sum(lp, ch["hl"], num_segments=ch["hid"].shape[0])  # (Hc, R)


def chunk_nll(p, model, ch, data, kdt):
    S = chunk_S(p, model, ch, data, kdt)
    return -(ch["hmask"] * (logsumexp(S, 1) - jnp.log(S.shape[1]))).sum()


@partial(jax.jit, static_argnums=(0, 3))
def value_grad(model, p, data, kdt=KDT):
    vals, grads = jax.lax.map(lambda ch: jax.value_and_grad(chunk_nll)(p, model, ch, data, kdt), data["chunks"])
    return vals.sum(), jax.tree.map(lambda g: g.sum(0), grads)


@partial(jax.jit, static_argnums=(0, 3))
def nll_only(model, p, data, kdt=KDT):
    return jax.lax.map(lambda ch: chunk_nll(p, model, ch, data, kdt), data["chunks"]).sum()


@partial(jax.jit, static_argnums=(0, 3))
def S_chunks(model, p, data, kdt=KDT):
    return jax.lax.map(lambda ch: chunk_S(p, model, ch, data, kdt), data["chunks"])


@partial(jax.jit, static_argnums=0)
def predict_chunk(model, p, Zt, wt, wk, LPX, A, z, lw):
    v = alpha_of(p, model, Zt) + offsets(p, LPX, A)[wk][:, None, :]
    return jnp.einsum("trk,tr->tk", pattern_probs(p, model, v, z, lw), wt)


def bounds_for(p):
    lo = {k: np.full(np.shape(v), -30.) for k, v in p.items()}
    hi = {k: np.full(np.shape(v), 30.) for k, v in p.items()}
    lo["Ld"], hi["Ld"] = np.full(M, -5.), np.full(M, 2.)
    return list(zip(np.asarray(ravel_pytree(lo)[0]), np.asarray(ravel_pytree(hi)[0])))


def fit(model, data, p0, tag):
    p0 = {k: jnp.asarray(v, jnp.float64) for k, v in p0.items()}
    x0, unravel = ravel_pytree(p0)
    n, t0, calls = data["n_train"], time.time(), [0]

    def f(x):
        val, g = value_grad(model, unravel(jnp.asarray(x)), data)
        calls[0] += 1
        if calls[0] % 200 == 0:
            log(f"  {tag}: {calls[0]} evaluations, nll/trip {float(val) / n:.7f}, {time.time() - t0:.0f}s")
        return float(val) / n, np.asarray(ravel_pytree(g)[0]) / n

    r = minimize(f, np.asarray(x0), jac=True, method="L-BFGS-B", bounds=bounds_for(p0), options=OPT)
    gmax = float(np.abs(r.jac).max())
    out = dict(p={k: np.asarray(v) for k, v in unravel(jnp.asarray(r.x)).items()}, nll=r.fun * n,
               ok=bool(r.success or gmax < 1e-4), nfev=r.nfev, secs=time.time() - t0, gmax=gmax, msg=str(r.message))
    log(f"{tag}: nll {out['nll']:.2f}, {r.nfev} evaluations, {out['secs']:.0f}s, |grad|max/trip {gmax:.1e}"
        + ("" if out["ok"] else f"  NOT CONVERGED ({r.message})"))
    return out


def n_params(model, nb):
    p = init_params(model, dict(train=np.ones(2, bool), y=np.ones((2, M), bool)), nb)
    return sum(np.size(v) for v in p.values())


ORDER = ["P-faithful", "P-generous", "M-noC0", "M-full"]  # fit order: warm starts follow it


def start_for(model, D, nb, fits):
    """P-generous starts at the P-faithful fit, M-full at the M-noC0 fit (l0 = 0.29; 0 is a saddle)."""
    p0 = init_params(model, D, nb)
    if model == "P-generous":
        pf = fits["P-faithful"]["p"]
        p0.update({k: pf[k] for k in ("k1", "k2", "mu", "beta", "Ld")})
        Lo = np.zeros((M, M))
        Lo[BLOCK_OFF] = pf["Lo"]
        p0["Lo"] = Lo[FULL_OFF]
    elif model == "M-full":
        p0 = dict(fits["M-noC0"]["p"], a0=p0["a0"])
    return p0


def fit_four(data, D, nb, starts=None, tag=""):
    fits = {}
    for model in ORDER:
        p0 = starts[model] if starts is not None else start_for(model, D, nb, fits)
        fits[model] = fit(model, data, p0, f"{tag}{model}")
    return {m: fits[m] for m in MODELS}


def jp(p):
    return {k: jnp.asarray(v, jnp.float64) for k, v in p.items()}


# ---------------------------------------------------------------------------
# Holdout prediction and metrics
# ---------------------------------------------------------------------------
def posterior(model, p, data):
    """(N, R) weights: each household's draws weighted by its training likelihood."""
    S = np.asarray(S_chunks(model, jp(p), data))
    hid, hm = np.asarray(data["chunks"]["hid"]), np.asarray(data["chunks"]["hmask"]) > 0
    out = np.zeros((data["N"], data["R"]))
    out[hid[hm]] = S[hm]
    w = np.exp(out - out.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)


def predict_holdout(model, p, data):
    """(holdout trips, 256) pattern probabilities, posterior-weighted over draws."""
    W, hh, wk = posterior(model, p, data), data["ho_hh"], data["ho_wk"]
    n, out = len(hh), []
    for s in range(0, n, HO_CHUNK):
        i = np.minimum(np.arange(s, s + HO_CHUNK), n - 1)
        r = predict_chunk(model, jp(p), data["Z"][hh[i]], jnp.asarray(W[hh[i]]), jnp.asarray(wk[i]),
                          data["LPX"], data["A"], data["z"], data["lw"])
        out.append(np.asarray(r)[:min(HO_CHUNK, n - s)])
    return np.concatenate(out)


def ci(a):
    return tuple(np.percentile(a[1:], [2.5, 97.5]))


def holdout_metrics(P, data):
    """Per model: point value (index 0) and N_BOOT household-bootstrap values (1:) of every metric.
    The bootstrap resamples households in the holdout scoring only (no refitting)."""
    code, hh, N = data["ho_code"], data["ho_hh"], data["N"]
    rng = np.random.default_rng([SEED, 1])
    cnt = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(N_BOOT)])
    W = np.vstack([np.ones(len(code)), cnt[:, hh]]).astype(float)  # (1 + B, trips)
    tot = W.sum(1)
    feat = {S: BITS[:, list(S)].prod(1) for S in PAIRS + [(0, 1, 2)]}
    obs_m = W @ BITS[code] / tot[:, None]
    obs_j = {S: W @ f[code] / tot for S, f in feat.items()}
    n_both = {S: W @ f[code] for S, f in feat.items()}
    top5 = sorted(CROSS, key=lambda S: -n_both[S][0])[:5]
    obs_size = np.bincount(SIZE[code], minlength=M + 1) / len(code)
    with np.errstate(invalid="ignore", divide="ignore"):  # an item absent from a resample: its pairs have weight 0
        obs_lift = {S: obs_j[S] / np.prod([obs_m[:, i] for i in S], 0) for S in feat}
    res = dict(_obs=dict(m=obs_m, j=obs_j, lift=obs_lift, n_both=n_both, size=obs_size, top5=top5))
    for model, Pm in P.items():
        pm = W @ (Pm @ BITS) / tot[:, None]
        pj = {S: W @ (Pm @ f) / tot for S, f in feat.items()}
        lift = {S: pj[S] / np.prod([pm[:, i] for i in S], 0) for S in feat}
        err = {S: np.where(n_both[S] > 0, np.abs(lift[S] - obs_lift[S]), 0.) for S in CROSS}  # weight-0 pairs
        wsum = sum(n_both[S] for S in CROSS)
        size = np.bincount(SIZE, weights=Pm.mean(0), minlength=M + 1)
        res[model] = dict(ll=W @ np.log(Pm[np.arange(len(code)), code]), m=pm, j=pj, lift=lift, err=err,
                          wmae=sum(n_both[S] * err[S] for S in CROSS) / wsum,
                          top5=np.mean([err[S] for S in top5], 0), size=size,
                          tv=.5 * np.abs(size - obs_size).sum())
    return res


def decision(res):
    d = {k: res[a]["ll"] - res[b]["ll"] for k, a, b in (("full_pf", "M-full", "P-faithful"),
                                                       ("full_pg", "M-full", "P-generous"),
                                                       ("noc_pg", "M-noC0", "P-generous"))}
    werr_pf = res["M-full"]["wmae"] - res["P-faithful"]["wmae"]
    impr = 1 - res["M-full"]["wmae"][0] / res["P-generous"]["wmae"][0]
    lo_pf, hi_pf = ci(d["full_pf"])
    lo_pg, hi_pg = ci(d["full_pg"])
    gain, kept = d["full_pg"][0], d["noc_pg"][0]
    abl_loses = gain > 0 and kept < .5 * gain
    beats_pf, beats_pg, ties_pg = lo_pf > 0, lo_pg > 0, lo_pg <= 0 <= hi_pg
    ties_pf = lo_pf <= 0 <= hi_pf and ci(werr_pf)[0] <= 0 <= ci(werr_pf)[1]
    facts = (f"M-full - P-generous holdout LL {gain:+.1f} [{lo_pg:+.1f}, {hi_pg:+.1f}]; M-full - P-faithful "
             f"{d['full_pf'][0]:+.1f} [{lo_pf:+.1f}, {hi_pf:+.1f}]; cross-cluster lift error vs P-generous "
             f"{100 * impr:+.0f}% (improvement); M-noC0 keeps {kept:+.1f} of the {gain:+.1f} gain over P-generous")
    if beats_pg and impr >= .25 and abl_loses:
        call = "Edge = occasion factor"
    elif beats_pf and (ties_pg or (gain > 0 and not abl_loses and impr < .25)):
        call = "Edge = cluster structure only"
    elif ties_pf:
        call = "Workflow only"
    else:
        call = "None of the three rules holds"
    return call, facts, dict(d=d, werr_pf=werr_pf, impr=impr, abl_loses=abl_loses)


# ---------------------------------------------------------------------------
# Checks: pattern assembly, 9^3 nodes, orthant engine
# ---------------------------------------------------------------------------
def consistency(fits, data, n=64):
    """logp of the observed pattern equals log of the assembled 256-pattern probability."""
    hh, wk, y = data["tr_hh"][:n], data["tr_wk"][:n], data["tr_y"][:n]
    code = codes(y)
    worst = 0.
    for model in MODELS:
        p = jp(fits[model]["p"])
        v = alpha_of(p, model, data["Z"][hh]) + offsets(p, data["LPX"], data["A"])[wk][:, None, :]
        a = logp(p, model, v, jnp.asarray(2. * y - 1)[:, None], jnp.asarray(code & 7), jnp.asarray((code >> 3) & 3),
                 data["z"], data["lw"])
        pp = np.asarray(pattern_probs(p, model, v, data["z"], data["lw"]))
        worst = max(worst, np.abs(np.asarray(a) - np.log(pp[np.arange(n), :, code])).max(), np.abs(pp.sum(-1) - 1).max())
    return worst


def gh_check(p, D, des):
    """Training nll per trip at the fitted M-full parameters, 9^3 vs 7^3 nodes, first 100 households."""
    Dq = subset(D, QUICK_HH)
    d7, d9 = make_data(Dq, des, QUICK_R, K_NODES), make_data(Dq, des, QUICK_R, K_CHECK)
    return abs(float(nll_only("M-full", jp(p), d9, "float64")) - float(nll_only("M-full", jp(p), d7, "float64"))) / d7["n_train"]


def kernel_check(fits, data):
    """max over M models of |nll(float32 kernel) - nll(float64 kernel)| per trip at the fitted parameters."""
    return max(abs(float(nll_only(m, jp(fits[m]["p"]), data, "float32")) - float(nll_only(m, jp(fits[m]["p"]), data, "float64")))
               for m in ("M-full", "M-noC0")) / data["n_train"]


def engine():
    try:
        for line in open(ROOT / ".env"):
            k, _, v = line.strip().partition("=")
            if k and not k.startswith("#"):
                os.environ.setdefault(k, v.strip().strip("'\""))
    except OSError:
        pass
    try:
        from multivariate_probit import orthant
    except ImportError:
        return None
    return orthant if getattr(orthant, "tier", "") == "paid" else None


def crosscheck(p, data):
    """1,000 random training trips (one random draw each): the 7^3 conditioned probability vs the
    orthant engine on the full 8-dim orthant (resolution 'high'); SciPy on the 20 largest gaps."""
    eng = engine()
    rng = np.random.default_rng([SEED, 2])
    t = rng.choice(data["n_train"], N_CROSS, replace=False)
    r = rng.integers(0, data["R"], N_CROSS)
    hh, wk, y = data["tr_hh"][t], data["tr_wk"][t], data["tr_y"][t]
    pj = jp(p)
    v = np.asarray(alpha_of(pj, "M-full", data["Z"][hh, r])) + np.asarray(offsets(pj, data["LPX"], data["A"]))[wk]
    sg = 2. * y - 1
    zero = jnp.zeros(N_CROSS, int)

    def gh(K):
        z, lw = rt.gh(K)
        return np.exp(np.asarray(logp(pj, "M-full", jnp.asarray(v[:, None]), jnp.asarray(sg[:, None]), zero, zero,
                                      jnp.asarray(z), jnp.asarray(lw))))[:, 0]
    g7, g9 = gh(K_NODES), gh(K_CHECK)
    out = dict(gh9_vs_gh7=float(np.abs(g9 - g7).max()), engine=eng is not None, n=N_CROSS)
    if eng is None:
        return out
    l0, lC, _ = (np.asarray(a) for a in loadings(pj, "M-full"))
    same = np.array([[block(i) == block(j) for j in range(M)] for i in range(M)])
    Rm = np.outer(l0, l0) + np.outer(lC, lC) * same
    np.fill_diagonal(Rm, 1)
    cov = sg[:, :, None] * Rm * sg[:, None, :]
    e = np.asarray(eng.cdf(sg * v, cov, resolution="high"), float)
    d = np.abs(e - g7)
    worst = np.argsort(d)[-20:]
    ref = np.array([multivariate_normal(np.zeros(M), cov[i]).cdf(sg[i] * v[i]) for i in worst])
    out.update(max_abs=float(d.max()), median_abs=float(np.median(d)), max_rel=float((d / g7).max()),
               scipy_vs_gh7=float(np.abs(ref - g7[worst]).max()), scipy_vs_engine=float(np.abs(ref - e[worst]).max()))
    return out


def fmt_cross(x):
    if not x:
        return "not run"
    if not x.get("engine"):
        return f"engine not available on this machine (9^3 vs 7^3 max |d| {x['gh9_vs_gh7']:.1e}); run --crosscheck"
    return (f"max |d| {x['max_abs']:.1e} (median {x['median_abs']:.1e}, max relative {x['max_rel']:.1e}) over "
            f"{x['n']} trips; on the 20 largest gaps SciPy differs from 7^3 by {x['scipy_vs_gh7']:.1e} and from "
            f"the engine by {x['scipy_vs_engine']:.1e}; 9^3 vs 7^3 max |d| {x['gh9_vs_gh7']:.1e}")


# ---------------------------------------------------------------------------
# Fairness controls
# ---------------------------------------------------------------------------
def simulate(model, p, D, des, rng):
    """Patterns for every trip (training and holdout) from a fitted generator, fresh alpha draws."""
    pj = jp(p)
    N = int(D["hh"].max()) + 1
    a = np.asarray(mu8(pj, model)) + rng.standard_normal((N, M)) @ np.asarray(chol(pj, model)).T
    v = a[D["hh"]] + np.asarray(offsets(pj, jnp.asarray(des["LPX"]), jnp.asarray(des["A"])))[D["week"]]
    n = len(v)
    if model[0] == "P":
        K1, K2 = (np.asarray(k) for k in pconst(pj))
        c1 = np.argmax(K1 + v[:, S1] @ B1.T + rng.gumbel(size=(n, 8)), 1)
        c2 = np.argmax(K2 + v[:, S2] @ B2.T + rng.gumbel(size=(n, 4)), 1)
        return np.column_stack([B1[c1], B2[c2], rng.random((n, 3)) < expit(v[:, SS])]).astype(bool)
    l0, lC, s = (np.asarray(x) for x in loadings(pj, model))
    f = rng.standard_normal((n, 3))
    fC = np.where(np.isin(np.arange(M), C1), f[:, [1]], f[:, [2]])
    return v + l0 * f[:, [0]] + lC * fC + rng.standard_normal((n, M)) / s > 0


def loading_table(p):
    """Sign-normalized loadings (f0 and fC1 signs are not identified). In a 2-item cluster only
    the product lC_hotdogs * lC_buns is identified; the individual |lC2| are reported, not tested."""
    l0, lC, _ = (np.asarray(x).copy() for x in loadings(jp(p), "M-full"))
    l0 *= np.sign(l0.sum()) or 1
    lC[:3] *= np.sign(lC[:3].sum()) or 1
    out = {f"l0 {NAMES[m]}": (l0[m], True) for m in range(M)}
    out.update({f"lC1 {NAMES[m]}": (lC[m], True) for m in C1})
    out["lC2 hot dogs * lC2 buns"] = (lC[3] * lC[4], True)
    out.update({f"|lC2 {NAMES[m]}| (not identified)": (abs(lC[m]), False) for m in C2})
    return out


def run_controls(real, D, des, nb):
    rows, verdict = [], {}
    truth = loading_table(real["M-full"]["p"])
    for c, gen in (("a", "P-generous"), ("b", "M-full")):
        diffs, wins, est = [], [], []
        for rep in range(N_REP):
            t0 = time.time()
            y = simulate(gen, real[gen]["p"], D, des, np.random.default_rng([SEED, ord(c), rep]))
            data = make_data(D, des, R_DRAWS, y=y)
            fits = fit_four(data, D, nb, starts={m: real[m]["p"] for m in MODELS}, tag=f"control ({c}) rep {rep} ")
            res = holdout_metrics({m: predict_holdout(m, fits[m]["p"], data) for m in MODELS}, data)
            for m in MODELS:
                dd = res[m]["ll"] - res["P-generous"]["ll"]
                lo, hi = ci(dd) if m != "P-generous" else (np.nan, np.nan)
                rows.append(dict(control=c, generator=gen, rep=rep, row="holdout_ll", name=m, value=res[m]["ll"][0],
                                 diff_vs_P_generous=dd[0], ci_lo=lo, ci_hi=hi, fit_ok=fits[m]["ok"]))
            d = res["M-full"]["ll"] - res["P-generous"]["ll"]
            diffs.append((d[0],) + ci(d))
            wins.append([res["M-full"]["ll"][0] - res[m]["ll"][0] for m in MODELS if m != "M-full"])
            if c == "b":
                est.append(loading_table(fits["M-full"]["p"]))
                for k, (val, _) in est[-1].items():
                    rows.append(dict(control=c, generator=gen, rep=rep, row="loading", name=k, value=val,
                                     truth=truth[k][0], fit_ok=fits["M-full"]["ok"]))
            log(f"control ({c}) rep {rep}: M-full - P-generous holdout LL {d[0]:+.1f} [{diffs[-1][1]:+.1f}, "
                f"{diffs[-1][2]:+.1f}]; fits ok {all(f['ok'] for f in fits.values())}; {time.time() - t0:.0f}s")
        if c == "a":
            n_win = sum(lo > 0 for _, lo, _ in diffs)
            verdict[c] = dict(passed=n_win == 0, text=f"M-full's CI over P-generous excluded 0 in {n_win} of {N_REP} "
                                                      f"replications (pass: 0)")
        else:
            bias = {k: np.mean([e[k][0] for e in est]) - truth[k][0] for k in truth}
            for k, b in bias.items():
                rows.append(dict(control=c, generator=gen, rep="mean", row="loading_bias", name=k,
                                 value=np.mean([e[k][0] for e in est]), truth=truth[k][0], bias=b,
                                 tested=truth[k][1]))
            mb = max(abs(b) for k, b in bias.items() if truth[k][1])
            mw = np.mean(wins, 0)
            ok = mb < .05 and (mw > 0).all()
            verdict[c] = dict(passed=bool(ok), text=f"max |bias| over the 12 identified loading quantities {mb:.3f} "
                              f"(pass: < 0.05); mean holdout LL of M-full minus "
                              + ", ".join(f"{m} {w:+.1f}" for m, w in zip([m for m in MODELS if m != 'M-full'], mw))
                              + " (pass: all > 0)")
        log(f"control ({c}): {'PASS' if verdict[c]['passed'] else 'FAIL'}: {verdict[c]['text']}")
    return rows, verdict


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
RES_COLS = ["model", "metric", "subject", "predicted", "observed", "value", "ci_lo", "ci_hi", "n"]
CTRL_COLS = ["control", "generator", "rep", "row", "name", "value", "truth", "bias", "tested",
             "diff_vs_P_generous", "ci_lo", "ci_hi", "fit_ok"]


def g(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.6g}" if isinstance(x, (float, np.floating)) else x)


def write_csv(name, cols, rows):
    with open(HERE / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: g(r.get(c)) for c in cols})


def result_rows(st):
    res, fits, o = st["res"], st["fits"], st["res"]["_obs"]
    rows = []
    for m in MODELS:
        r = res[m]
        rows.append(dict(model=m, metric="1 holdout LL (256 patterns)", subject="all", value=r["ll"][0]))
        for ref in ("P-faithful", "P-generous"):
            if m != ref:
                d = r["ll"] - res[ref]["ll"]
                rows.append(dict(model=m, metric=f"1 LL minus {ref}", subject="all", value=d[0],
                                 ci_lo=ci(d)[0], ci_hi=ci(d)[1]))
        for S in WITHIN:
            rows.append(dict(model=m, metric="2 within-cluster rate", subject=label(S), predicted=r["j"][S][0],
                             observed=o["j"][S][0], n=o["n_both"][S][0]))
            rows.append(dict(model=m, metric="2 within-cluster lift", subject=label(S), predicted=r["lift"][S][0],
                             observed=o["lift"][S][0], value=abs(r["lift"][S][0] - o["lift"][S][0])))
        for S in CROSS:
            rows.append(dict(model=m, metric="3 cross-cluster lift", subject=label(S), predicted=r["lift"][S][0],
                             observed=o["lift"][S][0], value=r["err"][S][0], n=o["n_both"][S][0]))
        rows.append(dict(model=m, metric="3 cross-cluster lift error (weighted MAE)", subject="24 pairs",
                         value=r["wmae"][0]))
        for ref in ("P-faithful", "P-generous"):
            if m != ref:
                d = r["wmae"] - res[ref]["wmae"]
                rows.append(dict(model=m, metric=f"3 weighted MAE minus {ref}", subject="24 pairs", value=d[0],
                                 ci_lo=ci(d)[0], ci_hi=ci(d)[1]))
        rows.append(dict(model=m, metric="3 cross-cluster lift error (top-5 MAE)", subject="; ".join(map(label, o["top5"])),
                         value=r["top5"][0]))
        for s in range(M + 1):
            rows.append(dict(model=m, metric="4 basket size", subject=s, predicted=r["size"][s], observed=o["size"][s]))
        rows.append(dict(model=m, metric="4 basket size TV distance", subject="0-8", value=r["tv"]))
        f = fits[m]
        for k, v in (("parameters", st["npar"][m]), ("fit seconds", f["secs"]), ("fit evaluations", f["nfev"]),
                     ("fit converged", f["ok"]), ("training nll", f["nll"])):
            rows.append(dict(model=m, metric=f"5 {k}", subject="real data", value=v))
    x = st["cross"]
    if x.get("engine"):
        rows.append(dict(model="M-full", metric="5 engine cross-check max |d|", subject=f"{x['n']} trips",
                         value=x["max_abs"]))
    rows.append(dict(model="M-full", metric="5 9^3 vs 7^3 max |d| (cross-check trips)", subject=f"{x['n']} trips",
                     value=x["gh9_vs_gh7"]))
    rows.append(dict(model="M-full", metric="5 9^3 vs 7^3 training nll per trip", subject=f"{QUICK_HH} households",
                     value=st["k9"]))
    return rows


FINDINGS = "(to be written from the tables above)"


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_summary(st):
    L = ["# Structured multivariate probit vs a Sawtooth best-practice proxy on joint patterns", "",
         f"Generated by `run.py` (seed {SEED}; JAX {jax.default_backend()}, probit kernel {KDT}; "
         f"{st['secs'] / 60:.0f} min). Wall clock per stage: "
         + ", ".join(f"{k} {v / 60:.1f} min" for k, v in st["stages"].items()) + ".", ""]
    v = st["verdict"]
    L += ["## Fairness controls", "",
          table(["control", "generator", "result", "detail"],
                [(c, g_, "PASS" if v[c]["passed"] else "FAIL", v[c]["text"])
                 for c, g_ in (("a", "P-generous"), ("b", "M-full"))]), ""]
    if not all(x["passed"] for x in v.values()):
        L += ["**A control failed; the real-data results are not interpreted** (results.csv is empty).", ""]
        (HERE / "SUMMARY.md").write_text("\n".join(L) + "\n" + methods(st))
        return
    res, o = st["res"], st["res"]["_obs"]
    call, facts, _ = decision(res)
    L += [f"## Decision: **{call}**", "", facts + ".", "", FINDINGS, "", f"## Metrics (holdout, {st['n_holdout']:,} trips)", ""]

    def cil(m, ref):
        if m == ref:
            return "-"
        d = res[m]["ll"] - res[ref]["ll"]
        return f"{d[0]:+.1f} [{ci(d)[0]:+.1f}, {ci(d)[1]:+.1f}]"
    f = st["fits"]
    L += [table(["metric"] + MODELS, [
        ["1 holdout LL (256 patterns)"] + [f"{res[m]['ll'][0]:.1f}" for m in MODELS],
        ["1 LL minus P-faithful [95% CI]"] + [cil(m, "P-faithful") for m in MODELS],
        ["1 LL minus P-generous [95% CI]"] + [cil(m, "P-generous") for m in MODELS],
        ["2 within-cluster mean abs lift error (5 joints)"]
        + [f"{np.mean([abs(res[m]['lift'][S][0] - o['lift'][S][0]) for S in WITHIN]):.3f}" for m in MODELS],
        ["3 cross-cluster lift error, weighted MAE (24 pairs)"] + [f"{res[m]['wmae'][0]:.3f}" for m in MODELS],
        ["3 cross-cluster lift error, top-5 pairs MAE"] + [f"{res[m]['top5'][0]:.3f}" for m in MODELS],
        ["4 basket-size TV distance"] + [f"{res[m]['tv']:.4f}" for m in MODELS],
        ["5 parameters"] + [str(st["npar"][m]) for m in MODELS],
        ["5 fit wall clock (s) / evaluations"] + [f"{f[m]['secs']:.0f} / {f[m]['nfev']}" for m in MODELS],
        ["5 fit converged"] + [str(f[m]["ok"]) for m in MODELS]]), "",
        "Cross-cluster weighted MAE minus P-faithful / P-generous [95% CI]: "
        + "; ".join(f"{m} {d[0]:+.3f} [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}] / {e[0]:+.3f} [{ci(e)[0]:+.3f}, {ci(e)[1]:+.3f}]"
                    for m in MODELS[1:] for d, e in [(res[m]["wmae"] - res["P-faithful"]["wmae"],
                                                      res[m]["wmae"] - res["P-generous"]["wmae"])]) + ".", "",
        "### Within-cluster joints (rate / lift)", "",
        table(["joint", "holdout trips with all", "observed"] + MODELS,
              [[label(S), int(o["n_both"][S][0]), f"{o['j'][S][0]:.4f} / {o['lift'][S][0]:.2f}"]
               + [f"{res[m]['j'][S][0]:.4f} / {res[m]['lift'][S][0]:.2f}" for m in MODELS] for S in WITHIN]), "",
        "### Five most frequent cross-cluster pairs (lift)", "",
        table(["pair", "holdout trips with both", "observed"] + MODELS,
              [[label(S), int(o["n_both"][S][0]), f"{o['lift'][S][0]:.2f}"]
               + [f"{res[m]['lift'][S][0]:.2f}" for m in MODELS] for S in o["top5"]]), "",
        "### Basket size (number of the 8 items on the trip)", "",
        table(["size", "observed"] + MODELS,
              [[s, f"{o['size'][s]:.4f}"] + [f"{res[m]['size'][s]:.4f}" for m in MODELS] for s in range(M + 1)]
              + [["TV", "-"] + [f"{res[m]['tv']:.4f}" for m in MODELS]]), "",
        "### Ablation (M-noC0 = M-full without the occasion factor)", ""]
    _, _, dd = decision(res)
    d = dd["d"]
    L += [f"Holdout LL over P-generous: M-full {d['full_pg'][0]:+.1f} [{ci(d['full_pg'])[0]:+.1f}, "
          f"{ci(d['full_pg'])[1]:+.1f}], M-noC0 {d['noc_pg'][0]:+.1f} [{ci(d['noc_pg'])[0]:+.1f}, "
          f"{ci(d['noc_pg'])[1]:+.1f}]. Cross-cluster weighted MAE: M-full {res['M-full']['wmae'][0]:.3f}, "
          f"M-noC0 {res['M-noC0']['wmae'][0]:.3f}, P-generous {res['P-generous']['wmae'][0]:.3f}. Basket-size TV: "
          f"M-full {res['M-full']['tv']:.4f}, M-noC0 {res['M-noC0']['tv']:.4f}. Removing the occasion factor "
          + ("loses most of the gain." if dd["abl_loses"] else "does not lose most of the gain."), "",
          "Fitted M-full loadings (sign-normalized): "
          + ", ".join(f"{k} {val:+.2f}" for k, (val, _) in loading_table(st["fits"]["M-full"]["p"]).items()) + ".", ""]
    (HERE / "SUMMARY.md").write_text("\n".join(L) + "\n" + methods(st))


def methods(st):
    D = st["data_desc"]
    return "\n".join([
        "## Method", "",
        f"- Data: the rho-test pipeline imported unchanged ({D}). Prices weekly, the same for every household; "
        "promotion-driven, not randomized.",
        "- Clusters: C1 = pasta, pasta sauce, shredded cheese; C2 = hot dogs, buns; singletons refrigerated OJ, "
        "shelf-stable juice, pet food. No trip-size covariate in any model.",
        f"- Price terms (all models): own log price where it varies, plus the cross log prices kept by the rho test's "
        f"pooled binary probit screen at p < 0.20: {st['price_terms']}.",
        f"- Heterogeneity (all models): alpha_i ~ N(mu, Sigma_alpha), Cholesky-parameterized; simulated ML with the same "
        f"{R_DRAWS} scrambled-Halton draws per household (seed {SEED}) for every model. P models: the pattern constants "
        "carry the cluster items' means (their alpha has mean 0); P-faithful's Sigma_alpha is block-diagonal "
        "(C1 3x3, C2 2x2, singletons 1x1).",
        f"- M models: loadings parameterized so l0^2 + lC^2 < {LAM_MAX}; {K_NODES}^3 Gauss-Hermite nodes, nested "
        f"(given f0 the clusters are independent). 9^3 vs 7^3 on the first {QUICK_HH} households ({QUICK_R} draws): "
        f"training nll differs by {st['k9']:.1e} per trip.",
        f"- Engine cross-check: {fmt_cross(st['cross'])}.",
        f"- Probit kernel computed in {KDT}; float32 vs float64 training nll at the fitted M parameters differs by "
        f"{st['kcheck']:.1e} per trip. Likelihood sums, draws and the optimizer are float64.",
        "- Optimizer: L-BFGS-B on the per-trip negative log-likelihood, JAX gradients. Fit order P-faithful -> "
        "P-generous (warm start) and M-noC0 -> M-full (warm start, l0 = 0.29; l0 = 0 is a stationary point). "
        "Converged = optimizer success or max |gradient| per trip < 1e-4.",
        f"- Holdout: last {rt.N_HOLDOUT} trips per household. Each household's draws are weighted by its training "
        "likelihood; pattern probabilities are averaged over draws. Metric 1 is the sum over holdout trips of log "
        f"P(observed 8-item pattern). CIs: {N_BOOT} household-bootstrap resamples of the holdout scoring, no refitting, "
        "percentile intervals.",
        "- Lift = P(joint) / product of marginals, predicted (means of predicted probabilities over holdout trips) vs "
        "observed. The weighted MAE weights each cross-cluster pair by its holdout trips with both items; the top 5 "
        "pairs are the most frequent by that count.",
        f"- Controls: {N_REP} replications each; every trip's pattern (training and holdout) simulated from the real-data "
        "fit of the generator with fresh alpha draws, same households, trips and prices; all four models refit "
        "(started at their real-data fits). (a) passes if M-full's 95% CI over P-generous never lies above 0. "
        "(b) passes if every identified loading quantity has |mean bias| < 0.05 and M-full's mean holdout LL beats "
        "each other model. In C2 (two items) only lC_hotdogs * lC_buns is identified, so that product is tested and "
        "the individual C2 loadings are only reported.",
        "- Decision rules as specified. 'Ties' = 95% CI includes 0. 'Cross-cluster lift error improves by >= 25%' is "
        "M-full vs P-generous on the weighted MAE. 'M-noC0 loses most of that gain' = M-noC0 keeps < half of M-full's "
        "holdout-LL gain over P-generous.",
        ""])


def write_all(st):
    write_csv("controls.csv", CTRL_COLS, st["ctrl_rows"])
    passed = all(x["passed"] for x in st["verdict"].values())
    write_csv("results.csv", RES_COLS, result_rows(st) if passed else [])
    write_summary(st)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def projection(data, nfev):
    """Seconds for the remaining fits: one timed full-size evaluation per model x quick-mode evaluations."""
    t = {}
    for m in MODELS:
        p = jp(init_params(m, dict(train=np.ones(2, bool), y=np.eye(2, M, dtype=bool)), data["LPX"].shape[1]))
        value_grad(m, p, data)[0].block_until_ready()
        t0 = time.time()
        value_grad(m, p, data)[0].block_until_ready()
        t[m] = time.time() - t0
    total = sum(t[m] * nfev[m] for m in MODELS) * (1 + 2 * N_REP)
    log("full-size evaluation: " + ", ".join(f"{m} {t[m]:.2f}s" for m in MODELS))
    return total


STATE = HERE / "fit_state.pkl"


def main():
    import pickle
    t_start = time.time()
    log(f"JAX backend {jax.default_backend()}, probit kernel {KDT}, chunk {CHUNK} trips")
    if "--crosscheck" in sys.argv:  # engine check after a run on a machine without the engine
        st = pickle.loads(STATE.read_bytes())
        D = rt.load()
        st["cross"] = crosscheck(st["fits"]["M-full"]["p"], make_data(D, st["des"], R_DRAWS))
        log(f"engine cross-check: {fmt_cross(st['cross'])}")
        write_all(st)
        STATE.write_bytes(pickle.dumps(st))
        return 0
    D = rt.load()
    cross, _ = rt.cross_screen(D)
    des = design(D, cross)
    nb = len(des["flat"])
    price_terms = "; ".join(f"{NAMES[m]}: " + (", ".join(NAMES[k] for k in des["cols"][m]) or "none") for m in range(M))
    log(f"data: {len(D['hh'])} trips, {D['train'].sum()} training; price terms {price_terms}")
    stages = {}

    # stage 1: quick mode
    t0 = time.time()
    Dq = subset(D, QUICK_HH)
    dq = make_data(Dq, des, QUICK_R)
    qf = fit_four(dq, Dq, nb, tag="quick ")
    cons = consistency(qf, dq)
    rq = holdout_metrics({m: predict_holdout(m, qf[m]["p"], dq) for m in MODELS}, dq)
    log(f"quick: pattern assembly vs observed-pattern likelihood max |d| {cons:.1e}")
    for m in MODELS:
        log(f"quick {m}: holdout LL {rq[m]['ll'][0]:.1f}, cross-cluster wMAE {rq[m]['wmae'][0]:.3f}, "
            f"basket TV {rq[m]['tv']:.4f}, converged {qf[m]['ok']}")
    log(f"quick: {decision(rq)[1]}")
    log("quick M-full loadings: " + ", ".join(f"{k} {v:+.3f}" for k, (v, _) in loading_table(qf["M-full"]["p"]).items()))
    if os.environ.get("JV_SAVE_QUICK"):
        import pickle
        Path(os.environ["JV_SAVE_QUICK"]).write_bytes(pickle.dumps(qf))
    xq = crosscheck(qf["M-full"]["p"], dq)
    log(f"quick engine cross-check: {fmt_cross(xq)}")
    log(f"quick: 9^3 vs 7^3 training nll per trip {gh_check(qf['M-full']['p'], D, des):.1e}")
    log(f"quick: float32 vs float64 probit kernel, training nll per trip {kernel_check(qf, dq):.1e}")
    stages["1 quick mode"] = time.time() - t0
    if cons > 1e-8 or not all(f["ok"] for f in qf.values()):
        log("quick mode found a problem (see above); stopping")
        return 2
    if "--quick" in sys.argv:
        log(f"quick mode done in {(time.time() - t_start) / 60:.1f} min")
        return 0

    # projection gate
    data = make_data(D, des, R_DRAWS)
    log(f"{data['n_train']} training trips enter as {data['n_unique']} distinct (household, week, pattern) rows "
        f"in {data['nC']} chunks of {data['Tc']}")
    proj = projection(data, {m: qf[m]["nfev"] for m in MODELS})
    log(f"projected remaining time {proj / 3600:.1f} h (target {TARGET_H} h, stop above {BUDGET_H} h)")
    if proj > BUDGET_H * 3600 and "--go" not in sys.argv:
        log("projection exceeds the budget; stopping before the long stages (rerun with --go to proceed)")
        return 3

    # stage 2: real-data fits (control generators), then controls
    t0 = time.time()
    real = fit_four(data, D, nb, tag="real ")
    stages["2a real-data fits"] = time.time() - t0
    t0 = time.time()
    ctrl_rows, verdict = run_controls(real, D, des, nb)
    stages["2b fairness controls"] = time.time() - t0
    st = dict(fits=real, ctrl_rows=ctrl_rows, verdict=verdict, stages=stages, des=des, price_terms=price_terms,
              npar={m: n_params(m, nb) for m in MODELS}, cross={}, k9=np.nan, kcheck=np.nan, n_holdout=int((~D["train"]).sum()),
              data_desc=f"{int(D['hh'].max()) + 1} households, {len(D['hh'])} trips, {D['train'].sum()} training, "
                        f"{(~D['train']).sum()} holdout")
    if all(x["passed"] for x in verdict.values()):
        # stage 3: real-data holdout metrics
        t0 = time.time()
        st["res"] = holdout_metrics({m: predict_holdout(m, real[m]["p"], data) for m in MODELS}, data)
        st["cross"] = crosscheck(real["M-full"]["p"], data)
        st["k9"] = gh_check(real["M-full"]["p"], D, des)
        st["kcheck"] = kernel_check(real, data)
        stages["3 real-data metrics and checks"] = time.time() - t0
        log(f"decision: {decision(st['res'])[0]}: {decision(st['res'])[1]}")
    else:
        log("a control failed; real-data results not interpreted")
    st["secs"] = time.time() - t_start
    write_all(st)
    STATE.write_bytes(pickle.dumps(st))
    log(f"done in {st['secs'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
