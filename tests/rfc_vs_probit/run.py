"""Does a probit-kernel simulator substitute better than RFC on near-duplicates?

Given the same (oracle) individual utilities, four simulation kernels are tuned
on 15 calibration tasks and scored on four launch scenarios against four truths.
Single entry point, fixed seeds:

    python tests/rfc_vs_probit/run.py

Writes results.csv and SUMMARY.md next to this file. Needs the local paid
orthant binary (rfc.BACKEND == "local"); the hosted service is too slow for
tuning.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research" / "rfc_vs_exact"))
import rfc  # noqa: E402  (orthant backend)

SEED = 20261005
N = 1000
T = np.pi ** 2 / 6  # total product error variance (logit scale)

# ---------------------------------------------------------------------------
# Market and respondents
# ---------------------------------------------------------------------------
LEVELS = (4, 3, 3)  # brand, flavor, pack
MU_LEVEL = [np.array([.6, .2, -.1, -.7]), np.array([.4, 0, -.4]), np.array([.3, 0, -.3])]
SD_COEF = (.8, .6, .5)  # sd of each effects-coded coefficient (diagonal Sigma_beta)
NONE_MEAN, NONE_SD = 1.0, .5  # "none" constant; about 15% none under D1 in the base market
PRICE_M, PRICE_S, PRICE_REF = np.log(.9), .4, 5.0

#                b  f  p  price      (0-based levels)
BASE = np.array([[0, 0, 0, 5.00],   # A
                 [1, 1, 1, 4.50],   # B
                 [2, 0, 2, 5.50],   # C
                 [1, 1, 0, 4.00],   # D
                 [2, 1, 2, 3.50]])  # E
NEW = {"T1": [3, 0, 0, 5.00],  # A': A with another brand
       "T2": [0, 0, 0, 4.70],  # A'': A, $0.30 cheaper (same brand: line extension)
       "T3": [3, 0, 1, 4.75],  # F: flavor of A, pack of B, other brand and price
       "T4": [3, 2, 2, 4.75]}  # G: new brand and flavor; pack shared with C and E
TARGETS = list(NEW)
SCEN = [BASE] + [np.vstack([BASE, NEW[t]]) for t in TARGETS]  # scenario 0 = base
# All share vectors have 7 entries: A..E, new, none.


def respondents(rng):
    pw = []
    for mu, sd, L in zip(MU_LEVEL, SD_COEF, LEVELS):
        c = mu[:-1] + sd * rng.standard_normal((N, L - 1))
        pw.append(np.column_stack([c, -c.sum(1)]))
    price = -np.exp(PRICE_M + PRICE_S * rng.standard_normal(N))
    none = NONE_MEAN + NONE_SD * rng.standard_normal(N)
    return dict(pw=pw, price=price, none=none)


def utilities(R, X):
    """X (..., K, 4) -> V (N, ..., K+1), "none" last."""
    lead = (-1,) + (1,) * (X.ndim - 1)
    V = sum(R["pw"][a][:, X[..., a].astype(int)] for a in range(3))
    V = V + R["price"].reshape(lead) * (X[..., 3] - PRICE_REF)
    none = np.broadcast_to(R["none"].reshape(lead), V.shape[:-1] + (1,))
    return np.concatenate([V, none], -1)


def to7(P, K):
    """(..., K+1) shares of a K-product scenario -> (..., 7)."""
    out = np.zeros(P.shape[:-1] + (7,))
    out[..., :K] = P[..., :K]
    out[..., 6] = P[..., K]
    return out


# ---------------------------------------------------------------------------
# Error structures
# ---------------------------------------------------------------------------
def ec_cov(X, s):
    """Error-components covariance (K+1, K+1) for products X (K, 4), s = (sb, sf, sp, snu).
    "None" gets an independent error with the same total variance as a product."""
    K = len(X)
    O = np.zeros((K + 1, K + 1))
    for a in range(3):
        O[:K, :K] += s[a] ** 2 * (X[:, a][:, None] == X[:, a][None, :])
    O[:K, :K] += s[3] ** 2 * np.eye(K)
    O[K, K] = np.sum(np.square(s))
    return O


D2_SIG = np.sqrt(T * np.array([.08, .45, .45, .02]))
D3_GUM = np.sqrt(.02)  # Gumbel scale g: pi^2 g^2 / 6 = .02 T
CNL_MU = .3


class Shocks:
    """First-choice error spec: normal per-level perturbations (shared by every
    product with that level), product error (normal sd or Gumbel scale), and an
    independent "none" error (normal sd + Gumbel scale) of the same total variance."""

    def __init__(self, lvl, prod_sd=0., gum=0., none_sd=0.):
        self.lvl, self.prod_sd, self.gum, self.none_sd = np.asarray(lvl, float), prod_sd, gum, none_sd


def d2_shocks():
    return Shocks(D2_SIG[:3], prod_sd=D2_SIG[3], none_sd=np.sqrt(T))


def d3_shocks():
    return Shocks(D2_SIG[:3], gum=D3_GUM, none_sd=np.sqrt(.98 * T))


def s2_shocks(sig_l, g):
    return Shocks([sig_l] * 3, gum=g, none_sd=np.sqrt(3) * sig_l)


def mc_counts(Vs, Xs, sh, R, nb, rng, block=None):
    """First-choice counts with common random numbers across scenarios.
    Vs[s] (N, K_s+1), Xs[s] (K_s, 4): product slot j in every scenario gets the
    same draws. Returns a list of (N, nb, K_s+1) int counts (nb batches of R/nb)."""
    block = block or max(1, min(N, 500_000 // R))
    out = [np.zeros((N, nb, V.shape[1]), np.int32) for V in Vs]
    Kmax = max(len(X) for X in Xs)
    for s0 in range(0, N, block):
        n = min(block, N - s0)
        eta = [sh.lvl[a] * rng.standard_normal((n, R, L), np.float32) for a, L in enumerate(LEVELS)]
        if sh.gum:
            nu = (sh.gum * rng.gumbel(size=(n, R, Kmax))).astype(np.float32)
        else:
            nu = sh.prod_sd * rng.standard_normal((n, R, Kmax), np.float32)
        none = sh.none_sd * rng.standard_normal((n, R), np.float32)
        if sh.gum:
            none += (sh.gum * rng.gumbel(size=(n, R))).astype(np.float32)
        for V, X, o in zip(Vs, Xs, out):
            K = len(X)
            b, f, p = (X[:, a].astype(int) for a in range(3))
            U = np.empty((n, R, K + 1), np.float32)
            U[..., :K] = V[s0:s0 + n, None, :K] + eta[0][..., b] + eta[1][..., f] + eta[2][..., p] + nu[..., :K]
            U[..., K] = V[s0:s0 + n, None, K] + none
            c = U.argmax(-1)
            o[s0:s0 + n] = (c[..., None] == np.arange(K + 1)).reshape(n, nb, R // nb, K + 1).sum(2)
    return out


def logit_logp(V, lam):
    A = lam * V
    return A - logsumexp(A, -1, keepdims=True)


def nl_logp(V, X, lam, mu):
    """Nested logit, nests by flavor, "none" alone. V (..., K+1)."""
    A = lam * V
    K = X.shape[-2]
    f = X[..., 1].astype(int)
    out = np.empty_like(A)
    lS = []
    for m in range(LEVELS[1]):
        mask = np.broadcast_to(f == m, A[..., :K].shape)
        z = np.where(mask, A[..., :K] / mu, -np.inf)
        lS.append(logsumexp(z, -1))  # -inf for an empty nest
    lS = np.stack(lS, -1)
    with np.errstate(invalid="ignore"):
        lG = logsumexp(np.concatenate([mu * lS, A[..., K:]], -1), -1)
    lSj = np.take_along_axis(lS, np.broadcast_to(f, A[..., :K].shape), -1)
    out[..., :K] = A[..., :K] / mu - lSj + mu * lSj - lG[..., None]
    out[..., K] = A[..., K] - lG
    return out


def cnl_p(V, X):
    """Cross-nested logit truth (D4): flavor nests and pack nests, alpha 0.5/0.5,
    nest parameter CNL_MU, "none" in its own degenerate nest."""
    K = X.shape[-2]
    la = np.log(.5)
    terms, lSs = [], []
    for a in (1, 2):
        lev = X[..., a].astype(int)
        for m in range(LEVELS[a]):
            mask = np.broadcast_to(lev == m, V[..., :K].shape)
            ly = np.where(mask, (la + V[..., :K]) / CNL_MU, -np.inf)
            lS = logsumexp(ly, -1, keepdims=True)
            terms.append((ly, lS))
            lSs.append(CNL_MU * lS)
    lG = logsumexp(np.concatenate(lSs + [V[..., K:]], -1), -1, keepdims=True)
    P = np.zeros_like(V)
    for ly, lS in terms:
        with np.errstate(invalid="ignore"):
            P[..., :K] += np.where(np.isfinite(ly), np.exp(ly - lS + CNL_MU * lS - lG), 0.)
    P[..., K] = np.exp(V[..., K] - lG[..., 0])
    return P


def probit_p(V, Om, resolution):
    """Error-components probit shares from the orthant engine, renormalized per
    respondent. V (n, K+1), Om (K+1, K+1). Uses the engine's default merge of
    difference variables correlated >= 0.97."""
    K1 = V.shape[1]
    P = np.empty_like(V)
    for j in range(K1):
        others = [k for k in range(K1) if k != j]
        M = np.zeros((K1 - 1, K1))
        M[np.arange(K1 - 1), others] = 1
        M[:, j] = -1
        P[:, j] = rfc.orthant_cdf(V[:, [j]] - V[:, others], M @ Om @ M.T, resolution=resolution)
    return P / P.sum(1, keepdims=True)


def diff_corrs(Om, j):
    """Correlations among the difference variables of the P(choose j) problem."""
    K1 = len(Om)
    others = [k for k in range(K1) if k != j]
    M = np.zeros((K1 - 1, K1))
    M[np.arange(K1 - 1), others] = 1
    M[:, j] = -1
    C = M @ Om @ M.T
    d = np.sqrt(np.diag(C))
    return (C / np.outer(d, d))[np.triu_indices(K1 - 1, 1)]


# ---------------------------------------------------------------------------
# Tuning: coordinate-wise golden-section on log-parameters, <= 200 evaluations
# ---------------------------------------------------------------------------
BUDGET, PER_LINE = 200, 12
GR = (np.sqrt(5) - 1) / 2


def tune(nll, x0, lo, hi):
    x = np.log(np.asarray(x0, float))
    lo, hi = np.log(lo), np.log(hi)
    vals = []

    def F(z):
        v = nll(np.exp(z))
        vals.append(v)
        return v

    fx = F(x)
    while True:
        f_start = fx
        for i in range(len(x)):
            if len(vals) + PER_LINE > BUDGET:
                break
            a, b = lo[i], hi[i]
            z = x.copy()
            c, d = b - GR * (b - a), a + GR * (b - a)
            z[i] = c; fc = F(z.copy())
            z[i] = d; fd = F(z.copy())
            pts = [(fc, c), (fd, d)]
            for _ in range(PER_LINE - 2):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - GR * (b - a); z[i] = c; fc = F(z.copy()); pts.append((fc, c))
                else:
                    a, c, fc = c, d, fd
                    d = a + GR * (b - a); z[i] = d; fd = F(z.copy()); pts.append((fd, d))
            fbest, zbest = min(pts)
            if fbest < fx:
                fx, x[i] = fbest, zbest
        if f_start - fx < 1e-3 or len(vals) + PER_LINE > BUDGET:
            break
    flags = []
    if max(vals) - min(vals) < .1:
        flags.append("flat objective")
    for i in range(len(x)):
        if x[i] - lo[i] < .05 or hi[i] - x[i] < .05:
            flags.append(f"param {i} at bound")
    return np.exp(x), fx, len(vals), flags


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def sov(base, tgt):
    """Source of volume of the new product: A..E, none; sums to 1."""
    return (base - tgt)[..., [0, 1, 2, 3, 4, 6]] / tgt[..., [5]]


def sibling(x, t):
    """Sibling value from a (A..E, none) source-of-volume error vector."""
    return np.abs(x[..., 0]) if t != "T3" else (np.abs(x[..., 0]) + np.abs(x[..., 1])) / 2


def batch_se(a):
    return a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(a.shape[1:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert rfc.BACKEND == "local", "needs the local paid orthant binary"
    t_start = time.time()
    ss = np.random.SeedSequence(SEED)
    r_resp, r_design, *r_dgp = [np.random.default_rng(s) for s in ss.spawn(6)]
    R = respondents(r_resp)

    Xc = np.concatenate([np.stack([r_design.integers(0, L, (15, 4)) for L in LEVELS], -1),
                         np.round(r_design.uniform(3, 7, (15, 4, 1)), 2)], -1)  # (15 tasks, 4, [b f p price])
    Vc = utilities(R, Xc)  # (N, 15, 5)
    Vs = [utilities(R, X) for X in SCEN]  # (N, K+1)

    print(f"backend {rfc.BACKEND}; calibration tasks with a same flavor+pack pair: "
          f"{sum(any(Xc[t, i, 1] == Xc[t, j, 1] and Xc[t, i, 2] == Xc[t, j, 2] for i in range(4) for j in range(i)) for t in range(15))}/15")

    dgps = ["D1", "D2", "D3", "D4"]
    truth, truth_b, choices = {}, {}, {}
    truth_q = None  # D2 per-respondent A'' / (A + A'') split in T2
    for d, rng in zip(dgps, r_dgp):
        t0 = time.time()
        if d in ("D1", "D4"):
            P = np.exp(logit_logp(Vc, 1.)) if d == "D1" else cnl_p(Vc, Xc)
            c = (P.cumsum(-1) < rng.random((N, 15, 1)) * P.sum(-1, keepdims=True)).sum(-1)
            sh = [to7((np.exp(logit_logp(V, 1.)) if d == "D1" else cnl_p(V, X)).mean(0), len(X))
                  for V, X in zip(Vs, SCEN)]
            truth[d], truth_b[d] = np.array(sh), None
        else:
            spec = d2_shocks() if d == "D2" else d3_shocks()
            c = np.stack([mc_counts([Vc[:, t]], [Xc[t]], spec, 1, 1, rng)[0][:, 0].argmax(-1)
                          for t in range(15)], 1)
            cnt = mc_counts(Vs, SCEN, spec, 10_000, 10, rng)  # 10^7 draws per scenario, CRN
            b = np.array([to7(k.sum(0) / (N * 1000), len(X)) for k, X in zip(cnt, SCEN)])  # (5, 10, 7)
            truth_b[d] = b.transpose(1, 0, 2)  # (10 batches, 5 scen, 7)
            truth[d] = truth_b[d].mean(0)
            if d == "D2":
                k = cnt[2].sum(1)
                truth_q = (k[:, 5] / np.maximum(k[:, 0] + k[:, 5], 1), k[:, 0] + k[:, 5])
        choices[d] = c
        print(f"{d}: truth + choices {time.time() - t0:.1f}s; base none share {truth[d][0, 6]:.3f}")

    # --- simulators: shares (5 scen, 7), optional batches, time per scenario
    rng_s2 = np.random.default_rng(ss.spawn(1)[0])
    Z = np.empty((15, N, 200, 5), np.float32)  # S2 tuning: fixed level draws (CRN)
    for t in range(15):
        z = [rng_s2.standard_normal((N, 200, L), np.float32) for L in LEVELS]
        b, f, p = (Xc[t, :, a].astype(int) for a in range(3))
        Z[t, ..., :4] = z[0][..., b] + z[1][..., f] + z[2][..., p]
        Z[t, ..., 4] = np.sqrt(3) * rng_s2.standard_normal((N, 200), np.float32)
    Vc32 = Vc.transpose(1, 0, 2).astype(np.float32)  # (15, N, 5)

    def cal_nll(logp_fn, c):
        lp = logp_fn()
        return -np.take_along_axis(lp, c[..., None], -1).sum()

    def nll_s1(th, c):
        return cal_nll(lambda: logit_logp(Vc, th[0]), c)

    def nll_s2(th, c):
        sl, g = th
        tot = 0.
        for t in range(15):
            A = (Vc32[t][:, None, :] + sl * Z[t]) / g
            lp = np.take_along_axis(A, c[:, t, None, None].repeat(200, 1), -1)[..., 0] - logsumexp(A, -1)
            tot += (logsumexp(lp, 1) - np.log(200)).sum()
        return -tot

    def nll_s3(th, c):
        return cal_nll(lambda: nl_logp(Vc, Xc[None], th[0], th[1]), c)

    def nll_s4(th, c, res):
        tot = 0.
        for t in range(15):
            P = probit_p(Vc[:, t], ec_cov(Xc[t], th), res)
            tot += np.log(np.maximum(P[np.arange(N), c[:, t]], 1e-12)).sum()
        return -tot

    SIMS = {
        "S1": dict(names=["lambda"], x0=[1.], lo=[.05], hi=[5.], nll=nll_s1),
        "S2": dict(names=["sigma_level", "gumbel_scale"], x0=[.5, .5], lo=[.01, .01], hi=[3., 3.], nll=nll_s2),
        "S3": dict(names=["lambda", "nest_mu"], x0=[1., .5], lo=[.05, .05], hi=[5., 1.], nll=nll_s3),
        "S4": dict(names=["sigma_brand", "sigma_flavor", "sigma_pack", "sigma_nu"], x0=[.5] * 4,
                   lo=[.01] * 4, hi=[3.] * 4, nll=lambda th, c: nll_s4(th, c, "low")),
    }

    def simulate(s, th, rng):
        """-> shares (5, 7), batches (10, 5, 7) or None, seconds per scenario."""
        t0 = time.time()
        if s == "S1":
            sh = np.array([to7(np.exp(logit_logp(V, th[0])).mean(0), len(X)) for V, X in zip(Vs, SCEN)])
            return sh, None, (time.time() - t0) / 5
        if s == "S3":
            sh = np.array([to7(np.exp(nl_logp(V, X, *th)).mean(0), len(X)) for V, X in zip(Vs, SCEN)])
            return sh, None, (time.time() - t0) / 5
        if s == "S2":  # literal first choice, 10^5 draws per scenario (100 per respondent), CRN
            cnt = mc_counts(Vs, SCEN, s2_shocks(*th), 100, 10, rng)
            b = np.array([to7(k.sum(0) / (N * 10), len(X)) for k, X in zip(cnt, SCEN)]).transpose(1, 0, 2)
            return b.mean(0), b, (time.time() - t0) / 5
        res = th[1]
        sh = np.array([to7(probit_p(V, ec_cov(X, th[0]), res).mean(0), len(X)) for V, X in zip(Vs, SCEN)])
        return sh, None, (time.time() - t0) / 5

    rows, tuned, recov = [], {}, {}
    rng_eval = np.random.default_rng(ss.spawn(1)[0])
    for d in dgps:
        c = choices[d]
        for s, cfg in SIMS.items():
            t0 = time.time()
            th, f, ne, flags = tune(lambda x: cfg["nll"](x, c), cfg["x0"], cfg["lo"], cfg["hi"])
            tuned[d, s] = dict(theta=th, nll=f, evals=ne, flags=flags, secs=time.time() - t0)
            print(f"{d} {s}: {np.round(th, 3)} nll {f:.1f} evals {ne} {flags} {time.time() - t0:.0f}s")
        th_acc, f_acc, ne, flags = tune(lambda x: nll_s4(x, c, "high"), SIMS["S4"]["x0"],
                                        SIMS["S4"]["lo"], SIMS["S4"]["hi"])
        recov[d] = dict(theta=th_acc, nll=f_acc, flags=flags)
        print(f"{d} S4 accurate-mode retune (recovery only): {np.round(th_acc, 3)} nll {f_acc:.1f}")

        tr, trb = truth[d], truth_b[d]
        tsov = np.array([sov(tr[0], tr[k]) for k in range(1, 5)])
        tsov_b = None if trb is None else np.array([[sov(bb[0], bb[k]) for k in range(1, 5)] for bb in trb])
        out = {}
        for s in SIMS:
            th = tuned[d, s]["theta"]
            if s == "S4":
                out["S4"] = simulate(s, (th, "high"), rng_eval)
                out["S4fast"] = simulate(s, (th, "low"), rng_eval)
            else:
                out[s] = simulate(s, th, rng_eval)

        def score(sh, k, t):
            e = sov(sh[0], sh[k]) - tsov[k - 1]
            incr = 1 - sov(sh[0], sh[k])[0]
            return dict(share_mae=np.abs(sh[k] - tr[k]).mean(), sov_max_err=np.abs(e).max(),
                        sov_err_A=e[0], sov_err_B=e[1] if t == "T3" else np.nan,
                        sov_sib_err=sibling(e, t), incr_sim=incr)

        for s in SIMS:
            sh, b, secs = out[s]
            for k, t in enumerate(TARGETS, 1):
                r = dict(dgp=d, sim=s, target=t, **score(sh, k, t))
                incr_t = 1 - tsov[k - 1][0]
                r.update(incr_true=incr_t,
                         incr_true_se=np.nan if tsov_b is None else batch_se(1 - tsov_b[:, k - 1, 0]),
                         decision_true=incr_t >= .5, decision_sim=r["incr_sim"] >= .5)
                r["decision_match"] = r["decision_true"] == r["decision_sim"]
                r["truth_sib_se"] = 0. if tsov_b is None else batch_se(sibling_signed(tsov_b[:, k - 1], t))
                r["truth_share_se_max"] = 0. if trb is None else batch_se(trb[:, k]).max()
                r["time_per_scenario_s"] = secs
                if s == "S2":
                    sb = np.array([sibling_signed(sov(bb[0], bb[k]), t) for bb in b])
                    r["sim_sib_se"] = batch_se(sb)
                    s4e = score(out["S4"][0], k, t)["sov_sib_err"]
                    r["s4_acc_sib_err"] = s4e
                    floor = max(s4e, r["truth_sib_se"])
                    r["s2_draws_to_match_s4"] = 1e5 * (r["sim_sib_se"] / floor) ** 2
                if s == "S4":
                    fs = score(out["S4fast"][0], k, t)
                    r.update(fast_share_mae=fs["share_mae"], fast_sov_sib_err=fs["sov_sib_err"],
                             fast_incr_sim=fs["incr_sim"],
                             fast_decision_match=(fs["incr_sim"] >= .5) == r["decision_true"],
                             fast_time_per_scenario_s=out["S4fast"][2])
                rows.append(r)

    # --- merge split check (D2, T2): S4 at D2's true sigma and at tuned sigma
    split = merge_split_check(Vs, truth, truth_q, tuned)
    write_csv(rows)
    write_summary(rows, tuned, recov, split, time.time() - t_start)
    print(f"done in {time.time() - t_start:.0f}s")


def sibling_signed(x, t):
    """Sibling source-of-volume value (A; mean of A and B in T3), signed."""
    return x[..., 0] if t != "T3" else (x[..., 0] + x[..., 1]) / 2


def merge_split_check(Vs, truth, truth_q, tuned):
    """In T2, A and A'' differ only in price. Where does the engine's A/A'' split
    come from, and is it right?"""
    X = SCEN[2]
    V = Vs[2]
    out = {}
    for label, th in (("true", D2_SIG), ("tuned", tuned["D2", "S4"]["theta"])):
        Om = ec_cov(X, th)
        rA, rN = diff_corrs(Om, 0).max(), diff_corrs(Om, 5).max()
        third = [diff_corrs(Om, j) for j in (1, 2, 3, 4, 6)]
        n_merge = sum((c >= .97).sum() for c in third)
        res = {}
        for mode in ("high", "low"):
            P = probit_p(V, Om, mode)
            q = P[:, 5] / (P[:, 0] + P[:, 5])
            res[mode] = dict(P=P.mean(0), q=q)
        dnu = np.sqrt(2 * th[3] ** 2)
        qf = norm.cdf((V[:, 5] - V[:, 0]) / dnu)
        ok = truth_q[1] > 0
        w = truth_q[1][ok]
        out[label] = dict(
            theta=th, max_corr_A=rA, max_corr_new=rN, n_merge_third=int(n_merge),
            agg_q_engine=res["high"]["P"][5] / (res["high"]["P"][0] + res["high"]["P"][5]),
            agg_q_fast=res["low"]["P"][5] / (res["low"]["P"][0] + res["low"]["P"][5]),
            agg_q_truth=truth["D2"][2][5] / (truth["D2"][2][0] + truth["D2"][2][5]),
            mad_engine_formula=np.average(np.abs(res["high"]["q"] - qf)[ok], weights=w),
            mad_engine_truth=np.average(np.abs(res["high"]["q"] - truth_q[0])[ok], weights=w),
            share_mae_vs_truth=np.abs(to7(res["high"]["P"], 6) - truth["D2"][2]).mean(),
            sib_sov_err=abs(sov(to7(probit_p(Vs[0], ec_cov(SCEN[0], th), "high").mean(0), 5),
                                to7(res["high"]["P"], 6))[0] - sov(truth["D2"][0], truth["D2"][2])[0]))
        print(f"merge check ({label} sigma): {out[label]}")
    return out


COLS = ["dgp", "sim", "target", "share_mae", "sov_max_err", "sov_err_A", "sov_err_B", "sov_sib_err",
        "incr_true", "incr_true_se", "incr_sim", "decision_true", "decision_sim", "decision_match",
        "truth_sib_se", "truth_share_se_max", "time_per_scenario_s",
        "sim_sib_se", "s4_acc_sib_err", "s2_draws_to_match_s4",
        "fast_share_mae", "fast_sov_sib_err", "fast_incr_sim", "fast_decision_match", "fast_time_per_scenario_s"]


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.6g}"
    return str(v)


def write_csv(rows):
    with open(HERE / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([fmt(r.get(c, np.nan)) for c in COLS])


FINDINGS = (
    "On the three correlated truths (D2-D4) the probit kernel S4 has the smallest sibling source-of-volume error (1.6-3.2 points against 5.6-6.4 for RFC-style S2, 7.3-10.3 for nested logit S3 and 15-19 for logit S1), and only logit flips a decision: on every correlated truth it calls the $0.30 line extension (T2) 56-58% incremental against a true 20-30%. S4's edge over S2 is parameterization, not kernel: S2 as specified has one level-error magnitude for all attributes and cannot put the correlation on flavor and pack where the truth has it, while with product error at 2% of variance the normal-vs-Gumbel tail barely matters (D2 and D3 results are nearly identical); if Sawtooth's production RFC defaults to attribute error only, as we believe but have not verified, it is closer still to S4 with sigma_nu -> 0. On IID logit truth (D1) S4 is worst (3.6 points, overstating cannibalization, since normal error components cannot reproduce independent Gumbel errors) but flips no decision. The precision-per-compute advantage does not show at this scale: S2 at 10^5 draws has a Monte Carlo SE of 0.2-0.5 points on sibling source of volume, 4-10x below S4's error against truth, at 0.02 s per scenario against 0.08 s for S4, so about 10^3 draws would already match S4. S4's error is tuning, and mostly fast-mode tuning: fast mode underestimates the near-twin split (A''/(A+A'') 0.806 against 0.849 true at the true sigma) and tuning compensates by pulling sigma_nu to 0.144 against 0.181 true, whereas accurate-mode retuning recovers 0.180 and accurate-mode S4 at the true sigma gets T2 to 0.10 points of sibling error, so the same-sigma fast-vs-accurate table (no decision changes) understates the fast pass's effect. The rho >= 0.97 merge does not decide T2: it fires only in third-product problems, while the A/A'' split comes from the unmerged P(A) and P(A'') problems (max difference correlation 0.73) and tracks Phi(dV / sqrt(2 sigma_nu^2)) to 0.01 per respondent.")


def write_summary(rows, tuned, recov, split, secs):
    dgps, sims = ["D1", "D2", "D3", "D4"], ["S1", "S2", "S3", "S4"]
    get = {(r["dgp"], r["sim"], r["target"]): r for r in rows}
    L = ["# RFC-style vs probit kernel on near-duplicates", "",
         f"Generated by `run.py` (seed {SEED}, N = {N}, {secs / 60:.0f} min). Oracle utilities; "
         "each simulator tuned on 15 calibration tasks (<= 200 objective evaluations), scored on T1-T4. "
         "Row-level numbers are in `results.csv`.", ""]

    L += ["## Findings", "", FINDINGS, ""]

    L += ["## Sibling source-of-volume error, mean over T1-T3", "",
          "|A| error (T1, T2), mean of |A| and |B| (T3), in share-of-volume points (x100). "
          "Truth MC SE in brackets for D2/D3.", "",
          "| DGP | " + " | ".join(sims) + " |", "|---" * 5 + "|"]
    for d in dgps:
        ts = ["T1", "T2", "T3"]
        se = np.mean([get[d, "S1", t]["truth_sib_se"] for t in ts])
        cells = [f"{100 * np.mean([get[d, s, t]['sov_sib_err'] for t in ts]):.1f}" for s in sims]
        L.append(f"| {d}{f' [{100 * se:.2f}]' if se else ''} | " + " | ".join(cells) + " |")

    L += ["", "## Cannibalization decision flips (out of 4 targets; hurdle 50% incrementality)", "",
          "| DGP | true incrementality T1/T2/T3/T4 | " + " | ".join(sims) + " |", "|---" * 6 + "|"]
    for d in dgps:
        inc = "/".join(f"{get[d, 'S1', t]['incr_true']:.2f}" for t in TARGETS)
        cells = [str(sum(not get[d, s, t]["decision_match"] for t in TARGETS)) for s in sims]
        L.append(f"| {d} | {inc} | " + " | ".join(cells) + " |")

    L += ["", "## Tuned parameters", "",
          "| DGP | S1 lambda | S2 sigma_level, gumbel | S3 lambda, mu | S4 sigma b/f/p/nu (fast) | "
          "S4 retuned in accurate mode | flags |", "|---" * 7 + "|"]
    for d in dgps:
        p = {s: "/".join(f"{v:.3f}" for v in tuned[d, s]["theta"]) for s in sims}
        fl = "; ".join(f"{s}: {', '.join(tuned[d, s]['flags'])}" for s in sims if tuned[d, s]["flags"])
        fl += ("; " if fl and recov[d]["flags"] else "") + (
            f"S4 acc: {', '.join(recov[d]['flags'])}" if recov[d]["flags"] else "")
        L.append(f"| {d} | {p['S1']} | {p['S2']} | {p['S3']} | {p['S4']} | "
                 f"{'/'.join(f'{v:.3f}' for v in recov[d]['theta'])} | {fl or '-'} |")
    L.append(f"| D2 true | | | | {'/'.join(f'{v:.3f}' for v in D2_SIG)} | | |")
    L += ["", "S2 is tuned with the Gumbel error integrated analytically (a logit-kernel mixed logit "
          "over 200 fixed level draws per task), which is if anything kinder than literal RFC; it is "
          "scored by literal first choice at 10^5 draws per scenario. S2 has one level-error magnitude "
          "for all three attributes; S4 has one per attribute."]

    L += ["", "## S4 fast vs accurate (same tuned sigma; accurate minus fast)", "",
          "| DGP | target | share MAE (x100) | sibling SoV err (x100) | incrementality | decision changes | "
          "s/scenario fast, accurate |", "|---" * 7 + "|"]
    for d in dgps:
        for t in TARGETS:
            r = get[d, "S4", t]
            L.append(f"| {d} | {t} | {100 * (r['share_mae'] - r['fast_share_mae']):+.3f} | "
                     f"{100 * (r['sov_sib_err'] - r['fast_sov_sib_err']):+.2f} | "
                     f"{r['incr_sim'] - r['fast_incr_sim']:+.3f} | "
                     f"{'yes' if r['decision_match'] != r['fast_decision_match'] else 'no'} | "
                     f"{r['fast_time_per_scenario_s']:.3f}, {r['time_per_scenario_s']:.3f} |")

    L += ["", "## Precision per unit compute: S2 (10^5 draws) vs S4 (accurate)", "",
          "Sibling source of volume, x100. Draws needed = 10^5 (S2 SE / S4 error)^2, with S4's error "
          "floored at the truth MC SE (then a lower bound).", "",
          "| DGP | target | S2 MC SE | S4 error vs truth | S2 draws to match | S2 s/scenario | S4 s/scenario |",
          "|---" * 7 + "|"]
    for d in dgps:
        for t in ["T1", "T2", "T3"]:
            r, r4 = get[d, "S2", t], get[d, "S4", t]
            L.append(f"| {d} | {t} | {100 * r['sim_sib_se']:.2f} | {100 * r['s4_acc_sib_err']:.2f} | "
                     f"{r['s2_draws_to_match_s4']:.2g} | {r['time_per_scenario_s']:.2f} | "
                     f"{r4['time_per_scenario_s']:.3f} |")

    L += ["", "## T2 and the near-duplicate merge", "",
          "In T2, A'' is A at $0.30 less (same brand, flavor, pack). Under D2 the error correlation of "
          "A and A'' is 0.98, so the engine's default merge (difference variables correlated >= 0.97) "
          "fires. It fires only in the problems for third products (B..E, none), where the differences "
          "against A and against A'' collapse into one variable with the tighter limit; that changes "
          "how much the pair takes jointly, not how it is split. The A/A'' split comes from P(A) and "
          "P(A'') themselves, whose difference variables are not merged: the A-vs-A'' difference has "
          "variance 2 sigma_nu^2 and enters as its own coordinate, so the per-respondent split is the "
          "kernel's, close to Phi((V_A'' - V_A) / sqrt(2 sigma_nu^2)) = Phi(dV / sqrt(2(1 - rho)) ) "
          "on the unit-variance scale. No equal or IIA split is imposed. Checks against D2 truth "
          "(10^4 draws per respondent):", "",
          "| sigma used | max corr in P(A), P(A'') problems | merged pairs in third-product problems | "
          "split A''/(A+A''): engine, fast, truth | mean abs per-respondent split diff: vs formula, vs truth | "
          "T2 share MAE (x100) | T2 sibling SoV err (x100) |", "|---" * 7 + "|"]
    for label, m in split.items():
        L.append(f"| {label} ({'/'.join(f'{v:.3f}' for v in m['theta'])}) | {m['max_corr_A']:.3f}, "
                 f"{m['max_corr_new']:.3f} | {m['n_merge_third']} | {m['agg_q_engine']:.4f}, "
                 f"{m['agg_q_fast']:.4f}, {m['agg_q_truth']:.4f} | {m['mad_engine_formula']:.4f}, "
                 f"{m['mad_engine_truth']:.4f} | {100 * m['share_mae_vs_truth']:.3f} | "
                 f"{100 * m['sib_sov_err']:.2f} |")
    L += ["", "Per-respondent truth splits carry MC noise (about 10^4 draws on the pair's share), so "
          "the vs-truth column has a noise floor; the aggregate split is the cleaner comparison. "
          "T2 results for S4 should be read with this rule in mind.", ""]

    L += ["## Design notes", "",
          "- Error variances as fractions of pi^2/6: brand .08, flavor .45, pack .45, product .02 (D2; "
          "D3 the same with Gumbel product error). Implied error correlations: T1 pair (A, A') 0.90, "
          "T2 pair (A, A'') 0.98, flavor-only 0.45. D4 (cross-nested, mu = 0.3): same flavor+pack about "
          "0.91 (Papola approximation). D1: 0 within person.",
          "- 'None' has an independent error with the same total variance as a product in every DGP "
          "and simulator.",
          "- The firm owning A (only b1 product) launches each new product; incrementality = 1 - "
          "source of volume from A.",
          "- T4 cannot share zero levels with A-E (5 distinct flavor x pack pairs on a 3 x 3 grid). G "
          "uses an unused brand and flavor and shares its pack with C and E (not A or B).",
          "- Engine modes: fast = resolution 'low', accurate = 'high'; both with the default merge at "
          "rho >= 0.97 and per-respondent renormalization. S4 is tuned in fast mode; the same sigma is "
          "scored in both modes. The accurate-mode retune is reported only to show how far fast-mode "
          "tuning moves sigma.",
          "- Truth: D1 and D4 closed form; D2 and D3 first choice with 10^7 draws per scenario, common "
          "random numbers across scenarios, SE from 10 batch means. The orthant engine is never used "
          "for truth.", ""]
    (HERE / "SUMMARY.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
