"""Do joint (multivariate probit) effects change menu decisions?

Pick-any menu of 6 add-ons. Given the same (oracle) individual utilities, which
simulator best reproduces joint take rates (pairs, triples), the "which pair to
bundle" decision, and the joint response to a price change? Four truths, four
simulators. Single entry point, fixed seeds:

    python tests/joint_menu/run.py

Writes results.csv and SUMMARY.md next to this file. Needs the local paid
orthant binary (rfc.BACKEND == "local").
"""
import csv
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_expit, log_ndtr, logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research" / "rfc_vs_exact"))
import rfc  # noqa: E402  (orthant backend)

SEED = 20261006
N, M, NCAL = 1000, 6, 15

# ---------------------------------------------------------------------------
# Menu, respondents, truths
# ---------------------------------------------------------------------------
P_BASE = np.array([3.0, 3.2, 2.5, 4.0, 3.5, 2.0])
P_T4 = P_BASE * np.r_[1, 1, 1, 1.3, 1, 1]  # T4: raise p_4 by 30%
B_M, B_S = np.log(.8), .3  # b_i lognormal
MU_A = .8 * P_BASE - .6
SD_A = .8


def corr(pairs):
    C = np.eye(M)
    for (j, k), r in pairs.items():
        C[j, k] = C[k, j] = r
    return C


CORR_A = corr({(0, 1): .6, (0, 2): .4, (1, 2): .4, (3, 4): .3})  # taste heterogeneity (mechanism 1)
R_TRUE = corr({(0, 1): .9, (0, 2): .5, (1, 2): .5, (3, 4): .5})  # J2/J4 error correlation
GAMMA = 1.5  # J3 complementarity of items 4 and 5

S = (np.arange(64)[:, None] >> np.arange(M)) & 1  # (64, 6) patterns
POW = 1 << np.arange(M)
PAIRS = list(combinations(range(M), 2))
TRIPLES = [(0, 1, 2), (2, 3, 4)]
DGPS = ["J1", "J2", "J3", "J4"]
SIMS = ["M1", "M2", "M3", "M4"]
item = lambda j: j + 1  # noqa: E731  (1-based item labels for reports)


def respondents(rng):
    a = MU_A + SD_A * rng.standard_normal((N, M)) @ np.linalg.cholesky(CORR_A).T
    b = np.exp(B_M + B_S * rng.standard_normal(N))
    return a, b


def util(a, b, p):
    """p (M,) or (N, ..., M) -> V (N, ..., M)."""
    p = np.asarray(p)
    if p.ndim == 1:
        return a - b[:, None] * p
    return a.reshape((N,) + (1,) * (p.ndim - 2) + (M,)) - b.reshape((N,) + (1,) * (p.ndim - 1)) * p


def logistic_eps(z):
    """Logistic margins through a Gaussian copula: F_L^{-1}(Phi(z))."""
    return log_ndtr(z) - log_ndtr(-z)


def take_j3(V, eps):
    """J3 choice given item utilities V and item-level logistic eps (..., M)."""
    U = V + eps
    s = U > 0
    opts = np.stack([np.zeros_like(U[..., 3]), U[..., 3], U[..., 4], U[..., 3] + U[..., 4] + GAMMA], -1)
    c = opts.argmax(-1)
    s[..., 3] = (c == 1) | (c == 3)
    s[..., 4] = (c == 2) | (c == 3)
    return s


def simulate_choices(d, Vc, rng):
    eps = rng.logistic(size=Vc.shape)
    if d == "J1":
        return Vc + eps > 0
    if d == "J3":
        return take_j3(Vc, eps)
    z = rng.standard_normal(Vc.shape) @ np.linalg.cholesky(R_TRUE).T
    return Vc + (z if d == "J2" else logistic_eps(z)) > 0


def indep_patterns(pm):
    """(N, M) take probabilities, independent items -> (N, 64) pattern probabilities."""
    return np.prod(np.where(S[None], pm[:, None, :], 1 - pm[:, None, :]), -1)


def j3_patterns(V):
    """J3 closed form: items 1-3, 6 independent logits; the 4/5 block by a 1-D integral."""
    V4, V5 = V[:, 3], V[:, 4]
    p10 = expit(V4) * expit(-GAMMA - V5)
    p01 = expit(V5) * expit(-GAMMA - V4)
    x, w = np.polynomial.legendre.leggauss(64)
    u = (x - 1) * GAMMA / 2  # [-gamma, 0]
    f = expit(u[None] - V4[:, None]) * expit(V4[:, None] - u[None])  # logistic density of U4 at u
    p11 = expit(V5 + GAMMA) * expit(V4) + GAMMA / 2 * (f * expit(V5[:, None] + GAMMA + u[None])) @ w
    p00 = 1 - p10 - p01 - p11
    cell = np.stack([p00, p10, p01, p11], -1)  # index s4 + 2 s5
    pm = expit(V)
    P = np.prod(np.where(S[None][..., [0, 1, 2, 5]], pm[:, None, [0, 1, 2, 5]], 1 - pm[:, None, [0, 1, 2, 5]]), -1)
    return P * cell[:, S[:, 3] + 2 * S[:, 4]]


def mc_patterns(d, Vs, rng, R=10_000, nb=10, block=50):
    """J2/J4 truth by Monte Carlo, common random numbers across price scenarios.
    Returns (nb, len(Vs), 64) population pattern frequencies."""
    L = np.linalg.cholesky(R_TRUE).astype(np.float32)
    out = np.zeros((nb, len(Vs), 64))
    for s0 in range(0, N, block):
        n = min(block, N - s0)
        z = rng.standard_normal((n, R, M), np.float32) @ L.T
        eps = z if d == "J2" else logistic_eps(z).astype(np.float32)
        for k, V in enumerate(Vs):
            idx = ((V[s0:s0 + n, None, :] + eps > 0) @ POW).reshape(n, nb, R // nb)
            for b in range(nb):
                out[b, k] += np.bincount(idx[:, b].ravel(), minlength=64)
    return out / (N * R // nb)


def true_patterns(d, a, b, rng):
    """-> (nb, 2, 64) population pattern probabilities at base and T4 prices (nb = 1 if exact)."""
    Vs = [util(a, b, P_BASE), util(a, b, P_T4)]
    if d == "J1":
        return np.array([[indep_patterns(expit(V)).mean(0) for V in Vs]])
    if d == "J3":
        return np.array([[j3_patterns(V).mean(0) for V in Vs]])
    return mc_patterns(d, Vs, rng)


# ---------------------------------------------------------------------------
# Simulators: fit on calibration, then population pattern probabilities
# ---------------------------------------------------------------------------
def fit_m1(Vc, Y):
    def nll(lam):
        z = lam * Vc
        return -np.where(Y, log_expit(z), log_expit(-z)).sum()
    r = minimize_scalar(nll, bounds=(.05, 5), method="bounded", options=dict(maxiter=200))
    return dict(lam=r.x)


def fit_m2(Vc, Y, Pc, lam):
    """Serial cross-effect logits: logit P(take m) = lam V_m + sum_{k != m} c_mk (p_k - p_k^base).
    lam fixed at M1's value; cross effects by ML, item by item."""
    C = np.zeros((M, M))
    dP = (Pc - P_BASE).reshape(-1, M)
    for m in range(M):
        others = [k for k in range(M) if k != m]
        X, off, y = dP[:, others], lam * Vc[..., m].ravel(), Y[..., m].ravel()

        def f(c):
            z = off + X @ c
            pr = expit(z)
            return -np.where(y, log_expit(z), log_expit(-z)).sum(), -X.T @ (y - pr)
        C[m, others] = minimize(f, np.zeros(M - 1), jac=True, method="BFGS").x
    return dict(lam=lam, C=C)


def bvn(u, sg, rho, res):
    """P(Z1 <= u1, Z2 <= u2) for unit normals with correlation sg * rho, sg = +-1 per row."""
    out = np.empty(len(u))
    for g in (1., -1.):
        m = sg == g
        if m.any():
            r = g * rho
            out[m] = rfc.orthant_cdf(u[m], np.array([[1, r], [r, 1]]), resolution=res)
    return out


def nearest_corr(A, iters=200):
    """Higham (2002) alternating projections onto correlation matrices."""
    Y, dS = A.copy(), np.zeros_like(A)
    for _ in range(iters):
        Rk = Y - dS
        w, Q = np.linalg.eigh(Rk)
        X = Q @ np.diag(np.maximum(w, 1e-8)) @ Q.T
        dS = X - Rk
        Y = X.copy()
        np.fill_diagonal(Y, 1)
    return Y


def fit_m3(Vc, Y, res):
    """IFM: margins Phi(lam V) with one scale fitted first; then each rho_jk by
    pairwise likelihood with the margins fixed; project to the nearest correlation matrix."""
    def nll1(lam):
        return -log_ndtr(np.where(Y, 1, -1) * lam * Vc).sum()
    lam = minimize_scalar(nll1, bounds=(.05, 5), method="bounded", options=dict(maxiter=200)).x
    U = (np.where(Y, 1., -1.) * lam * Vc).reshape(-1, M)
    sg = np.where(Y, 1., -1.).reshape(-1, M)
    R = np.eye(M)
    for j, k in PAIRS:
        u, s = U[:, [j, k]], sg[:, j] * sg[:, k]

        def nll2(rho):
            return -np.log(np.maximum(bvn(u, s, rho, res), 1e-300)).sum()
        R[j, k] = R[k, j] = minimize_scalar(nll2, bounds=(-.99, .99), method="bounded",
                                            options=dict(maxiter=200, xatol=1e-4)).x
    projected = np.linalg.eigvalsh(R).min() < 1e-8
    return dict(lam=lam, R_raw=R, R=nearest_corr(R) if projected else R, projected=projected)


def fit_m4(Vc, Y):
    """One MNL over the 64 patterns: W_s = lam sum_m s_m V_m + kappa_s, kappa_0 = 0."""
    VS = Vc.reshape(-1, M) @ S.T  # (obs, 64)
    y = (Y.reshape(-1, M) @ POW)
    cnt = np.bincount(y, minlength=64)
    vs_y = VS[np.arange(len(y)), y].sum()

    def f(th):
        lam, kap = th[0], np.r_[0, th[1:]]
        W = lam * VS + kap
        lse = logsumexp(W, 1, keepdims=True)
        P = np.exp(W - lse)
        ll = lam * vs_y + kap[y].sum() - lse.sum()
        g = np.r_[vs_y - (P * VS).sum(), (cnt - P.sum(0))[1:]]
        return -ll, -g
    r = minimize(f, np.r_[1., np.zeros(63)], jac=True, method="L-BFGS-B",
                 bounds=[(.01, 10)] + [(-30, 30)] * 63, options=dict(maxiter=2000))
    return dict(lam=r.x[0], kappa=np.r_[0, r.x[1:]], converged=r.success, n_empty=int((cnt == 0).sum()))


def m3_patterns(V, fit, res):
    lam, R = fit["lam"], fit["R"]
    P = np.empty((N, 64))
    for k in range(64):
        D = 2. * S[k] - 1
        P[:, k] = rfc.orthant_cdf(D * lam * V, R * np.outer(D, D), resolution=res)
    return P / P.sum(1, keepdims=True)


def sim_patterns(s, fit, a, b, p, res="high"):
    """Population pattern probabilities (64,) at prices p, and seconds taken."""
    t0 = time.time()
    V = util(a, b, p)
    if s == "M1":
        P = indep_patterns(expit(fit["lam"] * V))
    elif s == "M2":
        P = indep_patterns(expit(fit["lam"] * V + (p - P_BASE) @ fit["C"].T))
    elif s == "M3":
        P = m3_patterns(V, fit, res)
    else:
        W = fit["lam"] * V @ S.T + fit["kappa"]
        P = np.exp(W - logsumexp(W, 1, keepdims=True))
    return P.mean(0), time.time() - t0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def stats(pi):
    """From population pattern probabilities (..., 64)."""
    marg = pi @ S
    joint = np.stack([pi @ (S[:, j] * S[:, k]) for j, k in PAIRS], -1)
    lift = joint / np.stack([marg[..., j] * marg[..., k] for j, k in PAIRS], -1)
    trip = np.stack([pi @ S[:, list(t)].prod(1) for t in TRIPLES], -1)
    return dict(marg=marg, joint=joint, lift=lift, trip=trip)


def pick(x):
    """Best pair (1-based label) and margin between the top two."""
    o = np.argsort(x)[::-1]
    j, k = PAIRS[o[0]]
    return f"{item(j)}-{item(k)}", x[o[0]] - x[o[1]]


def se(x):
    return x.std(0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.zeros(np.shape(x)[1:])


def target_metrics(base, t4, tb, tt, tb_b, tt_b):
    """Metrics of a simulator (base, t4 pattern probs) against truth (tb, tt; batches tb_b, tt_b)."""
    s, s4, t, t4s = stats(base), stats(t4), stats(tb), stats(tt)
    sb, sb4 = stats(tb_b), stats(tt_b)
    ej, el = np.abs(s["joint"] - t["joint"]), np.abs(s["lift"] - t["lift"])
    pj_t, mj_t = pick(t["joint"])
    pl_t, ml_t = pick(t["lift"])
    d5 = lambda x, y: y["marg"][..., 4] - x["marg"][..., 4]  # noqa: E731
    d45 = lambda x, y: y["joint"][..., PAIRS.index((3, 4))] - x["joint"][..., PAIRS.index((3, 4))]  # noqa: E731
    return {
        "T1": dict(pair_joint_mae=ej.mean(), pair_joint_max=ej.max(), lift_mae=el.mean(), lift_max=el.max(),
                   truth_joint_se_max=se(sb["joint"]).max(), truth_lift_se_max=se(sb["lift"]).max()),
        "T2": dict(triple123_true=t["trip"][0], triple123_err=s["trip"][0] - t["trip"][0],
                   triple345_true=t["trip"][1], triple345_err=s["trip"][1] - t["trip"][1],
                   truth_triple_se_max=se(sb["trip"]).max()),
        "T3": dict(pick_joint_true=pj_t, pick_joint_sim=pick(s["joint"])[0], margin_joint_true=mj_t,
                   pick_joint_match=pj_t == pick(s["joint"])[0],
                   pick_lift_true=pl_t, pick_lift_sim=pick(s["lift"])[0], margin_lift_true=ml_t,
                   pick_lift_match=pl_t == pick(s["lift"])[0]),
        "T4": dict(dP5_true=d5(t, t4s), dP5_sim=d5(s, s4), dP5_err=d5(s, s4) - d5(t, t4s),
                   dP45_true=d45(t, t4s), dP45_sim=d45(s, s4), dP45_err=d45(s, s4) - d45(t, t4s),
                   truth_dP5_se=se(d5(sb, sb4)), truth_dP45_se=se(d45(sb, sb4))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    assert rfc.BACKEND == "local", "needs the local paid orthant binary"
    t_start = time.time()
    ss = np.random.SeedSequence(SEED)
    r_resp, r_cal, *r_dgp = [np.random.default_rng(s) for s in ss.spawn(6)]
    a, b = respondents(r_resp)
    Pc = P_BASE * r_cal.uniform(.6, 1.4, (N, NCAL, M))  # calibration menus: +-40% around base
    Vc = util(a, b, Pc)

    rows, fits, timing, rec = [], {}, {}, {}
    for d, rng in zip(DGPS, r_dgp):
        t0 = time.time()
        Y = simulate_choices(d, Vc, rng)
        tr = true_patterns(d, a, b, rng)  # (nb, 2, 64)
        tb, tt = tr[:, 0].mean(0), tr[:, 1].mean(0)
        st = stats(tb)
        print(f"{d}: truth {time.time() - t0:.1f}s; take rates {np.round(st['marg'], 3)}; "
              f"calibration take rate {Y.mean():.3f}")

        f = {"M1": fit_m1(Vc, Y)}
        f["M2"] = fit_m2(Vc, Y, Pc, f["M1"]["lam"])
        t0 = time.time()
        f["M3"] = fit_m3(Vc, Y, "low")
        f["M3"]["secs"] = time.time() - t0
        f["M4"] = fit_m4(Vc, Y)
        if d == "J2":
            rec["R_acc"] = fit_m3(Vc, Y, "high")["R"]
        fits[d] = f
        print(f"{d}: M1 lam {f['M1']['lam']:.3f}; M2 c_54 {f['M2']['C'][4, 3]:+.3f}; M3 lam {f['M3']['lam']:.3f} "
              f"R12 {f['M3']['R'][0, 1]:.3f} R45 {f['M3']['R'][3, 4]:.3f} projected {f['M3']['projected']} "
              f"({f['M3']['secs']:.0f}s); M4 lam {f['M4']['lam']:.3f} converged {f['M4']['converged']} "
              f"empty patterns {f['M4']['n_empty']}")

        for s in SIMS:
            base, secs = sim_patterns(s, f[s], a, b, P_BASE)
            t4, _ = sim_patterns(s, f[s], a, b, P_T4)
            met = target_metrics(base, t4, tb, tt, tr[:, 0], tr[:, 1])
            timing[d, s] = secs
            if s == "M3":
                fb, fsecs = sim_patterns(s, f[s], a, b, P_BASE, "low")
                ft4, _ = sim_patterns(s, f[s], a, b, P_T4, "low")
                fmet = target_metrics(fb, ft4, tb, tt, tr[:, 0], tr[:, 1])
                timing[d, "M3 fast"] = fsecs
            for t in ["T1", "T2", "T3", "T4"]:
                r = dict(dgp=d, sim=s, target=t, time_pattern_eval_s=secs, **met[t])
                if s == "M3":
                    r.update({f"fast_{k}": v for k, v in fmet[t].items() if not k.startswith(("truth", "pick_joint_true",
                              "pick_lift_true", "margin", "triple123_true", "triple345_true", "dP5_true", "dP45_true"))})
                    r["fast_time_pattern_eval_s"] = fsecs
                rows.append(r)

    write_csv(rows)
    write_summary(rows, fits, timing, rec, time.time() - t_start)
    print(f"done in {time.time() - t_start:.0f}s")


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.6g}"
    return str(v)


def write_csv(rows):
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(HERE / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c, np.nan)) for c in cols])


FINDINGS = (
    "With correlated errors in the truth (J2, J4), the independent logits M1/M2 miss pair joints by 1.3-1.9 points on average (up to 12), understate P(1,2,3) by 6-10 points and pick the wrong bundle by joint take (4-5 instead of the near-duplicates 1-2), while M3 and M4 get every bundle pick right in every DGP. M3 is the most accurate on correlated truths and its pairwise R carries the higher-order structure (J2 R within 0.02 of truth, triple errors at most 1.6 points), but M4 is close behind (mean pair error 0.31 vs 0.19 points in J2, 0.38 vs 0.31 in J4). Under complementarity (J3) M3 is misspecified and it shows: the pairwise fit absorbs gamma as rho_45 = 0.77, yet pair error is 4.2 points (max 16.7, against 0.22 for M4), P(3,4,5) is 6.9 points low, and it predicts no change in P(5) when p_4 rises against a true -4.9 points; it still picks the right bundle only because the true margin is 21 points. M4 is the only simulator never badly wrong on joints, but its pattern constants turn error correlation into complementarity, so in J2/J4 it invents a -2.1/-2.8 point response of P(5) to p_4 that the truth does not have; M2's cross effects get J3's dP(5) right (error +0.08) at the cost of a spurious +1.0 point response in J1. Fast mode is fine for estimating R (2-D problems, identical to accurate to 3 decimals) but not for the 6-D pattern pass: it worsens M3's mean pair error by 1.5-2.3 points and P(1,2,3) by 4-6 points, and flips the joint-take bundle pick in J2 and J4. M3 costs about 0.75 s per full 64-pattern evaluation against 0.002-0.005 s for M1, M2 and M4.")


def write_summary(rows, fits, timing, rec, secs):
    g = {(r["dgp"], r["sim"], r["target"]): r for r in rows}
    L = ["# Joint (multivariate probit) effects in a pick-any menu", "",
         f"Generated by `run.py` (seed {SEED}, N = {N}, {NCAL} calibration menus per respondent, "
         f"{secs / 60:.1f} min). Oracle a_i, b_i. Row-level numbers in `results.csv`. "
         "Probabilities in percentage points (x100) unless noted.", "",
         "## Findings", "", FINDINGS, ""]

    L += ["## Mean pair-joint error (T1), x100; lift mean error in brackets", "",
          "| DGP | " + " | ".join(SIMS) + " | truth SE (max) |", "|---" * 6 + "|"]
    for d in DGPS:
        cells = [f"{100 * g[d, s, 'T1']['pair_joint_mae']:.2f} ({g[d, s, 'T1']['lift_mae']:.3f})" for s in SIMS]
        L.append(f"| {d} | " + " | ".join(cells) + f" | {100 * g[d, 'M1', 'T1']['truth_joint_se_max']:.3f} |")
    L += ["", "Max over pairs, x100:", "", "| DGP | " + " | ".join(SIMS) + " |", "|---" * 5 + "|"]
    for d in DGPS:
        L.append(f"| {d} | " + " | ".join(f"{100 * g[d, s, 'T1']['pair_joint_max']:.2f}" for s in SIMS) + " |")

    L += ["", "## Bundle pick (T3): matches truth? (highest joint / highest lift)", "",
          "| DGP | true pick joint (margin x100) | true pick lift (margin) | " + " | ".join(SIMS) + " |",
          "|---" * 7 + "|"]
    for d in DGPS:
        r = g[d, "M1", "T3"]
        cells = []
        for s in SIMS:
            x = g[d, s, "T3"]
            cells.append(f"{'yes' if x['pick_joint_match'] else 'no (' + x['pick_joint_sim'] + ')'} / "
                         f"{'yes' if x['pick_lift_match'] else 'no (' + x['pick_lift_sim'] + ')'}")
        L.append(f"| {d} | {r['pick_joint_true']} ({100 * r['margin_joint_true']:.2f}) | "
                 f"{r['pick_lift_true']} ({r['margin_lift_true']:.3f}) | " + " | ".join(cells) + " |")

    L += ["", "## Triples (T2): simulated minus true, x100", "",
          "| DGP | true P(1,2,3) | true P(3,4,5) | " + " | ".join(f"{s} 123 / 345" for s in SIMS) + " | truth SE |",
          "|---" * 8 + "|"]
    for d in DGPS:
        r = g[d, "M1", "T2"]
        cells = [f"{100 * g[d, s, 'T2']['triple123_err']:+.2f} / {100 * g[d, s, 'T2']['triple345_err']:+.2f}"
                 for s in SIMS]
        L.append(f"| {d} | {100 * r['triple123_true']:.2f} | {100 * r['triple345_true']:.2f} | "
                 + " | ".join(cells) + f" | {100 * r['truth_triple_se_max']:.3f} |")

    L += ["", "## Price counterfactual (T4): p_4 +30%; simulated minus true change, x100", "",
          "| DGP | true dP(5) | true dP(4,5) | " + " | ".join(f"{s} dP5 / dP45" for s in SIMS) + " | truth SE |",
          "|---" * 8 + "|"]
    for d in DGPS:
        r = g[d, "M1", "T4"]
        cells = [f"{100 * g[d, s, 'T4']['dP5_err']:+.2f} / {100 * g[d, s, 'T4']['dP45_err']:+.2f}" for s in SIMS]
        L.append(f"| {d} | {100 * r['dP5_true']:+.2f} | {100 * r['dP45_true']:+.2f} | " + " | ".join(cells)
                 + f" | {100 * max(r['truth_dP5_se'], r['truth_dP45_se']):.3f} |")

    L += ["", "## M3: fast minus accurate (same estimated R)", "",
          "| DGP | pair-joint MAE x100 | lift MAE | triple 123 / 345 err x100 | picks joint/lift | "
          "T4 dP5 / dP45 err x100 |", "|---" * 6 + "|"]
    for d in DGPS:
        r1, r2, r3, r4 = (g[d, "M3", t] for t in ["T1", "T2", "T3", "T4"])
        L.append(f"| {d} | {100 * (r1['fast_pair_joint_mae'] - r1['pair_joint_mae']):+.3f} | "
                 f"{r1['fast_lift_mae'] - r1['lift_mae']:+.4f} | "
                 f"{100 * (r2['fast_triple123_err'] - r2['triple123_err']):+.3f} / "
                 f"{100 * (r2['fast_triple345_err'] - r2['triple345_err']):+.3f} | "
                 f"{'same' if r3['fast_pick_joint_sim'] == r3['pick_joint_sim'] else 'changed'} / "
                 f"{'same' if r3['fast_pick_lift_sim'] == r3['pick_lift_sim'] else 'changed'} | "
                 f"{100 * (r4['fast_dP5_err'] - r4['dP5_err']):+.3f} / "
                 f"{100 * (r4['fast_dP45_err'] - r4['dP45_err']):+.3f} |")

    L += ["", "## M3 estimated R against true R (J2)", "",
          "| pair | true | fast (used) | accurate |", "|---" * 4 + "|"]
    for j, k in PAIRS:
        L.append(f"| {item(j)}-{item(k)} | {R_TRUE[j, k]:.2f} | {fits['J2']['M3']['R'][j, k]:.3f} | "
                 f"{rec['R_acc'][j, k]:.3f} |")

    L += ["", "## Fitted parameters", "",
          "| DGP | M1 lambda | M2 c_54 (P(5) per $ of p_4), max abs c | M3 lambda, R_12, R_45, projected | "
          "M4 lambda, 4x5 interaction kappa_45 - kappa_4 - kappa_5, empty patterns |", "|---" * 5 + "|"]
    for d in DGPS:
        f = fits[d]
        k = f["M4"]["kappa"]
        inter = k[POW[3] + POW[4]] - k[POW[3]] - k[POW[4]]
        L.append(f"| {d} | {f['M1']['lam']:.3f} | {f['M2']['C'][4, 3]:+.3f}, {np.abs(f['M2']['C']).max():.3f} | "
                 f"{f['M3']['lam']:.3f}, {f['M3']['R'][0, 1]:.3f}, {f['M3']['R'][3, 4]:.3f}, "
                 f"{f['M3']['projected']} | {f['M4']['lam']:.3f}, {inter:+.3f}, {f['M4']['n_empty']} |")

    L += ["", "## Time per full 64-pattern evaluation (1,000 respondents, one price vector), seconds", "",
          "| DGP | M1 | M2 | M3 accurate | M3 fast | M4 |", "|---" * 6 + "|"]
    for d in DGPS:
        L.append(f"| {d} | {timing[d, 'M1']:.4f} | {timing[d, 'M2']:.4f} | {timing[d, 'M3']:.3f} | "
                 f"{timing[d, 'M3 fast']:.3f} | {timing[d, 'M4']:.4f} |")

    L += ["", "## Design notes", "",
          f"- Base prices {P_BASE.tolist()}; mu_a = 0.8 p - 0.6; a_i sd {SD_A}, taste correlations 1-2 .6, "
          "1-3 .4, 2-3 .4, 4-5 .3; b_i = exp(N(log .8, .3^2)). Calibration prices: base x U(0.6, 1.4), "
          "drawn per respondent-menu.",
          "- J2 errors MVN(0, R) with unit variance as specified (R: 1-2 .9, 1-3 .5, 2-3 .5, 4-5 .5, rest 0); "
          "J1, J3, J4 margins standard logistic. J3: gamma = 1.5 with item-level logistic errors, so items "
          "1-3 and 6 are independent and only the 4/5 block is joint.",
          "- Truth: J1 closed form; J3 closed form plus 64-node Gauss-Legendre for P(4,5); J2, J4 Monte Carlo "
          "with 10^7 draws per price scenario, common random numbers across scenarios, SE from 10 batch means. "
          "The orthant engine is never used for truth.",
          "- With oracle a_i, P(5) of each respondent does not depend on p_4 under J1, J2, J4, so the true "
          "dP(5) there is exactly 0; only J3 moves it.",
          "- M1: one logit scale by ML. M2: M1's scale plus 30 cross-price coefficients on prices centred at "
          "base (by ML, item by item), so M2 equals M1 at base prices and differs only in T4.",
          "- M3 (IFM): margins Phi(lambda V) with lambda fitted first (the truths' margins are logistic in J1, "
          "J3, J4), then each rho_jk by pairwise likelihood with the margins fixed (fast mode), projected to "
          "the nearest correlation matrix if not positive definite. Patterns from the orthant engine, "
          "renormalized per respondent; accurate mode for the reported results, fast mode for the deltas.",
          "- M4: one MNL over the 64 patterns, scale and 63 pattern constants by ML (bounded at +-30). Under "
          "J1 it nests the truth (lambda = 1, kappa = 0).",
          "- Lift = P(j,k) / (P(j) P(k)) on population take rates.", ""]
    (HERE / "SUMMARY.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
