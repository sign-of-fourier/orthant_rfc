"""T22 · Swissmetro, real stated choice: mixed MNP (MVP) vs HB-MNL + RFC-S vs nested logit.

    python analysis/T22/t22.py local               # MNL, NL, HB-MNL + RFC-S for every training size (local CPU)
    python analysis/T22/t22.py mvp                 # MVP fits on Modal T4 (paid; quote first)
    python analysis/T22/t22.py score               # holdout scoring, bootstrap, out/report.md

Data: EPFL Swissmetro SP survey (transp-or.epfl.ch/data/swissmetro.dat) in data/ (not committed). Commute and
business trips (PURPOSE 1, 3), CHOICE > 0, respondents with a car available: 623 respondents x 9 tasks, alternatives
train, Swissmetro (SM), car. Variables (alternative order train, SM, car): ASC train, ASC SM, time / 100 min,
cost / 100 CHF (0 for train and SM when the respondent holds a GA pass), headway / 100 min (0 for car).

Split: 25% of respondents held out (seed 22), scored on all 9 tasks: population predictive probabilities, as a
market simulator would produce for people not in the sample. Training respondents: tasks 1-7 fit every model;
tasks 8-9 tune RFC-S only (Sawtooth's holdout-task tuning). Training sizes: all training respondents, and 3 random
subsets each of 100 and 200.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
T1 = HERE.parent / "T1"
SEED = 22
T_VAR = np.pi ** 2 / 6                          # MNP total error variance per alternative (logit scale, as T1)
NAMES = ["asc_train", "asc_sm", "time", "cost", "headway"]
P = len(NAMES)
FIT_T, TUNE_T = 7, 2
HB_ITERS, R_L = 20000, 100
SIZES = [("all", 0)] + [(n, s) for n in (100, 200) for s in range(3)]
RAIL = np.array([[1., 1, 0], [1, 1, 0], [0, 0, 0]])   # shared train / SM error component
POLICY = 1.2                                     # SM cost x 1.2


# ----------------------------------------------------------------------------------- data

def load():
    d = pd.read_csv(HERE / "data" / "swissmetro.dat", sep="\t")
    d = d[(d.CHOICE > 0) & d.PURPOSE.isin([1, 3]) & (d.CAR_AV == 1)].sort_index()
    ids = d.ID.unique()
    N = len(ids)
    ga = (d.GA == 0).values
    X = np.zeros((len(d), 3, P))
    X[:, 0, 0] = 1
    X[:, 1, 1] = 1
    for j, a in enumerate(("TRAIN", "SM", "CAR")):
        X[:, j, 2] = d[f"{a}_TT"] / 100
        X[:, j, 3] = d[f"{a}_CO"] / 100 * (ga if a != "CAR" else 1)
        X[:, j, 4] = d[f"{a}_HE"] / 100 if a != "CAR" else 0
    X = X.reshape(N, 9, 3, P)
    y = (d.CHOICE.values - 1).reshape(N, 9)
    return X, y, ids


def split(N):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    nh = N // 4
    return np.sort(perm[nh:]), np.sort(perm[:nh])


def subsets(tr):
    out = {}
    for n, s in SIZES:
        if n == "all":
            out[f"all_s{s}"] = tr
        else:
            out[f"{n}_s{s}"] = np.sort(np.random.default_rng([SEED, n, s]).choice(tr, n, replace=False))
    return out


# ----------------------------------------------------------------------------------- pooled logits

def mnl_fit(X, y):
    Xf, yf = X.reshape(-1, 3, P), y.ravel()

    def f(b):
        ls = log_softmax(Xf @ b, 1)
        p = np.exp(ls)
        return -ls[np.arange(len(yf)), yf].sum(), -(Xf[np.arange(len(yf)), yf] - (p[..., None] * Xf).sum(1)).sum(0)
    return minimize(f, np.zeros(P), jac=True, method="BFGS").x


def nl_logp(X, b, lam):
    """log P(each alternative), nest {train, SM} with dissimilarity lam, car alone."""
    V = X @ b
    iv = lam * logsumexp(V[..., :2] / lam, -1)
    top = np.stack([iv, V[..., 2]], -1)
    lt = log_softmax(top, -1)
    within = log_softmax(V[..., :2] / lam, -1)
    return np.concatenate([lt[..., :1] + within, lt[..., 1:]], -1)


def nl_fit(X, y, b0):
    Xf, yf = X.reshape(-1, 3, P), y.ravel()

    def f(th):
        return -nl_logp(Xf, th[:P], 1 / (1 + np.exp(-th[P])))[np.arange(len(yf)), yf].sum()
    r = minimize(f, np.r_[b0, 2.0], method="L-BFGS-B")
    return r.x[:P], float(1 / (1 + np.exp(-r.x[P])))


# ----------------------------------------------------------------------------------- HB-MNL + RFC-S

def run_hb(X, y, seed):
    N, T = y.shape
    rows = [[n, t, k, int(y[n, t] == k)] + list(X[n, t, k]) for n in range(N) for t in range(T) for k in range(3)]
    work = Path(tempfile.mkdtemp())
    path = work / "hb_in.csv"
    np.savetxt(path, np.array(rows), delimiter=",", comments="",
               header="resp,task,alt,chosen," + ",".join(f"x{i}" for i in range(P)), fmt="%.6g")
    t0 = time.time()
    r = subprocess.run(["Rscript", str(T1 / "hb.R"), str(path), str(work / "hb"), str(HB_ITERS), "10", str(seed)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return np.loadtxt(work / "hb_beta.csv", delimiter=",", skiprows=1), time.time() - t0


class RFC:
    """Sawtooth RFC on HB point estimates, adapted to alternative-specific continuous attributes: per draw, a
    normal error sigma_l on each attribute coefficient, shared by the alternatives of a task (so alternatives with
    similar time / cost / headway are correlated), an independent normal sigma_l error on each ASC'd alternative,
    and Gumbel product error with scale g (integrated analytically). RFC-S: one sigma_l for everything."""

    def __init__(self, B, rng):
        self.B = B
        self.Z = rng.standard_normal((len(B), R_L, P))          # coefficient errors; ASC columns act per alternative
        self.Zc = rng.standard_normal((len(B), R_L))            # car's own error (car has no ASC)

    def probs(self, X, sl, g, max_elems=4_000_000):
        """Population P(alternative) for tasks X (m, 3, P): mean over respondents and draws. Respondents are
        chunked so one block holds at most max_elems utilities (the box has 4 GB)."""
        import torch
        m = X.shape[0]
        Xt = torch.as_tensor(X.reshape(m * 3, P).T)
        out = torch.zeros((m, 3), dtype=torch.float64)
        chunk = max(1, max_elems // (R_L * m * 3))
        for s in range(0, len(self.B), chunk):
            Bd = torch.as_tensor(self.B[s:s + chunk, None, :] + sl * self.Z[s:s + chunk]).reshape(-1, P)
            u = (Bd @ Xt).reshape(-1, m, 3)
            u[..., 2] += sl * torch.as_tensor(self.Zc[s:s + chunk]).reshape(-1, 1)
            out += torch.softmax(u / g, -1).sum(0)
        return out.numpy() / (len(self.B) * R_L)


def tune_rfc(R, X, y, lo=(0.01, 0.05), hi=(5.0, 5.0)):
    """RFC-S: (sigma_l, g) maximising the tuning tasks' population log-likelihood; log grid then Nelder-Mead."""
    Xf, yf = X.reshape(-1, 3, P), y.ravel()
    llo, lhi = np.log(lo), np.log(hi)

    def nll(x):
        x = np.clip(x, llo, lhi)
        p = R.probs(Xf, *np.exp(x))
        return -np.log(np.clip(p[np.arange(len(yf)), yf], 1e-300, None)).mean()
    grid = [np.array([a, b]) for a in np.linspace(llo[0], lhi[0], 6) for b in np.linspace(llo[1], lhi[1], 6)]
    v = [nll(x) for x in grid]
    r = minimize(nll, grid[int(np.argmin(v))], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6))
    x = np.clip(r.x, llo, lhi)
    flags = [n for i, n in enumerate(("sigma_l", "g")) if x[i] - llo[i] < .05 or lhi[i] - x[i] < .05]
    return np.exp(x), float(r.fun), flags


# ----------------------------------------------------------------------------------- MVP (mixed MNP)

def diff_arrays(X, y):
    """Xd (N, T, 2, P), G (2, N, T, 2, 2): differences chosen - other; G components (rail, iid)."""
    N, T = y.shape
    oth = np.array([[k for k in range(3) if k != c] for c in range(3)])[y]
    M = np.zeros((N, T, 2, 3))
    np.put_along_axis(M, oth[..., None], -1.0, axis=3)
    np.put_along_axis(M, np.broadcast_to(y[:, :, None, None], (N, T, 2, 1)), 1.0, axis=3)
    S = np.stack([RAIL, np.eye(3)])
    return M @ X, np.einsum("ntik,akl,ntjl->antij", M, S, M)


def mvp_unpack(th):
    """b, w (taste sd), (sigma2_rail, sigma2_iid), as ghk_gpu.sig2_of with tvar = T_VAR."""
    p = np.exp(np.r_[th[2 * P], 0.0] - logsumexp(np.r_[th[2 * P], 0.0]))
    return th[:P], np.exp(th[P:2 * P]), T_VAR * (p * (1 - 4e-6) + 1e-6)


def mvp_probs(X, th):
    """Population P(alternative): bivariate orthant of the differences, exact."""
    from multivariate_probit import bvn_cdf
    b, w, s2 = mvp_unpack(th)
    E = s2[0] * RAIL + s2[1] * np.eye(3)
    out = np.zeros(X.shape[:2])
    for c in range(3):
        o = [k for k in range(3) if k != c]
        D = np.zeros((2, 3))
        D[:, c] = 1
        D[[0, 1], o] = -1
        mu = np.einsum("ij,mjp,p->mi", D, X, b)
        C = np.einsum("ij,mjp,p,mkp,lk->mil", D, X, w ** 2, X, D) + D @ E @ D.T
        sd = np.sqrt(np.einsum("mii->mi", C))
        out[:, c] = bvn_cdf(mu[:, 0] / sd[:, 0], mu[:, 1] / sd[:, 1], C[:, 0, 1] / (sd[:, 0] * sd[:, 1]))
    return out


# ----------------------------------------------------------------------------------- runs

def local():
    X, y, ids = load()
    tr, ho = split(len(ids))
    OUT.mkdir(exist_ok=True)
    for tag, idx in subsets(tr).items():
        path = OUT / f"local_{tag}.json"
        if path.exists():
            continue
        Xf, yf = X[idx, :FIT_T], y[idx, :FIT_T]
        t0 = time.time()
        b_mnl = mnl_fit(Xf, yf)
        b_nl, lam = nl_fit(Xf, yf, b_mnl)
        t_pooled = time.time() - t0
        B, t_hb = run_hb(Xf, yf, SEED)
        R = RFC(B, np.random.default_rng([SEED, 1]))
        t0 = time.time()
        (sl, g), tune_nll, flags = tune_rfc(R, X[idx, FIT_T:], y[idx, FIT_T:])
        rec = dict(tag=tag, n=len(idx), resp=idx.tolist(), mnl=b_mnl.tolist(), nl=b_nl.tolist(), nl_lambda=lam,
                   hb_mean=B.mean(0).tolist(), hb_sd=B.std(0).tolist(), rfc=dict(sigma_l=sl, g=g, tune_nll=tune_nll,
                   flags=flags), seconds=dict(pooled=t_pooled, hb=t_hb, tune=time.time() - t0))
        np.save(OUT / f"hb_{tag}.npy", B)
        path.write_text(json.dumps(rec, indent=1))
        print(f"{tag}: n {len(idx)}, NL lambda {lam:.2f}, HB {t_hb:.0f} s, RFC-S sigma_l {sl:.3f} g {g:.3f} {flags}",
              flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "local"
    if cmd == "local":
        local()
    elif cmd == "mvp":
        import t22_modal
        t22_modal.launch()
    elif cmd == "score":
        import t22_score
        t22_score.main()


if __name__ == "__main__":
    main()
