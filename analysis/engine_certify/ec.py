"""Engine-until-it-gets-bad (DESIGN.md): shrunk shelf probit, engine and GHK likelihoods, fits.

    python ec.py stage_a [--reps 3] [--J 8,12] [--truths weak,medium,shelf,neardup]
    python ec.py stage_b            # picks rho* from stage A, runs the hybrid
    python ec.py smoke              # timings and finite-difference smoothness, tiny
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent / "out"
for line in open(ROOT / ".env"):
    k, _, v = line.strip().partition("=")
    if k and not k.startswith("#"):
        os.environ.setdefault(k, v.strip().strip("'\""))
from multivariate_probit import orthant  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
f64 = torch.float64
SEED = 20261009
T_VAR = math.pi ** 2 / 6
PRICES = np.array([3.5, 4.0, 4.5, 5.0, 5.5])
PRICE_REF = 4.5
TRUTHS = {"weak": (.05, .10, .10, .75), "medium": (.10, .25, .25, .40),
          "shelf": (.10, .35, .35, .20), "neardup": (.05, .45, .45, .05)}
SHELVES = {8: (2, 2, 2), 12: (3, 2, 2)}
B_TRUE = {8: np.array([.5, .4, .3, -.975, 1.0]), 12: np.array([.5, .2, .4, .3, -.975, 1.3])}
U_START = np.log(np.array([.05, .05, .05]) / .85)      # weak correlation start
N_RESP, N_TASK = 300, 8
TOL = 0.25                                             # SE units


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Shelf:
    def __init__(self, J):
        self.J, self.K1 = J, J + 1
        nb, nf, npk = self.lev = SHELVES[J]
        self.levels = np.array([[b, f, p] for b in range(nb) for f in range(nf) for p in range(npk)])
        self.P = nb + nf + npk - 3 + 2
        off = (0, nb, nb + nf)
        self.A = np.zeros((self.K1, nb + nf + npk))
        self.stat = np.zeros((J, self.P - 2))
        for k, lv in enumerate(self.levels):
            for a in range(3):
                self.A[k, off[a] + lv[a]] = 1
            self.stat[k] = np.r_[np.eye(nb)[lv[0]][1:], np.eye(nf)[lv[1]][1:], np.eye(npk)[lv[2]][1:]]
        D = np.zeros((self.K1, J, self.K1))
        self.oth = np.array([[k for k in range(self.K1) if k != j] for j in range(self.K1)])
        for j in range(self.K1):
            D[j, np.arange(J), self.oth[j]] = 1
            D[j, :, j] = -1
        self.A_t, self.D_t = torch.tensor(self.A, dtype=f64), torch.tensor(D, dtype=f64)
        self.oth_t = torch.tensor(self.oth)

    def X(self, prices):
        N, T, J = prices.shape
        X = np.zeros((N, T, self.K1, self.P))
        X[:, :, :J, :self.P - 2] = self.stat
        X[:, :, :J, self.P - 2] = prices - PRICE_REF
        X[:, :, J, self.P - 1] = 1
        return X

    def covs(self, u):
        """Difference covariances (K1, J, J), one per chosen alternative. u torch (3,)."""
        s2 = T_VAR * torch.softmax(torch.cat([u, u.new_zeros(1)]), 0)
        sv = torch.cat([s2[a].expand(n) for a, n in enumerate(self.lev)])
        S = (self.A_t * sv) @ self.A_t.T + torch.diag(torch.cat([s2[3].expand(self.J), u.new_tensor([T_VAR])]))
        return self.D_t @ S @ self.D_t.transpose(1, 2)

    def rho_max(self, th):
        C = self.covs(torch.as_tensor(np.asarray(th)[self.P:], dtype=f64)).numpy()
        d = np.sqrt(np.einsum("kii->ki", C))
        R = C / d[:, :, None] / d[:, None, :]
        R[:, np.arange(self.J), np.arange(self.J)] = 0
        return float(R.max())

    def simulate(self, truth, rep):
        rng = np.random.default_rng([SEED, self.J, list(TRUTHS).index(truth), rep])
        prices = PRICES[rng.integers(0, 5, (N_RESP, N_TASK, self.J))]
        X = self.X(prices)
        V = X @ B_TRUE[self.J]
        sh = np.array(TRUTHS[truth]) * T_VAR
        sv = np.repeat(np.sqrt(sh[:3]), self.lev)
        e = (rng.standard_normal((N_RESP, N_TASK, self.A.shape[1])) * sv) @ self.A.T
        e[..., :self.J] += np.sqrt(sh[3]) * rng.standard_normal((N_RESP, N_TASK, self.J))
        e[..., self.J] = np.sqrt(T_VAR) * rng.standard_normal((N_RESP, N_TASK))
        y = np.argmax(V + e, 2)
        th = np.r_[B_TRUE[self.J], np.log(sh[:3] / sh[3])]
        return dict(X=X, y=y, th_true=th)


# ---------------------------------------------------------------------------
# Likelihoods
# ---------------------------------------------------------------------------
def _uppers(sh, X, y, b):
    """Upper limits (M, J) of U_k - U_chosen <= 0; works for numpy and torch b."""
    if isinstance(b, torch.Tensor):
        V = torch.as_tensor(X, dtype=f64).reshape(-1, sh.K1, sh.P) @ b
        yt = torch.as_tensor(y.ravel())
        Vj = V.gather(1, yt[:, None])
        return Vj - V.gather(1, sh.oth_t[yt])
    V = X.reshape(-1, sh.K1, sh.P) @ b
    yf = y.ravel()
    m = np.arange(len(yf))
    return V[m, yf][:, None] - V[m[:, None], sh.oth[yf]]


def ll_engine(sh, d, th, dup=0.0, resolution="high"):
    th = np.asarray(th)
    C = sh.covs(torch.as_tensor(th[sh.P:], dtype=f64)).numpy()
    up = _uppers(sh, d["X"], d["y"], th[:sh.P])
    p = orthant.cdf(up, C[d["y"].ravel()], dup_corr=dup, resolution=resolution)
    return float(np.log(np.clip(p, 1e-300, None)).sum())


def ll_engine_rows(sh, d, th, dup=0.0):
    th = np.asarray(th)
    C = sh.covs(torch.as_tensor(th[sh.P:], dtype=f64)).numpy()
    up = _uppers(sh, d["X"], d["y"], th[:sh.P])
    return np.log(np.clip(orthant.cdf(up, C[d["y"].ravel()], dup_corr=dup), 1e-300, None))


class GHK:
    """GHK with R scrambled-Sobol points, one random shift per choice, fixed for the fit."""

    def __init__(self, sh, d, R, seed=1):
        self.sh, self.d, self.R = sh, d, R
        self.M = d["y"].size
        g = torch.Generator().manual_seed(seed)
        self.base = torch.quasirandom.SobolEngine(sh.J - 1, scramble=True, seed=seed).draw(R, dtype=f64)
        self.shift = torch.rand(self.M, sh.J - 1, generator=g, dtype=f64)
        self.yt = torch.as_tensor(d["y"].ravel())
        self.chunk = max(N_TASK, (800_000 // (R * sh.J)) // N_TASK * N_TASK)   # whole respondents

    def rows(self, th, lo, hi):
        """log P per choice for choices lo:hi, (hi - lo,)."""
        sh = self.sh
        L = torch.linalg.cholesky(sh.covs(th[sh.P:]))[self.yt[lo:hi]]           # (m, J, J)
        ub = _uppers(sh, self.d["X"][lo // N_TASK:hi // N_TASK], self.d["y"][lo // N_TASK:hi // N_TASK], th[:sh.P])
        w = torch.remainder(self.base[None] + self.shift[lo:hi, None], 1.0)   # (m, R, J-1)
        m = hi - lo
        logp = ub.new_zeros(m, self.R)
        etas = []
        for i in range(sh.J):
            c = ub[:, i, None].expand(m, self.R)
            if i:
                c = c - (torch.stack(etas, -1) * L[:, None, i, :i]).sum(-1)
            lp = torch.special.log_ndtr(c / L[:, i, i, None])
            logp = logp + lp
            if i < sh.J - 1:
                etas.append(torch.special.ndtri((w[:, :, i] * lp.exp()).clamp(1e-300, 1 - 1e-15)))
        return torch.logsumexp(logp, 1) - math.log(self.R)

    def ll_grad(self, th_np):
        th = torch.as_tensor(th_np, dtype=f64).requires_grad_(True)
        tot = 0.0
        for lo in range(0, self.M, self.chunk):
            l = self.rows(th, lo, min(lo + self.chunk, self.M)).sum()
            l.backward()
            tot += l.item()
        return tot, th.grad.numpy().copy()

    def ll_rows_nograd(self, th_np):
        th = torch.as_tensor(th_np, dtype=f64)
        with torch.no_grad():
            return torch.cat([self.rows(th, lo, min(lo + self.chunk, self.M))
                              for lo in range(0, self.M, self.chunk)]).numpy()

    def sandwich(self, th_np):
        """Sandwich SEs at th: -H from autograd Hessian, scores per respondent."""
        th0 = torch.as_tensor(th_np, dtype=f64)
        nth = th0.numel()
        H = np.zeros((nth, nth))
        S = []
        for lo in range(0, self.M, self.chunk):
            hi = min(lo + self.chunk, self.M)
            H += torch.autograd.functional.hessian(lambda t: self.rows(t, lo, hi).sum(), th0).numpy()
            S.append(torch.autograd.functional.jacobian(
                lambda t: self.rows(t, lo, hi).view(-1, N_TASK).sum(1), th0).numpy())
        S = np.concatenate(S)
        Hi = np.linalg.inv(-H)
        V = Hi @ (S.T @ S) @ Hi
        return np.sqrt(np.clip(np.diag(V), 0, None))


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fit_engine(sh, d, th0, rho_stop=None, h=1e-4, dup=0.0, maxiter=200):
    """L-BFGS on the engine likelihood, central finite differences. Stops early (returns the last
    iterate) once rho_max(theta) > rho_stop. Returns theta, path [(theta, rho_max)], counts."""
    path, nev = [], [0]

    def fg(th):
        f = -ll_engine(sh, d, th, dup)
        g = np.empty_like(th)
        for i in range(len(th)):
            e = np.zeros_like(th)
            e[i] = h
            g[i] = -(ll_engine(sh, d, th + e, dup) - ll_engine(sh, d, th - e, dup)) / (2 * h)
        nev[0] += 1
        return f, g

    stopped = [False]

    def cb(intermediate_result):
        x = intermediate_result.x
        r = sh.rho_max(x)
        path.append((x.tolist(), r))
        if rho_stop is not None and r > rho_stop:
            stopped[0] = True
            raise StopIteration

    t0 = time.time()
    res = minimize(fg, th0, jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=maxiter, gtol=1e-5, ftol=1e-12))
    return dict(theta=res.x.tolist(), nll=float(res.fun), n_iter=int(res.nit), n_evals=nev[0],
                stopped=stopped[0], path=path, seconds=time.time() - t0, message=str(res.message))


def fit_qmc(ghk, th0, maxiter=200):
    nev = [0]

    def fg(th):
        f, g = ghk.ll_grad(th)
        nev[0] += 1
        return -f, -g

    t0 = time.time()
    res = minimize(fg, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-5, ftol=1e-12))
    return dict(theta=res.x.tolist(), nll=float(res.fun), n_iter=int(res.nit), n_evals=nev[0],
                seconds=time.time() - t0, message=str(res.message))


def start(sh):
    return np.r_[np.zeros(sh.P), U_START]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_a(args):
    OUT.mkdir(exist_ok=True)
    for J in [int(v) for v in args.J.split(",")]:
        sh = Shelf(J)
        for truth in args.truths.split(","):
            for rep in range(args.reps):
                f = OUT / f"a_{truth}_J{J}_r{rep}.json"
                if f.exists():
                    continue
                d = sh.simulate(truth, rep)
                eng = fit_engine(sh, d, start(sh))
                ghk = GHK(sh, d, args.R)
                q = fit_qmc(ghk, start(sh))
                se = ghk.sandwich(q["theta"])
                tE, tQ = np.array(eng["theta"]), np.array(q["theta"])
                z = np.abs(tE - tQ) / se
                calib = GHK(sh, d, 8192, seed=7).ll_rows_nograd(d["th_true"])
                diff = ll_engine_rows(sh, d, d["th_true"]) - calib
                rec = dict(truth=truth, J=J, rep=rep, th_true=d["th_true"].tolist(), engine=eng, qmc=q,
                           se=se.tolist(), D=float(z.max()), D_shares=float(z[sh.P:].max()), z=z.tolist(),
                           rho_true=sh.rho_max(d["th_true"]), rho_E=sh.rho_max(tE), rho_Q=sh.rho_max(tQ),
                           calib=dict(mean=float(diff.mean()), mean_abs=float(np.abs(diff).mean()),
                                      max_abs=float(np.abs(diff).max())))
                if truth == "neardup":
                    rec["engine_merge_on"] = fit_engine(sh, d, start(sh), dup=0.03)
                    rec["D_merge_on"] = float((np.abs(np.array(rec["engine_merge_on"]["theta"]) - tQ) / se).max())
                f.write_text(json.dumps(rec))
                print(f"{truth:8s} J={J:2d} r{rep}  D {rec['D']:.2f} (shares {rec['D_shares']:.2f})  "
                      f"rho_E {rec['rho_E']:.3f}  calib bias/choice {diff.mean():+.4f}  "
                      f"engine {eng['seconds']:.0f}s/{eng['n_iter']}it  qmc {q['seconds']:.0f}s/{q['n_iter']}it",
                      flush=True)


def shares(sh, th):
    u = np.asarray(th)[sh.P:]
    e = np.exp(np.r_[u, 0.0] - max(u.max(), 0.0))
    return e / e.sum()


def lr_dist(ghk, th, nll_ref):
    """sqrt(2 (nll_Q(th) - nll_Q(theta_Q))): QMC log-likelihood lost at th, as a distance in SE units
    (>= every per-parameter |z|, invariant to the parameterisation, fine at share = 0)."""
    return float(np.sqrt(max(0.0, 2 * (-ghk.ll_rows_nograd(th).sum() - nll_ref))))


def metrics(args):
    """Adds D_LR and share differences to every stage A record (idempotent)."""
    for p in sorted(OUT.glob("a_*.json")):
        r = json.loads(p.read_text())
        if "D_LR" in r:
            continue
        sh = Shelf(r["J"])
        d = sh.simulate(r["truth"], r["rep"])
        ghk = GHK(sh, d, args.R)
        nq = -ghk.ll_rows_nograd(r["qmc"]["theta"]).sum()
        r["nll_Q_at_Q"] = float(nq)
        r["D_LR"] = lr_dist(ghk, r["engine"]["theta"], nq)
        if "engine_merge_on" in r:
            r["D_LR_merge_on"] = lr_dist(ghk, r["engine_merge_on"]["theta"], nq)
        sE, sQ = shares(sh, r["engine"]["theta"]), shares(sh, r["qmc"]["theta"])
        r["shares_E"], r["shares_Q"], r["share_diff_max"] = sE.tolist(), sQ.tolist(), float(np.abs(sE - sQ).max())
        r["z_b_max"] = float(np.max(r["z"][:sh.P]))
        p.write_text(json.dumps(r))
        print(f"{r['truth']:8s} J={r['J']:2d} r{r['rep']}  D_LR {r['D_LR']:.2f}  |z_b| max {r['z_b_max']:.2f}  "
              f"shares E {np.round(sE, 3)} Q {np.round(sQ, 3)}  rho_E {r['rho_E']:.3f}"
              + (f"  merge-on D_LR {r['D_LR_merge_on']:.2f}" if "D_LR_merge_on" in r else ""), flush=True)


def stage_b(args):
    metrics(args)
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("a_*.json"))]
    rhos = sorted({r["rho_E"] for r in recs})
    ok = [rho for rho in rhos if all(r["D_LR"] <= TOL for r in recs if r["rho_E"] <= rho)]
    rho_star = max(ok) if ok else None
    print("rho* =", rho_star)
    if rho_star is None:
        return
    out = []
    for r in recs:
        sh = Shelf(r["J"])
        d = sh.simulate(r["truth"], r["rep"])
        ghk = GHK(sh, d, args.R)
        eng = fit_engine(sh, d, start(sh), rho_stop=rho_star)
        if eng["stopped"]:
            q = fit_qmc(ghk, np.array(eng["theta"]))
            th = np.array(q["theta"])
        else:
            q, th = None, np.array(eng["theta"])
        out.append(dict(truth=r["truth"], J=r["J"], rep=r["rep"], handoff=eng["stopped"],
                        engine_iter=eng["n_iter"], qmc_iter=q["n_iter"] if q else 0,
                        qmc_iter_cold=r["qmc"]["n_iter"], D_LR=lr_dist(ghk, th, r["nll_Q_at_Q"]),
                        share_diff_max=float(np.abs(shares(sh, th) - np.array(r["shares_Q"])).max())))
        print(out[-1], flush=True)
    (OUT / "b.json").write_text(json.dumps(dict(rho_star=rho_star, tol=TOL, runs=out), indent=1))


def smoke(args):
    for J in (8, 12):
        sh = Shelf(J)
        d = sh.simulate("shelf", 0)
        th = d["th_true"]
        t = time.time(); l = ll_engine(sh, d, th); te = time.time() - t
        fd = [(ll_engine(sh, d, th + h * np.eye(len(th))[-1]) - ll_engine(sh, d, th - h * np.eye(len(th))[-1])) / (2 * h)
              for h in (1e-3, 1e-4, 1e-5)]
        ghk = GHK(sh, d, args.R)
        t = time.time(); lq, gq = ghk.ll_grad(th); tq = time.time() - t
        print(f"J={J}: engine ll {l:.2f} ({te:.2f}s)  qmc ll {lq:.2f} (+grad {tq:.2f}s)  "
              f"d/du3 FD h=1e-3,4,5: {np.round(fd, 3)}  qmc grad {gq[-1]:.3f}  rho_true {sh.rho_max(th):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage_a", "stage_b", "smoke", "metrics"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--J", default="8,12")
    ap.add_argument("--truths", default=",".join(TRUTHS))
    ap.add_argument("--R", type=int, default=1024)
    a = ap.parse_args()
    {"stage_a": stage_a, "stage_b": stage_b, "smoke": smoke, "metrics": metrics}[a.stage](a)
