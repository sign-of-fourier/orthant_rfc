"""X1 heterogeneity phase: latent classes (design: DESIGN.md, "Heterogeneity phase").

Arms (same splits as the pooled phase):
  LC-L-RFC   latent-class logit (EM) + RFC tuned on validation
  LC-P-LIN   latent-class linear probit, shared variance-share Sigma (EM)
  LC-P-XGB   latent-class tree probit, shared variance-share Sigma (EM, weighted boosting per class)
HB-RFC is taken from the pooled run (out/splits).

EM: person n's likelihood is sum_k pi_k prod_t P(y_nt | class k). E-step: class posteriors tau_nk from
training tasks. M-step: refit each class's utility with task weights tau_{n(t)k}; update pi; for the
probit arms refit the shared shares with utilities fixed. Prediction for a test task of person n:
sum_k tau_nk P(y | class k), with tau from the person's training tasks.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (COLS, F64, SEED, Probit, RFC, fit_xgb, probit_choice_probs, shares_from_logits,  # noqa: E402
                  xgb_V)


def resp_index(data):
    ids, inv = np.unique(data.resp, return_inverse=True)
    return ids, inv


def estep(task_ll, inv, n_resp, log_pi):
    """task_ll (T, K) -> tau (n_resp, K), total log-likelihood."""
    L = np.zeros((n_resp, task_ll.shape[1]))
    np.add.at(L, inv, task_ll)
    L = L + log_pi
    m = L.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(L - m).sum(1))
    return np.exp(L - lse[:, None]), float(lse.sum())


# --------------------------------------------------------------------------- weighted M-steps
def mnl_weighted(data, w, b0=None, iters=50):
    X = torch.as_tensor(data.X)
    y = torch.as_tensor(data.y)
    wt = torch.as_tensor(w, dtype=F64)
    b = torch.zeros(len(COLS), dtype=F64) if b0 is None else torch.tensor(b0, dtype=F64)
    b.requires_grad_(True)
    opt = torch.optim.LBFGS([b], max_iter=iters, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        loss = (wt * torch.nn.functional.cross_entropy(X @ b, y, reduction="none")).sum()
        loss.backward()
        return loss

    opt.step(cl)
    return b.detach().numpy().copy()


def mnl_task_ll(data, b):
    V = data.X @ b
    V = V - V.max(1, keepdims=True)
    return V[np.arange(data.T), data.y] - np.log(np.exp(V).sum(1))


def probit_lin_weighted(pr, X, w, shares, b0, iters=30):
    wt = torch.as_tensor(w, dtype=F64)
    s = torch.as_tensor(shares, dtype=F64)
    b = torch.tensor(b0, dtype=F64, requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=iters, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        loss = -(wt * pr.loglik(X @ b, s)).sum()
        loss.backward()
        return loss

    opt.step(cl)
    return b.detach().numpy().copy()


def shares_weighted(pr, Vs, W, w0, iters=30):
    """Shared shares given class utilities Vs[k] (T, 5) and weights W (T, K)."""
    Vs = [torch.as_tensor(v) for v in Vs]
    Wt = torch.as_tensor(W, dtype=F64)
    w = torch.tensor(w0, dtype=F64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad()
        s = shares_from_logits(w)
        loss = -sum((Wt[:, k] * pr.loglik(Vs[k], s)).sum() for k in range(len(Vs)))
        loss.backward()
        return loss

    opt.step(cl)
    return w.detach().numpy().copy()


# --------------------------------------------------------------------------- EM drivers
def init_tau(n_resp, K, rng):
    return rng.dirichlet(np.ones(K), size=n_resp)


def em_mnl(data, K, rng, iters=200, tol=1e-6):
    ids, inv = resp_index(data)
    tau = init_tau(len(ids), K, rng)
    B = [None] * K
    prev = -np.inf
    for it in range(iters):
        B = [mnl_weighted(data, tau[inv, k], B[k]) for k in range(K)]
        log_pi = np.log(tau.mean(0))
        tll = np.stack([mnl_task_ll(data, b) for b in B], 1)
        tau, ll = estep(tll, inv, len(ids), log_pi)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return dict(B=B, pi=np.exp(log_pi), tau=dict(zip(ids, tau)), ll=ll, iters=it + 1)


def em_probit_lin(data, K, rng, w0, b0, M=128, iters=40, tol=1e-5):
    ids, inv = resp_index(data)
    pr = Probit(data, M)
    X = torch.as_tensor(data.X)
    tau = init_tau(len(ids), K, rng)
    # starting betas: pooled beta jittered per class
    B = [b0 + 0.2 * rng.standard_normal(len(b0)) for _ in range(K)]
    w = np.asarray(w0, float)
    prev = -np.inf
    for it in range(iters):
        s = shares_from_logits(torch.as_tensor(w)).numpy()
        B = [probit_lin_weighted(pr, X, tau[inv, k], s, B[k]) for k in range(K)]
        Vs = [data.X @ b for b in B]
        w = shares_weighted(pr, Vs, tau[inv], w)
        s = shares_from_logits(torch.as_tensor(w)).numpy()
        with torch.no_grad():
            tll = np.stack([pr.loglik(torch.as_tensor(v), torch.as_tensor(s)).numpy() for v in Vs], 1)
        log_pi = np.log(tau.mean(0))
        tau, ll = estep(tll, inv, len(ids), log_pi)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return dict(B=B, w=w, shares=s, pi=np.exp(log_pi), tau=dict(zip(ids, tau)), ll=ll, iters=it + 1)


def em_probit_xgb(data, K, rng, w0, tau0, rounds, depth=3, M=128, iters=8, tol=1e-4):
    """Generalized EM: each M-step boosts each class from scratch with task weights (fixed rounds, from
    the pooled P-XGB early stopping), then refits the shared shares. tau0: starting posteriors (from
    LC-P-LIN, so both arms start from the same classes)."""
    ids, inv = resp_index(data)
    pr = Probit(data, M)
    tau = np.stack([tau0[i] for i in ids])
    w = np.asarray(w0, float)
    prev = -np.inf
    models = None
    for it in range(iters):
        s = shares_from_logits(torch.as_tensor(w)).numpy()
        models = [fit_xgb(data, None, s, depth, rounds=rounds, weights=tau[inv, k])[0] for k in range(K)]
        Vs = [xgb_V(m, data, rounds) for m in models]
        w = shares_weighted(pr, Vs, tau[inv], w)
        s = shares_from_logits(torch.as_tensor(w)).numpy()
        with torch.no_grad():
            tll = np.stack([pr.loglik(torch.as_tensor(v), torch.as_tensor(s)).numpy() for v in Vs], 1)
        log_pi = np.log(tau.mean(0))
        tau, ll = estep(tll, inv, len(ids), log_pi)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return dict(models=models, w=w, shares=s, pi=np.exp(log_pi), tau=dict(zip(ids, tau)), ll=ll, iters=it + 1)


# --------------------------------------------------------------------------- prediction
def tau_for(fit, data):
    K = len(fit["pi"])
    return np.stack([fit["tau"].get(r, fit["pi"]) for r in data.resp]).reshape(data.T, K)


def predict_mnl_rfc(fit, test, sl, g, seed):
    rfc = RFC(test, seed=seed)
    tau = tau_for(fit, test)
    return sum(tau[:, [k]] * rfc.probs(np.tile(b, (test.T, 1)), sl, g) for k, b in enumerate(fit["B"]))


def tune_lc_rfc(fit, val, seed, lo=(0.01, 0.05), hi=(3.0, 5.0)):
    """RFC (sl, g) maximizing validation log-likelihood of the class mixture of RFC shares."""
    from scipy.optimize import minimize
    rfc = RFC(val, seed=seed)
    tau = tau_for(fit, val)
    llo, lhi = np.log(lo), np.log(hi)

    def nll(x):
        x = np.clip(x, llo, lhi)
        P = sum(tau[:, [k]] * rfc.probs(np.tile(b, (val.T, 1)), *np.exp(x)) for k, b in enumerate(fit["B"]))
        return -np.log(np.maximum(P[np.arange(val.T), val.y], 1e-300)).mean()

    grid = [(a, b) for a in np.linspace(llo[0], lhi[0], 6) for b in np.linspace(llo[1], lhi[1], 6)]
    x0 = np.array(grid[int(np.argmin([nll(np.array(x)) for x in grid]))])
    r = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-7, maxiter=200))
    return np.exp(np.clip(r.x, llo, lhi))


def predict_probit_lin(fit, test):
    tau = tau_for(fit, test)
    return sum(tau[:, [k]] * probit_choice_probs(test, test.X @ b, fit["shares"]) for k, b in enumerate(fit["B"]))


def predict_probit_xgb(fit, test, rounds):
    tau = tau_for(fit, test)
    return sum(tau[:, [k]] * probit_choice_probs(test, xgb_V(m, test, rounds), fit["shares"])
               for k, m in enumerate(fit["models"]))


def val_ll(P, data):
    return float(np.mean(np.log(np.maximum(P[np.arange(data.T), data.y], 1e-300))))
