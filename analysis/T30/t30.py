"""T30. Client line + prices from a candidate set too large to enumerate: greedy + swap vs wide search (TEST.md).

    python analysis/T30/t30.py run --m 4,6 [--truths est,stress]

T29's world, survey, fits (rep 0) and RFC-S tuning, unchanged. The decision adds price: every SKU in the client's
line (the top 2 always in, the rest from the other 15) takes one of the survey's price levels, base x {0.8, ..., 1.2}.
Unit margin = price - 0.7 base (T29's 30% at base). A candidate is a (SKU, price) pair: 17 x 5 = 85. Lines at m:
C(15, m - 2) x 5^m (65,625 at m = 4; 2.1e7 at m = 6; 2.0e9 at m = 8).

Scorers (all on fixed draws, so lines are compared on common random numbers):
  SoP    HB-MNL posterior means, logit shares averaged over respondents
  RFC-S  HB-MNL + the tuned attribute-error sd + Gumbel, 100 draws per respondent
  MNP    the T29 MNP fit scored by a 100k-draw simulation (T29's MNP-sim)
  truth  the truth, 200k-draw simulation
Searches:
  greedy  start from the top 2 at base price; add the best (SKU, price) until m; then best-improvement 1-swap
          (replace any line member by any (SKU, price), the top 2 may only change price) until no move improves
  wide    every SKU subset; prices by coordinate ascent from base until no single price change improves
Truth optimum = the best truth value of any line found (truth wide, truth greedy, and every arm's line); at m = 4
also the exhaustive optimum. Regret = truth optimum - truth value of the arm's line, $ per 1,000 category trips.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parent / "T29"))
import t29 as T  # noqa: E402

COST = 1 - T.MARGIN
K = len(T.MULT)
BASE = 2                                    # index of mult 1.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Rows:
    """Draw rows (people x draws): U0 = best of the competitors + none, Uc (client SKU x price level, rows)."""

    def __init__(self, U0, Uc, prof):
        self.U0, self.prof = U0, prof
        self.Uc = np.ascontiguousarray(Uc.reshape(len(U0), -1).T)
        self.evals = 0

    def __call__(self, cs, ks):
        self.evals += 1
        X = self.Uc[cs * K + ks]
        a = X.argmax(0)
        win = X.max(0) > self.U0
        s = np.bincount(a[win], minlength=len(cs)) / len(a)
        return float(s @ self.prof[cs, ks])

    def sweep(self, cs, ks, i):
        """Values of line (cs, ks) with position i at each of the K prices: one pass for the other members."""
        self.evals += K
        o = np.r_[0:i, i + 1:len(cs)]
        n = len(self.U0)
        if len(o):
            X = self.Uc[cs[o] * K + ks[o]]
            a, mo = X.argmax(0), X.max(0)
            ow = mo > self.U0
            base = np.where(ow, mo, self.U0)
            po = np.where(ow, self.prof[cs[o], ks[o]][a], 0.0)
        else:
            base, po = self.U0, np.zeros(n)
        out = np.empty(K)
        for k in range(K):
            wi = self.Uc[cs[i] * K + k] > base
            out[k] = (wi.sum() * self.prof[cs[i], k] + po[~wi].sum()) / n
        return out


class Logit:
    def __init__(self, E, S0, prof):
        self.E, self.S0, self.prof = E, S0, prof
        self.evals = 0

    def __call__(self, cs, ks):
        self.evals += 1
        X = self.E[:, cs, ks]
        s = (X / (self.S0 + X.sum(1))[:, None]).mean(0)
        return float(s @ self.prof[cs, ks])

    def sweep(self, cs, ks, i):
        self.evals += K
        o = np.r_[0:i, i + 1:len(cs)]
        X = self.E[:, cs[o], ks[o]]
        so, num = self.S0 + X.sum(1), X @ self.prof[cs[o], ks[o]]
        x = self.E[:, cs[i], :]
        return ((num[:, None] + x * self.prof[cs[i]]) / (so[:, None] + x)).mean(0)


def setup(w):
    client = w["client"]
    comp = np.setdiff1d(np.arange(w["J"]), client)
    pc = w["price"][client][:, None] * T.MULT[None, :]                       # (C, K)
    prof = 1000 * (pc - COST * w["price"][client][:, None])
    return client, comp, pc, prof


def sim_rows(w, b, sd, shares, n, seed):
    """Truth / fitted-MNP draws as in t29.Sim, with each person's price coefficient kept for the client prices."""
    client, comp, pc, prof = setup(w)
    rng = np.random.default_rng(seed)
    J, lv = w["J"], w["levels"]
    beta = b + sd * rng.standard_normal((n, J + 1))
    e = np.sqrt(T.TVAR * shares[3]) * rng.standard_normal((n, J))
    for a in range(3):
        e += np.sqrt(T.TVAR * shares[a]) * rng.standard_normal((n, lv[:, a].max() + 1))[:, lv[:, a]]
    none = np.sqrt(T.TVAR) * rng.standard_normal(n)
    U0 = np.maximum((beta[:, comp] + beta[:, J:] * w["price"][comp] + e[:, comp]).max(1), none)
    Uc = beta[:, client, None] + beta[:, J, None, None] * pc[None] + e[:, client, None]
    return Rows(U0.astype(np.float32), Uc.astype(np.float32), prof)


def rfc_rows(w, R, sig):
    client, comp, pc, prof = setup(w)
    B, lv, J = R.B, R.lv, w["J"]
    err = R.g[:, :, :J].copy()
    for a in range(3):
        err += sig * R.z[a][:, :, lv[:, a]]
    V0 = B[:, None, comp] + B[:, -1:, None] * w["price"][comp] + err[:, :, comp]
    U0 = np.maximum(V0.max(2), R.g[:, :, J]).reshape(-1)
    Uc = (B[:, None, client, None] + B[:, -1:, None, None] * pc + err[:, :, client, None]).reshape(-1, len(client), K)
    return Rows(U0.astype(np.float32), Uc.astype(np.float32), prof)


def logit_rows(w, B):
    client, comp, pc, prof = setup(w)
    S0 = 1 + np.exp(B[:, comp] + B[:, -1:] * w["price"][comp]).sum(1)
    E = np.exp(B[:, client, None] + B[:, -1, None, None] * pc[None])
    return Logit(E, S0, prof)


def greedy(f, m):
    C = f.prof.shape[0]
    cs, ks = [0, 1], [BASE, BASE]
    while len(cs) < m:
        best = (-np.inf, None, None)
        for c in range(T.MUST, C):
            if c not in cs:
                v = f.sweep(np.r_[cs, c], np.r_[ks, BASE], len(cs))
                if v.max() > best[0]:
                    best = (v.max(), c, int(v.argmax()))
        cs, ks = cs + [best[1]], ks + [best[2]]
    cs, ks = np.array(cs), np.array(ks)
    v = f(cs, ks)
    while True:
        move = None
        for i in range(m):
            for c in ([cs[i]] if i < T.MUST else [c for c in range(T.MUST, C) if c == cs[i] or c not in cs]):
                c2 = cs.copy()
                c2[i] = c
                u = f.sweep(c2, ks, i)
                if c == cs[i]:
                    u[ks[i]] = -np.inf
                k = int(u.argmax())
                if u[k] > v + 1e-9 and (move is None or u[k] > move[0]):
                    k2 = ks.copy()
                    k2[i] = k
                    move = (u[k], c2, k2)
        if move is None:
            return cs.tolist(), ks.tolist(), v
        v, cs, ks = move


def ascend(f, cs):
    ks = np.full(len(cs), BASE)
    v = f(cs, ks)
    improved = True
    while improved:
        improved = False
        for i in range(len(cs)):
            u = f.sweep(cs, ks, i)
            k = int(u.argmax())
            if u[k] > v + 1e-9:
                v, ks[i], improved = u[k], k, True
    return ks, v


def wide(f, m):
    C = f.prof.shape[0]
    best = (-np.inf, None, None)
    for extra in itertools.combinations(range(T.MUST, C), m - T.MUST):
        cs = np.r_[0:T.MUST, extra].astype(int)
        ks, v = ascend(f, cs)
        if v > best[0]:
            best = (v, cs.tolist(), ks.tolist())
    return best[1], best[2], best[0]


def exhaustive(f, m):
    C = f.prof.shape[0]
    best = (-np.inf, None, None)
    for extra in itertools.combinations(range(T.MUST, C), m - T.MUST):
        cs = np.r_[0:T.MUST, extra].astype(int)
        for head in itertools.product(range(K), repeat=m - 1):
            ks = np.r_[head, BASE].astype(int)
            u = f.sweep(cs, ks, m - 1)
            k = int(u.argmax())
            if u[k] > best[0]:
                best = (u[k], cs.tolist(), list(head) + [k])
    return best[1], best[2], best[0]


def run(ms, tags, fit=""):
    w = T.world()
    b, sd = T.load_truth()
    res = []
    for tag in tags:
        sh = np.array(T.TRUTHS[tag])
        truth = sim_rows(w, b, sd, sh, 200_000, [T.SEED, 500, list(T.TRUTHS).index(tag)])
        d = T.make_data(tag, 0)
        B = np.load(T.OUT / f"hb_{tag}_r0.npz")["B"]
        R = T.RFC(B, w["levels"], np.random.default_rng([T.SEED, 99, 0]))
        sig, _ = T.rfc_tune(R, d, w)
        th = np.array(json.loads((T.OUT / f"mnp_{tag}_r0{fit}.json").read_text())["theta"])
        P = w["J"] + 1
        e = np.exp(np.r_[th[2 * P:], 0.0])
        fsh = e / e.sum() * (1 - 4e-6) + 1e-6
        scorers = {"SoP": logit_rows(w, B), "RFC-S": rfc_rows(w, R, sig),
                   "MNP": sim_rows(w, th[:P], np.exp(th[P:2 * P]), fsh, 100_000, [T.SEED, 600, 0])}
        log(f"{tag}: RFC-S sd {sig}; scorers built")
        for m in ms:
            found = {}
            for search, fn in (("greedy", greedy), ("wide", wide)):
                t0, e0 = time.time(), truth.evals
                cs, ks, v = fn(truth, m)
                found[("truth", search)] = (cs, ks, v, time.time() - t0, truth.evals - e0)
            if m == 4:
                t0 = time.time()
                cs, ks, v = exhaustive(truth, m)
                found[("truth", "exhaustive")] = (cs, ks, v, time.time() - t0, None)
            for name, f in scorers.items():
                for search, fn in (("greedy", greedy), ("wide", wide)):
                    t0, e0 = time.time(), f.evals
                    cs, ks, v = fn(f, m)
                    found[(name, search)] = (cs, ks, v, time.time() - t0, f.evals - e0)
            tv = {key: truth(np.array(cs), np.array(ks)) for key, (cs, ks, *_) in found.items()}
            opt = max(tv.values())
            for (name, search), (cs, ks, v, sec, ev) in found.items():
                row = dict(truth=tag, m=m, arm=name, search=search, line=[int(w["client"][c]) for c in cs],
                           mult=[float(T.MULT[k]) for k in ks], predicted=v, true=tv[(name, search)], opt=opt,
                           regret=opt - tv[(name, search)], seconds=sec, evals=ev)
                res.append(row)
                log(f"{tag} m={m} {name:5s} {search:10s}: regret {row['regret']:6.3f} $/1000 trips "
                    f"({100 * row['regret'] / opt:.2f}%), predicted {v:7.2f}, truth {row['true']:7.2f}, "
                    f"{ev or 0:7d} evals, {sec:5.0f} s, prices {''.join('LlBhH'[k] for k in ks)}")
    f = OUT / f"runs{fit}.json"
    old = json.loads(f.read_text()) if f.exists() else []
    keep = [r for r in old if (r["truth"], r["m"]) not in {(x["truth"], x["m"]) for x in res}]
    f.write_text(json.dumps(keep + res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--m", default="4,6")
    ap.add_argument("--truths", default="est,stress")
    ap.add_argument("--fit", default="", help="MNP fit file suffix, e.g. _bc")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    run([int(x) for x in a.m.split(",")], a.truths.split(","), a.fit)
