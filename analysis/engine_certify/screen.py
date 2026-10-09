"""ORTHANT_PLAN test S1: the engine as a screen over candidate scenarios, exact on the shortlist.

T4's J = 20 shelf and DGP at the truth. Candidates: A = assortment x near-duplicate extension (448),
B = 400 random firm price vectors. Engine margins (Q Sobol taste draws x one orthant per firm SKU)
vs T4's frequency simulator (10^6 draws, common random numbers, two seeds).

    python screen.py time                 # engine seconds per candidate
    python screen.py run [--truths shelf,neardup,medium,weak] [--Q 256] [--n_sim 1000000]
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
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "T4"))
import t4  # noqa: E402
import ec  # noqa: E402,F401  (loads .env for the engine)
from multivariate_probit import orthant  # noqa: E402

J = 20
OWN = list(range(6))
SHARES = {"weak": (.05, .10, .10, .75), "medium": (.10, .25, .25, .40),
          "shelf": (.10, .35, .35, .20), "neardup": (.05, .45, .45, .05),
          "g1": (.08, .45, .45, .02)}     # T1 G1: the clone correlates .98 with its original
KEEP = (0.01, 0.05, 0.10)
MATERIAL = 0.5
OUT = HERE / "out" / "screen"


def candidates():
    bp = t4.base_prices(J)
    out = []
    for mask in range(64):
        for ext in [None] + OWN:
            p = np.r_[bp, 0.0]
            m = np.r_[np.ones(J, bool), False]
            m[OWN] = [(mask >> k) & 1 == 1 for k in OWN]
            if ext is not None:
                p[J] = max(bp[ext] - t4.EXT_CUT, 0.0)
                m[J] = True
            out.append(dict(fam="A", prices=p, present=m, ext=ext))
    rng = np.random.default_rng([t4.SEED, 99])
    for _ in range(400):
        p = np.r_[bp, 0.0]
        p[OWN] = t4.PRICES[rng.integers(0, 5, 6)]
        out.append(dict(fam="B", prices=p, present=np.r_[np.ones(J, bool), False], ext=None))
    return out


def own_slots(s):
    return [j for j in OWN if s["present"][j]] + ([J] if s["ext"] is not None else [])


def margin(s, sh):
    """sh: shares indexed like slots (J+1 SKU/clone slots; extra entries ignored)."""
    return sum((s["prices"][j] - t4.COST) * sh[j] for j in own_slots(s))


class Engine:
    def __init__(self, sig, Q, seed=3):
        self.sig = np.asarray(sig)
        z = torch.quasirandom.SobolEngine(t4.P, scramble=True, seed=seed).draw(Q, dtype=torch.float64).numpy()
        from scipy.special import ndtri
        self.beta = t4.b_true(J) + t4.W_TRUE * ndtri(np.clip(z, 1e-12, 1 - 1e-12))   # (Q, P)
        self.levels = t4.ALL[t4.shelf(J)]

    def shares(self, s):
        """Population shares of the firm's present slots: (J+1,) with zeros elsewhere."""
        lv = t4.scen_levels(self.levels, s)                    # (J+1, 3), slot J = clone
        alts = [j for j in range(J + 1) if s["present"][j]]
        X = t4.task_X(lv, s["prices"])                         # (J+2, P), last row = none
        idx = alts + [J + 1]
        V = self.beta @ X[idx].T                               # (Q, K)
        A = t4.member(lv)[idx]                                 # (K, NLEV), none row zero
        sv = np.repeat(self.sig[:3], t4.LEVELS)
        S = (A * sv ** 2) @ A.T + np.diag(np.r_[np.full(len(alts), self.sig[3] ** 2), t4.T_VAR])
        K = len(idx)
        mine = [i for i, j in enumerate(alts) if j in own_slots(s)]
        if not mine:
            return np.zeros(J + 1)
        ups, covs = [], []
        for i in mine:
            o = [k for k in range(K) if k != i]
            D = np.zeros((K - 1, K))
            D[np.arange(K - 1), o] = 1
            D[:, i] = -1
            ups.append(V[:, [i]] - V[:, o])                    # (Q, K-1)
            covs.append(np.broadcast_to(D @ S @ D.T, (len(V), K - 1, K - 1)))
        p = orthant.cdf(np.concatenate(ups), np.concatenate(covs), dup_corr=0.0)
        out = np.zeros(J + 1)
        for n, i in enumerate(mine):
            out[alts[i]] = p[n * len(V):(n + 1) * len(V)].mean()
        return out


def reference(sig, cands, n_sim, seed):
    sh = t4.shares_probit(t4.b_true(J), t4.W_TRUE, np.asarray(sig), t4.ALL[t4.shelf(J)],
                          [dict(prices=c["prices"], present=c["present"], ext=c["ext"]) for c in cands],
                          n_sim, seed)
    return np.array([margin(c, sh[i]) for i, c in enumerate(cands)])


def evaluate(eng, r1, r2):
    ref = (r1 + r2) / 2
    best = ref.max()
    noise = 100 * 2 * np.std(r1 - r2) / best                  # regret % below this is noise
    reg = 100 * (best - ref) / best
    order = np.argsort(-eng)
    out = dict(n=len(ref), ref_noise_pct=float(noise), spearman=float(spearmanr(eng, ref)[0]),
               rank_of_best=int(np.where(order == ref.argmax())[0][0]) + 1,
               engine_pick_regret=float(reg[order[0]]))
    for k in KEEP:
        top = order[:max(1, math.ceil(k * len(ref)))]
        out[f"screen_{k:.2f}_regret"] = float(reg[top[np.argmax(ref[top])]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "run"])
    ap.add_argument("--truths", default="shelf,neardup,medium,weak")
    ap.add_argument("--Q", type=int, default=256)
    ap.add_argument("--n_sim", type=int, default=1_000_000)
    a = ap.parse_args()
    cands = candidates()
    if a.mode == "time":
        eng = Engine(np.sqrt(t4.T_VAR * np.array(SHARES["shelf"])), a.Q)
        for i in (63 * 7, 63 * 7 + 3, 500):
            t = time.time(); sh = eng.shares(cands[i])
            print(f"cand {i} ({cands[i]['fam']}): {time.time() - t:.2f}s  shares {np.round(sh[own_slots(cands[i])], 4)}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for truth in a.truths.split(","):
        f = OUT / f"{truth}.json"
        if f.exists():
            continue
        sig = np.sqrt(t4.T_VAR * np.array(SHARES[truth]))
        t0 = time.time()
        eng_obj = Engine(sig, a.Q)
        eng = np.array([margin(c, eng_obj.shares(c)) for c in cands])
        t_eng = time.time() - t0
        t0 = time.time()
        r1, r2 = reference(sig, cands, a.n_sim, 11), reference(sig, cands, a.n_sim, 12)
        t_ref = time.time() - t0
        fam = np.array([c["fam"] for c in cands])
        res = dict(truth=truth, Q=a.Q, n_sim=a.n_sim, seconds=dict(engine=t_eng, reference=t_ref),
                   engine=eng.tolist(), ref1=r1.tolist(), ref2=r2.tolist())
        for F in ("A", "B"):
            m = fam == F
            res[F] = evaluate(eng[m], r1[m], r2[m])
            rm = (r1 + r2) / 2
            mm = m & (rm > 0)
            res[F]["engine_margin_bias_pct"] = float(100 * np.mean((eng[mm] - rm[mm]) / rm[mm]))
            print(truth, F, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[F].items()}),
                  flush=True)
        print(f"{truth}: engine {t_eng:.0f}s, reference {t_ref:.0f}s", flush=True)
        f.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
