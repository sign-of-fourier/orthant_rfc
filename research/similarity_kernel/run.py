"""One replication: every truth x every fix. Usage: python run.py [seed] [N]."""
import json
import sys
import time

import numpy as np

import sim

FIXES = ["none", "rfc", "kernel"]


def twin_share(P, task):
    return P[:, list(task["twins"])].sum(1).mean()


def replication(seed, N=400, R_oracle=2000):
    rng = np.random.default_rng(seed)
    tasks = sim.make_holdouts(rng)
    B = sim.respondents(N, rng)
    fit_i, ev_i = np.arange(N // 2), np.arange(N // 2, N)
    out = []
    for truth, prm in sim.TRUTH_DEFAULTS.items():
        Y = sim.simulate_choices(tasks, B, truth, prm, rng)
        actual = [np.bincount(Y[ev_i, t], minlength=sim.K) / len(ev_i) for t in range(len(tasks))]
        rep = [i for i, t in enumerate(tasks) if t["kind"] == "repeat"]
        retest = np.mean([np.abs(np.bincount(Y[:, i], minlength=5) - np.bincount(Y[:, tasks[i]["of"]], minlength=5)).mean() / N
                          for i in rep])
        oracle = [sim.true_probs(t, B[ev_i], truth, prm, R_oracle, rng) for t in tasks]
        rows = {"oracle": dict(P=oracle, th=prm, ll=np.sum([np.log(np.maximum(oracle[t][np.arange(len(ev_i)), Y[ev_i, t]], 1e-4)).sum()
                                                            for t in range(len(tasks))]), sec=0.0)}
        for model in FIXES:
            t0 = time.time()
            th, _, res = sim.fit(tasks, B[fit_i], Y[fit_i], model)
            P = [sim.fix_probs(t, B[ev_i], model, th) for t in tasks]
            ll = sum(np.log(np.maximum(P[t][np.arange(len(ev_i)), Y[ev_i, t]], 1e-10)).sum() for t in range(len(tasks)))
            rows[model] = dict(P=P, th=th, ll=ll, sec=time.time() - t0, nfev=res.nfev)
        for name, r in rows.items():
            for ti, t in enumerate(tasks):
                if t["kind"] == "repeat":
                    continue
                pred = r["P"][ti].mean(0)
                out.append(dict(seed=seed, truth=truth, fix=name, task=ti, k=t["k"], kind=t["kind"], d=t["d_twin"],
                                twin_pred=float(pred[list(t["twins"])].sum()),
                                twin_act=float(actual[ti][list(t["twins"])].sum()),
                                mae=float(np.abs(pred - actual[ti]).mean()), retest=float(retest),
                                ll=float(r["ll"]), th={k: float(v) for k, v in r["th"].items()},
                                sec=float(r["sec"]), nfev=int(r.get("nfev", 0))))
        print(truth, {m: (round(rows[m]["ll"], 1), {k: round(float(v), 3) for k, v in rows[m]["th"].items()},
                          round(rows[m]["sec"], 1)) for m in rows}, flush=True)
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rows = replication(seed, N)
    json.dump(rows, open(f"results/rep_{seed}_{N}.json", "w"))
