"""T20 sample-size curves: MVP (GHK, GPU) vs HB-MNL + RFC-S, T5's generator, models and decisions.

    python analysis/T20/run.py fit --truths G1 --Ns 50,1500 --reps 1     # smoke (ask first: paid)
    python analysis/T20/run.py fit                                        # full grid (ask first: paid)
    python analysis/T20/run.py assemble                                   # local: score decisions
    python analysis/T20/summarize.py

fit launches both arms on Modal at once (modal_t20.py) and writes one JSON per cell replicate to
out/mvp/ and out/inc/; existing files are skipped, so a stopped run resumes. assemble scores each arm's
decisions against the truth (orthant binary for the probit truths, mixed logit for G0) with T1's score()
and writes out/results.json.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modal_t20 as m  # noqa: E402

OUT = HERE / "out"
NS = (50, 75, 100, 150, 200, 300, 450, 600, 800, 1000, 1500)
TRUTHS = ("G1", "G2", "G0")
REPS = 20


def path(arm, truth, N, rep):
    return OUT / arm / f"{truth}_N{N}_r{rep:02d}.json"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "log.txt", "a") as f:
        f.write(line + "\n")


def fit(cells, arms=("mvp", "inc")):
    jobs = {arm: [c for c in cells if not path(arm, *c).exists()] for arm in arms}
    log(f"fit: " + ", ".join(f"{len(j)} {a} jobs" for a, j in jobs.items()) + f" ({len(cells)} cells per arm requested)")
    for arm in jobs:
        (OUT / arm).mkdir(parents=True, exist_ok=True)
    fns = {"mvp": m.mvp, "inc": m.incumbent}

    def drain(arm):
        todo = jobs[arm]
        if not todo:
            return
        done = 0
        for res in fns[arm].starmap(todo, order_outputs=False, return_exceptions=True,
                                    wrap_returned_exceptions=False):
            if isinstance(res, Exception):
                log(f"{arm} FAILED: {res!r}"[:500])
                continue
            path(arm, res["truth"], res["N"], res["rep"]).write_text(json.dumps(res))
            done += 1
            secs = res["seconds"]["fit"] if arm == "mvp" else res["seconds"]["hb"] + res["seconds"]["tune"]
            log(f"{arm} {done}/{len(todo)}: {res['truth']} N={res['N']} rep {res['rep']}, {secs:.0f} s")

    t0 = time.time()
    with m.app.run():
        th = [threading.Thread(target=drain, args=(a,)) for a in jobs]
        for t in th:
            t.start()
        for t in th:
            t.join()
    log(f"fit done: {time.time() - t0:.0f} s wall")


def assemble():
    t1 = m.t1_module()
    log(f"assemble: orthant backend {t1.rfc.BACKEND}")
    truth_dec = {}
    out = []
    for truth in TRUTHS:
        sig = np.sqrt(t1.T_VAR * t1.TRUTHS[truth])
        for N in NS:
            for rep in range(REPS):
                pm, pi = path("mvp", truth, N, rep), path("inc", truth, N, rep)
                if not (pm.exists() or pi.exists()):
                    continue
                if truth not in truth_dec:
                    fn = (t1.mixed_logit_shares if truth in t1.LOGIT_TRUTHS
                          else (lambda pr, s=sig: t1.pop_shares(pr, t1.B_TRUE, t1.W_TRUE, s)))
                    truth_dec[truth] = t1.decisions(fn)
                tdec = truth_dec[truth]
                rec = dict(truth=truth, N=N, rep=rep, mvp=None, rfc_s=None)
                rm = json.loads(pm.read_text()) if pm.exists() else None
                ri = json.loads(pi.read_text()) if pi.exists() else None
                if rm and ri and rm["y_sha"] != ri["y_sha"]:
                    raise SystemExit(f"{truth} N={N} rep {rep}: arms saw different data")
                if rm:
                    b, w, s_hat = t1.unpack_total(np.array(rm["theta"]))
                    mdec = t1.decisions(lambda pr: t1.pop_shares(pr, b, w, s_hat))
                    rec["mvp"] = dict(converged=rm["n_iter"] < rm["max_iter"], iters=rm["n_iter"],
                                      grad_max=rm["grad_max"], nll=rm["nll"], nll_true=rm["nll_true"],
                                      sigma=s_hat.tolist(), seconds=rm["seconds"], gpu=rm["gpu"],
                                      **t1.score(mdec, tdec))
                if ri:
                    rec["rfc_s"] = dict(sigma_l=ri["sigma_l"], g=ri["g"], holdout_mae=ri["holdout_mae"],
                                        flags=ri["flags"], hb_loglike_drift=ri["hb_loglike_drift"],
                                        seconds=ri["seconds"], **t1.score(ri["decisions"], tdec))
                out.append(rec)
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    log(f"assemble: {len(out)} records -> out/results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "assemble"])
    ap.add_argument("--truths", default=",".join(TRUTHS))
    ap.add_argument("--Ns", default=",".join(map(str, NS)))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--arms", default="mvp,inc", help="mvp, inc or both")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.cmd == "fit":
        cells = [(t, int(n), r) for t in a.truths.split(",") for n in a.Ns.split(",") for r in range(a.reps)]
        fit(cells, tuple(a.arms.split(",")))
    else:
        assemble()


if __name__ == "__main__":
    main()
