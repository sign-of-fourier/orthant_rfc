"""T4 recovery gate (DESIGN.md): J = 20, S1, N = 1,000, one dataset; MVP-F at two simulation sizes.
MVP-F (factor) or MVP-G (per-task GHK), --method. Pass: every error share and the mean of b within 2 SEs of the truth, and the two sizes within 1 SE.

    python gate.py [--N 1000] [--method G] [--sizes 256x128,512x256] [--chunk 2] [--tag g]
"""
import argparse
import json
import pathlib

import numpy as np

import mvp_gpu
import t4

OUT = pathlib.Path(__file__).parent / "out" / "gate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--J", type=int, default=20)
    ap.add_argument("--sizes", default="128x128,512x256")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--method", default="F")
    ap.add_argument("--tag", default="")
    ap.add_argument("--status", action="store_true", help="print progress checkpoints and exit")
    a = ap.parse_args()
    if a.status:
        import modal
        d = modal.Dict.from_name("orthant-rfc-t4-progress")
        for Q, R in [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]:
            k = f"gate_{a.tag}_{Q}x{R}"
            v = d.get(k)
            if v is None:
                print(k, "no checkpoint")
                continue
            v = {kk: vv for kk, vv in v.items() if kk != "theta"}
            print(k, v, "| fit result saved" if d.get(k + ":fit") else "", "| done" if d.get(k + ":done") else "")
        return
    out = OUT.with_name(OUT.name + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    d = t4.make_data("S1", a.J, a.N, 12, 0)
    X = t4.task_X(d["levels"], d["prices"]).astype(np.float32)
    A = t4.member(d["levels"]).astype(np.float32)
    s2 = t4.sig_of("S1") ** 2 / t4.T_VAR
    th_true = np.r_[t4.b_true(a.J), np.log(t4.W_TRUE), np.log(s2[:3] / s2[3])]
    th0 = np.r_[np.zeros(t4.P), np.log(np.full(t4.P, 0.5)), np.zeros(3)]
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]

    with mvp_gpu.app.run():
        calls = [mvp_gpu.fit.spawn(X, d["y"], A, Q, R, a.method, th0, th_true, 1, a.chunk,
                                      progress=f"gate_{a.tag}_{Q}x{R}") for Q, R in sizes]
        res = []
        for (Q, R), c in zip(sizes, calls):     # saved one by one: a later timeout keeps earlier fits
            res.append(c.get())
            (out / f"fit_{Q}x{R}.json").write_text(json.dumps(res[-1]))

    P = t4.P
    rows = []
    for (Q, R), r in zip(sizes, res):
        th, se = np.array(r["theta"]), np.array(r["se_sandwich"])
        z = (th - th_true) / se
        rows.append(dict(Q=Q, R=R, z=z.tolist(), seconds=r["seconds"], n_iter=r["n_iter"],
                         nll=r["nll"], nll_true=r["nll_true"]))
        print(f"Q={Q} R={R}: fit {r['seconds']['fit']:.0f}s se {r['seconds'].get('se', 0):.0f}s "
              f"iter {r['n_iter']} nll {r['nll']:.1f} (true {r['nll_true']:.1f})")
        print("  shares est", np.round(mvp_gpu.sig2_of(__import__("torch").tensor(th[2 * P:])).numpy()
                                         / t4.T_VAR, 3), "true", np.round(s2, 3))
        print("  |z| b max %.2f  w max %.2f  u %s" % (np.abs(z[:P]).max(), np.abs(z[P:2 * P]).max(),
                                                      np.round(z[2 * P:], 2)))
    t0, t1 = [np.array(r["theta"]) for r in res]
    agree = np.abs(t1 - t0) / np.array(res[1]["se_sandwich"])
    print("size agreement |diff|/SE max %.2f" % agree.max())
    (out / "gate.json").write_text(json.dumps(dict(N=a.N, J=a.J, method=a.method, th_true=th_true.tolist(), fits=res,
                                                   summary=rows, agree=agree.tolist()), indent=1))


if __name__ == "__main__":
    main()
