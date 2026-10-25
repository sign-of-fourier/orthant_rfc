"""T37: T34's assortment win (MNP vs the incumbent) against logits that estimate the error dependence (TEST.md).
S1 truth, J = 20, N = 1,000, 12 tasks, reps 0-4; MNP picks, truth best and T34's incumbent arms reused.

Arms:
  ec   error-component mixed logit: diagonal normal tastes + normal error components per brand / flavor / pack
       level per choice occasion + one on "none", logit kernel; simulated ML on the GPU.
  cnl  pooled cross-nested logit: one nest per brand / flavor / pack level, allocation per attribute and nest
       parameter per attribute estimated; "none" in its own nest.

    python t37.py run 0              # Modal H100: fits, exhaustive D1 searches, truth scores
    python t37.py check              # Modal H100: EC fit on the logit truth S0 (EC sds should go to 0)
    python t37.py summary
"""
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
T34 = HERE.parent / "T34"
if len(HERE.parents) > 2:                                      # host only
    for d in ("T4", "T15", "T34"):
        sys.path.insert(0, str(HERE.parent / d))

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy")
         .add_local_python_source("t4", "t15", "t34"))
app = modal.App("orthant-rfc-t37", image=image)

OUT = HERE / "out"
N_RESP, T_TASKS, J = 1000, 12, 20
R_B, R_E = 250, 500                     # EC fit: taste draws per respondent, error-component draws per occasion
R_POP, R_POP_E = 20000, 5               # EC decision: population taste draws x error-component draws
CHUNK = 5                               # respondents per likelihood chunk
M_TRUTH = 65536
MATERIAL = 0.5
ARMS = ("ec", "cnl")


# ---------------------------------------------------------------------------
# Error-component mixed logit
# ---------------------------------------------------------------------------
def ec_unpack(th, P):
    b, lw, ls = th[:P], th[P:2 * P], th[2 * P:]
    return b, lw.exp(), ls.exp()                                   # ls: brand, flavor, pack, none


def ec_fit(d, seed, dev, maxiter=400):
    import torch
    from scipy.optimize import minimize
    import t4
    X = torch.as_tensor(t4.task_X(d["levels"], d["prices"]), dtype=torch.float32, device=dev)     # (N, T, 21, P)
    N, T, K1, P = X.shape
    y = torch.as_tensor(d["y"], device=dev)
    A = torch.as_tensor(t4.member(d["levels"]), dtype=torch.float32, device=dev)                  # (21, 12)
    lvl = torch.as_tensor(np.repeat(np.arange(3), t4.LEVELS), device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    Zb = torch.randn(N, R_B, P, generator=g, device=dev)
    Ze = torch.randn(N, T, R_E, t4.NLEV + 1, generator=g, device=dev)                             # last: none
    Xy = X.gather(2, y[:, :, None, None].expand(-1, -1, 1, P))[:, :, 0]                            # (N, T, P)

    def nll(th_np, grad=True):
        th = torch.as_tensor(th_np, dtype=torch.float64, device=dev).requires_grad_(grad)
        tot, gr = 0.0, np.zeros_like(th_np)
        for lo in range(0, N, CHUNK):                                  # graph per chunk: unpack inside the loop
            b, w, s = ec_unpack(th, P)
            b32, w32 = b.float(), w.float()
            sl = s[:3].float()[lvl]                                                                # (12,)
            sl_ = slice(lo, lo + CHUNK)
            beta = b32 + w32 * Zb[sl_]                                                             # (n, R_B, P)
            V = torch.einsum("ntkp,nrp->nrtk", X[sl_], beta)                                       # (n, R_B, T, 21)
            Vy = torch.einsum("ntp,nrp->nrt", Xy[sl_], beta)
            E = (Ze[sl_, :, :, :-1] * sl) @ A.T                                                    # (n, T, R_E, 21)
            E = torch.cat([E[..., :-1], E[..., -1:] + s[3].float() * Ze[sl_, :, :, -1:]], -1)
            Ey = E.gather(3, y[sl_, :, None, None].expand(-1, -1, R_E, 1))[..., 0]                 # (n, T, R_E)
            U = V[:, :, :, None, :] + E[:, None]                                                   # (n, R_B, T, R_E, 21)
            lp = Vy[:, :, :, None] + Ey[:, None] - torch.logsumexp(U, -1)                          # (n, R_B, T, R_E)
            lt = torch.logsumexp(lp, -1) - np.log(R_E)                                             # (n, R_B, T)
            ln = torch.logsumexp(lt.sum(-1).double(), -1) - np.log(R_B)                            # (n,)
            loss = -ln.sum()
            if grad:
                gg, = torch.autograd.grad(loss, th)
                gr += gg.cpu().numpy()
            tot += float(loss.detach())
        return (tot, gr) if grad else tot

    th0 = np.r_[np.zeros(P), np.full(P, np.log(0.3)), np.zeros(4)]          # EC sds start at 1: the score is 0 at sd = 0
    th0[P - 2] = -1.0
    t0 = time.time()
    r = minimize(nll, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-5))
    return dict(theta=r.x.tolist(), nll=float(r.fun), nit=int(r.nit), success=bool(r.success), msg=str(r.message),
                seconds=time.time() - t0)


def ec_exp_utils(fit, inst, seed, dev):
    """(R_POP * R_POP_E, 21) exp utilities of the 20 shelf SKUs + none at base prices, population draws."""
    import torch
    import t4
    lev = t4.ALL[np.array(inst["shelf"])]
    X = torch.as_tensor(t4.task_X(lev, np.array(inst["bp"])), device=dev)                         # (21, P)
    P = X.shape[1]
    b, w, s = ec_unpack(torch.as_tensor(fit["theta"], dtype=torch.float64, device=dev), P)
    g = torch.Generator(device=dev).manual_seed(seed)
    beta = b + w * torch.randn(R_POP, P, generator=g, device=dev, dtype=torch.float64)
    V = (beta @ X.T)[:, None, :].expand(-1, R_POP_E, -1)                                           # (R, Re, 21)
    sl = s[:3][torch.as_tensor(np.repeat(np.arange(3), t4.LEVELS), device=dev)]
    A = torch.as_tensor(t4.member(lev), device=dev)                                                # (21, 12)
    Z = torch.randn(R_POP, R_POP_E, t4.NLEV + 1, generator=g, device=dev, dtype=torch.float64)
    E = (Z[..., :-1] * sl) @ A.T
    E[..., -1] += s[3] * Z[..., -1]
    U = (V + E).reshape(-1, 21)
    return (U - U.max(1, keepdim=True)[0]).exp()


# ---------------------------------------------------------------------------
# Pooled cross-nested logit (nests = brand / flavor / pack levels; none alone)
# ---------------------------------------------------------------------------
def cnl_logp(V, A, th, present=None):
    """V (..., 21) utilities, none last; A (21, 12) level membership; th: alloc logits (3), nest logit-mu (3).
    Returns log choice probabilities (..., 21). present (..., 21) bool masks absent SKUs."""
    import torch
    import t4
    lvl = torch.as_tensor(np.repeat(np.arange(3), t4.LEVELS), device=V.device)
    la = torch.log_softmax(th[:3], 0)[lvl]                                                         # (12,) log alloc
    mu = torch.sigmoid(th[3:6])[lvl].clamp_min(1e-3)                                               # (12,)
    Vs = V[..., :-1]
    t = (la + Vs[..., None]) / mu                                                                  # (..., 20, 12)
    mask = A[:-1].bool()
    if present is not None:
        mask = mask & present[..., :-1, None]
    t = t.masked_fill(~mask, -torch.inf)
    lG = torch.logsumexp(t, -2)                                                                    # (..., 12)
    lD = torch.logsumexp(torch.cat([mu * lG, V[..., -1:]], -1), -1, keepdim=True)
    term = (t - lG[..., None, :] + mu * lG[..., None, :]).masked_fill(~mask, -torch.inf)          # empty nests: nan
    lp = torch.logsumexp(term, -1) - lD                                                            # (..., 20)
    return torch.cat([lp, V[..., -1:] - lD], -1)


def cnl_fit(d, dev, maxiter=500):
    import torch
    from scipy.optimize import minimize
    import t4
    X = torch.as_tensor(t4.task_X(d["levels"], d["prices"]), device=dev).reshape(-1, J + 1, t4.P)
    y = torch.as_tensor(d["y"].reshape(-1), device=dev)
    A = torch.as_tensor(t4.member(d["levels"]), device=dev)
    P = t4.P

    def nll(th_np):
        th = torch.as_tensor(th_np, device=dev).requires_grad_(True)
        lp = cnl_logp(X @ th[:P], A, th[P:])
        loss = -lp.gather(1, y[:, None]).sum()
        g, = torch.autograd.grad(loss, th)
        return float(loss.detach()), g.cpu().numpy()

    th0 = np.r_[np.zeros(P), np.zeros(3), np.zeros(3)]
    th0[P - 2] = -1.0
    t0 = time.time()
    r = minimize(nll, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-6))
    return dict(theta=r.x.tolist(), nll=float(r.fun), nit=int(r.nit), success=bool(r.success), msg=str(r.message),
                seconds=time.time() - t0)


def cnl_d1_values(fit, inst, mg, cand, dev, chunk=4096):
    import torch
    import t4
    lev = t4.ALL[np.array(inst["shelf"])]
    X = torch.as_tensor(t4.task_X(lev, np.array(inst["bp"])), device=dev)
    th = torch.as_tensor(fit["theta"], dtype=torch.float64, device=dev)
    V = X @ th[:t4.P]
    A = torch.as_tensor(t4.member(lev), device=dev)
    out = torch.empty(cand.shape[0], dtype=torch.float64, device=dev)
    for lo in range(0, cand.shape[0], chunk):
        c = cand[lo:lo + chunk]
        pres = torch.zeros(len(c), J + 1, dtype=torch.bool, device=dev).scatter_(1, c, True)
        pres[:, -1] = True
        lp = cnl_logp(V.expand(len(c), -1), A, th[t4.P:], pres)
        out[lo:lo + chunk] = (lp[:, :-1].exp().nan_to_num(0.0) * mg).sum(1)
    return out


# ---------------------------------------------------------------------------
# Modal
# ---------------------------------------------------------------------------
def run_impl(m, insts, rep, truth="S1", search=True):
    import torch
    import t4
    import t15
    import t34
    from itertools import combinations
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = t4.make_data(truth, J, N_RESP, T_TASKS, rep)
    fits = dict(ec=ec_fit(d, 3700 + rep, dev), cnl=cnl_fit(d, dev))
    out = dict(rep=rep, truth=truth, fits=fits, inst={})
    if not search:
        return out
    th = torch.as_tensor(m["th_true"], dtype=torch.float64, device=dev)
    for inst in insts:
        mg = torch.as_tensor(np.array(inst["bp"]) - np.array(inst["cost"])[inst["shelf"]], device=dev)
        Xs = torch.as_tensor(np.array(list(combinations(range(20), inst["k"]))), device=dev)
        r = {}
        t0 = time.perf_counter()
        v = t34.d1_values(ec_exp_utils(fits["ec"], inst, 900 + rep, dev), mg, Xs)
        i = int(v.argmax())
        r["ec"] = dict(pick=Xs[i].tolist(), own_value=float(v[i]), seconds=time.perf_counter() - t0)
        t0 = time.perf_counter()
        v = cnl_d1_values(fits["cnl"], inst, mg, Xs, dev)
        i = int(v.argmax())
        r["cnl"] = dict(pick=Xs[i].tolist(), own_value=float(v[i]), seconds=time.perf_counter() - t0)
        sh = t15.Shelf(m, inst, dev)
        tv = t15.score(sh, t15.GHKScorer(sh, th, M_TRUTH), torch.as_tensor([r[a]["pick"] for a in ARMS], device=dev))
        for a, x in zip(ARMS, tv.cpu().numpy()):
            r[a]["true_value"] = float(x)
        out["inst"][inst["name"]] = r
    return out


@app.function(gpu="H100", timeout=3 * 3600)
def run_rep(m, insts, rep, truth="S1", search=True):
    return run_impl(m, insts, rep, truth, search)


def run(reps):
    import t34
    OUT.mkdir(parents=True, exist_ok=True)
    m, insts = t34.setup()
    with modal.enable_output(), app.run():
        for res in run_rep.starmap([(m, insts, rep) for rep in reps], order_outputs=False):
            rep = res["rep"]
            t34r = json.loads((T34 / "out" / f"S1_rep{rep}.json").read_text())
            for n, r in res["inst"].items():
                o = t34r["inst"][n]
                best = max([o["best_true"]] + [r[a]["true_value"] for a in ARMS])
                reg = lambda v: 100 * (best - v) / best      # noqa: E731
                r["best_true"] = best
                r["mnp"] = dict(pick=o["mnp"]["pick"], regret=reg(o["t15_rescored"]["mnp"]))
                r["rfc_draws"] = dict(pick=o["rfc_draws"]["pick"], regret=reg(o["rfc_draws"]["true_value"]))
                for a in ARMS:
                    r[a]["regret"] = reg(r[a]["true_value"])
                    r[a]["forecast"] = 100 * (r[a]["own_value"] - r[a]["true_value"]) / r[a]["true_value"]
            (OUT / f"S1_rep{rep}.json").write_text(json.dumps(res, indent=1))
            print(f"rep {rep}: " + "  ".join(f"{a} fit {f['seconds']:.0f} s nit {f['nit']} {f['msg'][:30]}"
                                             for a, f in res["fits"].items()))
            print(f"  EC theta: {np.round(res['fits']['ec']['theta'], 3).tolist()}")
            print(f"  CNL theta: {np.round(res['fits']['cnl']['theta'], 3).tolist()}")
            for n, r in res["inst"].items():
                print(f"  {n}: MNP {r['mnp']['regret']:.2f}  RFC-S draws {r['rfc_draws']['regret']:.2f}  "
                      + "  ".join(f"{a} {r[a]['regret']:.2f} (fc {r[a]['forecast']:+.1f}%)" for a in ARMS), flush=True)


def check():
    """EC and CNL on the logit truth S0: EC error-component sds and CNL nesting should vanish."""
    import t34
    m, insts = t34.setup()
    with modal.enable_output(), app.run():
        res = run_rep.remote(m, insts, 0, "S0", False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check_S0.json").write_text(json.dumps(res, indent=1))
    P = 11
    th = np.array(res["fits"]["ec"]["theta"])
    print(f"S0 EC: {res['fits']['ec']['seconds']:.0f} s, nit {res['fits']['ec']['nit']}; error-component sds "
          f"(brand, flavor, pack, none) {np.round(np.exp(th[2 * P:]), 3).tolist()}")
    tc = np.array(res["fits"]["cnl"]["theta"])
    print(f"S0 CNL: mu {np.round(1 / (1 + np.exp(-tc[P + 3:])), 3).tolist()}")


def summary():
    from scipy import stats
    reps = [json.loads(p.read_text()) for p in sorted(OUT.glob("S1_rep*.json"))]
    k = len(reps)
    q = stats.t.ppf(0.975, k - 1)
    rows = {}
    for a in ("rfc_draws",) + ARMS:
        reg = [np.mean([r["inst"][n][a]["regret"] for n in r["inst"]]) for r in reps]
        per = [np.mean([r["inst"][n][a]["regret"] - r["inst"][n]["mnp"]["regret"] for n in r["inst"]]) for r in reps]
        dd, se = float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(k))
        p = float(2 * stats.t.sf(abs(dd) / se, k - 1)) if se > 0 else 0.0
        rows[a] = dict(regret=float(np.mean(reg)), delta=dd, se=se, lo=dd - q * se, hi=dd + q * se, p=p, per=per)
    mnp = np.mean([np.mean([r["inst"][n]["mnp"]["regret"] for n in r["inst"]]) for r in reps])
    print(f"{k} datasets x 3 instances; MNP regret {mnp:.2f}")
    for a, r in rows.items():
        print(f"{a:10s} regret {r['regret']:.2f}  delta {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] SE {r['se']:.2f}"
              f"  p {r['p']:.3f}  per " + " ".join(f"{x:.2f}" for x in r["per"]))
    b = rows["ec"]
    v = ("WIN IS THE PROBIT" if b["delta"] > 2 * b["se"] and b["delta"] >= MATERIAL else
         "WIN IS ESTIMATING THE DEPENDENCE" if b["delta"] < MATERIAL else "INCONCLUSIVE")
    print(f"EC delta {b['delta']:+.2f} (SE {b['se']:.2f}) -> {v}")
    (OUT / "summary.json").write_text(json.dumps(dict(rows=rows, verdict=v), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "run":
        run([int(x) for x in sys.argv[2].split(",")])
    elif cmd == "check":
        check()
    else:
        summary()
