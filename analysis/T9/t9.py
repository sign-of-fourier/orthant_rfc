"""T9: MACML (Bhat's pairwise composite likelihood, OVUS) vs MVP by GPU GHK on T1's DGP.

    python analysis/T9/t9.py check            # local CPU: torch OVUS port vs pybhatlib on T1 pair orthants
    python analysis/T9/t9.py gate             # G1 rep 0, N = 600, Godambe SEs; vs analysis/T1/out/step0_G1
    python analysis/T9/t9.py run --truth G2   # 20 reps, N = 600, no SEs; GHK arm from T1/out/T5_<truth>_N600
    python analysis/T9/t9.py summary

Design and kill criteria: PLAN.md T9. Data per replicate are T1's own (make_data on the replicate's
r_data stream), so the GHK fits of T5's N = 600 cells are the paired comparison arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T1 = HERE.parent / "T1"
sys.path.insert(0, str(T1))
import run  # noqa: E402  (T1: DGP, parameterisation, shares, decisions, score)
from ghk_gpu import panel_arrays  # noqa: E402

N, T, REPS = 600, 12, 20
FLIPS = ("price_flip", "ext_flip_at_placeholder_h", "cost_flip")
LOGF = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


def sig_of(truth):
    return np.sqrt(run.T_VAR * run.TRUTHS[truth])


def data(truth, rep):
    r_data = np.random.default_rng(np.random.SeedSequence([run.SEED, rep]).spawn(3)[0])
    d = run.make_data(r_data, N, T, sig_of(truth), truth in run.LOGIT_TRUTHS)
    return panel_arrays(d["prods"], d["y"], run.task_X)


def th0():
    return np.r_[np.zeros(run.P), np.full(run.P, np.log(.5)), np.zeros(3)]


def se_log_sigma(th, V):
    """Delta method, as run.replicate: log sigma_a = (log T_VAR + u_a - logsumexp(u, 0)) / 2."""
    u = np.asarray(th)[2 * run.P:]
    J = 0.5 * (np.eye(3) - run.softmax0(u)[None, :3])
    return np.sqrt(np.diag(J @ np.asarray(V)[2 * run.P:, 2 * run.P:] @ J.T))


def err_corr(s):
    """Error correlations among T1's 36 products (all level combinations), off-diagonal."""
    import itertools
    prods = np.array([[b, f, p, 5.0] for b, f, p in itertools.product(*[range(L) for L in run.LEVELS])])
    E = run.err_cov(prods, s)[:-1, :-1]
    d = np.sqrt(np.diag(E))
    C = E / np.outer(d, d)
    return C[~np.eye(len(C), dtype=bool)]


TYPES = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]   # attributes two distinct products share


def corr_types(u):
    """Error correlation of two products by the attributes they share (the distinct off-diagonal values
    of err_corr): sum of those attributes' variance shares, as run.unpack_total."""
    sh = run.softmax0(np.asarray(u)) * (1 - 4e-6) + 1e-6
    return np.array([sh[list(t)].sum() if t else 0.0 for t in TYPES])


def corr_jac(u, h=1e-6):
    u = np.asarray(u, float)
    return np.stack([(corr_types(u + h * e) - corr_types(u - h * e)) / (2 * h) for e in np.eye(3)], 1)


def ghk_cov_rep0():
    """GHK sandwich covariance for step0_G1 rep 0 (step0 stored only diagonal SEs): the same fit as
    run.ghk_fits (M = 4096, seed SEED + rep, total scale), rerun with SEs."""
    import ghk_gpu
    import modal
    sig = sig_of("G1")
    Xd, G = data("G1", 0)
    th0g = np.r_[np.zeros(run.P), np.full(run.P, np.log(.5)), np.zeros(3)]
    with modal.enable_output(), ghk_gpu.app.run():
        r = ghk_gpu.fit.remote(Xd, G, 0.0, th0g, run.theta_true(sig, "total"), 4096, run.SEED + 0,
                               tvar=float(run.T_VAR), se=True)
    log(f"GHK rep 0 refit with SEs: {r['seconds']['fit']:.0f} s + SE {r['seconds']['se']:.0f} s")
    return r


def check(args):
    """CPU: the composite likelihood's pair orthants at the GHK fit, torch port vs pybhatlib OVUS."""
    import torch
    from macml import cl_rows, ovus_log, pair_index
    from pybhatlib.gradmvn._mvncd import _mvncd_ovus

    Xd, G = data("G2", 0)
    n = 20
    th = np.array(json.loads((T1 / "out" / "T5_G2_N600" / "results.json").read_text())[0]["mvp"]["theta"])
    Xt, Gt = torch.tensor(Xd[:n]), torch.tensor(G[:, :n])
    IDX = torch.as_tensor(pair_index(T, 3))
    thr = torch.tensor(th).expand(n, -1)
    cl = cl_rows(thr, Xt, Gt, float(run.T_VAR), IDX).numpy()
    # rebuild the pair orthants in numpy and evaluate each with pybhatlib. Exact |a/sd| ties are
    # compared separately: OVUS depends on the variable order, and pybhatlib's tie order is numpy's
    # default argsort (not stable on numpy 2.x); the port breaks ties stably.
    b, w, s = run.unpack_total(th)
    A, S = [], []
    for i in range(n):
        Z = Xd[i].reshape(T * 3, run.P)
        eta = Z @ b
        C = Z @ np.diag(w ** 2) @ Z.T
        for t in range(T):
            C[3 * t:3 * t + 3, 3 * t:3 * t + 3] += np.einsum("a,aij->ij", s ** 2, G[:, i, t])
        for ix in IDX.numpy():
            A.append(eta[ix])
            S.append(C[np.ix_(ix, ix)])
    A, S = np.array(A), np.array(S)
    ref = np.log([_mvncd_ovus(a, c) for a, c in zip(A, S)])
    got = ovus_log(torch.tensor(A), torch.tensor(S)).numpy()
    z = np.abs(A / np.sqrt(np.diagonal(S, axis1=1, axis2=2)))
    tie = np.array([len(np.unique(r)) < len(r) for r in z])
    d = np.abs(got - ref)
    log(f"check (G2 rep 0, {n} respondents x 66 pairs = {len(A)} orthants, at the GHK fit): composite ll "
        f"{cl.sum():.6f}, sum of torch orthants {got.sum():.6f}, pybhatlib {ref.sum():.6f}")
    log(f"  no exact tie ({(~tie).sum()}): max |dlog P| {d[~tie].max():.2e}; exact ties ({tie.sum()}): "
        f"max |dlog P| {d[tie].max() if tie.any() else 0:.2e}")
    ok = d[~tie].max() < 1e-6   # the composite ll vs the orthant sum differs only by tie order (arrays built in torch vs numpy)
    log("CHECK PASS" if ok else "CHECK FAIL")


def fits(jobs, se):
    import macml
    import modal
    t0 = time.time()
    with modal.enable_output(), macml.app.run():
        res = list(macml.fit.starmap(jobs, kwargs=dict(se=se)))
    log(f"GPU fits done: {time.time() - t0:.0f} s wall; GPU fit-seconds "
        f"{sum(r['seconds']['fit'] + r['seconds']['se'] for r in res):.0f}")
    return res


def gate(args):
    sig = sig_of("G1")
    ghk = json.loads((T1 / "out" / "step0_G1" / "results.json").read_text())[0]["mvp"]
    Xd, G = data("G1", 0)
    log("gate: G1 rep 0, N = 600, MACML (OVUS pairs) with Godambe SEs, vs step0_G1 rep 0 GHK")
    r = fits([(Xd, G, th0(), run.theta_true(sig, "total"), float(run.T_VAR))], se=True)[0]
    th, V = np.array(r["theta"]), np.array(r["cov_sandwich"])
    se = np.sqrt(np.diag(V))
    thg, seg = np.array(ghk["theta"]), np.array(ghk["se_sandwich"])
    P = run.P
    zb = (th[:P] - thg[:P]) / np.sqrt(se[:P] ** 2 + seg[:P] ** 2)
    zw = (th[P:2 * P] - thg[P:2 * P]) / np.sqrt(se[P:2 * P] ** 2 + seg[P:2 * P] ** 2)
    s_m, s_g = run.unpack_total(th)[2], np.array(ghk["sigma"])
    sls_m, sls_g = se_log_sigma(th, V), np.array(ghk["se_log_sigma"])
    zs = (np.log(s_m[:3]) - np.log(s_g[:3])) / np.sqrt(sls_m ** 2 + sls_g ** 2)
    cerr = float(np.abs(err_corr(s_m) - err_corr(s_g)).max())
    # correlation gate (user, 2026-10-10: option 1): every distinct correlation's MACML - GHK gap within
    # 2 SE of the difference, delta method on u with each fit's sandwich (MACML Godambe, GHK sandwich)
    g = ghk_cov_rep0()
    if np.abs(np.array(g["theta"]) - thg).max() > 1e-6:
        log(f"  WARNING: GHK refit theta differs from step0 by {np.abs(np.array(g['theta']) - thg).max():.2e}")
    um, ug = th[2 * P:], np.array(g["theta"])[2 * P:]
    Vm, Vg = V[2 * P:, 2 * P:], np.array(g["cov_sandwich"])[2 * P:, 2 * P:]
    Jm, Jg = corr_jac(um), corr_jac(ug)
    gap = corr_types(um) - corr_types(ug)
    se_gap = np.sqrt(np.diag(Jm @ Vm @ Jm.T) + np.diag(Jg @ Vg @ Jg.T))
    zc = np.where(se_gap > 0, gap / np.where(se_gap > 0, se_gap, 1), 0.0)
    for t, a, b_, z in zip(TYPES, gap, se_gap, zc):
        if t:
            log(f"  corr, shared {'+'.join(('brand', 'flavor', 'pack')[i] for i in t):18s} gap {a:+.3f} (SE {b_:.3f}) z {z:+.2f}")
    log(f"  fit {r['seconds']['fit']:.0f} s + SE {r['seconds']['se']:.0f} s, {r['n_iter']} iters, "
        f"max|grad| {r['grad_max']:.1e}; ncl {r['ncl']:.1f} vs truth {r['ncl_true']:.1f}; order rounds "
        f"{r['order_rounds']}, orthants re-sorted {r['order_changes']}, iters {r['iters_per_round']}")
    for nm, a, b_, s1, s2, z in zip(run.NAMES, th[:P], thg[:P], se[:P], seg[:P], zb):
        log(f"  b {nm:7s} MACML {a:+.3f} ({s1:.3f})  GHK {b_:+.3f} ({s2:.3f})  z {z:+.2f}")
    for nm, a, b_, z in zip(("brand", "flavor", "pack"), s_m ** 2 / run.T_VAR, s_g ** 2 / run.T_VAR, zs):
        log(f"  share {nm:6s} MACML {a:.3f}  GHK {b_:.3f}  z(log sigma) {z:+.2f}")
    log(f"  taste sds (reported, not gated): z {np.round(zw, 2).tolist()}")
    log(f"  implied error correlation max |diff| {cerr:.3f} (reported; gate is |z| <= 2 per correlation)")
    ok = bool(np.all(np.abs(zb) <= 2) and np.all(np.abs(zs) <= 2) and np.all(np.abs(zc) <= 2))
    log("GATE PASS" if ok else "GATE FAIL: STOP and diagnose")
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "gate.json").write_text(json.dumps(dict(
        macml=r, ghk_theta=thg.tolist(), ghk_se=seg.tolist(), z_b=zb.tolist(), z_logsigma=zs.tolist(),
        z_logw=zw.tolist(), share_macml=(s_m ** 2 / run.T_VAR).tolist(), share_ghk=(s_g ** 2 / run.T_VAR).tolist(),
        corr_max_abs_diff=cerr, corr_types=[list(t) for t in TYPES], corr_gap=gap.tolist(), corr_gap_se=se_gap.tolist(),
        z_corr=zc.tolist(), ghk_refit=dict(theta=g["theta"], cov_sandwich=g["cov_sandwich"], seconds=g["seconds"]),
        PASS=ok), indent=1))


def diag(args):
    """Gate diagnosis (not an arm): reps 0-2 of step0_G1 (GHK fits with SEs), MACML from th0 and from
    the GHK theta; composite ll at both optima, shares, correlation gap."""
    sig = sig_of("G1")
    ghk = json.loads((T1 / "out" / "step0_G1" / "results.json").read_text())
    jobs, keys = [], []
    for rep in range(3):
        Xd, G = data("G1", rep)
        for start in ("th0", "ghk"):
            st = th0() if start == "th0" else np.array(ghk[rep]["mvp"]["theta"])
            jobs.append((Xd, G, st, np.array(ghk[rep]["mvp"]["theta"]), float(run.T_VAR)))
            keys.append((rep, start))
    res = fits(jobs, se=False)
    out = []
    for (rep, start), r in zip(keys, res):
        s_m, s_g = run.unpack_total(np.array(r["theta"]))[2], np.array(ghk[rep]["mvp"]["sigma"])
        row = dict(rep=rep, start=start, ncl=r["ncl"], ncl_at_ghk=r["ncl_true"], n_iter=r["n_iter"],
                   grad_max=r["grad_max"], share_macml=(s_m ** 2 / run.T_VAR).tolist(),
                   share_ghk=(s_g ** 2 / run.T_VAR).tolist(), corr_diff=float(np.abs(err_corr(s_m) - err_corr(s_g)).max()))
        out.append(row)
        log(f"rep {rep} start {start}: ncl {r['ncl']:.2f} (at GHK theta {r['ncl_true']:.2f}), {r['n_iter']} it, "
            f"max|grad| {r['grad_max']:.1e}; shares b/f/p MACML {np.round(row['share_macml'][:3], 3).tolist()} "
            f"GHK {np.round(row['share_ghk'][:3], 3).tolist()}; corr diff {row['corr_diff']:.3f}")
    (HERE / "out" / "diag.json").write_text(json.dumps(out, indent=1))


def run_truth(args):
    truth = args.truth
    sig = sig_of(truth)
    ghk = json.loads((T1 / "out" / f"T5_{truth}_N600" / "results.json").read_text())
    log(f"T9 {truth}: {REPS} reps, N {N}; MACML on GPU in parallel; GHK arm from T5_{truth}_N600")
    jobs = [(*data(truth, rep), th0(), run.theta_true(sig, "total"), float(run.T_VAR)) for rep in range(REPS)]
    res = fits(jobs, se=False)
    recs = []
    for rep, r in enumerate(res):
        g = ghk[rep]
        assert g["rep"] == rep and g["N"] == N
        b, w, s = run.unpack_total(np.array(r["theta"]))
        dec = run.decisions(lambda pr: run.pop_shares(pr, b, w, s))
        sc = run.score(dec, g["truth"])
        m = dict(theta=r["theta"], ncl=r["ncl"], n_iter=r["n_iter"], grad_max=r["grad_max"], seconds=r["seconds"],
                 S1=float(run.pair_corr_probit(s)), flips=int(sum(sc[k] for k in FLIPS)), **sc)
        gm = dict(S1=g["mvp"]["S1"], seconds=g["mvp"]["seconds"], flips=int(sum(g["mvp"][k] for k in FLIPS)),
                  **{k: g["mvp"][k] for k in ("price_flip", "price_regret_pct", "ext_flip_at_placeholder_h",
                                              "cost_flip", "cost_regret_pct")})
        recs.append(dict(rep=rep, truth_S1=g["truth"]["S1"], macml=m, ghk=gm))
        log(f"rep {rep}: MACML flips {m['flips']} regret {m['cost_regret_pct']:.2f}% ({r['seconds']['fit']:.0f} s, "
            f"{r['n_iter']} it)  GHK flips {gm['flips']} regret {gm['cost_regret_pct']:.2f}%")
    out = HERE / "out" / f"full_{truth}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(truth=truth, N=N, records=recs), indent=1))
    log("done")


def summary(args):
    out = {}
    for truth in ("G0", "G1", "G2"):
        p = HERE / "out" / f"full_{truth}" / "results.json"
        if not p.exists():
            continue
        R = json.loads(p.read_text())["records"]
        reg = {a: np.array([r[a]["cost_regret_pct"] for r in R]) for a in ("ghk", "macml")}
        d = reg["macml"] - reg["ghk"]                       # > 0: GHK better
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        row = dict(R=len(R), regret={a: float(v.mean()) for a, v in reg.items()},
                   flips={a: int(sum(r[a]["flips"] for r in R)) for a in ("ghk", "macml")},
                   macml_minus_ghk=(float(m), float(se)),
                   S1_err={a: float(np.mean([r[a]["S1"] - r["truth_S1"] for r in R])) for a in ("ghk", "macml")},
                   fit_s={a: [float(min(x)), float(np.median(x)), float(max(x))] for a, x in
                          {a: [r[a]["seconds"]["fit"] for r in R] for a in ("ghk", "macml")}.items()})
        if truth == "G2":
            row["WIN"] = bool(m > 2 * se)
        if truth == "G1":
            row["FLAG_macml_beats_ghk"] = bool(-m > 2 * se)
        out[truth] = row
        print(f"\n## {truth}  (R = {len(R)})")
        for a in ("ghk", "macml"):
            print(f"  {a:6s} cost regret {row['regret'][a]:.2f}%  flips {row['flips'][a]}  S1 err {row['S1_err'][a]:+.3f}  "
                  f"fit s min/med/max {np.round(row['fit_s'][a]).astype(int).tolist()}")
        print(f"  MACML - GHK regret {m:+.2f} (paired SE {se:.2f})"
              + (f"  WIN (GHK better by > 2 SE): {row['WIN']}" if truth == "G2" else "")
              + (f"  FLAG (MACML better by > 2 SE): {row['FLAG_macml_beats_ghk']}" if truth == "G1" else ""))
    (HERE / "out" / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "gate", "diag", "run", "summary"])
    ap.add_argument("--truth", choices=list(run.TRUTHS), default="G2")
    a = ap.parse_args()
    if a.stage != "summary":
        (HERE / "out").mkdir(exist_ok=True)
        LOGF = open(HERE / f"{a.stage}{'_' + a.truth if a.stage == 'run' else ''}.log", "w")
    dict(check=check, gate=gate, diag=diag, run=run_truth, summary=summary)[a.stage](a)
