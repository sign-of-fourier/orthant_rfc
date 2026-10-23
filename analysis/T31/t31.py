"""T31. B5: MNP fit at large J (design: analysis/T31/TEST.md).

    python analysis/T31/t31.py fit  --truths est,stress --J 20,40,60 --reps 0      # Modal H100: per-task GHK fits
                                    [--tol 1e-6]                                   # looser stop (file suffix _tol)
    python analysis/T31/t31.py hb   --truths est,stress --J 20,40,60 --reps 0      # Modal T4: HB-MNL (T29 sampler)
    python analysis/T31/t31.py partA                                                # local: holdout shares vs truth
    python analysis/T31/t31.py partB --reps 0,1,2,3,4                               # local: T30 line + price decision
    python analysis/T31/t31.py hbcomp --reps 0,...,9                                # Modal T4: HB-MNL per compartment
    python analysis/T31/t31.py partC --reps 0,...,9                                 # local: compartmentalized incumbent

Truths: est, stress (T29's probit truths) and mxl (HB-MNL's own model: est's tastes, iid Gumbel errors).
World and survey: T29's (J = 60 uses T29's make_data, so rep 0 matches T29's fits); J < 60 keeps the J SKUs with
the largest truth shares at base prices (plus none), data re-simulated on that shelf.
Fit: per-task composite likelihood. Tastes are folded into each task's covariance; each choice is one orthant
P(chosen beats the other J alternatives), dimension J, by GHK at 1,024 scrambled Sobol points (Genz order fixed at
the start), L-BFGS with autograd gradients on an H100. Parameters: J + 1 means, J + 1 log taste sds, and for stress
the 3 variance-share logits (est fixes iid errors).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE      # /root inside Modal containers
sys.path.insert(0, str(ROOT / "analysis" / "T29"))
sys.path.insert(0, str(ROOT / "analysis" / "T30"))
try:
    import t29 as T  # noqa: E402
except ModuleNotFoundError:      # inside Modal containers
    T = None
TVAR = np.pi ** 2 / 6
GHK_M = 1024
TAGS = ["est", "stress", "mxl"]          # mxl: HB-MNL's own model (est's tastes, iid Gumbel errors, same variance)
IID = ("est", "mxl")


def sfx(tol):
    return "" if tol == 1e-8 else f"_tol{tol:g}"


def simulate(rng, prices, beta, tag, levels):
    """T29's simulate; mxl draws iid Gumbel errors (variance pi^2 / 6, as the probit truths' total) instead."""
    if tag != "mxl":
        return T.simulate(rng, prices, beta, T.TRUTHS[tag], levels)
    n, t, J = prices.shape
    U = np.concatenate([beta[:, None, :J] + beta[:, None, J:] * prices, np.zeros((n, t, 1))], 2)
    return (U + rng.gumbel(size=U.shape)).argmax(2)

gpu_image = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")
             .add_local_file(ROOT / "analysis" / "T18" / "t18.py", "/root/t18.py"))
app = modal.App("orthant-rfc-t31")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data (local)
# ---------------------------------------------------------------------------
def shelf(J):
    """Indices of the J SKUs kept: largest truth shares at base prices (est truth, 50k-draw simulation)."""
    w = T.world()
    if J >= w["J"]:
        return np.arange(w["J"])
    b, sd = T.load_truth()
    sim = T.Sim(b, sd, T.TRUTHS["est"], w["levels"], w["price"], 50_000, [T.SEED, 31])
    s = sim.shares(np.arange(w["J"]), np.arange(w["J"]))
    return np.sort(np.argsort(-s, kind="stable")[:J])


def make_data(tag, rep, J):
    """J = 60: T29's make_data. J < 60: the same construction on the kept SKUs, seeds keyed by J."""
    w = T.world()
    if J >= w["J"] and tag == "mxl":
        return shelf_data(tag, rep, np.arange(w["J"]), J)
    if J >= w["J"]:
        b, sd = T.load_truth()
        d = T.make_data(tag, rep)
        return dict(d, keep=np.arange(w["J"]), b=b, sd=sd)
    return shelf_data(tag, rep, shelf(J), J)


def shelf_data(tag, rep, keep, key):
    """A survey whose tasks show only the SKUs in keep (plus none), drawn from the full truth; seeds keyed by key."""
    w = T.world()
    b, sd = T.load_truth()
    J = len(keep)
    bi = np.r_[keep, w["J"]]
    r_data, r_hold, r_hb = [np.random.default_rng(s) for s in
                            np.random.SeedSequence([T.SEED, TAGS.index(tag), rep, key]).spawn(3)]
    lv = w["levels"][keep]
    price = w["price"][keep]
    beta = b[bi] + sd[bi] * r_data.standard_normal((T.N, J + 1))
    prices = price * T.MULT[r_data.integers(0, 5, (T.N, T.T, J))]
    y = simulate(r_data, prices, beta, tag, lv)
    Hp = price * T.MULT[np.random.default_rng([T.SEED, 7, J]).integers(0, 5, (T.NH, J))]
    yh = simulate(r_hold, np.broadcast_to(Hp, (T.N, T.NH, J)), beta, tag, lv)
    obs = np.stack([np.bincount(yh[:, i], minlength=J + 1) / T.N for i in range(T.NH)])
    return dict(keep=keep, prices=prices.astype(np.float32), y=y.astype(np.int16), Hp=Hp, obs=obs, levels=lv,
                b=b[bi], sd=sd[bi], hb_seed=int(r_hb.integers(2 ** 31)))


def level_mats(levels, J):
    S = np.zeros((4, J + 1, J + 1))
    for a in range(3):
        S[a, :J, :J] = levels[:, a, None] == levels[None, :, a]
    S[3, :J, :J] = np.eye(J)
    return S


def theta_start(J, iid):
    return np.r_[np.zeros(J + 1), np.log(np.full(J + 1, 0.5)), np.zeros(0 if iid else 3)]


def unpack(th, J):
    K1 = J + 1
    if len(th) == 2 * K1:
        return th[:K1], np.exp(th[K1:2 * K1]), np.array([0, 0, 0, 1.0])
    e = np.exp(np.r_[th[2 * K1:], 0.0])
    return th[:K1], np.exp(th[K1:2 * K1]), e / e.sum() * (1 - 4e-6) + 1e-6


# ---------------------------------------------------------------------------
# Per-task composite fit (Modal H100)
# ---------------------------------------------------------------------------
def moments(th, pr, y, S_lv, iid):
    """th (P,) or (n, P), pr (n, J), y (n,) -> eta (n, J), cov (n, J, J) of U_chosen - U_other (others in index
    order). Tastes folded: Omega = diag(sd_asc^2) + sd_price^2 p p' + level / product errors; none has TVAR."""
    import torch

    n, J = pr.shape
    th = th.expand(n, -1) if th.dim() == 1 else th
    K1 = J + 1
    b, sd = th[:, :K1], th[:, K1:2 * K1].exp()
    if iid:
        s2 = torch.zeros(n, 4, dtype=th.dtype, device=th.device)
        s2[:, 3] = TVAR
    else:
        u = th[:, 2 * K1:]
        p = torch.softmax(torch.cat([u, torch.zeros_like(u[:, :1])], 1), 1)
        s2 = TVAR * (p * (1 - 4e-6) + 1e-6)
    z1 = torch.zeros(n, 1, dtype=th.dtype, device=th.device)
    q = torch.cat([pr, z1], 1)
    mu = torch.cat([b[:, :J] + b[:, J:] * pr, z1], 1)
    Om = torch.einsum("na,aij->nij", s2, S_lv)
    Om[:, J, J] = TVAR
    Om = (Om + torch.diag_embed(torch.cat([sd[:, :J] ** 2, z1], 1))
          + (sd[:, J] ** 2)[:, None, None] * q[:, :, None] * q[:, None, :])
    ar = torch.arange(K1, device=pr.device)
    yc = y.long()
    oth = torch.stack([torch.cat([ar[:c], ar[c + 1:]]) for c in range(K1)])[yc]
    rows = torch.arange(n, device=pr.device)
    Oco = Om[rows, yc].gather(1, oth)
    Ooo = Om.gather(1, oth[:, :, None].expand(-1, -1, K1)).gather(2, oth[:, None, :].expand(-1, J, -1))
    cov = Ooo - Oco[:, :, None] - Oco[:, None, :] + Om[rows, yc, yc][:, None, None]
    eta = mu[rows, yc][:, None] - mu.gather(1, oth)
    return eta, cov


@app.function(image=gpu_image, gpu="H100", timeout=3 * 3600)
def fit_task(th0, pr, y, S_lv, iid, M=GHK_M, max_iter=1000, tol_change=1e-8):
    """L-BFGS (strong Wolfe) on the mean per-choice negative log-likelihood; stops when a step changes it by less
    than tol_change (1e-8 x 12,000 choices = 1.2e-4 nats in total) or the gradient is flat."""
    import torch

    sys.path.insert(0, "/root")
    import t18

    dev = torch.device("cuda")
    pr_t = torch.as_tensor(pr, dtype=torch.float64, device=dev)
    y_t = torch.as_tensor(y, device=dev)
    S_t = torch.as_tensor(S_lv, dtype=torch.float64, device=dev)
    n, J = pr.shape
    with torch.no_grad():
        e0, c0 = moments(torch.as_tensor(th0, dtype=torch.float64, device=dev), pr_t, y_t, S_t, iid)
        perm = t18.genz_order(e0, c0)
    del e0, c0
    base = torch.quasirandom.SobolEngine(J - 1, scramble=True, seed=7).draw(M, dtype=torch.float64).to(dev)
    shf = torch.rand((n, J - 1), generator=torch.Generator().manual_seed(8), dtype=torch.float64).to(dev)
    ch = max(1, int(2 ** 33 // (M * J * 8 * 8)))

    def nll_grad(th):
        th = th.detach().clone().requires_grad_(True)
        tot, g = 0.0, torch.zeros_like(th)
        for lo in range(0, n, ch):
            hi = min(lo + ch, n)
            e, c = moments(th, pr_t[lo:hi], y_t[lo:hi], S_t, iid)
            e, c = t18.permute(e, c, perm[lo:hi])
            w = torch.remainder(base[None] + shf[lo:hi, None], 1.0)
            lp = t18.ghk(e[:, None, :], torch.linalg.cholesky(c), w)
            ll = (torch.logsumexp(lp, 1) - np.log(M)).sum()
            g += torch.autograd.grad(ll, th)[0]
            tot += ll.item()
        return -tot, -g

    theta = torch.tensor(th0, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-7,
                            tolerance_change=tol_change, line_search_fn="strong_wolfe")
    nev, trace = [0], []

    def closure():
        f, g = nll_grad(theta)
        theta.grad = g / n
        nev[0] += 1
        trace.append(f)
        return torch.tensor(f / n, dtype=torch.float64)

    torch.cuda.synchronize()
    t0 = time.time()
    opt.step(closure)
    torch.cuda.synchronize()
    sec = time.time() - t0
    f, g = nll_grad(theta)
    return dict(theta=theta.detach().cpu().numpy().tolist(), nll=f, grad_max=float(g.abs().max()), seconds=sec,
                n_evals=nev[0], n_iter=int(opt.state[opt._params[0]]["n_iter"]), trace=trace,
                gpu=torch.cuda.get_device_name(), M=M)


def run_fit(tags, Js, reps, tol=1e-8):
    with modal.enable_output(), app.run():
        calls = {}
        for tag in tags:
            for J in Js:
                for rep in reps:
                    if (OUT / f"fit_{tag}_J{J}_r{rep}{sfx(tol)}.json").exists():
                        continue
                    d = make_data(tag, rep, J)
                    iid = tag in IID
                    pr = d["prices"].reshape(-1, J).astype(np.float64)
                    y = d["y"].reshape(-1).astype(np.int64)
                    calls[(tag, J, rep)] = fit_task.spawn(theta_start(J, iid), pr, y, level_mats(d["levels"], J), iid,
                                                          tol_change=tol)
        for (tag, J, rep), c in calls.items():
            r = c.get()
            (OUT / f"fit_{tag}_J{J}_r{rep}{sfx(tol)}.json").write_text(json.dumps(r))
            log(f"fit {tag} J={J} r{rep}: {r['seconds'] / 60:.1f} min, {r['n_iter']} iter / {r['n_evals']} evals, "
                f"nll {r['nll']:.1f}, grad max {r['grad_max']:.2e}")


def run_hb(tags, Js, reps):
    """HB-MNL with T29's GPU sampler (its app); J = 60 rep 0 reuses T29's file."""
    with modal.enable_output(), T.app.run():
        calls = {}
        for tag in tags:
            for J in Js:
                for rep in reps:
                    f = OUT / f"hb_{tag}_J{J}_r{rep}.npz"
                    if f.exists():
                        continue
                    if J == 60 and (T.OUT / f"hb_{tag}_r{rep}.npz").exists():
                        f.write_bytes((T.OUT / f"hb_{tag}_r{rep}.npz").read_bytes())
                        continue
                    d = make_data(tag, rep, J)
                    calls[(tag, J, rep)] = T.hb_fit.spawn(d["prices"], d["y"], d["hb_seed"])
        for (tag, J, rep), c in calls.items():
            r = c.get()
            np.savez_compressed(OUT / f"hb_{tag}_J{J}_r{rep}.npz", B=r["B"], loglike=r["loglike"],
                                seconds=r["seconds"])
            log(f"hb {tag} J={J} r{rep}: {r['seconds'] / 60:.1f} min")


# ---------------------------------------------------------------------------
# Part A: holdout shares vs truth (local)
# ---------------------------------------------------------------------------
def holdout_shares(b, sd, shares, Hp, levels, n, seed):
    J = Hp.shape[1]
    out = []
    for i in range(len(Hp)):
        sim = T.Sim(b, sd, shares, levels, Hp[i], n, [seed, i])
        s = np.r_[sim.shares(np.arange(J), np.arange(J)), 0.0]
        s[-1] = 1 - s[:-1].sum()
        out.append(s)
    return np.array(out)


def hb_holdout(B, Hp):
    J = Hp.shape[1]
    out = []
    for i in range(len(Hp)):
        V = np.c_[B[:, :J] + B[:, -1:] * Hp[i], np.zeros(len(B))]
        P = np.exp(V - V.max(1, keepdims=True))
        out.append((P / P.sum(1, keepdims=True)).mean(0))
    return np.array(out)


def mxl_shares(b, sd, Hp, n, seed):
    """Mixed-logit truth: shares (holdout tasks x J + 1) by simulation, Gumbel errors, chunked."""
    J = Hp.shape[1]
    out = np.zeros((len(Hp), J + 1))
    rng = np.random.default_rng(seed)
    for lo in range(0, n, 20_000):
        m = min(20_000, n - lo)
        beta = b + sd * rng.standard_normal((m, J + 1))
        for i in range(len(Hp)):
            U = np.c_[beta[:, :J] + beta[:, J:] * Hp[i], np.zeros(m)] + rng.gumbel(size=(m, J + 1))
            out[i] += np.bincount(U.argmax(1), minlength=J + 1)
    return out / n


def part_a():
    rows = []
    for f in sorted(OUT.glob("fit_*_J*_r*.json")):
        if "_tol" in f.stem:
            continue
        _, tag, Js, rs = f.stem.split("_")
        J, rep = int(Js[1:]), int(rs[1:])
        hbf = OUT / f"hb_{tag}_J{J}_r{rep}.npz"
        if not hbf.exists():
            continue
        d = make_data(tag, rep, J)
        r = json.loads(f.read_text())
        truth = (mxl_shares(d["b"], d["sd"], d["Hp"], 200_000, 900) if tag == "mxl" else
                 holdout_shares(d["b"], d["sd"], np.array(T.TRUTHS[tag]), d["Hp"], d["levels"], 200_000, 900))
        mnp = holdout_shares(*unpack(np.array(r["theta"]), J), d["Hp"], d["levels"], 100_000, 901)
        hb = np.load(hbf)
        arms = {"MNP": mnp, "HB-MNL": hb_holdout(hb["B"], d["Hp"]), "observed": d["obs"]}
        row = dict(truth=tag, J=J, rep=rep, mnp_min=r["seconds"] / 60, mnp_iter=r["n_iter"],
                   hb_min=float(hb["seconds"]) / 60 if "seconds" in hb else None)
        for nm, s in arms.items():
            e = 100 * (s - truth)
            row[nm] = dict(mae=float(np.abs(e).mean()), max=float(np.abs(e).max()), none=float(e[:, -1].mean()))
        rows.append(row)
        log(f"{tag} J={J} r{rep}: MNP {row['mnp_min']:.1f} min ({r['n_iter']} iter) | MAE pts: MNP "
            f"{row['MNP']['mae']:.3f} (max {row['MNP']['max']:.2f}, none {row['MNP']['none']:+.2f}), HB-MNL "
            f"{row['HB-MNL']['mae']:.3f} (max {row['HB-MNL']['max']:.2f}, none {row['HB-MNL']['none']:+.2f}), "
            f"observed {row['observed']['mae']:.3f}")
    (OUT / "partA.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------
# Part B: T30's line + price decision, greedy + swap (local)
# ---------------------------------------------------------------------------
def mxl_rows(w, b, sd, n, seed):
    """t30.sim_rows for the mixed-logit truth: iid Gumbel errors."""
    import t30

    client, comp, pc, prof = t30.setup(w)
    rng = np.random.default_rng(seed)
    J = w["J"]
    beta = b + sd * rng.standard_normal((n, J + 1))
    e = rng.gumbel(size=(n, J + 1))
    U0 = np.maximum((beta[:, comp] + beta[:, J:] * w["price"][comp] + e[:, comp]).max(1), e[:, J])
    Uc = beta[:, client, None] + beta[:, J, None, None] * pc[None] + e[:, client, None]
    return t30.Rows(U0.astype(np.float32), Uc.astype(np.float32), prof)


def part_b(reps, ms=(6, 8), tags=("est", "stress")):
    import t30

    w = T.world()
    b, sd = T.load_truth()
    runs = json.loads((ROOT / "analysis" / "T30" / "out" / "runs.json").read_text())
    res = []
    for tag in tags:
        if tag == "mxl":           # no T30 optimum: truth greedy + swap, raised below to the best line any arm found
            truth = mxl_rows(w, b, sd, 200_000, [T.SEED, 500, TAGS.index(tag)])
            opt = {}
            for m in ms:
                cs, ks, _ = t30.greedy(truth, m)
                opt[m] = truth(np.array(cs), np.array(ks))
        else:
            truth = t30.sim_rows(w, b, sd, np.array(T.TRUTHS[tag]), 200_000, [T.SEED, 500, TAGS.index(tag)])
            opt = {m: max(r["opt"] for r in runs if r["truth"] == tag and r["m"] == m) for m in ms}
        for rep in reps:
            fit, hbf = OUT / f"fit_{tag}_J60_r{rep}.json", OUT / f"hb_{tag}_J60_r{rep}.npz"
            if not (fit.exists() and hbf.exists()):
                continue
            d = make_data(tag, rep, 60)
            B = np.load(hbf)["B"]
            R = T.RFC(B, w["levels"], np.random.default_rng([T.SEED, 99, rep]))
            sig, _ = T.rfc_tune(R, d, w)
            bb, ss, sh = unpack(np.array(json.loads(fit.read_text())["theta"]), w["J"])
            scorers = {"SoP": t30.logit_rows(w, B), "RFC-S": t30.rfc_rows(w, R, sig),
                       "MNP": t30.sim_rows(w, bb, ss, sh, 100_000, [T.SEED, 600, rep])}
            hbd = OUT / f"hbd_{tag}_J60_r{rep}.npz"
            if hbd.exists():                   # SoP averaged over posterior draws (draws x respondents as rows)
                Bd = np.load(hbd)["draws"].astype(np.float64)
                scorers["SoP-draws"] = t30.logit_rows(w, Bd.reshape(-1, Bd.shape[-1]))
                del Bd
            for m in ms:
                for name, f in scorers.items():
                    cs, ks, v = t30.greedy(f, m)
                    tv = truth(np.array(cs), np.array(ks))
                    row = dict(truth=tag, rep=rep, m=m, arm=name, regret=opt[m] - tv, opt=opt[m], true=tv,
                               predicted=v, line=[int(w["client"][c]) for c in cs],
                               mult=[float(T.MULT[k]) for k in ks])
                    res.append(row)
                    log(f"{tag} r{rep} m={m} {name:5s}: regret {row['regret']:6.3f} $/1000 trips "
                        f"({100 * row['regret'] / opt[m]:.2f}%), prices {''.join('LlBhH'[k] for k in ks)}")
            del scorers, R
        if tag == "mxl":
            for m in ms:
                rs = [r for r in res if r["truth"] == tag and r["m"] == m]
                o = max([opt[m]] + [r["true"] for r in rs])
                for r in rs:
                    r["opt"], r["regret"] = o, o - r["true"]
        del truth
    f = OUT / "partB.json"
    old = json.loads(f.read_text()) if f.exists() else []
    keep = [r for r in old if (r["truth"], r["rep"]) not in {(x["truth"], x["rep"]) for x in res}]
    f.write_text(json.dumps(keep + res, indent=1))


# ---------------------------------------------------------------------------
# Part C: the compartmentalized incumbent (one survey + HB-MNL per pack-format compartment)
# ---------------------------------------------------------------------------
COMPARTMENTS = {"2L": ("SFT DRNK 2 LITER",), "single": ("SFT DRNK SNGL SRV",),
                "multi": ("SOFT DRINKS 12/18", "SFT DRNK MLT-PK", "SOFT DRINKS 20PK")}


def compartments():
    """name -> SKU indices, by sub-commodity prefix; every SKU in exactly one."""
    sub = np.load(T.T25 / "prep.npz", allow_pickle=True)["sub"].astype(str)
    out = {g: np.flatnonzero([any(s.startswith(p) for p in ps) for s in sub]) for g, ps in COMPARTMENTS.items()}
    assert sorted(np.concatenate(list(out.values()))) == list(range(len(sub)))
    return out


def comp_data(tag, rep, g):
    return shelf_data(tag, rep, compartments()[g], 1000 + list(COMPARTMENTS).index(g))


def run_hb_comp(tags, reps):
    with modal.enable_output(), T.app.run():
        calls = {}
        for tag in tags:
            for rep in reps:
                for g in COMPARTMENTS:
                    if (OUT / f"hbc_{tag}_{g}_r{rep}.npz").exists():
                        continue
                    d = comp_data(tag, rep, g)
                    calls[(tag, g, rep)] = T.hb_fit.spawn(d["prices"], d["y"], d["hb_seed"])
        for (tag, g, rep), c in calls.items():
            r = c.get()
            np.savez_compressed(OUT / f"hbc_{tag}_{g}_r{rep}.npz", B=r["B"], loglike=r["loglike"],
                                seconds=r["seconds"])
            log(f"hb {tag} {g} r{rep}: {r['seconds'] / 60:.1f} min")


class Comp:
    """Sum of per-compartment scorers, each over its own client SKUs (the naive combination of separate studies:
    each compartment is its own market with its own none)."""

    def __init__(self, parts, prof):
        self.parts, self.prof = parts, prof             # [(scorer, client positions in this compartment, weight)]

    def _sel(self, cs, idx):
        sel = [j for j, c in enumerate(cs) if c in idx]
        return sel, np.array([idx.index(cs[j]) for j in sel], dtype=int)

    def __call__(self, cs, ks):
        v = 0.0
        for f, idx, wt in self.parts:
            sel, loc = self._sel(cs, idx)
            if sel:
                v += wt * f(loc, np.asarray(ks)[sel])
        return v

    def sweep(self, cs, ks, i):
        out = np.zeros(K_PRICES)
        for f, idx, wt in self.parts:
            sel, loc = self._sel(cs, idx)
            if not sel:
                continue
            kk = np.asarray(ks)[sel]
            out += wt * (f.sweep(loc, kk, sel.index(i)) if i in sel else f(loc, kk))
        return out


K_PRICES = 5


def comp_scorers(tag, rep, weighted=False):
    """weighted: each compartment's profit scaled so its predicted purchase volume at the surveyed shelf (all its
    SKUs, base prices) matches its share of category purchases (real occasions, 85% of trips buy), as a
    practitioner calibrating separate studies to sales would; otherwise summed unweighted."""
    import t30

    w = T.world()
    client = list(w["client"])
    occ = np.load(T.T25 / "prep.npz", allow_pickle=True)["occ"]
    sop, rfc = [], []
    for g, keep in compartments().items():
        idx = [p for p, j in enumerate(client) if j in keep]
        if not idx:
            continue
        d = comp_data(tag, rep, g)
        wg = dict(J=len(keep), price=w["price"][keep], levels=w["levels"][keep],
                  client=np.array([int(np.flatnonzero(keep == client[p])[0]) for p in idx]))
        B = np.load(OUT / f"hbc_{tag}_{g}_r{rep}.npz")["B"]
        R = T.RFC(B, wg["levels"], np.random.default_rng([T.SEED, 98, rep, len(keep)]))
        sig, _ = T.rfc_tune(R, d, wg)
        vol = 0.85 * occ[keep].sum() / occ.sum()
        V = B[:, :-1] + B[:, -1:] * wg["price"]
        buy_sop = 1 - (1 / (1 + np.exp(V).sum(1))).mean()
        buy_rfc = 1 - R.shares(wg["price"], np.arange(len(keep)), sig)[-1]
        sop.append((t30.logit_rows(wg, B), idx, vol / buy_sop if weighted else 1.0))
        rfc.append((t30.rfc_rows(wg, R, sig), idx, vol / buy_rfc if weighted else 1.0))
    prof = t30.setup(w)[3]
    sfx = "-w" if weighted else ""
    return {f"SoP-comp{sfx}": Comp(sop, prof), f"RFC-S-comp{sfx}": Comp(rfc, prof)}


def part_c(reps, ms=(6,), weighted=False):
    import t30

    w = T.world()
    b, sd = T.load_truth()
    runs = json.loads((ROOT / "analysis" / "T30" / "out" / "runs.json").read_text())
    res = []
    for tag in T.TRUTHS:
        truth = t30.sim_rows(w, b, sd, np.array(T.TRUTHS[tag]), 200_000, [T.SEED, 500, list(T.TRUTHS).index(tag)])
        opt = {m: max(r["opt"] for r in runs if r["truth"] == tag and r["m"] == m) for m in ms}
        for rep in reps:
            if not all((OUT / f"hbc_{tag}_{g}_r{rep}.npz").exists() for g in COMPARTMENTS):
                continue
            for name, f in comp_scorers(tag, rep, weighted).items():
                for m in ms:
                    cs, ks, v = t30.greedy(f, m)
                    tv = truth(np.array(cs), np.array(ks))
                    row = dict(truth=tag, rep=rep, m=m, arm=name, regret=opt[m] - tv, opt=opt[m], true=tv,
                               predicted=v, line=[int(w["client"][c]) for c in cs],
                               mult=[float(T.MULT[k]) for k in ks])
                    res.append(row)
                    log(f"{tag} r{rep} m={m} {name:10s}: regret {row['regret']:6.3f} $/1000 trips "
                        f"({100 * row['regret'] / opt[m]:.2f}%), predicted {v:7.2f} vs true {tv:7.2f}, "
                        f"prices {''.join('LlBhH'[k] for k in ks)}")
        del truth
    f = OUT / "partC.json"
    old = json.loads(f.read_text()) if f.exists() else []
    keep = [r for r in old if (r["truth"], r["rep"], r["arm"]) not in {(x["truth"], x["rep"], x["arm"]) for x in res}]
    f.write_text(json.dumps(keep + res, indent=1))


def run_hb_draws(tags, reps, draws=100):
    """HB-MNL refit (same data, same seed) keeping posterior draws, for simulation over draws."""
    with modal.enable_output(), T.app.run():
        calls = {}
        for tag in tags:
            for rep in reps:
                if (OUT / f"hbd_{tag}_J60_r{rep}.npz").exists():
                    continue
                d = make_data(tag, rep, 60)
                calls[(tag, rep)] = T.hb_fit.spawn(d["prices"], d["y"], d["hb_seed"], draws=draws)
        for (tag, rep), c in calls.items():
            r = c.get()
            np.savez_compressed(OUT / f"hbd_{tag}_J60_r{rep}.npz", B=r["B"], draws=r["draws"], seconds=r["seconds"])
            log(f"hb draws {tag} r{rep}: {r['seconds'] / 60:.1f} min")


def hb_draw_check(tags, reps):
    """Holdout shares from HB: posterior means + SoP vs SoP averaged over posterior draws; error vs truth."""
    for tag in tags:
        for rep in reps:
            f = OUT / f"hbd_{tag}_J60_r{rep}.npz"
            if not f.exists():
                continue
            d = make_data(tag, rep, 60)
            tr = (mxl_shares(d["b"], d["sd"], d["Hp"], 200_000, 900) if tag == "mxl" else
                  holdout_shares(d["b"], d["sd"], np.array(T.TRUTHS[tag]), d["Hp"], d["levels"], 200_000, 900))
            z = np.load(f)
            Bd = z["draws"].astype(np.float32)
            mean = hb_holdout(z["B"], d["Hp"])
            over = np.mean([hb_holdout(Bk, d["Hp"]) for Bk in Bd], 0)
            for nm, s in (("posterior means", mean), ("over draws", over)):
                e = s - tr
                log(f"{tag} r{rep} HB {nm:15s}: misplaced {100 * 0.5 * np.abs(e).sum(1).mean():.2f} pts, SKUs "
                    f"{100 * 0.5 * np.abs(e[:, :-1]).sum(1).mean():.2f}, none bias {100 * e[:, -1].mean():+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "hb", "partA", "partB", "hbcomp", "partC", "hbdraws", "drawcheck"])
    ap.add_argument("--truths", default="est,stress")
    ap.add_argument("--J", default="60")
    ap.add_argument("--reps", default="0")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-8, help="fit: stop when a step changes the mean nll by less")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tags, Js, reps = a.truths.split(","), [int(x) for x in a.J.split(",")], [int(x) for x in a.reps.split(",")]
    if a.stage == "fit":
        run_fit(tags, Js, reps, a.tol)
    elif a.stage == "hb":
        run_hb(tags, Js, reps)
    elif a.stage == "partA":
        part_a()
    elif a.stage == "hbdraws":
        run_hb_draws(tags, reps)
    elif a.stage == "drawcheck":
        hb_draw_check(tags, reps)
    elif a.stage == "hbcomp":
        run_hb_comp(tags, reps)
    elif a.stage == "partC":
        part_c(reps, weighted=a.weighted)
    else:
        part_b(reps, tags=tags)
