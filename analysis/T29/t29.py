"""T29. Client line from a large candidate set: survey -> fitted model -> exhaustive line search (TEST.md).

    python analysis/T29/t29.py truth                 # local: calibrated truths -> out/truth.json
    python analysis/T29/t29.py fit --reps 0          # Modal: MNP fit (H100) and HB-MNL (CPU) per (truth, rep)
    python analysis/T29/t29.py score --reps 0        # local: RFC-S tuning, line search per arm, regret vs truth

World: T25's 60 dunnhumby SOFT DRINKS SKUs (cluster c1, 53 stores). A shelf CBC survey: N = 1,000 respondents,
T = 12 tasks, every task shows all 60 SKUs plus "none" at prices base x {0.8, 0.9, 1.0, 1.1, 1.2}; 6 fixed holdout
tasks. Truth (MNP, logit scale, total error variance pi^2 / 6):
  U_nj = asc_nj + b_price,n price_j + e_j,   asc_n ~ N(a, diag s_a^2), b_price,n ~ N(-1.2, 0.3^2) (assumed)
  e = sum over (manufacturer, sub-commodity, cell) of a level effect shared by the SKUs at that level, plus a
  product error; "none" has its own independent error of the total variance.
  a, s_a: the T25 LC-MNL (c1) segment mean and spread at base prices, shifted so "none" takes ~15%.
  cell = manufacturer x sub-commodity x size band (14 cells; T25's near-duplicates).
  est:    variance shares 0 / 0 / 0 / 1 (T25's household-panel MLE on the real occasions: Sigma = I)
  stress: 0.25 / 0.02 / 0.23 / 0.50 (assumed; T25's stress truth: rho 0.25 within manufacturer, 0.5 within cell)
Decision: client = manufacturer 103 (17 SKUs); line of m in {4, 6, 8} SKUs, its top 2 by real occasions always in;
the other 43 SKUs and "none" stay on the shelf at base prices. Objective = client margin per 1,000 category
trips = 1000 sum_{j in line} P(j) 0.30 price_j. Every arm enumerates every line (C(15, m - 2)).
Arms: SoP (HB-MNL posterior means, logit shares averaged), RFC-S (HB-MNL + one tuned attribute-error sd, Gumbel,
fixed draws), MNP (T18 panel GHK fit, population shares by the orthant engine; MNP-sim: the same fit
scored by a 100k-draw simulation). Truth values by a 200k-draw first-choice simulation with fixed draws. Regret = truth optimum minus the
truth value of the arm's line.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
T25 = HERE.parent / "T25" / "out"
SEED = 20261022
N, T, NH = 1000, 12, 6
MULT = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
PRICE_B, PRICE_W = -1.2, 0.3               # assumed: the real occasions give b ~ 0 (promotions are endogenous)
TVAR = np.pi ** 2 / 6
NONE_SHARE = 0.15
TRUTHS = {"est": [0.0, 0.0, 0.0, 1.0], "stress": [0.25, 0.02, 0.23, 0.50]}   # mfr, sub, cell, product
CLIENT, MUST, LINE_M = "103", 2, (4, 6, 8)
MARGIN = 0.30
GHK_M = 1024
HB_ITERS = 20000
RFC_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
RFC_R = 100

gpu_image = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy")
             .add_local_file(HERE.parent / "T18" / "t18.py", "/root/t18.py"))
app = modal.App("orthant-rfc-t29")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Truth (local)
# ---------------------------------------------------------------------------
def world():
    z = np.load(T25 / "prep.npz", allow_pickle=True)
    mfr, sub, band = z["mfr"].astype(str), z["sub"].astype(str), z["band"]
    lv = np.stack([np.unique(mfr, return_inverse=True)[1], np.unique(sub, return_inverse=True)[1],
                   np.unique(np.char.add(np.char.add(mfr, sub), band.astype(str)), return_inverse=True)[1]], 1)
    client = np.flatnonzero(mfr == CLIENT)
    client = client[np.argsort(-z["occ"][client], kind="stable")]
    return dict(price=z["price"].astype(float), levels=lv, client=client, J=len(mfr))


def simulate(rng, prices, beta, shares, levels):
    """prices (N, T, J), beta (N, J + 1) [asc, price] -> first choices (N, T) in 0..J (J = none)."""
    n, t, J = prices.shape
    V = beta[:, None, :J] + beta[:, None, J:] * prices
    sd = np.sqrt(TVAR * np.asarray(shares))
    e = sd[3] * rng.standard_normal((n, t, J))
    for a in range(3):
        e += sd[a] * rng.standard_normal((n, t, levels[:, a].max() + 1))[..., levels[:, a]]
    U = np.concatenate([V + e, np.sqrt(TVAR) * rng.standard_normal((n, t, 1))], 2)
    return U.argmax(2)


def truth():
    w = world()
    lc = json.loads((T25 / "lc_c1.json").read_text())
    a, wt = np.array(lc["a"]), np.array(lc["w"])
    m = wt @ a
    s = np.sqrt(wt @ (a - m) ** 2)
    asc = m - PRICE_B * w["price"]                 # utilities at base prices match the LC segment mean
    rng = np.random.default_rng([SEED, 0])
    n = 20000
    zb = rng.standard_normal((n, w["J"] + 1))
    pr = np.broadcast_to(w["price"], (n, 1, w["J"]))

    def none_share(c):
        beta = np.c_[asc + c + s * zb[:, :-1], PRICE_B + PRICE_W * zb[:, -1]]
        return (simulate(np.random.default_rng([SEED, 1]), pr, beta, TRUTHS["est"], w["levels"]) == w["J"]).mean()

    lo, hi = -10.0, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if none_share(mid) > NONE_SHARE else (lo, mid)
    shift = (lo + hi) / 2
    out = dict(b=np.r_[asc + shift, PRICE_B].tolist(), w=np.r_[s, PRICE_W].tolist(), truths=TRUTHS,
               none_share=float(none_share(shift)))
    asc = asc + shift
    (OUT / "truth.json").write_text(json.dumps(out, indent=1))
    log(f"asc mean {asc.mean():.2f} (range {asc.min():.2f}..{asc.max():.2f}), taste sd median {np.median(s):.2f}, "
        f"none share {out['none_share']:.3f}")


def load_truth():
    t = json.loads((OUT / "truth.json").read_text())
    return np.array(t["b"]), np.array(t["w"])


def make_data(tag, rep):
    w = world()
    b, sd = load_truth()
    r_data, r_hold, r_hb = [np.random.default_rng(s) for s in
                            np.random.SeedSequence([SEED, list(TRUTHS).index(tag), rep]).spawn(3)]
    J = w["J"]
    beta = b + sd * r_data.standard_normal((N, J + 1))
    prices = w["price"] * MULT[r_data.integers(0, 5, (N, T, J))]
    y = simulate(r_data, prices, beta, TRUTHS[tag], w["levels"])
    Hp = w["price"] * MULT[np.random.default_rng([SEED, 7]).integers(0, 5, (NH, J))]      # fixed across reps
    yh = simulate(r_hold, np.broadcast_to(Hp, (N, NH, J)), beta, TRUTHS[tag], w["levels"])
    obs = np.stack([np.bincount(yh[:, i], minlength=J + 1) / N for i in range(NH)])
    return dict(prices=prices.astype(np.float32), y=y.astype(np.int16), Hp=Hp, obs=obs, levels=w["levels"],
                hb_seed=int(r_hb.integers(2 ** 31)))


# ---------------------------------------------------------------------------
# MNP fit (Modal H100): T18's panel GHK, Genz order, design built on the GPU
# ---------------------------------------------------------------------------
def theta_true(tag):
    b, sd = load_truth()
    sh = np.maximum(np.array(TRUTHS[tag]), 1e-3)
    return np.r_[b, np.log(sd), np.log(sh[:3] / sh[3])]


@app.function(image=gpu_image, gpu="H100", timeout=4 * 3600, memory=32768)
def mnp_fit(prices, y, levels, th_true, M=GHK_M, bc=False, start=None):
    import torch

    sys.path.insert(0, "/root")
    import t18

    dev = torch.device("cuda")
    n, t, J = prices.shape
    K1, P = J + 1, J + 1
    pr = torch.as_tensor(prices, dtype=torch.float64, device=dev)
    yy = torch.as_tensor(y.astype(np.int64), device=dev)
    lv = torch.as_tensor(levels, device=dev)
    S = torch.zeros(4, K1, K1, dtype=torch.float64, device=dev)
    for a in range(3):
        S[a, :J, :J] = (lv[:, a, None] == lv[None, :, a]).double()
    S[3, :J, :J] = torch.eye(J, dtype=torch.float64, device=dev)
    S[:, J, J] = 1
    Xd, G = [], []
    oth_all = torch.stack([torch.cat([torch.arange(c), torch.arange(c + 1, K1)]) for c in range(K1)]).to(dev)
    for lo in range(0, n, 100):
        p, yc = pr[lo:lo + 100], yy[lo:lo + 100]
        c = p.shape[0]
        X = torch.zeros(c, t, K1, P, dtype=torch.float64, device=dev)
        X[..., :J, :J] = torch.eye(J, dtype=torch.float64, device=dev)
        X[..., :J, J] = p
        Mx = torch.zeros(c, t, K1 - 1, K1, dtype=torch.float64, device=dev)
        Mx.scatter_(3, oth_all[yc][..., None], -1.0)
        Mx.scatter_(3, yc[..., None, None].expand(c, t, K1 - 1, 1), 1.0)
        Xd.append((Mx @ X).float())
        G.append(torch.einsum("ntik,akl,ntjl->antij", Mx, S, Mx).to(torch.int8))
        del X, Mx
    Xd = torch.cat(Xd).double()
    G = torch.cat(G, 1).double()
    spec = dict(kind="panel", M=M, order="genz", bc=bc)
    t0 = time.time()
    with torch.no_grad():
        eta, cov = t18.panel_moments(torch.as_tensor(t18.theta0(P), device=dev).expand(n, -1), Xd, G, TVAR)
        perm = t18.genz_order(eta, cov)
    del eta, cov
    r = t18.fit_lbfgs(spec, Xd, G, TVAR, np.asarray(th_true), perm, max_iter=500, start=start)
    r.update(order_seconds=time.time() - t0 - r["seconds"], gpu=torch.cuda.get_device_name(), M=M, bc=bc)
    return r


# ---------------------------------------------------------------------------
# HB-MNL (Modal T4): Sawtooth CBC/HB's sampler in torch, hb.R's prior (prior variance 1, df P + 5), full
# covariance. bayesm took ~0.4 s per iteration at N = 300 with 61 parameters (~9 h for a fit), so the same model
# runs on a GPU: per respondent a random-walk Metropolis step with proposal rho L_D z (rho adapted to 30%
# acceptance), alpha | beta, D Gibbs, D | beta, alpha inverse Wishart. Posterior means over every 10th draw of
# the second half.
# ---------------------------------------------------------------------------
@app.function(image=gpu_image, gpu="T4", timeout=2 * 3600)
def hb_fit(prices, y, seed, iters=HB_ITERS, keep=10):
    import torch

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    n, t, J = prices.shape
    P = J + 1
    pr = torch.as_tensor(prices, dtype=torch.float32, device=dev)
    yy = torch.as_tensor(y.astype(np.int64), device=dev)

    def loglik(B):
        V = torch.cat([B[:, None, :J] + B[:, None, J:] * pr, torch.zeros(n, t, 1, device=dev)], 2)
        return (V.gather(2, yy[..., None])[..., 0] - torch.logsumexp(V, 2)).sum(1)

    nu = P + 5
    Vp = nu * torch.eye(P, device=dev)
    B = torch.zeros(n, P, device=dev)
    alpha = torch.zeros(P, device=dev)
    D = torch.eye(P, device=dev)
    rho = 0.1
    ll = loglik(B)
    t0 = time.time()
    acc_hist, llh, Bsum, nk = [], [], torch.zeros(n, P, device=dev), 0
    for it in range(iters):
        L = torch.linalg.cholesky(D)
        Di = torch.cholesky_inverse(L)
        prop = B + rho * torch.randn(n, P, device=dev, generator=g) @ L.T
        llp = loglik(prop)
        dq = lambda x: (((x - alpha) @ Di) * (x - alpha)).sum(1)  # noqa: E731
        logr = llp - ll - 0.5 * (dq(prop) - dq(B))
        acc = torch.log(torch.rand(n, device=dev, generator=g)) < logr
        B = torch.where(acc[:, None], prop, B)
        ll = torch.where(acc, llp, ll)
        a = acc.float().mean().item()
        rho *= 1.1 if a > 0.3 else 0.9
        acc_hist.append(a)
        alpha = B.mean(0) + torch.linalg.cholesky(D / n) @ torch.randn(P, device=dev, generator=g)
        R = B - alpha
        S = Vp + R.T @ R
        W = torch.distributions.Wishart(df=torch.tensor(float(nu + n), device=dev),
                                        precision_matrix=S).sample()    # D^-1 ~ Wishart(nu + n, S^-1)
        D = torch.cholesky_inverse(torch.linalg.cholesky(W))
        if it % keep == 0:
            llh.append(ll.sum().item())
            if it >= iters // 2:
                Bsum += B
                nk += 1
    return dict(B=(Bsum / nk).cpu().numpy(), loglike=np.array(llh), accept=float(np.mean(acc_hist[iters // 2:])),
                rho=rho, alpha=alpha.cpu().numpy(), seconds=time.time() - t0)


def run_fit(reps, tags, arms, bc=False):
    """bc: bias-corrected simulated likelihood, warm-started from the plain fit; writes mnp_<tag>_r<rep>_bc.json."""
    sfx = "_bc" if bc else ""
    with modal.enable_output(), app.run():
        calls = {}
        for tag in tags:
            for rep in reps:
                d = make_data(tag, rep)
                if "mnp" in arms and not (OUT / f"mnp_{tag}_r{rep}{sfx}.json").exists():
                    start = json.loads((OUT / f"mnp_{tag}_r{rep}.json").read_text())["theta"] if bc else None
                    calls[("mnp", tag, rep)] = mnp_fit.spawn(d["prices"], d["y"], d["levels"], theta_true(tag),
                                                             bc=bc, start=start)
                if "hb" in arms and not (OUT / f"hb_{tag}_r{rep}.npz").exists():
                    calls[("hb", tag, rep)] = hb_fit.spawn(d["prices"], d["y"], d["hb_seed"])
        for (arm, tag, rep), c in calls.items():
            r = c.get()
            if arm == "mnp":
                (OUT / f"mnp_{tag}_r{rep}{sfx}.json").write_text(json.dumps(r))
                log(f"mnp {tag} r{rep}: {r['seconds'] / 60:.1f} min, {r['n_iter']} iter / {r['n_evals']} evals, "
                    f"nll {r['nll']:.1f} (truth {r['nll_true']:.1f}), grad max {r['grad_max']:.2e}, {r['gpu']}")
            else:
                np.savez_compressed(OUT / f"hb_{tag}_r{rep}.npz", B=r["B"], loglike=r["loglike"])
                log(f"hb {tag} r{rep}: {r['seconds'] / 60:.1f} min")


# ---------------------------------------------------------------------------
# Shares and scoring (local)
# ---------------------------------------------------------------------------
def err_cov(levels, shares):
    J = len(levels)
    C = np.zeros((J + 1, J + 1))
    for a in range(3):
        C[:J, :J] += shares[a] * (levels[:, a, None] == levels[None, :, a])
    C[:J, :J] += shares[3] * np.eye(J)
    C[J, J] = 1.0
    return TVAR * C


BOAZ = Path("/home/ubuntu/projects/boaz/modal")       # private engine variants (local scoring only)


def engines():
    """Orthant scorers for the MNP fit: the released binary ("high" = order 1), the library at order 2 (what
    "highest" ships), and the fast-engine prototype."""
    sys.path.insert(0, str(HERE.parents[1] / "research" / "rfc_vs_exact"))
    sys.path[:0] = [str(BOAZ), str(BOAZ / "prototypes" / "mnp_engine")]
    import orthant
    import pairx_fast
    import rfc

    return {"MNP": lambda u, c: rfc.orthant_cdf(u, c, resolution="high"),
            "MNP-o2": lambda u, c: orthant.cdf(u, c, order=2),
            "MNP-fast": lambda u, c: pairx_fast.cdf(u, c, order=2)}


def mnp_shares(b, sd, shares, levels, price, shelf, which, engine=None):
    """Population shares of SKUs `which` (subset of shelf) on shelf + none; tastes folded (normal, diag)."""
    engine = engine or engines()["MNP"]

    J = len(price)
    alts = np.r_[shelf, J]
    X = np.zeros((len(alts), J + 1))
    X[np.arange(len(shelf)), shelf] = 1
    X[:len(shelf), J] = price[shelf]
    mu = X @ b
    C = X @ np.diag(sd ** 2) @ X.T + err_cov(levels, shares)[np.ix_(alts, alts)]
    ups, covs = [], []
    for j in which:
        k = int(np.flatnonzero(alts == j)[0])
        oth = np.r_[np.arange(k), np.arange(k + 1, len(alts))]
        D = np.zeros((len(oth), len(alts)))
        D[np.arange(len(oth)), oth] = 1
        D[:, k] = -1
        ups.append(-(D @ mu))
        covs.append(D @ C @ D.T)
    return engine(np.array(ups), np.array(covs))


class Sim:
    """First-choice simulation with fixed draws: every person's utilities for all 60 SKUs + none at base prices
    are drawn once, so each line is one argmax over its shelf's columns (common random numbers across lines)."""

    def __init__(self, b, sd, shares, levels, price, n, seed):
        rng = np.random.default_rng(seed)
        J = len(price)
        beta = b + sd * rng.standard_normal((n, J + 1))
        self.U = np.c_[beta[:, :J] + beta[:, J:] * price, np.zeros(n)]
        sdv = np.sqrt(TVAR * np.asarray(shares))
        self.U[:, :J] += sdv[3] * rng.standard_normal((n, J))
        for a in range(3):
            self.U[:, :J] += sdv[a] * rng.standard_normal((n, levels[:, a].max() + 1))[:, levels[:, a]]
        self.U[:, J] = np.sqrt(TVAR) * rng.standard_normal(n)
        self.U = self.U.astype(np.float32)
        self.J = J

    def shares(self, shelf, which):
        cols = np.r_[shelf, self.J]
        k = self.U[:, cols].argmax(1)
        c = np.bincount(k, minlength=len(cols)) / len(k)
        return c[[int(np.flatnonzero(cols == j)[0]) for j in which]]


def profit_sim(w, sim, line):
    comp = np.setdiff1d(np.arange(w["J"]), w["client"])
    return 1000 * float(sim.shares(np.r_[comp, line], line) @ (MARGIN * w["price"][line]))


def lines(w, m):
    free = w["client"][MUST:]
    for extra in itertools.combinations(free, m - MUST):
        yield np.r_[w["client"][:MUST], extra]


def profit_mnp(w, b, sd, shares, line, engine=None):
    comp = np.setdiff1d(np.arange(w["J"]), w["client"])
    shelf = np.r_[comp, line]
    p = mnp_shares(b, sd, shares, w["levels"], w["price"], shelf, line, engine)
    return 1000 * float(p @ (MARGIN * w["price"][line]))


def logit_profit(B, w, line):
    comp = np.setdiff1d(np.arange(w["J"]), w["client"])
    shelf = np.r_[comp, line]
    V = B[:, shelf] + B[:, -1:] * w["price"][shelf]
    V = np.c_[V, np.zeros(len(B))]
    P = np.exp(V - V.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    s = P[:, len(comp):len(shelf)].mean(0)
    return 1000 * float(s @ (MARGIN * w["price"][line]))


class RFC:
    """HB posterior means + one attribute-error sd (mfr, sub, cell levels) + Gumbel, fixed draws (common numbers)."""

    def __init__(self, B, levels, rng, R=RFC_R):
        self.B, self.lv = B, levels
        n, J = len(B), len(levels)
        self.z = [rng.standard_normal((n, R, levels[:, a].max() + 1)).astype(np.float32) for a in range(3)]
        self.g = rng.gumbel(size=(n, R, J + 1)).astype(np.float32)

    def shares(self, prices, shelf, sig):
        """prices (J,), shelf SKU indices -> shares over shelf + none, averaged over respondents and draws."""
        J = len(self.lv)
        V = (self.B[:, shelf] + self.B[:, -1:] * prices[shelf]).astype(np.float32)
        U = V[:, None, :] + self.g[:, :, shelf]
        for a in range(3):
            U += sig * self.z[a][:, :, self.lv[shelf, a]]
        U = np.concatenate([U, self.g[:, :, J:]], 2)
        k = U.argmax(2)
        return np.bincount(k.ravel(), minlength=len(shelf) + 1) / k.size


def rfc_tune(R, d, w):
    J = w["J"]
    mae = {}
    for s in RFC_GRID:
        pred = np.stack([R.shares(d["Hp"][i], np.arange(J), s) for i in range(NH)])
        mae[s] = float(np.abs(pred - d["obs"]).mean())
    return min(mae, key=mae.get), mae


def score(reps, tags):
    w = world()
    b, sd = load_truth()
    res = []
    for tag in tags:
        shares = np.array(TRUTHS[tag])
        tf = OUT / f"truthvals_{tag}.json"
        if tf.exists():
            tv = {int(k): v for k, v in json.loads(tf.read_text()).items()}
        else:
            t0 = time.time()
            sim = Sim(b, sd, shares, w["levels"], w["price"], 200_000, [SEED, 500, list(TRUTHS).index(tag)])
            tv = {m: [profit_sim(w, sim, L) for L in lines(w, m)] for m in LINE_M}
            del sim
            tf.write_text(json.dumps(tv))
            log(f"truth {tag}: {sum(len(v) for v in tv.values())} lines in {time.time() - t0:.0f} s")
        for rep in reps:
            d = make_data(tag, rep)
            arms = {}
            hbf, mf = OUT / f"hb_{tag}_r{rep}.npz", OUT / f"mnp_{tag}_r{rep}.json"
            if hbf.exists():
                B = np.load(hbf)["B"]
                R = RFC(B, w["levels"], np.random.default_rng([SEED, 99, rep]))
                sig, mae = rfc_tune(R, d, w)
                log(f"{tag} r{rep}: RFC-S sd {sig}, holdout MAE {mae[sig]:.4f} (sd 0: {mae[0.0]:.4f})")
                arms["SoP"] = lambda L, B=B: logit_profit(B, w, L)
                comp = np.setdiff1d(np.arange(w["J"]), w["client"])

                def rfc_val(L, R=R, sig=sig):
                    s = R.shares(w["price"], np.r_[comp, L], sig)[len(comp):len(comp) + len(L)]
                    return 1000 * float(s @ (MARGIN * w["price"][L]))
                arms["RFC-S"] = rfc_val
            if mf.exists():
                th = np.array(json.loads(mf.read_text())["theta"])
                P = w["J"] + 1
                e = np.exp(np.r_[th[2 * P:], 0.0])
                sh = e / e.sum() * (1 - 4e-6) + 1e-6
                for name, eng in engines().items():
                    arms[name] = lambda L, th=th, sh=sh, eng=eng: profit_mnp(w, th[:P], np.exp(th[P:2 * P]), sh, L, eng)
                fsim = Sim(th[:P], np.exp(th[P:2 * P]), sh, w["levels"], w["price"], 100_000, [SEED, 600, rep])
                arms["MNP-sim"] = lambda L, fsim=fsim: profit_sim(w, fsim, L)
            for m in LINE_M:
                Ls = list(lines(w, m))
                opt = max(tv[m])
                for name, f in arms.items():
                    t0 = time.time()
                    v = [f(L) for L in Ls]
                    i = int(np.argmax(v))
                    row = dict(truth=tag, rep=rep, m=m, arm=name, n_lines=len(Ls), opt=opt, true=tv[m][i],
                               regret=opt - tv[m][i], regret_pct=100 * (opt - tv[m][i]) / opt,
                               predicted=v[i], seconds=time.time() - t0,
                               line=[int(x) for x in Ls[i]], best=[int(x) for x in Ls[int(np.argmax(tv[m]))]])
                    res.append(row)
                    log(f"{tag} r{rep} m={m} {name:5s}: regret {row['regret']:.3f} $/1000 trips "
                        f"({row['regret_pct']:.2f}%), opt {opt:.2f}, predicted {v[i]:.2f}, {row['seconds']:.0f} s")
    f = OUT / "scores.json"
    old = json.loads(f.read_text()) if f.exists() else []
    keep = [r for r in old if (r["truth"], r["rep"]) not in {(x["truth"], x["rep"]) for x in res}]
    f.write_text(json.dumps(keep + res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["truth", "fit", "score"])
    ap.add_argument("--reps", default="0")
    ap.add_argument("--truths", default="est,stress")
    ap.add_argument("--arms", default="mnp,hb")
    ap.add_argument("--bc", action="store_true", help="MNP: bias-corrected likelihood, warm start (fit stage)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    reps = [int(x) for x in a.reps.split(",")]
    tags = a.truths.split(",")
    if a.stage == "truth":
        truth()
    elif a.stage == "fit":
        run_fit(reps, tags, a.arms.split(","), a.bc)
    else:
        score(reps, tags)
