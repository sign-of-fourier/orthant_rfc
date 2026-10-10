"""T7. Engine vs shared-draw simulator for TURF search (design: PLAN.md T7).

    python analysis/T7/t7.py            # full run (one Modal T4)
    python analysis/T7/t7.py --check    # small wiring check

Model: T6's window-1 IFM fit (soft drinks, J = 30; intercept-only marginals mu, one R), recomputed
locally. reach(S) = 1 - P(Z_S < -mu_S), Z ~ N(0, R): one orthant per portfolio.

Scorers, all on one T4, each timed with its own setup (CUDA context warmed first):
  engine : modal_gp_api.orthant_prob (boaz GPU engine, order 1, float64), as score_bench
  sim    : best-practice shared-draw simulator. N households drawn once, acceptance stored
           bit-sliced (per item a bitvector over households, 64-bit words); reach(S) =
           popcount(OR_{j in S} B_j) / N; common draws for every portfolio; batched on GPU
  ghk    : GHK with fixed scrambled Sobol (T6's integrator), float64
Searches: exhaustive k = 3-6, greedy k = 3-12. Reference: GHK M = 2^16 re-scores the union of every
scorer setting's top 100 per k (exhaustive) and greedy finalists + last-step top 100 (greedy);
assumption: the true best is in that union. Writes analysis/T7/out/results[_check].json.
"""
import argparse
import json
import pathlib
import sys
import time

import modal
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BOAZ = pathlib.Path("/home/ubuntu/projects/boaz/modal")
SEED = 20261010
TOP = 100

FULL = dict(k_exh=(3, 4, 5, 6), k_greedy=tuple(range(3, 13)), sim_N=tuple(2 ** e for e in (12, 14, 16, 18, 20)),
            ghk_M=tuple(2 ** e for e in (10, 12, 14)), ref_M=2 ** 16)
CHECK = dict(k_exh=(3, 4), k_greedy=(3, 4, 5, 6, 7), sim_N=(2 ** 12, 2 ** 14), ghk_M=(2 ** 10,), ref_M=2 ** 14)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy", "gpytorch", "fastapi[standard]", "pydantic")
         .add_local_file(str(BOAZ / "modal_gp_api.py"), "/root/modal_gp_api.py"))
app = modal.App("orthant-rfc-t7-turf")


# ---------------------------------------------------------------------------
# Scorers (run in the container). Each returns reach for a (C, k) long tensor of portfolios.
# ---------------------------------------------------------------------------
def _sync():
    import torch
    torch.cuda.synchronize()


class Engine:
    def __init__(self, mu, R):
        import torch
        import modal_gp_api as M
        self.M, self.mu, self.R, self.dev = M, mu, R, mu.device

    def __call__(self, S):
        import torch
        C, k = S.shape
        u = -self.mu[S]
        s = self.R[S[:, :, None], S[:, None, :]]
        out = torch.empty(C, dtype=torch.float64, device=self.dev)
        chunk = self.M.orthant_rows_per_chunk(k, 27, 5, 8, self.dev)
        for a in range(0, C, chunk):
            b = min(a + chunk, C)
            uu, ss = self.M.resolve_degenerate_coords(u[a:b].contiguous(), s[a:b].contiguous(), 0.0)
            out[a:b] = 1 - self.M.orthant_prob(uu, ss, k, self.dev, order=1, chunk=b - a)[:, 0].clamp(0, 1)
        return out


class GHK:
    def __init__(self, mu, R, M):
        import torch
        self.mu, self.R, self.M, self.dev = mu, R, M, mu.device
        self.sob = {}

    def _sobol(self, k):
        import torch
        if k not in self.sob:
            self.sob[k] = torch.quasirandom.SobolEngine(k, scramble=True, seed=SEED + k).draw(
                self.M, dtype=torch.float64).to(self.dev)
        return self.sob[k]

    def __call__(self, S):
        import torch
        from torch.special import log_ndtr, ndtri
        C, k = S.shape
        f64 = torch.float64
        tiny, one = torch.finfo(f64).tiny, 1.0 - torch.finfo(f64).eps
        sob = self._sobol(k)
        eta = -self.mu[S]
        L = torch.linalg.cholesky(self.R[S[:, :, None], S[:, None, :]])
        out = torch.empty(C, dtype=f64, device=self.dev)
        chunk = max(1, 2 ** 23 // self.M)
        for a in range(0, C, chunk):
            b = min(a + chunk, C)
            e, Lc = eta[a:b], L[a:b]
            logp = e.new_zeros(b - a, self.M)
            us = []
            for j in range(k):
                c = e[:, j, None]
                for i in range(j):
                    c = c + Lc[:, j, i, None] * us[i]
                lp = log_ndtr(c / Lc[:, j, j, None])
                logp = logp + lp
                if j < k - 1:
                    us.append(-ndtri((sob[None, :, j] * lp.exp()).clamp(tiny, one)))
            out[a:b] = 1 - logp.exp().mean(1)
        return out


class Sim:
    """Bit-sliced shared-draw simulator: B[j] = item j's acceptance over N households, packed in
    64-bit words; reach(S) = popcount(OR_j B[S_j]) / N."""

    def __init__(self, mu, R, N):
        import torch
        assert N % 64 == 0
        dev = mu.device
        g = torch.Generator(device=dev).manual_seed(SEED)
        L = torch.linalg.cholesky(R)
        acc = (torch.randn(N, len(mu), generator=g, device=dev, dtype=torch.float64) @ L.T + mu) > 0  # (N, J)
        bits = acc.T.contiguous().view(len(mu), N // 8, 8).to(torch.uint8)
        w = (2 ** torch.arange(8, device=dev, dtype=torch.uint8))
        bytes_ = (bits * w).sum(-1, dtype=torch.uint8)                                             # (J, N/8)
        self.B = bytes_.view(torch.int64)                                                          # (J, N/64)
        self.lut = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int32, device=dev)
        self.N, self.dev = N, dev
        self.batch = max(1, 2 ** 26 // (N // 8))   # ~64 MB of uint8 per batch before the LUT gather

    def __call__(self, S):
        import torch
        C, k = S.shape
        out = torch.empty(C, dtype=torch.float64, device=self.dev)
        for a in range(0, C, self.batch):
            b = min(a + self.batch, C)
            acc = self.B[S[a:b, 0]]
            for i in range(1, k):
                acc = acc | self.B[S[a:b, i]]
            cnt = self.lut[acc.view(torch.uint8).long()].sum(1)
            out[a:b] = cnt.double() / self.N
        return out


def _combos(J, k, dev):
    import itertools
    import torch
    return torch.tensor(list(itertools.combinations(range(J), k)), dtype=torch.long, device=dev)


@app.function(image=image, gpu="T4", timeout=3600)
def bench(mu_np, R_np, cfg):
    import torch
    dev = torch.device("cuda")
    mu = torch.as_tensor(mu_np, dtype=torch.float64, device=dev)
    R = torch.as_tensor(R_np, dtype=torch.float64, device=dev)
    J = len(mu)
    # warm the CUDA context and kernels (not part of any scorer's setup)
    _ = GHK(mu, R, 64)(_combos(J, 2, dev)[:8]); _ = Engine(mu, R)(_combos(J, 2, dev)[:8]); _sync()

    settings = [("engine", None)] + [("sim", N) for N in cfg["sim_N"]] + [("ghk", M) for M in cfg["ghk_M"]]

    def make(name, par):
        return Engine(mu, R) if name == "engine" else Sim(mu, R, par) if name == "sim" else GHK(mu, R, par)

    exh, greedy, union_e, union_g = {}, {}, {k: set() for k in cfg["k_exh"]}, {k: set() for k in cfg["k_greedy"]}
    for name, par in settings:
        key = f"{name}" + (f"_{par}" if par else "")
        # exhaustive: setup counted at every k (each k is a separate decision)
        for k in cfg["k_exh"]:
            P = _combos(J, k, dev)
            _sync(); t0 = time.time()
            sc = make(name, par)
            r = sc(P)
            _sync(); t = time.time() - t0
            o = torch.argsort(r, descending=True)[:TOP]
            top = [tuple(x) for x in P[o].tolist()]
            exh.setdefault(key, {})[k] = dict(pick=list(top[0]), score=float(r[o[0]]), seconds=t, n=len(P), top=top)
            union_e[k].update(top)
            del P, r
        # greedy: setup once, path to max k (time = setup + path to k, cumulative)
        _sync(); t0 = time.time()
        sc = make(name, par)
        _sync(); t_setup = time.time() - t0
        S, gk, t_run = [], {}, t_setup
        for k in range(1, max(cfg["k_greedy"]) + 1):
            cand = [j for j in range(J) if j not in S]
            P = torch.tensor([S + [j] for j in cand], dtype=torch.long, device=dev)
            _sync(); t0 = time.time()
            r = torch.special.ndtr(mu[P[:, 0]]) if k == 1 else sc(P)  # k = 1: Phi(mu_j), identical for all
            _sync(); t_run += time.time() - t0
            o = torch.argsort(r, descending=True)
            S = P[o[0]].tolist()
            if k in cfg["k_greedy"]:
                top = [tuple(sorted(x)) for x in P[o[:TOP]].tolist()]
                gk[k] = dict(pick=sorted(S), score=float(r[o[0]]), seconds=t_run)
                union_g[k].update(top)
        greedy[key] = gk
        del sc
        torch.cuda.empty_cache()

    # reference: GHK at ref_M on the unions
    ref = GHK(mu, R, cfg["ref_M"])
    _sync(); t0 = time.time()
    refsc = {}
    for kind, U in (("exh", union_e), ("greedy", union_g)):
        for k, us in U.items():
            P = torch.tensor(sorted(us), dtype=torch.long, device=dev)
            r = ref(P)
            refsc[(kind, k)] = {tuple(p): float(v) for p, v in zip(P.tolist(), r.tolist())}
    _sync(); t_ref = time.time() - t0

    # every setting's own scores on the union (for near-tie ordering)
    own = {}
    for name, par in settings:
        key = f"{name}" + (f"_{par}" if par else "")
        sc = make(name, par)
        for kind, U in (("exh", union_e), ("greedy", union_g)):
            for k, us in U.items():
                P = torch.tensor(sorted(us), dtype=torch.long, device=dev)
                own[(key, kind, k)] = dict(zip([tuple(p) for p in P.tolist()], sc(P).tolist()))
        del sc
        torch.cuda.empty_cache()

    return dict(exh=exh, greedy=greedy, ref_seconds=t_ref, gpu=torch.cuda.get_device_name(),
                ref={f"{kind}|{k}": {",".join(map(str, p)): v for p, v in d.items()} for (kind, k), d in refsc.items()},
                own={f"{key}|{kind}|{k}": {",".join(map(str, p)): v for p, v in d.items()}
                     for (key, kind, k), d in own.items()})


# ---------------------------------------------------------------------------
# Local: fit, run, verdict
# ---------------------------------------------------------------------------
def verdict(res, cfg):
    out = {"exhaustive": {}, "greedy": {}}
    keys = list(res["exh"].keys())
    for kind, store, ks in (("exhaustive", res["exh"], cfg["k_exh"]), ("greedy", res["greedy"], cfg["k_greedy"])):
        rk = "exh" if kind == "exhaustive" else "greedy"
        for k in ks:
            ref = res["ref"][f"{rk}|{k}"]
            best_p, best = max(ref.items(), key=lambda x: x[1])
            row = {"ref_best": best_p, "ref_best_reach_pct": 100 * best, "settings": {}}
            for key in keys:
                e = store[key][k]
                pk = ",".join(map(str, sorted(e["pick"])))
                reg = 100 * (best - ref[pk])
                row["settings"][key] = dict(regret_pts=reg, seconds=e["seconds"])
            eng = row["settings"]["engine"]
            sims = sorted([(int(kk.split("_")[1]), v) for kk, v in row["settings"].items() if kk.startswith("sim_")])
            ok = [(N, v) for N, v in sims if v["regret_pts"] <= 0.5]
            row["sim_smallest_N_ok"] = ok[0][0] if ok else None
            if ok and eng["regret_pts"] <= 0.5:
                row["speedup_engine_vs_sim"] = ok[0][1]["seconds"] / eng["seconds"]
                row["engine_5x"] = row["speedup_engine_vs_sim"] >= 5
            else:
                row["speedup_engine_vs_sim"] = None
                row["engine_5x"] = False
            # secondary: near-tie pairs within 0.5 pts of the best, ordered as the reference orders them
            near = [p for p, v in ref.items() if best - v <= 0.005]
            pairs = [(a, b) for i, a in enumerate(near) for b in near[i + 1:]]
            row["near_tie_portfolios"] = len(near)
            row["near_tie_order"] = {}
            for key in keys:
                own = res["own"][f"{key}|{rk}|{k}"]
                good = sum((ref[a] - ref[b]) * (own[a] - own[b]) > 0 for a, b in pairs)
                row["near_tie_order"][key] = good / len(pairs) if pairs else None
            out[kind][k] = row
    k6 = out["exhaustive"].get(6, {}).get("engine_5x", False)
    gk = [k for k in cfg["k_greedy"] if k >= 6]
    gwins = sum(out["greedy"][k]["engine_5x"] for k in gk)
    out["PASS"] = bool(k6 and gwins > len(gk) / 2)
    out["greedy_engine_5x_count"] = f"{gwins} of {len(gk)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cfg = CHECK if a.check else FULL
    sys.path.insert(0, str(HERE.parent / "T6"))
    import t6
    Y1, _ = t6.load()
    fit = t6.fit_ifm(Y1)
    print(f"T6 fit: J = {len(fit['mu'])}, intercept-only (no covariates); rho median "
          f"{np.median(fit['R'][np.triu_indices(len(fit['mu']), 1)]):+.3f}", flush=True)
    t0 = time.time()
    with modal.enable_output(), app.run():
        res = bench.remote(fit["mu"], fit["R"], cfg)
    wall = time.time() - t0
    v = verdict(res, cfg)
    out = dict(cfg={k: list(v_) if isinstance(v_, tuple) else v_ for k, v_ in cfg.items()}, wall_seconds=wall,
               items=t6.ITEMS, gpu=res["gpu"], ref_seconds=res["ref_seconds"], verdict=v,
               exh={k: {kk: {x: y for x, y in vv.items() if x != "top"} for kk, vv in d.items()}
                    for k, d in res["exh"].items()},
               greedy=res["greedy"])
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"results{'_check' if a.check else ''}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"wall {wall:.0f} s (incl. container start); GPU {res['gpu']}; reference {res['ref_seconds']:.1f} s")
    for kind in ("exhaustive", "greedy"):
        for k, row in v[kind].items():
            s = row["settings"]
            print(f"{kind:10s} k={k:2d} best {row['ref_best_reach_pct']:.2f}%  near-ties {row['near_tie_portfolios']:3d}  "
                  + "  ".join(f"{kk} {x['regret_pts']:.2f}pt/{x['seconds']:.2f}s" for kk, x in s.items())
                  + f"  | sim N* {row['sim_smallest_N_ok']} speedup {row['speedup_engine_vs_sim']}")
    print("PASS" if v["PASS"] else "FAIL", "| greedy k>=6 engine>=5x:", v["greedy_engine_5x_count"])


if __name__ == "__main__":
    main()
