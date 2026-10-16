"""T14: MBC linked sets at k > 12 (analysis/T14/TEST.md).

Part A reuses T2's DGP with the menu size M = k; Part B reuses T3's loader and covariates on a 24-item
home-baking cluster.

    python analysis/T14/t14.py step0        # ML timing on an H100 (k = 30 L2 rep 0; baking) + GHK vs OVUS bench (T4)
"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
if len(HERE.parents) > 2:                                      # host only (not in the Modal container)
    for d in ("T2", "T3", "T9", "T11"):
        sys.path.insert(0, str(HERE.parent / d))
    sys.path.insert(0, str(HERE.parents[2] / "boaz" / "modal"))  # mvp_fit's fitting code (private sibling checkout)

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "scipy", "pybhatlib==0.4.0")
         .add_local_python_source("macml", "t11"))
# The deployed mvp_fit service's own code and packages, run in an ephemeral app on an H100 (user 2026-10-12).
fit_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("fastapi[standard]", "torch==2.5.1", "numpy", "scipy", "pydantic")
             .add_local_python_source("modal_gp_api"))
app = modal.App("orthant-rfc-t14")
OUT = HERE / "out"
ML_CAP_S = 3600            # approved cap; the deployed mvp_fit job's own timeout is also 3,600 s
STEP0_TIMEOUT = 360        # per H100 container (user cap ~$0.40 for the step-0 rerun)

BAKING = {
    "eggs": [("EGGS", ["EGGS - LARGE", "EGGS - X-LARGE", "EGGS - MEDIUM"])],
    "sugar": [("SUGARS/SWEETNERS", ["SUGAR"])],
    "flour": [("FLOUR & MEALS", ["FLOUR: WHITE & SELF RISING"])],
    "butter": [("BUTTER", ["BUTTER"])],
    "cream cheese": [("CHEESE", ["CREAM CHEESE"])],
    "cake mix": [("BAKING MIXES", ["LAYER CAKE MIX"])],
    "frosting": [("BAKING MIXES", ["FROSTING"])],
    "brownie mix": [("BAKING MIXES", ["BROWNIE MIX"])],
    "muffin/cornbread mix": [("BAKING MIXES", ["MUFFIN & CORN BREAD MIX"])],
    "chocolate chips": [("BAKING NEEDS", ["BITS & MORSELS"])],
    "marshmallows": [("BAKING NEEDS", ["MARSHMALLOWS"])],
    "baking nuts": [("BAKING NEEDS", ["BAKING NUTS"])],
    "oil": [("SHORTENING/OIL", ["VEGETABLE/SALAD OIL", "CANOLA OILS"])],
    "cooking spray": [("SHORTENING/OIL", ["COOKING SPRAYS"])],
    "shortening": [("SHORTENING/OIL", ["SOLID SHORTENING"])],
    "baking powder/soda": [("BAKING NEEDS", ["BAKING POWDER & SODA"])],
    "extracts": [("SPICES & EXTRACTS", ["PURE EXTRACTS", "IMITATION EXTRACTS"])],
    "pie filling": [("BAKING NEEDS", ["PIE FILLING MINCEMEAT GLAZES"])],
    "pie crust": [("BAKING MIXES", ["PIE CRUST MIXES & SHELLS"]), ("FROZEN PIE/DESSERTS", ["FRZN PIE SHELLS PASTRY SHELL"])],
    "whipped topping": [("FROZEN PIE/DESSERTS", ["FRZN WHIPPED TOPPING"])],
    "cake decor/icing": [("BAKING NEEDS", ["CAKE DECORS & ICING", "CAKE DECORS - BIRTHDAY CANDLES"])],
    "graham crackers": [("COOKIES/CONES", ["GRAHAM CRACKERS"])],
    "aluminum foil": [("FD WRAPS/BAGS/TRSH BG", ["ALUMINUM FOIL"])],
    "refrig. cookie dough": [("REFRGRATD DOUGH PRODUCTS", ["REFRIGERATED COOKIES-CHUB"])],
}
STAPLES = ("eggs", "sugar", "oil", "aluminum foil")


def sim_data(truth, k, rep):
    """T2's DGP with menu size M = k."""
    import t2
    t2.M = k
    return t2.make_data(truth, k, rep)


def remote_client():
    spec = importlib.util.spec_from_file_location(
        "mvp_remote", Path.home() / "projects" / "multivariate_probit" / "src" / "multivariate_probit" / "_remote.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_ml(client, X, Y, seed):
    """As T2/T3: the deployed mvp_fit service, 512 draws, no SEs; capped at ML_CAP_S."""
    t0 = time.time()
    try:
        out = client.fit_remote(X, Y.astype(float), n_draws=512, seed=seed, alpha=1e-6, se=False,
                                timeout=ML_CAP_S + 120)
    except Exception as e:                                      # timeout or the job's own 3,600 s limit
        return dict(ok=False, error=f"{type(e).__name__}: {e}"[:300], seconds=time.time() - t0)
    d, p = Y.shape[1], X.shape[1]
    return dict(ok=True, c=np.asarray(out["intercept"], float).tolist(),
                B=np.asarray(out["coef"], float).reshape(d, p).tolist(),
                R=np.asarray(out["corr"], float).reshape(d, d).tolist(),
                n_iter=int(out["n_iter"]), converged=bool(out["converged"]), seconds=time.time() - t0)


@app.function(image=image, gpu="T4", timeout=600)
def bench(R_by_d, n_obs=20_000, n_draws=512):
    """Per-observation log-likelihood value + gradient (w.r.t. the limits and the covariance), d = 20 / 30:
    GHK with n_draws scrambled Sobol points vs OVUS (T9's port). Batch n_obs, best of 3 after a warm-up."""
    import torch

    import macml
    import t11
    dev = torch.device("cuda")
    out = {}
    for d, R in R_by_d.items():
        R = torch.as_tensor(np.asarray(R), dtype=torch.float64, device=dev)
        g = torch.Generator(device=dev).manual_seed(int(d))
        a0 = torch.randn(n_obs, int(d), generator=g, dtype=torch.float64, device=dev) * 0.8
        sg = torch.where(torch.rand(n_obs, int(d), generator=g, device=dev, dtype=torch.float64) < 0.3, 1.0, -1.0)
        S0 = sg[:, :, None] * R * sg[:, None, :]
        u = torch.quasirandom.SobolEngine(int(d) - 1, scramble=True, seed=1).draw(n_draws, dtype=torch.float64).to(dev)

        def ghk():
            a, S = a0.clone().requires_grad_(True), S0.clone().requires_grad_(True)
            for lo in range(0, n_obs, 2000):
                p = t11.ghk_p(a[lo:lo + 2000] * sg[lo:lo + 2000], torch.linalg.cholesky(S[lo:lo + 2000]), u)
                p.clamp_min(1e-300).log().sum().backward()

        def ovus():
            a, S = a0.clone().requires_grad_(True), S0.clone().requires_grad_(True)
            for lo in range(0, n_obs, 5000):
                macml.ovus_log(a[lo:lo + 5000] * sg[lo:lo + 5000], S[lo:lo + 5000]).sum().backward()

        res = {}
        for name, fn in (("ghk", ghk), ("ovus", ovus)):
            fn()
            ts = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            res[name + "_us_per_obs"] = 1e6 * min(ts) / n_obs
        out[int(d)] = res
    out["gpu"] = torch.cuda.get_device_name()
    out["n_draws"], out["n_obs"] = n_draws, n_obs
    return out


@app.function(image=fit_image, gpu="H100", timeout=STEP0_TIMEOUT)
def fit_steps(X, Y, seed):
    """mvp_fit's fit (512 draws, float32 GHK, no SEs) at max_iter 1 then 5 in one container, yielding each.
    Phase-2 seconds per L-BFGS iteration = (t5 - t1) / 4; t1 also holds the margins phase and setup."""
    import torch

    import modal_gp_api as g
    dev = torch.device("cuda")
    for it in (1, 5):
        t0 = time.time()
        out = g.mvp_fit_core(X, Y.astype(float), None, g.MvpFitParams(n_draws=512, seed=seed, alpha=1e-6, se=False,
                                                                      max_iter=it), dev)
        torch.cuda.synchronize()
        yield dict(max_iter=it, seconds=time.time() - t0, n_iter=int(out["n_iter"]), nll=float(out["nll"]),
                   grad_max=float(out["grad_max"]), gpu=torch.cuda.get_device_name(),
                   peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)


def _save(name, obj):
    (OUT / "step0").mkdir(parents=True, exist_ok=True)
    (OUT / "step0" / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(name, obj, flush=True)


def step0():
    """Each result is saved as it arrives; a failure in one part does not discard the others."""
    import t2
    import t3
    X, Y, _, _ = sim_data("L2", 30, 0)
    tp = t2.truth_params("L2", 30)
    t3.CLUSTERS["baking"] = BAKING
    d = t3.load("baking", HERE.parents[1] / "data" / "completejourney")
    tr = d["train"]
    _save("data", dict(sim=dict(truth="L2", k=30, rep=0, n_obs=len(Y), n_cov=X.shape[1]),
                       real=dict(cluster="baking", n_hh=d["n_hh"], n_obs=int(tr.sum()), n_cov=d["X"].shape[1],
                                 names=d["names"], rate=d["rate"], lp_dropped=d["lp_dropped"])))

    def run_fit(name, X, Y):
        try:
            for r in fit_steps.remote_gen(X, Y, 20261011):
                _save(f"{name}_iter{r['max_iter']}", r)
        except Exception as e:
            _save(f"{name}_error", dict(error=f"{type(e).__name__}: {e}"[:300]))

    def run_bench():
        try:
            _save("bench", bench.remote({20: tp["R"][:20, :20].tolist(), 30: tp["R"].tolist()}))
        except Exception as e:
            _save("bench_error", dict(error=f"{type(e).__name__}: {e}"[:300]))

    t0 = time.time()
    with modal.enable_output(), app.run():
        with ThreadPoolExecutor(3) as ex:
            for f in (ex.submit(run_fit, "sim", X, Y), ex.submit(run_fit, "real", d["X"][tr], d["Y"][tr]), ex.submit(run_bench)):
                f.result()
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    {"step0": step0}[sys.argv[1]]()
