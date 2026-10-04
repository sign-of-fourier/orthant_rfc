import os
import nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# How accurate is `orthant_cdf` on choice shares?

This is a one-time accuracy test. Later comparisons of `orthant_cdf` with RFC
cite its table and do not recompute SciPy references. Each choice share at
$K = 5$ is a 4-dimensional orthant probability of the utility differences (see
`rfc_vs_exact.ipynb`, section 1). The model, design and part-worths are the
same as there.

**Scenarios**

- 300 stratified sets: 100 with an exact twin, 100 with a near-twin
  (differing only on A4 and A5), and 100 with no similar products.
- A **sweep** of 50 twin sets, each with $\sigma_P$ chosen so that the largest
  correlation of its difference covariances hits a target between 0.55 and
  0.999. As $\sigma_P \to \infty$ that correlation tends to 0.5, so 0.5 itself
  cannot be reached.

**Reference**: SciPy `multivariate_normal.cdf` with `abseps = releps = 1e-7`
and `maxpts = 4·10⁷`, computed once and cached in
`orthant_accuracy_ref.npz`. A rerun reuses the cache and only recomputes
`orthant_cdf`.

**Aggregation** uses 1,000 respondents ($\beta_n = \beta + 0.4 z_n$) on 2
scenarios per group. Its reference uses `abseps = releps = 1e-6`, 8 times
faster. That is still about 1,000 times smaller than the errors being
measured, and it is cached too.

Error means `orthant_cdf` − SciPy, in share units.
""")
code(r"""
import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import rfc
pd.set_option("display.precision", 5)
CACHE = "orthant_accuracy_ref.npz"
rng = np.random.default_rng(31)

# 300 stratified scenarios
X_s, kind_s = rfc.make_sets(300, rng)
V, S = rfc.utility_moments(X_s)
up_s, cov_s = rfc.orthant_problems(V, S)                       # (1500, 4), (1500, 4, 4)
group_s = np.repeat(kind_s, 5)
scen_s = np.repeat(np.arange(300), 5)

# correlation sweep: twin sets, sigma_P solved for each target max correlation
targets = 1 - np.geomspace(0.45, 0.001, 50)
X_tw, _ = rfc.make_sets(200, rng, kinds=("twin",))
sweep_X, sweep_sp, sweep_target = [], [], []
for x in X_tw:
    if len(sweep_X) == len(targets):
        break
    sp = rfc.sigma_p_for_corr(x, targets[len(sweep_X)])
    if sp is not None:
        sweep_X.append(x); sweep_sp.append(sp); sweep_target.append(targets[len(sweep_X) - 1])
sweep_X, sweep_sp = np.array(sweep_X), np.array(sweep_sp)
up_w, cov_w = [], []
for x, sp in zip(sweep_X, sweep_sp):
    u, c = rfc.orthant_problems(*rfc.utility_moments(x[None], sigma_p=sp))
    up_w.append(u); cov_w.append(c)
up_w, cov_w = np.concatenate(up_w), np.concatenate(cov_w)     # (250, 4), (250, 4, 4)
scen_w = np.repeat(np.arange(len(sweep_X)), 5)

# aggregation: 1,000 respondents on 2 scenarios per group
N_RESP = 1000
agg_idx = np.concatenate([np.where(kind_s == g)[0][:2] for g in ("twin", "near", "plain")])
B = rfc.respondent_betas(N_RESP, rng, spread=0.4)
up_a = np.array([[u for u, _ in rfc.shared_problems(X_s[i], B)] for i in agg_idx])    # (6, 5, 1000, 4)
cov_a = np.array([[c for _, c in rfc.shared_problems(X_s[i], B)] for i in agg_idx])   # (6, 5, 4, 4)
print(f"{len(up_s)} stratified orthants, {len(up_w)} sweep orthants (sigma_P {sweep_sp.min():.3f} to {sweep_sp.max():.2f}), "
      f"{up_a[..., 0].size} aggregation orthants")
""")
code(r"""
def load_or_compute():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if all(np.array_equal(z[k], v) for k, v in
               (("up_s", up_s), ("cov_s", cov_s), ("up_w", up_w), ("cov_w", cov_w), ("up_a", up_a), ("cov_a", cov_a))):
            print("reference loaded from", CACHE, f"(computed in {float(z['seconds']):.0f} s)")
            return z["ref_s"], z["ref_w"], z["ref_a"]
    t = time.perf_counter()
    ref_s = rfc.scipy_orthants(up_s, cov_s, tol=1e-7)
    ref_w = rfc.scipy_orthants(up_w, cov_w, tol=1e-7)
    flat_up = up_a.reshape(-1, 4)
    flat_cov = np.repeat(cov_a.reshape(-1, 4, 4), N_RESP, axis=0)
    ref_a = rfc.scipy_orthants(flat_up, flat_cov, tol=1e-6).reshape(up_a.shape[:3])
    seconds = time.perf_counter() - t
    np.savez(CACHE, up_s=up_s, cov_s=cov_s, ref_s=ref_s, up_w=up_w, cov_w=cov_w, ref_w=ref_w,
             up_a=up_a, cov_a=cov_a, ref_a=ref_a, seconds=seconds)
    print(f"reference computed in {seconds:.0f} s and cached")
    return ref_s, ref_w, ref_a

ref_s, ref_w, ref_a = load_or_compute()
print("reference self-check: stratified shares sum to 1 within", f"{np.abs(ref_s.reshape(300, 5).sum(1) - 1).max():.1e}")
""")
code(r"""
rfc.orthant_cdf(up_s[:5], cov_s[:5])                                 # wake the service
oc_s = rfc.orthant_cdf(up_s, cov_s)
oc_w = rfc.orthant_cdf(up_w, cov_w)
oc_a = np.array([[rfc.orthant_cdf(up_a[i, k], cov_a[i, k]) for k in range(5)] for i in range(len(agg_idx))])
err_s, err_w = oc_s - ref_s, oc_w - ref_w
corr_s, corr_w = rfc.max_diff_corr(cov_s), rfc.max_diff_corr(cov_w)  # per orthant
""")
md(r"""
## Error by scenario type

The bias test is clustered by scenario, because the 5 shares of one scenario
share their errors. It is a one-sample t-test on the per-scenario mean
signed error. "worst share at" gives the role of the share with the largest
error: a twin or near-twin member (alternatives 0 and 1) or another product.
""")
code(r"""
def summarize(err, scen, ref, role=None):
    per_scen = pd.Series(err).groupby(scen).mean()
    t = stats.ttest_1samp(per_scen, 0.0)
    i = np.argmax(np.abs(err))
    return dict(n_shares=len(err), bias=err.mean(), sd=err.std(ddof=1), max_abs=np.abs(err).max(),
                median_abs=np.median(np.abs(err)), bias_p_value=t.pvalue,
                worst_share_ref_p=ref[i], worst_share_at=None if role is None else role[i])

role_s = np.where(np.tile(np.arange(5), 300) < 2, "pair member", "other")
rows = {}
for g in ("twin", "near", "plain"):
    m = group_s == g
    rows[g] = summarize(err_s[m], scen_s[m], ref_s[m], np.where(role_s[m] == "pair member", f"{g} member", "other") if g != "plain" else role_s[m].astype(object) * 0 + "any")
rows["all 300"] = summarize(err_s, scen_s, ref_s)
rows["sweep (50 twin sets)"] = summarize(err_w, scen_w, ref_w)
by_group = pd.DataFrame(rows).T
by_group
""")
code(r"""
# twin and near groups split by share role: is the error on the pair members or on the others?
role_rows = {}
for g in ("twin", "near"):
    for r in ("pair member", "other"):
        m = (group_s == g) & (role_s == r)
        e = err_s[m]
        role_rows[f"{g}: {r}"] = dict(n=len(e), bias=e.mean(), sd=e.std(ddof=1), max_abs=np.abs(e).max(),
                                      bias_p_value=stats.ttest_1samp(pd.Series(e).groupby(scen_s[m]).mean(), 0).pvalue)
pd.DataFrame(role_rows).T
""")
code(r"""
# by size of the true share
pbins = [0, 0.05, 0.15, 0.3, 0.5, 1.0]
size_rows = {}
for lo, hi in zip(pbins[:-1], pbins[1:]):
    m = (ref_s >= lo) & (ref_s < hi)
    e = err_s[m]
    size_rows[f"[{lo}, {hi})"] = dict(n=m.sum(), bias=e.mean(), sd=e.std(ddof=1), max_abs=np.abs(e).max(),
                                     relative_bias=e.mean() / ref_s[m].mean(),
                                     bias_p_value=stats.ttest_1samp(pd.Series(e).groupby(scen_s[m]).mean(), 0).pvalue)
print("stratified 300, by true share"); display(pd.DataFrame(size_rows).T)
print(f"shares of a scenario sum to: mean {oc_s.reshape(300, 5).sum(1).mean():.5f}, "
      f"min {oc_s.reshape(300, 5).sum(1).min():.5f} (exact: 1)")
""")
md(r"""
## Error by correlation of the difference covariance

The bins use each orthant's own largest correlation. The sweep is reported
separately because it also changes $\sigma_P$.
""")
code(r"""
bins = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0 + 1e-12]
def by_corr(err, corr, scen):
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (corr >= lo) & (corr < hi)
        if m.sum() < 5:
            continue
        per = pd.Series(err[m]).groupby(scen[m]).mean()
        out[f"[{lo}, {min(hi, 1):.3g})"] = dict(n=m.sum(), bias=err[m].mean(), sd=err[m].std(ddof=1),
                                               max_abs=np.abs(err[m]).max(),
                                               bias_p_value=stats.ttest_1samp(per, 0).pvalue if len(per) > 2 else np.nan)
    return pd.DataFrame(out).T
print("stratified 300"); display(by_corr(err_s, corr_s, scen_s))
print("sweep");          display(by_corr(err_w, corr_w, scen_w))
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for g, mk in (("twin", "o"), ("near", "s"), ("plain", "^")):
    m = group_s == g
    ax[0].scatter(1 - corr_s[m], err_s[m], s=8, marker=mk, alpha=.5, label=g)
    ax[1].scatter(1 - corr_s[m], np.abs(err_s[m]) + 1e-8, s=8, marker=mk, alpha=.5, label=g)
ax[0].scatter(1 - corr_w, err_w, s=14, c="k", marker="x", label="sweep")
ax[1].scatter(1 - corr_w, np.abs(err_w) + 1e-8, s=14, c="k", marker="x", label="sweep")
for a in ax:
    a.set_xscale("log"); a.invert_xaxis(); a.set_xlabel("1 - max correlation of the difference covariance")
ax[0].axhline(0, c="gray", lw=.8); ax[0].set(ylabel="signed error (orthant_cdf - SciPy)", title="Signed error")
ax[1].set_yscale("log"); ax[1].set(ylabel="|error|", title="Absolute error")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()
""")
md(r"""
## Does the error average out across respondents?

For each of the 6 scenarios and 5 products, the market-share error is the
mean over 1,000 respondents of the per-respondent error. If the per-respondent
errors were independent noise, it would be about $\text{sd}/\sqrt{1000}$. If
they share a bias, it stays near the mean signed error.
""")
code(r"""
e_ind = oc_a - ref_a                                  # (6, 5, 1000)
agg_rows = []
for i, s in enumerate(agg_idx):
    for k in range(5):
        e = e_ind[i, k]
        agg_rows.append(dict(scenario=int(s), group=kind_s[s], product=k, market_share=ref_a[i, k].mean(),
                             individual_mean_abs=np.abs(e).mean(), individual_sd=e.std(ddof=1),
                             aggregate_error=e.mean(), noise_only_expect=e.std(ddof=1) / np.sqrt(N_RESP)))
agg = pd.DataFrame(agg_rows)
agg["aggregate / noise-only"] = agg.aggregate_error.abs() / agg.noise_only_expect
display(agg)
print(f"mean |individual error| {agg.individual_mean_abs.mean():.5f}; mean |aggregate error| "
      f"{agg.aggregate_error.abs().mean():.5f}; max |aggregate error| {agg.aggregate_error.abs().max():.5f}")
""")
md(r"""
## Summary
""")
code(r"""
summary = by_group[["n_shares", "bias", "sd", "max_abs", "bias_p_value"]].copy()
ag = agg.groupby("group").aggregate_error.agg(lambda x: x.abs().max())
summary["max |market-share error|, 1,000 resp."] = [ag.get(g, np.nan) for g in summary.index]
summary
""")
md(open(os.path.join(HERE, "acc_conclusion.md")).read())
md(r"""
## Renormalized shares

Each scenario's `orthant_cdf` shares are divided by their sum (for the
aggregation scenarios, each respondent's shares), and the error is recomputed
against the same cached reference.

Renormalizing makes a scenario's signed errors sum to zero, so the
per-scenario mean error, and with it the group "bias", is zero by
construction. A remaining bias would instead show up as a pattern across
shares: some shares too low and others too high. That is why the bias is
also reported by share size and by share role, with the t-test clustered by
scenario within each bin.
""")
code(r"""
def renorm(p, n_alt=5):
    q = p.reshape(-1, n_alt)
    return (q / q.sum(axis=1, keepdims=True)).ravel()

rn_s, rn_w = renorm(oc_s), renorm(oc_w)
rn_a = oc_a / oc_a.sum(axis=1, keepdims=True)                 # per respondent, over the 5 products
er_s, er_w = rn_s - ref_s, rn_w - ref_w
e_agg_rn = (rn_a - ref_a).mean(axis=2)                        # (6, 5) market-share errors
e_agg_raw = (oc_a - ref_a).mean(axis=2)

def stats_row(e, scen, agg_raw=None, agg_rn=None):
    return dict(n=len(e), bias=e.mean(), sd=e.std(ddof=1), max_abs=np.abs(e).max(),
                **({} if agg_rn is None else {"max |market-share error|, 1,000 resp.": np.abs(agg_rn).max(),
                                              "(before renormalizing)": np.abs(agg_raw).max()}))

agg_group = kind_s[agg_idx]
rows = {}
for g in ("twin", "near", "plain"):
    m = group_s == g
    rows[g] = stats_row(er_s[m], scen_s[m], e_agg_raw[agg_group == g], e_agg_rn[agg_group == g])
rows["sweep, all"] = stats_row(er_w, scen_w)
hi_w = np.repeat(np.array([rfc.max_diff_corr(cov_w[scen_w == j]).max() for j in range(len(sweep_X))]) >= 0.9, 5)
rows["sweep, max corr < 0.9"] = stats_row(er_w[~hi_w], scen_w[~hi_w])
rows["sweep, max corr >= 0.9"] = stats_row(er_w[hi_w], scen_w[hi_w])
renorm_tab = pd.DataFrame(rows).T
renorm_tab
""")
code(r"""
def binned(e, ref, scen, mask, label):
    out = {}
    for lo, hi in zip(pbins[:-1], pbins[1:]):
        m = mask & (ref >= lo) & (ref < hi)
        if m.sum() < 5:
            continue
        per = pd.Series(e[m]).groupby(scen[m]).mean()
        out[(label, f"[{lo}, {hi})")] = dict(n=m.sum(), bias=e[m].mean(), sd=e[m].std(ddof=1), max_abs=np.abs(e[m]).max(),
                                            bias_p_value=stats.ttest_1samp(per, 0).pvalue if len(per) > 2 else np.nan)
    return out

size_tab = {}
size_tab.update(binned(er_s, ref_s, scen_s, np.ones(len(er_s), bool), "stratified 300"))
size_tab.update(binned(er_w, ref_w, scen_w, ~hi_w, "sweep, corr < 0.9"))
size_tab.update(binned(er_w, ref_w, scen_w, hi_w, "sweep, corr >= 0.9"))
size_tab = pd.DataFrame(size_tab).T
display(size_tab)
role_rn = {}
for g in ("twin", "near"):
    for r in ("pair member", "other"):
        m = (group_s == g) & (role_s == r)
        role_rn[f"{g}: {r}"] = dict(n=m.sum(), bias=er_s[m].mean(), max_abs=np.abs(er_s[m]).max(),
                                    bias_p_value=stats.ttest_1samp(pd.Series(er_s[m]).groupby(scen_s[m]).mean(), 0).pvalue)
display(pd.DataFrame(role_rn).T)
""")
md(open(os.path.join(HERE, "acc_verdict.md")).read())
nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, os.path.join(HERE, "..", "orthant_accuracy.ipynb"))
