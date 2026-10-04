import os
import nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
md(r"""
# Fair product-line search: RFC with common random numbers vs `orthant_cdf`

Section 7 of `rfc_vs_exact.ipynb` compared exact scoring with RFC at
$R = 250{,}000$, which is the number of draws needed for 0.1-point accuracy
on every share. To rank lines, RFC only needs the differences between lines
to be right. This notebook gives RFC that advantage: one set of $R$ draws,
shared by every line (common random numbers), at $R = 10^2, 10^3, 10^4$.

**Setup: the same as section 7.**

- 720 candidate products, lines of 3, the same 3 competitors ($K = 6$, 5-dimensional orthants).
- The same margin objective.
- The aggregate model: true $\beta$, $\Sigma_A$, $\sigma_P$.

Section 7 has no respondents or posterior means, so none are used here.
Scoring $10^6$ lines over 1,000 respondents would take about $3 \cdot 10^9$
orthants, far beyond the 30-minute cap.

**Differences from section 7**

- Section 7 did not save its random lines, so $10^6$ lines are drawn again
  from a fixed seed. Section 7's best line, (279, 454, 459) from the greedy
  search, is added to them.
- Shares are renormalized `orthant_cdf`: all 6 shares of each line are
  computed and divided by their sum (see `orthant_accuracy.ipynb`). Lines
  where any difference covariance reaches a correlation of 0.9 or more are
  flagged, because there the error is larger (sd 0.0065, max 0.034 at
  $d = 4$).
- No SciPy.

**Reference ranking**: renormalized `orthant_cdf` profit. The profit gap of
each RFC pick is also measured with `orthant_cdf`.

**Runtime**: every configuration ran on all lines. The timing check put the
total at about 15 minutes, so no subset was needed.
""")
code(r"""
import json, time
import numpy as np, pandas as pd
from scipy import stats
import rfc
pd.set_option("display.precision", 4)

lev_all, price_all, X_all = rfc.all_products()
comp_lev = np.array([[1, 2, 0, 0, 1], [3, 2, 2, 0, 0], [1, 1, 2, 1, 0]])
comp_price = np.array([-1.0, 0.5, -0.5])
comp_idx = np.array([np.flatnonzero((lev_all == l).all(1) & (price_all == p))[0] for l, p in zip(comp_lev, comp_price)])
X_comp = X_all[comp_idx]
print("competitors (section 7):", comp_idx.tolist(), " margins", rfc.dollars(X_comp))

rng = np.random.default_rng(2026)
lines = rfc.random_lines(1_000_000, len(X_all), 3, rng)
sec7_best = np.array([279, 454, 459])
if not (lines == sec7_best).all(1).any():
    lines[-1] = sec7_best
print(f"{len(lines):,} distinct lines; section 7 best line included")
""")
code(r"""
t = time.perf_counter()
ref, ref_sh, ref_corr = rfc.line_profit_orthant_renorm(X_all[lines], X_comp)
t_orth = time.perf_counter() - t
flag = ref_corr >= 0.9
order = np.argsort(-ref)
best = order[0]
print(f"orthant_cdf (renormalized): {len(lines):,} lines, {6 * len(lines):,} orthants in {t_orth:.1f} s")
print(f"lines flagged (max difference-covariance corr >= 0.9): {flag.sum():,} of {len(lines):,}; "
      f"in the top 10: {flag[order[:10]].sum()}, top 100: {flag[order[:100]].sum()}, top 1,000: {flag[order[:1000]].sum()}")
print("reference best line:", lines[best].tolist(), f"profit {ref[best]:.4f}", "(flagged)" if flag[best] else "")
for i in lines[best]:
    print("   levels", lev_all[i].tolist(), f"price {price_all[i]:+.1f}", f"margin ${rfc.dollars(X_all[i]):.2f}")
print(f"profit spread: top 10 {ref[order[0]] - ref[order[9]]:.4f}, top 100 {ref[order[0]] - ref[order[99]]:.4f}")
""")
md(r"""
## RFC with common random numbers

Each $R$ uses one fixed draw of $(E_A, E_P)$ for all $10^6$ lines (RFC variant
a, the same error model as the exact shares). Wall times are measured on the
full set of lines. Spearman correlation is computed on the reference's top
1,000 lines.
""")
code(r"""
rows, picks = [], {}
for R in (100, 1_000, 10_000):
    t = time.perf_counter()
    p_rfc, _ = rfc.line_profit_rfc_crn(X_all[lines], X_comp, R, np.random.default_rng(R))
    dt = time.perf_counter() - t
    o = np.argsort(-p_rfc, kind="stable")
    picks[R] = o[0]
    rows.append(dict(R=R, seconds=dt, same_best=bool(o[0] == best),
                     top10_overlap=len(np.intersect1d(o[:10], order[:10])),
                     top100_overlap=len(np.intersect1d(o[:100], order[:100])),
                     spearman_top1000=stats.spearmanr(ref[order[:1000]], p_rfc[order[:1000]]).statistic,
                     pick=lines[o[0]].tolist(), pick_ref_rank=int(np.flatnonzero(order == o[0])[0]) + 1,
                     profit_gap=ref[best] - ref[o[0]], pick_flagged=bool(flag[o[0]])))
rows.append(dict(R="orthant_cdf (reference)", seconds=t_orth, same_best=True, top10_overlap=10, top100_overlap=100,
                 spearman_top1000=1.0, pick=lines[best].tolist(), pick_ref_rank=1,
                 profit_gap=0.0, pick_flagged=bool(flag[best])))
fair = pd.DataFrame(rows).set_index("R")
fair
""")
code(r"""
ok = fair.iloc[:-1]
ok = ok[ok.same_best & (ok.top10_overlap >= 9)]
if len(ok):
    R_min = int(ok.index[0]); t_min = float(ok.seconds.iloc[0])
    verdict = (f"smallest R with the same best line and top-10 overlap >= 9: R = {R_min:,}, "
               f"{t_min:.0f} s; orthant_cdf took {t_orth:.0f} s, speedup {t_min / t_orth:.2f}x")
else:
    R_min, t_min = None, None
    verdict = f"no R up to 10,000 picks the same best line with top-10 overlap >= 9; orthant_cdf took {t_orth:.0f} s"
print(verdict)
json.dump(dict(R_min=R_min, t_rfc=t_min, t_orth=t_orth, n_lines=len(lines), verdict=verdict,
               table=json.loads(fair.reset_index().to_json(orient="records"))),
          open("product_line_fair.json", "w"), indent=1)
""")
md(open(os.path.join(HERE, "fair_verdict.md")).read())
nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, os.path.join(HERE, "..", "product_line_fair.ipynb"))
