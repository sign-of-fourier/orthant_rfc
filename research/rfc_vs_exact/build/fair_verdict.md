**Verdict: no R up to 10⁴ meets the bar.** None picks the same best line and
also gets at least 9 of the reference's top 10. `orthant_cdf` scored all 10⁶
lines in 41 s.

- **R = 100 draws** (5 s) is useless for this. Its pick ranks 25,648th, with a profit
  gap of 0.31.
- **R = 1,000 draws** (46 s, about the same time as `orthant_cdf`) already picks the
  same best line, (279, 454, 459), so its profit gap is 0. It gets only 5 of
  the top 10 (Spearman 0.63 on the top 1,000).
- **R = 10,000 draws** (436 s, 10.6 times as long as `orthant_cdf`'s 41 s) also picks the same
  best line. It gets 8 of the top 10 and 84 of the top 100 (Spearman 0.82).
  The bar therefore needs more than 10,000 draws, so any RFC setting that
  meets it takes more than 10.6 times as long as `orthant_cdf`.

**Flag.** The best line is flagged: 454 and 459 differ only on A5, and its
difference covariances reach a correlation of 0.9 or more. So are 3 of the top
10 lines, 7 of the top 100 and 38 of the top 1,000. That is the regime where
renormalized `orthant_cdf` has share sd 0.0065 and max error 0.034 at
$d = 4$, so top-10 overlap is measured against a reference that is itself
uncertain there. In section 7, SciPy agreed with `orthant_cdf` on this winner
and on the top six lines.
