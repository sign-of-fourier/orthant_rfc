"""T1 summary over replicates: python analysis/T1/summarize.py out/full_G1/results.json"""
import json, sys, numpy as np
r = json.load(open(sys.argv[1]))
P = 9
def a(m, k): return np.array([x[m][k] for x in r], dtype=float)
print("reps", len(r), "S1 true", r[0]["truth"]["S1"])
for m in ("mvp", "rfc_s"):
    e = a(m, "S1_err")
    print(f"{m}: S1 median {np.median(a(m,'S1')):.3f}, median|err| {np.median(abs(e)):.3f}, mean err {e.mean():+.3f} sd {e.std(ddof=1):.3f}; S3 median {np.median(a(m,'S3_ratio')):.3f}")
    print(f"   flips price {int(a(m,'price_flip').sum())} ext {int(a(m,'ext_flip_at_placeholder_h').sum())} cost {int(a(m,'cost_flip').sum())}; "
          f"regret mean price {a(m,'price_regret_pct').mean():.2f}% cost {a(m,'cost_regret_pct').mean():.2f}%; ext_incr_err mean {a(m,'ext_incr_err').mean():+.3f} mae {abs(a(m,'ext_incr_err')).mean():.3f}; holdout MAE {a(m,'holdout_mae').mean():.4f}")
d = (abs(a("rfc_s","S1_err")) - abs(a("mvp","S1_err")))
print(f"paired |S1 err| rfc - mvp: mean {d.mean():.3f}, 95% CI ±{1.96*d.std(ddof=1)/np.sqrt(len(d)):.3f}")
S2 = np.array([x["mvp"]["S2"] for x in r]); print("MVP S2 mean", S2.mean(0).round(3), "true", (np.square(r[0]["sigma_true"])/np.sum(np.square(r[0]["sigma_true"]))).round(3))
st = np.array(r[0]["sigma_true"][:3]); cov = []
for x in r:
    th, se = np.array(x["mvp"]["theta"]), np.array(x["mvp"]["se_sandwich"])
    tt = np.array(x["mvp"]["theta_true"])
    cov.append(np.abs((th-tt)/se)[2*P:] <= 1.96)
print("S4 coverage (log sigma) per attr", np.mean(cov, 0), "sigma mean", np.array([x["mvp"]["sigma"][:3] for x in r]).mean(0).round(3), "true", st.round(3))
z = np.array([(np.array(x["mvp"]["theta"])-np.array(x["mvp"]["theta_true"]))/np.array(x["mvp"]["se_sandwich"]) for x in r])
print("mean z all params", z.mean(0).round(2))
print("rfc flags", sum(bool(x["rfc_s"]["flags"]) for x in r), "| mvp ll>truth", sum(x["mvp"]["ll"]>=x["mvp"]["ll_true"] for x in r), "| mvp fit s median", np.median([x["mvp"]["seconds"]["fit"] for x in r]))
