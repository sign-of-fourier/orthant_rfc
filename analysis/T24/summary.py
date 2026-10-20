"""T24 step 0 summary of out/clickouts.parquet, streamed in batches (4 GB box)."""
import collections, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

t = time.time()
pf = pq.ParquetFile("out/clickouts.parquet")
cols = ["user_id", "session_id", "reference", "platform", "city", "current_filters", "impressions"]

J = collections.Counter(); pos = collections.Counter()
users = collections.Counter(); cities = collections.Counter()
sessions = set(); platforms = set(); n = 0; n_filt = 0

for b in pf.iter_batches(batch_size=100_000, columns=cols):
    d = b.to_pandas()
    n += len(d)
    J.update((d.impressions.str.count(r"\|") + 1).value_counts().to_dict())
    for r, imp in zip(d.reference, d.impressions):
        k = 0
        for i, x in enumerate(imp.split("|"), 1):
            if x == r:
                k = i; break
        pos[k] += 1
    users.update(d.user_id.value_counts().to_dict())
    cities.update(d.city.value_counts().to_dict())
    sessions.update(d.session_id.unique()); platforms.update(d.platform.unique())
    n_filt += d.current_filters.notna().sum()

js = np.repeat(np.array(list(J)), np.array(list(J.values())))
print("clickouts", n, "secs", round(time.time() - t))
print("shelf size", {q: int(np.quantile(js, q)) for q in (0, .25, .5, .75, 1)})
hit = n - pos[0]
print("clicked item in shelf", round(hit / n, 3),
      "position of click (in shelf)", {k: round(pos[k] / hit, 3) for k in range(1, 11)})
u = pd.Series(users)
print("users", len(u), "users with >=2 clickouts", int((u >= 2).sum()), ">=5", int((u >= 5).sum()),
      "clickouts by users>=2", int(u[u >= 2].sum()))
print("sessions", len(sessions), "cities", len(cities), "platforms", len(platforms))
print("top cities", dict(cities.most_common(8)))
print("with filters", round(n_filt / n, 3))
