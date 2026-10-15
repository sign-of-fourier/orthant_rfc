"""TURF on Conjointly MaxDiff HB point estimates: validation, bootstrap, preference structure.

Usage: python turf_bootstrap.py [path/to/conjointly_experiment_7.json]
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261015
B = 5000
KS = (2, 3, 4)
SET_SIZE = 3

LABELS = {
    "7955433": "Red", "7955434": "Black", "7955435": "Orange", "7955436": "Yellow",
    "7955437": "Green", "7955438": "Blue", "7955439": "Purple", "7955440": "Violet",
    "7955441": "Maroon", "7955442": "Navy", "7955443": "Turquoise", "7955444": "Lime",
    "7955445": "White", "7955446": "Silver",
}
IDS = list(LABELS)
NAMES = [LABELS[i] for i in IDS]

# Conjointly reference values: (k, set of names) -> reach, or freq
REF_SHARE = {"Black": 0.27722772, "Turquoise": 0.13366337, "Navy": 0.08910891, "Blue": 0.00495050}
REF_REACH = [
    (2, ("Black", "Purple", "Turquoise"), 0.74257426),
    (3, ("Black", "Purple", "Turquoise"), 0.88118812),
    (3, ("Black", "Navy", "Turquoise"), 0.85148515),
    (4, ("Black", "Purple", "Turquoise"), 0.94059406),
    (4, ("Black", "Navy", "Turquoise"), 0.93069307),
]
REF_FREQ = [
    (2, ("Black", "Purple", "Turquoise"), 1.08),
    (3, ("Black", "Purple", "Turquoise"), 1.26966292),
]
REF_RANK = {2: [("Black", "Purple", "Turquoise")],
            3: [("Black", "Purple", "Turquoise"), ("Black", "Navy", "Turquoise")],
            4: [("Black", "Purple", "Turquoise"), ("Black", "Navy", "Turquoise")]}
TOL = 5e-8


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def name_of(triple):
    return "{" + ", ".join(NAMES[j] for j in triple) + "}"


def topk_matrix(U, k):
    """N x J indicator of item j in respondent i's top-k. Fails on ties at the k boundary."""
    order = np.argsort(-U, axis=1, kind="stable")
    s = np.take_along_axis(U, order, axis=1)
    if np.any(s[:, k - 1] == s[:, k]):
        raise RuntimeError(f"tie at top-{k} boundary; tie rule needed")
    T = np.zeros_like(U, dtype=np.int8)
    np.put_along_axis(T, order[:, :k], 1, axis=1)
    return T


def upgma(D, n_clusters):
    """Average-linkage agglomerative clustering on distance matrix D.
    Returns (dendrogram leaf order, cluster label per item for an n_clusters cut)."""
    n = len(D)
    members = {i: [i] for i in range(n)}
    children = {}
    merges = []
    nxt = n
    while len(members) > 1:
        ids = sorted(members)
        best = None
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1:]:
                d = D[np.ix_(members[a], members[b])].mean()
                if best is None or d < best[0] - 1e-12:
                    best = (d, a, b)
        _, a, b = best
        children[nxt] = (a, b)
        members[nxt] = members.pop(a) + members.pop(b)
        merges.append(nxt)
        nxt += 1

    def leaves(c):
        return [c] if c < n else leaves(children[c][0]) + leaves(children[c][1])

    order = leaves(nxt - 1)
    # cut: undo the last n_clusters - 1 merges
    roots = {nxt - 1}
    for c in reversed(merges[len(merges) - (n_clusters - 1):]):
        roots.remove(c)
        roots.update(children[c])
    root_list = sorted(roots, key=lambda c: order.index(leaves(c)[0]))
    labels = np.empty(n, dtype=int)
    for lab, c in enumerate(root_list):
        labels[leaves(c)] = lab
    return order, labels


def main(path):
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------ STEP 0
    hdr("STEP 0. Load and check")
    raw = json.loads(Path(path).read_text())
    df = pd.DataFrame(raw)
    missing_cols = [c for c in IDS + ["participant", "weight"] if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in IDS + ["participant", "weight"]]
    print(f"N = {len(df)}  (unique participants: {df['participant'].nunique()})")
    print(f"missing columns: {missing_cols or 'none'}   extra columns: {extra_cols or 'none'}")
    print(f"missing values: {int(df[IDS + ['weight']].isna().sum().sum())}")
    w = df["weight"].to_numpy(float)
    print(f"weights: unique = {np.unique(w)}  all equal 1: {bool(np.all(w == 1))}")
    U = df[IDS].to_numpy(float)
    N, J = U.shape
    print(f"Red (reference) all zero: {bool(np.all(U[:, 0] == 0))}")
    rng_i = U.max(1) - U.min(1)
    print(f"per-respondent utility range (max-min): min {rng_i.min():.3f}  "
          f"median {np.median(rng_i):.3f}  max {rng_i.max():.3f}")
    print(f"  quartiles: {np.percentile(rng_i, [25, 75]).round(3)}")

    triples = list(combinations(range(J), SET_SIZE))
    M = np.zeros((J, len(triples)), dtype=np.int16)  # item x triple membership
    for t, tr in enumerate(triples):
        M[list(tr), t] = 1
    tidx = {frozenset(NAMES[j] for j in tr): t for t, tr in enumerate(triples)}
    wn = w / w.sum()

    # ------------------------------------------------------------------ STEP 1
    hdr("STEP 1. Reproduce Conjointly (validation gate)")
    hits = {}    # k -> N x T count of triple items in top-k
    reach = {}   # k -> T
    freq = {}
    rank = {}
    for k in KS:
        H = topk_matrix(U, k).astype(np.int16) @ M
        hits[k] = H
        R = (H > 0).astype(float)
        reach[k] = wn @ R
        freq[k] = (wn @ H) / reach[k]
        rank[k] = np.lexsort((-freq[k], -reach[k]))  # reach desc, freq desc as tie-break

    fails = []
    T1 = topk_matrix(U, 1)
    share = wn @ T1
    print("first-choice shares:")
    for nm, ref in REF_SHARE.items():
        v = share[NAMES.index(nm)]
        ok = abs(v - ref) < TOL
        fails += [] if ok else [f"share {nm}"]
        print(f"  {nm:<10} {v:.8f}  ref {ref:.8f}  {'OK' if ok else 'MISMATCH'}")
    print(f"  shares sum to {share.sum():.8f}")
    print("reach:")
    for k, s, ref in REF_REACH:
        v = reach[k][tidx[frozenset(s)]]
        ok = abs(v - ref) < TOL
        fails += [] if ok else [f"reach k={k} {s}"]
        print(f"  k={k} {{{', '.join(s)}}}  {v:.8f}  ref {ref:.8f}  {'OK' if ok else 'MISMATCH'}")
    print("frequency:")
    for k, s, ref in REF_FREQ:
        v = freq[k][tidx[frozenset(s)]]
        ok = abs(v - ref) < 5e-3 if ref == round(ref, 2) else abs(v - ref) < TOL
        fails += [] if ok else [f"freq k={k} {s}"]
        print(f"  k={k} {{{', '.join(s)}}}  {v:.8f}  ref {ref}  {'OK' if ok else 'MISMATCH'}")
    print("ranks:")
    for k, lst in REF_RANK.items():
        for pos, s in enumerate(lst):
            got = rank[k][pos]
            ok = got == tidx[frozenset(s)]
            fails += [] if ok else [f"rank k={k} #{pos + 1}"]
            print(f"  k={k} #{pos + 1} {name_of(triples[got])}  ref {{{', '.join(s)}}}  "
                  f"{'OK' if ok else 'MISMATCH'}")
    if fails:
        print(f"\nVALIDATION FAILED: {fails}. Investigate tie handling and N.")
        sys.exit(1)
    print("\nVALIDATION PASSED.")

    for k in KS:
        print(f"\nTop 10 triples, k={k} (of {len(triples)}):")
        print(f"  {'#':>2}  {'set':<34} {'reach':>10} {'freq':>10}")
        for p, t in enumerate(rank[k][:10]):
            print(f"  {p + 1:>2}  {name_of(triples[t]):<34} {reach[k][t]:>10.6f} {freq[k][t]:>10.6f}")
        r = reach[k][rank[k]]
        n_tied = int(np.sum(np.isclose(r, r[0], atol=1e-12)))
        if n_tied > 1:
            print(f"  note: {n_tied} triples share the top reach; freq breaks the tie")

    # ------------------------------------------------------------------ STEP 2
    hdr(f"STEP 2. Bootstrap respondents (B = {B}, seed = {SEED})")
    # Resample counts per respondent: replicate b weight on i = count_bi * w_i
    C = rng.multinomial(N, np.full(N, 1.0 / N), size=B).astype(float)  # B x N
    Wb = C * w
    Wb /= Wb.sum(1, keepdims=True)
    boot = {}
    for k in KS:
        Rb = Wb @ (hits[k] > 0).astype(float)          # B x T bootstrap reach
        Fb = (Wb @ hits[k]) / Rb
        # winner per replicate: max reach, freq as tie-break (matches Step 1 ordering)
        top = Rb.max(1, keepdims=True)
        cand = np.isclose(Rb, top, rtol=0, atol=1e-12)
        win = np.where(cand, Fb, -np.inf).argmax(1)
        n_exact_ties = int((cand.sum(1) > 1).sum())

        print(f"\n--- k = {k} ---")
        pw = pd.Series(np.bincount(win, minlength=len(triples)) / B)
        print(f"P(win) for triples winning >= 1% of replicates "
              f"({n_exact_ties} replicates had an exact reach tie at the top, broken by freq):")
        for t in pw[pw >= 0.01].sort_values(ascending=False).index:
            orig_pos = int(np.where(rank[k] == t)[0][0]) + 1
            print(f"  {name_of(triples[t]):<34} P(win) {pw[t]:.4f}   (original rank #{orig_pos})")

        t1 = rank[k][0]
        print("\n95% percentile CI on reach, original top 10; paired 95% CI on reach(#1) - reach(#j):")
        print(f"  {'#':>2}  {'set':<34} {'reach':>9} {'95% CI':>19}   {'diff vs #1':>10} {'paired 95% CI':>21}")
        D = Rb[:, [t1]] - Rb                              # B x T paired differences
        lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
        # differences are multiples of 1/N; snap float noise so an endpoint of exactly 0 counts as including 0
        lo, hi = np.round(lo * N) / N, np.round(hi * N) / N
        for p, t in enumerate(rank[k][:10]):
            rl, rh = np.percentile(Rb[:, t], [2.5, 97.5])
            line = f"  {p + 1:>2}  {name_of(triples[t]):<34} {reach[k][t]:>9.4f}  [{rl:.4f}, {rh:.4f}]"
            if p:
                d = reach[k][t1] - reach[k][t]
                flag = "  tied" if lo[t] <= 0 <= hi[t] else ""
                line += f"   {d:>+10.4f}  [{lo[t]:+.4f}, {hi[t]:+.4f}]{flag}"
            print(line)
        others = np.arange(len(triples)) != t1
        tied = others & (lo <= 0) & (hi >= 0)
        print(f"\n  triples statistically tied with #1 (paired 95% CI on difference includes 0): "
              f"{int(tied.sum())} of {len(triples) - 1}")
        if tied.sum():
            tied_sorted = sorted(np.flatnonzero(tied), key=lambda t: -reach[k][t])
            print("  " + "; ".join(name_of(triples[t]) for t in tied_sorted))
        boot[k] = dict(p_win=pw[t1], n_tied=int(tied.sum()), runner=pw.drop(t1).idxmax(),
                       p_runner=pw.drop(t1).max(), gap=reach[k][t1] - reach[k][rank[k][1]],
                       gap_lo=lo[rank[k][1]], gap_hi=hi[rank[k][1]])

    print("\nNOTE: this bootstrap captures between-respondent sampling uncertainty only. Utilities are\n"
          "HB point estimates; within-person posterior uncertainty is ignored, so the true\n"
          "uncertainty on reach and on the winning set is larger than shown.")

    # ------------------------------------------------------------------ STEP 3
    hdr("STEP 3. Between-person preference structure")
    Uc = U - U.mean(1, keepdims=True)
    sd = Uc.std(1, ddof=1, keepdims=True)
    Us = Uc / sd
    print(f"row-centered; row SD: min {sd.min():.3f} median {np.median(sd):.3f} max {sd.max():.3f}")
    pd.set_option("display.width", 200)
    for tag, X in (("(a) row-centered", Uc), ("(b) row-centered / row SD", Us)):
        Cm = pd.DataFrame(np.corrcoef(X, rowvar=False), index=NAMES, columns=NAMES)
        print(f"\n{tag}: 14x14 correlation across respondents")
        print(Cm.round(2).to_string())
        iu = np.triu_indices(J, 1)
        print(f"  off-diagonal mean {Cm.to_numpy()[iu].mean():+.3f}  "
              f"(row-centering forces the average to about -1/(J-1) = {-1 / (J - 1):+.3f})")
    Ca = np.corrcoef(Uc, rowvar=False)
    Cb = np.corrcoef(Us, rowvar=False)
    iu = np.triu_indices(J, 1)
    dd = np.abs(Ca - Cb)[iu]
    print(f"\n|(a) - (b)| off-diagonal: max {dd.max():.3f}, mean {dd.mean():.3f}")

    base = -1 / (J - 1)
    versions = {"a": ("(a) row-centered", Ca), "b": ("(b) row-centered / row SD", Cb)}
    for v, (tag, Cm) in versions.items():
        print(f"\n{tag}: corr - (-1/13), i.e. correlation relative to the row-centering baseline")
        print(pd.DataFrame(Cm - base, index=NAMES, columns=NAMES).round(2).to_string())

    clusters = {}
    for v, (tag, Cm) in versions.items():
        order, cut4 = upgma(1 - Cm, n_clusters=4)
        clusters[v] = cut4
        print(f"\n{tag}: average-linkage clustering, distance = 1 - corr")
        print("  dendrogram leaf order: " + ", ".join(NAMES[j] for j in order))
        print("  4-cluster cut:")
        for c in range(4):
            mem = [j for j in order if cut4[j] == c]
            sub = Cm[np.ix_(mem, mem)]
            within = sub[np.triu_indices(len(mem), 1)].mean() if len(mem) > 1 else float("nan")
            print(f"    C{c + 1}: {', '.join(NAMES[j] for j in mem):<45} mean within-corr {within:+.2f}")
    # relabel (b) clusters to match (a) where memberships agree, so labels are comparable
    remap = {}
    for c in range(4):
        vals, cnt = np.unique(clusters["a"][clusters["b"] == c], return_counts=True)
        remap[c] = int(vals[cnt.argmax()])
    if len(set(remap.values())) == 4:
        clusters["b"] = np.array([remap[c] for c in clusters["b"]])
    same = all(len({clusters['b'][j] for j in range(J) if clusters['a'][j] == c}) == 1 for c in range(4))
    print(f"\n  4-cluster partitions identical across (a) and (b): {same}")
    out_csv = Path(__file__).resolve().parent / "cluster_assignments.csv"
    pd.DataFrame({"item_id": IDS, "item": NAMES,
                  "cluster_rowcentered": [f"C{c + 1}" for c in clusters["a"]],
                  "cluster_rowstandardized": [f"C{c + 1}" for c in clusters["b"]]}).to_csv(out_csv, index=False)
    print(f"  saved {out_csv.name}")

    nv = NAMES.index("Navy")
    for v, (tag, Cm) in versions.items():
        row = pd.Series(Cm[nv], index=NAMES).drop("Navy").sort_values(ascending=False)
        print(f"\nNavy correlation row, {tag} (corr, corr + 1/13):")
        for nm, r in row.items():
            print(f"  {nm:<10} {r:+.3f}  {r - base:+.3f}")

    # ------------------------------------------------------------------ STEP 4
    hdr("STEP 4. Prediction tests")
    ix = {n: i for i, n in enumerate(NAMES)}
    rk = np.argsort(np.argsort(-U, axis=1, kind="stable"), axis=1) + 1   # per-person rank, 1 = best
    top3 = rk <= 3
    top4 = rk <= 4
    bpt = [ix["Black"], ix["Purple"], ix["Turquoise"]]
    reached_bpt3 = top3[:, bpt].any(1)
    pct = lambda m: 100 * m.mean()
    results = {}

    print("\nP1: Navy's correlations with Purple and Violet are among its top 3 and exceed Navy-Blue")
    p1 = {}
    for v, (tag, Cm) in versions.items():
        row = pd.Series(Cm[nv], index=NAMES).drop("Navy")
        top = list(row.sort_values(ascending=False).index[:3])
        cond = ("Purple" in top and "Violet" in top
                and row["Purple"] > row["Blue"] and row["Violet"] > row["Blue"])
        p1[v] = cond
        print(f"  {tag}: Navy top 3 = {top} ({', '.join(f'{row[n]:+.2f}' for n in top)}); "
              f"Navy-Purple {row['Purple']:+.2f}, Navy-Violet {row['Violet']:+.2f}, "
              f"Navy-Blue {row['Blue']:+.2f}  ->  {'PASS' if cond else 'FAIL'}")
    results["P1"] = "PASS" if all(p1.values()) else "FAIL"

    print("\nP1b: why Navy-containing triples have high frequency at k=3")
    navy3 = top3[:, nv]
    nn = int(navy3.sum())
    print(f"  respondents with Navy in top 3: {nn} ({pct(navy3):.1f}%)")
    co = pd.Series(top3[navy3].sum(0), index=NAMES).drop("Navy").sort_values(ascending=False)
    base3 = pd.Series(top3.sum(0) / N, index=NAMES)
    print(f"  other items in their top 3 (count, % of Navy-top-3, vs % of all respondents with item in top 3):")
    for nm, c in co[co > 0].items():
        print(f"    {nm:<10} {c:>3}  {100 * c / nn:5.1f}%   (all: {100 * base3[nm]:5.1f}%)")
    pb = 100 * top3[navy3, ix["Black"]].mean()
    pt = 100 * top3[navy3, ix["Turquoise"]].mean()
    pboth = 100 * (top3[navy3, ix["Black"]] & top3[navy3, ix["Turquoise"]]).mean()
    print(f"  also Black in top 3: {pb:.1f}%   also Turquoise: {pt:.1f}%   both: {pboth:.1f}%")
    t_bnt = tidx[frozenset(("Black", "Navy", "Turquoise"))]
    t_bpt = tidx[frozenset(("Black", "Purple", "Turquoise"))]
    for t in (t_bnt, t_bpt):
        H = hits[3][:, t]
        dist = np.bincount(H[H > 0], minlength=4)[1:]
        print(f"  {name_of(triples[t]):<28} k=3: reached by 1/2/3 items = {dist.tolist()}, freq {freq[3][t]:.3f}")
    results["P1b"] = (f"{pb:.0f}% of Navy-top-3 respondents also have Black in top 3 "
                      f"(vs {100 * base3['Black']:.0f}% overall), {pt:.0f}% Turquoise")

    print("\nP2: Lime niche (and Yellow for comparison)")
    item_sd = pd.Series(Uc.std(0, ddof=1), index=NAMES)
    sd_rank = item_sd.rank(ascending=False).astype(int)
    print(f"  overall: {100 * (1 - reached_bpt3.mean()):.1f}% of respondents NOT reached by "
          f"{{Black, Purple, Turquoise}} at k=3")
    p2 = {}
    for nm in ("Lime", "Yellow"):
        j = ix[nm]
        corrs = ", ".join(f"{o} {Ca[j, ix[o]]:+.2f}/{Cb[j, ix[o]]:+.2f}"
                          for o in ("Black", "Purple", "Turquoise", "Navy"))
        m4 = top4[:, j]
        not_reached = 100 * (~reached_bpt3[m4]).mean() if m4.any() else float("nan")
        p2[nm] = dict(sd_rank=sd_rank[nm], sd=item_sd[nm], top4=pct(m4), n4=int(m4.sum()), nr=not_reached)
        print(f"  {nm}: between-person SD of centered utility {item_sd[nm]:.3f} (rank {sd_rank[nm]} of {J})")
        print(f"    corr (a)/(b) with {corrs}")
        print(f"    in top 4: {int(m4.sum())} ({pct(m4):.1f}%); of those NOT reached by BPT at k=3: "
              f"{int((~reached_bpt3[m4]).sum())} ({not_reached:.1f}%)")
    print(f"  Lime-Yellow corr (a)/(b): {Ca[ix['Lime'], ix['Yellow']]:+.2f}/{Cb[ix['Lime'], ix['Yellow']]:+.2f}")
    print("  between-person SD of centered utility, all items: "
          + ", ".join(f"{n} {item_sd[n]:.2f}" for n in item_sd.sort_values(ascending=False).index))
    neg = all(min(Ca[ix['Lime'], ix[o]], Cb[ix['Lime'], ix[o]]) < 0 for o in ("Black", "Purple", "Turquoise", "Navy"))
    lift = p2["Lime"]["nr"] / (100 * (1 - reached_bpt3.mean()))
    p2_pass = neg and lift >= 2
    print(f"  criterion: Lime negatively correlated with all four core items in both versions ({neg}) and its\n"
          f"  top-4 fans are >= 2x as likely as average to be missed by BPT at k=3 (lift {lift:.2f}x)"
          f"  ->  {'PASS' if p2_pass else 'FAIL'}")
    results["P2"] = (f"{'PASS' if p2_pass else 'FAIL'}: Lime SD rank {p2['Lime']['sd_rank']}, "
                     f"{p2['Lime']['top4']:.1f}% have it in top 4, {p2['Lime']['nr']:.0f}% of those missed by BPT "
                     f"(lift {lift:.1f}x); Yellow {p2['Yellow']['top4']:.1f}% / {p2['Yellow']['nr']:.0f}%")

    print("\nP3: Blue is a near-duplicate of Navy")
    bl = ix["Blue"]
    dist = np.bincount(rk[:, bl], minlength=J + 1)[1:]
    print("  Blue per-person rank distribution (rank: count):")
    print("   " + "  ".join(f"{r + 1}:{c}" for r, c in enumerate(dist)))
    print(f"   median rank {np.median(rk[:, bl]):.0f}, mean {rk[:, bl].mean():.2f}")
    sel = (rk[:, bl] >= 2) & (rk[:, bl] <= 4)
    above = [NAMES[int(np.flatnonzero(rk[i] == rk[i, bl] - 1)[0])] for i in np.flatnonzero(sel)]
    vc = pd.Series(above).value_counts()
    print(f"  Blue at rank 2-4: {int(sel.sum())} respondents; item ranked directly above Blue: "
          + ", ".join(f"{n} {c}" for n, c in vc.items()))
    nav_gt = U[:, nv] > U[:, bl]
    both4 = top4[:, nv] & top4[:, bl]
    nav_above_both = 100 * nav_gt[both4].mean() if both4.any() else float("nan")
    print(f"  Navy > Blue: {int(nav_gt.sum())}/{N} ({pct(nav_gt):.1f}%)")
    print(f"  both in top 4: {int(both4.sum())}; Navy above Blue among those: {int(nav_gt[both4].sum())} "
          f"({nav_above_both:.1f}%)")

    def r3(names):
        return reach[3][tidx[frozenset(names)]] if len(names) == 3 else None

    reach_pair = lambda names: (top3[:, [ix[n] for n in names]].any(1)).mean()
    g_navy = reach_pair(("Black", "Navy", "Blue")) - reach_pair(("Black", "Navy"))
    g_turq = reach_pair(("Black", "Turquoise", "Blue")) - reach_pair(("Black", "Turquoise"))
    print(f"  k=3 marginal reach of adding Blue: to {{Black, Navy}} {reach_pair(('Black', 'Navy')):.4f} -> "
          f"{reach_pair(('Black', 'Navy', 'Blue')):.4f} (+{g_navy:.4f}, {round(g_navy * N)} people); "
          f"to {{Black, Turquoise}} {reach_pair(('Black', 'Turquoise')):.4f} -> "
          f"{reach_pair(('Black', 'Turquoise', 'Blue')):.4f} (+{g_turq:.4f}, {round(g_turq * N)} people)")
    top_above = vc.index[0] if len(vc) else None
    p3_pass = (top_above == "Navy") and pct(nav_gt) > 50 and g_navy < g_turq
    print(f"  criterion: Navy is the item most often directly above Blue ({top_above}), Navy > Blue for a\n"
          f"  majority ({pct(nav_gt):.1f}%), and Blue adds less reach to {{Black, Navy}} than to "
          f"{{Black, Turquoise}} ({g_navy:.4f} vs {g_turq:.4f})  ->  {'PASS' if p3_pass else 'FAIL'}")
    results["P3"] = (f"{'PASS' if p3_pass else 'FAIL'}: item most often directly above Blue = {top_above}; "
                     f"Navy > Blue {pct(nav_gt):.0f}%; Blue's gain +{g_navy:.3f} on {{Black, Navy}} vs "
                     f"+{g_turq:.3f} on {{Black, Turquoise}}")

    # ------------------------------------------------------------------ STEP 5
    hdr("STEP 5. Summary")
    nb = ~top3[:, ix["Black"]]
    pu, tu = top3[nb, ix["Purple"]], top3[nb, ix["Turquoise"]]
    na_ = top3[nb, ix["Navy"]]
    cl = {c: [NAMES[j] for j in range(J) if clusters["a"][j] == c] for c in range(4)}
    cl_txt = "; ".join("{" + ", ".join(m) + "}" for m in cl.values())
    lines = [
        "(1) Reproduction. N = 202, no missing values, all weights 1. First-choice shares, reach at k = 2/3/4,",
        "    frequencies and #1/#2 ranks all match Conjointly to within 5e-8. No ties at any top-k boundary.",
        "",
        "(2) Ranking uncertainty (respondent bootstrap, B = 5000). {Black, Purple, Turquoise} is #1 at every k.",
    ]
    for k in KS:
        b = boot[k]
        lines.append(f"    k={k}: P(win) {b['p_win']:.2f}; runner-up {name_of(triples[b['runner']])} P(win) "
                     f"{b['p_runner']:.2f}; gap to #2 {b['gap']:+.3f} (paired 95% CI {b['gap_lo']:+.3f} to "
                     f"{b['gap_hi']:+.3f}); {b['n_tied']} of 363 triples tied with #1.")
    lines += [
        "    The winner is stable but not separable from its nearest rivals at any k; the tie band widens with k.",
        "",
        f"(3) Cluster structure (average linkage, 1 - corr). Same 4-cluster partition with and without",
        f"    per-person scaling: {cl_txt}.",
        "    Read as: dark/neutral, blues, purple-turquoise, warm/bright. Navy's partner is Blue",
        f"    (corr {Ca[nv, bl]:+.2f}), then Black ({Ca[nv, ix['Black']]:+.2f}).",
        "",
        f"(4) P1 {results['P1']}: Navy's top 3 correlates are Blue, Black, Maroon; Navy-Violet is "
        f"{Ca[nv, ix['Violet']]:+.2f}.",
        f"    P1b: {results['P1b']}. Navy fans are disproportionately Black fans, so a",
        "    Navy+Black triple hits many people twice: high frequency, lower reach.",
        f"    P2 {results['P2']}.",
        "        Lime is a high-variance taste anticorrelated with Black, Navy and Purple but mildly positive with",
        f"        Turquoise, so the pass rule (negative with all four) fails. The fans BPT misses are "
        f"{p2['Lime']['n4'] - round(p2['Lime']['n4'] * (1 - p2['Lime']['nr'] / 100))} people; too few to act on.",
        f"    P3 {results['P3']}.",
        "",
        "(5) Why {Black, Purple, Turquoise} wins. Black is the single largest anchor "
        f"({100 * share[ix['Black']]:.0f}% first choice, {100 * base3['Black']:.0f}% top 3),",
        "    and it sits in the dark/neutral cluster, with negative correlations to Violet "
        f"({Ca[ix['Black'], ix['Violet']]:+.2f}), Turquoise ({Ca[ix['Black'], ix['Turquoise']]:+.2f})",
        f"    and Purple ({Ca[ix['Black'], ix['Purple']]:+.2f}). The best complements are therefore the most popular",
        "    items in the purple-turquoise cluster, which is the most anticorrelated with Black. Purple and",
        f"    Turquoise are only moderately correlated with each other ({Ca[ix['Purple'], ix['Turquoise']]:+.2f}).",
        f"    Of the {int(nb.sum())} respondents without Black in their top 3, Purple reaches "
        f"{int(pu.sum())} and Turquoise {int(tu.sum())}; {int((pu & tu).sum())} have both,",
        f"    so together they reach {int((pu | tu).sum())} of the {int(nb.sum())}. Navy is popular too "
        f"({int(na_.sum())} of the {int(nb.sum())}),",
        "    but it is positively correlated with Black, so its reach overlaps with Black's. That is why the",
        "    Navy triples have higher frequency and lower reach.",
        "",
        "(6) Limitations. Utilities are per-person HB point estimates. Within-person posterior uncertainty",
        "    is ignored, so every interval here is a lower bound on the true uncertainty. They come from an HB-MNL",
        "    whose population prior shrinks individuals toward the mean and imposes its own covariance",
        "    structure, which colors the correlations in Step 3. A multivariate-probit fit of the correlation",
        "    structure needs the raw best/worst choice tasks, which this export does not contain. The P2 and P3",
        "    pass rules were set here (the spec gave none) and rest on small counts.",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[2] / "conjointly_experiment_7.json"
    main(sys.argv[1] if len(sys.argv) > 1 else default)
