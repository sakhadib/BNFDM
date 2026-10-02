"""E7 permutation audit.

For each item evaluated under all 24 orderings of a fixed 4-option set, map each
slot's returned probability back to the *original* option identity using the
recorded permutation, then measure how much a single option's own score moves
across the 24 orderings. A scorer that is order-independent by construction
should show spreads at the level of floating-point noise.
"""
import numpy as np, pandas as pd, sys, itertools, json

ARMS = {
    "Laya-ml":        "LAYA_RUN/results/laya-ml/e7_order/raw.csv",
    "Lod-lille-0.6B": "LOD_RUN/results/lod-lille-0.6b/e7_order/raw.csv",
    "Jev-1.13":       "JEV_RUN/results/jev-1.13/j1_order/raw.csv",
}
PCOLS = ["p_A", "p_B", "p_C", "p_D"]
out = {}

for name, path in ARMS.items():
    df = pd.read_csv(path, dtype={"perm": str})
    df["perm"] = df["perm"].str.zfill(4)
    spreads, changed, n = [], 0, 0
    for rid, g in df.groupby("id"):
        if len(g) != 24:
            continue
        M = np.full((24, 4), np.nan)
        am = []
        for r, (_, row) in enumerate(g.iterrows()):
            perm = [int(c) for c in row["perm"]]
            probs = np.array([row[c] for c in PCOLS], float)
            # slot s displays original option perm[s]
            for s, orig in enumerate(perm):
                M[r, orig] = probs[s]
            am.append(perm[int(np.argmax(probs))])
        if np.isnan(M).any():
            continue
        n += 1
        spreads.append((M.max(0) - M.min(0)).max())
        if len(set(am)) > 1:
            changed += 1
    s = np.array(spreads)
    out[name] = dict(
        n=n, median=float(np.median(s)), mean=float(s.mean()),
        p95=float(np.percentile(s, 95)), max=float(s.max()),
        share_below_1e4=float((s < 1e-4).mean()),
        changed=changed, change_rate=changed / n,
    )
    print(f"{name}: items={n}")
    print(f"  max within-option spread across the 24 orderings:")
    print(f"    median {np.median(s):.3e} | mean {s.mean():.3e} | "
          f"p95 {np.percentile(s,95):.3e} | max {s.max():.3e}")
    print(f"  share with spread < 1e-4 (numerically order-invariant): "
          f"{(s<1e-4).mean():.3f}")
    print(f"  items whose argmax IDENTITY changes: {changed}/{n} = {changed/n:.3f}")

json.dump(out, open("analysis/e7_order_audit.json", "w"), indent=2)
