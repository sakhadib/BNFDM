"""Binning-free and equal-mass calibration metrics from the recorded outputs.

ECE with equal-width bins is sensitive to the binning choice, which is the
criticism \citet{nixon2019measuring} make of it. This script adds two metrics
computed from the same per-item probability vectors already released:

  Brier  the multiclass Brier score, mean_i sum_k (p_ik - y_ik)^2, a strictly
         proper score with no binning and no free parameter. Range [0, 2]; the
         value for a uniform 4-way distribution is 0.750.
  ACE    adaptive calibration error on the top label: |confidence - accuracy|
         averaged over 15 EQUAL-MASS bins rather than equal-width ones, so no
         bin is empty and no bin dominates.

No new model calls are made. Everything here is a re-reading of released files.
"""
import numpy as np, pandas as pd, json

P = ["p_A", "p_B", "p_C", "p_D"]
ARMS = {
    "laya-ml": ("LAYA_RUN/results/laya-ml/e1_ladder/raw.csv", "conf"),
    "lod-lille-0.6b": ("LOD_RUN/results/lod-lille-0.6b/e1_ladder/raw.csv", "max_prob"),
    "jev-1.13": ("JEV_RUN/results/jev-1.13/j2_literal_en/raw.csv", "max_prob"),
}

def brier(d):
    Pm = d[P].to_numpy(float)
    Pm = Pm / np.clip(Pm.sum(1, keepdims=True), 1e-12, None)
    Y = np.zeros_like(Pm)
    for j, k in enumerate(P):
        Y[:, j] = (d.gold_key.values == k[-1]).astype(float)
    return float(((Pm - Y) ** 2).sum(1).mean())

def ece(conf, corr, nb=15):          # equal-width, for reference
    e, n = 0.0, len(conf)
    edges = np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(conf[m].mean() - corr[m].mean())
    return float(e)

def ace(conf, corr, nb=15):          # equal-mass
    o = np.argsort(conf); conf, corr = conf[o], corr[o]
    parts = np.array_split(np.arange(len(conf)), nb)
    return float(np.mean([abs(conf[ix].mean() - corr[ix].mean())
                          for ix in parts if len(ix)]))

rows = []
for arm, (path, cc) in ARMS.items():
    d = pd.read_csv(path)
    for fam in ("D-rand", "D-lit", "D-hard"):
        g = d[d.family == fam]
        if not len(g):
            continue
        conf = g[cc].to_numpy(float); corr = g.correct.to_numpy(float)
        rows.append(dict(arm=arm, family=fam, n=len(g),
                         acc=float(corr.mean()), conf=float(conf.mean()),
                         brier=brier(g), ece=ece(conf, corr), ace=ace(conf, corr)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nreference: Brier of the uniform 4-way distribution = 0.750")
t.to_csv("analysis/tables/brier_ace.csv", index=False)
json.dump(rows, open("analysis/brier_ace.json", "w"), indent=2)
