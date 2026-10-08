"""AURC and AUGRC for every arm, plus Jev's Bangla calibration.

\\citet{traub2024selective} argue that AURC mis-ranks selective classifiers
because the risk it integrates is conditioned on the accepted set, so a system
that accepts almost nothing scores well. Their AUGRC integrates the generalized
risk, the probability that an item is both accepted and wrong, which is not
conditioned that way. We report both.
"""
import numpy as np, pandas as pd, json

def curves(conf, correct):
    o = np.argsort(-np.asarray(conf, float))
    err = 1.0 - np.asarray(correct, float)[o]
    n = len(err)
    k = np.arange(1, n + 1)
    cum = np.cumsum(err)
    aurc = float(np.mean(cum / k))          # risk | accepted
    augrc = float(np.mean(cum / n))         # risk of (accepted AND wrong)
    return aurc, augrc

def rand_baseline(correct, seed=0, reps=200):
    y = np.asarray(correct, float); rng = np.random.default_rng(seed)
    a = [curves(rng.random(len(y)), y) for _ in range(reps)]
    return float(np.mean([x[0] for x in a])), float(np.mean([x[1] for x in a]))

rows = []
SRC = [
 ("laya-ml","D-lit","LAYA_RUN/results/laya-ml/e1_ladder/raw.csv","conf","family"),
 ("laya-ml","D-rand","LAYA_RUN/results/laya-ml/e1_ladder/raw.csv","conf","family"),
 ("lod-lille-0.6b","D-lit","LOD_RUN/results/lod-lille-0.6b/e1_ladder/raw.csv","max_prob","family"),
 ("lod-lille-0.6b","D-rand","LOD_RUN/results/lod-lille-0.6b/e1_ladder/raw.csv","max_prob","family"),
 ("jev-1.13","D-lit","JEV_RUN/results/jev-1.13/j2_literal_en/raw.csv","max_prob","family"),
 ("jev-1.13","D-rand","JEV_RUN/results/jev-1.13/j2_literal_en/raw.csv","max_prob","family"),
]
for arm, fam, path, cc, fcol in SRC:
    d = pd.read_csv(path); d = d[d[fcol] == fam]
    a, g = curves(d[cc], d.correct)
    ra, rg = rand_baseline(d.correct)
    rows.append(dict(arm=arm, family=fam, lang="bn" if arm!="jev-1.13" else "en",
                     n=len(d), acc=float(d.correct.mean()),
                     aurc=a, aurc_rand=ra, augrc=g, augrc_rand=rg))
# Jev in Bangla, from the matrix
for lg in ("bn","en"):
    d = pd.read_csv(f"added/results/jev-1.13/matrix_{lg}/raw.csv")
    a, g = curves(d.max_prob, d.correct); ra, rg = rand_baseline(d.correct)
    rows.append(dict(arm="jev-1.13", family="D-lit(matrix)", lang=lg, n=len(d),
                     acc=float(d.correct.mean()), aurc=a, aurc_rand=ra,
                     augrc=g, augrc_rand=rg))
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nratios (random / confidence-gated), AURC then AUGRC:")
for r in rows:
    print(f"  {r['arm']:16s} {r['family']:14s} {r['lang']}  "
          f"AURC {r['aurc']:.3f} vs rand {r['aurc_rand']:.3f} ({r['aurc_rand']/r['aurc']:.2f}x)   "
          f"AUGRC {r['augrc']:.3f} vs rand {r['augrc_rand']:.3f} ({r['augrc_rand']/r['augrc']:.2f}x)")
t.to_csv("analysis/tables/selective_augrc.csv", index=False)
json.dump(rows, open("analysis/selective_augrc.json","w"), indent=2)
