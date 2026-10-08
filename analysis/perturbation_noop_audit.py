"""How often do the E3 and E6 perturbations change the string at all?

A word shuffle on a one-word idiom is a no-op; on a two-word idiom it either
does nothing or equals the reversal. A six-word truncation does nothing to an
option already at or below six words. Both bound how much E3 and E6 could have
shown, so we report them rather than leaving the reader to infer them.
"""
import re, json, unicodedata
import numpy as np, pandas as pd

PUNCT = "।,;:!?\"'()[]{}—–-"
def words(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return [w for w in t.split() if w]

res = {}
print("E3: share of items where the perturbed state differs from the intact one")
for arm, path in (("laya-ml","LAYA_RUN/results/laya-ml/e3_unit/raw.csv"),
                  ("lod-lille-0.6b","LOD_RUN/results/lod-lille-0.6b/e3_unit/raw.csv")):
    d = pd.read_csv(path)
    base = d[d.variant=="V0_intact"].set_index("id").text
    d["nw"] = d.id.map(base.map(lambda s: len(words(s))))
    out = {}
    for v in ("V1_shuffle","V2_reverse","V3_delete1","V4_substitute1","V5_in_sentence"):
        g = d[d.variant==v].set_index("id").text
        common = base.index.intersection(g.index)
        changed = (base.loc[common].values != g.loc[common].values).mean()
        out[v] = float(changed)
    print(f"  {arm:16s} " + "  ".join(f"{k.split('_')[1]} {v:.3f}" for k,v in out.items()))
    res[arm+"_e3_changed"] = out
    nw = base.map(lambda s: len(words(s)))
    print(f"{'':18s} single-word items in E3: {(nw==1).mean():.3f} (n={(nw==1).sum()}/{len(nw)})")
    res[arm+"_e3_oneword"] = float((nw==1).mean())

    # shuffle==reverse collision, and accuracy restricted to items the shuffle moved
    g1 = d[d.variant=="V1_shuffle"].set_index("id"); g2 = d[d.variant=="V2_reverse"].set_index("id")
    common = g1.index.intersection(g2.index)
    coll = (g1.loc[common].text.values == g2.loc[common].text.values).mean()
    print(f"{'':18s} shuffle and reversal give the same string on {coll:.3f} of items")
    res[arm+"_e3_collision"] = float(coll)

    mw = set(nw[nw>=3].index)
    b = d[(d.variant=="V0_intact") & (d.id.isin(mw))].correct.mean()
    s = d[(d.variant=="V1_shuffle") & (d.id.isin(mw))].correct.mean()
    print(f"{'':18s} 3+ word items only (n={len(mw)}): intact {b:.3f} -> shuffle {s:.3f} ({s-b:+.3f})")
    res[arm+"_e3_multiword"] = dict(n=len(mw), intact=float(b), shuffle=float(s))

print("\nE6: how often six-word truncation changes an option at all")
dd = pd.read_json("JEV_RUN/data/splits/full.jsonl", lines=True)
for lab, col in (("Bangla gold gloss","figurative_meaning_bn"),
                 ("English gold gloss","figurative_meaning_en")):
    nw = dd[col].map(lambda s: len(words(s)))
    print(f"  {lab:20s} median {nw.median():.0f} words; "
          f"{(nw>6).mean():.3f} exceed six, so truncation is a no-op on "
          f"{1-(nw>6).mean():.3f} of options")
    res["trunc_"+col] = float((nw>6).mean())
json.dump(res, open("analysis/perturbation_noop_audit.json","w"), indent=2)
