"""How much of literal capture could be string matching?

Three properties of the corpus make the literal option an easy lexical target:
some idioms are single words whose literal gloss repeats the idiom verbatim,
some multiword idioms have a literal gloss that contains the idiom string, and
the Bangla literal gloss is on average the longest option in the list. This
script measures all three and breaks LAR down by each.
"""
import re, json, unicodedata
import numpy as np, pandas as pd

PUNCT = "।,;:!?\"'()[]{}—–-"
def words(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return [w for w in t.split() if w]
def norm(s):
    return " ".join(words(s))

dd = pd.read_json("JEV_RUN/data/splits/full.jsonl", lines=True)
dd["n_words"]   = dd.idiom.map(lambda s: len(words(s)))
dd["lit_norm"]  = dd.literal_meaning.map(norm)
dd["idi_norm"]  = dd.idiom.map(norm)
# strip the English parenthetical before asking about Bangla-side identity
PAREN = re.compile(r"\([^)]*\)")
dd["lit_bn_only"] = dd.literal_meaning.map(lambda s: norm(PAREN.sub(" ", str(s))))
dd["identical"] = dd.lit_bn_only == dd.idi_norm
dd["contains"]  = [r.idi_norm in r.lit_bn_only for r in dd.itertuples()]

n = len(dd)
print(f"items                                   : {n}")
for k in (1, 2, 3):
    m = (dd.n_words == k).sum()
    print(f"  idioms of {k} word(s)                    : {m} ({m/n:.3f})")
print(f"  idioms of 4+ words                     : {(dd.n_words>=4).sum()}")
print(f"  literal gloss IDENTICAL to the idiom    : {dd.identical.sum()} ({dd.identical.mean():.3f})")
print(f"  literal gloss CONTAINS the idiom        : {dd.contains.sum()} ({dd.contains.mean():.3f})")
print(f"  of single-word items, identical         : "
      f"{dd[dd.n_words==1].identical.mean():.3f}  (n={(dd.n_words==1).sum()})")

# option length, as presented
dd["lit_chars"] = dd.literal_meaning.str.len()
dd["gold_chars"] = dd.figurative_meaning_bn.str.len()
print(f"\nmean option length, Bangla literal gloss : {dd.lit_chars.mean():.1f} chars")
print(f"mean option length, Bangla gold gloss    : {dd.gold_chars.mean():.1f} chars")
longest = []
for r in dd.itertuples():
    longest.append(r.lit_chars > r.gold_chars)
print(f"literal option longer than the gold      : {np.mean(longest):.3f} of items")

info = dd.set_index("id")[["n_words","identical","contains","lit_chars","gold_chars"]]
info.columns = ["nw","identical","contains","lit_chars","gold_chars"]

ARMS = {"laya-ml":        "LAYA_RUN/results/laya-ml/e1_ladder/raw.csv",
        "lod-lille-0.6b": "LOD_RUN/results/lod-lille-0.6b/e1_ladder/raw.csv"}
res = {}
print("\nBangla LAR under D-LIT, by idiom length:")
for arm, path in ARMS.items():
    d = pd.read_csv(path); d = d[d.family == "D-lit"].copy()
    d = d.join(info, on="id")
    row = {}
    for lab, mask in (("1 word", d.nw == 1), ("2 words", d.nw == 2),
                      ("3 words", d.nw == 3), ("4+ words", d.nw >= 4)):
        if mask.sum():
            row[lab] = (d[mask].chose_literal.mean(), int(mask.sum()))
    print(f"  {arm:16s} " + "  ".join(f"{k} {v[0]:.3f} (n={v[1]})" for k, v in row.items()))
    res[arm] = {k: v[0] for k, v in row.items()}

print("\nBangla LAR under D-LIT, by verbatim overlap:")
for arm, path in ARMS.items():
    d = pd.read_csv(path); d = d[d.family == "D-lit"].copy(); d = d.join(info, on="id")
    a = d[~d.contains].chose_literal; b = d[d.contains].chose_literal
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(b.values, b.size).mean() - rng.choice(a.values, a.size).mean()
                   for _ in range(2000)])
    ci = np.percentile(bs, [2.5, 97.5])
    print(f"  {arm:16s} idiom NOT in gloss {a.mean():.3f} (n={a.size})  "
          f"idiom IN gloss {b.mean():.3f} (n={b.size})  "
          f"delta {b.mean()-a.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    res[arm+"_overlap"] = dict(no=float(a.mean()), yes=float(b.mean()),
                               n_no=int(a.size), n_yes=int(b.size),
                               delta=float(b.mean()-a.mean()),
                               ci=[float(ci[0]), float(ci[1])])

# the strictest subset: multiword AND no verbatim overlap AND no parenthetical
PARENX = re.compile(r"\(([^)]*)\)")
def has_paren(s):
    for x in PARENX.findall(str(s)):
        c = x.strip().replace(" ", "")
        if c and sum(ch.isascii() and ch.isalpha() for ch in c) >= 3 and \
           sum(ch.isascii() and ch.isalpha() for ch in c)/len(c) >= 0.6:
            return True
    return False
dd["paren"] = dd.literal_meaning.map(has_paren)
clean = set(dd[(dd.n_words >= 2) & (~dd.contains) & (~dd.paren)].id)
print(f"\nStrict subset (multiword, no verbatim overlap, no parenthetical): {len(clean)} items")
for arm, path in ARMS.items():
    d = pd.read_csv(path); d = d[(d.family == "D-lit") & (d.id.isin(clean))]
    print(f"  {arm:16s} LAR {d.chose_literal.mean():.3f}  acc {d.correct.mean():.3f}  n={len(d)}")
    res[arm+"_strict"] = dict(lar=float(d.chose_literal.mean()),
                              acc=float(d.correct.mean()), n=int(len(d)))
res["n_clean"] = len(clean)
res["counts"] = dict(n=n, one_word=int((dd.n_words==1).sum()),
                     identical=int(dd.identical.sum()), contains=int(dd.contains.sum()))
json.dump(res, open("analysis/item_construction_audit.json","w"), indent=2)
