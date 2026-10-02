"""Provenance and symmetry audit of the Bangla/English literal glosses.

The English literal gloss is NOT a machine translation. It is the
annotator-written English parenthetical already present inside the corpus's
Bangla `literal_meaning` field, recovered by the same regex the runs used. This
script documents that, and measures three asymmetries between the two language
cells of the capture matrix that the paper must declare:

  A. the Bangla literal option retains the English parenthetical, so it is a
     bilingual string, while the English literal option is the parenthetical alone;
  B. the English literal gloss was filtered to be lexically distinct from the
     English gold gloss (Jaccard < 0.34) while the Bangla side was not;
  C. the state differs in kind: Bangla uses the idiom itself, English uses the
     annotator's nearest English idiom, not a translation of the Bangla one.
"""
import re, json, unicodedata
import numpy as np, pandas as pd

PAREN = re.compile(r"\(([^)]*)\)")
def eng_literal(s):
    out = []
    for x in PAREN.findall(str(s)):
        x = x.strip(); core = x.replace(" ", "")
        if not core: continue
        asc = sum(c.isascii() and c.isalpha() for c in core)
        if asc >= 3 and asc / len(core) >= 0.6:
            out.append(x)
    return " / ".join(out)

STOP = set("the and for one who that with from not are was his her its into out off whom".split())
def _tok(s): return set(re.findall(r"[a-z]{3,}", str(s).lower())) - STOP
def jac(a, b):
    A, B = _tok(a), _tok(b)
    return len(A & B) / max(1, len(A | B))

PUNCT = "।,;:!?\"'()[]{}—–-"
def words(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return {w for w in t.split() if len(w) > 1}

dd = pd.read_json("JEV_RUN/data/splits/full.jsonl", lines=True)
dd["sim_en"] = dd.similar_in_english.map(
    lambda v: (v if isinstance(v, list) else [])[:1] or [""]).str[0]
dd["lit_en"] = dd.literal_meaning.map(eng_literal)
dd["has_lit_en"] = dd.lit_en.str.len() > 0
dd["has_sim_en"] = dd.sim_en.str.len() > 0
dd["lit_fig_overlap"] = [jac(r.lit_en, r.figurative_meaning_en) if r.has_lit_en
                         else np.nan for r in dd.itertuples()]
dd["usable"] = dd.has_lit_en & dd.has_sim_en & (dd.lit_fig_overlap < 0.34)

n = len(dd)
print(f"corpus items                              : {n}")
print(f"  carry an English parenthetical gloss     : {dd.has_lit_en.sum()} "
      f"({dd.has_lit_en.mean():.3f})")
print(f"  also carry an English equivalent idiom   : {(dd.has_lit_en & dd.has_sim_en).sum()}")
print(f"  survive the gold/literal distinctness cut: {dd.usable.sum()}")

P = dd[dd.usable]
res = {"n_corpus": int(n), "has_lit_en": int(dd.has_lit_en.sum()),
       "n_pool": int(len(P))}

# A. bilinguality and length of the literal option as actually presented
res["bn_lit_contains_english"] = float(P.literal_meaning.map(
    lambda s: bool(eng_literal(s))).mean())
res["bn_lit_chars"] = float(P.literal_meaning.str.len().mean())
res["en_lit_chars"] = float(P.lit_en.str.len().mean())
res["bn_gold_chars"] = float(P.figurative_meaning_bn.str.len().mean())
res["en_gold_chars"] = float(P.lit_en.str.len().mean())
print(f"\nA. Bangla literal option contains an English parenthetical: "
      f"{res['bn_lit_contains_english']:.3f} of pool items")
print(f"   mean length, Bangla literal option: {res['bn_lit_chars']:.0f} chars; "
      f"English literal option: {res['en_lit_chars']:.0f} chars")

# B. gold/literal lexical distance, both sides
bn_gl = np.array([len(words(r.figurative_meaning_bn) & words(r.literal_meaning)) /
                  max(1, len(words(r.figurative_meaning_bn) | words(r.literal_meaning)))
                  for r in P.itertuples()])
en_gl = np.array([jac(r.figurative_meaning_en, r.lit_en) for r in P.itertuples()])
res["bn_gold_lit_jaccard"] = float(bn_gl.mean())
res["en_gold_lit_jaccard"] = float(en_gl.mean())
print(f"\nB. gold-vs-literal Jaccard, Bangla {bn_gl.mean():.3f} | "
      f"English {en_gl.mean():.3f}  (English capped at 0.34 by construction)")

# C. state-vs-literal overlap: how far the literal gloss echoes the state string
bn_sl = np.array([(len(words(r.idiom) & words(r.literal_meaning)) > 0)
                  for r in P.itertuples()], float)
en_sl = np.array([(len(_tok(r.sim_en) & _tok(r.lit_en)) > 0)
                  for r in P.itertuples()], float)
res["bn_state_lit_share"] = float(bn_sl.mean())
res["en_state_lit_share"] = float(en_sl.mean())
print(f"\nC. literal gloss shares a content word with the state: "
      f"Bangla {bn_sl.mean():.3f} | English {en_sl.mean():.3f}")
print(f"   (Bangla state = the idiom; English state = the nearest English idiom)")

json.dump(res, open("analysis/e4_gloss_audit.json", "w"), indent=2)


# ── D. does the surface hook actually drive Bangla capture? ────────────────
# Stratify the Bangla cell of the capture matrix by whether the literal gloss
# shares a content word with the idiom. If LAR is comparable in both strata the
# lexical hook is not what produces the capture, which bounds how much of the
# Bangla-English gap asymmetry C could explain.
print("\nD. Bangla LAR stratified by state/literal word overlap")
hook = {int(r.id): bool(words(r.idiom) & words(r.literal_meaning))
        for r in P.itertuples()}
strat = {}
for arm, path in (("laya-ml", "added/results/laya-ml/matrix_bn/raw.csv"),
                  ("lod-lille-0.6b", "added/results/lod-lille-0.6b/matrix_bn/raw.csv"),
                  ("jev-1.13", "added/results/jev-1.13/matrix_bn/raw.csv")):
    d = pd.read_csv(path)
    d["hook"] = d.id.map(hook)
    d = d[d.hook.notna()]
    g = d.groupby("hook").chose_literal.agg(["mean", "size"])
    lo = g.loc[False, "mean"]; hi = g.loc[True, "mean"]
    strat[arm] = dict(lar_no_overlap=float(lo), n_no_overlap=int(g.loc[False, "size"]),
                      lar_overlap=float(hi), n_overlap=int(g.loc[True, "size"]),
                      delta=float(hi - lo), lar_all=float(d.chose_literal.mean()))
    print(f"  {arm:16s} no overlap {lo:.3f} (n={g.loc[False,'size']})  "
          f"overlap {hi:.3f} (n={g.loc[True,'size']})  delta {hi-lo:+.3f}")
res["bn_lar_stratified"] = strat
json.dump(res, open("analysis/e4_gloss_audit.json", "w"), indent=2)


# ── E. the matrix gap restricted to the no-hook stratum ───────────────────
# The strongest available answer to the asymmetry: if the Bangla-English gap
# survives on items whose Bangla literal gloss shares NO content word with the
# idiom, the surface hook cannot be what produces it.
print("\nE. capture gap restricted to items with no state/literal word overlap")
res["gap_no_hook"] = {}
for arm in ("laya-ml", "lod-lille-0.6b", "jev-1.13"):
    out = {}
    for lang in ("bn", "en"):
        d = pd.read_csv(f"added/results/{arm}/matrix_{lang}/raw.csv")
        d["hook"] = d.id.map(hook)
        out[lang] = float(d[d.hook == False].chose_literal.mean())
    out["gap"] = out["bn"] - out["en"]
    out["n"] = int((pd.Series(list(hook.values())) == False).sum())
    res["gap_no_hook"][arm] = out
    print(f"  {arm:16s} bn {out['bn']:.3f}  en {out['en']:.3f}  "
          f"gap {out['gap']:+.3f}  (n={out['n']})")
json.dump(res, open("analysis/e4_gloss_audit.json", "w"), indent=2)
