"""Test the 'pick the option containing Latin script' rival explanation.

The Bangla literal option is the corpus's `literal_meaning` field as written, and
for some idioms that field carries an English parenthetical. A reviewer can argue
that English-centric scorers doing bag-of-words matching select that option
because it is the only one containing Latin script, which would explain Bangla
capture without any literal-versus-figurative reading.

This needs no new experiments. The capture matrix is restricted to the 3,265
items whose literal gloss HAS a parenthetical, because the English cell needs
one. But the Bangla E1 run covers all 8,876 items, and on 5,485 of them the
literal gloss has NO parenthetical and therefore offers no Latin-script cue.
Comparing LAR across those two strata tests the explanation directly.
"""
import re, json
import numpy as np, pandas as pd

PAREN = re.compile(r"\(([^)]*)\)")
LATIN = re.compile(r"[A-Za-z]")

def eng_literal(s):
    out = []
    for x in PAREN.findall(str(s)):
        x = x.strip(); core = x.replace(" ", "")
        if not core: continue
        asc = sum(c.isascii() and c.isalpha() for c in core)
        if asc >= 3 and asc / len(core) >= 0.6:
            out.append(x)
    return " / ".join(out)

dd = pd.read_json("JEV_RUN/data/splits/full.jsonl", lines=True)
dd["lit_en"] = dd.literal_meaning.map(eng_literal)
dd["has_paren"] = dd.lit_en.str.len() > 0

# How distinctive is the cue? Do the distractor glosses also carry Latin script?
fig_latin = dd.figurative_meaning_bn.map(lambda s: bool(LATIN.search(str(s))))
print(f"items                                        : {len(dd)}")
print(f"  literal gloss carries an English parenthetical: {dd.has_paren.sum()} "
      f"({dd.has_paren.mean():.3f})")
print(f"  Bangla figurative gloss contains any Latin    : {fig_latin.sum()} "
      f"({fig_latin.mean():.3f})")
print("  -> when the literal gloss has a parenthetical, it is usually the only")
print("     option in the list containing Latin script, so the cue is real.\n")

# length, the other half of the cue
print(f"mean length, literal gloss WITH paren : "
      f"{dd[dd.has_paren].literal_meaning.str.len().mean():.1f} chars")
print(f"mean length, literal gloss WITHOUT    : "
      f"{dd[~dd.has_paren].literal_meaning.str.len().mean():.1f} chars")
print(f"mean length, Bangla figurative gloss  : "
      f"{dd.figurative_meaning_bn.str.len().mean():.1f} chars\n")

paren = dict(zip(dd.id, dd.has_paren))

ARMS = {
    "laya-ml":        "LAYA_RUN/results/laya-ml/e1_ladder/raw.csv",
    "lod-lille-0.6b": "LOD_RUN/results/lod-lille-0.6b/e1_ladder/raw.csv",
}
res = {}
print("LAR on the Bangla D-LIT condition, stratified by whether the literal")
print("gloss carries an English parenthetical (the Latin-script cue):")
for arm, path in ARMS.items():
    d = pd.read_csv(path)
    d = d[d.family == "D-lit"].copy()
    d["paren"] = d.id.map(paren)
    d = d[d.paren.notna()]
    g = d.groupby("paren").chose_literal.agg(["mean", "size"])
    lo, hi = g.loc[False, "mean"], g.loc[True, "mean"]
    # bootstrap CI on the difference
    a = d[~d.paren.astype(bool)].chose_literal.to_numpy(float)
    b = d[d.paren.astype(bool)].chose_literal.to_numpy(float)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(b, b.size).mean() - rng.choice(a, a.size).mean()
                   for _ in range(2000)])
    ci = np.percentile(bs, [2.5, 97.5])
    res[arm] = dict(lar_no_paren=float(lo), n_no_paren=int(g.loc[False, "size"]),
                    lar_paren=float(hi), n_paren=int(g.loc[True, "size"]),
                    delta=float(hi - lo), ci=[float(ci[0]), float(ci[1])])
    print(f"  {arm:16s} no parenthetical {lo:.3f} (n={int(g.loc[False,'size'])})  "
          f"parenthetical {hi:.3f} (n={int(g.loc[True,'size'])})  "
          f"delta {hi-lo:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")

json.dump(res, open("analysis/parenthetical_cue_audit.json", "w"), indent=2)

# ── Specificity: the cue should move D-LIT and nothing else ───────────────
# If the parenthetical stratum simply contained easier or harder idioms, the
# split would also move accuracy on families that carry no literal option.
print("\nSpecificity check, accuracy by stratum on families with no literal option:")
for arm, path in ARMS.items():
    d = pd.read_csv(path)
    d["paren"] = d.id.map(paren)
    d = d[d.paren.notna()]
    row = []
    for fam in ("D-rand", "D-surf", "D-sem"):
        g = d[d.family == fam].groupby("paren").correct.mean()
        row.append(f"{fam} {g.loc[True]-g.loc[False]:+.3f}")
    print(f"  {arm:16s} " + "   ".join(row))

# ── How much of the Bangla-English gap could the cue explain? ─────────────
print("\nBound on what the cue can explain (Laya-ml / Lod-lille):")
for arm in ARMS:
    r = res[arm]
    print(f"  {arm:16s} removing the cue entirely takes Bangla LAR to "
          f"{r['lar_no_paren']:.3f}")
