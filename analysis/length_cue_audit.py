"""Is the literal option selected because it is the longest option?

The cue-free subset of the construction audit removes the lexical and the
Latin-script cue but not length, so length has to be measured separately: how
often the literal option is the longest of the four actually presented, what LAR
is on each side of that split, and whether the length gap persists inside the
cue-free subset.
"""
import re, json, unicodedata, random
import numpy as np, pandas as pd

PUNCT = "।,;:!?\"'()[]{}—–-"
def words(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return [w for w in t.split() if w]
def norm(s): return " ".join(words(s))
PARENX = re.compile(r"\([^)]*\)")
def has_paren(s):
    for x in re.findall(r"\(([^)]*)\)", str(s)):
        c = x.strip().replace(" ", "")
        if c and sum(ch.isascii() and ch.isalpha() for ch in c) >= 3 and \
           sum(ch.isascii() and ch.isalpha() for ch in c)/len(c) >= 0.6:
            return True
    return False

dd = pd.read_json("JEV_RUN/data/splits/full.jsonl", lines=True)
dd["nw"]      = dd.idiom.map(lambda s: len(words(s)))
dd["litbn"]   = dd.literal_meaning.map(lambda s: norm(PARENX.sub(" ", str(s))))
dd["idin"]    = dd.idiom.map(norm)
dd["contains"]= [r.idin in r.litbn for r in dd.itertuples()]
dd["paren"]   = dd.literal_meaning.map(has_paren)
MEAN = dict(zip(dd.id, dd.figurative_meaning_bn))
LIT  = dict(zip(dd.id, dd.literal_meaning))
ALL  = list(dd.id)
K = 4

def _fill(pool, need, exclude, rng):
    pool = [i for i in pool if i not in exclude]; rng.shuffle(pool)
    out = pool[:need]
    while len(out) < need:
        c = rng.choice(ALL)
        if c not in exclude and c not in out: out.append(c)
    return out

# rebuild each E1 D-lit option list under the run's own seed namespace (1000)
rows = []
for r in dd.itertuples():
    rng = random.Random(1000 + int(r.id))
    pool = _fill(list(ALL), K - 2, {r.id}, rng)
    opts = [MEAN[r.id], LIT[r.id]] + [MEAN[i] for i in pool]
    opts = [o for o in opts if str(o).strip()]
    opts = list(dict.fromkeys(opts))[:K]
    while len(opts) < K:
        c = MEAN[rng.choice(ALL)]
        if c and c not in opts: opts.append(c)
    L = [len(str(o)) for o in opts]
    li = opts.index(LIT[r.id]) if LIT[r.id] in opts else None
    if li is None: continue
    rows.append(dict(id=int(r.id), lit_len=L[li], max_len=max(L),
                     longest=L[li] == max(L), mean_other=np.mean([x for j,x in enumerate(L) if j!=li])))
opt = pd.DataFrame(rows).set_index("id")
print(f"items with a reconstructible literal option : {len(opt)}")
print(f"literal option is the LONGEST of the four   : {opt.longest.mean():.3f}")
print(f"mean literal length {opt.lit_len.mean():.1f} vs mean other option {opt.mean_other.mean():.1f}")

info = dd.set_index("id")[["nw","contains","paren"]]
clean = set(dd[(dd.nw>=2) & (~dd.contains) & (~dd.paren)].id)
print(f"\ncue-free subset ({len(clean)} items):")
c = opt[opt.index.isin(clean)]
print(f"  literal option is the longest of the four : {c.longest.mean():.3f}")
print(f"  mean literal length {c.lit_len.mean():.1f} vs mean other option {c.mean_other.mean():.1f}")

res = {"longest_all": float(opt.longest.mean()), "longest_clean": float(c.longest.mean()),
       "lit_len_clean": float(c.lit_len.mean()), "other_len_clean": float(c.mean_other.mean()),
       "lit_len_all": float(opt.lit_len.mean()), "other_len_all": float(opt.mean_other.mean())}

ARMS = {"laya-ml":"LAYA_RUN/results/laya-ml/e1_ladder/raw.csv",
        "lod-lille-0.6b":"LOD_RUN/results/lod-lille-0.6b/e1_ladder/raw.csv"}
rng0 = np.random.default_rng(0)
for lab, keep in (("all items", None), ("cue-free subset", clean)):
    print(f"\nBangla LAR under D-LIT by whether the literal option is longest, {lab}:")
    for arm, path in ARMS.items():
        d = pd.read_csv(path); d = d[d.family=="D-lit"].copy()
        d = d.join(opt[["longest"]], on="id"); d = d[d.longest.notna()]
        if keep is not None: d = d[d.id.isin(keep)]
        a = d[~d.longest.astype(bool)].chose_literal.to_numpy(float)
        b = d[d.longest.astype(bool)].chose_literal.to_numpy(float)
        if len(a) < 5 or len(b) < 5:
            print(f"  {arm:16s} too few items in one cell (n={len(a)}/{len(b)})"); continue
        bs = np.array([rng0.choice(b, b.size).mean() - rng0.choice(a, a.size).mean()
                       for _ in range(2000)])
        ci = np.percentile(bs, [2.5, 97.5])
        print(f"  {arm:16s} not longest {a.mean():.3f} (n={a.size})  longest {b.mean():.3f} "
              f"(n={b.size})  delta {b.mean()-a.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
        res[f"{arm}_{'all' if keep is None else 'clean'}"] = dict(
            not_longest=float(a.mean()), n_not=int(a.size), longest=float(b.mean()),
            n_long=int(b.size), delta=float(b.mean()-a.mean()), ci=[float(ci[0]),float(ci[1])])
json.dump(res, open("analysis/length_cue_audit.json","w"), indent=2)
