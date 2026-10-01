# BNFDM — Jev 1.13 arm (OpenRouter Decisions API)

Third arm. **CPU runtime is fine** — no model is loaded locally; everything goes over the OpenRouter Decisions API. Root `/content/drive/MyDrive/BNFDM`, results under `results/jev-1.13/`.

Jev is English-only, so this arm is built around an **English** version of the instrument. That is the point, not a workaround: it separates *literal capture* from *the Bangla representation gap*. If capture persists in English — where Lod already scores 0.864 — it is a property of figurative processing, not a symptom of low-resource failure.

## Interface

`POST https://openrouter.ai/api/alpha/decisions` — a custom Decisions API, **not** chat completions. The response shape is the one you already handle:

```json
{"type":"choice","choice":"B","confidence":0.75,
 "probabilities":{"A":0.0,"B":0.84,"C":0.16,"D":0.0}}
```

`confidence` is a **separate signal**, not the max probability (0.75 vs 0.84 in the docs' own example) — same situation as Lod's head. It is logged as `conf_head`; every cross-model comparison uses max-probability.

Pricing: $0.042/M input, output free, 0.18s P50, 32K context. The whole suite is ≈100k calls ≈ **$1.35**. The cell-by-cell estimates print live spend as they run.

## What this arm runs, and why

| Block | Items | Calls | Purpose |
|---|---|---|---|
| **J1** order audit | 300 × 24 | 7,200 | The only behavioural way to measure order-sensitivity in a closed API. Run first — independent of every data question. |
| **J2** literal capture, EN | 3,265 × 2 | 6,530 | The reason to include Jev. Needs the Laya/Lod backfill to be paired. |
| **J3** English ceiling | 8,761 × 3 | 26,283 | Three-way English comparison. |
| **J4** Bengali control | 8,761 | 8,761 | English-only model on Bengali script; should sit at chance, mirroring Laya's `bn_en` = 0.283. |
| **J5** option ablation, EN | 888 × 5 | 4,440 | Option-side attribution on the shared subset. |

E2 (surface invariance) is **skipped**: `alternative_idioms` exist only in Bangla, so there is no English pair. Said plainly rather than improvised.

## The 3,265 items, and a caveat that must go in the paper

`literal_meaning` is Bangla, but **38%** of entries carry an English parenthetical written by the annotators — `কাঠ (wood)`, `অগ্নিকন্যা (fire's daughter)`. Those are extracted as the English literal gloss. Items are kept only when an English idiom equivalent exists and the literal gloss is textually distinct from the figurative one (Jaccard < 0.34 on content words), which removes 91 semantically transparent idioms where choosing the literal reading would not be an error.

**That pool is not neutral.** On Laya's existing Bangla run, in-pool LAR is **0.747** against **0.583** out-of-pool, while D-rand accuracy is identical (+0.008). Annotators added parentheticals where the literal image was vivid, and vivid literal images are what attract the model. So every English LAR is compared against the Bangla LAR **restricted to the same 3,265 items** — never against the full-set 0.644. Cell 10 computes both.

---

## 01 — Root and API key

```python
# ── 01 ── paths + OpenRouter key from Colab secrets ─────────────────────────
from google.colab import drive, userdata
drive.mount('/content/drive')

import os, json, platform, datetime as dt
from pathlib import Path

ROOT  = Path("/content/drive/MyDrive/BNFDM")
MODEL = "jev-1.13"
RUN   = dt.datetime.now().strftime("%Y%m%d-%H%M")

for d in ["data/splits", "configs", "results"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

def R(*p):
    q = ROOT / "results" / MODEL / Path(*p); q.parent.mkdir(parents=True, exist_ok=True); return q
def D(*p):
    q = ROOT / "data" / Path(*p); q.parent.mkdir(parents=True, exist_ok=True); return q

API_KEY = userdata.get("OPENROUTER_API_KEY")
assert API_KEY and len(API_KEY) > 20, "OPENROUTER_API_KEY missing from Colab secrets"
print("key loaded:", API_KEY[:8] + "…" + API_KEY[-4:])

ENV = {"run": RUN, "model": MODEL, "python": platform.python_version(),
       "runtime": "cpu (API only)"}
print(json.dumps(ENV, indent=2)); print("root:", ROOT)
```

---

## 02 — Dependencies

```python
# ── 02 ── requests + kagglehub. Never -U a preinstalled Colab package. ──────
import sys, subprocess
subprocess.run(f"{sys.executable} -m pip -q install kagglehub requests",
               shell=True, check=False)
import requests, pandas as pd, numpy as np
print("requests", requests.__version__, "| pandas", pd.__version__)
```

---

## 03 — Dataset, English literal extraction, pools

```python
# ── 03 ── identical pipeline to the other arms, plus the English pools ──────
import os, re, json, unicodedata, numpy as np, pandas as pd, kagglehub

RAW = D("bagdhara_raw.jsonl")
if RAW.exists():
    df = pd.read_json(RAW, lines=True); print("cached:", df.shape)
else:
    p  = kagglehub.dataset_download("sakhadib/bangla-bagdhara")
    fs = os.listdir(p)
    jl, js = [f for f in fs if f.endswith(".jsonl")], [f for f in fs if f.endswith(".json")]
    df = pd.read_json(os.path.join(p, (jl or js)[0]), lines=bool(jl))
    df.to_json(RAW, orient="records", lines=True, force_ascii=False)
    print("downloaded:", df.shape)

def nfc(x):
    if isinstance(x, str):  return unicodedata.normalize("NFC", x).strip()
    if isinstance(x, list): return [nfc(v) for v in x]
    return x
for c in ["idiom","literal_meaning","figurative_meaning_bn","figurative_meaning_en",
          "note","example_sentences_in_bangla","example_sentences_in_english",
          "alternative_idioms","similar_in_english","tags","usage_domain","history"]:
    if c in df.columns: df[c] = df[c].map(nfc)

def lst(v): return v if isinstance(v, list) else ([] if pd.isna(v) else [v])
def one(v):
    L = lst(v); return L[0] if L else ""
df["ex_bn1"] = df["example_sentences_in_bangla"].map(one)
df["alt"]    = df["alternative_idioms"].map(lst)
df["sim_en"] = df["similar_in_english"].map(one)
df["tag_lc"] = df["tags"].map(lambda v: [str(t).strip().lower() for t in lst(v)])

n0 = len(df)
dd = (df.sort_values("id").drop_duplicates(subset=["idiom"], keep="first")
        .reset_index(drop=True))
dd = dd[(dd.figurative_meaning_bn.str.len() > 0) &
        (dd.literal_meaning.str.len() > 0)].reset_index(drop=True)
print(f"dedupe + both glosses: {n0} -> {len(dd)}")
dd.to_json(D("splits","full.jsonl"), orient="records", lines=True, force_ascii=False)

# ── extract the English literal gloss from the annotators' parenthetical ───
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
def jaccard(a, b):
    A, B = _tok(a), _tok(b)
    return len(A & B) / max(1, len(A | B))

dd["lit_en"] = dd.literal_meaning.map(eng_literal)
dd["has_lit_en"] = dd.lit_en.str.len() > 0
dd["has_sim_en"] = dd.sim_en.str.len() > 0
dd["lit_fig_overlap"] = [jaccard(r.lit_en, r.figurative_meaning_en) if r.has_lit_en
                         else np.nan for r in dd.itertuples()]
dd["usable_en_lit"] = (dd.has_lit_en & dd.has_sim_en & (dd.lit_fig_overlap < 0.34))

POOL_EN   = dd[dd.has_sim_en].reset_index(drop=True)              # English ceiling
POOL_LIT  = dd[dd.usable_en_lit].reset_index(drop=True)           # English literal capture
SUB_PATH  = D("splits","sub10.jsonl")
SUB = (pd.read_json(SUB_PATH, lines=True) if SUB_PATH.exists()
       else dd.sample(frac=0.10, random_state=909).sort_values("id").reset_index(drop=True))
if not SUB_PATH.exists():
    SUB.to_json(SUB_PATH, orient="records", lines=True, force_ascii=False)

print(f"\nPOOL_EN  (English idiom equivalent) : {len(POOL_EN)}")
print(f"POOL_LIT (usable for D-lit-EN)      : {len(POOL_LIT)}")
print(f"  rejected, literal ~= figurative   : {int((dd.has_lit_en & (dd.lit_fig_overlap>=0.34)).sum())}")
print(f"SUB (shared 10% subset)             : {len(SUB)}")

POOL_LIT[["id","idiom","sim_en","lit_en","figurative_meaning_en","lit_fig_overlap"]] \
    .to_json(D("splits","pool_en_literal.jsonl"), orient="records", lines=True, force_ascii=False)
json.dump({"n_raw": n0, "n_usable": len(dd), "n_pool_en": len(POOL_EN),
           "n_pool_en_literal": len(POOL_LIT), "n_subset": len(SUB),
           "literal_overlap_threshold": 0.34},
          open(D("splits","meta_en.json"),"w"), indent=2)

for r in POOL_LIT.head(3).itertuples():
    print(f"\n  {r.idiom}\n    EN idiom  : {r.sim_en}\n    literal   : {r.lit_en}"
          f"\n    figurative: {r.figurative_meaning_en[:70]}")
```

---

## 04 — Task instrument (English + Bangla), seeds pinned

```python
# ── 04 ── same contract as the other arms; new seed bases for EN conditions ─
import random
from collections import defaultdict

K        = 4
OPT_KEYS = ["A","B","C","D"]
INSTR_BN = "নিচের বাংলা বাগধারাটির প্রকৃত ভাবার্থ কোনটি?"
INSTR_EN = "What is the actual figurative meaning of this idiom?"
PUNCT    = "।,;:!?\"'()[]{}—–-"

# seed bases. 1000-7000 are used by the Bangla arms; English conditions get
# 11000+ so nothing collides, and Laya/Lod backfill must reuse these exactly.
SEEDS = {"j1_order": 17000, "j2_lit_en": 12000, "j3_ceiling_en": 11000,
         "j4_bn_control": 14000, "j5_options_en": 16000}

full = pd.read_json(D("splits","full.jsonl"), lines=True)

def words(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return [w for w in t.split() if w]

tok2ids, tag2ids = defaultdict(set), defaultdict(set)
for r in full.itertuples():
    for w in set(words(r.idiom)):
        if len(w) >= 2: tok2ids[w].add(r.id)
    for t in r.tag_lc: tag2ids[t].add(r.id)

MEAN_BN = dict(zip(full.id, full.figurative_meaning_bn))
MEAN_EN = dict(zip(full.id, full.figurative_meaning_en))
LIT_BN  = dict(zip(full.id, full.literal_meaning))
LIT_EN  = dict(zip(full.id, full.lit_en))
ALL_IDS = list(full.id)

def _fill(pool, need, exclude, rng):
    pool = [i for i in pool if i not in exclude]; rng.shuffle(pool)
    out = pool[:need]
    while len(out) < need:
        c = rng.choice(ALL_IDS)
        if c not in exclude and c not in out: out.append(c)
    return out

def distractor_ids(row, family, rng):
    """Return K-1 distractor ITEM IDS; the caller picks the language."""
    ex = {row.id}
    if family == "D-rand":
        return _fill(list(ALL_IDS), K-1, ex, rng), None
    if family == "D-lit":
        return _fill(list(ALL_IDS), K-2, ex, rng), "literal"
    if family == "D-surf":
        c = set()
        for w in set(words(row.idiom)):
            if len(w) >= 2: c |= tok2ids.get(w, set())
        return _fill(list(c), K-1, ex, rng), None
    if family == "D-sem":
        c = set()
        for t in row.tag_lc: c |= tag2ids.get(t, set())
        return _fill(list(c), K-1, ex, rng), None
    if family == "D-hard":
        surf = set()
        for w in set(words(row.idiom)):
            if len(w) >= 2: surf |= tok2ids.get(w, set())
        sem = set()
        for t in row.tag_lc: sem |= tag2ids.get(t, set())
        ids = _fill(list(surf), 1, ex, rng) + _fill(list(sem), K-3, ex, rng)
        return ids[:K-2], "literal"
    raise ValueError(family)

def build(row, family, lang, rng):
    """(question, gold_key, texts, literal_key). lang in {'en','bn'}."""
    MEAN = MEAN_EN if lang == "en" else MEAN_BN
    LIT  = LIT_EN  if lang == "en" else LIT_BN
    gold = MEAN[row.id]
    ids, lit_flag = distractor_ids(row, family, rng)
    opts = [gold] + ([LIT[row.id]] if lit_flag else []) + [MEAN[i] for i in ids]
    opts = [o for o in opts if str(o).strip()]
    opts = list(dict.fromkeys(opts))[:K]
    while len(opts) < K:
        c = MEAN[rng.choice(ALL_IDS)]
        if c not in opts: opts.append(c)
    order = list(range(K)); rng.shuffle(order)
    texts = [opts[i] for i in order]
    q = {"type": "choice",
         "instructions": INSTR_EN if lang == "en" else INSTR_BN,
         "criteria": {k: t for k, t in zip(OPT_KEYS, texts)}}
    gk = OPT_KEYS[texts.index(gold)]
    lk = (OPT_KEYS[texts.index(LIT[row.id])]
          if lit_flag and LIT[row.id] in texts else None)
    return q, gk, texts, lk

def jsd(p, q):
    p, q = np.asarray(p,float), np.asarray(q,float); m = (p+q)/2
    def kl(a,b):
        a=np.clip(a,1e-12,1); b=np.clip(b,1e-12,1); return float((a*np.log(a/b)).sum())
    return 0.5*kl(p,m) + 0.5*kl(q,m)

json.dump({"k": K, "option_keys": OPT_KEYS, "instructions_en": INSTR_EN,
           "instructions_bn": INSTR_BN, "seeds": SEEDS,
           "families": ["D-rand","D-surf","D-sem","D-lit","D-hard"]},
          open(ROOT/"configs/task_en.json","w"), ensure_ascii=False, indent=2)

r0 = POOL_LIT.iloc[0]
q,gk,tx,lk = build(r0, "D-lit", "en", random.Random(SEEDS["j2_lit_en"]+int(r0.id)))
print("probe —", r0.sim_en, "| gold", gk, "| literal", lk)
for k,v in q["criteria"].items(): print(f"  {k}: {str(v)[:64]}")
```

---

## 05 — Jev client, endpoint probe, response contract

```python
# ── 05 ── Decisions API client. Resolves the endpoint, then asserts shape. ──
import requests, time, json, threading

CANDIDATE_URLS = ["https://openrouter.ai/api/alpha/decisions",
                  "https://openrouter.ai/api/v1/alpha/decisions",
                  "https://openrouter.ai/api/v1/api/alpha/decisions"]
HEADERS = {"Authorization": f"Bearer {API_KEY}",
           "Content-Type": "application/json",
           "HTTP-Referer": "https://github.com/sakhadib/BNFDM",
           "X-Title": "BNFDM"}
JEV = "typesafe/jev-1.13"

_probe = {"model": JEV, "state": "The checkout page is blank after I click Pay.",
          "questions": {"q": {"type":"choice",
              "instructions":"Which team should own this ticket?",
              "criteria":{"A":"Payment and checkout issues.",
                          "B":"Login and account issues."}}}}

URL = None
for u in CANDIDATE_URLS:
    try:
        r = requests.post(u, headers=HEADERS, json=_probe, timeout=30)
        print(f"{r.status_code}  {u}")
        if r.status_code == 200 and "answers" in r.json():
            URL = u; print(json.dumps(r.json(), indent=2)[:700]); break
        if r.status_code in (401, 402, 403):
            raise SystemExit(f"auth/credit problem: {r.status_code} {r.text[:200]}")
    except requests.RequestException as e:
        print("  ", type(e).__name__, e)
assert URL, "no Decisions endpoint responded 200 — check the API reference"
print("\nendpoint:", URL)

COST = {"usd": 0.0, "calls": 0}
_lock = threading.Lock()

def ask(state, q, qid="q", retries=6):
    """One typed question. Returns (answer_dict, usage). Retries 429/5xx."""
    body = {"model": JEV, "state": state, "questions": {qid: q}}
    delay = 1.0
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers=HEADERS, json=body, timeout=60)
        except requests.RequestException:
            time.sleep(delay); delay = min(delay*2, 30); continue
        if r.status_code == 200:
            j = r.json(); u = j.get("usage", {})
            with _lock:
                COST["usd"] += float(u.get("cost", 0) or 0); COST["calls"] += 1
            return j["answers"][qid], u
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", delay))
            time.sleep(wait); delay = min(delay*2, 30); continue
        if r.status_code >= 500:
            time.sleep(delay); delay = min(delay*2, 30); continue
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    raise RuntimeError("exhausted retries")

def unpack(a):
    pr  = {k: float(a["probabilities"].get(k, 0.0)) for k in OPT_KEYS}
    top = a.get("choice") or max(pr, key=pr.get)
    return {"pred": top, "max_prob": pr[top],
            "conf_head": float(a["confidence"]) if "confidence" in a else None,
            "probs": pr}

# contract check on a real item
r0 = POOL_LIT.iloc[0]
q, gk, tx, lk = build(r0, "D-lit", "en", random.Random(SEEDS["j2_lit_en"]+int(r0.id)))
a, usage = ask(r0.sim_en, q)
print(json.dumps(a, indent=2)[:500]); print("usage:", usage)
assert set(a["probabilities"]) <= set(OPT_KEYS), a["probabilities"]
assert abs(sum(a["probabilities"].values()) - 1) < 1e-2
u = unpack(a)
print(f"\npred {u['pred']} | gold {gk} | literal {lk}")
print(f"max_prob {u['max_prob']:.4f}  <- cross-model comparable")
print(f"conf_head {u['conf_head']}  <- separate signal, NOT comparable")
print(f"\nconfidence == max_prob ? {abs((u['conf_head'] or -1) - u['max_prob']) < 1e-6}")

json.dump({**ENV, "endpoint": URL, "model_id": JEV},
          open(R("env.json"),"w"), indent=2, default=str)
```

Note what the last line prints. If `confidence == max_prob` is **False**, Jev's confidence is a separate head like Lod's and belongs in `conf_head`. If **True**, it is just the top probability and the two columns are redundant — record which, it changes how E5 is read.

---

## 06 — Concurrency probe

```python
# ── 06 ── find a safe worker count BEFORE launching 100k calls ─────────────
import time, random
from concurrent.futures import ThreadPoolExecutor

rows = list(POOL_EN.head(64).itertuples())
def _one(r):
    rg = random.Random(SEEDS["j3_ceiling_en"] + int(r.id))
    q,_,_,_ = build(r, "D-rand", "en", rg)
    t0 = time.time(); ask(r.sim_en, q); return time.time()-t0

for W in (1, 4, 8, 16):
    sel = rows[:32]
    t0 = time.time()
    try:
        with ThreadPoolExecutor(W) as ex: list(ex.map(_one, sel))
        wall = time.time()-t0
        print(f"workers={W:>2}  {wall:5.1f}s for {len(sel)}  "
              f"-> {wall/len(sel)*1000:5.0f} ms/call effective")
    except Exception as e:
        print(f"workers={W:>2}  FAILED: {type(e).__name__} {e}"); break

WORKERS = 8          # raise only if the sweep above stayed clean
print(f"\nWORKERS = {WORKERS}")
print(f"spend so far: ${COST['usd']:.4f} over {COST['calls']} calls")
```

If 16 workers trips sustained 429s, stay at 8. The suite is ~100k calls; at 8 workers and 0.18s that is roughly 40 minutes.

---

## 07 — Shared runner (resume-safe, concurrent)

```python
# ── 07 ── every block below uses this. Appends JSONL, skips done keys. ─────
import json, random
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def run_block(name, jobs, make, key_fields, workers=None):
    """jobs: iterable of job dicts. make(job) -> (state, question, extra_dict).
    Writes one JSONL line per job, keyed by key_fields, resumable."""
    JL = R(name, "raw.jsonl")
    done = set()
    if JL.exists():
        for l in open(JL, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); done.add(tuple(d[k] for k in key_fields))
    jobs = [j for j in jobs if tuple(j[k] for k in key_fields) not in done]
    print(f"{name}: {len(jobs)} to run ({len(done)} already done)")
    if not jobs: return

    f = open(JL, "a", encoding="utf-8"); lk = threading.Lock()
    def work(job):
        state, q, extra = make(job)
        a, usage = ask(state, q)
        u = unpack(a)
        rec = {**job, **extra, "pred": u["pred"], "max_prob": u["max_prob"],
               "conf_head": u["conf_head"],
               **{f"p_{k}": u["probs"][k] for k in OPT_KEYS},
               "input_tokens": usage.get("input_tokens"),
               "cost": usage.get("cost")}
        with lk:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

    with ThreadPoolExecutor(workers or WORKERS) as ex:
        list(tqdm(ex.map(work, jobs), total=len(jobs), desc=name))
    f.close()
    print(f"  spend so far: ${COST['usd']:.4f} / {COST['calls']} calls")

def load_block(name, key_fields):
    d = pd.read_json(R(name, "raw.jsonl"), lines=True)
    return d.drop_duplicates(key_fields, keep="last")
```

---

## 08 — J1: option-order audit (run this first)

The one measurement nobody outside TypeSafe can make.

```python
# ── 08 ── J1: all 24 orderings of a fixed option set, 300 items ────────────
import itertools

PERMS = list(itertools.permutations(range(K)))
probe = POOL_LIT.sample(min(300, len(POOL_LIT)), random_state=7).reset_index(drop=True)

jobs, cache = [], {}
for r in probe.itertuples():
    rng = random.Random(SEEDS["j1_order"] + int(r.id))
    q, gk, texts, lk = build(r, "D-hard", "en", rng)
    cache[r.id] = (texts, r.figurative_meaning_en, r.sim_en)
    for pi in range(len(PERMS)):
        jobs.append({"id": int(r.id), "perm_id": pi})

def make_j1(job):
    texts, gold, state = cache[job["id"]]
    perm = PERMS[job["perm_id"]]
    t = [texts[i] for i in perm]
    q = {"type":"choice","instructions":INSTR_EN,
         "criteria":{k: v for k, v in zip(OPT_KEYS, t)}}
    gk = OPT_KEYS[t.index(gold)]
    return state, q, {"perm": "".join(map(str, perm)), "gold_key": gk,
                      "gold_pos": OPT_KEYS.index(gk)}

print(f"{len(jobs):,} calls  (~${len(jobs)*320*0.042/1e6:.2f})")
run_block("j1_order", jobs, make_j1, ["id","perm_id"])

J1 = load_block("j1_order", ["id","perm_id"])
J1["correct"] = (J1.pred == J1.gold_key).astype(int)
J1["perm"] = J1.perm.astype(str).str.zfill(K)
# chosen OPTION IDENTITY, not slot key: a key necessarily moves with content
J1["chosen_opt"] = [int(p[OPT_KEYS.index(pr)]) for p, pr in zip(J1.perm, J1.pred)]
J1.to_csv(R("j1_order","raw.csv"), index=False, encoding="utf-8")

J1["p_gold"] = J1.apply(lambda r: r[f"p_{r.gold_key}"], axis=1)
g = J1.groupby("id")
per = pd.DataFrame({"n_distinct_option": g.chosen_opt.nunique(),
                    "n_distinct_key": g.pred.nunique(),
                    "accuracy": g.correct.mean(),
                    "p_gold_mean": g.p_gold.mean(),
                    "p_gold_std": g.p_gold.std(),
                    "p_gold_range": g.p_gold.agg(lambda s: s.max()-s.min()),
                    "conf_range": g.max_prob.agg(lambda s: s.max()-s.min())}).reset_index()
per["answer_unstable"] = per.n_distinct_option > 1
per["correctness_varies"] = (per.accuracy > 0) & (per.accuracy < 1)
per.to_csv(R("j1_order","per_item.csv"), index=False)

print(f"\nitems whose chosen gloss changes : {per.answer_unstable.mean():.3f}")
print(f"items whose correctness varies   : {per.correctness_varies.mean():.3f}")
print(f"mean distinct glosses (of 4)     : {per.n_distinct_option.mean():.3f}")
print(f"mean within-item p(gold) range   : {per.p_gold_range.mean():.4f}")
print(f"overall accuracy                 : {J1.correct.mean():.4f}")
print("\nreference — Laya 0.837 unstable / range 0.346 ; Lod 0.107 / 0.029")
J1.groupby("gold_pos").agg(n=("correct","size"), acc=("correct","mean"),
    p_gold=("p_gold","mean")).reset_index().to_csv(
    R("j1_order","by_gold_position.csv"), index=False)
J1.pred.value_counts(normalize=True).sort_index().rename("share").reset_index().to_csv(
    R("j1_order","selection_bias.csv"), index=False)
```

---

## 09 — J2: English literal capture

```python
# ── 09 ── J2: D-lit-EN and D-hard-EN on the 3,265 usable items ─────────────
FAM_LIT = ["D-rand", "D-lit", "D-hard"]     # D-rand as the in-pool reference
jobs, cache = [], {}
for r in POOL_LIT.itertuples():
    for fam in FAM_LIT:
        rng = random.Random(SEEDS["j2_lit_en"] + int(r.id))
        q, gk, texts, lk = build(r, fam, "en", rng)
        cache[(r.id, fam)] = (r.sim_en, q, gk, lk, texts)
        jobs.append({"id": int(r.id), "family": fam})

def make_j2(job):
    state, q, gk, lk, texts = cache[(job["id"], job["family"])]
    return state, q, {"gold_key": gk, "lit_key": lk,
                      "gold_pos": OPT_KEYS.index(gk)}

print(f"{len(jobs):,} calls  (~${len(jobs)*340*0.042/1e6:.2f})")
run_block("j2_literal_en", jobs, make_j2, ["id","family"])

J2 = load_block("j2_literal_en", ["id","family"])
J2["correct"] = (J2.pred == J2.gold_key).astype(int)
J2["chose_literal"] = ((J2.lit_key.notna()) & (J2.pred == J2.lit_key)).astype(int)
J2["p_gold"] = J2.apply(lambda r: r[f"p_{r.gold_key}"], axis=1)
J2["p_literal"] = J2.apply(
    lambda r: r[f"p_{r.lit_key}"] if isinstance(r.lit_key, str) else np.nan, axis=1)
J2.to_csv(R("j2_literal_en","raw.csv"), index=False, encoding="utf-8")

s = (J2.groupby("family").apply(lambda g: pd.Series({
        "n": len(g), "accuracy": g.correct.mean(), "chance": 1/K,
        "LAR": g.chose_literal.mean() if g.lit_key.notna().any() else np.nan,
        "mean_p_gold": g.p_gold.mean(), "mean_p_literal": g.p_literal.mean(),
        "mean_max_prob": g.max_prob.mean(), "mean_conf_head": g.conf_head.mean()}))
     .reset_index())
s.to_csv(R("j2_literal_en","summary.csv"), index=False)
print(s.round(4).to_string(index=False))
print("\nreference (Bangla, same-pool restriction computed in cell 12):")
print("  Laya full-set LAR 0.644 ; Lod 0.885 — do NOT compare to these directly")
```

---

## 10 — J3: English ceiling

```python
# ── 10 ── J3: D-rand / D-surf / D-sem on all items with an English idiom ───
FAM_CEIL = ["D-rand", "D-surf", "D-sem"]
jobs, cache = [], {}
for r in POOL_EN.itertuples():
    for fam in FAM_CEIL:
        rng = random.Random(SEEDS["j3_ceiling_en"] + int(r.id))
        q, gk, texts, lk = build(r, fam, "en", rng)
        cache[(r.id, fam)] = (r.sim_en, q, gk)
        jobs.append({"id": int(r.id), "family": fam})

def make_j3(job):
    state, q, gk = cache[(job["id"], job["family"])]
    return state, q, {"gold_key": gk, "gold_pos": OPT_KEYS.index(gk)}

print(f"{len(jobs):,} calls  (~${len(jobs)*340*0.042/1e6:.2f})")
run_block("j3_ceiling_en", jobs, make_j3, ["id","family"])

J3 = load_block("j3_ceiling_en", ["id","family"])
J3["correct"] = (J3.pred == J3.gold_key).astype(int)
J3["p_gold"] = J3.apply(lambda r: r[f"p_{r.gold_key}"], axis=1)
J3.to_csv(R("j3_ceiling_en","raw.csv"), index=False, encoding="utf-8")
(J3.groupby("family").agg(n=("correct","size"), accuracy=("correct","mean"),
    p_gold=("p_gold","mean"), max_prob=("max_prob","mean"),
    conf_head=("conf_head","mean")).reset_index()
   .to_csv(R("j3_ceiling_en","summary.csv"), index=False))
print(pd.read_csv(R("j3_ceiling_en","summary.csv")).round(4).to_string(index=False))
print("\nreference — Laya en_ml 0.617, en_en 0.733 ; Lod en 0.864")
```

---

## 11 — J4: Bengali-script control

```python
# ── 11 ── J4: an English-only model fed the Bangla idiom. Expect chance. ───
jobs, cache = [], {}
for r in POOL_EN.itertuples():
    rng = random.Random(SEEDS["j4_bn_control"] + int(r.id))
    q, gk, texts, lk = build(r, "D-rand", "bn", rng)
    cache[r.id] = (r.idiom, q, gk)
    jobs.append({"id": int(r.id)})

def make_j4(job):
    state, q, gk = cache[job["id"]]
    return state, q, {"gold_key": gk, "gold_pos": OPT_KEYS.index(gk)}

print(f"{len(jobs):,} calls  (~${len(jobs)*340*0.042/1e6:.2f})")
run_block("j4_bn_control", jobs, make_j4, ["id"])

J4 = load_block("j4_bn_control", ["id"])
J4["correct"] = (J4.pred == J4.gold_key).astype(int)
J4["p_gold"] = J4.apply(lambda r: r[f"p_{r.gold_key}"], axis=1)
J4.to_csv(R("j4_bn_control","raw.csv"), index=False, encoding="utf-8")
pd.DataFrame([{"n": len(J4), "accuracy": J4.correct.mean(), "chance": 1/K,
               "mean_p_gold": J4.p_gold.mean(), "mean_max_prob": J4.max_prob.mean(),
               "mean_conf_head": J4.conf_head.mean()}]).to_csv(
    R("j4_bn_control","summary.csv"), index=False)
print(pd.read_csv(R("j4_bn_control","summary.csv")).round(4).to_string(index=False))
print("reference — Laya's English checkpoint on Bengali: 0.283 at conf 0.390")
```

---

## 12 — J5: option-side ablation (English)

```python
# ── 12 ── J5: perturb the OPTION text, hold state and gold fixed ───────────
sub_ids = set(SUB.id) & set(POOL_EN.id)
pool = POOL_EN[POOL_EN.id.isin(sub_ids)].reset_index(drop=True)

def trunc(t, n=6): return " ".join(str(t).split()[:n])
def scram(t, rng):
    w = str(t).split(); rng.shuffle(w); return " ".join(w)

jobs, cache = [], {}
for r in pool.itertuples():
    rng = random.Random(SEEDS["j5_options_en"] + int(r.id))
    ids = [r.id] + _fill(list(ALL_IDS), K-1, {r.id}, rng)
    order = list(range(K)); rng.shuffle(order); ids = [ids[i] for i in order]
    gk = OPT_KEYS[ids.index(r.id)]
    forms = {"O0_full_en":   [MEAN_EN[i] for i in ids],
             "O1_trunc6":    [trunc(MEAN_EN[i]) for i in ids],
             "O2_scrambled": [scram(MEAN_EN[i], rng) for i in ids],
             "O3_bangla":    [MEAN_BN[i] for i in ids],
             "O4_labels":    [f"meaning {j+1}" for j in range(K)]}
    for fname, texts in forms.items():
        cache[(r.id, fname)] = (r.sim_en, texts, gk)
        jobs.append({"id": int(r.id), "form": fname})

def make_j5(job):
    state, texts, gk = cache[(job["id"], job["form"])]
    q = {"type":"choice","instructions":INSTR_EN,
         "criteria":{k: t for k, t in zip(OPT_KEYS, texts)}}
    return state, q, {"gold_key": gk,
                      "opt_tokens": sum(len(str(t).split()) for t in texts)}

print(f"{len(jobs):,} calls  (~${len(jobs)*340*0.042/1e6:.2f})")
run_block("j5_options_en", jobs, make_j5, ["id","form"])

J5 = load_block("j5_options_en", ["id","form"])
J5["correct"] = (J5.pred == J5.gold_key).astype(int)
J5["p_gold"] = J5.apply(lambda r: r[f"p_{r.gold_key}"], axis=1)
J5.to_csv(R("j5_options_en","raw.csv"), index=False, encoding="utf-8")
(J5.groupby("form").agg(n=("correct","size"), accuracy=("correct","mean"),
    p_gold=("p_gold","mean"), max_prob=("max_prob","mean"),
    conf_head=("conf_head","mean"), opt_tokens=("opt_tokens","mean"))
   .reset_index().to_csv(R("j5_options_en","summary.csv"), index=False))
print(pd.read_csv(R("j5_options_en","summary.csv")).round(4).to_string(index=False))
```

---

## 13 — Manifest

```python
# ── 13 ── inventory + everything the cross-model pass needs ────────────────
import hashlib
def sha(p):
    h = hashlib.sha256()
    with open(p,"rb") as fh:
        while (b := fh.read(1<<20)): h.update(b)
    return h.hexdigest()[:16]

arte = sorted([*(ROOT/"results"/MODEL).rglob("*.csv"),
               *(ROOT/"results"/MODEL).rglob("*.json*"),
               *(ROOT/"data/splits").glob("*en*"),
               *(ROOT/"configs").glob("task_en.json")])
json.dump({**ENV, "endpoint": URL, "model_id": JEV,
           "k": K, "option_keys": OPT_KEYS, "seeds": SEEDS,
           "instructions_en": INSTR_EN, "instructions_bn": INSTR_BN,
           "n_pool_en": int(len(POOL_EN)), "n_pool_en_literal": int(len(POOL_LIT)),
           "literal_overlap_threshold": 0.34,
           "total_cost_usd": COST["usd"], "total_calls": COST["calls"],
           "artifacts": [{"path": str(a.relative_to(ROOT)), "bytes": a.stat().st_size,
                          "sha256_16": sha(a)} for a in arte]},
          open(R(f"manifest_{RUN}.json"),"w"), indent=2, default=str, ensure_ascii=False)
for a in arte: print(f"{a.stat().st_size:>10,}  {a.relative_to(ROOT)}")
print(f"\n{len(arte)} artefacts | total spend ${COST['usd']:.4f} over {COST['calls']:,} calls")
```

---

## Output

```
BNFDM/
├── configs/task_en.json
├── data/splits/  full · sub10 · pool_en_literal · meta_en
└── results/jev-1.13/
    ├── env.json · manifest_*.json
    ├── j1_order/        raw · per_item · by_gold_position · selection_bias
    ├── j2_literal_en/   raw · summary
    ├── j3_ceiling_en/   raw · summary
    ├── j4_bn_control/   raw · summary
    └── j5_options_en/   raw · summary
```

Push `results/jev-1.13/`, `data/splits/pool_en_literal.jsonl` and `configs/task_en.json` to the repo as `JEV_RUN/`.

## Clearance

Run 01 → 07 and check four things:

1. **Cell 03** prints POOL_EN ≈ 8,761, POOL_LIT ≈ 3,265, rejected ≈ 91.
2. **Cell 05** resolves an endpoint, the assertions pass, and it tells you whether `confidence == max_prob`. Record that answer — it decides how E5 reads.
3. **Cell 06** finds a worker count that does not trip sustained 429s.
4. Spend after the probes is a fraction of a cent.

Then 08 → 13. J1 first: it is independent of every data question and is the finding nobody else can produce. Each block resumes from its JSONL, so a disconnect costs the in-flight calls only.

## What still has to happen after this

J2 is only half an experiment until Laya and Lod run the **same** English literal condition, with seed base `12000` and the same `pool_en_literal.jsonl`. Until then you have Jev's English LAR with nothing to pair it against. Tell me when this run is done and I will write that backfill — it is ~13k calls on Lod and seconds on Laya.
