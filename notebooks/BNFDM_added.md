# BNFDM — the missing matrix cells

One notebook, one new result: **literal capture as a complete model × language
matrix**, on the same 3,265-item pool, with one option-construction function and
one seed namespace.

The paper currently reports literal capture for the two Bangla arms only,
because Jev ran the literal condition in English while Laya and Lod ran it in
Bangla, and the two off-diagonal cells were never measured. This fills all six
cells so the comparison stops being confounded.

| | Bangla | English |
|---|---|---|
| Laya-ml | new | new |
| Lod-lille | new | new |
| Jev-1.13 | **new (the one that matters)** | new (replicates the existing run) |

Running Jev in both languages rather than reusing its old English numbers costs
about eleven cents and buys a free reproducibility check: the new English cell
should land on the existing $\LAR = 0.047$.

**Budget.** Jev ≈ 6,530 calls ≈ \$0.11, a few minutes threaded. Laya ≈ 6,530
single calls at 27 ms ≈ 3 min. Lod ≈ 6,530 at 86 ms ≈ 9 min. Model loading
dominates the wall clock, not inference.

**Runtime:** T4. **Root:** `/content/drive/MyDrive/BNFDM`. Everything this
notebook writes goes to `BNFDM/added/`; nothing existing is touched.

Every cell appends to JSONL and skips work already done, so you can lose the
session and re-run from the top.

---

## 01 — Root, paths, environment

```python
# ── 01 ── same project root, isolated output folder ────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, json, platform, subprocess, datetime as dt
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/BNFDM")
OUT  = ROOT / "added"                      # everything new lands here
RUN  = dt.datetime.now().strftime("%Y%m%d-%H%M")

for d in ["data/splits", "configs", "cache/hf"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
(OUT / "results").mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(ROOT / "cache/hf")

def D(*p):
    q = ROOT / "data" / Path(*p); q.parent.mkdir(parents=True, exist_ok=True); return q
def A(*p):
    q = OUT / Path(*p); q.parent.mkdir(parents=True, exist_ok=True); return q

ENV = {"run": RUN, "python": platform.python_version()}
try:
    ENV["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    ).decode().strip()
except Exception:
    ENV["gpu"] = "cpu"

print(json.dumps(ENV, indent=2))
print("root:", ROOT)
print("out :", OUT)
```

---

## 02 — Install

```python
# ── 02 ── only what is missing. Never -U a preinstalled Colab package. ─────
import sys, subprocess

def need(mod, pkg=None):
    try:
        __import__(mod); return False
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", pkg or mod],
                       check=True); return True

for m, p in [("kagglehub", None), ("laya", "laya"), ("tqdm", None)]:
    print(m, "installed" if need(m, p) else "already present")

import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available())
```

---

## 03 — Corpus, identical preprocessing

Reuses `data/splits/full.jsonl` if this root already has it (it will, if you run
in the Drive the Jev arm used). Otherwise it rebuilds it with exactly the same
steps, so the item set and ids match the existing runs either way.

```python
# ── 03 ── NFC, dedupe on idiom, require both glosses. Same as every arm. ───
import os, re, json, unicodedata, numpy as np, pandas as pd

SEED = 20261001
FULL = D("splits", "full.jsonl")

if FULL.exists():
    dd = pd.read_json(FULL, lines=True)
    print("reused existing full.jsonl:", dd.shape)
else:
    import kagglehub
    RAW = D("bagdhara_raw.jsonl")
    if RAW.exists():
        df = pd.read_json(RAW, lines=True)
    else:
        p  = kagglehub.dataset_download("sakhadib/bangla-bagdhara")
        fs = os.listdir(p)
        jl, js = [f for f in fs if f.endswith(".jsonl")], [f for f in fs if f.endswith(".json")]
        df = pd.read_json(os.path.join(p, (jl or js)[0]), lines=bool(jl))
        df.to_json(RAW, orient="records", lines=True, force_ascii=False)

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

    dd = (df.sort_values("id").drop_duplicates(subset=["idiom"], keep="first")
            .reset_index(drop=True))
    dd = dd[(dd.figurative_meaning_bn.str.len() > 0) &
            (dd.literal_meaning.str.len() > 0)].reset_index(drop=True)
    dd.to_json(FULL, orient="records", lines=True, force_ascii=False)
    print("rebuilt full.jsonl:", dd.shape)

# English columns are derived here rather than trusted from the file
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

if "sim_en" not in dd.columns:
    def lst(v): return v if isinstance(v, list) else ([] if pd.isna(v) else [v])
    dd["sim_en"] = dd["similar_in_english"].map(lambda v: (lst(v) or [""])[0])
if "tag_lc" not in dd.columns:
    dd["tag_lc"] = dd["tags"].map(lambda v: [str(t).strip().lower()
                                             for t in (v if isinstance(v, list) else [])])

dd["lit_en"]          = dd.literal_meaning.map(eng_literal)
dd["has_lit_en"]      = dd.lit_en.str.len() > 0
dd["has_sim_en"]      = dd.sim_en.str.len() > 0
dd["lit_fig_overlap"] = [jaccard(r.lit_en, r.figurative_meaning_en) if r.has_lit_en
                         else np.nan for r in dd.itertuples()]
dd["usable_en_lit"]   = (dd.has_lit_en & dd.has_sim_en & (dd.lit_fig_overlap < 0.34))

POOL = dd[dd.usable_en_lit].reset_index(drop=True)
print(f"\nPOOL (both languages have a usable literal gloss): {len(POOL)}")
assert len(POOL) == 3265, f"pool is {len(POOL)}, expected 3265 — check the corpus version"
```

---

## 04 — One question builder for all six cells

The same distractor item pool is drawn once per item and reused for both
languages, so Bangla and English differ only in the gloss strings. The seed
namespace is new (`21000`), so nothing here collides with the existing runs.

```python
# ── 04 ── D-lit in both languages, from one distractor draw per item ───────
import random
from collections import defaultdict

K        = 4
OPT_KEYS = ["A", "B", "C", "D"]
SEED_M   = 21000                      # new namespace: matrix cells only
PUNCT    = "।,;:!?\"'()[]{}—–-"

def words(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return [w for w in t.split() if w]

MEAN_BN = dict(zip(dd.id, dd.figurative_meaning_bn))
MEAN_EN = dict(zip(dd.id, dd.figurative_meaning_en))
LIT_BN  = dict(zip(dd.id, dd.literal_meaning))
LIT_EN  = dict(zip(dd.id, dd.lit_en))
IDIOM   = dict(zip(dd.id, dd.idiom))
SIM_EN  = dict(zip(dd.id, dd.sim_en))
ALL_IDS = list(dd.id)

INSTR_BN = "নিচের বাংলা বাগধারাটির প্রকৃত ভাবার্থ কোনটি?"
INSTR_EN = "What is the actual figurative meaning of this idiom?"

def _fill(pool, need, exclude, rng):
    pool = [i for i in pool if i not in exclude]; rng.shuffle(pool)
    out = pool[:need]
    while len(out) < need:
        c = rng.choice(ALL_IDS)
        if c not in exclude and c not in out: out.append(c)
    return out

def build_pair(row):
    """One item -> (bn, en) each a dict with question/state/gold_key/lit_key.

    D-lit: gold + the idiom's own literal gloss + K-2 random other glosses.
    Both languages share the distractor ITEM ids and get independent shuffles
    from the same seeded generator, so neither language sees a privileged order.
    """
    rng  = random.Random(SEED_M + int(row.id))
    pool = _fill(list(ALL_IDS), K - 2, {row.id}, rng)
    outs = {}
    for lang in ("bn", "en"):
        MEAN = MEAN_BN if lang == "bn" else MEAN_EN
        LIT  = LIT_BN  if lang == "bn" else LIT_EN
        gold = MEAN[row.id]
        lit  = LIT[row.id]
        opts = [gold, lit] + [MEAN[i] for i in pool]
        opts = [o for o in opts if str(o).strip()]
        opts = list(dict.fromkeys(opts))[:K]
        while len(opts) < K:
            c = MEAN[rng.choice(ALL_IDS)]
            if c and c not in opts: opts.append(c)
        order = list(range(K)); rng.shuffle(order)
        texts = [opts[i] for i in order]
        outs[lang] = {
            "state": IDIOM[row.id] if lang == "bn" else SIM_EN[row.id],
            "q": {"type": "choice",
                  "instructions": INSTR_BN if lang == "bn" else INSTR_EN,
                  "criteria": {k: t for k, t in zip(OPT_KEYS, texts)}},
            "gold_key": OPT_KEYS[texts.index(gold)],
            "lit_key":  OPT_KEYS[texts.index(lit)] if lit in texts else None,
            "texts": texts,
        }
    return outs

json.dump({"k": K, "option_keys": OPT_KEYS, "seed_matrix": SEED_M,
           "family": "D-lit", "n_pool": int(len(POOL)),
           "instructions_bn": INSTR_BN, "instructions_en": INSTR_EN},
          open(A("configs_matrix.json"), "w"), ensure_ascii=False, indent=2)

# eyeball one item in both languages
r0 = POOL.iloc[0]
pair = build_pair(r0)
for lang in ("bn", "en"):
    p = pair[lang]
    print(f"\n[{lang}] state: {p['state'][:60]}  | gold {p['gold_key']} | literal {p['lit_key']}")
    for k, v in p["q"]["criteria"].items():
        print(f"   {k}: {str(v)[:64]}")
```

---

## 05 — Shared row writer

```python
# ── 05 ── one schema for all six cells, resumable by (id, language) ────────
import numpy as np, pandas as pd, json

def row_from(ans, pr, meta, arm, lang, item_id):
    pred = ans.get("choice") or max(pr, key=pr.get)
    gk, lk = meta["gold_key"], meta["lit_key"]
    return {"id": int(item_id), "arm": arm, "language": lang, "family": "D-lit",
            "gold_key": gk, "lit_key": lk,
            "gold_pos": OPT_KEYS.index(gk),
            "pred": pred,
            "correct": int(pred == gk),
            "chose_literal": int(lk is not None and pred == lk),
            "p_gold": pr[gk],
            "p_literal": pr[lk] if lk else None,
            "max_prob": pr[pred],
            "conf_head": meta.get("conf_head"),
            **{f"p_{k}": pr.get(k, np.nan) for k in OPT_KEYS}}

def cell_path(arm, lang): return A("results", arm, f"matrix_{lang}", "raw.jsonl")

def load_done(arm, lang):
    p = cell_path(arm, lang)
    if not p.exists(): return set()
    return {(json.loads(l)["id"], json.loads(l)["language"])
            for l in open(p, encoding="utf-8")}

def finalise(arm, lang):
    p = cell_path(arm, lang)
    if not p.exists():
        print(f"  {arm:<16} {lang}  nothing written"); return None
    df = pd.read_json(p, lines=True).drop_duplicates(["id","language"], keep="last")
    df.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8")
    lar = df.chose_literal.mean(); acc = df.correct.mean()
    print(f"  {arm:<16} {lang}  n={len(df):>5}  acc={acc:.4f}  LAR={lar:.4f}")
    return df
```

---

## 06 — Arm 1: Jev (API only, no GPU)

Run this first: it needs no accelerator, so nothing is loaded yet.

```python
# ── 06 ── Jev on both languages. Bangla is the cell the paper is missing. ──
import requests, time, threading, json
from concurrent.futures import ThreadPoolExecutor
from google.colab import userdata
from tqdm.auto import tqdm

URL     = "https://openrouter.ai/api/alpha/decisions"
JEV     = "typesafe/jev-1.13"
HEADERS = {"Authorization": f"Bearer {userdata.get('OPENROUTER_API_KEY')}",
           "Content-Type": "application/json"}
COST    = {"usd": 0.0, "calls": 0}
_lock   = threading.Lock()

def jev_ask(state, q, qid="q", retries=6):
    body, delay = {"model": JEV, "state": state, "questions": {qid: q}}, 1.0
    for _ in range(retries):
        try:
            r = requests.post(URL, headers=HEADERS, json=body, timeout=60)
        except requests.RequestException:
            time.sleep(delay); delay = min(delay*2, 30); continue
        if r.status_code == 200:
            j = r.json(); u = j.get("usage", {})
            with _lock:
                COST["usd"] += float(u.get("cost", 0) or 0); COST["calls"] += 1
            return j["answers"][qid]
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", delay)))
            delay = min(delay*2, 30); continue
        if r.status_code >= 500:
            time.sleep(delay); delay = min(delay*2, 30); continue
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    raise RuntimeError("exhausted retries")

# contract check before spending anything
_p = build_pair(POOL.iloc[0])["bn"]
_a = jev_ask(_p["state"], _p["q"])
assert set(_a["probabilities"]) <= set(OPT_KEYS), _a["probabilities"]
assert abs(sum(_a["probabilities"].values()) - 1) < 1e-2
print("contract ok:", {k: round(v,3) for k,v in _a["probabilities"].items()},
      "| choice", _a.get("choice"))

print(f"\nplanned: {len(POOL)*2:,} calls  (~${len(POOL)*2*1.65e-5:.2f})")

for lang in ("bn", "en"):
    done = load_done("jev-1.13", lang)
    todo = [r for r in POOL.itertuples() if (int(r.id), lang) not in done]
    print(f"\njev-1.13 / {lang}: {len(todo)} to run ({len(done)} already done)")
    if todo:
        f = open(cell_path("jev-1.13", lang), "a", encoding="utf-8")
        def work(row):
            m = build_pair(row)[lang]
            a = jev_ask(m["state"], m["q"])
            pr = {k: float(a["probabilities"].get(k, 0.0)) for k in OPT_KEYS}
            meta = dict(m); meta["conf_head"] = (float(a["confidence"])
                                                 if "confidence" in a else None)
            return row_from(a, pr, meta, "jev-1.13", lang, row.id)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for rec in tqdm(ex.map(work, todo), total=len(todo), desc=f"jev/{lang}"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.close()
    finalise("jev-1.13", lang)

print(f"\nspend so far: ${COST['usd']:.4f} over {COST['calls']:,} calls")
```

---

## 07 — Clear the GPU before loading anything

Nothing is loaded yet, but run it anyway so the cell is in the right order when
you re-run from the top.

```python
# ── 07 ── free the device ──────────────────────────────────────────────────
import gc, torch

for name in ("agent", "model"):
    if name in globals():
        try: del globals()[name]
        except Exception: pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    print(f"allocated {torch.cuda.memory_allocated()/1e9:.2f} GB  "
          f"| reserved {torch.cuda.memory_reserved()/1e9:.2f} GB")
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 08 — Arm 2: Laya (multilingual checkpoint, both languages)

The multilingual checkpoint runs **both** languages here. That is the point: the
model is held fixed and only the language changes, so the English cell is not
confounded by also swapping the checkpoint.

```python
# ── 08 ── Laya, one checkpoint, two languages ──────────────────────────────
import laya, random, json
from tqdm.auto import tqdm

agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")
agent.cfg["max_len"], agent.cfg["head_max_len"] = 1024, 384
print("cfg:", {k: agent.cfg.get(k) for k in ("max_len","head_max_len","amp_dtype")})

def laya_ask(state, q, qid="q"):
    return agent.predict(state, {qid: q})

_p = build_pair(POOL.iloc[0])["bn"]
_r = laya_ask(_p["state"], _p["q"]); _a = _r["answers"]["q"]
assert set(_a["probabilities"]) == set(OPT_KEYS)
assert abs(sum(_a["probabilities"].values()) - 1) < 1e-3
assert not _r["usage"].get("truncated"), "options truncated — raise head_max_len"
print("contract ok | choice", _a["choice"], "| conf", round(_a["answer_confidence"],4))

json.dump({**ENV, "arm": "laya-ml", "checkpoint": "convaiinnovations/laya:multilingual",
           "cfg": {k: agent.cfg.get(k) for k in
                   ("max_len","head_max_len","amp_dtype","temperature",
                    "temperature_by_options")}},
          open(A("results","laya-ml","env.json"), "w"), indent=2, default=str)

for lang in ("bn", "en"):
    done = load_done("laya-ml", lang)
    todo = [r for r in POOL.itertuples() if (int(r.id), lang) not in done]
    print(f"\nlaya-ml / {lang}: {len(todo)} to run")
    if todo:
        f = open(cell_path("laya-ml", lang), "a", encoding="utf-8")
        for row in tqdm(todo, desc=f"laya/{lang}"):
            m = build_pair(row)[lang]
            a = laya_ask(m["state"], m["q"])["answers"]["q"]
            pr = {k: float(a["probabilities"][k]) for k in OPT_KEYS}
            meta = dict(m); meta["conf_head"] = float(a["answer_confidence"])
            f.write(json.dumps(row_from(a, pr, meta, "laya-ml", lang, row.id),
                               ensure_ascii=False) + "\n")
            if row.Index % 250 == 0: f.flush()
        f.close()
    finalise("laya-ml", lang)
```

---

## 09 — Clear the GPU again

Laya must be gone before Lod loads; both on a T4 will not fit comfortably.

```python
# ── 09 ── free the device, properly ────────────────────────────────────────
import gc, torch

try:
    agent.to("cpu")
except Exception:
    pass
for name in ("agent", "_r", "_a"):
    if name in globals():
        try: del globals()[name]
        except Exception: pass
gc.collect()
torch.cuda.empty_cache(); torch.cuda.ipc_collect()
print(f"allocated {torch.cuda.memory_allocated()/1e9:.2f} GB  "
      f"| reserved {torch.cuda.memory_reserved()/1e9:.2f} GB")
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

If `memory.used` is still high, the cleanest fix is **Runtime → Restart session**
and then re-run cells 01, 03, 04, 05 and 10. Every cell resumes from JSONL, so
nothing already finished is recomputed.

---

## 10 — Arm 3: Lod (fp32, both languages)

Lod ran a 10% subset originally. Here it runs the full 3,265-item pool, which
supersedes the thin 316-item in-pool estimate the paper currently carries.

```python
# ── 10 ── Lod-lille, fp32 for exact option-order independence ──────────────
import torch, json
from transformers import AutoModel
from tqdm.auto import tqdm

REPO  = "mrn-dk/lod-lille-0.6B"
DTYPE = torch.float32

model = AutoModel.from_pretrained(REPO, trust_remote_code=True,
                                  dtype=DTYPE).eval().to("cuda")
cfgd = getattr(model, "config", None)
print(type(model).__name__, "| dtype", next(model.parameters()).dtype,
      "| temperature:", getattr(cfgd, "temperature", "?"),
      "| confidence_mode:", getattr(cfgd, "confidence_mode", "?"))

def lod_ask(state, q, qid="q"):
    st = state if isinstance(state, dict) else {"text": state}
    out = model.score(state=st, questions={qid: q})
    return out["answers"][qid], out.get("usage", {})

_p = build_pair(POOL.iloc[0])["bn"]
_a, _u = lod_ask(_p["state"], _p["q"])
assert set(_a["probabilities"]) == set(OPT_KEYS), _a["probabilities"]
assert abs(sum(_a["probabilities"].values()) - 1) < 1e-2
assert not _u.get("state_truncated", False)
print("contract ok | choice", _a.get("choice"))

json.dump({**ENV, "arm": "lod-lille-0.6b", "repo": REPO, "dtype": str(DTYPE),
           "temperature": getattr(cfgd, "temperature", None),
           "confidence_mode": getattr(cfgd, "confidence_mode", None)},
          open(A("results","lod-lille-0.6b","env.json"), "w"), indent=2, default=str)

for lang in ("bn", "en"):
    done = load_done("lod-lille-0.6b", lang)
    todo = [r for r in POOL.itertuples() if (int(r.id), lang) not in done]
    print(f"\nlod-lille / {lang}: {len(todo)} to run")
    if todo:
        f = open(cell_path("lod-lille-0.6b", lang), "a", encoding="utf-8")
        for row in tqdm(todo, desc=f"lod/{lang}"):
            m = build_pair(row)[lang]
            a, _ = lod_ask(m["state"], m["q"])
            pr = {k: float(a["probabilities"][k]) for k in OPT_KEYS}
            meta = dict(m); meta["conf_head"] = (float(a["confidence"])
                                                 if "confidence" in a else None)
            f.write(json.dumps(row_from(a, pr, meta, "lod-lille-0.6b", lang, row.id),
                               ensure_ascii=False) + "\n")
            if row.Index % 250 == 0: f.flush()
        f.close()
    finalise("lod-lille-0.6b", lang)
```

---

## 11 — The matrix

```python
# ── 11 ── six cells, one table, with paired tests down the language axis ───
import numpy as np, pandas as pd, json
from scipy.stats import binomtest

ARMS = ["laya-ml", "lod-lille-0.6b", "jev-1.13"]

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); rs = np.random.RandomState(seed)
    bs = x[rs.randint(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

cells, frames = [], {}
for arm in ARMS:
    for lang in ("bn", "en"):
        p = cell_path(arm, lang).with_suffix(".csv")
        if not p.exists(): print("missing:", arm, lang); continue
        d = pd.read_csv(p); frames[(arm, lang)] = d
        a, alo, ahi = boot_ci(d.correct)
        l, llo, lhi = boot_ci(d.chose_literal)
        cells.append({"arm": arm, "language": lang, "n": len(d),
                      "accuracy": a, "acc_lo": alo, "acc_hi": ahi,
                      "LAR": l, "LAR_lo": llo, "LAR_hi": lhi,
                      "mean_p_gold": d.p_gold.mean(),
                      "mean_p_literal": d.p_literal.mean(),
                      "mean_max_prob": d.max_prob.mean()})

M = pd.DataFrame(cells)
M.to_csv(A("results", "matrix_summary.csv"), index=False)
print(M[["arm","language","n","accuracy","LAR","mean_p_literal"]].round(4).to_string(index=False))

# within-model language effect, paired by item
rows = []
for arm in ARMS:
    if (arm,"bn") not in frames or (arm,"en") not in frames: continue
    b = frames[(arm,"bn")].set_index("id"); e = frames[(arm,"en")].set_index("id")
    j = b[["chose_literal"]].join(e[["chose_literal"]], rsuffix="_en", how="inner")
    b_only = int(((j.chose_literal==1)&(j.chose_literal_en==0)).sum())
    e_only = int(((j.chose_literal==0)&(j.chose_literal_en==1)).sum())
    p = binomtest(b_only, b_only+e_only, 0.5).pvalue if b_only+e_only else 1.0
    rows.append({"arm": arm, "n_paired": len(j),
                 "LAR_bn": j.chose_literal.mean(), "LAR_en": j.chose_literal_en.mean(),
                 "delta_bn_minus_en": j.chose_literal.mean()-j.chose_literal_en.mean(),
                 "captured_bn_only": b_only, "captured_en_only": e_only,
                 "mcnemar_p": p})
P = pd.DataFrame(rows)
P.to_csv(A("results", "matrix_language_effect.csv"), index=False)
print("\nwithin-model language effect on literal attraction")
print(P.round(4).to_string(index=False))

json.dump({**ENV, "seed_matrix": SEED_M, "n_pool": int(len(POOL)),
           "cells": len(M), "jev_cost_usd": COST["usd"], "jev_calls": COST["calls"]},
          open(A("manifest_matrix.json"), "w"), indent=2, default=str)
print("\nwrote:", A("results","matrix_summary.csv"))
```

---

## 12 — Sanity check against the existing run

The Jev English cell is a fresh measurement of a condition the paper already
reports. It should land near $\LAR = 0.047$. If it does, the construction here
matches the original; if it does not, say so rather than quietly using the new
number, because then the two runs are not measuring the same thing.

```python
# ── 12 ── does the replicated cell reproduce? ──────────────────────────────
j_en = M[(M.arm=="jev-1.13") & (M.language=="en")]
if len(j_en):
    got, lo, hi = float(j_en.LAR.iat[0]), float(j_en.LAR_lo.iat[0]), float(j_en.LAR_hi.iat[0])
    print(f"Jev English LAR: {got:.4f}  [{lo:.4f}, {hi:.4f}]   paper reports 0.0472")
    print("reproduces" if lo <= 0.0472 <= hi else
          "DOES NOT reproduce — report this, do not paper over it")
```

---

## 13 — Zip for hand-off

```python
# ── 13 ── one archive to attach ────────────────────────────────────────────
import shutil
z = shutil.make_archive("/content/bnfdm_added", "zip", OUT)
print(z, f"{os.path.getsize(z)/1e6:.1f} MB")
from google.colab import files; files.download(z)
```

---

## What to send back

`bnfdm_added.zip`, or push `BNFDM/added/` to the repo. The two files that matter
are `results/matrix_summary.csv` and `results/matrix_language_effect.csv`; the
per-cell `raw.csv` files carry the item-level rows for paired tests.

## What this changes in the paper

If Jev's Bangla \textsc{lar} stays low, capture is a model property and the
cross-model claim becomes clean three-arm evidence: §5.1 loses its scoping
caveat, Figure 6 comes out of the appendix with every cell measured, and the
Limitations paragraph about the unrun cells is deleted.

If Jev's Bangla \textsc{lar} jumps toward the open arms, that is the more
interesting result: capture tracks the language, not capability, and the paper's
framing inverts into something stronger than it currently claims.

Either answer is publishable. The current state, where neither is known, is the
one a reviewer objects to.
