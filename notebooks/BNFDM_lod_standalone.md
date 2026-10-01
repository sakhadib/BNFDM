# BNFDM — Lod-lille-0.6B arm (standalone)

Self-contained. Builds its own root, downloads its own copy of the dataset, rebuilds the splits and the task instrument from scratch. Nothing from the Laya run is read. Runs on a fixed 10% subset because this model is far slower than Laya.

Root: `/content/drive/MyDrive/BNFDM`. Results under `results/lod-lille-0.6b/`.

**The repo moved.** `thefloydd/qwen3-0.6b-rlcd` is now **`mrn-dk/lod-lille-0.6B`** — same model (Qwen3-0.6B-Base + merged LoRA), renamed account and repo. That's why `bouncy` isn't on PyPI: there is no package any more. It ships `modeling_lod.py` as `trust_remote_code`, so `torch` + `transformers` is the whole dependency list and the call is `model.score(state=..., questions=...)`.

## Reproducibility contract

These constants must match the Laya notebook exactly, or the two arms can't be paired later. They are hardcoded in cells 03 and 04 rather than read from a shared config:

| | value |
|---|---|
| dataset | `sakhadib/bangla-bagdhara` |
| normalisation | Unicode NFC, strip |
| dedupe | `sort_values("id")`, `drop_duplicates("idiom", keep="first")` |
| filter | non-empty `figurative_meaning_bn` **and** `literal_meaning` |
| `K` | 4 |
| `OPT_KEYS` | `["A","B","C","D"]` |
| `INSTR` | `নিচের বাংলা বাগধারাটির প্রকৃত ভাবার্থ কোনটি?` |
| families | `D-rand`, `D-lit`, `D-surf`, `D-sem`, `D-hard` |
| per-item seeds | E1 `1000+id` · E2 `2000+id` · E3 `3000+id` · E4 `4000+id` · E6 `6000+id` · E7 `7000+id` |
| 10% subset | `full.sample(frac=0.10, random_state=909)` |

Because the option sets are rebuilt from the same `full` table with the same per-item seed, this model sees byte-identical questions to Laya's on the overlapping items. The dev/test split is irrelevant to that — only `full` and the seeds matter.

**Four things that differ from Laya:**

1. **`confidence` means something else.** Laya's `answer_confidence` is the top-class probability. Lod's `confidence` is a separate head estimating whether the right answer is among the listed options at all (AURC 0.091 vs 0.113 for top probability). Compare across models on **max-probability**; the head is logged as `conf_head`, never as the comparable quantity.
2. **Order independence by construction in fp32.** Each option attends only to the state, its question and itself, from the same start position. The card reports at most 0.011 drift in bf16 and exact independence in fp32. Laya has a live issue (#779) claiming the opposite. E7 is a verification here, and ~0 is the finding.
3. **No fitted temperature.** `config.json` sets `temperature: 1.0` — the authors fitted one and it made held-out calibration worse. Laya ships per-bucket temperatures including the broken `choice:11+` entry.
4. **English only.** Bangla is out of distribution. Near-chance is a plausible, reportable outcome.

Cross-model comparison is **not** in this notebook. Push both result folders to the repo and it happens in a separate analysis pass.

---

## 01 — Root

```python
# ── 01 ── standalone project root ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, json, platform, subprocess, datetime as dt
from pathlib import Path

ROOT  = Path("/content/drive/MyDrive/BNFDM")
MODEL = "lod-lille-0.6b"
RUN   = dt.datetime.now().strftime("%Y%m%d-%H%M")

for d in ["data/splits", "configs", "results", "cache/hf"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(ROOT / "cache/hf")

def R(*p):
    q = ROOT / "results" / MODEL / Path(*p); q.parent.mkdir(parents=True, exist_ok=True); return q
def D(*p):
    q = ROOT / "data" / Path(*p); q.parent.mkdir(parents=True, exist_ok=True); return q

ENV = {"run": RUN, "model": MODEL, "python": platform.python_version()}
try:
    ENV["gpu"] = subprocess.check_output(
        ["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"]).decode().strip()
except Exception:
    ENV["gpu"] = "cpu"
print(json.dumps(ENV, indent=2)); print("root:", ROOT)
```

---

## 02 — Dependencies

```python
# ── 02 ── kagglehub for the data; torch+transformers are already here ───────
import sys, subprocess
subprocess.run(f"{sys.executable} -m pip -q install kagglehub", shell=True, check=False)

import torch, transformers, pandas as pd, numpy as np
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| pandas", pd.__version__)
print("bf16:", torch.cuda.is_bf16_supported(), "(False on T4 — we use fp32)")

# Flip only if cell 05 fails on an unknown architecture, then RESTART RUNTIME.
UPGRADE_TRANSFORMERS = False
if UPGRADE_TRANSFORMERS:
    subprocess.run(f"{sys.executable} -m pip -q install -U transformers", shell=True)
    print("restart the runtime now, then rerun from cell 01")
```

Note: `huggingface-cli` is retired; the CLI is now `hf`. Nothing here needs it.

---

## 03 — Dataset and splits (rebuilt from scratch)

```python
# ── 03 ── identical pipeline to the Laya arm: NFC, dedupe, gloss filter ────
import os, json, unicodedata, numpy as np, pandas as pd, kagglehub

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
print(f"dedupe + require both glosses: {n0} -> {len(dd)}")

dd.to_json(D("splits","full.jsonl"), orient="records", lines=True, force_ascii=False)

# ── the fixed 10% subset: drawn once, saved, reused on every rerun ────────
SUB_PATH = D("splits","sub10.jsonl")
if SUB_PATH.exists():
    SUB = pd.read_json(SUB_PATH, lines=True); print(f"loaded subset: {len(SUB)}")
else:
    SUB = dd.sample(frac=0.10, random_state=909).sort_values("id").reset_index(drop=True)
    SUB.to_json(SUB_PATH, orient="records", lines=True, force_ascii=False)
    print(f"drew subset: {len(SUB)} of {len(dd)}")
SUB_IDS = set(SUB.id)

print("\nsubset vs full, for the stratification columns:")
for c in ["frequency","scape","historical_significance","religious_significance"]:
    print(f"  {c:<24} full={dd[c].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  {'':24} sub ={SUB[c].value_counts(normalize=True).round(3).to_dict()}")

json.dump({"n_raw": n0, "n_usable": len(dd), "n_subset": len(SUB),
           "subset_seed": 909, "subset_frac": 0.10,
           "dataset": "sakhadib/bangla-bagdhara"},
          open(D("splits","meta.json"),"w"), indent=2)
```

---

## 04 — Task instrument (hardcoded to match the Laya arm)

```python
# ── 04 ── k-way meaning-choice with controlled distractors ─────────────────
import re, random, unicodedata, json, numpy as np, pandas as pd
from collections import defaultdict

K        = 4
OPT_KEYS = ["A","B","C","D"]
INSTR    = "নিচের বাংলা বাগধারাটির প্রকৃত ভাবার্থ কোনটি?"
FAMILIES = ["D-rand","D-lit","D-surf","D-sem","D-hard"]
PUNCT    = "।,;:!?\"'()[]{}—–-"

def words(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return [w for w in t.split() if w]

full = pd.read_json(D("splits","full.jsonl"), lines=True)
SUB  = pd.read_json(D("splits","sub10.jsonl"), lines=True)
SUB_IDS = set(SUB.id)

# distractor pools are drawn from the FULL table, not the subset — that keeps
# the option sets identical to the Laya arm's for the overlapping items.
tok2ids, tag2ids = defaultdict(set), defaultdict(set)
for r in full.itertuples():
    for w in set(words(r.idiom)):
        if len(w) >= 2: tok2ids[w].add(r.id)
    for t in r.tag_lc: tag2ids[t].add(r.id)

MEAN    = dict(zip(full.id, full.figurative_meaning_bn))
MEAN_EN = dict(zip(full.id, full.figurative_meaning_en))
LIT     = dict(zip(full.id, full.literal_meaning))
ALL_IDS = list(full.id)

def _fill(pool, need, exclude, rng):
    pool = [i for i in pool if i not in exclude]; rng.shuffle(pool)
    out = pool[:need]
    while len(out) < need:
        c = rng.choice(ALL_IDS)
        if c not in exclude and c not in out: out.append(c)
    return out

def distractors(row, family, rng):
    ex = {row.id}
    if family == "D-rand": return [MEAN[i] for i in _fill(list(ALL_IDS), K-1, ex, rng)]
    if family == "D-lit":
        return [LIT[row.id]] + [MEAN[i] for i in _fill(list(ALL_IDS), K-2, ex, rng)]
    if family == "D-surf":
        c = set()
        for w in set(words(row.idiom)):
            if len(w) >= 2: c |= tok2ids.get(w, set())
        return [MEAN[i] for i in _fill(list(c), K-1, ex, rng)]
    if family == "D-sem":
        c = set()
        for t in row.tag_lc: c |= tag2ids.get(t, set())
        return [MEAN[i] for i in _fill(list(c), K-1, ex, rng)]
    if family == "D-hard":
        surf = set()
        for w in set(words(row.idiom)):
            if len(w) >= 2: surf |= tok2ids.get(w, set())
        sem = set()
        for t in row.tag_lc: sem |= tag2ids.get(t, set())
        d  = [LIT[row.id]]
        d += [MEAN[i] for i in _fill(list(surf), 1, ex, rng)]
        d += [MEAN[i] for i in _fill(list(sem), K-3, ex|set(), rng)]
        return d[:K-1]
    raise ValueError(family)

def make_question(row, family, rng, options=None, instr=INSTR):
    if options is None:
        options = [row.figurative_meaning_bn] + distractors(row, family, rng)
        options = list(dict.fromkeys(options))[:K]
        while len(options) < K:
            c = MEAN[rng.choice(ALL_IDS)]
            if c not in options: options.append(c)
    gold_text = row.figurative_meaning_bn
    order = list(range(len(options))); rng.shuffle(order)
    texts = [options[i] for i in order]
    return ({"type":"choice","instructions":instr,
             "criteria":{k:t for k,t in zip(OPT_KEYS, texts)}},
            OPT_KEYS[texts.index(gold_text)], texts)

def jsd(p, q):
    p, q = np.asarray(p,float), np.asarray(q,float); m = (p+q)/2
    def kl(a,b):
        a=np.clip(a,1e-12,1); b=np.clip(b,1e-12,1); return float((a*np.log(a/b)).sum())
    return 0.5*kl(p,m) + 0.5*kl(q,m)

json.dump({"k": K, "option_keys": OPT_KEYS, "instructions": INSTR,
           "families": FAMILIES,
           "seeds": {"e1":1000,"e2":2000,"e3":3000,"e4":4000,"e6":6000,"e7":7000}},
          open(ROOT/"configs/task.json","w"), ensure_ascii=False, indent=2)

rng = random.Random(1000 + int(full.iloc[0].id))
_q,_gk,_ = make_question(full.iloc[0], "D-lit", rng)
print("parity probe — idiom:", full.iloc[0].idiom, "| gold:", _gk)
for k,v in _q["criteria"].items(): print(f"  {k}: {v[:60]}")
```

Keep that parity probe output. Comparing it against the Laya notebook's identical probe is the one-line proof that both arms saw the same questions.

---

## 05 — Load the model

```python
# ── 05 ── trust_remote_code; fp32 for exact option-order independence ──────
import torch, json, random, pandas as pd
from transformers import AutoModel

REPO  = "mrn-dk/lod-lille-0.6B"
DTYPE = torch.float32          # card default; order-independent by construction

model = AutoModel.from_pretrained(REPO, trust_remote_code=True,
                                  dtype=DTYPE).eval().to("cuda")
cfgd = getattr(model, "config", None)
print(type(model).__name__, "| dtype", next(model.parameters()).dtype,
      "| device", next(model.parameters()).device)
print("temperature:", getattr(cfgd, "temperature", "?"),
      "| confidence_mode:", getattr(cfgd, "confidence_mode", "?"))

def ask(state, q, qid="q"):
    st = state if isinstance(state, dict) else {"text": state}
    out = model.score(state=st, questions={qid: q})
    return out["answers"][qid], out.get("usage", {})

json.dump({**ENV, "repo": REPO, "dtype": str(DTYPE),
           "temperature": getattr(cfgd, "temperature", None),
           "confidence_mode": getattr(cfgd, "confidence_mode", None)},
          open(R("env.json"),"w"), indent=2, default=str)
```

Unknown-architecture or import error here means the transformers version — go to cell 02, flip `UPGRADE_TRANSFORMERS`, restart, rerun from 01.

---

## 06 — Contract check

```python
# ── 06 ── answer schema, and the two different confidence notions ──────────
import random, json, numpy as np, pandas as pd

row = SUB.iloc[0]
rng = random.Random(1000 + int(row.id))
q, gk, tx = make_question(row, "D-lit", rng)
a, usage = ask(row.idiom, q)
print(json.dumps(a, indent=2, ensure_ascii=False)[:800])

assert "probabilities" in a, f"unexpected keys: {list(a)}"
assert set(a["probabilities"]) == set(OPT_KEYS), a["probabilities"]
assert abs(sum(a["probabilities"].values()) - 1) < 1e-2

def unpack(a):
    pr  = {k: float(a["probabilities"][k]) for k in OPT_KEYS}
    top = a.get("choice") or max(pr, key=pr.get)
    return {"pred": top, "max_prob": pr[top],
            "conf_head": float(a["confidence"]) if "confidence" in a else None,
            "probs": pr}

u = unpack(a)
print("\npred", u["pred"], "| gold", gk)
print("max_prob ", round(u["max_prob"],4), " <- cross-model comparable")
print("conf_head", u["conf_head"], " <- Lod-only separate head, NOT comparable")
print("\nusage:", usage)
assert not usage.get("state_truncated", False)
```

---

## 07 — Latency

```python
# ── 07 ── per-call latency drives every estimate below ─────────────────────
import time, torch, random, pandas as pd

rows = list(SUB.head(40).itertuples())
for r in rows[:5]:
    rg = random.Random(1000+int(r.id)); qq,_,_ = make_question(r,"D-rand",rg); ask(r.idiom, qq)

torch.cuda.synchronize(); t0 = time.time()
for r in rows:
    rg = random.Random(1000+int(r.id)); qq,_,_ = make_question(r,"D-rand",rg)
    ask(r.idiom, qq)
torch.cuda.synchronize()
SINGLE_MS = (time.time()-t0)/len(rows)*1000
PEAK = torch.cuda.max_memory_allocated()/1e9
print(f"single call: {SINGLE_MS:.1f} ms | peak {PEAK:.2f} GB")
pd.DataFrame([{"single_ms": SINGLE_MS, "peak_gb": PEAK, "dtype": str(DTYPE),
               "n_subset": len(SUB)}]).to_csv(R("bench.csv"), index=False)

tot = 300*24 + len(SUB)*5 + int(len(SUB)*0.24)*3 + len(SUB)*6 + len(SUB)*2 + len(SUB)*5
print(f"\nsuite on the 10% subset ≈ {tot:,} calls ≈ {tot*SINGLE_MS/3600000:.1f} h")
```

If that still projects past ~4 hours, drop E7 to 150 items and E6 to half the subset. E1 stays whole — it's the headline.

---

## 08 — E7: option-order sensitivity

The card claims scores are order-independent **by construction** in fp32. This verifies it; ~0 is the finding.

```python
# ── 08 ── all 24 orderings per item, 300 items ─────────────────────────────
import itertools, random, json, numpy as np, pandas as pd
from tqdm.auto import tqdm

probe = SUB.sample(min(300, len(SUB)), random_state=7).reset_index(drop=True)
PERMS = list(itertools.permutations(range(K)))
JL    = R("e7_order","raw.jsonl")
done  = {json.loads(l)["id"] for l in open(JL, encoding="utf-8")} if JL.exists() else set()
f     = open(JL, "a", encoding="utf-8")

for row in tqdm(list(probe.itertuples()), desc="E7"):
    if row.id in done: continue
    rng  = random.Random(7000 + int(row.id))
    opts = [row.figurative_meaning_bn] + distractors(row, "D-hard", rng)
    opts = list(dict.fromkeys(opts))[:K]
    if len(opts) < K: continue
    gold_text = row.figurative_meaning_bn
    for pi, perm in enumerate(PERMS):
        texts = [opts[i] for i in perm]
        crit  = {k:t for k,t in zip(OPT_KEYS, texts)}
        gk    = OPT_KEYS[texts.index(gold_text)]
        a,_   = ask(row.idiom, {"type":"choice","instructions":INSTR,"criteria":crit})
        u     = unpack(a)
        f.write(json.dumps({"id": int(row.id), "perm_id": pi,
            "perm": "".join(map(str,perm)), "gold_key": gk, "pred": u["pred"],
            "correct": int(u["pred"]==gk), "gold_pos": OPT_KEYS.index(gk),
            "pred_pos": OPT_KEYS.index(u["pred"]), "max_prob": u["max_prob"],
            "conf_head": u["conf_head"], "p_gold": u["probs"][gk],
            **{f"p_{k}": u["probs"][k] for k in OPT_KEYS}}) + "\n")
    f.flush()
f.close()

E7 = pd.read_json(JL, lines=True).drop_duplicates(["id","perm_id"], keep="last")
E7.to_csv(R("e7_order","raw.csv"), index=False, encoding="utf-8")
per_item = E7.groupby("id").agg(
    n_distinct_pred=("pred","nunique"), acc=("correct","mean"),
    p_gold_mean=("p_gold","mean"), p_gold_std=("p_gold","std"),
    p_gold_range=("p_gold", lambda s: s.max()-s.min()),
    conf_range=("max_prob", lambda s: s.max()-s.min())).reset_index()
per_item["unstable"] = per_item.n_distinct_pred > 1
per_item.to_csv(R("e7_order","per_item.csv"), index=False)
E7.groupby("gold_pos").agg(n=("correct","size"), acc=("correct","mean"),
    p_gold=("p_gold","mean")).reset_index().to_csv(R("e7_order","by_gold_position.csv"), index=False)
E7.pred_pos.value_counts(normalize=True).sort_index().rename("share").reset_index().to_csv(
    R("e7_order","selection_bias.csv"), index=False)
print(f"unstable {per_item.unstable.mean():.1%} | mean P(gold) range "
      f"{per_item.p_gold_range.mean():.6f} | max {per_item.p_gold_range.max():.6f}")
```

---

## 09 — E1: distractor ladder

```python
# ── 09 ── accuracy and Literal Attraction Rate across distractor families ──
import random, json, numpy as np, pandas as pd
from tqdm.auto import tqdm

samp = SUB
print(f"{len(samp)} x {len(FAMILIES)} = {len(samp)*len(FAMILIES):,} calls "
      f"(~{len(samp)*len(FAMILIES)*SINGLE_MS/60000:.0f} min)")

JL   = R("e1_ladder","raw.jsonl")
done = {(d["id"], d["family"]) for d in
        (json.loads(l) for l in open(JL, encoding="utf-8"))} if JL.exists() else set()
f    = open(JL, "a", encoding="utf-8")

for row in tqdm(list(samp.itertuples()), desc="E1"):
    rng = random.Random(1000 + int(row.id))
    for fam in FAMILIES:
        q, gk, texts = make_question(row, fam, rng)   # advance rng before skipping
        if (row.id, fam) in done: continue
        a, usage = ask(row.idiom, q); u = unpack(a)
        lit_key = (OPT_KEYS[texts.index(row.literal_meaning)]
                   if fam in ("D-lit","D-hard") and row.literal_meaning in texts else None)
        f.write(json.dumps({"id": int(row.id), "family": fam, "idiom": row.idiom,
            "gold_key": gk, "pred": u["pred"], "correct": int(u["pred"]==gk),
            "lit_key": lit_key,
            "chose_literal": int(lit_key is not None and u["pred"]==lit_key),
            "p_gold": u["probs"][gk],
            "p_literal": u["probs"][lit_key] if lit_key else None,
            "max_prob": u["max_prob"], "conf_head": u["conf_head"],
            "gold_pos": OPT_KEYS.index(gk), "input_tokens": usage.get("input_tokens"),
            "frequency": row.frequency, "scape": row.scape,
            "hist_sig": int(row.historical_significance),
            "relig_sig": int(row.religious_significance),
            "n_words": len(words(row.idiom)),
            **{f"p_{k}": u["probs"][k] for k in OPT_KEYS}}, ensure_ascii=False) + "\n")
    f.flush()
f.close()

E1 = pd.read_json(JL, lines=True).drop_duplicates(["id","family"], keep="last")
E1.to_csv(R("e1_ladder","raw.csv"), index=False, encoding="utf-8")

def ece(c, y, bins=15):
    c,y = np.asarray(c,float), np.asarray(y,float); ed=np.linspace(0,1,bins+1); e=0.
    for lo,hi in zip(ed[:-1],ed[1:]):
        m=(c>lo)&(c<=hi)
        if m.sum(): e += m.mean()*abs(y[m].mean()-c[m].mean())
    return float(e)

(E1.groupby("family").apply(lambda g: pd.Series({
    "n": len(g), "accuracy": g.correct.mean(), "chance": 1/K,
    "LAR": g.chose_literal.mean() if g.chose_literal.notna().any() else np.nan,
    "mean_p_gold": g.p_gold.mean(), "mean_p_literal": g.p_literal.mean(),
    "mean_max_prob": g.max_prob.mean(), "mean_conf_head": g.conf_head.mean(),
    "ece_maxprob": ece(g.max_prob, g.correct),
    "ece_confhead": ece(g.conf_head, g.correct) if g.conf_head.notna().any() else np.nan}))
   .reset_index().to_csv(R("e1_ladder","summary.csv"), index=False))

for col in ["frequency","scape","hist_sig","relig_sig","n_words","gold_pos"]:
    (E1.groupby(["family",col]).agg(n=("correct","size"), accuracy=("correct","mean"),
        LAR=("chose_literal","mean"), p_gold=("p_gold","mean")).reset_index()
       .to_csv(R("e1_ladder", f"by_{col}.csv"), index=False))
print(pd.read_csv(R("e1_ladder","summary.csv")).to_string(index=False))
```

---

## 10 — E2: surface invariance

```python
# ── 10 ── alternative_idioms: same gold meaning, different surface ─────────
import random, json, numpy as np, pandas as pd
from tqdm.auto import tqdm

def clean_alts(v, canon):
    out = []
    for a in (v if isinstance(v, list) else []):
        if a is None: continue
        s = str(a).strip()
        if s and s != str(canon).strip() and s not in out: out.append(s)
    return out

SUB2 = SUB.copy()
SUB2["alt_clean"] = [clean_alts(r.alt, r.idiom) for r in SUB2.itertuples()]
pairs = SUB2[SUB2.alt_clean.map(len) > 0].copy()
print(f"{len(pairs)} subset items with usable variants "
      f"(~{len(pairs)*3*SINGLE_MS/60000:.0f} min)")

JL   = R("e2_invariance","raw.jsonl")
done = {json.loads(l)["id"] for l in open(JL, encoding="utf-8")} if JL.exists() else set()
f    = open(JL, "a", encoding="utf-8")

for row in tqdm(list(pairs.itertuples()), desc="E2"):
    if row.id in done: continue
    rng  = random.Random(2000 + int(row.id))
    opts = [row.figurative_meaning_bn] + distractors(row, "D-hard", rng)
    opts = list(dict.fromkeys(opts))[:K]
    if len(opts) < K: continue
    gold_text = row.figurative_meaning_bn
    order = list(range(K)); rng.shuffle(order); texts = [opts[i] for i in order]
    q  = {"type":"choice","instructions":INSTR,"criteria":{k:t for k,t in zip(OPT_KEYS,texts)}}
    gk = OPT_KEYS[texts.index(gold_text)]
    a0,_ = ask(row.idiom, q); u0 = unpack(a0)
    p0 = [u0["probs"][k] for k in OPT_KEYS]
    for vi, alt in enumerate(row.alt_clean[:2]):
        if not isinstance(alt, str) or not alt.strip(): continue
        a1,_ = ask(alt, q); u1 = unpack(a1)
        p1 = [u1["probs"][k] for k in OPT_KEYS]
        f.write(json.dumps({"id": int(row.id), "variant_i": vi, "idiom": row.idiom,
            "alt": alt, "gold_key": gk, "pred_canon": u0["pred"], "pred_alt": u1["pred"],
            "agree": int(u0["pred"]==u1["pred"]),
            "correct_canon": int(u0["pred"]==gk), "correct_alt": int(u1["pred"]==gk),
            "jsd": jsd(p0,p1), "p_gold_canon": u0["probs"][gk], "p_gold_alt": u1["probs"][gk],
            "maxp_canon": u0["max_prob"], "maxp_alt": u1["max_prob"],
            "confhead_canon": u0["conf_head"], "confhead_alt": u1["conf_head"]},
            ensure_ascii=False) + "\n")
    f.flush()
f.close()

E2 = pd.read_json(JL, lines=True).drop_duplicates(["id","variant_i"], keep="last")
E2.to_csv(R("e2_invariance","raw.csv"), index=False, encoding="utf-8")
pd.DataFrame([{"n_pairs": len(E2), "agreement": E2.agree.mean(),
    "acc_canonical": E2.correct_canon.mean(), "acc_alt": E2.correct_alt.mean(),
    "both_correct": ((E2.correct_canon==1)&(E2.correct_alt==1)).mean(),
    "mean_jsd": E2.jsd.mean(), "median_jsd": E2.jsd.median(),
    "mean_dP_gold": (E2.p_gold_canon-E2.p_gold_alt).abs().mean()}]
           ).to_csv(R("e2_invariance","summary.csv"), index=False)
print(pd.read_csv(R("e2_invariance","summary.csv")).to_string(index=False))
```

---

## 11 — E3: unit integrity

```python
# ── 11 ── perturb the idiom, hold the options fixed ────────────────────────
import random, json, numpy as np, pandas as pd
from tqdm.auto import tqdm

samp  = SUB[SUB.idiom.map(lambda s: len(words(s)) >= 3)].reset_index(drop=True)
VOCAB = sorted({w for s in full.idiom for w in words(s) if len(w) >= 2})  # full vocab
print(f"{len(samp)} items x 6 = {len(samp)*6:,} calls (~{len(samp)*6*SINGLE_MS/60000:.0f} min)")

def variants(idiom, example, rng):
    w = words(idiom); sh = w[:]
    for _ in range(8):
        rng.shuffle(sh)
        if sh != w: break
    dl = w[:]; di = rng.randrange(len(w)); dropped = dl.pop(di)
    sb = w[:]; si = rng.randrange(len(w)); sb[si] = rng.choice(VOCAB)
    return ({"V0_intact": idiom, "V1_shuffle": " ".join(sh),
             "V2_reverse": " ".join(reversed(w)), "V3_delete1": " ".join(dl),
             "V4_substitute1": " ".join(sb),
             "V5_in_sentence": example if example else idiom}, dropped, di, si)

JL   = R("e3_unit","raw.jsonl")
done = {json.loads(l)["id"] for l in open(JL, encoding="utf-8")} if JL.exists() else set()
f    = open(JL, "a", encoding="utf-8")

for row in tqdm(list(samp.itertuples()), desc="E3"):
    if row.id in done: continue
    rng  = random.Random(3000 + int(row.id))
    opts = [row.figurative_meaning_bn] + distractors(row, "D-hard", rng)
    opts = list(dict.fromkeys(opts))[:K]
    if len(opts) < K: continue
    gold_text = row.figurative_meaning_bn
    order = list(range(K)); rng.shuffle(order); texts = [opts[i] for i in order]
    q  = {"type":"choice","instructions":INSTR,"criteria":{k:t for k,t in zip(OPT_KEYS,texts)}}
    gk = OPT_KEYS[texts.index(gold_text)]
    vs, dropped, di, si = variants(row.idiom, row.ex_bn1, rng)
    base_p = None
    for vname, vtext in vs.items():
        a,_ = ask(vtext, q); u = unpack(a)
        p = [u["probs"][k] for k in OPT_KEYS]
        if vname == "V0_intact": base_p = p
        f.write(json.dumps({"id": int(row.id), "variant": vname, "text": vtext,
            "idiom": row.idiom, "gold_key": gk, "pred": u["pred"],
            "correct": int(u["pred"]==gk), "p_gold": u["probs"][gk],
            "max_prob": u["max_prob"], "conf_head": u["conf_head"],
            "jsd_vs_intact": float(jsd(base_p,p)) if base_p else 0.0,
            "n_words": len(words(row.idiom)), "dropped_word": dropped,
            "delete_pos": di, "sub_pos": si}, ensure_ascii=False) + "\n")
    f.flush()
f.close()

E3 = pd.read_json(JL, lines=True).drop_duplicates(["id","variant"], keep="last")
E3.to_csv(R("e3_unit","raw.csv"), index=False, encoding="utf-8")
first = E3.groupby("id").pred.transform(lambda s: s.iloc[0])
E3["kept_answer"] = (E3.pred == first).astype(int)
(E3.groupby("variant").agg(n=("correct","size"), accuracy=("correct","mean"),
    p_gold=("p_gold","mean"), jsd=("jsd_vs_intact","mean"),
    answer_retained=("kept_answer","mean"), max_prob=("max_prob","mean"))
   .reset_index().to_csv(R("e3_unit","summary.csv"), index=False))
(E3.groupby(["variant","n_words"]).agg(n=("correct","size"),
    accuracy=("correct","mean"), jsd=("jsd_vs_intact","mean")).reset_index()
   .to_csv(R("e3_unit","by_length.csv"), index=False))
print(pd.read_csv(R("e3_unit","summary.csv")).to_string(index=False))
```

---

## 12 — E4: Bangla vs English

```python
# ── 12 ── same item, Bangla vs its gold English equivalent ─────────────────
import random, json, numpy as np, pandas as pd
from tqdm.auto import tqdm

samp = SUB[(SUB.sim_en.str.len()>0) & (SUB.figurative_meaning_en.str.len()>0)
          ].reset_index(drop=True)
INSTR_EN = "What is the actual figurative meaning of this idiom?"
print(f"{len(samp)} x 2 = {len(samp)*2:,} calls (~{len(samp)*2*SINGLE_MS/60000:.0f} min)")

JL   = R("e4_language","raw.jsonl")
done = {json.loads(l)["id"] for l in open(JL, encoding="utf-8")} if JL.exists() else set()
f    = open(JL, "a", encoding="utf-8")

for row in tqdm(list(samp.itertuples()), desc="E4"):
    if row.id in done: continue
    rng  = random.Random(4000 + int(row.id))
    pool = _fill(list(ALL_IDS), K-1, {row.id}, rng)

    tb = [row.figurative_meaning_bn] + [MEAN[i] for i in pool]
    ob = list(range(K)); rng.shuffle(ob); tb = [tb[i] for i in ob]
    qb = {"type":"choice","instructions":INSTR,"criteria":{k:t for k,t in zip(OPT_KEYS,tb)}}
    gb = OPT_KEYS[tb.index(row.figurative_meaning_bn)]

    te = [row.figurative_meaning_en] + [MEAN_EN.get(i, MEAN[i]) for i in pool]
    oe = list(range(K)); rng.shuffle(oe); te = [te[i] for i in oe]
    qe = {"type":"choice","instructions":INSTR_EN,"criteria":{k:t for k,t in zip(OPT_KEYS,te)}}
    ge = OPT_KEYS[te.index(row.figurative_meaning_en)]

    rec = {"id": int(row.id), "idiom": row.idiom, "sim_en": row.sim_en}
    for name, (st, q, gk) in {"bn": (row.idiom, qb, gb), "en": (row.sim_en, qe, ge)}.items():
        a,_ = ask(st, q); u = unpack(a)
        rec[f"{name}_pred"]      = u["pred"]
        rec[f"{name}_correct"]   = int(u["pred"]==gk)
        rec[f"{name}_p_gold"]    = u["probs"][gk]
        rec[f"{name}_max_prob"]  = u["max_prob"]
        rec[f"{name}_conf_head"] = u["conf_head"]
    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
f.close()

E4 = pd.read_json(JL, lines=True).drop_duplicates("id", keep="last")
E4.to_csv(R("e4_language","raw.csv"), index=False, encoding="utf-8")
pd.DataFrame([{"arm": a, "accuracy": E4[f"{a}_correct"].mean(),
    "mean_p_gold": E4[f"{a}_p_gold"].mean(), "mean_max_prob": E4[f"{a}_max_prob"].mean(),
    "mean_conf_head": E4[f"{a}_conf_head"].mean()} for a in ["bn","en"]]
           ).to_csv(R("e4_language","summary.csv"), index=False)
E4["cell"] = np.select(
    [(E4.bn_correct==1)&(E4.en_correct==1), (E4.bn_correct==0)&(E4.en_correct==1),
     (E4.bn_correct==1)&(E4.en_correct==0)],
    ["both_right","bangla_gap","english_gap"], default="both_wrong")
E4.cell.value_counts(normalize=True).rename("share").reset_index().to_csv(
    R("e4_language","localisation.csv"), index=False)
print(pd.read_csv(R("e4_language","summary.csv")).to_string(index=False))
print("\n", E4.cell.value_counts(normalize=True).to_string())
```

---

## 13 — E5: calibration

```python
# ── 13 ── temperature, risk-coverage, head vs max-prob selective prediction ─
import numpy as np, pandas as pd, json
from scipy.optimize import minimize_scalar

E1 = pd.read_csv(R("e1_ladder","raw.csv"))
P  = [f"p_{k}" for k in OPT_KEYS]

def temper(pr, T):
    lg = np.log(np.clip(pr,1e-12,1))/T
    e = np.exp(lg - lg.max(axis=1, keepdims=True)); return e/e.sum(axis=1, keepdims=True)
def nll(T, pr, gi):
    q = temper(pr, T); return float(-np.log(np.clip(q[np.arange(len(q)),gi],1e-12,1)).mean())
def ece(c, y, bins=15):
    c,y = np.asarray(c,float), np.asarray(y,float); ed=np.linspace(0,1,bins+1); e=0.
    for lo,hi in zip(ed[:-1],ed[1:]):
        m=(c>lo)&(c<=hi)
        if m.sum(): e += m.mean()*abs(y[m].mean()-c[m].mean())
    return float(e)
def aurc(score, correct):
    o = np.argsort(-np.asarray(score,float)); y = np.asarray(correct,float)[o]
    return float((1 - np.cumsum(y)/np.arange(1,len(y)+1)).mean())

rows, curves = [], []
for fam, g in E1.groupby("family"):
    pr = g[P].values
    gi = g.gold_key.map({k:i for i,k in enumerate(OPT_KEYS)}).values
    corr, mp, ch = g.correct.values, pr.max(axis=1), g.conf_head.values
    rs = np.random.RandomState(5); m = rs.rand(len(g)) < 0.5
    T = minimize_scalar(lambda t: nll(t, pr[m], gi[m]), bounds=(0.25,8.0), method="bounded").x
    q = temper(pr[~m], T); mpT = q.max(axis=1)
    corrT = (np.array(OPT_KEYS)[q.argmax(axis=1)] == g.gold_key.values[~m]).astype(int)
    rows.append({"family": fam, "T_fitted": float(T),
        "ece_maxprob_raw": ece(mp[~m], corr[~m]), "ece_maxprob_tempered": ece(mpT, corrT),
        "ece_confhead": ece(ch[~m], corr[~m]) if not np.isnan(ch).all() else np.nan,
        "aurc_maxprob": aurc(mp, corr),
        "aurc_confhead": aurc(np.nan_to_num(ch, nan=0.0), corr),
        "acc_raw": corr[~m].mean(), "acc_tempered": corrT.mean()})
    for sig, nm in [(mp,"max_prob"), (ch,"conf_head")]:
        if np.isnan(sig).all(): continue
        o = np.argsort(-np.nan_to_num(sig, nan=-1)); cc = np.cumsum(corr[o])/np.arange(1,len(o)+1)
        for i in range(0, len(o), max(1, len(o)//100)):
            curves.append({"family": fam, "signal": nm, "coverage": (i+1)/len(o),
                           "accuracy": cc[i]})
pd.DataFrame(rows).to_csv(R("e5_calibration","temperature.csv"), index=False)
pd.DataFrame(curves).to_csv(R("e5_calibration","risk_coverage.csv"), index=False)

lit = E1[E1.family.isin(["D-lit","D-hard"])].dropna(subset=["chose_literal"])
pd.DataFrame([{"group": gn, "n": len(d), "mean_max_prob": d.max_prob.mean(),
    "mean_conf_head": d.conf_head.mean(), "mean_p_gold": d.p_gold.mean(),
    "mean_p_literal": d.p_literal.mean()}
    for gn, d in [("correct", lit[lit.correct==1]),
                  ("chose_literal", lit[lit.chose_literal==1]),
                  ("other_wrong", lit[(lit.correct==0)&(lit.chose_literal==0)])]]
           ).to_csv(R("e5_calibration","confidence_by_outcome.csv"), index=False)

rel = []
for fam, g in E1.groupby("family"):
    for sig, nm in [(g.max_prob.values,"max_prob"), (g.conf_head.values,"conf_head")]:
        if np.isnan(sig).all(): continue
        ed = np.linspace(0,1,11)
        for lo,hi in zip(ed[:-1],ed[1:]):
            m = (sig>lo)&(sig<=hi)
            rel.append({"family":fam,"signal":nm,"bin_lo":lo,"bin_hi":hi,"n":int(m.sum()),
                "mean_conf": float(sig[m].mean()) if m.sum() else np.nan,
                "accuracy": float(g.correct.values[m].mean()) if m.sum() else np.nan})
pd.DataFrame(rel).to_csv(R("e5_calibration","reliability.csv"), index=False)
print(pd.read_csv(R("e5_calibration","temperature.csv")).to_string(index=False))
```

---

## 14 — E6: option-side attribution

```python
# ── 14 ── perturb the OPTION text, hold state and gold fixed ───────────────
import random, json, numpy as np, pandas as pd
from tqdm.auto import tqdm

samp = SUB[SUB.figurative_meaning_en.str.len()>0].reset_index(drop=True)
print(f"{len(samp)} x 5 = {len(samp)*5:,} calls (~{len(samp)*5*SINGLE_MS/60000:.0f} min)")

def trunc(t, n=6): return " ".join(words(t)[:n])
def scram(t, rng):
    w = words(t); rng.shuffle(w); return " ".join(w)

JL   = R("e6_options","raw.jsonl")
done = {json.loads(l)["id"] for l in open(JL, encoding="utf-8")} if JL.exists() else set()
f    = open(JL, "a", encoding="utf-8")

for row in tqdm(list(samp.itertuples()), desc="E6"):
    if row.id in done: continue
    rng  = random.Random(6000 + int(row.id))
    pool = _fill(list(ALL_IDS), K-1, {row.id}, rng)
    ids  = [row.id] + pool
    order = list(range(K)); rng.shuffle(order); ids = [ids[i] for i in order]
    gk = OPT_KEYS[ids.index(row.id)]
    forms = {"O0_full_bn":   [MEAN[i] for i in ids],
             "O1_trunc6":    [trunc(MEAN[i]) for i in ids],
             "O2_scrambled": [scram(MEAN[i], rng) for i in ids],
             "O3_english":   [MEAN_EN.get(i, MEAN[i]) for i in ids],
             "O4_labels":    [f"অর্থ {j+1}" for j in range(K)]}
    base_p = None
    for fname, texts in forms.items():
        q = {"type":"choice","instructions":INSTR,"criteria":{k:t for k,t in zip(OPT_KEYS,texts)}}
        a, usage = ask(row.idiom, q); u = unpack(a)
        p = [u["probs"][k] for k in OPT_KEYS]
        if fname == "O0_full_bn": base_p = p
        f.write(json.dumps({"id": int(row.id), "form": fname, "gold_key": gk,
            "pred": u["pred"], "correct": int(u["pred"]==gk), "p_gold": u["probs"][gk],
            "max_prob": u["max_prob"], "conf_head": u["conf_head"],
            "jsd_vs_full": float(jsd(base_p,p)) if base_p else 0.0,
            "opt_tokens": sum(len(words(t)) for t in texts),
            "input_tokens": usage.get("input_tokens")}, ensure_ascii=False) + "\n")
    f.flush()
f.close()

E6 = pd.read_json(JL, lines=True).drop_duplicates(["id","form"], keep="last")
E6.to_csv(R("e6_options","raw.csv"), index=False, encoding="utf-8")
(E6.groupby("form").agg(n=("correct","size"), accuracy=("correct","mean"),
    p_gold=("p_gold","mean"), max_prob=("max_prob","mean"),
    conf_head=("conf_head","mean"), jsd_vs_full=("jsd_vs_full","mean"),
    opt_tokens=("opt_tokens","mean")).reset_index()
   .to_csv(R("e6_options","summary.csv"), index=False))
print(pd.read_csv(R("e6_options","summary.csv")).to_string(index=False))
```

---

## 15 — Manifest

```python
# ── 15 ── inventory, with everything a later cross-model pass needs ────────
import json, hashlib
def sha(p):
    h = hashlib.sha256()
    with open(p,"rb") as fh:
        while (b := fh.read(1<<20)): h.update(b)
    return h.hexdigest()[:16]

arte = sorted([*(ROOT/"results"/MODEL).rglob("*.csv"),
               *(ROOT/"results"/MODEL).rglob("*.json*"),
               *(ROOT/"data/splits").rglob("*.json*"),
               *(ROOT/"configs").rglob("*.json")])
json.dump({**ENV, "repo": REPO, "dtype": str(DTYPE),
           "n_full": int(len(full)), "n_subset": int(len(SUB)),
           "subset_seed": 909, "subset_frac": 0.10,
           "k": K, "option_keys": OPT_KEYS, "families": FAMILIES,
           "instructions": INSTR,
           "seeds": {"e1":1000,"e2":2000,"e3":3000,"e4":4000,"e6":6000,"e7":7000},
           "artifacts":[{"path": str(a.relative_to(ROOT)), "bytes": a.stat().st_size,
                         "sha256_16": sha(a)} for a in arte]},
          open(R(f"manifest_{RUN}.json"),"w"), indent=2, default=str, ensure_ascii=False)
for a in arte: print(f"{a.stat().st_size:>10,}  {a.relative_to(ROOT)}")
print(f"\n{len(arte)} artefacts")
```

---

## What lands on disk

```
BNFDM/
├── configs/task.json
├── data/
│   ├── bagdhara_raw.jsonl
│   └── splits/  full.jsonl · sub10.jsonl · meta.json
└── results/lod-lille-0.6b/
    ├── env.json · bench.csv · manifest_*.json
    ├── e7_order/        raw · per_item · by_gold_position · selection_bias
    ├── e1_ladder/       raw · summary · by_{frequency,scape,hist_sig,relig_sig,n_words,gold_pos}
    ├── e2_invariance/   raw · summary
    ├── e3_unit/         raw · summary · by_length
    ├── e4_language/     raw · summary · localisation
    ├── e5_calibration/  temperature · risk_coverage · confidence_by_outcome · reliability
    └── e6_options/      raw · summary
```

For the repo, push `results/lod-lille-0.6b/` plus `data/splits/sub10.jsonl` and `configs/task.json`. The Laya folder goes in as `results/laya-ml/`. The cross-model pass then joins on `(id, family)` — `sub10.jsonl` tells it which ids to restrict the Laya side to, and `manifest_*.json` carries every constant needed to verify the two arms were run identically.

## Clearance

Run 01 → 07, then check:

1. **Cell 03** prints `n_usable` near 10,300 and a subset near 1,030, with subset and full metadata distributions tracking closely.
2. **Cell 04**'s parity probe prints an idiom with four options. Save that output — it's the proof the two arms saw identical questions.
3. **Cell 05** loads and prints dtype `float32`, device `cuda`, `temperature: 1.0`.
4. **Cell 06**'s assertions pass; it prints both `max_prob` and `conf_head` and `state_truncated` is False.
5. **Cell 07**'s suite estimate is tolerable. Past ~4 hours, drop E7 to 150 items and E6 to half the subset.

Then 08 → 15 straight through. Every heavy cell appends JSONL and skips completed ids, so a disconnect costs one item — rerun the cell.

Two outcomes to expect, both reportable: E7 should return ~0 instability (order independence by construction), and Bangla may sit near chance in E4 while English does not (the card says English-only training).
