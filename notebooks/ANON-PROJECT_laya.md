# BnFiDM — Laya experiments (data collection only)

Eleven cells, copy-paste in order. Laya multilingual only. No plotting, no other backends — every experiment dumps CSV/JSONL so figures can be built later from the saved files.

**Why the last run crashed:** `pip install -U pandas pyarrow` upgraded pyarrow under an already-imported pyarrow, breaking the C extension (`Array size changed, Expected 80 ... got 72`). This file installs only `laya` and `kagglehub`, touches no preinstalled scientific package, and uses JSONL instead of parquet — which also preserves the list-valued columns that CSV would flatten.

---

## 01 — Paths

```python
# ── 01 ── project root ──────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, json, sys, platform, subprocess, datetime as dt
from pathlib import Path

ROOT     = Path("/content/drive/MyDrive/Articles/Research/BnFiDM")
MODEL    = "laya-ml"                       # namespaces results/
RUN_ID   = dt.datetime.now().strftime("%Y%m%d-%H%M")

for d in ["data", "data/splits", "configs", "results", "logs", "cache/hf"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(ROOT / "cache/hf")   # survive runtime restarts

def R(*parts) -> Path:
    q = ROOT / "results" / MODEL / Path(*parts)
    q.parent.mkdir(parents=True, exist_ok=True); return q

def D(*parts) -> Path:
    q = ROOT / "data" / Path(*parts)
    q.parent.mkdir(parents=True, exist_ok=True); return q

ENV = {"run_id": RUN_ID, "model": MODEL, "python": platform.python_version()}
try:
    ENV["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    ).decode().strip()
except Exception:
    ENV["gpu"] = "cpu"
print(json.dumps(ENV, indent=2)); print("root:", ROOT)
```

---

## 02 — Install (minimal, no upgrades)

```python
# ── 02 ── ONLY what is missing. Never -U a preinstalled package in Colab. ───
import sys, subprocess
subprocess.run(f"{sys.executable} -m pip -q install laya kagglehub",
               shell=True, check=False)

import laya, pandas as pd, numpy as np
print("laya  :", laya.__version__)
print("pandas:", pd.__version__)
print("numpy :", np.__version__)
```

If this cell ever prints a pyarrow or numpy ABI error, **Runtime → Restart session** and rerun from 01. Do not add `-U`.

---

## 03 — Dataset

```python
# ── 03 ── bangla-bagdhara: load, normalise, dedupe, split. JSONL throughout. ─
import os, json, unicodedata, numpy as np, pandas as pd, kagglehub

RAW = D("bagdhara_raw.jsonl")

if RAW.exists():
    df = pd.read_json(RAW, lines=True)
    print("cached:", df.shape)
else:
    path  = kagglehub.dataset_download("sakhadib/bangla-bagdhara")
    files = os.listdir(path)
    jl, js = [f for f in files if f.endswith(".jsonl")], [f for f in files if f.endswith(".json")]
    df = pd.read_json(os.path.join(path, (jl or js)[0]), lines=bool(jl))
    df.to_json(RAW, orient="records", lines=True, force_ascii=False)
    print("downloaded:", df.shape)

def nfc(x):
    if isinstance(x, str):  return unicodedata.normalize("NFC", x).strip()
    if isinstance(x, list): return [nfc(v) for v in x]
    return x

for c in ["idiom", "literal_meaning", "figurative_meaning_bn", "figurative_meaning_en",
          "note", "example_sentences_in_bangla", "example_sentences_in_english",
          "alternative_idioms", "similar_in_english", "history"]:
    if c in df.columns: df[c] = df[c].map(nfc)

def join(v):
    if isinstance(v, list): return " ".join(map(str, v))
    return "" if pd.isna(v) else str(v)

df["ex_bn"] = df["example_sentences_in_bangla"].map(join)

LABELS = ["negative", "neutral", "positive", "mixed"]
n0 = len(df)
dedup = (df.sort_values("id").drop_duplicates(subset=["idiom"], keep="first")
           .query("sentiment in @LABELS").reset_index(drop=True))
print(f"dedupe: {n0} -> {len(dedup)}")

from sklearn.model_selection import train_test_split
SEED = 20261001
dev, test = train_test_split(dedup, test_size=0.5, random_state=SEED,
                             stratify=dedup.sentiment)
probe, _  = train_test_split(dev, train_size=400, random_state=SEED,
                             stratify=dev.sentiment)

for name, part in [("probe_400", probe), ("dev", dev), ("test", test), ("full", dedup)]:
    part.reset_index(drop=True).to_json(D("splits", f"{name}.jsonl"),
                                        orient="records", lines=True, force_ascii=False)
    print(f"{name:<10} {len(part):>6}", dict(part.sentiment.value_counts()))

MAJORITY = float(dedup.sentiment.value_counts(normalize=True).iloc[0])
json.dump({"majority_baseline": MAJORITY, "n_raw": n0, "n_dedup": len(dedup),
           "seed": SEED, "labels": LABELS,
           "class_prior": dedup.sentiment.value_counts(normalize=True).to_dict()},
          open(D("splits", "meta.json"), "w"), indent=2)
print(f"\nmajority baseline: {MAJORITY:.4f}")
```

---

## 04 — Question and state conditions

```python
# ── 04 ── sentiment question, four Bangla-only state conditions ─────────────
import json

SENTIMENT_Q = {
    "type": "choice",
    "instructions": "এই বাংলা বাগধারাটি যে অনুভূতি বা মনোভাব প্রকাশ করে তা কোনটি?",
    "criteria": {
        "negative": "নিন্দা, ক্ষতি, দুঃখ, রাগ, প্রতারণা, ব্যর্থতা বা অপমান বোঝায়",
        "neutral":  "বর্ণনামূলক বা তথ্যগত, স্পষ্ট প্রশংসা বা নিন্দা নেই",
        "positive": "প্রশংসা, সাফল্য, আনন্দ, গুণ বা সৌভাগ্য বোঝায়",
        "mixed":    "একই সঙ্গে ইতিবাচক ও নেতিবাচক দিক আছে, বা প্রসঙ্গভেদে বদলায়",
    },
}
LABELS = ["negative", "neutral", "positive", "mixed"]

CONDS = {
    "S1_idiom":        lambda r: r["idiom"],
    "S2_idiom_lit":    lambda r: f"বাগধারা: {r['idiom']}\nআক্ষরিক অর্থ: {r['literal_meaning']}",
    "S3_idiom_fig":    lambda r: f"বাগধারা: {r['idiom']}\nভাবার্থ: {r['figurative_meaning_bn']}",
    "S4_idiom_fig_ex": lambda r: (f"বাগধারা: {r['idiom']}\nভাবার্থ: {r['figurative_meaning_bn']}"
                                  f"\nউদাহরণ: {r['ex_bn']}"),
}

json.dump({"id": "sentiment_v1", "labels": LABELS, "question": SENTIMENT_Q,
           "conditions": list(CONDS)},
          open(ROOT / "configs/sentiment_v1.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(SENTIMENT_Q, ensure_ascii=False, indent=2))
```

---

## 05 — Load Laya and check the contract

```python
# ── 05 ── multilingual checkpoint, pinned. No Router: a state carrying any ──
#          English field detects as latin and would hit the English checkpoint,
#          documented at 0.000 accuracy / 0.952 confidence on Bengali.
import laya, json, pandas as pd

agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")
print("cfg:", {k: agent.cfg.get(k) for k in ("max_len", "head_max_len", "amp_dtype")})

def decide(state, q=SENTIMENT_Q, qid="q", **kw):
    return agent.predict(state, {qid: q}, **kw)["answers"][qid]

def decide_batch(states, q=SENTIMENT_Q, qid="q", batch_size=None, **kw):
    rs = agent.predict_batch(list(states), {qid: q},
                             batch_size=batch_size or globals().get("BATCH", 32), **kw)
    return [r["answers"][qid] for r in rs]

probe = pd.read_json(D("splits", "probe_400.jsonl"), lines=True)
a = decide(CONDS["S3_idiom_fig"](probe.iloc[0]))
assert set(a["probabilities"]) == set(LABELS)
assert abs(sum(a["probabilities"].values()) - 1) < 1e-3
print("\nlabel :", a["choice"], "| gold:", probe.iloc[0]["sentiment"])
print("probs :", {k: round(v, 4) for k, v in a["probabilities"].items()})
print("conf  :", round(a["answer_confidence"], 4), "(top-mass; use this for ECE)")
print("entrop:", round(a["confidence"], 4), "(normalised entropy; NOT calibrated)")

json.dump({**ENV, "cfg": {k: agent.cfg.get(k) for k in
           ("max_len", "head_max_len", "amp_dtype", "temperature", "temperature_by_options")}},
          open(R("env.json"), "w"), indent=2, default=str)
```

---

## 06 — Batch sweep

```python
# ── 06 ── T4 is Turing: fp16, no bf16. Find the batch size that pays. ───────
import torch, time, pandas as pd

print("gpu :", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0),
      "| bf16", torch.cuda.is_bf16_supported())

probe  = pd.read_json(D("splits", "probe_400.jsonl"), lines=True)
states = [CONDS["S3_idiom_fig"](r) for _, r in probe.head(256).iterrows()]
decide_batch(states[:16], batch_size=16)                 # warm up

rows = []
for bs in (16, 32, 64, 128, 256):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    decide_batch(states, batch_size=bs)
    torch.cuda.synchronize(); dtms = (time.time() - t0) / len(states) * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    rows.append({"batch_size": bs, "ms_per_item": dtms, "peak_gb": peak})
    print(f"bs={bs:>3}  {dtms:5.1f} ms/item  peak {peak:.2f} GB")

bench = pd.DataFrame(rows); bench.to_csv(R("bench_batch.csv"), index=False)
BATCH = int(bench.loc[bench.ms_per_item.idxmin(), "batch_size"])
print(f"\nBATCH = {BATCH}")
```

---

## 07 — E0: state ablation (the gate)

```python
# ── 07 ── does Laya read Bangla figurative semantics zero-shot? ─────────────
import numpy as np, pandas as pd, json, time

probe = pd.read_json(D("splits", "probe_400.jsonl"), lines=True)
rows  = []

for cond, fn in CONDS.items():
    states = [fn(r) for _, r in probe.iterrows()]
    t0 = time.time(); answers = decide_batch(states); el = time.time() - t0
    for (_, r), a in zip(probe.iterrows(), answers):
        rows.append({"condition": cond, "id": int(r["id"]), "idiom": r["idiom"],
                     "gold": r["sentiment"], "pred": a["choice"],
                     "confidence": a["answer_confidence"],
                     "entropy_conf": a["confidence"],
                     **{f"p_{k}": a["probabilities"].get(k, np.nan) for k in LABELS}})
    acc = np.mean([a["choice"] == g for a, g in zip(answers, probe.sentiment)])
    print(f"{cond:<18} acc={acc:.4f}  ({el:.1f}s)")

abl = pd.DataFrame(rows)
abl.to_csv(R("e0_ablation", "predictions.csv"), index=False, encoding="utf-8")

from sklearn.metrics import f1_score, precision_recall_fscore_support

def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

MAJORITY = json.load(open(D("splits", "meta.json")))["majority_baseline"]
summ = []
for cond, g in abl.groupby("condition"):
    corr = (g.pred == g.gold).values
    p, r, f1, sup = precision_recall_fscore_support(g.gold, g.pred, labels=LABELS,
                                                    zero_division=0)
    summ.append({"condition": cond, "n": len(g), "accuracy": corr.mean(),
                 "macro_f1": f1_score(g.gold, g.pred, average="macro", zero_division=0),
                 "weighted_f1": f1_score(g.gold, g.pred, average="weighted", zero_division=0),
                 "ece": ece(g.confidence, corr), "mean_conf": g.confidence.mean(),
                 "majority_baseline": MAJORITY,
                 **{f"f1_{l}": v for l, v in zip(LABELS, f1)},
                 **{f"precision_{l}": v for l, v in zip(LABELS, p)},
                 **{f"recall_{l}": v for l, v in zip(LABELS, r)},
                 **{f"support_{l}": int(v) for l, v in zip(LABELS, sup)}})
summary = pd.DataFrame(summ).sort_values("condition")
summary.to_csv(R("e0_ablation", "summary.csv"), index=False)

# confusion counts, long form — enough to rebuild any confusion plot later
from sklearn.metrics import confusion_matrix
cm_rows = []
for cond, g in abl.groupby("condition"):
    cm = confusion_matrix(g.gold, g.pred, labels=LABELS)
    for i, gl in enumerate(LABELS):
        for j, pl in enumerate(LABELS):
            cm_rows.append({"condition": cond, "gold": gl, "pred": pl,
                            "count": int(cm[i, j]),
                            "row_rate": float(cm[i, j] / max(cm[i].sum(), 1))})
pd.DataFrame(cm_rows).to_csv(R("e0_ablation", "confusion.csv"), index=False)

# reliability bins, long form
rel = []
for cond, g in abl.groupby("condition"):
    corr, conf = (g.pred == g.gold).values.astype(float), g.confidence.values
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rel.append({"condition": cond, "bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                    "mean_conf": float(conf[m].mean()) if m.sum() else np.nan,
                    "accuracy":  float(corr[m].mean()) if m.sum() else np.nan})
pd.DataFrame(rel).to_csv(R("e0_ablation", "reliability.csv"), index=False)

print("\n", summary[["condition", "accuracy", "macro_f1", "ece", "mean_conf"]]
      .to_string(index=False))
best = summary.accuracy.max()
print(f"\nbest {best:.4f} vs majority {MAJORITY:.4f}  ->  "
      f"{'GATE PASS' if best > MAJORITY else 'GATE FAIL (explaining noise)'}")
```

---

## 08 — E5: criteria-order sensitivity

Independent of the gate. Run it either way.

```python
# ── 08 ── does permuting option order change the decision? (issue #779) ────
import itertools, numpy as np, pandas as pd
from tqdm.auto import tqdm

probe = pd.read_json(D("splits", "probe_400.jsonl"), lines=True)
sub   = probe.sample(150, random_state=7)
PERMS = list(itertools.permutations(LABELS))          # 24
rows  = []

for pi, perm in enumerate(tqdm(PERMS, desc="orderings")):
    q = {**SENTIMENT_Q, "criteria": {k: SENTIMENT_Q["criteria"][k] for k in perm}}
    states = [CONDS["S3_idiom_fig"](r) for _, r in sub.iterrows()]
    for (_, r), a in zip(sub.iterrows(), decide_batch(states, q)):
        rows.append({"perm_id": pi, "perm": "|".join(perm), "id": int(r["id"]),
                     "gold": r["sentiment"], "pred": a["choice"],
                     "confidence": a["answer_confidence"],
                     **{f"p_{k}": a["probabilities"].get(k, np.nan) for k in LABELS}})

order = pd.DataFrame(rows)
order.to_csv(R("e5_order", "permutations.csv"), index=False, encoding="utf-8")

per_item = order.groupby("id").agg(
    gold=("gold", "first"), n_distinct=("pred", "nunique"),
    modal=("pred", lambda s: s.mode().iat[0]),
    p_neg_mean=("p_negative", "mean"), p_neg_std=("p_negative", "std"),
    conf_mean=("confidence", "mean"),
    conf_range=("confidence", lambda s: s.max() - s.min())).reset_index()
per_item["unstable"] = per_item.n_distinct > 1
per_item.to_csv(R("e5_order", "per_item.csv"), index=False, encoding="utf-8")

by_perm = (order.assign(ok=order.pred == order.gold)
                .groupby(["perm_id", "perm"]).ok.mean()
                .reset_index().rename(columns={"ok": "accuracy"}))
by_perm.to_csv(R("e5_order", "by_permutation.csv"), index=False)

print(f"unstable items          : {per_item.unstable.mean():.1%} "
      f"({per_item.unstable.sum()}/{len(per_item)})")
print(f"mean distinct answers   : {per_item.n_distinct.mean():.2f} of 24")
print(f"accuracy across orders  : {by_perm.accuracy.min():.4f} – "
      f"{by_perm.accuracy.max():.4f}  (sd {by_perm.accuracy.std():.4f})")
print(f"mean sd of P(negative)  : {per_item.p_neg_std.mean():.4f}")
```

---

## 09 — E1: exact Shapley, occlusion, random control

```python
# ── 09 ── word-level attribution on idiom-only states ──────────────────────
import itertools, math, re, json, unicodedata, numpy as np, pandas as pd
from tqdm.auto import tqdm

MAX_WORDS  = 10          # 2^n per item per mode; 10 caps the tail at 1024
PER_BUCKET = 40
PUNCT      = "।,;:!?\"'()[]{}—–-"

def words_of(t):
    t = re.sub(f"[{re.escape(PUNCT)}]", " ", unicodedata.normalize("NFC", str(t)))
    return [w for w in t.split() if w]

def coalition(words, keep, mode="drop", filler="…"):
    if mode == "drop":
        return " ".join(w for w, k in zip(words, keep) if k)
    return " ".join(w if k else filler for w, k in zip(words, keep))

def v_of(words, masks, target, mode):
    states = [coalition(words, m, mode) or "…" for m in masks]
    return np.array([a["probabilities"].get(target, 0.0)
                     for a in decide_batch(states)])

def shapley(text, target, mode):
    w = words_of(text)[:MAX_WORDS]; n = len(w)
    masks = [tuple(bool(i >> b & 1) for b in range(n)) for i in range(1 << n)]
    V = dict(zip(masks, v_of(w, masks, target, mode)))
    phi = np.zeros(n)
    for i in range(n):
        for m in masks:
            if m[i]: continue
            s = sum(m)
            coef = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            mi = list(m); mi[i] = True
            phi[i] += coef * (V[tuple(mi)] - V[m])
    return w, phi, V[tuple([True]*n)], V[tuple([False]*n)]

def occlude(text, target, mode):
    w = words_of(text)[:MAX_WORDS]; n = len(w)
    masks = [tuple([True]*n)] + [tuple(j != i for j in range(n)) for i in range(n)]
    v = v_of(w, masks, target, mode)
    return w, v[0] - v[1:], v[0]

# ── stratified sample by word count ───────────────────────────────────────
probe = pd.read_json(D("splits", "probe_400.jsonl"), lines=True)
cand  = probe[probe.idiom.map(lambda s: 2 <= len(words_of(s)) <= MAX_WORDS)].copy()
cand["nw"] = cand.idiom.map(lambda s: len(words_of(s)))
sample = (cand.groupby("nw", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), PER_BUCKET), random_state=11))
              .reset_index(drop=True))
print(sample.nw.value_counts().sort_index().to_string())
print(f"\n{len(sample)} idioms  ~{int((2**sample.nw + sample.nw + 1).sum()*2):,} passes")

# ── resume-safe loop ──────────────────────────────────────────────────────
JW, JI = R("e1_attribution", "words.jsonl"), R("e1_attribution", "items.jsonl")
done = {json.loads(l)["id"] for l in open(JI, encoding="utf-8")} if JI.exists() else set()
if done: print(f"resuming, {len(done)} done")

rng = np.random.default_rng(3)
fw, fi = open(JW, "a", encoding="utf-8"), open(JI, "a", encoding="utf-8")
for _, r in tqdm(list(sample.iterrows()), desc="attribution"):
    if int(r["id"]) in done: continue
    text   = r["idiom"]
    target = decide(text)["choice"]
    for mode in ("drop", "mask"):
        w, phi, full, empty = shapley(text, target, mode)
        _, occ, _ = occlude(text, target, mode)
        rnd = rng.normal(size=len(w))
        for k, word in enumerate(w):
            fw.write(json.dumps({"id": int(r["id"]), "mode": mode, "pos": k,
                "word": word, "idiom": text, "shapley": float(phi[k]),
                "occlusion": float(occ[k]), "random": float(rnd[k]),
                "target": target, "gold": r["sentiment"]}, ensure_ascii=False) + "\n")
        fi.write(json.dumps({"id": int(r["id"]), "mode": mode, "idiom": text,
            "n_words": len(w), "target": target, "gold": r["sentiment"],
            "p_full": float(full), "p_empty": float(empty),
            "shapley_sum": float(phi.sum()),
            "efficiency_gap": float(full - empty - phi.sum()),
            "spearman_shap_occ": float(pd.Series(phi).corr(pd.Series(occ),
                                                           method="spearman"))
        }, ensure_ascii=False) + "\n")
    fw.flush(); fi.flush()
fw.close(); fi.close()

items = pd.read_json(JI, lines=True).drop_duplicates(["id", "mode"], keep="last")
wrds  = pd.read_json(JW, lines=True).drop_duplicates(["id", "mode", "pos"], keep="last")
items.to_csv(R("e1_attribution", "items.csv"), index=False, encoding="utf-8")
wrds.to_csv(R("e1_attribution", "word_attributions.csv"), index=False, encoding="utf-8")
print(f"\n{len(items)} item-rows, {len(wrds)} word-rows")
print(items.groupby("mode")[["efficiency_gap", "spearman_shap_occ"]].mean())
```

---

## 10 — E1b: faithfulness curves

```python
# ── 10 ── deletion / insertion / comprehensiveness / sufficiency ───────────
import numpy as np, pandas as pd, json
from tqdm.auto import tqdm

W = pd.read_csv(R("e1_attribution", "word_attributions.csv"))
MODE, METHODS = "drop", ["shapley", "occlusion", "random"]
JC = R("e1_attribution", "faith_curves.jsonl")
done = {json.loads(l)["id"] for l in open(JC, encoding="utf-8")} if JC.exists() else set()

fc = open(JC, "a", encoding="utf-8")
for iid, g in tqdm(list(W[W["mode"] == MODE].groupby("id")), desc="faithfulness"):
    if iid in done: continue
    g = g.sort_values("pos")
    w, target, n = list(g.word), g.target.iat[0], len(g)
    for meth in METHODS:
        rank = np.argsort(-g[meth].values)
        dm = [tuple(j not in set(rank[:k]) for j in range(n)) for k in range(n + 1)]
        im = [tuple(j in     set(rank[:k]) for j in range(n)) for k in range(n + 1)]
        dv, iv = v_of(w, dm, target, MODE), v_of(w, im, target, MODE)
        frac = np.arange(n + 1) / n
        k = max(1, int(round(0.25 * n)))
        fc.write(json.dumps({"id": int(iid), "method": meth, "n_words": n,
            "frac": frac.tolist(), "deletion": dv.tolist(), "insertion": iv.tolist(),
            "del_auc": float(np.trapz(dv, frac)), "ins_auc": float(np.trapz(iv, frac)),
            "comprehensiveness": float(dv[0] - dv[k]),
            "sufficiency": float(dv[0] - iv[k])}) + "\n")
    fc.flush()
fc.close()

C = pd.read_json(JC, lines=True).drop_duplicates(["id", "method"], keep="last")
C[["id", "method", "n_words", "del_auc", "ins_auc",
   "comprehensiveness", "sufficiency"]].to_csv(
    R("e1_attribution", "faith_summary.csv"), index=False)

long = [{"id": r.id, "method": r.method, "frac": f, "deletion": d, "insertion": i}
        for r in C.itertuples()
        for f, d, i in zip(r.frac, r.deletion, r.insertion)]
pd.DataFrame(long).to_csv(R("e1_attribution", "faith_curves.csv"), index=False)

print(C.groupby("method")[["del_auc", "ins_auc", "comprehensiveness", "sufficiency"]]
       .agg(["mean", "std"]).round(4).to_string())
```

---

## 11 — Manifest

```python
# ── 11 ── inventory every artefact ─────────────────────────────────────────
import json, hashlib
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(1 << 20)): h.update(b)
    return h.hexdigest()[:16]

arte = sorted([*(ROOT / "results" / MODEL).rglob("*.csv"),
               *(ROOT / "results" / MODEL).rglob("*.json*"),
               *(ROOT / "data").rglob("*.json*")])
man = {**ENV, "majority_baseline": MAJORITY,
       "artifacts": [{"path": str(a.relative_to(ROOT)), "bytes": a.stat().st_size,
                      "sha256_16": sha(a)} for a in arte]}
json.dump(man, open(R(f"manifest_{RUN_ID}.json"), "w"), indent=2, default=str)
for a in arte: print(f"{a.stat().st_size:>10,}  {a.relative_to(ROOT)}")
print(f"\n{len(arte)} artefacts")
```

---

## Files this produces

```
BnFiDM/
├── configs/sentiment_v1.json
├── data/
│   ├── bagdhara_raw.jsonl
│   └── splits/  probe_400 · dev · test · full · meta.json
└── results/laya-ml/
    ├── env.json  bench_batch.csv  manifest_*.json
    ├── e0_ablation/   predictions.csv  summary.csv  confusion.csv  reliability.csv
    ├── e5_order/      permutations.csv  per_item.csv  by_permutation.csv
    └── e1_attribution/ items.csv  word_attributions.csv
                        faith_summary.csv  faith_curves.csv
                        words.jsonl  items.jsonl  faith_curves.jsonl  (resume state)
```

Every number any future figure could need is in those CSVs — per-item predictions with full probability vectors, long-form confusion counts, pre-binned reliability, per-permutation accuracy, per-word attributions under two masking modes and three methods, and both raw curves and AUC summaries. Plotting is a separate pass over these files; nothing above needs rerunning for it.

Cells 09 and 10 write JSONL as they go and skip completed ids, so a Colab disconnect costs one item. Rerun the cell to resume.
