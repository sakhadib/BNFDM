# BNFDM — Bangla Figurative Decision-Model diagnostics

Behavioural diagnostics for *decision models* — non-autoregressive, option-scoring
models that return calibrated probabilities over a caller-supplied option set and
never generate text — applied to Bangla idioms (বাগধারা).

Because these models cannot explain themselves (no chain of thought, no
self-rationalisation), every diagnostic here is an **intervention on the input** with
a gold counterfactual already present in the resource. Nothing depends on a model's
self-report.

Two architecturally unrelated models, run on byte-identical questions:

| | Laya-multilingual | Lod-lille |
|---|---|---|
| checkpoint | `convaiinnovations/laya` · `multilingual` | `mrn-dk/lod-lille-0.6B` |
| architecture | mmBERT encoder + option-marker scorer | Qwen3-0.6B + independent-option scoring |
| params | 322M | 0.6B |
| items | 8,876 (full) | 888 (fixed 10%) |

**Headline.** Both models answer a 4-way meaning-choice question well above chance
against random distractors (0.367 / 0.484). Swap one distractor for the idiom's own
expert-annotated **literal** gloss and both fall **below the 0.25 chance floor**
(0.151 / 0.066), taking the literal reading **64.4% / 88.5%** of the time — and both
are *more confident* when captured than when correct. The stronger model is the more
captured one.

→ **[FINDINGS.md](FINDINGS.md)** for the full write-up.

## Four claims

1. **Literal capture** is severe, measurable without annotation, and does not improve
   with capability.
2. **Calibration claims do not survive**, under either philosophy — fitted
   temperatures (Laya) or deliberately unfitted (Lod). Gating on confidence is worse
   than not gating.
3. **Option-order dependence dissociates cleanly**: 83.7% of Laya's items change
   answer under reordering versus 10.7% of Lod's, a 12× gap in probability stability.
   Independent-option scoring delivers what it claims; option-marker scoring does not.
4. **Failure localises to Bangla**, not to figurativeness: +0.246 / +0.380 from
   swapping language alone on the same model.

## Layout

```
BNFDM/
├── FINDINGS.md                      write-up: 8 result sections, limitations
├── LAYA_RUN/  results/laya-ml/      full set, n = 8,876
├── LOD_RUN/   results/lod-lille-0.6b/  10% subset, n = 888
│              data/splits/          full · sub10 · meta (shared contract)
└── analysis/
    ├── bnfdm_style.py               CMYK palette, ACL style, bootstrap/ECE/AURC
    ├── analyze.py                   raws -> tables/<arm>/*.csv
    ├── crossmodel.py                pairing check, McNemar, subset check
    ├── tables_tex.py                -> tables/tables.tex
    ├── figures.py                   -> figures/*.pdf
    ├── tables/   <arm>/ and _cross/
    └── figures/  11 figures; each .pdf has a .png preview and a .csv of
                  exactly the data plotted
```

Reproduce:

```bash
python3 analysis/analyze.py      # per-arm tables
python3 analysis/crossmodel.py   # pairing verification + paired tests
python3 analysis/tables_tex.py   # LaTeX
python3 analysis/figures.py      # figures
```

Needs `pandas`, `numpy`, `scipy`, `matplotlib`. No GPU, no model download — the
analysis runs entirely off the committed artifacts.

## The instrument

Given a Bangla idiom, choose its gold `figurative_meaning_bn` from k = 4 candidates.
Exactly scored, no rubric, no human judgment — which turns the generative benchmark
of the [LREC 2026 Bangla Bagdhara paper](https://aclanthology.org/2026.lrec-1.546.pdf)
into a discriminative one. The **distractor family is the independent variable**;
item, instruction, k and per-item option-order randomisation are held fixed.

| Family | Distractors drawn from | Isolates |
|---|---|---|
| `D-rand` | figurative glosses of random other idioms | ceiling |
| `D-surf` | glosses of idioms sharing surface tokens | surface reliance |
| `D-sem` | glosses of idioms sharing `tags` | semantic granularity |
| `D-lit` | **the idiom's own `literal_meaning`** + random | **literal attraction** |
| `D-hard` | literal + surface + tag combined | compound |

Seven experiments per arm: **E1** distractor ladder · **E2** surface invariance via
`alternative_idioms` · **E3** unit integrity under shuffle/delete/substitute ·
**E4** cross-lingual failure localisation · **E5** calibration and selective
prediction · **E6** option-side attribution · **E7** option-order sensitivity.

### Pairing

The arms were run in different accounts from separate notebooks, each rebuilding the
corpus and the task instrument from a hardcoded contract (same dataset,
dedup/filter pipeline, k, option keys, instruction string, families and per-item
seeds). `crossmodel.py` verifies this empirically before reporting any paired test:
on all 4,440 overlapping rows the gold key, gold position, idiom and literal-key
match at **1.000**. A failure there is fatal, not cosmetic.

## Figures

| | |
|---|---|
| `f1_distractor_ladder` | accuracy across the ladder, both arms; LAR |
| `f2_literal_capture` | outcome decomposition; p(gold) vs p(literal) |
| `f3_confidence_anti_diagnostic` | confidence by outcome; inverted reliability |
| `f4_position_bias` | selection share by slot; accuracy by gold slot |
| `f5_order_instability` | distinct glosses chosen across 24 orderings |
| `f6_crosslingual` | six input/checkpoint arms; failure localisation |
| `f7_unit_integrity` | accuracy and agreement under perturbation |
| `f8_option_ablation` | option-text ablation; semantic signal retained |
| `f9_selective_prediction` | risk-coverage; AURC vs a random gate |
| `f10_cross_model` | paired McNemar deltas; four claims side by side |
| `f11_stratified_and_subset` | by idiom length; 10% subset representativeness |

No titles (captions belong in the paper), ACL single/double column widths, embedded
TrueType, CMYK palette on white. Laya is drawn in cyan, Lod in magenta throughout.

**No Bangla glyphs in any figure** — matplotlib does not use HarfBuzz and cannot
shape Bengali, and no Bengali font was available on the analysis machine. Every
figure is quantitative by design; qualitative examples live in `FINDINGS.md`, which
renders Bangla natively.

## Data

[Bangla Bagdhara](https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara) — 10,361 published rows (10,359 on load, 8,876 usable); Sakhawat et al., LREC 2026, doi:10.63317/546w2cys6m6t
idioms under a 19-field schema, expert-validated by consensus of ≥2 Bangla
linguists. After deduplication on the idiom string and requiring both a literal and a
figurative gloss: **8,876 items**.

Sakhawat et al., *When Words Don't Mean What They Say: Figurative Understanding in
Bengali Idioms*, LREC 2026.
