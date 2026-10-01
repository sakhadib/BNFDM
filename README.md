# BNFDM — Bangla Figurative Decision-Model diagnostics

Behavioural diagnostics for *decision models* — non-autoregressive, option-scoring
models that return calibrated probabilities over a caller-supplied option set and
never generate text — applied to Bangla idioms (বাগধারা).

Because these models cannot explain themselves (no chain of thought, no
self-rationalisation), every diagnostic here is an **intervention on the input** with
a gold counterfactual already present in the resource. Nothing depends on a model's
self-report.

**Headline:** Laya-multilingual answers a 4-way meaning-choice question correctly
36.7% of the time against random distractors. Swap one distractor for the idiom's own
expert-annotated *literal* gloss and accuracy falls to **15.1% — below the 25% chance
floor** — because the model takes the literal reading **64.4%** of the time, and is
*more confident* when it does so than when it is right.

→ **[FINDINGS.md](FINDINGS.md)** for the full write-up.

## Layout

```
BNFDM/
├── FINDINGS.md                 write-up: 8 result sections, limitations
├── LAYA_RUN/                   raw experiment output (Colab, Tesla T4)
│   ├── configs/task.json       the task instrument
│   ├── data/splits/            dev · test · probe_400 · meta
│   └── results/laya-ml/        e1_ladder, e2_invariance, e3_unit,
│                               e4_crosslingual, e5_calibration,
│                               e6_options, e7_order  (raw.csv + summaries)
└── analysis/
    ├── bnfdm_style.py          CMYK palette, ACL figure style, bootstrap/ECE/AURC
    ├── analyze.py              raws -> analysis/tables/*.csv + tables.tex
    ├── figures.py              tables -> analysis/figures/*.pdf
    ├── tables/                 24 derived tables + LaTeX
    └── figures/                10 figures; each .pdf has a .png preview and a
                                .csv of exactly the data plotted
```

Reproduce:

```bash
python3 analysis/analyze.py    # tables
python3 analysis/figures.py    # figures
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

Seven experiments: **E1** distractor ladder · **E2** surface invariance via
`alternative_idioms` · **E3** unit integrity under shuffle/delete/substitute ·
**E4** cross-lingual failure localisation · **E5** calibration and selective
prediction · **E6** option-side attribution · **E7** option-order sensitivity.

## Figures

| | |
|---|---|
| `f1_distractor_ladder` | accuracy across the ladder; Literal Attraction Rate |
| `f2_literal_capture` | outcome decomposition; p(gold) vs p(literal) |
| `f3_confidence_anti_diagnostic` | confidence by outcome; inverted reliability |
| `f4_position_bias` | selection share by slot; accuracy by gold slot |
| `f5_order_instability` | distinct answers across 24 orderings; p(gold) range |
| `f6_crosslingual` | four input/checkpoint arms; failure localisation |
| `f7_unit_integrity` | accuracy and agreement under perturbation |
| `f8_option_ablation` | option-text ablation; semantic signal retained |
| `f9_selective_prediction` | risk-coverage curves; AURC vs a random gate |
| `f10_stratified` | by idiom length and corpus frequency |

No titles (captions belong in the paper), ACL single/double column widths, embedded
TrueType, CMYK palette on white.

## Data

[Bangla Bagdhara](https://www.kaggle.com/datasets/sakhadib/bangla-bagdhara) — 10,361
idioms under a 19-field schema, expert-validated by consensus of ≥2 Bangla
linguists. After deduplication on the idiom string and requiring both a literal and a
figurative gloss: **8,876 items**.

Sakhawat et al., *When Words Don't Mean What They Say: Figurative Understanding in
Bengali Idioms*, LREC 2026.

## Status

The Laya arm is complete on all 8,876 items. A second arm
(`mrn-dk/lod-lille-0.6B`, ex `thefloydd/qwen3-0.6b-rlcd`) is planned on a fixed 10%
subset; option sets are rebuilt from identical per-item seeds so the two arms pair
item-for-item and support McNemar rather than two accuracy numbers side by side.
