# Findings — Laya (multilingual) on Bangla figurative meaning choice

All numbers from `LAYA_RUN/results/laya-ml/`, reproduced by `analysis/analyze.py`.
Tables in `analysis/tables/`, figures in `analysis/figures/` (PDF + PNG + the plotted
data as a sibling CSV). 95% CIs are percentile bootstrap, 2,000 resamples over items.

**Setup.** 8,876 Bangla idioms (deduplicated on the idiom string, both glosses
required). The task is 4-way meaning choice: given the idiom, pick its gold
`figurative_meaning_bn` from four candidates. Chance is 0.250. Option order is
randomised per item, seeded by item id. Checkpoint pinned to
`convaiinnovations/laya` subfolder `multilingual`; no Router, so no state is ever
routed to the English checkpoint by accident.

---

## 1. Literal capture: adding the idiom's own literal gloss drives accuracy below chance

The distractor family is the only thing that varies. Everything else — item,
instruction, k, option-order randomisation — is held fixed.

| Distractor set | Accuracy [95% CI] | vs. chance | LAR | ECE |
|---|---|---|---|---|
| random other glosses | 0.367 [0.357, 0.377] | +0.117 | — | 0.071 |
| surface-matched | 0.338 [0.328, 0.347] | +0.088 | — | 0.099 |
| tag-matched | 0.333 [0.323, 0.342] | +0.083 | — | 0.100 |
| **+ own literal gloss** | **0.151 [0.144, 0.159]** | **−0.099** | **0.644** | 0.385 |
| **literal + surface + tag** | **0.144 [0.137, 0.151]** | **−0.106** | **0.631** | 0.389 |

Against random distractors the model is meaningfully above chance. Replace one
distractor with the item's *own expert-annotated literal gloss* and accuracy falls to
**less than half of chance**. The **Literal Attraction Rate** — the share of items
where the model prefers the literal reading to the figurative one — is **0.644**
[0.634, 0.653].

The full outcome decomposition (D-lit): gold 0.151, literal 0.644, some other idiom's
gloss 0.207. Paired within item against random distractors, the swap costs
−0.216 accuracy; 2,667 items flip right→wrong against 747 wrong→right
(McNemar *p* < 10⁻³⁰⁰).

This is not a near-miss. On the clearest cases the model puts **p = 1.000 on the
literal gloss and 0.000 on the gold figurative gloss**:

| Idiom | Literal gloss (chosen) | Gold figurative gloss (rejected) |
|---|---|---|
| কব্জা | কব্জি বা হাতের আঁচড়ানো অংশ | অবাঞ্ছিত অধিকার, প্রভাব |
| কাঠ | কাঠ (wood) | কঙ্কাল |
| চামচা | চামচ (spoon) | খোসামুদে; চাটুকার; তোষামুদে |
| গ্যাঁড়াকল | গ্যাঁড়া (স্ক্রু বা ফাঁদের যন্ত্র) এবং কল | লোক ঠকাবার কৌশল; বিপদ, ফাঁদ |

→ `f1_distractor_ladder.pdf`, `f2_literal_capture.pdf`

## 2. Confidence is anti-diagnostic — gating on it is worse than not gating

| Outcome | Share | Mean confidence [95% CI] | *p*(gold) |
|---|---|---|---|
| correct | 0.148 | 0.432 [0.428, 0.436] | 0.432 |
| **captured by literal** | **0.637** | **0.596 [0.593, 0.600]** | 0.145 |
| other error | 0.217 | 0.421 [0.418, 0.425] | 0.182 |

The model is **more confident when it is captured by the literal reading than when
it is right**. The reliability diagram for the literal families slopes *downward*:
items in the 0.9–1.0 confidence bin are correct about **1%** of the time.
Consequently selective prediction inverts. Restricting to the most confident 5% of
D-lit items gives **0.007** accuracy versus 0.151 at full coverage — the risk-coverage
curve **rises** with coverage. AURC with confidence as the gate (0.924) is *worse
than a random gate* (0.849). The model card's claim that RLCD yields honest
probabilities does not transfer to this distribution.

→ `f3_confidence_anti_diagnostic.pdf`, `f9_selective_prediction.pdf`

## 3. Option order decides the answer

992 items × all 24 orderings of a fixed option set.

- **100%** of items change their answer under some permutation.
- **82.8%** produce **all four** distinct answers; mean 3.82 distinct answers of 4.
- Mean within-item range of *p*(gold) across orderings: **0.346** (max 0.884).
- Mean within-item confidence range: 0.391.

The bias is positional, not random. With the gold gloss placed uniformly across
slots, the model selects slot B 37.2% of the time and slot A only 14.6%. Accuracy
when the gold sits in slot A is **0.066** versus 0.215 in slot B — a 3.3× swing
driven purely by position. The same skew appears independently in E1, where option
order is randomised per item.

This empirically confirms upstream issue
[#779](https://github.com/NandhaKishorM/laya/issues/779) at scale. For an
architecture whose options are meant to be independent parallel predicates scored at
their own `[MASK]` tokens, order-dependence of this magnitude is a defect, not a
tolerance.

**Methodological consequence:** any evaluation of this model class that fixes option
order is measuring position as much as meaning. All results here randomise per item
and are therefore averages over position.

→ `f4_position_bias.pdf`, `f5_order_instability.pdf`

## 4. The failure is Bangla representation, not the figurative concept

Same items, same distractor items, four input/checkpoint combinations.

| Input / checkpoint | Accuracy [95% CI] | Confidence |
|---|---|---|
| Bangla idiom / multilingual | 0.371 [0.362, 0.382] | 0.438 |
| English equivalent / multilingual | 0.617 [0.608, 0.628] | 0.584 |
| English equivalent / English | 0.733 [0.724, 0.743] | 0.613 |
| Bangla idiom / English *(control)* | 0.283 [0.274, 0.292] | 0.390 |

Holding the checkpoint fixed and swapping only the language gives **+0.246**
accuracy (McNemar *p* < 10⁻³⁰⁰). Localisation: **37.4%** of items are right in
English and wrong in Bangla; only 12.8% the reverse. The control arm confirms the
documented failure mode — the English checkpoint on Bengali script sits at 0.283,
barely above chance.

So the model largely *has* the figurative concept and cannot reach it through
Bangla. That is a representation gap, not a conceptual one, and it is the sharpest
single result for a low-resource figurative-language argument.

→ `f6_crosslingual.pdf`

## 5. The idiom is not processed as a unit

Perturb the idiom, hold the options fixed (n = 1,778 idioms of ≥3 words).

| Perturbation | Accuracy | Δ vs intact | Agreement with intact | JSD |
|---|---|---|---|---|
| intact | 0.157 | — | 1.000 | 0.000 |
| word shuffle | 0.168 | +0.011 | 0.742 | 0.015 |
| reversed | 0.178 | +0.021 | 0.706 | 0.017 |
| delete 1 word | 0.158 | +0.001 | 0.747 | 0.014 |
| substitute 1 word | 0.162 | +0.005 | 0.692 | 0.020 |
| **in carrier sentence** | **0.237** | **+0.080** | 0.551 | 0.046 |

Scrambling the word order of a non-compositional expression **does not hurt** — it
is within noise of intact, and reversal is nominally *better*. A model holding the
idiom as a stored lexical unit should collapse under shuffling. This one does not,
which is the bag-of-words signature.

Note the asymmetry: answer agreement with intact drops to 0.74 under shuffling even
though accuracy is unchanged. The perturbation moves the answer around without
moving it toward or away from the truth — consistent with §3, where the answer is
substantially determined by factors orthogonal to meaning.

The one manipulation that helps is embedding the idiom in its gold carrier sentence
(+0.080). Context, not the idiom string, is what carries usable signal.

→ `f7_unit_integrity.pdf`

## 6. Surface form matters more than meaning

2,423 items with expert-validated alternative surface forms (3,143 pairs): same gold
meaning, different words.

- Answer agreement between canonical and variant: **0.657** [0.640, 0.674]
- Both correct: 0.085. Neither correct: 0.754.
- Mean JSD between the two distributions: 0.039 (median 0.015)

A model reading meaning would be invariant across a pair the annotators certified as
synonymous. A third of the time the answer changes.

## 7. The model does read option semantics — about 86% of its input is options

Perturb the option text, hold the state and gold fixed.

| Option text | Accuracy | Δ | Semantic signal retained | Agreement with full |
|---|---|---|---|---|
| full Bangla gloss | 0.365 | — | 1.000 | 1.000 |
| truncated to 6 words | 0.361 | −0.004 | 0.967 | 0.704 |
| word-scrambled | 0.357 | −0.008 | 0.925 | 0.528 |
| English gloss | 0.347 | −0.018 | 0.835 | 0.364 |
| semantics removed | 0.256 | −0.109 | 0.000 | 0.115 |

Stripping the option text to bare labels drops the model to chance, so option
semantics genuinely drive the decision. But **scrambling the words inside each option
costs only 7.5% of that signal** — bag-of-words on the option side too, mirroring §5.
Replacing Bangla glosses with their English translations costs only 0.018 accuracy
while confidence *rises* from 0.438 to 0.499, which is the §4 representation gap
visible from the option side.

→ `f8_option_ablation.pdf`

## 8. Stratification

Accuracy declines with idiom length on the non-literal families (random: 0.364 at
1 word → 0.301 at 5) but *rises* on the literal families (0.146 → 0.193) — longer
idioms give the literal reading less of a grip. Frequency effects are non-monotonic
and the `very rare` cell has only 27 items, so it should not be read as a trend.

→ `f10_stratified.pdf`

---

## What this adds up to

Four claims, each supported by an independent instrument on the same 8,876 items:

1. **Literal capture is measurable and severe.** A 4-way choice between an idiom's own
   gold literal and gold figurative glosses is answered literally 64% of the time,
   putting accuracy below chance. LAR is a clean, annotation-free diagnostic that
   transfers to any idiom resource carrying both glosses.
2. **Calibration claims do not survive distribution shift.** Confidence is higher on
   captured items than correct ones and the risk-coverage curve inverts, so the
   advertised selective-prediction property is not merely weak here — it is harmful.
3. **Option-order dependence is a first-order confound** in this architecture class,
   large enough to change the answer on 100% of items.
4. **Failure localises to Bangla, not to figurativeness.** +0.246 from swapping
   language alone, with 37.4% of items right in English and wrong in Bangla.

(1), (2) and (3) are properties of the decision-model architecture and do not depend
on how the Bangla arm performs. (4) is the low-resource contribution.

## Limitations

- Single checkpoint, zero-shot. The model card is explicit that the base checkpoints
  are near chance on its own typed-decisions benchmark and that the family is "a fast
  base to specialise". Nothing here speaks to a fine-tuned Laya.
- `temperature_by_options` was empty in this run and `temperature` was `[1.0,1.0,1.0]`
  (see `results/laya-ml/env.json`), so the shipped per-bucket temperatures — including
  the known-broken `choice:11+` entry — were **not** applied. ECE figures are
  therefore raw, not post-temperature.
- E7 covers 992 items × 24 orderings, not the full set.
- Distractors are sampled from the same corpus, so difficulty is corpus-relative.
- `cultural_significance` is unusable as a stratifier (10,337 of 10,359 rows are 1).
- No figure contains Bangla glyphs: matplotlib cannot shape Bengali and no Bengali
  font was available on the analysis machine. Qualitative examples appear in this
  document instead, where they render natively.
