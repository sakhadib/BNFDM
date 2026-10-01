# Findings — two decision models on Bangla figurative meaning choice

| | Laya-multilingual | Lod-lille |
|---|---|---|
| checkpoint | `convaiinnovations/laya` · `multilingual` | `mrn-dk/lod-lille-0.6B` |
| backbone | mmBERT-base, bidirectional encoder | Qwen3-0.6B-Base + merged LoRA |
| params | 322M | 0.6B |
| items | 8,876 (full) | 888 (fixed 10% subset) |
| precision | bf16 | fp32 |
| latency / call | 27.2 ms | 86.1 ms |

Numbers from `LAYA_RUN/` and `LOD_RUN/`, reproduced by `analysis/analyze.py` and
`analysis/crossmodel.py`. 95% CIs are percentile bootstrap, 2,000 resamples over
items. Chance is 0.250 throughout.

**Task.** Given a Bangla idiom, choose its gold `figurative_meaning_bn` from k = 4
candidates. The **distractor family is the independent variable**; item,
instruction, k and per-item option-order randomisation are held fixed.

**Pairing verified.** The arms ran in different accounts from separate notebooks,
reconstructing the corpus and the task instrument independently from a hardcoded
contract. On all 4,440 overlapping rows the gold key, gold position, idiom string
and literal-distractor key match at **1.000** (`_cross/pairing_verification.csv`).
Every paired test below is therefore on byte-identical questions.

---

## 1. Literal capture — and a crossover

| Distractor set | Laya | Lod | Lod − Laya |
|---|---|---|---|
| random other glosses | 0.367 [0.357, 0.377] | **0.484** [0.453, 0.516] | +0.133 \*\*\* |
| surface-matched | 0.338 [0.328, 0.347] | **0.412** [0.380, 0.444] | +0.073 \*\*\* |
| tag-matched | 0.333 [0.324, 0.343] | **0.446** [0.414, 0.478] | +0.120 \*\*\* |
| **+ own literal gloss** | **0.151** [0.144, 0.159] | **0.066** [0.051, 0.084] | **−0.070** \*\*\* |
| **literal + surface + tag** | **0.144** [0.137, 0.151] | **0.063** [0.048, 0.080] | **−0.086** \*\*\* |

(McNemar on the 888 shared items; \*\*\* = *p* < 0.001.)

Lod is the stronger model on every non-literal condition — up to +13 points. Add the
idiom's own expert-annotated **literal** gloss and the ordering **reverses**: Lod
falls to 0.066, less than half of Laya's already-below-chance 0.151, and barely a
quarter of chance.

**Literal Attraction Rate** — the share of items where the model prefers the literal
reading to the figurative one:

| | Laya | Lod |
|---|---|---|
| LAR (D-lit) | 0.644 [0.634, 0.653] | **0.885** [0.864, 0.905] |
| LAR (D-hard) | 0.631 [0.621, 0.641] | 0.855 [0.832, 0.877] |
| mean *p*(literal) | 0.464 | **0.726** |
| mean *p*(gold) | 0.195 | 0.128 |

**The better model is more captured.** Capability on the surrounding task does not
protect against literal capture; here it is associated with more of it. That
dissociation is the central result — it means LAR measures something the accuracy
number does not, and that a model can be selected as "better" on a standard
benchmark while being strictly worse at the thing the benchmark is a proxy for.

Full decomposition for D-lit — Laya: gold 0.151, literal 0.644, other 0.207.
Lod: gold 0.066, literal 0.885, other 0.048.

On the clearest cases Laya puts **p = 1.000 on the literal gloss and 0.000 on the
gold**:

| Idiom | Literal gloss (chosen) | Gold figurative gloss (rejected) |
|---|---|---|
| কব্জা | কব্জি বা হাতের আঁচড়ানো অংশ | অবাঞ্ছিত অধিকার, প্রভাব |
| কাঠ | কাঠ (wood) | কঙ্কাল |
| চামচা | চামচ (spoon) | খোসামুদে; চাটুকার; তোষামুদে |
| গ্যাঁড়াকল | গ্যাঁড়া (স্ক্রু বা ফাঁদের যন্ত্র) এবং কল | লোক ঠকাবার কৌশল; বিপদ, ফাঁদ |

→ `f1_distractor_ladder.pdf`, `f2_literal_capture.pdf`, `f10_cross_model.pdf`

## 2. Confidence is anti-diagnostic in both models

Mean probability assigned to the chosen option, by outcome, on the literal-bearing
families:

| Outcome | Laya | Lod |
|---|---|---|
| correct | 0.432 [0.428, 0.436] | 0.453 [0.421, 0.487] |
| **captured by literal** | **0.596 [0.593, 0.600]** | **0.787 [0.779, 0.796]** |
| other error | 0.421 [0.418, 0.425] | 0.399 [0.368, 0.432] |

Both models are **more confident when captured than when correct**, and Lod more
extremely so. The reliability curve for the literal families slopes *downward* in
both.

Selective prediction therefore inverts. Restricting Laya's D-lit to the most
confident 5% gives **0.007** accuracy against 0.151 at full coverage. AURC under a
confidence gate is *worse than a random gate* for both models on the literal
families (Laya 0.924 vs 0.849; Lod 0.989 vs 0.933).

Lod ships a **separate confidence head** whose stated purpose is estimating whether
the answer is in the option list at all, and whose AURC is reported better than top
probability on its own benchmark. It does not rescue this: mean head confidence is
0.631 on captured items against 0.295 on correct ones — the same inversion. Both
calibration philosophies fail here, and they fail the same way: Laya ships fitted
per-bucket temperatures, Lod deliberately ships none (`temperature: 1.0`, the
authors having found fitting hurt held-out calibration).

→ `f3_confidence_anti_diagnostic.pdf`, `f9_selective_prediction.pdf`

## 3. Option order: a clean architectural dissociation

> **Correction.** An earlier version of this analysis counted distinct option *keys*
> (slot letters) across permutations and reported ~100% instability for Laya. That
> metric is an artifact: a key necessarily changes when the same gloss moves slots.
> The correct unit is the identity of the **chosen gloss**, recovered as
> `perm[slot(pred)]`. All numbers below use the corrected metric; the Laya figure
> drops from 1.000 to 0.837, and the finding stands.

992 Laya items and 299 Lod items, each under all 24 orderings of a fixed option set.

| | Laya | Lod |
|---|---|---|
| items whose **chosen gloss** changes under some ordering | **0.837** | **0.107** |
| items whose correctness varies | 0.686 | 0.070 |
| mean distinct glosses chosen (of 4) | 2.83 | 1.14 |
| items perfectly stable | 0.163 | **0.893** |
| mean within-item range of *p*(gold) | 0.346 | **0.029** |
| mean within-item confidence range | 0.391 | 0.044 |

A **12× difference in probability stability** and a 7.8× difference in answer
stability. Lod's card states that each option attends only to the state, its question
and itself from the same start position, making scores order-independent by
construction in fp32; run in fp32, it very nearly is. Laya's option-marker scorer is
not, confirming upstream issue
[#779](https://github.com/NandhaKishorM/laya/issues/779) at scale.

The residual positional bias follows. With the gold placed uniformly, Laya selects
slot B 37.2% of the time and slot A only 14.6%; accuracy is 0.066 when the gold sits
in A against 0.215 in B — a 3.3× swing from position alone. Lod's selection shares
are 0.265 / 0.258 / 0.242 / 0.235, near-uniform.

**Methodological consequence:** any evaluation of this model class that fixes option
order is partly measuring position. All results here randomise per item, seeded by
item id, and are therefore averages over position.

→ `f4_position_bias.pdf`, `f5_order_instability.pdf`

## 4. The failure localises to Bangla in both models

| Input / checkpoint | Accuracy |
|---|---|
| **Laya** Bangla idiom / multilingual | 0.371 [0.362, 0.382] |
| **Laya** English equivalent / multilingual | 0.617 [0.608, 0.628] |
| **Laya** English equivalent / English ckpt | 0.733 [0.724, 0.743] |
| **Laya** Bangla idiom / English ckpt *(control)* | 0.283 [0.274, 0.292] |
| **Lod** Bangla idiom | 0.483 [0.452, 0.516] |
| **Lod** English equivalent | **0.864 [0.841, 0.886]** |

Holding the model fixed and swapping only the language: Laya **+0.246**, Lod
**+0.380** (both McNemar *p* < 0.001). Localisation — items right in English and
wrong in Bangla: Laya 37.4%, Lod 43.3%. The reverse is 12.8% and 5.3%.

Lod reaches **0.864 in English**, so the figurative concepts are largely present; it
simply cannot reach them through Bangla. Its card says the training evaluation was
English-only, and this is what that looks like from the outside. Laya's control arm
reproduces the documented failure mode — the English checkpoint on Bengali script sits
at 0.283, barely above chance.

→ `f6_crosslingual.pdf`

## 5. Neither model treats the idiom as a unit

Perturb the idiom, hold the options fixed (Laya n = 1,778; Lod n = 180).

| Perturbation | Laya acc | Δ | Lod acc | Δ |
|---|---|---|---|---|
| intact | 0.157 | — | 0.072 | — |
| word shuffle | 0.168 | +0.011 | 0.044 | −0.028 |
| reversed | 0.178 | +0.021 | 0.056 | −0.017 |
| delete 1 word | 0.158 | +0.001 | 0.089 | +0.017 |
| substitute 1 word | 0.162 | +0.005 | 0.056 | −0.017 |
| **in carrier sentence** | **0.237** | **+0.080** | **0.189** | **+0.117** |

Scrambling the word order of a non-compositional expression does not meaningfully
hurt either model — the effects are small and, on Lod's 180 items, mostly within
noise (McNemar against Laya is n.s. for intact, delete-1 and in-sentence). A model
holding the idiom as a stored lexical unit should collapse under shuffling. Neither
does.

The one manipulation that clearly helps both is embedding the idiom in its gold
carrier sentence: +0.080 and +0.117. Context, not the idiom string, carries the
usable signal.

Note Lod's answer agreement under shuffling is 0.917 against Laya's 0.742 — Lod is
more *stable*, consistent with §3, while being no more *compositional*.

→ `f7_unit_integrity.pdf`

## 6. Options carry the signal, as a bag of words

| Option text | Laya acc | Signal retained | Lod acc | Signal retained |
|---|---|---|---|---|
| full Bangla gloss | 0.365 | 1.000 | 0.492 | 1.000 |
| truncated to 6 words | 0.361 | 0.967 | 0.469 | 0.909 |
| word-scrambled | 0.357 | 0.925 | 0.479 | 0.948 |
| English gloss | 0.347 | 0.835 | 0.398 | 0.636 |
| semantics removed | 0.256 | 0.000 | 0.233 | 0.000 |

Stripping option text to bare labels drops both models to chance (0.256 / 0.233,
McNemar between them n.s., *p* = 0.76) — the instrument's null behaves identically in
both, which is a useful check that the task is measuring option semantics at all.

But **scrambling the words inside each option costs only 5–8% of that signal** in
both models — bag-of-words on the option side too, mirroring §5. Replacing Bangla
glosses with English ones costs Laya little (0.835 retained) but Lod much more
(0.636), again consistent with Lod's English-only training leaving it with a
Bangla-side deficit rather than a conceptual one.

→ `f8_option_ablation.pdf`

## 7. Surface invariance

Items with expert-validated alternative surface forms: same gold meaning, different
words. Laya 2,423 items / 3,143 pairs; Lod 234 items / 310 pairs.

| | Laya | Lod |
|---|---|---|
| answer agreement across the pair | 0.657 [0.640, 0.674] | 0.677 [0.626, 0.729] |
| both correct | 0.085 | 0.016 |
| mean JSD | 0.039 | 0.085 |

A model reading meaning would be invariant across a pair the annotators certified as
synonymous. About a third of the time, both change their answer.

## 8. Is the 10% subset representative?

Laya's accuracy on the 888 Lod items against its own full 8,876:

| Family | full | subset | drift |
|---|---|---|---|
| D-rand | 0.367 | 0.351 | −0.016 |
| D-surf | 0.338 | 0.339 | +0.001 |
| D-sem | 0.333 | 0.325 | −0.008 |
| D-lit | 0.151 | 0.136 | −0.015 |
| D-hard | 0.144 | 0.149 | +0.004 |

Maximum drift 0.016. Two of five subset estimates fall marginally outside the
full-set bootstrap CI (D-rand, D-lit), which is expected at n = 888 and does not
change any sign or ordering. Lod's numbers should be read with ±3-point CIs; its
stratified tables (`by_frequency`, `by_scape`, the significance flags) have thin
cells and are indicative only — cite Laya's full-set versions as the precise ones.

→ `f11_stratified_and_subset.pdf`

---

## What this adds up to

Four claims, each from an independent instrument, on byte-identical questions across
two architecturally unrelated models:

1. **Literal capture is severe, measurable, and does not improve with capability.**
   LAR 0.644 (Laya) and 0.885 (Lod); both below chance when the literal gloss is
   present. The stronger model on every other condition is the more captured one.
2. **Calibration claims do not survive this distribution shift, in either
   philosophy.** Fitted temperatures (Laya) and deliberately unfitted ones (Lod)
   both yield confidence that is higher when wrong, and selective prediction worse
   than no gating. Lod's dedicated confidence head does not rescue it.
3. **Option-order dependence is architecture-specific, and the dissociation is
   clean.** 83.7% vs 10.7% of items change answer; 12× difference in probability
   stability. Independent-option scoring delivers what it claims; option-marker
   scoring does not.
4. **Failure localises to Bangla, not to figurativeness.** +0.246 and +0.380 from
   swapping language alone, with 37.4% and 43.3% of items right in English only.

(1), (2) and (3) are properties of the decision-model class, established by
replication and dissociation rather than by a single model. (4) is the low-resource
contribution, and Lod's 0.864 in English against 0.483 in Bangla is the sharpest form
of it.

## Limitations

- Both models zero-shot. Laya's card is explicit that its base checkpoints are near
  chance on its own typed-decisions benchmark and that the family is "a fast base to
  specialise". Nothing here speaks to a fine-tuned model of either family.
- Laya's run had `temperature_by_options` empty and `temperature = [1,1,1]`
  (`LAYA_RUN/.../env.json`), so its shipped per-bucket temperatures — including the
  known-broken `choice:11+` entry — were **not** applied. Laya ECE figures are raw.
  Lod ships `temperature: 1.0` by design, so both arms are effectively untempered
  and comparable on that axis.
- Lod ran 888 items (10%), so its CIs are ~±3 points and its stratified cells are
  thin. §8 bounds the resulting bias at ≤0.016.
- E3 on the Lod arm is 180 items; several of its perturbation contrasts are not
  individually significant.
- E7 covers 992 (Laya) and 299 (Lod) items × 24 orderings, not the full sets.
- Distractors are sampled from the same corpus, so difficulty is corpus-relative.
- `cultural_significance` is unusable as a stratifier (10,337 of 10,359 rows are 1).
- No figure contains Bangla glyphs: matplotlib cannot shape Bengali and no Bengali
  font was available on the analysis machine. Qualitative examples appear in this
  document, which renders Bangla natively.
