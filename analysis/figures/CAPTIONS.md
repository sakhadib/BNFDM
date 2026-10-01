# Figure notes — what each mark means, and what the caption has to say

The figures carry **no legends, no series names, no value labels, no in-axes
text**. Everything that identifies a series lives here and must be written into
the LaTeX caption. If a caption does not state the colour key, the figure is
unreadable — that is by design, not an oversight.

## Global key — use these words in every caption that needs them

| mark | meaning |
|---|---|
| cyan `#27EBF5` | **Laya-ml** (mmBERT encoder, multilingual, full 10k Bangla set) |
| magenta `#F127F5` | **Lod-lille-0.6B** (Qwen3-0.6B scorer, 10% random subset) |
| yellow `#F5C827` | **Jev-1.13** (closed commercial decision model via OpenRouter) |
| grey `#CCCCCC` / `#999999` / `#4D4D4D` | control, neutral or reference series — never a model |
| black dashed horizontal line | chance, 0.25 (k = 4 options) |
| black thin bar outline / caps | bar edge and bootstrap 95% CI (2000 resamples) |
| diagonal hatching on a **coloured** bar | the **English-language** condition |
| diagonal hatching on a **light grey** block | condition **not run** |

Ordering convention, relied on by every caption: **within a panel, bars run
Laya → Lod → Jev left to right; panels run Laya → Lod → Jev left to right.**
Two-arm figures are Laya → Lod.

Bangla script cannot be shaped by matplotlib, so no figure contains Bangla
glyphs; every category is named in Latin script or by an ASCII family code.

---

## Where each figure sits in the paper

Figure numbers are assigned by LaTeX, so they do not match the file names.
Current mapping (`paper/acl_latex.tex`):

| file | paper | location |
|---|---|---|
| (TikZ, in acl_latex.tex) | Figure 1 | body, page 1 teaser |
| (TikZ, in acl_latex.tex) | Figure 2 | body, Methodology pipeline |
| fig1_literal_capture | Figure 3 | body, 5.1 (the central result) |
| fig4_confidence_reliability | Figure 4 | body, 5.2 |
| fig5_no_signal_control | Figure 5 | body, 5.3 |
| fig2_capture_matrix | Figure 4 | body, 5.1 (six measured cells) |
| fig10_crosslingual | Figure 7 | Appendix, cross-lingual arms |
| fig3_three_way_bangla | Figure 8 | Appendix, matched three-way |
| fig9_unit_integrity | Figure 9 | Appendix, unit integrity |
| fig8_option_ablation | Figure 10 | Appendix, option ablation |
| fig7_position_bias | Figure 11 | Appendix, order and position |
| fig6_order_sensitivity | Figure 12 | Appendix, order and position |

After the October revision the body keeps the ladder (the central result), the
reliability curves and the semantics-free control. Everything else moved to the
appendix, either because a table already carries its numbers or because the
claim it supports is now explicitly scoped rather than headline.

Captions now carry colour swatches (the swL / swD / swJ / swG / swK macros in
the LaTeX), so the reader gets the key from the caption itself. Those macros are
defined in the preamble to the same hex values as the figure fills.


## fig1_literal_capture

**Plot.** Left: accuracy across the five distractor families (random,
surface-matched, tag-matched, + own literal gloss, hard = all three combined).
Right: literal attraction rate — the share of items on which the model picks the
idiom's own literal gloss over the gold figurative meaning — for the two
families that contain a literal gloss. Two bars per group, Laya then Lod.
Error bars 95% bootstrap CI. Dashed line = chance.

**Caption must say.** Cyan Laya-ml (n = 8,876), magenta Lod-lille (n = 888);
dashed line chance (0.25); error bars 95% bootstrap CI.

**Key numbers.** Accuracy Laya/Lod: D-rand .367/.484, D-surf .338/.412,
D-sem .333/.446, D-lit **.151/.066**, D-hard **.144/.063**.
LAR D-lit .644/.885, D-hard .633/.858.
The point: surface and tag distractors cost a few points; adding the literal
gloss collapses both models well below chance, and the smaller model collapses
harder.

---

## fig2_capture_matrix

**Plot.** Literal attraction rate on the shared English-literal pool
(n = 3,265 items), by model × task language. Solid coloured bar = Bangla
condition, hatched coloured bar = English condition, light grey hatched block
running off the top of the axis = **that cell was never run** (it is not a
value of 1.0). Dashed line = chance.

**Caption must say.** Explicitly: the grey hatched blocks are unmeasured cells,
not measurements; Jev's low value is its **English** condition and is therefore
not directly comparable to the Bangla values beside it.

**Key numbers.** Laya BN **.747**, Lod BN **.908**, Jev EN **.047**.
Anti-diagonal (Laya/Lod in English, Jev in Bangla) never run.

**Caveat to carry into the text.** The pool is not neutral: in-pool Laya LAR is
.747 against .583 out of pool, while D-rand accuracy is identical to within
.008. Items with an English literal gloss are the more literally-attracting
ones.

---

## fig3_three_way_bangla

**Plot.** Accuracy on the one condition that is byte-identical across all three
models: Bangla idiom → Bangla figurative gloss, random distractors.
Error bars 95% bootstrap CI, dashed line chance.

**Caption must say.** Bars left to right Laya-ml, Lod-lille, Jev-1.13 (colours
as the global key); this is the only fully-matched three-way comparison in the
paper.

**Key numbers.** Laya .3672 [.3574, .3766] n = 8,876; Lod .4842 [.4527, .5180]
n = 888; Jev **.7355** [.7266, .7445] n = 8,761.
McNemar: Jev − Laya +.3694 (p ≈ 0), Jev − Lod +.2589 (p ≈ 0),
Lod − Laya +.1329 (p ≈ 0).

---

## fig4_confidence_reliability

**Plot.** Reliability curves, one panel per model, 15 equal-width confidence
bins. Grey dashed diagonal = perfect calibration. In each panel the **dark grey
circles** are the random-distractor condition and the **model-coloured squares**
are the + literal-gloss condition.

**Caption must say.** Panels left to right Laya-ml, Lod-lille, Jev-1.13;
grey circles D-rand, coloured squares D-lit; grey diagonal perfect calibration;
Jev's panel is English, the other two Bangla.

**Key numbers.** ECE D-rand / D-lit: Laya .071/.385, Lod .073/.679,
Jev .005/.014. Under D-lit the first two models' curves bend **downward** —
higher stated confidence buys lower accuracy, which is the inverted-reliability
result. Jev stays on the diagonal.

---

## fig5_no_signal_control

**Plot.** The O4 ablation: option text replaced by bare labels, so no semantic
signal remains and accuracy is chance by construction. Per model, **grey bar =
accuracy** (with CI), **coloured bar = mean reported confidence**. Dashed line
chance.

**Caption must say.** Grey = accuracy, colour = mean confidence; accuracy is at
chance by construction, so the gap between the two bars is pure overconfidence.
Jev's condition is English.

**Key numbers.** Laya acc .256 / conf .506 (gap .250);
Lod .233 / .282 (gap .049); Jev .254 / **.815** (gap **.561**, ECE .561).
The headline reversal: the best-calibrated model under real options is the worst
when the options carry no information at all.

---

## fig6_order_sensitivity

**Plot.** Left: distribution of how many **distinct glosses** a model selects
across all 24 permutations of the same four options (1 = fully stable), three
bars per group in model order. Right: per model, **coloured bar = share of items
whose chosen answer is not stable across orderings**, **grey bar = mean range of
p(gold) across the 24 orderings**.

**Caption must say.** Left panel bars left to right Laya, Lod, Jev within each
group; right panel colour = unstable share, grey = mean p(gold) range.
The metric is the chosen option's **identity**, not its slot key.

**Key numbers.** Fully stable share (1 distinct gloss): Laya .163, Lod .893,
Jev .867. Unstable share .837 / .107 / .133; mean p(gold) range .346 / .029 /
.128; selection-share spread .226 / .052 / .072.

**Note for the text.** An earlier version of this metric counted option keys
rather than identities and reported Laya at 1.000; the corrected value is .837.
This is recorded in FINDINGS.md.

---

## fig7_position_bias

**Plot.** Share of selections falling on each option slot A–D, one panel per
model. Gold is placed uniformly at random, so any departure from the dashed
0.25 line is positional bias.

**Caption must say.** Panels left to right Laya-ml, Lod-lille, Jev-1.13;
dashed line = uniform expectation (.25); gold position is uniform by
construction.

**Key numbers.** Laya A .146 / B **.372** / C .267 / D .215 — a strong slot-B
preference. Lod .270/.268/.244/.217. Jev .287/.256/.241/.216 — a mild
primacy gradient in both of the latter two.

---

## fig8_option_ablation

**Plot.** Accuracy under five option-side rewrites: full text, truncated to six
tokens, token-scrambled, translated into the other language, and reduced to
semantics-free labels. **The fifth (grey) bar is the no-semantics control.**
The whole **Jev panel is hatched because its task is in English.**
Error bars 95% bootstrap CI, dashed line chance.

**Caption must say.** Panels left to right Laya, Lod, Jev; the grey bar in each
panel is the semantics-free control; hatching marks the English-language panel.

**Key numbers.** Laya .365/.362/.358/.348/.256; Lod .492/.468/.479/.400/.233;
Jev .970/.958/.974/.963/.254. Truncation and scrambling of the option text cost
almost nothing in all three models — the decision is being made on surface
overlap long before the option is read as a sentence.

---

## fig9_unit_integrity

**Plot.** The idiom is perturbed while the options are held fixed. Left:
accuracy. Right: agreement with the answer given on the intact idiom. Variants:
intact, word-shuffled, reversed, one word deleted, one word substituted, and
embedded in a carrier sentence. Two bars per group, Laya then Lod.

**Caption must say.** Cyan Laya-ml, magenta Lod-lille; error bars 95% bootstrap
CI; dashed line chance (left panel only); options are identical across variants,
only the idiom changes.

**Key numbers.** Laya accuracy barely moves from intact (.158) through shuffled
(.168) and reversed (.180); agreement with intact is .75 under shuffling. Lod is
flat too. Because an idiom is a non-compositional unit, a model that reads it as
one would have to degrade when the unit is broken — neither does. This is the
bag-of-words signature, and it is the experiment that replaces word-level
attribution (which is ill-posed for idioms).

---

## fig10_crosslingual

**Plot.** Accuracy by condition, grouped by model, groups left to right
Laya (4 bars), Lod (2), Jev (2). Within a group, **solid = Bangla input,
hatched = English input**. Tick labels give the input language and, for Laya's
two extra conditions, which model variant scored it.
Error bars 95% bootstrap CI, dashed line chance.

**Caption must say.** Group order Laya-ml / Lod-lille / Jev-1.13 by colour;
solid bars Bangla, hatched bars English; Laya's third and fourth bars are the
English-only model variant, included as the cross-lingual control.

**Key numbers.** Laya: BN .371, EN .617, EN(en-model) .733,
BN(en-model) .283 (control, near chance — confirms the Bangla result is not an
artefact of the English scorer). Lod: BN .484 → EN .863.
Jev: BN .736 → EN .968.
Every model gains 20–45 points simply by having the same idiom presented in
English. The figurative-meaning task is not language-neutral.

---

## Files

Each figure writes three siblings in this directory: `.pdf` (for the paper),
`.png` at 320 dpi (for review), `.csv` (the exact plotted values). Quote numbers
in the text from the `.csv`, not from reading the bars.

Regenerate with `python analysis/figures.py`. Colours and all shared helpers
live in `analysis/bnfdm_style.py`; that module deliberately contains **no**
label, legend or value-annotation helpers, so a figure cannot drift back into
carrying its own key.


## Bangla-script figures

`analysis/bangla_figures.py` produces three vector PDFs that carry the actual
Bengali text, because pdfLaTeX cannot shape Bengali:

| file | paper | shows |
|---|---|---|
| fig_bn_item | Figure 7 | one complete item as the models received it, with real option texts and returned probabilities |
| fig_bn_manipulations | Figure 8 | the E3 states and E6 option forms applied to real Bangla |
| fig_bn_examples | Figure 9 | five real literal-capture items with both glosses |

They are rendered by headless Chromium (HarfBuzz) from HTML, and are true
vector: embedded Noto Sans Bengali subsets, selectable text, no raster. The font
is fetched via `npm pack @fontsource/noto-sans-bengali` and converted woff2 to
ttf with fontTools; the script assumes it is installed system-wide.

Option sets are reconstructed with the same builder and seed the runs used, and
the script asserts the reconstruction matches the recorded gold and literal keys
before drawing. The E3 strings are the exact ones logged during the run.
