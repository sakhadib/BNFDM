# ARR submission sheet

Everything below is ready to copy. Each field is in its own fenced block so you
can select it cleanly. Nothing here contains author names or the de-anonymized
repository URL.

---

## 1. OpenReview submission fields

### Title

```
Literal Lock-In: How Test Language Shifts Literal Attraction in Typed Decision Models on Bangla Idioms
```

### Keywords

```
decision models, typed output, option scoring, literal attraction, idioms, figurative language, Bangla, Bengali, low-resource languages, cross-lingual evaluation, calibration, selective prediction, confidence, model auditing, guardrails
```

### TL;DR

```
Dropping an idiom's own literal gloss into the option list captures three decision models on Bangla, 1.9 to 6.4 times more than in English on the same paired items, and their confidence inverts, so gating on it is worse than gating at random.
```

### Abstract

```
Decision models, non-generative scorers that report a probability over caller-supplied options and emit no text, are increasingly deployed as routers, classifiers and guardrails. Because they generate nothing, rationales and chain-of-thought do not apply. We audit three: an encoder scorer, a decoder scorer and a closed API, entirely through interventions on the option set, using expert-annotated Bangla idioms carrying a gold figurative and a gold literal gloss. Dropping an idiom's own literal gloss into the option list draws every arm toward it, taking 66% to 96% of each arm's remaining Bangla errors. A model x language matrix over one shared item set shows how far this depends on the test language: every arm is 1.9 to 6.4 times more captured in Bangla than in English on the same paired entries, the commercial model's apparent resistance is a property of its test language, and capability does not order the failure. Calibration degrades in all three arms under the same shift, and on the open arms confidence inverts, so gating on it is worse than gating at random. On a semantics-free option list, where accuracy is at chance, the best-calibrated arm still reports mean confidence 0.82. Code and data are available at https://anonymous.4open.science/r/FSS-CC4D
```

### Research Area

```
Resources and Evaluation
```

Why this one: the contribution is an evaluation instrument plus a released
per-item artifact set, and the reviewers in that track are the ones who assess
measurement design, pooling and release quality, which is what the paper stands
or falls on.

If you would rather route it elsewhere, the two defensible alternates are
`Multilingualism and Cross-Lingual NLP` (the model x language matrix is the
headline result) and `Interpretability and Analysis of Models for NLP`. Be aware
that the third track's reviewers usually expect probing or mechanistic work, and
Section 4.4 and Appendix B state plainly that this paper measures behavior
rather than internals, so it is the weakest of the three fits.

### Research Area Keywords

Taken verbatim from the ARR areas page. Recommended list, primary-area keywords
first and then the four from adjacent areas that match what the paper actually
measures:

```
evaluation methodologies, evaluation, benchmarking, metrics, statistical testing for evaluation, reproducibility, datasets for low resource languages, calibration/uncertainty, robustness, data shortcuts/artifacts, multilingual evaluation, less-resourced languages
```

If you would rather keep every keyword inside Resources and Evaluation, use this
instead:

```
evaluation methodologies, evaluation, benchmarking, metrics, statistical testing for evaluation, reproducibility, datasets for low resource languages, language resources
```

Where each one is earned, so you can drop any that you think overclaims:

| keyword | area | what backs it |
| --- | --- | --- |
| evaluation methodologies | Resources and Evaluation | the whole instrument: auditing a non-generative scorer entirely through the option set, Section 4 and Appendix D |
| evaluation | Resources and Evaluation | the paper is an audit of three deployed arms |
| benchmarking | Resources and Evaluation | the distractor ladder and the model x language matrix |
| metrics | Resources and Evaluation | LAR, and ECE, ACE, Brier, AURC and AUGRC in Section 5.2 and Appendix F |
| statistical testing for evaluation | Resources and Evaluation | 95% percentile bootstrap and McNemar's exact test throughout, Appendix R |
| reproducibility | Resources and Evaluation | per-item outputs, seeds, split manifests, run records and per-figure CSVs, Appendix R |
| datasets for low resource languages | Resources and Evaluation | the Bangla corpus and the Bangla cells of every experiment |
| calibration/uncertainty | Interpretability and Analysis | Section 5.2 and Appendix F: three calibration metrics, the inverted reliability curves, selective prediction |
| robustness | Interpretability and Analysis | the perturbation, surface-form and option-order experiments, Appendices M, N and Q |
| data shortcuts/artifacts | Interpretability and Analysis | Section 5.6 and Appendix J: the verbatim-overlap and Latin-script cues, measured rather than assumed away |
| multilingual evaluation | Multilingualism and Cross-Lingual NLP | the model x language matrix over one shared item set, Section 5.1 and Appendix I |
| less-resourced languages | Multilingualism and Cross-Lingual NLP | Bangla, and the finding that the failure localizes to the test language |

Two Resources and Evaluation keywords I left out on purpose. **corpus creation**
and **NLP datasets**: we create no corpus. Section 3 says we add no annotation and
assemble every intervention from fields that already exist, so claiming either
would invite a reviewer to look for a resource contribution that is not there.
**multilingual corpora** is also out, since the source corpus is monolingual
Bangla and the English side comes from one annotated field.

### Contribution Types

Primary:

```
Model analysis & interpretability
```

Also applicable, if the form lets you tick more than one:

```
Data analysis
Approaches to low-resource settings
Publicly available software and/or pre-trained models
```

---

## 2. Responsible NLP Checklist

Every answered item has its elaboration below it in its own block, and each one
opens with the section or appendix number and title so the reviewer can find it
without searching. Section numbers are those of the current compiled PDF.

### Section A

**A1 potential risks**

```
Yes
```

**A1 elaboration**

```
Ethics Statement (unnumbered section, immediately after Section 7: Conclusion) and Limitations (unnumbered section, immediately before it). The Ethics Statement covers the three risks that apply to this work. First, the source corpus is a published expert-annotated resource of idioms containing no personal data and no content about identifiable individuals. Second, we report a named commercial endpoint's failure modes, so we state that the model alias is mutable, that at the time of our run it resolved to the 20260917 build, and that our figures should be read as a dated snapshot rather than a standing property of the product. Third, we flag against our own interest both the pool selection effect stated in Section 5.1: Literal capture, and what it tracks, and the lexical construction cue reported in Appendix J: How the two language cells were built, which accounts for part of the headline effect on one arm. Limitations sets out what the design cannot settle, including the relabeling confound in the order audit and the fact that the two language cells are paired entries rather than translations.
```

### Section B

**B use or create scientific artifacts**

```
Yes
```

**B elaboration**

```
Used: the Bangla Bagdhara corpus (Section 3: Dataset) and three model arms (Section 4.4: Models). Created and released: the instrument, the per-item probability outputs for every cell of every experiment, the split manifests, the task configurations, the analysis scripts and the figure sources, all enumerated in Appendix R: Run record.
```

**B1 cite creators of artifacts**

```
Yes
```

**B1 elaboration**

```
Section 3: Dataset cites the source corpus at first mention and again in the Ethics Statement. Section 4.4: Models names each of the three arms, and its footnote 2 gives the released artifact and hosting location for each one, including the exact endpoint and build for the commercial arm. Appendix R: Run record repeats the commercial build identifier.
```

**B2 discuss the license for artifacts**

```
Yes
```

**B2 elaboration**

```
Ethics Statement (unnumbered section, after Section 7: Conclusion) states that the source corpus is distributed under CC BY-SA 4.0 and that we release the instrument, the per-item outputs and the analysis code under the same license, as its share-alike term requires, with the two open checkpoints used under the terms their own model cards declare. Appendix R: Run record repeats the release license beside the repository link, and the repository carries the full CC BY-SA 4.0 text in its LICENSE file.
```

**B3 data contains personally identifying info**

```
Yes
```

**B3 elaboration**

```
Ethics Statement (unnumbered section, after Section 7: Conclusion): the corpus is a published resource of idioms, it contains no personal data and no content about identifiable individuals, and we add no annotation of our own, so nothing we release can introduce it. Section 3: Dataset describes the 19-field schema, none of whose fields carries personal data.
```

**B4 data contains offensive content**

```
N/A
```

**B4 elaboration**

```
Not applicable. Section 3: Dataset and Appendix R: Run record describe what we release: option lists assembled from fields that already exist in the published corpus, and the probability vectors the three arms returned for them. We add no annotation and the arms emit no text, as Appendix A: The typed question sets out, so the release opens no new surface on which objectionable content could appear.
```

**B5 documentation of artifacts**

```
Yes
```

**B5 elaboration**

```
Section 3: Dataset documents the corpus coverage, the language, the 19-field schema and the 8,876 rows that carry every field the instrument requires, including the footnote reconciling the published count of 10,361 against the 10,359 distributed rows. Appendix A: The typed question gives the exact interface, the request and response bodies and the two instruction strings. Appendix R: Run record, paragraph "What is released", enumerates every released file: the per-item outputs for every cell of every experiment, the five split manifests, the three run environment records, the task configurations with the verbatim instruction strings, the analysis scripts that produce every table in the paper, the figure sources, and a CSV of the exact plotted values beside each figure.
```

**B6 statistics for data**

```
Yes
```

**B6 elaboration**

```
Section 3: Dataset gives the corpus size and the usable subset. Table 4: The three model arms gives the item count per arm (8,876 / 888 / 8,761). Section 5.1: Literal capture, and what it tracks gives the 888-item shared subset and the 3,265-item matrix pool, and Appendix O: Subset representativeness bounds the drift from running one arm on the 10% subset at 0.016. Appendix J: How the two language cells were built documents every cut that produces the 3,265-item pool, and Appendix I: The capture matrix reports the per-cell counts and intervals. Appendix R: Run record gives the per-block call volume for the commercial arm, 56,533 calls in total. There are no train/dev/test splits because nothing here is trained; every split is an evaluation stratum and each one is documented where it is first used.
```

### Section C

**C computational experiments**

```
Yes
```

**C elaboration**

```
The paper is an audit. Section 5: Analysis reports the results of computational experiments on three model arms, and Section 4: Methodology defines the seven interventions that produce them.
```

**C1 model size and budget**

```
Yes
```

**C1 elaboration**

```
Table 4: The three model arms, referenced from Section 4.4: Models, gives the parameter count for each arm (322M, 0.6B, and closed for the commercial one) together with per-call latency. Appendix R: Run record, paragraph "Environment", gives the computing infrastructure: Python 3.13.15 throughout, with the two open arms on one Tesla T4 (15,360 MiB) and the commercial arm on CPU, and it notes that the T4 is Turing, so the configured bf16 setting falls back to ordinary fp32 CUDA cores. Appendix R, paragraph "Call volume and cost", gives the commercial arm's full budget: 56,533 calls for $0.9348, broken down per block.
```

**C2 experimental setup and hyperparameters**

```
Yes
```

**C2 elaboration**

```
Section 4: Methodology, and in particular Section 4.1: The meaning-choice task, Section 4.2: Distractor families and measures and Section 4.3: Interventions and statistics, defines the setup. Appendix A: The typed question gives the complete interface: one typed choice question and one probability vector, with no prompt template, no system message, no sampling parameter and no generated text, which is why the option set is the only intervention surface and why there is no prompt or decoding hyperparameter to search. Appendix C: What each experiment changes states what each of the seven interventions varies and holds fixed. Appendix R: Run record gives the per-experiment seeds, the split seed, and the temperature settings on both open arms with the reason they are untempered.
```

**C3 descriptive statistics**

```
Yes
```

**C3 elaboration**

```
Appendix R: Run record, paragraph "Statistics", states the convention once for the whole paper: all intervals are 95% percentile bootstrap over items, 2,000 resamples, fixed seed 0, and paired comparisons use McNemar's exact test via the binomial distribution. Every reported number is a single deterministic run rather than a max or a mean over runs, which is why the seeds are recorded per experiment in the same appendix. Intervals are reported alongside the results throughout: Section 5.1: Literal capture, and what it tracks, Table 15 in Appendix I: The capture matrix for the six matrix cells, and Appendix J: How the two language cells were built for the construction-cue effects. Figures carry 95% bootstrap error bars, stated in their captions.
```

**C4 parameters for packages**

```
Yes
```

**C4 elaboration**

```
Appendix R: Run record gives the environment and the metric definitions we implement rather than import: ECE over 15 equal-width bins, ACE as the adaptive equal-mass variant, multiclass Brier, and AURC against a random gate computed on the same items. Appendix F: Selective prediction in detail defines AUGRC and its analytic random baseline and reports all three calibration metrics on the same recorded distributions. The analysis scripts that compute every one of these are part of the release enumerated in Appendix R.
```

### Section D

**D human subjects including annotators**

```
No
```

**D elaboration**

```
We ran no annotation and recruited no participants. Section 3: Dataset states that the corpus is a pre-existing published resource annotated under a 19-field schema by expert consensus, that its annotation was carried out and documented by its own authors, and that we add no annotation: every intervention in the paper is assembled from fields that already exist in the corpus. The Ethics Statement repeats that our interventions rewrite option lists and perturb idiom strings and produce no new annotation.
```

### Section E

**E substantive ai assistance in research or writing**

```
Yes
```

**E1 information about substantive ai assistance**

```
Yes
```

**E1 elaboration**

```
Ethics Statement (unnumbered section, after Section 7: Conclusion), and the elaboration here. An AI coding assistant was used for the analysis and plotting code and for copy-editing passes over the manuscript. Every experimental design decision, every claim and every number reported in the paper is the authors'. All analysis code is released and every figure ships with a CSV of its exact plotted values, as Appendix R: Run record sets out, so each reported number can be recomputed from the released per-item outputs independently of how the code was written. No AI system was used to generate, paraphrase or annotate any item in the corpus, and no model output in the paper comes from an assistant: the three arms under audit are called through the typed interface described in Appendix A: The typed question.
```

> Two things to settle before you paste E1.
>
> 1. Edit the first sentence to match what you actually did. If you also used an
>    assistant for literature search, say so; if you did not, drop the mention.
> 2. The elaboration above points at the Ethics Statement, but the Ethics
>    Statement does not currently say anything about AI assistance. Either drop
>    that opening pointer and let the checklist text stand alone, which the form
>    explicitly allows, or tell me and I will add the sentence to the Ethics
>    Statement so the pointer is true.

### Readers

```
Everyone
```

Unless you have a reason to restrict it. Making the checklist public is the
default and is the stronger signal for a paper whose contribution is a release.
