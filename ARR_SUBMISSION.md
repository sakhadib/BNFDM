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

### Section A

**A1 potential risks**

```
Yes
```

**A1 elaboration**

```
Ethics Statement (unnumbered section after the Conclusion) and Limitations. The Ethics Statement covers the three risks that apply to this work: the source corpus is a published expert-annotated resource containing no personal data and no content about identifiable individuals; we report a named commercial endpoint's failure modes, so we state that the alias is mutable and that our figures are a dated snapshot of the 20260917 build rather than a standing property of the product; and we flag, against our own interest, both the pool selection effect in Section 5.1 and the lexical construction cue in Appendix J that accounts for part of the headline effect on one arm. Limitations states what the design cannot settle.
```

### Section B

**B use or create scientific artifacts**

```
Yes
```

**B1 cite creators of artifacts**

```
Yes
```

Cited in Section 3 (the Bangla Bagdhara corpus) and Section 4.4 with footnote 2,
which names each model arm by its released artifact and hosting location.

**B2 discuss the license for artifacts**

```
Yes
```

Ethics Statement: the source corpus is distributed under CC BY-SA 4.0, and we
release the instrument, the per-item outputs and the analysis code under the
same license, as its share-alike term requires. The two open checkpoints are
used under the terms their model cards declare. Appendix R repeats the release
license beside the repository link, and the repository carries the full CC BY-SA
4.0 text in its `LICENSE` file.

**B3 data contains personally identifying info**

```
Yes
```

Ethics Statement: the corpus is a published resource of idioms, it contains no
personal data and no content about identifiable individuals, and we add no
annotation, so nothing we release can carry it.

**B4 data contains offensive content**

```
N/A
```

We release option lists and probability vectors derived from a published
expert-annotated idiom corpus. We produce no new text and no generated output,
so there is no new surface on which objectionable content could appear.

**B5 documentation of artifacts**

```
Yes
```

Section 3 documents the corpus coverage, the 19-field schema and the 8,876 rows
that carry every field the instrument requires. Appendix A gives the exact
interface. Appendix R, "What is released", enumerates every released file: the
per-item outputs for every cell of every experiment, the five split manifests,
the three run environment records, the task configurations with the verbatim
instruction strings, the analysis scripts that produce every table, and a CSV of
the plotted values beside each figure.

**B6 statistics for data**

```
Yes
```

Section 3 and Table 4 give the item counts per arm (8,876 / 888 / 8,761), Section
5.1 and Appendix O give the shared 888-item subset and bound its drift at 0.016,
Section 5.1 and Appendix I give the 3,265-item matrix pool and Appendix J
documents how it was cut, and Appendix R gives the per-block call volume for the
commercial arm.

### Section C

**C computational experiments**

```
Yes
```

**C1 model size and budget**

```
Yes
```

Table 4 gives parameter counts (322M, 0.6B, closed) and per-call latency.
Appendix R gives the infrastructure (one Tesla T4, 15,360 MiB; Python 3.13.15),
the precision actually used on that card, and the commercial arm's full cost:
56,533 calls for $0.9348.

**C2 experimental setup and hyperparameters**

```
Yes
```

Section 4 and Appendix A give the typed-question setup in full: there is no
prompt template, no system message and no sampling parameter, so the option set
is the only intervention surface. Appendix R records the per-experiment seeds,
the split seed, the temperature settings on both open arms and the reason they
are untempered.

**C3 descriptive statistics**

```
Yes
```

All intervals are 95% percentile bootstrap over items, 2,000 resamples, fixed
seed 0; paired comparisons use McNemar's exact test. Appendix R, "Statistics",
states this once for the whole paper, and Appendix I tabulates the intervals for
the matrix cells. Every reported number is a single deterministic run, not a max
or a mean over runs, which is why the seeds are recorded per experiment.

**C4 parameters for packages**

```
Yes
```

Appendix R records the environment and the metric definitions we implement: ECE
over 15 equal-width bins, ACE as the adaptive equal-mass variant, multiclass
Brier, and AURC against a random gate on the same items. The analysis scripts
that compute them are part of the release.

### Section D

**D human subjects including annotators**

```
No
```

We ran no annotation and recruited no participants. The corpus is a
pre-existing published resource; its annotation was carried out and documented
by its own authors.

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
An AI coding assistant was used for the analysis and plotting code and for copy-editing passes over the manuscript. Every experimental design decision, every claim and every number reported in the paper is the authors'. All analysis code is released, and every figure ships with a CSV of its exact plotted values, so each reported number can be recomputed from the released per-item outputs independently of how the code was written. No AI system was used to generate, paraphrase or annotate any item in the corpus, and no model output in the paper comes from an assistant: the three arms under audit are called through the typed interface described in Appendix A.
```

> Adjust the first sentence to match what you actually did. If you also used an
> assistant for literature search, say so; if you did not, drop the mention.

### Readers

```
Everyone
```

Unless you have a reason to restrict it. Making the checklist public is the
default and is the stronger signal for a paper whose contribution is a release.
