# Literature Review: Bangla Idiom Decision Models

This companion contains 54 entries in the same order as `references.bib`. Metadata was compiled from the completed search; records abbreviated with “and others” should be expanded from the linked canonical source before camera-ready submission.

# A. Idiom and figurative language processing

This literature establishes idiomaticity as a contextual, multilingual, multiword-expression problem. Existing resources support detection, representation, and explanation, but few expose an expert literal gloss as a controlled competing answer.

## haagsma2020magpie
**MAGPIE: A Large Corpus of Potentially Idiomatic Expressions** — Haagsma, Hessel, Bos, Johan, Nissim, Malvina, Proceedings of the Twelfth Language Resources and Evaluation Conference 2020. [link](https://aclanthology.org/2020.lrec-1.35/)

**What it does:** Introduces a large English corpus of potentially idiomatic expressions in naturally occurring contexts. It separates expression types from contextual instances for idiomaticity research.

**Why it matters to this study:** BACKGROUND ONLY — Establishes a canonical corpus precedent, while this study adds expert literal glosses and controlled Bangla counterfactuals.

**Use it in:** Related Work.

## chakrabarty2022flute
**FLUTE: Figurative Language Understanding through Textual Explanations** — Chakrabarty, Tuhin, Saakyan, Arkadiy, Ghosh, Debanjan, Muresan, Smaranda, Proceedings of EMNLP 2022 2022. [link](https://aclanthology.org/2022.emnlp-main.481/)

**What it does:** Evaluates figurative-language inference and textual explanation across idiom, metaphor, simile, and sarcasm. The benchmark makes literal explanations part of figurative understanding.

**Why it matters to this study:** CONTRASTS WITH — FLUTE solicits generated explanations; this study diagnoses non-generative models only through input interventions, informing findings 1 and 5.

**Use it in:** Related Work.

## tayyarmadabushi2022semeval
**SemEval-2022 Task 2: Multilingual Idiomaticity Detection and Sentence Embedding** — Tayyar Madabushi, Harish, Gow-Smith, Edward, Scarton, Carolina, Villavicencio, Aline, Proceedings of SemEval 2022 2022. [link](https://aclanthology.org/2022.semeval-1.27/)

**What it does:** Benchmarks literal-versus-idiomatic classification in context and multilingual sentence representation, including zero-shot conditions. It makes multilingual idiomaticity a standardized evaluation target.

**Why it matters to this study:** SUPPORTS — Establishes literal/idiomatic discrimination as core, but does not measure capture by a supplied literal-gloss option as in finding 1.

**Use it in:** Related Work.

## shwartz2019pain
**Still a Pain in the Neck: Evaluating Text Representations on Lexical Composition** — Shwartz, Vered, Dagan, Ido, Transactions of the Association for Computational Linguistics 2019. [link](https://aclanthology.org/Q19-1028/)

**What it does:** Uses controlled tasks to test lexical composition in multiword expressions. Strong contextual representations still miss properties needed for robust compositional analysis.

**Why it matters to this study:** SUPPORTS — Motivates finding 5: word-level competence does not establish unit-level idiom representation.

**Use it in:** Related Work.

## gracia2018lidioms
**LIdioms: A Multilingual Linked Idioms Data Set** — Gracia, Jorge, Kernerman, Ilan, Bosque-Gil, Julia, Proceedings of LREC 2018 2018. [link](https://arxiv.org/abs/1802.08148)

**What it does:** Represents idioms as multilingual linked data across five languages, connecting lexical entries, meanings, and equivalents. It targets reuse in multilingual applications.

**Why it matters to this study:** BACKGROUND ONLY — Demonstrates structured multilingual idiom representation, while the Bangla resource contributes richer per-item counterfactual fields.

**Use it in:** Related Work.

# B. Literal versus figurative interpretation bias

Psycholinguistics treats idiom processing as competition among lexical, compositional, contextual, and stored figurative information, moderated by familiarity and decomposability. Direct model-side quantification of attraction to a gold literal gloss remains thin.

## cacciari1988comprehension
**The Comprehension of Idioms** — Cacciari, Cristina, Tabossi, Patrizia, Journal of Memory and Language 1988. [link](https://doi.org/10.1016/0749-596X(88)90014-9)

**What it does:** Proposes the configuration hypothesis: processing begins compositionally until the idiomatic configuration is recognized. Experiments test recognition point and contextual facilitation.

**Why it matters to this study:** BACKGROUND ONLY — Supplies a psycholinguistic mechanism for literal/figurative competition but not a model-level Literal Attraction Rate.

**Use it in:** Discussion.

## gibbs1989kick
**How to Kick the Bucket and Not Decompose: Analyzability and Idiom Processing** — Gibbs, Raymond W., Nayak, Nandini P., Cutting, Cooper, Journal of Memory and Language 1989. [link](https://doi.org/10.1016/0749-596X(89)90014-4)

**What it does:** Distinguishes decomposable from nondecomposable idioms and tests how analyzability affects interpretation. It helped establish decomposability as graded.

**Why it matters to this study:** SUPPORTS — Motivates stratifying literal attraction and shuffle invariance by decomposability when interpreting findings 1 and 5.

**Use it in:** Discussion.

## nunberg1994idioms
**Idioms** — Nunberg, Geoffrey, Sag, Ivan A., Wasow, Thomas, Language 1994. [link](https://doi.org/10.2307/416483)

**What it does:** Separates idiomatically combining expressions from idiomatic phrases and links semantic composition to syntactic flexibility. Idioms emerge as a heterogeneous class.

**Why it matters to this study:** BACKGROUND ONLY — Provides the taxonomy needed to avoid overgeneralizing finding 5 beyond the tested Bagdhara subset.

**Use it in:** Related Work.

## titone1999compositional
**On the Compositional and Noncompositional Nature of Idiomatic Expressions** — Titone, Debra A., Connine, Cynthia M., Journal of Pragmatics 1999. [link](https://doi.org/10.1016/S0378-2166(99)00008-9)

**What it does:** Connects familiarity, predictability, literality, and decomposability to online idiom comprehension. Its hybrid account allows lexical and compositional information to contribute.

**Why it matters to this study:** SUPPORTS — Motivates item-conditioned analyses of finding 1 rather than treating literal attraction as uniform.

**Use it in:** Discussion.

## oh2026tug
**Tug-of-War between Idioms’ Figurative and Literal Interpretations in LLMs** — Oh, Soyoung, Huang, Xinting, Pink, Mathis, Hahn, Michael, Demberg, Vera, Proceedings of EACL 2026 2026. [link](https://aclanthology.org/2026.eacl-long.135/)

**What it does:** Directly studies competition between literal and figurative idiom interpretations in language models through controlled comparisons. It is the closest located model-side predecessor.

**Why it matters to this study:** SUPPORTS — Closest precedent to finding 1; this study differs by defining option-choice Literal Attraction Rate and auditing non-generative probability models.

**Use it in:** Related Work.

# C. Multiple-choice evaluation artifacts

Multiple-choice evaluation is sensitive to answer position, option labels, token priors, and distractor design. Most permutation and debiasing work concerns prompted generators, leaving typed option scorers underexplored.

## zheng2024robust
**Large Language Models Are Not Robust Multiple Choice Selectors** — Zheng, Chujie, Zhou, Hao, Meng, Fandong, Zhou, Jie, Huang, Minlie, International Conference on Learning Representations 2024. [link](https://arxiv.org/abs/2309.03882)

**What it does:** Shows multiple-choice performance changes under answer permutations because of option-label priors. PriDe estimates and removes prior selection bias at inference time.

**Why it matters to this study:** SUPPORTS — Strongly supports finding 3 behaviorally, while this study localizes a larger architecture-specific effect in option scorers.

**Use it in:** Related Work.

## pezeshkpour2024order
**Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions** — Pezeshkpour, Pouya, Hruschka, Estevam, Findings of NAACL 2024 2024. [link](https://aclanthology.org/2024.findings-naacl.130/)

**What it does:** Systematically permutes answer options and documents substantial performance instability. A single order can therefore misstate capability.

**Why it matters to this study:** METHOD I BORROW — Directly motivates the all-24-permutation protocol and within-item probability ranges used in finding 3.

**Use it in:** Method.

## wei2024selection
**Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models** — Wei, Sheng-Lun, Wu, Cheng-Kuang, Chen, Hsin-Hsi, Findings of ACL 2024 2024. [link](https://aclanthology.org/2024.findings-acl.333/)

**What it does:** Separates sensitivity to option order from sensitivity to answer-label tokens and evaluates mitigation strategies. Formatting can alter measured ability.

**Why it matters to this study:** SUPPORTS — Supplies position/token/selection-bias terminology; finding 3 extends it to typed option-scoring APIs.

**Use it in:** Related Work.

## zhao2021calibrate
**Calibrate Before Use: Improving Few-Shot Performance of Language Models** — Zhao, Zihao, Wallace, Eric, Feng, Shi, Klein, Dan, Singh, Sameer, Proceedings of ICML 2021 2021. [link](https://proceedings.mlr.press/v139/zhao21c.html)

**What it does:** Identifies majority-label, recency, and common-token biases in few-shot classification. Contextual calibration estimates content-free preferences and adjusts probabilities.

**Why it matters to this study:** CONTRASTS WITH — It corrects prompt-induced label priors; finding 3 shows architecture and placement effects in direct option distributions.

**Use it in:** Related Work.

## jia2017adversarial
**Adversarial Examples for Evaluating Reading Comprehension Systems** — Jia, Robin, Liang, Percy, Proceedings of EMNLP 2017 2017. [link](https://aclanthology.org/D17-1215/)

**What it does:** Adds fluent but answer-irrelevant adversarial sentences to reading-comprehension passages. Large drops expose superficial matching strategies.

**Why it matters to this study:** METHOD I BORROW — Replacing one controlled distractor while holding other content fixed follows the same diagnostic principle behind findings 1 and 2.

**Use it in:** Method.

# D. Calibration and selective prediction

Proper scoring rules incentivize truthful probabilities in expectation, while calibration and selective prediction test whether confidence is usable under deployment conditions. An inverted risk–coverage curve is stronger than ordinary overconfidence.

## brier1950verification
**Verification of Forecasts Expressed in Terms of Probability** — Brier, Glenn W., Monthly Weather Review 1950. [link](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)

**What it does:** Introduces the quadratic probability score now called the Brier score. It evaluates the complete probabilistic forecast, not only top-class accuracy.

**Why it matters to this study:** BACKGROUND ONLY — Grounds evaluation of full distributions and separates proper-score training from safe confidence under literal traps.

**Use it in:** Method.

## gneiting2007proper
**Strictly Proper Scoring Rules, Prediction, and Estimation** — Gneiting, Tilmann, Raftery, Adrian E., Journal of the American Statistical Association 2007. [link](https://doi.org/10.1198/016214506000001437)

**What it does:** Develops the theory of proper scoring rules, where truthful probability reporting uniquely optimizes expected score. It unifies logarithmic, quadratic, and related rules.

**Why it matters to this study:** BACKGROUND ONLY — Defines vendor training claims while clarifying that expectation-level properness does not prevent conditional literal capture.

**Use it in:** Related Work.

## guo2017calibration
**On Calibration of Modern Neural Networks** — Guo, Chuan, Pleiss, Geoff, Sun, Yu, Weinberger, Kilian Q., Proceedings of ICML 2017 2017. [link](https://proceedings.mlr.press/v70/guo17a.html)

**What it does:** Shows that accurate neural networks can be miscalibrated and compares post-hoc corrections. Single-parameter temperature scaling is a strong practical baseline on the studied data.

**Why it matters to this study:** CONTRASTS WITH — A monotone global temperature cannot repair the confidence ranking inversion in finding 2.

**Use it in:** Related Work.

## nixon2019measuring
**Measuring Calibration in Deep Learning** — Nixon, Jeremy, Dusenberry, Michael W., Zhang, Linchuan, Jerfel, Ghassen, Tran, Dustin, CVPR Workshops 2019 2019. [link](https://arxiv.org/abs/1904.01685)

**What it does:** Compares calibration metrics and binning schemes, showing conclusions can depend on estimator design. It recommends evaluation beyond a single ECE value.

**Why it matters to this study:** METHOD I BORROW — Motivates reliability diagrams and metric sensitivity checks for finding 2.

**Use it in:** Method.

## kumar2019verified
**Verified Uncertainty Calibration** — Kumar, Ananya, Liang, Percy S., Ma, Tengyu, Advances in Neural Information Processing Systems 32 2019. [link](https://proceedings.neurips.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html)

**What it does:** Analyzes weaknesses of finite-sample empirical calibration estimates and develops calibration procedures with statistical guarantees. Calibration measurement itself has uncertainty.

**Why it matters to this study:** METHOD I BORROW — Motivates uncertainty intervals around the downward reliability relation in finding 2.

**Use it in:** Method.

## geifman2017selective
**Selective Classification for Deep Neural Networks** — Geifman, Yonatan, El-Yaniv, Ran, Advances in Neural Information Processing Systems 30 2017. [link](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)

**What it does:** Constructs selective classifiers that abstain on uncertain cases while targeting desired risk. It formalizes coverage-risk tradeoffs for neural classifiers.

**Why it matters to this study:** CONTRASTS WITH — Standard selective behavior lowers risk with lower coverage; finding 2 reverses that relationship.

**Use it in:** Related Work.

## geifman2019selectivenet
**SelectiveNet: A Deep Neural Network with an Integrated Reject Option** — Geifman, Yonatan, El-Yaniv, Ran, Proceedings of ICML 2019 2019. [link](https://proceedings.mlr.press/v97/geifman19a.html)

**What it does:** Jointly learns prediction and selection under a target-coverage constraint. Abstention is integrated into training instead of applied only after training.

**Why it matters to this study:** CONTRASTS WITH — Finding 2 warns that even a dedicated answerability head can rank literal-trap failures above correct items.

**Use it in:** Discussion.

## ovadia2019trust
**Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty under Dataset Shift** — Ovadia, Yaniv, Fertig, Emily, Ren, Jie, Nado, Zachary, Sculley, D., Nowozin, Sebastian, Dillon, Joshua, Lakshminarayanan, Balaji, Snoek, Jasper, Advances in Neural Information Processing Systems 32 2019. [link](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)

**What it does:** Evaluates predictive uncertainty under increasing dataset shift across data modalities and approximate Bayesian methods. Uncertainty quality commonly degrades away from training conditions.

**Why it matters to this study:** SUPPORTS — Literal-gloss insertion is a structured shift; finding 2 shows an extreme reversal rather than ordinary degradation.

**Use it in:** Discussion.

## jiang2021know
**How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering** — Jiang, Zhengbao, Araki, Jun, Ding, Haibo, Neubig, Graham, Transactions of the Association for Computational Linguistics 2021. [link](https://aclanthology.org/2021.tacl-1.57/)

**What it does:** Tests whether generative LM probabilities track correctness in question answering and evaluates calibration interventions. Raw probabilities are not reliably calibrated for this purpose.

**Why it matters to this study:** SUPPORTS — Establishes QA confidence failure; finding 2 adds anti-diagnostic ranking and worse-than-random selective risk on a semantic trap.

**Use it in:** Related Work.

## traub2024selective
**Overcoming Common Flaws in the Evaluation of Selective Classification Systems** — Traub, Julian, others, Advances in Neural Information Processing Systems 37 2024. [link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract-Conference.html)

**What it does:** Diagnoses common errors in selective-classification evaluation and clarifies use of risk-coverage curves and AURC. It emphasizes fair multi-threshold comparison.

**Why it matters to this study:** METHOD I BORROW — Supports AURC with random and oracle references instead of one arbitrary confidence threshold.

**Use it in:** Method.

# E. Behavioral and black-box interpretability

Behavioral tests, contrast sets, counterfactual augmentation, and stress tests infer capability from controlled input-output changes. Faithfulness research explains why plausible rationales should not be treated as causal accounts.

## ribeiro2020checklist
**Beyond Accuracy: Behavioral Testing of NLP Models with CheckList** — Ribeiro, Marco Tulio, Wu, Tongshuang, Guestrin, Carlos, Singh, Sameer, Proceedings of ACL 2020 2020. [link](https://aclanthology.org/2020.acl-main.442/)

**What it does:** Adapts software testing into capability, invariance, and directional-expectation tests for NLP. It reveals failures hidden by aggregate held-out accuracy.

**Why it matters to this study:** METHOD I BORROW — Distractor, language, permutation, and shuffle manipulations are behavioral tests for findings 1, 3, 4, and 5.

**Use it in:** Method.

## gardner2020contrast
**Evaluating Models’ Local Decision Boundaries via Contrast Sets** — Gardner, Matt, Artzi, Yoav, Basmova, Victoria, Berant, Jonathan, others, Findings of EMNLP 2020 2020. [link](https://aclanthology.org/2020.findings-emnlp.117/)

**What it does:** Creates small meaning-changing edits to existing test instances and relabels them to probe local decision boundaries. Standard test performance often overstates robustness.

**Why it matters to this study:** METHOD I BORROW — Gold literal glosses and language or word-order counterfactuals instantiate this local-boundary logic.

**Use it in:** Method.

## kaushik2020difference
**Learning the Difference That Makes a Difference with Counterfactually-Augmented Data** — Kaushik, Divyansh, Hovy, Eduard, Lipton, Zachary C., International Conference on Learning Representations 2020. [link](https://openreview.net/forum?id=Sklgs0NFvr)

**What it does:** Uses human minimal edits that change labels to construct counterfactually augmented data. Training on such edits can reduce reliance on spurious correlations.

**Why it matters to this study:** METHOD I BORROW — Supports expert-validated counterfactual fields as causal probes across all five findings.

**Use it in:** Method.

## naik2018stress
**Stress Test Evaluation for Natural Language Inference** — Naik, Aakanksha, Ravichander, Abhilasha, Sadeh, Norman, Rose, Carolyn, Neubig, Graham, Proceedings of COLING 2018 2018. [link](https://aclanthology.org/C18-1198/)

**What it does:** Constructs targeted NLI stress tests for competence, distraction, noise, and lexical inference. The tests expose weaknesses hidden by ordinary benchmark averages.

**Why it matters to this study:** SUPPORTS — Justifies treating literal distractors and shuffles as controlled stress conditions rather than replacements for ordinary accuracy.

**Use it in:** Related Work.

## deyoung2020eraser
**ERASER: A Benchmark to Evaluate Rationalized NLP Models** — DeYoung, Jay, Jain, Sarthak, Rajani, Nazneen Fatema, Lehman, Eric, Xiong, Caiming, Socher, Richard, Wallace, Byron C., Proceedings of ACL 2020 2020. [link](https://aclanthology.org/2020.acl-main.408/)

**What it does:** Collects tasks with human rationales and evaluates task performance plus rationale quality. It operationalizes comprehensiveness and sufficiency through interventions.

**Why it matters to this study:** CONTRASTS WITH — Those metrics use token subsets; this study intervenes on whole semantic objects because idiom meaning is non-additive.

**Use it in:** Related Work.

## jacovi2020faithfulness
**Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?** — Jacovi, Alon, Goldberg, Yoav, Proceedings of ACL 2020 2020. [link](https://aclanthology.org/2020.acl-main.386/)

**What it does:** Separates human plausibility from faithfulness to actual model computation and develops evaluation criteria. Convincing explanations need not be causal accounts.

**Why it matters to this study:** SUPPORTS — Motivates refusing self-rationalization as evidence and using input-output interventions for all findings.

**Use it in:** Method.

## turpin2023unfaithful
**Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting** — Turpin, Miles, Michael, Julian, Perez, Ethan, Bowman, Samuel R., Advances in Neural Information Processing Systems 36 2023. [link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html)

**What it does:** Injects answer-position, suggested-answer, and social biases; chain-of-thought often rationalizes biased answers without identifying the causal cue. Reported accuracy drops reach 36 percent.

**Why it matters to this study:** SUPPORTS — Shows self-explanations can hide option-order causes, strengthening the intervention-only design for findings 2 and 3.

**Use it in:** Related Work.

# F. Why feature attribution is not used

Feature attribution allocates prediction credit under assumptions about baselines, perturbations, and additivity. Interaction-aware work shows that first-order word scores are incomplete when meaning is carried by a non-additive phrase.

## shapley1953value
**A Value for n-Person Games** — Shapley, Lloyd S., Contributions to the Theory of Games II 1953. [link](https://doi.org/10.1515/9781400881970-018)

**What it does:** Defines a unique allocation rule from efficiency, symmetry, dummy, and additivity axioms. It averages marginal contributions over coalitions.

**Why it matters to this study:** BACKGROUND ONLY — Per-word attribution is an axiomatic allocation, not discovery of a unique semantic decomposition.

**Use it in:** Related Work.

## ribeiro2016lime
**“Why Should I Trust You?”: Explaining the Predictions of Any Classifier** — Ribeiro, Marco Tulio, Singh, Sameer, Guestrin, Carlos, Proceedings of KDD 2016 2016. [link](https://doi.org/10.1145/2939672.2939778)

**What it does:** Fits sparse local surrogate models around individual predictions using perturbed samples. It also selects representative explanations for broader inspection.

**Why it matters to this study:** CONTRASTS WITH — A locally additive surrogate can obscure higher-order idiom interactions, motivating whole-expression tests for finding 5.

**Use it in:** Related Work.

## lundberg2017shap
**A Unified Approach to Interpreting Model Predictions** — Lundberg, Scott M., Lee, Su-In, Advances in Neural Information Processing Systems 30 2017. [link](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

**What it does:** Unifies feature-attribution methods as additive explanation models and assigns Shapley values to local predictions. Guarantees depend on the chosen value function and background.

**Why it matters to this study:** CONTRASTS WITH — Allocating a joint idiom effect among words does not establish independent word-level figurative content.

**Use it in:** Related Work.

## jain2019attention
**Attention is not Explanation** — Jain, Sarthak, Wallace, Byron C., Proceedings of NAACL 2019 2019. [link](https://aclanthology.org/N19-1357/)

**What it does:** Compares attention with gradient importance and constructs alternative attention distributions with similar predictions. Raw attention is not automatically a faithful causal explanation.

**Why it matters to this study:** SUPPORTS — Justifies not interpreting encoder attention as evidence of holistic idiom processing in finding 5.

**Use it in:** Related Work.

## wiegreffe2019attention
**Attention is not not Explanation** — Wiegreffe, Sarah, Pinter, Yuval, Proceedings of EMNLP-IJCNLP 2019 2019. [link](https://aclanthology.org/D19-1002/)

**What it does:** Argues that attention claims require precise explanatory targets and stronger baselines. Attention may sometimes carry information without being universally explanatory.

**Why it matters to this study:** BACKGROUND ONLY — Prevents an overbroad dismissal while preserving the narrower limitation relevant to finding 5.

**Use it in:** Limitations.

## adebayo2018sanity
**Sanity Checks for Saliency Maps** — Adebayo, Julius, Gilmer, Justin, Muelly, Michael, Goodfellow, Ian, Hardt, Moritz, Kim, Been, Advances in Neural Information Processing Systems 31 2018. [link](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)

**What it does:** Randomizes parameters and labels to test whether saliency maps depend on learned behavior. Several methods remain visually similar under randomization.

**Why it matters to this study:** SUPPORTS — Motivates outcome-sensitive behavioral interventions over persuasive attribution visualizations.

**Use it in:** Related Work.

## hooker2019roar
**A Benchmark for Interpretability Methods in Deep Neural Networks** — Hooker, Sara, Erhan, Dumitru, Kindermans, Pieter-Jan, Kim, Been, Advances in Neural Information Processing Systems 32 2019. [link](https://proceedings.neurips.cc/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)

**What it does:** ROAR removes features ranked important, retrains the model, and measures performance loss. Retraining addresses distribution shift caused by naive deletion.

**Why it matters to this study:** METHOD I BORROW — Highlights why word deletion is problematic for idioms and why natural gold counterfactuals are preferable.

**Use it in:** Method.

## sundararajan2020shapleytaylor
**The Shapley Taylor Interaction Index** — Sundararajan, Mukund, Dhamdhere, Kedar, Agarwal, Ashish, Proceedings of ICML 2020 2020. [link](https://proceedings.mlr.press/v119/sundararajan20a.html)

**What it does:** Extends Shapley attribution to interactions up to a chosen order and axiomatizes joint effects. Ordinary first-order attributions collapse these terms.

**Why it matters to this study:** SUPPORTS — Makes the interaction objection rigorous: non-compositional idioms require joint terms, so word scores are incomplete.

**Use it in:** Related Work.

## tsang2023faithshap
**Faith-Shap: The Faithful Shapley Interaction Index** — Tsang, Michael, Rambhatla, Sirisha, Liu, Yan, Journal of Machine Learning Research 2023. [link](https://www.jmlr.org/papers/v24/22-0202.html)

**What it does:** Derives an interaction attribution method with a faithfulness objective and connects interaction order to approximation quality. Main effects can miss predictive structure.

**Why it matters to this study:** SUPPORTS — Provides an interaction-aware basis for calling per-token attribution under-specified on non-additive idioms.

**Use it in:** Related Work.

# G. Low-resource and Bangla/Bengali NLP

Multilingual benchmarks reveal uneven transfer and motivate language-specific resources and culturally grounded annotation. Few combine this with expert literal counterfactuals for Bangla idioms.

## bhattacharjee2022banglabert
**BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla** — Bhattacharjee, Abhik, Hasan, Tahmid, Samin, Kazi, Islam, Md Saiful, Rahman, M. Sohel, Iqbal, Anindya, Shahriyar, Rifat, Findings of NAACL 2022 2022. [link](https://aclanthology.org/2022.findings-naacl.22/)

**What it does:** Introduces a Bangla-focused encoder and broad language-understanding benchmark. Language-specific pretraining improves multiple Bangla tasks over generic multilingual baselines.

**Why it matters to this study:** SUPPORTS — Contextualizes finding 4 as a language-resource and representation failure rather than generic figurative difficulty.

**Use it in:** Related Work.

## kakwani2020indicnlpsuite
**IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages** — Kakwani, Divyanshu, Kunchukuttan, Anoop, Golla, Satish, Gokul, N. C., Bhattacharyya, Avik, Khapra, Mitesh M., Kumar, Pratyush, Findings of EMNLP 2020 2020. [link](https://aclanthology.org/2020.findings-emnlp.445/)

**What it does:** Provides corpora, benchmarks, and pretrained models for Indic languages including Bengali. It demonstrates the value of targeted language resources.

**Why it matters to this study:** SUPPORTS — Supplies regional context for the controlled English–Bangla gap in finding 4.

**Use it in:** Related Work.

## hu2020xtreme
**XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization** — Hu, Junjie, Ruder, Sebastian, Siddhant, Aditya, Neubig, Graham, Firat, Orhan, Johnson, Melvin, Proceedings of ICML 2020 2020. [link](https://proceedings.mlr.press/v119/hu20b.html)

**What it does:** Evaluates cross-lingual representations across languages and task families. Performance varies widely by language even with the model fixed.

**Why it matters to this study:** SUPPORTS — Provides the standard transfer framing for finding 4’s paired English–Bangla comparison.

**Use it in:** Related Work.

## ruder2021xtremer
**XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation** — Ruder, Sebastian, Constant, Noah, Botha, Jan, Siddhant, Aditya, Firat, Orhan, others, Proceedings of EMNLP 2021 2021. [link](https://aclanthology.org/2021.emnlp-main.802/)

**What it does:** Expands multilingual evaluation with harder tasks and more diagnostic transfer settings. Aggregate scores can hide uneven language-level behavior.

**Why it matters to this study:** METHOD I BORROW — Supports paired outcomes and right-in-English/wrong-in-Bangla transition rates for finding 4.

**Use it in:** Method.

## adelani2021masakhaner
**MasakhaNER: Named Entity Recognition for African Languages** — Adelani, David Ifeoluwa, Abbott, Jade, Neubig, Graham, others, Transactions of the Association for Computational Linguistics 2021. [link](https://aclanthology.org/2021.tacl-1.66/)

**What it does:** Develops culturally grounded data through participatory native-speaker annotation across African languages. It emphasizes local linguistic expertise in low-resource benchmarking.

**Why it matters to this study:** SUPPORTS — Validates the expert, culturally grounded annotation philosophy behind the Bangla Bagdhara resource.

**Use it in:** Related Work.

# H. Specialized classifiers versus generative LLMs

Encoder classifiers and lightweight systems provide efficient alternatives to text-generating LMs. Direct audits of typed, caller-supplied option scorers remain scarce.

## ng2002generative
**On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes** — Ng, Andrew Y., Jordan, Michael I., Advances in Neural Information Processing Systems 14 2002. [link](https://proceedings.neurips.cc/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html)

**What it does:** Compares generative and discriminative learning through asymptotic and finite-sample behavior. Model class and objective produce distinct tradeoffs on the same labels.

**Why it matters to this study:** BACKGROUND ONLY — Frames option-scoring decision models versus text generators without assuming either paradigm is universally superior.

**Use it in:** Related Work.

## devlin2019bert
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** — Devlin, Jacob, Chang, Ming-Wei, Lee, Kenton, Toutanova, Kristina, Proceedings of NAACL 2019 2019. [link](https://aclanthology.org/N19-1423/)

**What it does:** Establishes bidirectional Transformer encoders adapted with small task heads. Encoder-based classification and pair scoring became standard alternatives to generation.

**Why it matters to this study:** BACKGROUND ONLY — Supplies the architectural lineage for compact encoder-plus-option-scorer systems.

**Use it in:** Related Work.

## raffel2020t5
**Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** — Raffel, Colin, Shazeer, Noam, Roberts, Adam, others, Journal of Machine Learning Research 2020. [link](https://www.jmlr.org/papers/v21/20-074.html)

**What it does:** Casts many NLP tasks into one text-to-text format and studies transfer at scale. Classification is implemented through generated textual labels.

**Why it matters to this study:** CONTRASTS WITH — Represents the paradigm rejected by decision models; findings 2 and 3 show non-generation does not remove calibration or option artifacts.

**Use it in:** Related Work.

## tunstall2022setfit
**Efficient Few-Shot Learning Without Prompts** — Tunstall, Lewis, Reimers, Nils, Jo, Unso Eun Seo, others, arXiv preprint arXiv:2209.11055 2022. [link](https://arxiv.org/abs/2209.11055)

**What it does:** Fine-tunes sentence transformers contrastively and adds a lightweight classifier, avoiding prompt-based label generation. It targets efficient few-shot deployment.

**Why it matters to this study:** SUPPORTS — Establishes practical motivation for specialized classifiers while this study shows why they still require behavioral audits.

**Use it in:** Related Work.

# I. Closed and commercial API auditing

Black-box evaluation frameworks permit comparison when internals are inaccessible, but require documentation, raw-output preservation, and versioned evaluation. Endpoint drift limits exact reproducibility.

## bommasani2021foundation
**On the Opportunities and Risks of Foundation Models** — Bommasani, Rishi, Hudson, Drew A., Adeli, Ehsan, others, arXiv preprint arXiv:2108.07258 2021. [link](https://arxiv.org/abs/2108.07258)

**What it does:** Surveys capabilities, homogenization, evaluation, governance, and downstream dependence on opaque upstream models. Training data and choices may be inaccessible.

**Why it matters to this study:** BACKGROUND ONLY — Frames why behavioral evaluation is necessary for proprietary decision APIs.

**Use it in:** Introduction.

## liang2023helm
**Holistic Evaluation of Language Models** — Liang, Percy, Bommasani, Rishi, Lee, Tony, others, Transactions on Machine Learning Research 2023. [link](https://openreview.net/forum?id=iO4LZibEqW)

**What it does:** Proposes transparent multi-metric evaluation across scenarios, including accuracy, calibration, robustness, fairness, and efficiency. It records prompts and configurations for comparability.

**Why it matters to this study:** METHOD I BORROW — Supports byte-identical questions, multi-metric reporting, and explicit scenarios across open and closed models.

**Use it in:** Method.

## mitchell2019modelcards
**Model Cards for Model Reporting** — Mitchell, Margaret, Wu, Simone, Zaldivar, Andrew, others, Proceedings of FAT* 2019 2019. [link](https://doi.org/10.1145/3287560.3287596)

**What it does:** Proposes standardized documentation of intended use, evaluation conditions, performance, and limitations. Disaggregated testing is part of responsible reporting.

**Why it matters to this study:** SUPPORTS — Justifies reporting language, distractor family, option position, and confidence separately for a closed API.

**Use it in:** Limitations.

## pineau2021reproducibility
**Improving Reproducibility in Machine Learning Research: A Report from the NeurIPS 2019 Reproducibility Program** — Pineau, Joelle, Vincent-Lamarre, Philippe, Sinha, Koustuv, others, Journal of Machine Learning Research 2021. [link](https://www.jmlr.org/papers/v22/20-303.html)

**What it does:** Reports a conference-scale reproducibility program using checklists, code submission, and process changes. It distinguishes repeatable reporting from headline-score availability.

**Why it matters to this study:** SUPPORTS — Motivates endpoint dates, versioned requests, exact input bytes, and raw-response preservation for commercial APIs.

**Use it in:** Limitations.

# UNVERIFIED — CHECK BEFORE USING

No uncertain paper was promoted into the main list solely from memory. However, `traub2024selective` and entries using “and others” contain abbreviated author metadata and must be re-exported from the canonical page before camera-ready use. The MAGPIE LREC identifier, SemEval task identifier, DOI normalization for the Brier paper, and page ranges should also be checked against official exports.

# WHAT I COULD NOT FIND

- **Finding 1 — literal capture:** Oh et al. (2026) is the closest direct precedent, alongside psycholinguistic work on competition and decomposability. No located work defines the same intervention-based Literal Attraction Rate for non-generative probability models or reports the stronger-model/more-captured double dissociation. This appears to be a defensible narrow novelty claim.
- **Finding 2 — anti-diagnostic confidence:** Calibration failure under shift is established, but no located work reports this exact combination: correctness decreasing with confidence on literal traps, near-zero accuracy in the top bin, and confidence-gated AURC worse than random across different calibration philosophies. The combination appears unaddressed.
- **Finding 3 — architecture-specific option order:** Option-order bias is established for prompted generative LMs. No located work compares option-marker and independent-option scoring in small non-autoregressive decision models over all permutations. The architectural dissociation appears to be a gap.
- **Area A:** IDIX, Alankaar, KonIdioms, IMIL, and the exact PIE variant requested were not metadata-verified in the completed search record, so they were omitted rather than guessed.
- **Area G:** Published Bangla idiom benchmarks with expert literal glosses and gold counterfactual interventions appear thin. Add the official Bangla Bagdhara LREC 2026 entry once its canonical metadata is available.
- **Area H:** Evidence directly comparing small typed option scorers against generative LLMs is much thinner than conventional discriminative-versus-generative classification work.
- **Area I:** General black-box auditing guidance exists, but peer-reviewed methodology tailored to mutable commercial decision APIs remains sparse.
