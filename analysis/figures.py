#!/usr/bin/env python3
"""BNFDM figures, both arms.

Laya-multilingual is drawn in cyan, Lod-lille in magenta, throughout.
No titles (captions belong in the paper). No Bangla glyphs — matplotlib cannot
shape Bengali and no Bengali font is installed; qualitative examples live in
FINDINGS.md.

Each save() writes .pdf + .png preview + .csv of exactly the plotted data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import bnfdm_style as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TBL = ROOT / "analysis" / "tables"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
S.use_style()

CHANCE = 0.25
ARMS = ["laya-ml", "lod-lille-0.6b"]
ALAB = {"laya-ml": "Laya-ml (322M)", "lod-lille-0.6b": "Lod-lille (0.6B)"}
ACOL = {"laya-ml": S.C, "lod-lille-0.6b": S.M}
RAW = {"laya-ml": ROOT / "LAYA_RUN" / "results" / "laya-ml",
       "lod-lille-0.6b": ROOT / "LOD_RUN" / "results" / "lod-lille-0.6b"}

FAM = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]
FAM_SHORT = {"D-rand": "random", "D-surf": "surface", "D-sem": "tag",
             "D-lit": "+literal", "D-hard": "hard"}
VAR = ["V0_intact", "V1_shuffle", "V2_reverse", "V3_delete1",
       "V4_substitute1", "V5_in_sentence"]
VAR_SHORT = {"V0_intact": "intact", "V1_shuffle": "shuffled", "V2_reverse": "reversed",
             "V3_delete1": "-1 word", "V4_substitute1": "sub 1 word",
             "V5_in_sentence": "in sentence"}
FORM = ["O0_full_bn", "O1_trunc6", "O2_scrambled", "O3_english", "O4_labels"]
FORM_SHORT = {"O0_full_bn": "full BN", "O1_trunc6": "truncated",
              "O2_scrambled": "scrambled", "O3_english": "English",
              "O4_labels": "no semantics"}


def T(arm, name):
    return pd.read_csv(TBL / arm / f"{name}.csv")


def X(tbl, key, order):
    return tbl.set_index(key).reindex(order).reset_index()


def eb(ax, x, d, col, lo, hi, **kw):
    ax.errorbar(x, d[col], yerr=np.vstack([d[col] - d[lo], d[hi] - d[col]]),
                fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2, **kw)


# ════════════════════════════════════════════════════════════════════
def f1_ladder():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.3))
    fig.subplots_adjust(wspace=0.28)
    x = np.arange(len(FAM)); wd = 0.38
    data = []
    for i, arm in enumerate(ARMS):
        d = X(T(arm, "e1_summary"), "family", FAM)
        off = (i - 0.5) * wd
        ax.bar(x + off, d.accuracy, width=wd, color=ACOL[arm], edgecolor=S.K,
               linewidth=0.6, label=ALAB[arm])
        eb(ax, x + off, d, "accuracy", "acc_lo", "acc_hi")
        for xi, v in zip(x + off, d.accuracy):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=6)
        data.append(d)
    ax.axhline(CHANCE, color=S.K, ls="--", lw=1.0)
    ax.text(-0.52, CHANCE + 0.012, "chance", fontsize=7, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([FAM_SHORT[f] for f in FAM],
                                         rotation=15, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.58)
    ax.set_xlabel("distractor set")
    ax.legend(loc="upper right", fontsize=7)

    lf = ["D-lit", "D-hard"]
    xl = np.arange(len(lf))
    for i, arm in enumerate(ARMS):
        d = X(T(arm, "e1_summary"), "family", lf)
        off = (i - 0.5) * wd
        ax2.bar(xl + off, d.LAR, width=wd, color=ACOL[arm], edgecolor=S.K,
                linewidth=0.6, label=ALAB[arm])
        eb(ax2, xl + off, d, "LAR", "LAR_lo", "LAR_hi")
        for xi, v in zip(xl + off, d.LAR):
            ax2.text(xi, v + 0.018, f"{v:.3f}", ha="center", fontsize=7)
    ax2.axhline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax2.set_xticks(xl); ax2.set_xticklabels([FAM_SHORT[f] for f in lf])
    ax2.set_ylabel("Literal Attraction Rate"); ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("distractor set")
    S.save(fig, FIG / "f1_distractor_ladder.pdf", pd.concat(data))


def f2_literal_capture():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.3))
    fig.subplots_adjust(wspace=0.28)
    rows, labels = [], []
    for arm in ARMS:
        d = T(arm, "e1_outcome_decomposition")
        for f in ["D-lit", "D-hard"]:
            r = d[d.family == f].iloc[0]
            rows.append(r); labels.append(f"{FAM_SHORT[f]}\n{ALAB[arm].split()[0]}")
    dec = pd.DataFrame(rows)
    x = np.arange(len(dec)); b = np.zeros(len(dec))
    for col, c, lab in [("chose_gold", S.C, "gold figurative gloss"),
                        ("chose_literal", S.M, "own literal gloss"),
                        ("chose_other", S.Y, "another idiom's gloss")]:
        ax.bar(x, dec[col], bottom=b, width=0.62, color=c, edgecolor=S.K,
               linewidth=0.6, label=lab)
        for xi, (v, bb) in enumerate(zip(dec[col], b)):
            if v > 0.07:
                ax.text(xi, bb + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if c != S.Y else S.K)
        b = b + dec[col].values
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("share of items"); ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), fontsize=6.5, ncol=1)

    bins = np.linspace(0, 1, 41)
    for arm, ls in zip(ARMS, ["-", "--"]):
        g = pd.read_csv(RAW[arm] / "e1_ladder" / "raw.csv")
        g = g[(g.family == "D-lit") & g.lit_key.notna()]
        ax2.hist(g.p_gold, bins=bins, histtype="step", lw=1.3, ls=ls,
                 color=S.C, density=True, label=f"$p$(gold) {ALAB[arm].split()[0]}")
        ax2.hist(g.p_literal, bins=bins, histtype="step", lw=1.3, ls=ls,
                 color=S.M, density=True, label=f"$p$(literal) {ALAB[arm].split()[0]}")
    ax2.axvline(CHANCE, color=S.K, ls=":", lw=0.9)
    ax2.set_xlabel("probability assigned"); ax2.set_ylabel("density")
    ax2.legend(loc="upper center", fontsize=6)
    S.save(fig, FIG / "f2_literal_capture.pdf", dec)


def f3_confidence():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.25))
    fig.subplots_adjust(wspace=0.28)
    OUT = ["correct", "chose_literal", "other_wrong"]
    lab = {"correct": "correct", "chose_literal": "captured", "other_wrong": "other error"}
    x = np.arange(len(OUT)); wd = 0.38
    data = []
    for i, arm in enumerate(ARMS):
        d = X(T(arm, "e5_confidence_by_outcome"), "outcome", OUT)
        off = (i - 0.5) * wd
        ax.bar(x + off, d.mean_conf, width=wd, color=ACOL[arm], edgecolor=S.K,
               linewidth=0.6, label=ALAB[arm])
        eb(ax, x + off, d, "mean_conf", "conf_lo", "conf_hi")
        for xi, v in zip(x + off, d.mean_conf):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.5)
        data.append(d)
    ax.set_xticks(x); ax.set_xticklabels([lab[o] for o in OUT], rotation=12, ha="right")
    ax.set_ylabel("confidence in chosen option"); ax.set_ylim(0, 0.95)
    ax.legend(loc="upper left", fontsize=7)

    for arm, ls in zip(ARMS, ["-", "--"]):
        rel = T(arm, "e5_reliability")
        for fam, c in [("D-rand", S.C), ("D-lit", S.M)]:
            g = rel[rel.family == fam]
            ax2.plot(g.mean_conf, g.accuracy, "o" + ls, color=c, mfc="white",
                     mew=1.0, ms=3.2,
                     label=f"{FAM_SHORT[fam]} · {ALAB[arm].split()[0]}")
    ax2.plot([0, 1], [0, 1], color=S.K_L, ls=":", lw=0.9, zorder=0)
    ax2.set_xlabel("reported confidence"); ax2.set_ylabel("empirical accuracy")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=6)
    S.save(fig, FIG / "f3_confidence_anti_diagnostic.pdf", pd.concat(data))


def f4_position():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.2))
    fig.subplots_adjust(wspace=0.28)
    x = np.arange(4); wd = 0.38
    data = []
    for i, arm in enumerate(ARMS):
        d = T(arm, "e1_position_bias").sort_values("position")
        off = (i - 0.5) * wd
        ax.bar(x + off, d.selection_share, width=wd, color=ACOL[arm],
               edgecolor=S.K, linewidth=0.6, label=ALAB[arm])
        for xi, v in zip(x + off, d.selection_share):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5)
        data.append(d)
    ax.axhline(0.25, color=S.K, ls="--", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_xlabel("option slot"); ax.set_ylabel("share of selections")
    ax.set_ylim(0, 0.47); ax.legend(loc="upper right", fontsize=7)

    for i, arm in enumerate(ARMS):
        d = T(arm, "e7_position_bias").sort_values("position")
        off = (i - 0.5) * wd
        ax2.bar(x + off, d.accuracy_when_gold_here, width=wd, color=ACOL[arm],
                edgecolor=S.K, linewidth=0.6, label=ALAB[arm])
        eb(ax2, x + off, d, "accuracy_when_gold_here", "acc_lo", "acc_hi")
        for xi, v in zip(x + off, d.accuracy_when_gold_here):
            ax2.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.5)
    ax2.set_xticks(x); ax2.set_xticklabels(["A", "B", "C", "D"])
    ax2.set_xlabel("slot holding the gold gloss"); ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 0.27)
    S.save(fig, FIG / "f4_position_bias.pdf", pd.concat(data))


def f5_order_instability():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.2))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(1, 5); wd = 0.38
    data = []
    for i, arm in enumerate(ARMS):
        h = T(arm, "e7_instability_hist").set_index("n_distinct_option")
        h = h.reindex(range(1, 5)).fillna(0).reset_index()
        off = (i - 0.5) * wd
        ax.bar(x + off, h.share, width=wd, color=ACOL[arm], edgecolor=S.K,
               linewidth=0.6, label=ALAB[arm])
        for xi, v in zip(x + off, h.share):
            if v > 0.01:
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.5)
        h["arm"] = arm; data.append(h)
    ax.set_xticks(x)
    ax.set_xlabel("distinct glosses chosen / 24 orderings")
    ax.set_ylabel("share of items"); ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", fontsize=7)

    for arm in ARMS:
        per = T(arm, "e7_per_item")
        ax2.hist(per.p_gold_range, bins=np.linspace(0, 0.9, 46), histtype="step",
                 lw=1.3, color=ACOL[arm], density=True, label=ALAB[arm])
        ax2.axvline(per.p_gold_range.mean(), color=ACOL[arm], ls=":", lw=1.0)
    ax2.set_xlabel("within-item range of $p$(gold)")
    ax2.set_ylabel("density"); ax2.legend(loc="upper right", fontsize=7)
    S.save(fig, FIG / "f5_order_instability.pdf", pd.concat(data))


def f6_crosslingual():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.4),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(wspace=0.32)
    rows, labels, cols = [], [], []
    short = {"bn_ml": "BN\nmulti", "en_ml": "EN\nmulti", "en_en": "EN\nEnglish",
             "bn_en": "BN\nEnglish", "bn": "BN", "en": "EN"}
    for arm in ARMS:
        d = T(arm, "e4_arms")
        for r in d.itertuples():
            rows.append(r)
            labels.append(f"{short[r.condition]}\n{ALAB[arm].split()[0]}")
            cols.append(ACOL[arm] if r.condition.startswith("bn") else S.C_L
                        if arm == "laya-ml" else S.M_L)
    a = pd.DataFrame(rows)
    x = np.arange(len(a))
    ax.bar(x, a.accuracy, width=0.65, color=cols, edgecolor=S.K, linewidth=0.6)
    eb(ax, x, a, "accuracy", "acc_lo", "acc_hi")
    ax.axhline(CHANCE, color=S.K, ls="--", lw=1.0)
    ax.text(-0.55, CHANCE + 0.015, "chance", fontsize=7, ha="left")
    for xi, v in zip(x, a.accuracy):
        ax.text(xi, v + 0.016, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.99)

    order = ["both_right", "bangla_gap", "english_gap", "both_wrong"]
    lab = {"both_right": "both right", "bangla_gap": "English only",
           "english_gap": "Bangla only", "both_wrong": "both wrong"}
    cols2 = [S.C, S.M, S.Y, S.K_L]
    xb = np.arange(len(ARMS)); data = []
    for i, arm in enumerate(ARMS):
        loc = X(T(arm, "e4_localisation"), "cell", order).fillna(0)
        b = 0.0
        for r, c in zip(loc.itertuples(), cols2):
            ax2.bar(i, r.share, bottom=b, width=0.55, color=c, edgecolor=S.K,
                    linewidth=0.6, label=lab[r.cell] if i == 0 else None)
            if r.share > 0.08:
                ax2.text(i, b + r.share / 2, f"{r.share:.2f}", ha="center",
                         va="center", fontsize=6.5,
                         color="white" if c != S.Y else S.K)
            b += r.share
        loc["arm"] = arm; data.append(loc)
    ax2.set_xticks(xb); ax2.set_xticklabels([ALAB[a].split()[0] for a in ARMS],
                                            fontsize=7.5)
    ax2.set_ylabel("share of items"); ax2.set_ylim(0, 1)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    S.save(fig, FIG / "f6_crosslingual.pdf", a)


def f7_unit_integrity():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.45))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(VAR)); wd = 0.38
    data = []
    for i, arm in enumerate(ARMS):
        d = X(T(arm, "e3_summary"), "variant", VAR)
        off = (i - 0.5) * wd
        ax.bar(x + off, d.accuracy, width=wd, color=ACOL[arm], edgecolor=S.K,
               linewidth=0.6, label=ALAB[arm])
        eb(ax, x + off, d, "accuracy", "acc_lo", "acc_hi")
        ax.axhline(d.accuracy.iloc[0], color=ACOL[arm], ls=":", lw=0.8)
        ax2.bar(x + off, d.answer_agreement_with_intact, width=wd,
                color=ACOL[arm], edgecolor=S.K, linewidth=0.6, label=ALAB[arm])
        eb(ax2, x + off, d, "answer_agreement_with_intact", "agree_lo", "agree_hi")
        data.append(d)
    for a_, yl, lb in [(ax, 0.33, "accuracy"), (ax2, 1.12, "agreement with intact")]:
        a_.set_xticks(x); a_.set_xticklabels([VAR_SHORT[v] for v in VAR],
                                             rotation=28, ha="right", fontsize=7)
        a_.set_ylim(0, yl); a_.set_ylabel(lb)
    ax.legend(loc="upper left", fontsize=7)
    S.save(fig, FIG / "f7_unit_integrity.pdf", pd.concat(data))


def f8_options():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.45))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(FORM)); wd = 0.38
    data = []
    for i, arm in enumerate(ARMS):
        d = X(T(arm, "e6_summary"), "form", FORM)
        off = (i - 0.5) * wd
        ax.bar(x + off, d.accuracy, width=wd, color=ACOL[arm], edgecolor=S.K,
               linewidth=0.6, label=ALAB[arm])
        eb(ax, x + off, d, "accuracy", "acc_lo", "acc_hi")
        for xi, v in zip(x + off, d.accuracy):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6)
        ax2.plot(x, d.semantic_signal_retained, "o-", color=ACOL[arm],
                 mfc="white", mew=1.1, ms=4, label=ALAB[arm])
        data.append(d)
    ax.axhline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax.text(-0.55, CHANCE + 0.008, "chance", fontsize=7, ha="left")
    for a_ in (ax, ax2):
        a_.set_xticks(x); a_.set_xticklabels([FORM_SHORT[f] for f in FORM],
                                             rotation=28, ha="right", fontsize=7)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.60)
    ax.set_xlabel("option text"); ax.legend(loc="upper right", fontsize=7)
    ax2.set_ylabel("semantic signal retained"); ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(0, color=S.K_L, ls=":", lw=0.8)
    ax2.set_xlabel("option text")
    S.save(fig, FIG / "f8_option_ablation.pdf", pd.concat(data))


def f9_selective():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.2))
    fig.subplots_adjust(wspace=0.30)
    for arm, ls in zip(ARMS, ["-", "--"]):
        rc = T(arm, "e5_risk_coverage")
        for fam, c in [("D-rand", S.C), ("D-lit", S.M)]:
            g = rc[rc.family == fam].sort_values("coverage")
            ax.plot(g.coverage, g.accuracy, ls, color=c, lw=1.3,
                    label=f"{FAM_SHORT[fam]} · {ALAB[arm].split()[0]}")
    ax.set_xlabel("coverage (most confident first)")
    ax.set_ylabel("accuracy on covered items")
    ax.set_xlim(0.05, 1.0); ax.legend(loc="upper right", fontsize=6.5)

    x = np.arange(len(FAM)); wd = 0.38
    data = []
    for i, arm in enumerate(ARMS):
        d = X(T(arm, "e5_calibration"), "family", FAM)
        off = (i - 0.5) * wd
        ax2.bar(x + off, d.aurc_conf - d.aurc_random, width=wd, color=ACOL[arm],
                edgecolor=S.K, linewidth=0.6, label=ALAB[arm])
        data.append(d)
    ax2.axhline(0, color=S.K, lw=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels([FAM_SHORT[f] for f in FAM],
                                           rotation=20, ha="right", fontsize=7)
    ax2.set_ylabel("AURC $-$ random-gate AURC")
    ax2.text(0.02, 0.95, "above 0 = gating is worse than not gating",
             transform=ax2.transAxes, fontsize=6.5, va="top")
    ax2.legend(loc="lower left", fontsize=7)
    S.save(fig, FIG / "f9_selective_prediction.pdf", pd.concat(data))


def f10_paired():
    p = pd.read_csv(TBL / "_cross" / "paired_mcnemar.csv")
    e1 = p[p.experiment == "e1_ladder"].set_index("group").reindex(FAM).reset_index()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.3),
                                  gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(e1))
    cols = [S.C if v < 0 else S.M for v in e1.delta_lod_minus_laya]
    ax.bar(x, e1.delta_lod_minus_laya, width=0.6, color=cols, edgecolor=S.K,
           linewidth=0.6)
    for xi, (v, pv) in enumerate(zip(e1.delta_lod_minus_laya, e1.mcnemar_p)):
        star = "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < .05 else "n.s."
        ax.text(xi, v + (0.006 if v > 0 else -0.016), f"{v:+.3f}\n{star}",
                ha="center", fontsize=6.5,
                va="bottom" if v > 0 else "top")
    ax.axhline(0, color=S.K, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([FAM_SHORT[f] for f in FAM],
                                         rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Lod $-$ Laya accuracy"); ax.set_ylim(-0.16, 0.20)
    ax.set_xlabel("distractor set")

    h = pd.read_csv(TBL / "_cross" / "headline_both_arms.csv")
    metrics = [("literal_attraction_rate", "literal attraction"),
               ("order_unstable_share", "order instability"),
               ("language_gap", "EN $-$ BN gap"),
               ("bangla_gap_share", "right in EN only")]
    xm = np.arange(len(metrics)); wd = 0.38
    for i, arm in enumerate(ARMS):
        r = h[h.arm == arm].iloc[0]
        ax2.bar(xm + (i - 0.5) * wd, [r[m] for m, _ in metrics], width=wd,
                color=ACOL[arm], edgecolor=S.K, linewidth=0.6, label=ALAB[arm])
        for xi, (m, _) in zip(xm + (i - 0.5) * wd, metrics):
            ax2.text(xi, r[m] + 0.018, f"{r[m]:.2f}", ha="center", fontsize=6.5)
    ax2.set_xticks(xm); ax2.set_xticklabels([l for _, l in metrics],
                                            rotation=22, ha="right", fontsize=7)
    ax2.set_ylabel("value"); ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=7)
    S.save(fig, FIG / "f10_cross_model.pdf", p)


def f11_stratified():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.2))
    fig.subplots_adjust(wspace=0.28)
    nw = T("laya-ml", "e1_by_n_words")
    for fam, col in zip(FAM, [S.C, S.C_L, S.Y, S.M, S.K]):
        g = nw[(nw.family == fam) & (nw.n >= 50)].sort_values("n_words")
        ax.plot(g.n_words, g.accuracy, "o-", color=col, mfc="white", mew=1.0,
                ms=3.5, label=FAM_SHORT[fam])
    ax.axhline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax.set_xlabel("idiom length (words)"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.10, 0.47)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=6.5, ncol=3)

    sub = pd.read_csv(TBL / "_cross" / "subset_representativeness.csv")
    sub = sub.set_index("family").reindex(FAM).reset_index()
    x = np.arange(len(sub)); wd = 0.38
    ax2.bar(x - wd / 2, sub.laya_full, width=wd, color=S.C, edgecolor=S.K,
            linewidth=0.6, label="Laya, full 8,876")
    ax2.bar(x + wd / 2, sub.laya_subset, width=wd, color=S.C_L, edgecolor=S.K,
            linewidth=0.6, label="Laya, the 888 subset")
    eb(ax2, x + wd / 2, sub, "laya_subset", "sub_lo", "sub_hi")
    for xi, v in zip(x, sub.drift):
        ax2.text(xi, 0.012, f"{v:+.3f}", ha="center", fontsize=6, color=S.K)
    ax2.set_xticks(x); ax2.set_xticklabels([FAM_SHORT[f] for f in FAM],
                                           rotation=20, ha="right", fontsize=7)
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0, 0.45)
    ax2.legend(loc="upper right", fontsize=6.5)
    S.save(fig, FIG / "f11_stratified_and_subset.pdf", sub)


if __name__ == "__main__":
    for fn in [f1_ladder, f2_literal_capture, f3_confidence, f4_position,
               f5_order_instability, f6_crosslingual, f7_unit_integrity,
               f8_options, f9_selective, f10_paired, f11_stratified]:
        print(fn.__name__)
        fn()
    print("\nDone ->", FIG)
