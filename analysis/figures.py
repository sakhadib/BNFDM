#!/usr/bin/env python3
"""BNFDM figures for the Laya arm.

Reads analysis/tables/*.csv (and a few raws for distributions), writes
analysis/figures/*.pdf + *.png + a sibling *.csv holding the plotted data.

No titles anywhere — captions belong in the paper. ACL two-column widths.
No Bangla glyphs: matplotlib cannot shape Bengali and no Bengali font is
installed here, so all labels are English/ASCII by design.
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
RES = ROOT / "LAYA_RUN" / "results" / "laya-ml"
FIG = ROOT / "analysis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

S.use_style()
CHANCE = 0.25
FAM_SHORT = {"D-rand": "random", "D-surf": "surface", "D-sem": "tag",
             "D-lit": "+literal", "D-hard": "hard"}
ORDER = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]


def err(df, col, lo, hi):
    return np.vstack([df[col] - df[lo], df[hi] - df[col]])


# ════════════════════════════════════════════════════════════════════
def f1_ladder():
    d = pd.read_csv(TBL / "e1_summary.csv").set_index("family").loc[ORDER].reset_index()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.25),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(d))
    cols = [S.C if not b else S.M for b in d.below_chance]
    ax.bar(x, d.accuracy, width=0.62, color=cols, edgecolor=S.K, linewidth=0.6)
    ax.errorbar(x, d.accuracy, yerr=err(d, "accuracy", "acc_lo", "acc_hi"),
                fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    ax.axhline(CHANCE, color=S.K, ls="--", lw=1.0)
    ax.text(len(d) - 0.45, CHANCE + 0.008, "chance", fontsize=7, ha="right", color=S.K)
    for xi, v in zip(x, d.accuracy):
        ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", fontsize=7, color=S.K)
    ax.set_xticks(x); ax.set_xticklabels([FAM_SHORT[f] for f in d.family],
                                         rotation=15, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.46)
    ax.set_xlabel("distractor set")

    lit = d[d.LAR.notna()]
    xl = np.arange(len(lit)); wd = 0.36
    ax2.bar(xl - wd / 2, lit.LAR, width=wd, color=S.M, edgecolor=S.K,
            linewidth=0.6, label="picks own literal gloss")
    ax2.errorbar(xl - wd / 2, lit.LAR, yerr=err(lit, "LAR", "LAR_lo", "LAR_hi"),
                 fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    ax2.bar(xl + wd / 2, lit.accuracy, width=wd, color=S.C, edgecolor=S.K,
            linewidth=0.6, label="picks gold figurative gloss")
    ax2.errorbar(xl + wd / 2, lit.accuracy,
                 yerr=err(lit, "accuracy", "acc_lo", "acc_hi"),
                 fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    for xi, l, a in zip(xl, lit.LAR, lit.accuracy):
        ax2.text(xi - wd / 2, l + 0.022, f"{l:.3f}", ha="center", fontsize=7)
        ax2.text(xi + wd / 2, a + 0.022, f"{a:.3f}", ha="center", fontsize=7)
    ax2.axhline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax2.set_xticks(xl); ax2.set_xticklabels([FAM_SHORT[f] for f in lit.family])
    ax2.set_ylabel("share of items"); ax2.set_ylim(0, 0.86)
    ax2.set_xlabel("distractor set")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=1, fontsize=7)
    S.save(fig, FIG / "f1_distractor_ladder.pdf", d)


def f2_literal_capture():
    dec = pd.read_csv(TBL / "e1_outcome_decomposition.csv")
    raw = pd.read_csv(RES / "e1_ladder" / "raw.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.15))

    x = np.arange(len(dec))
    b = np.zeros(len(dec))
    for col, c, lab in [("chose_gold", S.C, "gold figurative gloss"),
                        ("chose_literal", S.M, "own literal gloss"),
                        ("chose_other", S.Y, "another idiom's gloss")]:
        ax.bar(x, dec[col], bottom=b, width=0.5, color=c, edgecolor=S.K,
               linewidth=0.6, label=lab)
        for xi, (v, bb) in enumerate(zip(dec[col], b)):
            if v > 0.06:
                ax.text(xi, bb + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if c != S.Y else S.K)
        b = b + dec[col].values
    ax.set_xticks(x); ax.set_xticklabels([FAM_SHORT[f] for f in dec.family])
    ax.set_ylabel("share of items"); ax.set_ylim(0, 1)
    ax.set_xlabel("distractor set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=1, fontsize=7)

    g = raw[(raw.family == "D-lit") & raw.lit_key.notna()]
    bins = np.linspace(0, 1, 41)
    ax2.hist(g.p_gold, bins=bins, histtype="step", lw=1.3, color=S.C,
             label="$p$(gold gloss)")
    ax2.hist(g.p_literal, bins=bins, histtype="step", lw=1.3, color=S.M,
             label="$p$(literal gloss)")
    ax2.axvline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax2.set_xlabel("probability assigned"); ax2.set_ylabel("items")
    ax2.legend(loc="upper right")
    S.save(fig, FIG / "f2_literal_capture.pdf", dec)


def f3_confidence():
    c = pd.read_csv(TBL / "e5_confidence_by_outcome.csv")
    rel = pd.read_csv(TBL / "e5_reliability.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.15))

    lab = {"correct": "correct", "chose_literal": "captured by literal",
           "other_wrong": "other error"}
    x = np.arange(len(c))
    cols = [S.C, S.M, S.Y]
    ax.bar(x, c.mean_conf, width=0.55, color=cols, edgecolor=S.K, linewidth=0.6)
    ax.errorbar(x, c.mean_conf, yerr=err(c, "mean_conf", "conf_lo", "conf_hi"),
                fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    for xi, (v, n) in enumerate(zip(c.mean_conf, c.n)):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=S.K)
        ax.text(xi, 0.02, f"n={n:,}", ha="center", fontsize=6.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels([lab[o] for o in c.outcome],
                                         rotation=15, ha="right")
    ax.set_ylabel("reported confidence"); ax.set_ylim(0, 0.72)

    for fam, col in zip(ORDER, [S.C, S.C_L, S.Y, S.M, S.K]):
        g = rel[rel.family == fam]
        ax2.plot(g.mean_conf, g.accuracy, "o-", color=col, mfc="white",
                 mew=1.0, label=FAM_SHORT[fam], ms=3.5)
    ax2.plot([0, 1], [0, 1], color=S.K_L, ls="--", lw=0.9, zorder=0)
    ax2.set_xlabel("reported confidence"); ax2.set_ylabel("empirical accuracy")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=7)
    S.save(fig, FIG / "f3_confidence_anti_diagnostic.pdf", c)


def f4_position():
    e1p = pd.read_csv(TBL / "e1_position_bias.csv")
    e7p = pd.read_csv(TBL / "e7_position_bias.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.15))

    x = np.arange(4); wd = 0.38
    ax.bar(x - wd / 2, e1p.gold_share, width=wd, color=S.C_L, edgecolor=S.K,
           linewidth=0.6, label="gold placed here")
    ax.bar(x + wd / 2, e1p.selection_share, width=wd, color=S.M, edgecolor=S.K,
           linewidth=0.6, label="model selects here")
    ax.axhline(0.25, color=S.K, ls="--", lw=0.9)
    for xi, v in zip(x, e1p.selection_share):
        ax.text(xi + wd / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(e1p.key)
    ax.set_xlabel("option slot"); ax.set_ylabel("share")
    ax.set_ylim(0, 0.50); ax.legend(loc="upper left", fontsize=7, ncol=2,
                                    bbox_to_anchor=(0.0, 1.0))

    ax2.bar(x, e7p.accuracy_when_gold_here, width=0.55, color=S.C,
            edgecolor=S.K, linewidth=0.6)
    ax2.errorbar(x, e7p.accuracy_when_gold_here,
                 yerr=err(e7p, "accuracy_when_gold_here", "acc_lo", "acc_hi"),
                 fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    for xi, v in zip(x, e7p.accuracy_when_gold_here):
        ax2.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(e7p.key)
    ax2.set_xlabel("slot holding the gold gloss")
    ax2.set_ylabel("accuracy"); ax2.set_ylim(0, 0.26)
    S.save(fig, FIG / "f4_position_bias.pdf",
           e1p.assign(source="E1").merge(e7p.assign(source="E7"), on="position",
                                         suffixes=("_e1", "_e7")))


def f5_order_instability():
    h = pd.read_csv(TBL / "e7_instability_hist.csv")
    raw = pd.read_csv(RES / "e7_order" / "raw.csv")
    rng = raw.groupby("id").p_gold.agg(lambda s: s.max() - s.min())
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.1))
    fig.subplots_adjust(wspace=0.32)

    ax.bar(h.n_distinct_pred, h.share, width=0.55, color=S.M, edgecolor=S.K,
           linewidth=0.6)
    for xi, v, n in zip(h.n_distinct_pred, h.share, h["items"]):
        ax.text(xi, v + 0.015, f"{v:.3f}\n(n={n})", ha="center", fontsize=7)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("distinct answers / 24 orderings")
    ax.set_ylabel("share of items"); ax.set_ylim(0, 1.0)

    ax2.hist(rng.values, bins=40, color=S.C, edgecolor=S.K, linewidth=0.4)
    ax2.axvline(rng.mean(), color=S.M, lw=1.2,
                label=f"mean {rng.mean():.3f}")
    ax2.set_xlabel("within-item range of $p$(gold)")
    ax2.set_ylabel("items"); ax2.legend(loc="upper right")
    S.save(fig, FIG / "f5_order_instability.pdf",
           pd.DataFrame({"id": rng.index, "p_gold_range": rng.values}))


def f6_crosslingual():
    a = pd.read_csv(TBL / "e4_arms.csv")
    loc = pd.read_csv(TBL / "e4_localisation.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.3),
                                  gridspec_kw={"width_ratios": [1.3, 1]})
    short = {"bn_ml": "BN idiom\nmultiling.", "en_ml": "EN equiv.\nmultiling.",
             "en_en": "EN equiv.\nEnglish", "bn_en": "BN idiom\nEnglish"}
    x = np.arange(len(a))
    cols = [S.M, S.C, S.C, S.K_L]
    ax.bar(x, a.accuracy, width=0.6, color=cols, edgecolor=S.K, linewidth=0.6)
    ax.errorbar(x, a.accuracy, yerr=err(a, "accuracy", "acc_lo", "acc_hi"),
                fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    ax.axhline(CHANCE, color=S.K, ls="--", lw=1.0)
    ax.text(-0.42, CHANCE + 0.018, "chance", fontsize=7, ha="left")
    for xi, v in zip(x, a.accuracy):
        ax.text(xi, v + 0.016, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([short[k] for k in a.arm], fontsize=7.5)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.85)
    ax.set_xlabel("input language / checkpoint")

    order = ["both_right", "bangla_gap", "english_gap", "both_wrong"]
    lab = {"both_right": "both right", "bangla_gap": "English only",
           "english_gap": "Bangla only", "both_wrong": "both wrong"}
    loc = loc.set_index("cell").loc[order].reset_index()
    cols2 = [S.C, S.M, S.Y, S.K_L]
    b = 0.0
    for r, c in zip(loc.itertuples(), cols2):
        ax2.bar(0, r.share, bottom=b, width=0.5, color=c, edgecolor=S.K,
                linewidth=0.6, label=f"{lab[r.cell]} ({r.share:.3f})")
        b += r.share
    ax2.set_xlim(-0.6, 0.6); ax2.set_xticks([])
    ax2.set_ylabel("share of items"); ax2.set_ylim(0, 1)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5)
    S.save(fig, FIG / "f6_crosslingual.pdf", a)


def f7_unit_integrity():
    d = pd.read_csv(TBL / "e3_summary.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.4))
    fig.subplots_adjust(wspace=0.30)
    short = {"V0_intact": "intact", "V1_shuffle": "shuffled", "V2_reverse": "reversed",
             "V3_delete1": "-1 word", "V4_substitute1": "sub 1 word",
             "V5_in_sentence": "in sentence"}
    x = np.arange(len(d))
    cols = [S.K] + [S.C] * 4 + [S.M]
    ax.bar(x, d.accuracy, width=0.6, color=cols, edgecolor=S.K, linewidth=0.6)
    ax.errorbar(x, d.accuracy, yerr=err(d, "accuracy", "acc_lo", "acc_hi"),
                fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    ax.axhline(d.accuracy.iloc[0], color=S.K, ls=":", lw=0.9)
    for xi, v in zip(x, d.accuracy):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([short[v] for v in d.variant],
                                         rotation=25, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.31)

    ax2.bar(x, d.answer_agreement_with_intact, width=0.6, color=S.Y,
            edgecolor=S.K, linewidth=0.6)
    ax2.errorbar(x, d.answer_agreement_with_intact,
                 yerr=err(d, "answer_agreement_with_intact", "agree_lo", "agree_hi"),
                 fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    for xi, v in zip(x, d.answer_agreement_with_intact):
        ax2.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels([short[v] for v in d.variant],
                                           rotation=25, ha="right")
    ax2.set_ylabel("agreement with intact"); ax2.set_ylim(0, 1.12)
    S.save(fig, FIG / "f7_unit_integrity.pdf", d)


def f8_options():
    d = pd.read_csv(TBL / "e6_summary.csv")
    sig = pd.read_csv(TBL / "e6_signal_retained.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.35))
    short = {"O0_full_bn": "full BN", "O1_trunc6": "truncated",
             "O2_scrambled": "scrambled", "O3_english": "English",
             "O4_labels": "no semantics"}
    x = np.arange(len(d))
    cols = [S.K, S.C, S.C, S.Y, S.M]
    ax.bar(x, d.accuracy, width=0.6, color=cols, edgecolor=S.K, linewidth=0.6)
    ax.errorbar(x, d.accuracy, yerr=err(d, "accuracy", "acc_lo", "acc_hi"),
                fmt="none", ecolor=S.K, elinewidth=0.8, capsize=2)
    ax.axhline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax.text(-0.45, CHANCE + 0.008, "chance", fontsize=7, ha="left")
    for xi, v in zip(x, d.accuracy):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([short[f] for f in d.form],
                                         rotation=25, ha="right")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 0.45)
    ax.set_xlabel("option text")

    ax2.bar(x, sig.semantic_signal_retained, width=0.6, color=S.C,
            edgecolor=S.K, linewidth=0.6)
    for xi, v in zip(x, sig.semantic_signal_retained):
        ax2.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
    ax2b = ax2.twinx()
    ax2b.plot(x, d.answer_agreement_with_full, "o--", color=S.M, mfc="white",
              mew=1.1, label="answer agreement")
    ax2b.set_ylabel("answer agreement with full", color=S.M)
    ax2b.tick_params(axis="y", colors=S.M); ax2b.set_ylim(0, 1.05)
    ax2b.grid(False)
    ax2b.legend(loc="lower left", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels([short[f] for f in d.form],
                                           rotation=25, ha="right")
    ax2.set_ylabel("semantic signal retained"); ax2.set_ylim(0, 1.12)
    S.save(fig, FIG / "f8_option_ablation.pdf", d.merge(sig[["form", "semantic_signal_retained"]], on="form"))


def f9_risk_coverage():
    rc = pd.read_csv(TBL / "e5_risk_coverage.csv")
    cal = pd.read_csv(TBL / "e5_calibration.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.15))
    fig.subplots_adjust(wspace=0.30)
    for fam, col in zip(ORDER, [S.C, S.C_L, S.Y, S.M, S.K]):
        g = rc[rc.family == fam].sort_values("coverage")
        ax.plot(g.coverage, g.accuracy, "-", color=col, label=FAM_SHORT[fam])
        base = cal.loc[cal.family == fam, "accuracy"].iat[0]
        ax.axhline(base, color=col, ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("coverage (most confident first)")
    ax.set_ylabel("accuracy on covered items")
    ax.set_xlim(0.05, 1.0); ax.legend(loc="upper right", fontsize=7)

    x = np.arange(len(cal))
    wd = 0.38
    ax2.bar(x - wd / 2, cal.aurc_conf, width=wd, color=S.M, edgecolor=S.K,
            linewidth=0.6, label="gate on confidence")
    ax2.bar(x + wd / 2, cal.aurc_random, width=wd, color=S.K_L, edgecolor=S.K,
            linewidth=0.6, label="random gate")
    ax2.set_xticks(x); ax2.set_xticklabels([FAM_SHORT[f] for f in cal.family],
                                           rotation=20, ha="right")
    ax2.set_ylabel("AURC (lower is better)")
    ax2.legend(loc="upper left", fontsize=7)
    S.save(fig, FIG / "f9_selective_prediction.pdf", cal)


def f10_stratified():
    nw = pd.read_csv(TBL / "e1_by_n_words.csv")
    fr = pd.read_csv(TBL / "e1_by_frequency.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.15))
    for fam, col in zip(ORDER, [S.C, S.C_L, S.Y, S.M, S.K]):
        g = nw[(nw.family == fam) & (nw.n >= 50)].sort_values("n_words")
        ax.plot(g.n_words, g.accuracy, "o-", color=col, mfc="white", mew=1.0,
                ms=3.5, label=FAM_SHORT[fam])
    ax.axhline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax.set_xlabel("idiom length (words)"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.10, 0.47)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=3)

    freqs = ["very common", "common", "rare", "very rare"]
    fr = fr[fr.frequency.isin(freqs)]
    xs = np.arange(len(freqs)); wd = 0.16
    for i, (fam, col) in enumerate(zip(ORDER, [S.C, S.C_L, S.Y, S.M, S.K])):
        g = fr[fr.family == fam].set_index("frequency").reindex(freqs)
        ax2.bar(xs + (i - 2) * wd, g.accuracy.values, width=wd, color=col,
                edgecolor=S.K, linewidth=0.4, label=FAM_SHORT[fam])
    ax2.axhline(CHANCE, color=S.K, ls="--", lw=0.9)
    ax2.set_xticks(xs); ax2.set_xticklabels(freqs, rotation=20, ha="right")
    ax2.set_ylabel("accuracy")
    S.save(fig, FIG / "f10_stratified.pdf", nw)


if __name__ == "__main__":
    for fn in [f1_ladder, f2_literal_capture, f3_confidence, f4_position,
               f5_order_instability, f6_crosslingual, f7_unit_integrity,
               f8_options, f9_risk_coverage, f10_stratified]:
        print(fn.__name__)
        fn()
    print("\nDone ->", FIG)
