#!/usr/bin/env python3
"""BNFDM paper figures.

Design rule: the figure carries data, the caption carries description. No
legends, no series names, no value labels, no in-axes text of any kind.
Axis labels and tick labels only.

Colour is the only identifier and it is fixed, undarkened:
    #27EBF5  Laya-ml
    #F127F5  Lod-lille
    #F5C827  Jev-1.13
    grey     control / neutral / reference series
    black    dashed chance line, bar edges, error bars
    hatched  condition not run, or English-language condition

Every mark is documented in analysis/figures/CAPTIONS.md. Bar order inside a
panel is always the model order Laya, Lod, Jev; panel order left to right is
the same. That is what the caption keys off.
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
FAM = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]
FAM_SHORT = {"D-rand": "random", "D-surf": "surface", "D-sem": "tag",
             "D-lit": "literal", "D-hard": "hard"}
VAR = ["V0_intact", "V1_shuffle", "V2_reverse", "V3_delete1",
       "V4_substitute1", "V5_in_sentence"]
VAR_SHORT = {"V0_intact": "intact", "V1_shuffle": "shuffled", "V2_reverse": "reversed",
             "V3_delete1": "$-$1 word", "V4_substitute1": "substituted",
             "V5_in_sentence": "in sentence"}
HATCH = "////"


def T(arm, name):
    return pd.read_csv(TBL / arm / f"{name}.csv")


def C(name):
    return pd.read_csv(TBL / "_cross" / f"{name}.csv")


def X(t, key, order):
    return t.set_index(key).reindex(order).reset_index()


def ticks(ax, x, labels, rot=0, fs=7):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center",
                       fontsize=fs)


def bars(ax, x, y, color, width=0.6, hatch=None, hatch_color=None):
    """Filled bar with no outline. Hatching is drawn in a saturated version of
    the same hue, because matplotlib takes the hatch colour from the edge."""
    kw = dict(width=width, color=color, zorder=2, linewidth=0)
    if hatch:
        kw.update(hatch=hatch, edgecolor=hatch_color or S.G1, linewidth=0)
    return ax.bar(x, y, **kw)


# ════════════════════════════════════════════════════════════════════
def fig1_literal_capture():
    """Bangla ladder (left) and literal attraction rate (right), two arms."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.05),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(wspace=0.24)
    x = np.arange(len(FAM))
    wd = 0.38
    data = []
    for i, arm in enumerate(S.ARMS2):
        d = X(T(arm, "e1_summary"), "family", FAM)
        off = (i - 0.5) * wd
        bars(ax, x + off, d.accuracy, S.ARM_COLOR[arm], wd)
        S.eb(ax, x + off, d, "accuracy", "acc_lo", "acc_hi")
        data.append(d.assign(arm=arm))
    S.chance(ax)
    ticks(ax, x, [FAM_SHORT[f] for f in FAM])
    ax.set_xlabel("distractor set")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.56)

    lf = ["D-lit", "D-hard"]
    xl = np.arange(len(lf))
    for i, arm in enumerate(S.ARMS2):
        d = X(T(arm, "e1_summary"), "family", lf)
        off = (i - 0.5) * wd
        bars(ax2, xl + off, d.LAR, S.ARM_COLOR[arm], wd)
        S.eb(ax2, xl + off, d, "LAR", "LAR_lo", "LAR_hi")
    S.chance(ax2)
    ticks(ax2, xl, [FAM_SHORT[f] for f in lf])
    ax2.set_xlabel("distractor set")
    ax2.set_ylabel("literal attraction rate")
    ax2.set_ylim(0, 1.0)
    S.save(fig, FIG / "fig1_literal_capture.pdf", pd.concat(data))


def fig2_capture_matrix():
    """LAR by model x task language on the shared English-literal pool.
    Solid bar = Bangla condition, hatched colour bar = English condition,
    hatched grey band spanning the panel = that cell was never run."""
    m = C("literal_capture_matrix")
    m = m[(m.family == "D-lit") & (m.population == "en_literal_pool")]
    fig, ax = plt.subplots(figsize=(S.W1, 1.95))
    wd = 0.34
    rows = []
    for ci, arm in enumerate(S.ARMS3):
        for si, lang in enumerate(["bn", "en"]):
            xi = ci + (si - 0.5) * wd
            r = m[(m.arm == arm) & (m.language == lang)]
            if len(r):
                v = float(r.LAR.iat[0])
                bars(ax, [xi], [v], S.ARM_COLOR[arm], wd,
                     hatch=HATCH if lang == "en" else None,
                     hatch_color=S.ARM_LINE[arm])
                rows.append({"arm": arm, "language": lang, "LAR": v,
                             "n": int(r.n.iat[0]), "measured": True})
            else:
                # drawn past the top of the axis so it cannot read as a bar
                ax.add_patch(plt.Rectangle((xi - wd / 2, 0), wd, 1.06,
                                           facecolor="#F4F4F4", edgecolor=S.G2,
                                           linewidth=0, hatch=HATCH, zorder=1,
                                           clip_on=False))
                rows.append({"arm": arm, "language": lang, "LAR": np.nan,
                             "n": 0, "measured": False})
    S.chance(ax)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.0)
    ticks(ax, range(3), [S.ARM_LABEL[a] for a in S.ARMS3])
    ax.set_ylabel("literal attraction rate")
    S.save(fig, FIG / "fig2_capture_matrix.pdf", pd.DataFrame(rows))


def fig3_three_way_bangla():
    """The one condition identical across all three models."""
    t = C("three_way_bangla_drand").set_index("arm").reindex(S.ARMS3).reset_index()
    fig, ax = plt.subplots(figsize=(S.W1, 1.95))
    x = np.arange(len(t))
    bars(ax, x, t.accuracy, [S.ARM_COLOR[a] for a in t.arm], 0.58)
    S.eb(ax, x, t, "accuracy", "acc_lo", "acc_hi")
    S.chance(ax)
    ticks(ax, x, [S.ARM_LABEL[a] for a in t.arm])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.82)
    S.save(fig, FIG / "fig3_three_way_bangla.pdf", t)


def fig4_confidence():
    """Reliability curves, one panel per model, grey diagonal = ideal."""
    fig, axes = plt.subplots(1, 3, figsize=(S.W2, 2.0), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    data = []
    for ax, arm in zip(axes, S.ARMS3):
        rel = T(arm, "e5_reliability")
        ax.plot([0, 1], [0, 1], color=S.G2, ls=(0, (3, 2)), lw=0.8, zorder=1)
        for fam, col, mk in [("D-rand", S.G1, "o"), ("D-lit", S.ARM_LINE[arm], "s")]:
            g = rel[rel.family == fam]
            if not len(g):
                continue
            ax.plot(g.mean_conf, g.accuracy, mk + "-", color=col,
                    mec=col, mew=0.8, mfc="white", ms=3.2, lw=1.2, zorder=3)
            data.append(g.assign(arm=arm))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("reported confidence")
    axes[0].set_ylabel("empirical accuracy")
    S.save(fig, FIG / "fig4_confidence_reliability.pdf", pd.concat(data))


def fig5_no_signal():
    """Semantics-free options: grey = accuracy, colour = mean confidence."""
    t = C("no_signal_control").set_index("arm").reindex(S.ARMS3).reset_index()
    fig, ax = plt.subplots(figsize=(S.W1, 1.95))
    x = np.arange(len(t))
    wd = 0.34
    bars(ax, x - wd / 2, t.accuracy, S.G3, wd)
    S.eb(ax, x - wd / 2, t, "accuracy", "acc_lo", "acc_hi")
    bars(ax, x + wd / 2, t.mean_conf, [S.ARM_COLOR[a] for a in t.arm], wd)
    S.chance(ax)
    ticks(ax, x, [S.ARM_LABEL[a] for a in t.arm])
    ax.set_ylabel("accuracy / confidence")
    ax.set_ylim(0, 0.92)
    S.save(fig, FIG / "fig5_no_signal_control.pdf", t)


def fig6_order():
    """Left: distinct glosses chosen over 24 orderings. Right: instability
    (colour) beside mean p(gold) range (grey)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.0),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(wspace=0.26)
    x = np.arange(1, 5)
    wd = 0.27
    data = []
    for i, arm in enumerate(S.ARMS3):
        h = T(arm, "e7_instability_hist").set_index("n_distinct_option")
        h = h.reindex(range(1, 5)).fillna(0).reset_index()
        bars(ax, x + (i - 1) * wd, h.share, S.ARM_COLOR[arm], wd)
        data.append(h.assign(arm=arm))
    ax.set_xticks(x)
    ax.set_xlabel("distinct glosses chosen across 24 orderings")
    ax.set_ylabel("share of items")
    ax.set_ylim(0, 1.0)

    o = C("order_three_way").set_index("arm").reindex(S.ARMS3).reset_index()
    xo = np.arange(len(o))
    wd2 = 0.34
    bars(ax2, xo - wd2 / 2, o.answer_unstable_share,
         [S.ARM_COLOR[a] for a in o.arm], wd2)
    bars(ax2, xo + wd2 / 2, o.mean_p_gold_range, S.G3, wd2)
    ticks(ax2, xo, [S.ARM_LABEL[a] for a in o.arm])
    ax2.set_ylabel("share / range")
    ax2.set_ylim(0, 0.92)
    S.save(fig, FIG / "fig6_order_sensitivity.pdf", pd.concat(data))


def fig7_position():
    """Selection share by option slot, one panel per model."""
    fig, axes = plt.subplots(1, 3, figsize=(S.W2, 1.85), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    data = []
    for ax, arm in zip(axes, S.ARMS3):
        p = T(arm, "e1_position_bias").sort_values("position")
        x = np.arange(len(p))
        bars(ax, x, p.selection_share, S.ARM_COLOR[arm], 0.6)
        S.chance(ax)
        ticks(ax, x, list(p.key))
        ax.set_xlabel("option slot")
        data.append(p.assign(arm=arm))
    axes[0].set_ylabel("share of selections")
    axes[0].set_ylim(0, 0.42)
    S.save(fig, FIG / "fig7_position_bias.pdf", pd.concat(data))


def fig8_options():
    """Option-side ablation. Grey bar = options stripped of all semantics.
    Jev panel is hatched because its task language is English."""
    FORM_BN = ["O0_full_bn", "O1_trunc6", "O2_scrambled", "O3_english", "O4_labels"]
    FORM_EN = ["O0_full_en", "O1_trunc6", "O2_scrambled", "O3_bangla", "O4_labels"]
    SHORT = ["full", "truncated", "scrambled", "other lang.", "no semantics"]
    fig, axes = plt.subplots(1, 3, figsize=(S.W2, 2.15), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    data = []
    for ax, arm in zip(axes, S.ARMS3):
        en = arm == "jev-1.13"
        d = X(T(arm, "e6_summary"), "form", FORM_EN if en else FORM_BN)
        x = np.arange(len(d))
        for xi, (v, lastcol) in enumerate(zip(d.accuracy, [False] * 4 + [True])):
            bars(ax, [xi], [v], S.G3 if lastcol else S.ARM_COLOR[arm], 0.62,
                 hatch=HATCH if en else None,
                 hatch_color=S.G1 if lastcol else S.ARM_LINE[arm])
        S.eb(ax, x, d, "accuracy", "acc_lo", "acc_hi")
        S.chance(ax)
        ticks(ax, x, SHORT, rot=35, fs=6.5)
        data.append(d.assign(arm=arm))
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_xlabel("option text")
    S.save(fig, FIG / "fig8_option_ablation.pdf", pd.concat(data))


def fig9_unit_integrity():
    """Perturb the idiom, hold options fixed. Left accuracy, right agreement
    with the intact form."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(S.W2, 2.15))
    fig.subplots_adjust(wspace=0.24)
    x = np.arange(len(VAR))
    wd = 0.38
    data = []
    for i, arm in enumerate(S.ARMS2):
        d = X(T(arm, "e3_summary"), "variant", VAR)
        off = (i - 0.5) * wd
        bars(ax, x + off, d.accuracy, S.ARM_COLOR[arm], wd)
        S.eb(ax, x + off, d, "accuracy", "acc_lo", "acc_hi")
        bars(ax2, x + off, d.answer_agreement_with_intact, S.ARM_COLOR[arm], wd)
        S.eb(ax2, x + off, d, "answer_agreement_with_intact", "agree_lo", "agree_hi")
        data.append(d.assign(arm=arm))
    for a_, yl, lb in [(ax, 0.58, "accuracy"), (ax2, 1.05, "agreement with intact")]:
        ticks(a_, x, [VAR_SHORT[v] for v in VAR], rot=35, fs=6.5)
        a_.set_ylim(0, yl)
        a_.set_ylabel(lb)
    S.chance(ax)
    S.save(fig, FIG / "fig9_unit_integrity.pdf", pd.concat(data))


def fig10_crosslingual():
    """Accuracy by task language within each model. Hatched bars are English
    conditions, solid bars Bangla. Bars are grouped by model, left to right."""
    rows = []
    SHORT = {"bn_ml": "BN idiom", "en_ml": "EN idiom", "en_en": "EN idiom\nEN model",
             "bn_en": "BN idiom\nEN model", "bn": "BN idiom", "en": "EN idiom"}
    for arm in S.ARMS2:
        for r in T(arm, "e4_arms").itertuples():
            rows.append({"arm": arm, "cond": r.condition, "label": r.label,
                         "accuracy": r.accuracy, "acc_lo": r.acc_lo,
                         "acc_hi": r.acc_hi})
    jb = C("three_way_bangla_drand")
    jb = jb[jb.arm == "jev-1.13"].iloc[0]
    rows.append({"arm": "jev-1.13", "cond": "bn", "label": "Bangla idiom",
                 "accuracy": jb.accuracy, "acc_lo": jb.acc_lo, "acc_hi": jb.acc_hi})
    je = T("jev-1.13", "e1_summary")
    je = je[je.family == "D-rand"].iloc[0]
    rows.append({"arm": "jev-1.13", "cond": "en", "label": "English equivalent",
                 "accuracy": je.accuracy, "acc_lo": je.acc_lo, "acc_hi": je.acc_hi})
    t = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(S.W2, 2.05))
    xs, cols, hat, lab = [], [], [], []
    pos = 0.0
    for arm in S.ARMS3:
        for r in t[t.arm == arm].itertuples():
            xs.append(pos)
            lab.append(SHORT.get(str(r.cond), str(r.cond)))
            cols.append(S.ARM_COLOR[arm])
            hat.append(HATCH if not str(r.cond).startswith("bn") else None)
            pos += 1
        pos += 0.75
    for xi, v, c, h, a in zip(xs, t.accuracy, cols, hat, t.arm):
        bars(ax, [xi], [v], c, 0.74, hatch=h, hatch_color=S.ARM_LINE[a])
    ax.errorbar(xs, t.accuracy,
                yerr=np.vstack([t.accuracy - t.acc_lo, t.acc_hi - t.accuracy]),
                fmt="none", ecolor=S.K, elinewidth=0.7, capsize=1.6, zorder=3)
    S.chance(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=6.3)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    S.save(fig, FIG / "fig10_crosslingual.pdf", t)


if __name__ == "__main__":
    for fn in [fig1_literal_capture, fig2_capture_matrix, fig3_three_way_bangla,
               fig4_confidence, fig5_no_signal, fig6_order, fig7_position,
               fig8_options, fig9_unit_integrity, fig10_crosslingual]:
        print(fn.__name__)
        fn()
    print("\nDone ->", FIG)
