#!/usr/bin/env python3
"""LaTeX tables for the paper, both arms. Writes analysis/tables/tables.tex."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TBL = ROOT / "analysis" / "tables"
ARMS = ["laya-ml", "lod-lille-0.6b"]
LBL = {"laya-ml": "Laya-ml", "lod-lille-0.6b": "Lod-lille"}
FAM = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]
FAM_LBL = {"D-rand": "random", "D-surf": "surface-matched", "D-sem": "tag-matched",
           "D-lit": "+ own literal gloss", "D-hard": "literal + surface + tag"}


def T(arm, n):
    return pd.read_csv(TBL / arm / f"{n}.csv")


def stars(p):
    return "$^{***}$" if p < 1e-3 else "$^{**}$" if p < 1e-2 \
        else "$^{*}$" if p < .05 else ""


def tab_ladder():
    L = T("laya-ml", "e1_summary").set_index("family")
    D = T("lod-lille-0.6b", "e1_summary").set_index("family")
    mc = pd.read_csv(TBL / "_cross" / "paired_mcnemar.csv")
    mc = mc[mc.experiment == "e1_ladder"].set_index("group")
    rows = []
    for f in FAM:
        lar_l = f"{L.loc[f,'LAR']:.3f}" if not np.isnan(L.loc[f, "LAR"]) else "--"
        lar_d = f"{D.loc[f,'LAR']:.3f}" if not np.isnan(D.loc[f, "LAR"]) else "--"
        d = mc.loc[f, "delta_lod_minus_laya"]
        rows.append(f"{FAM_LBL[f]} & {L.loc[f,'accuracy']:.3f} & {lar_l} & "
                    f"{D.loc[f,'accuracy']:.3f} & {lar_d} & "
                    f"{d:+.3f}{stars(mc.loc[f,'mcnemar_p'])} \\\\")
    return (r"""% Distractor ladder, both arms. Delta is paired on the 888 shared items.
\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{2}{c}{Laya-ml (8{,}876)} & \multicolumn{2}{c}{Lod-lille (888)} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Distractor set & Acc. & LAR & Acc. & LAR & $\Delta$ \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
chance & 0.250 & -- & 0.250 & -- & \\
\bottomrule
\end{tabular}""")


def tab_confidence():
    rows = []
    lab = {"correct": "correct", "chose_literal": "captured by literal",
           "other_wrong": "other error"}
    L = T("laya-ml", "e5_confidence_by_outcome").set_index("outcome")
    D = T("lod-lille-0.6b", "e5_confidence_by_outcome").set_index("outcome")
    for o in ["correct", "chose_literal", "other_wrong"]:
        rows.append(f"{lab[o]} & {L.loc[o,'share']:.3f} & {L.loc[o,'mean_conf']:.3f} & "
                    f"{D.loc[o,'share']:.3f} & {D.loc[o,'mean_conf']:.3f} & "
                    f"{D.loc[o,'mean_conf_head']:.3f} \\\\")
    return (r"""% Confidence is anti-diagnostic in both models
\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{2}{c}{Laya-ml} & \multicolumn{3}{c}{Lod-lille} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
Outcome & Share & Conf. & Share & Conf. & Head \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def tab_order():
    L = T("laya-ml", "e7_headline").iloc[0]
    D = T("lod-lille-0.6b", "e7_headline").iloc[0]
    rows = [
        ("items whose chosen gloss changes", L.answer_unstable_share, D.answer_unstable_share),
        ("items whose correctness varies", L.correctness_varies_share, D.correctness_varies_share),
        ("mean distinct glosses (of 4)", L.mean_distinct_options, D.mean_distinct_options),
        ("mean range of $p$(gold)", L.mean_p_gold_range, D.mean_p_gold_range),
        ("mean range of confidence", L.mean_conf_range, D.mean_conf_range),
    ]
    return (r"""% Option-order sensitivity, all 24 orderings of a fixed option set
\begin{tabular}{lrr}
\toprule
& Laya-ml & Lod-lille \\
& ($n{=}992$) & ($n{=}299$) \\
\midrule
""" + "\n".join(f"{n} & {a:.3f} & {b:.3f} \\\\" for n, a, b in rows) + r"""
\bottomrule
\end{tabular}""")


def tab_crosslingual():
    rows = []
    for arm in ARMS:
        d = T(arm, "e4_arms")
        for r in d.itertuples():
            rows.append(f"{LBL[arm]} & {r.label} & {r.accuracy:.3f} "
                        f"\\tiny[{r.acc_lo:.3f},{r.acc_hi:.3f}] & {r.mean_conf:.3f} \\\\")
    return (r"""% Cross-lingual localisation
\begin{tabular}{llrr}
\toprule
Model & Input / checkpoint & Accuracy & Conf. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def tab_options():
    L = T("laya-ml", "e6_summary").set_index("form")
    D = T("lod-lille-0.6b", "e6_summary").set_index("form")
    order = ["O0_full_bn", "O1_trunc6", "O2_scrambled", "O3_english", "O4_labels"]
    rows = [f"{L.loc[f,'label']} & {L.loc[f,'accuracy']:.3f} & "
            f"{L.loc[f,'semantic_signal_retained']:.3f} & "
            f"{D.loc[f,'accuracy']:.3f} & "
            f"{D.loc[f,'semantic_signal_retained']:.3f} \\\\" for f in order]
    return (r"""% Option-side ablation
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{Laya-ml} & \multicolumn{2}{c}{Lod-lille} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Option text & Acc. & Retained & Acc. & Retained \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def tab_unit():
    L = T("laya-ml", "e3_summary").set_index("variant")
    D = T("lod-lille-0.6b", "e3_summary").set_index("variant")
    order = ["V0_intact", "V1_shuffle", "V2_reverse", "V3_delete1",
             "V4_substitute1", "V5_in_sentence"]
    rows = [f"{L.loc[v,'label']} & {L.loc[v,'accuracy']:.3f} & "
            f"{L.loc[v,'delta_acc_vs_intact']:+.3f} & {D.loc[v,'accuracy']:.3f} & "
            f"{D.loc[v,'delta_acc_vs_intact']:+.3f} \\\\" for v in order]
    return (r"""% Unit integrity
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{Laya-ml ($n{=}1{,}778$)} & \multicolumn{2}{c}{Lod-lille ($n{=}180$)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Perturbation & Acc. & $\Delta$ & Acc. & $\Delta$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def main():
    tex = [tab_ladder(), tab_confidence(), tab_order(), tab_crosslingual(),
           tab_options(), tab_unit()]
    (TBL / "tables.tex").write_text("\n\n".join(tex), encoding="utf-8")
    print("[tex] tables.tex")


if __name__ == "__main__":
    main()
