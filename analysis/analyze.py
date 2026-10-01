#!/usr/bin/env python3
"""BNFDM analysis: derived statistics for the Laya arm.

Reads LAYA_RUN/results/laya-ml/**, writes analysis/tables/*.csv and *.tex.
Pure aggregation + inference. Figures are in figures.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from bnfdm_style import boot_ci, ece, aurc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "LAYA_RUN" / "results" / "laya-ml"
OUT = ROOT / "analysis" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

K = 4
CHANCE = 1.0 / K
OPT_KEYS = ["A", "B", "C", "D"]
P = [f"p_{k}" for k in OPT_KEYS]
FAM_ORDER = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]
FAM_LABEL = {
    "D-rand": "random",
    "D-surf": "surface-matched",
    "D-sem": "tag-matched",
    "D-lit": "+ own literal gloss",
    "D-hard": "literal + surface + tag",
}


def w(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  [tbl] {name}.csv  ({len(df)} rows)")
    return df


# ════════════════════════════════════════════════════════════════════════
# E1 — distractor ladder, literal attraction
# ════════════════════════════════════════════════════════════════════════
def e1():
    print("E1 distractor ladder")
    d = pd.read_csv(RES / "e1_ladder" / "raw.csv")
    rows = []
    for fam in FAM_ORDER:
        g = d[d.family == fam]
        acc, lo, hi = boot_ci(g.correct.values, seed=1)
        # exact binomial test against chance
        p_chance = binomtest(int(g.correct.sum()), len(g), CHANCE).pvalue
        lar = lar_lo = lar_hi = np.nan
        if g.chose_literal.notna().any() and g.lit_key.notna().any():
            gl = g[g.lit_key.notna()]
            lar, lar_lo, lar_hi = boot_ci(gl.chose_literal.values, seed=2)
        rows.append({
            "family": fam, "label": FAM_LABEL[fam], "n": len(g),
            "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
            "chance": CHANCE, "vs_chance_p": p_chance,
            "below_chance": acc < CHANCE,
            "LAR": lar, "LAR_lo": lar_lo, "LAR_hi": lar_hi,
            "mean_p_gold": g.p_gold.mean(),
            "mean_p_literal": g.p_literal.mean(),
            "mean_conf": g.conf.mean(),
            "ece": ece(g.conf, g.correct),
            "aurc": aurc(g.conf.values, g.correct.values),
        })
    summ = w(pd.DataFrame(rows), "e1_summary")

    # outcome decomposition for the two literal-bearing families
    dec = []
    for fam in ["D-lit", "D-hard"]:
        g = d[(d.family == fam) & d.lit_key.notna()]
        n = len(g)
        dec.append({
            "family": fam, "n": n,
            "chose_gold": (g.correct == 1).mean(),
            "chose_literal": (g.chose_literal == 1).mean(),
            "chose_other": ((g.correct == 0) & (g.chose_literal == 0)).mean(),
        })
    w(pd.DataFrame(dec), "e1_outcome_decomposition")

    # paired within-item: does adding the literal gloss flip a correct answer?
    piv = d.pivot_table(index="id", columns="family", values="correct")
    flips = []
    for fam in ["D-lit", "D-hard"]:
        m = piv[["D-rand", fam]].dropna()
        n01 = int(((m["D-rand"] == 0) & (m[fam] == 1)).sum())
        n10 = int(((m["D-rand"] == 1) & (m[fam] == 0)).sum())
        flips.append({
            "family": fam, "n_paired": len(m),
            "acc_Drand": m["D-rand"].mean(), "acc_family": m[fam].mean(),
            "delta": m[fam].mean() - m["D-rand"].mean(),
            "rand_only_right": n10, "family_only_right": n01,
            "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else np.nan,
        })
    w(pd.DataFrame(flips), "e1_paired_vs_random")

    # stratified views, with the literal-bearing families kept separate
    for col in ["frequency", "scape", "hist_sig", "relig_sig", "n_words"]:
        rows = []
        for (fam, lev), g in d.groupby(["family", col]):
            if len(g) < 20:
                continue
            acc, lo, hi = boot_ci(g.correct.values, seed=3)
            rows.append({"family": fam, col: lev, "n": len(g),
                         "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                         "LAR": g.chose_literal.mean() if g.lit_key.notna().any() else np.nan,
                         "mean_conf": g.conf.mean()})
        w(pd.DataFrame(rows), f"e1_by_{col}")
    return d, summ


# ════════════════════════════════════════════════════════════════════════
# E7 + E1 — positional bias and option-order instability
# ════════════════════════════════════════════════════════════════════════
def e7(d1):
    print("E7 option-order sensitivity")
    d = pd.read_csv(RES / "e7_order" / "raw.csv")
    pit = pd.read_csv(RES / "e7_order" / "per_item.csv")
    # the shipped per_item.csv predates the p_gold_range column; derive it
    if "p_gold_range" not in pit.columns:
        rng = (d.groupby("id").p_gold.agg(lambda s: s.max() - s.min())
                 .rename("p_gold_range").reset_index())
        pit = pit.merge(rng, on="id", how="left")

    # instability: how many distinct answers does permuting the order produce?
    inst = pit.n_distinct_pred.value_counts().sort_index().rename("items").reset_index()
    inst.columns = ["n_distinct_pred", "items"]
    inst["share"] = inst["items"] / inst["items"].sum()
    w(inst, "e7_instability_hist")

    hdr = pd.DataFrame([{
        "n_items": int(pit.shape[0]), "n_permutations": int(d.perm_id.nunique()),
        "unstable_share": float((pit.n_distinct_pred > 1).mean()),
        "mean_distinct_answers": float(pit.n_distinct_pred.mean()),
        "mean_p_gold_std": float(pit.p_gold_std.mean()),
        "mean_p_gold_range": float(pit.p_gold_range.mean()),
        "max_p_gold_range": float(pit.p_gold_range.max()),
        "mean_conf_range": float(pit.conf_range.mean()),
        "overall_accuracy": float(d.correct.mean()),
    }]).round(6)
    w(hdr, "e7_headline")

    # accuracy and selection share by position, with CIs
    rows = []
    for pos in range(K):
        g = d[d.gold_pos == pos]
        acc, lo, hi = boot_ci(g.correct.values, seed=4)
        rows.append({"position": pos, "key": OPT_KEYS[pos], "n": len(g),
                     "accuracy_when_gold_here": acc, "acc_lo": lo, "acc_hi": hi,
                     "mean_p_gold": g.p_gold.mean(),
                     "selection_share": float((d.pred_pos == pos).mean()),
                     "mean_p_at_position": float(d[f"p_{OPT_KEYS[pos]}"].mean())})
    w(pd.DataFrame(rows), "e7_position_bias")

    # the same bias measured independently in E1 (order randomised per item)
    rows = []
    for pos in range(K):
        sel = float((d1.pred == OPT_KEYS[pos]).mean())
        g = d1[d1.gold_pos == pos]
        acc, lo, hi = boot_ci(g.correct.values, seed=5)
        rows.append({"position": pos, "key": OPT_KEYS[pos],
                     "gold_share": float((d1.gold_pos == pos).mean()),
                     "selection_share": sel,
                     "accuracy_when_gold_here": acc, "acc_lo": lo, "acc_hi": hi,
                     "mean_p_at_position": float(d1[f"p_{OPT_KEYS[pos]}"].mean())})
    w(pd.DataFrame(rows), "e1_position_bias")
    return d, pit


# ════════════════════════════════════════════════════════════════════════
# E2 — surface invariance
# ════════════════════════════════════════════════════════════════════════
def e2():
    print("E2 surface invariance")
    d = pd.read_csv(RES / "e2_invariance" / "raw.csv")
    ag, lo, hi = boot_ci(d.agree.values, seed=6)
    rows = [{
        "n_pairs": len(d), "n_items": int(d.id.nunique()),
        "agreement": ag, "agree_lo": lo, "agree_hi": hi,
        "acc_canonical": d.correct_canon.mean(), "acc_variant": d.correct_alt.mean(),
        "both_correct": ((d.correct_canon == 1) & (d.correct_alt == 1)).mean(),
        "neither_correct": ((d.correct_canon == 0) & (d.correct_alt == 0)).mean(),
        "mean_jsd": d.jsd.mean(), "median_jsd": d.jsd.median(),
        "mean_abs_dP_gold": (d.p_gold_canon - d.p_gold_alt).abs().mean(),
    }]
    w(pd.DataFrame(rows).round(6), "e2_summary")
    q = d.jsd.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).rename("jsd").reset_index()
    q.columns = ["quantile", "jsd"]
    w(q, "e2_jsd_quantiles")
    return d


# ════════════════════════════════════════════════════════════════════════
# E3 — unit integrity
# ════════════════════════════════════════════════════════════════════════
VAR_LABEL = {
    "V0_intact": "intact", "V1_shuffle": "word shuffle", "V2_reverse": "reversed",
    "V3_delete1": "delete 1 word", "V4_substitute1": "substitute 1 word",
    "V5_in_sentence": "in carrier sentence",
}


def e3():
    print("E3 unit integrity")
    d = pd.read_csv(RES / "e3_unit" / "raw.csv")
    base = d[d.variant == "V0_intact"].set_index("id")
    rows = []
    for v in VAR_LABEL:
        g = d[d.variant == v].set_index("id")
        acc, lo, hi = boot_ci(g.correct.values, seed=7)
        same = (g.pred == base.pred.reindex(g.index)).astype(float)
        agree, alo, ahi = boot_ci(same.values, seed=8)
        rows.append({"variant": v, "label": VAR_LABEL[v], "n": len(g),
                     "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "delta_acc_vs_intact": acc - base.correct.mean(),
                     "answer_agreement_with_intact": agree,
                     "agree_lo": alo, "agree_hi": ahi,
                     "mean_jsd_vs_intact": g.jsd_vs_intact.mean(),
                     "mean_p_gold": g.p_gold.mean(), "mean_conf": g.conf.mean()})
    w(pd.DataFrame(rows), "e3_summary")

    rows = []
    for (v, nw), g in d.groupby(["variant", "n_words"]):
        if len(g) < 20:
            continue
        rows.append({"variant": v, "n_words": nw, "n": len(g),
                     "accuracy": g.correct.mean(),
                     "mean_jsd": g.jsd_vs_intact.mean()})
    w(pd.DataFrame(rows), "e3_by_length")
    return d


# ════════════════════════════════════════════════════════════════════════
# E4 — cross-lingual failure localisation
# ════════════════════════════════════════════════════════════════════════
ARM_LABEL = {
    "bn_ml": "Bangla idiom / multilingual",
    "en_ml": "English equivalent / multilingual",
    "en_en": "English equivalent / English",
    "bn_en": "Bangla idiom / English (control)",
}


def e4():
    print("E4 cross-lingual")
    d = pd.read_csv(RES / "e4_crosslingual" / "raw.csv")
    rows = []
    for a in ["bn_ml", "en_ml", "en_en", "bn_en"]:
        acc, lo, hi = boot_ci(d[f"{a}_correct"].values, seed=9)
        rows.append({"arm": a, "label": ARM_LABEL[a], "n": len(d),
                     "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "mean_p_gold": d[f"{a}_p_gold"].mean(),
                     "mean_conf": d[f"{a}_conf"].mean(),
                     "ece": ece(d[f"{a}_conf"], d[f"{a}_correct"])})
    w(pd.DataFrame(rows), "e4_arms")

    # localisation: same checkpoint, two languages
    cell = np.select(
        [(d.bn_ml_correct == 1) & (d.en_ml_correct == 1),
         (d.bn_ml_correct == 0) & (d.en_ml_correct == 1),
         (d.bn_ml_correct == 1) & (d.en_ml_correct == 0)],
        ["both_right", "bangla_gap", "english_gap"], default="both_wrong")
    loc = pd.Series(cell).value_counts(normalize=True).rename("share").reset_index()
    loc.columns = ["cell", "share"]
    loc["n"] = pd.Series(cell).value_counts().values
    w(loc, "e4_localisation")

    n01 = int(((d.bn_ml_correct == 0) & (d.en_ml_correct == 1)).sum())
    n10 = int(((d.bn_ml_correct == 1) & (d.en_ml_correct == 0)).sum())
    w(pd.DataFrame([{
        "comparison": "en_ml vs bn_ml (same checkpoint, two languages)",
        "n_paired": len(d), "bn_only_right": n10, "en_only_right": n01,
        "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue,
        "delta_accuracy": d.en_ml_correct.mean() - d.bn_ml_correct.mean(),
    }]), "e4_mcnemar")
    return d


# ════════════════════════════════════════════════════════════════════════
# E5 — calibration and selective prediction
# ════════════════════════════════════════════════════════════════════════
def e5(d1):
    print("E5 calibration")
    rows = []
    for fam in FAM_ORDER:
        g = d1[d1.family == fam]
        rows.append({
            "family": fam, "label": FAM_LABEL[fam], "n": len(g),
            "accuracy": g.correct.mean(), "mean_conf": g.conf.mean(),
            "overconfidence": g.conf.mean() - g.correct.mean(),
            "ece": ece(g.conf, g.correct),
            "aurc_conf": aurc(g.conf.values, g.correct.values),
            "aurc_random": 1 - g.correct.mean(),
        })
    w(pd.DataFrame(rows), "e5_calibration")

    # the key safety result: is confidence higher when the model is captured?
    lit = d1[d1.family.isin(["D-lit", "D-hard"]) & d1.lit_key.notna()]
    rows = []
    for name, sub in [("correct", lit[lit.correct == 1]),
                      ("chose_literal", lit[lit.chose_literal == 1]),
                      ("other_wrong", lit[(lit.correct == 0) & (lit.chose_literal == 0)])]:
        m, lo, hi = boot_ci(sub.conf.values, seed=10)
        rows.append({"outcome": name, "n": len(sub), "share": len(sub) / len(lit),
                     "mean_conf": m, "conf_lo": lo, "conf_hi": hi,
                     "median_conf": sub.conf.median(),
                     "mean_p_gold": sub.p_gold.mean(),
                     "mean_p_literal": sub.p_literal.mean()})
    w(pd.DataFrame(rows), "e5_confidence_by_outcome")

    # risk-coverage: accuracy at each coverage level, per family
    rows = []
    for fam in FAM_ORDER:
        g = d1[d1.family == fam]
        o = np.argsort(-g.conf.values)
        corr = g.correct.values[o]
        cum = np.cumsum(corr) / np.arange(1, len(corr) + 1)
        for cov in np.arange(0.05, 1.001, 0.05):
            i = max(0, int(round(cov * len(corr))) - 1)
            rows.append({"family": fam, "coverage": round(cov, 2),
                         "accuracy": cum[i], "threshold": g.conf.values[o][i]})
    w(pd.DataFrame(rows), "e5_risk_coverage")

    # reliability bins
    rows = []
    edges = np.linspace(0, 1, 11)
    for fam in FAM_ORDER:
        g = d1[d1.family == fam]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (g.conf > lo) & (g.conf <= hi)
            if m.sum() < 10:
                continue
            rows.append({"family": fam, "bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                         "mean_conf": g.conf[m].mean(), "accuracy": g.correct[m].mean()})
    w(pd.DataFrame(rows), "e5_reliability")


# ════════════════════════════════════════════════════════════════════════
# E6 — option-side attribution
# ════════════════════════════════════════════════════════════════════════
FORM_LABEL = {
    "O0_full_bn": "full Bangla gloss", "O1_trunc6": "truncated to 6 words",
    "O2_scrambled": "word-scrambled", "O3_english": "English gloss",
    "O4_labels": "semantics removed",
}


def e6():
    print("E6 option-side attribution")
    d = pd.read_csv(RES / "e6_options" / "raw.csv")
    base = d[d.form == "O0_full_bn"].set_index("id")
    rows = []
    for f in FORM_LABEL:
        g = d[d.form == f].set_index("id")
        acc, lo, hi = boot_ci(g.correct.values, seed=11)
        same = (g.pred == base.pred.reindex(g.index)).astype(float)
        rows.append({"form": f, "label": FORM_LABEL[f], "n": len(g),
                     "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "delta_vs_full": acc - base.correct.mean(),
                     "answer_agreement_with_full": same.mean(),
                     "mean_jsd_vs_full": g.jsd_vs_full.mean(),
                     "mean_p_gold": g.p_gold.mean(), "mean_conf": g.conf.mean(),
                     "mean_opt_tokens": g.opt_tokens.mean()})
    w(pd.DataFrame(rows), "e6_summary")

    # how much of the gap between full options and no options does each form keep?
    full_acc = base.correct.mean()
    null_acc = d[d.form == "O4_labels"].correct.mean()
    rows = []
    for f in FORM_LABEL:
        a = d[d.form == f].correct.mean()
        rows.append({"form": f, "label": FORM_LABEL[f], "accuracy": a,
                     "semantic_signal_retained": (a - null_acc) / (full_acc - null_acc)})
    w(pd.DataFrame(rows), "e6_signal_retained")
    return d


# ════════════════════════════════════════════════════════════════════════
def latex_tables():
    print("LaTeX tables")
    tex = []

    s = pd.read_csv(OUT / "e1_summary.csv")
    tex.append(r"""% Main result: distractor ladder
\begin{tabular}{llrrr}
\toprule
Distractor set & & Accuracy & LAR & ECE \\
\midrule
""" + "\n".join(
        f"{r.label} & & {r.accuracy:.3f} \\tiny[{r.acc_lo:.3f},{r.acc_hi:.3f}] & "
        + (f"{r.LAR:.3f}" if not np.isnan(r.LAR) else "--")
        + f" & {r.ece:.3f} \\\\"
        for r in s.itertuples()
    ) + r"""
\midrule
chance & & 0.250 & -- & -- \\
\bottomrule
\end{tabular}""")

    a = pd.read_csv(OUT / "e4_arms.csv")
    tex.append(r"""% Cross-lingual localisation
\begin{tabular}{lrr}
\toprule
Input / checkpoint & Accuracy & Conf. \\
\midrule
""" + "\n".join(
        f"{r.label} & {r.accuracy:.3f} \\tiny[{r.acc_lo:.3f},{r.acc_hi:.3f}] & {r.mean_conf:.3f} \\\\"
        for r in a.itertuples()
    ) + r"""
\bottomrule
\end{tabular}""")

    c = pd.read_csv(OUT / "e5_confidence_by_outcome.csv")
    tex.append(r"""% Confidence is anti-diagnostic
\begin{tabular}{lrrr}
\toprule
Outcome & Share & Mean conf. & $p$(gold) \\
\midrule
""" + "\n".join(
        f"{r.outcome.replace('_',' ')} & {r.share:.3f} & {r.mean_conf:.3f} & {r.mean_p_gold:.3f} \\\\"
        for r in c.itertuples()
    ) + r"""
\bottomrule
\end{tabular}""")

    o = pd.read_csv(OUT / "e6_summary.csv")
    tex.append(r"""% Option-side ablation
\begin{tabular}{lrrr}
\toprule
Option text & Accuracy & $\Delta$ & JSD \\
\midrule
""" + "\n".join(
        f"{r.label} & {r.accuracy:.3f} & {r.delta_vs_full:+.3f} & {r.mean_jsd_vs_full:.3f} \\\\"
        for r in o.itertuples()
    ) + r"""
\bottomrule
\end{tabular}""")

    v = pd.read_csv(OUT / "e3_summary.csv")
    tex.append(r"""% Unit integrity
\begin{tabular}{lrrr}
\toprule
Perturbation & Accuracy & Agreement & JSD \\
\midrule
""" + "\n".join(
        f"{r.label} & {r.accuracy:.3f} & {r.answer_agreement_with_intact:.3f} & {r.mean_jsd_vs_intact:.3f} \\\\"
        for r in v.itertuples()
    ) + r"""
\bottomrule
\end{tabular}""")

    (OUT / "tables.tex").write_text("\n\n".join(tex), encoding="utf-8")
    print("  [tex] tables.tex")


def main():
    d1, _ = e1()
    e7(d1)
    e2()
    e3()
    e4()
    e5(d1)
    e6()
    latex_tables()
    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()
