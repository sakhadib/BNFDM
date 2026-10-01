#!/usr/bin/env python3
"""BNFDM analysis, both arms.

Arms:
  laya-ml        Laya multilingual (convaiinnovations/laya, subfolder multilingual)
                 full set, n = 8,876
  lod-lille-0.6b mrn-dk/lod-lille-0.6B, fp32
                 fixed 10% subset, n = 888

The two arms were run from identical per-item seeds over an identically built
`full` table, so option sets are byte-identical on the overlap — verified in
crossmodel.py before any paired test is reported.

Confidence is normalised across arms:
  confidence  = probability of the chosen option (comparable across models)
  conf_head   = Lod's separate "is the answer even here" head (Lod only)
Laya's `answer_confidence` is the top-class probability, so it maps to
`confidence`; its entropy-based `confidence` column is NOT comparable and is
carried through as `entropy_conf`.

Writes analysis/tables/<arm>/*.csv plus LaTeX in analysis/tables/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from bnfdm_style import boot_ci, ece, aurc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "tables"

K = 4
CHANCE = 1.0 / K
OPT_KEYS = ["A", "B", "C", "D"]
P = [f"p_{k}" for k in OPT_KEYS]
FAM_ORDER = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]
FAM_LABEL = {
    "D-rand": "random", "D-surf": "surface-matched", "D-sem": "tag-matched",
    "D-lit": "+ own literal gloss", "D-hard": "literal + surface + tag",
}

ARMS = {
    "laya-ml": {
        "dir": ROOT / "LAYA_RUN" / "results" / "laya-ml",
        "label": "Laya-multilingual (322M)",
        "conf": "conf", "e4": "e4_crosslingual",
        "e4_arms": ["bn_ml", "en_ml", "en_en", "bn_en"],
        "e3_conf": "conf",
    },
    "lod-lille-0.6b": {
        "dir": ROOT / "LOD_RUN" / "results" / "lod-lille-0.6b",
        "label": "Lod-lille (0.6B)",
        "conf": "max_prob", "e4": "e4_language",
        "e4_arms": ["bn", "en"],
        "e3_conf": "max_prob",
    },
}
E4_LABEL = {
    "bn_ml": "Bangla idiom / multilingual", "en_ml": "English equiv. / multilingual",
    "en_en": "English equiv. / English", "bn_en": "Bangla idiom / English (control)",
    "bn": "Bangla idiom", "en": "English equivalent",
}
VAR_LABEL = {
    "V0_intact": "intact", "V1_shuffle": "word shuffle", "V2_reverse": "reversed",
    "V3_delete1": "delete 1 word", "V4_substitute1": "substitute 1 word",
    "V5_in_sentence": "in carrier sentence",
}
FORM_LABEL = {
    "O0_full_bn": "full Bangla gloss", "O1_trunc6": "truncated to 6 words",
    "O2_scrambled": "word-scrambled", "O3_english": "English gloss",
    "O4_labels": "semantics removed",
}


def w(df, arm, name):
    d = OUT / arm
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  [tbl] {arm}/{name}.csv ({len(df)})")
    return df


def load_e1(arm):
    """E1 raw with a normalised `confidence` column."""
    c = ARMS[arm]
    d = pd.read_csv(c["dir"] / "e1_ladder" / "raw.csv")
    d["confidence"] = d[c["conf"]]
    if "conf_head" not in d.columns:
        d["conf_head"] = np.nan
    return d


# ════════════════════════════════════════════════════════════════════
def e1(arm):
    d = load_e1(arm)
    rows = []
    for fam in FAM_ORDER:
        g = d[d.family == fam]
        acc, lo, hi = boot_ci(g.correct.values, seed=1)
        lar = lar_lo = lar_hi = np.nan
        if g.lit_key.notna().any():
            gl = g[g.lit_key.notna()]
            lar, lar_lo, lar_hi = boot_ci(gl.chose_literal.values, seed=2)
        rows.append({
            "arm": arm, "family": fam, "label": FAM_LABEL[fam], "n": len(g),
            "accuracy": acc, "acc_lo": lo, "acc_hi": hi, "chance": CHANCE,
            "vs_chance_p": binomtest(int(g.correct.sum()), len(g), CHANCE).pvalue,
            "below_chance": acc < CHANCE,
            "LAR": lar, "LAR_lo": lar_lo, "LAR_hi": lar_hi,
            "mean_p_gold": g.p_gold.mean(), "mean_p_literal": g.p_literal.mean(),
            "mean_conf": g.confidence.mean(), "ece": ece(g.confidence, g.correct),
            "aurc": aurc(g.confidence.values, g.correct.values),
            "aurc_random": 1 - g.correct.mean(),
        })
    w(pd.DataFrame(rows), arm, "e1_summary")

    dec = []
    for fam in ["D-lit", "D-hard"]:
        g = d[(d.family == fam) & d.lit_key.notna()]
        dec.append({"arm": arm, "family": fam, "n": len(g),
                    "chose_gold": (g.correct == 1).mean(),
                    "chose_literal": (g.chose_literal == 1).mean(),
                    "chose_other": ((g.correct == 0) & (g.chose_literal == 0)).mean()})
    w(pd.DataFrame(dec), arm, "e1_outcome_decomposition")

    piv = d.pivot_table(index="id", columns="family", values="correct")
    flips = []
    for fam in ["D-lit", "D-hard"]:
        m = piv[["D-rand", fam]].dropna()
        n01 = int(((m["D-rand"] == 0) & (m[fam] == 1)).sum())
        n10 = int(((m["D-rand"] == 1) & (m[fam] == 0)).sum())
        flips.append({"arm": arm, "family": fam, "n_paired": len(m),
                      "acc_Drand": m["D-rand"].mean(), "acc_family": m[fam].mean(),
                      "delta": m[fam].mean() - m["D-rand"].mean(),
                      "rand_only_right": n10, "family_only_right": n01,
                      "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue
                      if (n01 + n10) else np.nan})
    w(pd.DataFrame(flips), arm, "e1_paired_vs_random")

    for col in ["frequency", "scape", "hist_sig", "relig_sig", "n_words"]:
        rows = []
        for (fam, lev), g in d.groupby(["family", col]):
            if len(g) < 20:
                continue
            acc, lo, hi = boot_ci(g.correct.values, seed=3)
            rows.append({"arm": arm, "family": fam, col: lev, "n": len(g),
                         "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                         "LAR": g.chose_literal.mean() if g.lit_key.notna().any() else np.nan,
                         "mean_conf": g.confidence.mean()})
        if rows:
            w(pd.DataFrame(rows), arm, f"e1_by_{col}")

    # positional bias measured in E1 (option order randomised per item)
    rows = []
    for pos in range(K):
        g = d[d.gold_pos == pos]
        acc, lo, hi = boot_ci(g.correct.values, seed=5)
        rows.append({"arm": arm, "position": pos, "key": OPT_KEYS[pos],
                     "gold_share": float((d.gold_pos == pos).mean()),
                     "selection_share": float((d.pred == OPT_KEYS[pos]).mean()),
                     "accuracy_when_gold_here": acc, "acc_lo": lo, "acc_hi": hi,
                     "mean_p_at_position": float(d[f"p_{OPT_KEYS[pos]}"].mean())})
    w(pd.DataFrame(rows), arm, "e1_position_bias")
    return d


# ════════════════════════════════════════════════════════════════════
def e7(arm):
    """Option-order sensitivity.

    IMPORTANT: `pred` is an option KEY (slot letter). A key necessarily changes
    when the same gloss moves slots, so counting distinct keys measures
    relabelling, not instability. The correct unit is the identity of the chosen
    OPTION TEXT, recovered as perm[slot(pred)].
    """
    c = ARMS[arm]
    d = pd.read_csv(c["dir"] / "e7_order" / "raw.csv")
    d["confidence"] = d[c["conf"]] if c["conf"] in d.columns else d["conf"]
    d["perm"] = d.perm.astype(str).str.zfill(K)
    d["chosen_opt"] = [int(p[OPT_KEYS.index(pr)]) for p, pr in zip(d.perm, d.pred)]
    d["gold_opt"] = [int(p[OPT_KEYS.index(g)]) for p, g in zip(d.perm, d.gold_key)]

    g = d.groupby("id")
    per = pd.DataFrame({
        "n_distinct_key": g.pred.nunique(),
        "n_distinct_option": g.chosen_opt.nunique(),
        "accuracy": g.correct.mean(),
        "p_gold_mean": g.p_gold.mean(),
        "p_gold_std": g.p_gold.std(),
        "p_gold_range": g.p_gold.agg(lambda s: s.max() - s.min()),
        "conf_range": g.confidence.agg(lambda s: s.max() - s.min()),
    }).reset_index()
    per["answer_unstable"] = per.n_distinct_option > 1
    per["correctness_varies"] = (per.accuracy > 0) & (per.accuracy < 1)
    per["arm"] = arm
    w(per, arm, "e7_per_item")

    hist = (per.n_distinct_option.value_counts().sort_index()
            .rename("items").reset_index())
    hist.columns = ["n_distinct_option", "items"]
    hist["share"] = hist["items"] / hist["items"].sum()
    hist["arm"] = arm
    w(hist, arm, "e7_instability_hist")

    w(pd.DataFrame([{
        "arm": arm, "n_items": len(per), "n_permutations": int(d.perm_id.nunique()),
        "answer_unstable_share": float(per.answer_unstable.mean()),
        "correctness_varies_share": float(per.correctness_varies.mean()),
        "mean_distinct_options": float(per.n_distinct_option.mean()),
        "mean_distinct_keys": float(per.n_distinct_key.mean()),
        "mean_p_gold_std": float(per.p_gold_std.mean()),
        "mean_p_gold_range": float(per.p_gold_range.mean()),
        "max_p_gold_range": float(per.p_gold_range.max()),
        "mean_conf_range": float(per.conf_range.mean()),
        "overall_accuracy": float(d.correct.mean()),
    }]).round(6), arm, "e7_headline")

    rows = []
    for pos in range(K):
        gg = d[d.gold_pos == pos]
        acc, lo, hi = boot_ci(gg.correct.values, seed=4)
        rows.append({"arm": arm, "position": pos, "key": OPT_KEYS[pos], "n": len(gg),
                     "accuracy_when_gold_here": acc, "acc_lo": lo, "acc_hi": hi,
                     "mean_p_gold": gg.p_gold.mean(),
                     "selection_share": float((d.pred_pos == pos).mean()),
                     "mean_p_at_position": float(d[f"p_{OPT_KEYS[pos]}"].mean())})
    w(pd.DataFrame(rows), arm, "e7_position_bias")
    return d


# ════════════════════════════════════════════════════════════════════
def e2(arm):
    d = pd.read_csv(ARMS[arm]["dir"] / "e2_invariance" / "raw.csv")
    ag, lo, hi = boot_ci(d.agree.values, seed=6)
    w(pd.DataFrame([{
        "arm": arm, "n_pairs": len(d), "n_items": int(d.id.nunique()),
        "agreement": ag, "agree_lo": lo, "agree_hi": hi,
        "acc_canonical": d.correct_canon.mean(), "acc_variant": d.correct_alt.mean(),
        "both_correct": ((d.correct_canon == 1) & (d.correct_alt == 1)).mean(),
        "neither_correct": ((d.correct_canon == 0) & (d.correct_alt == 0)).mean(),
        "mean_jsd": d.jsd.mean(), "median_jsd": d.jsd.median(),
        "mean_abs_dP_gold": (d.p_gold_canon - d.p_gold_alt).abs().mean(),
    }]).round(6), arm, "e2_summary")
    return d


def e3(arm):
    c = ARMS[arm]
    d = pd.read_csv(c["dir"] / "e3_unit" / "raw.csv")
    d["confidence"] = d[c["e3_conf"]]
    base = d[d.variant == "V0_intact"].set_index("id")
    rows = []
    for v in VAR_LABEL:
        g = d[d.variant == v].set_index("id")
        if not len(g):
            continue
        acc, lo, hi = boot_ci(g.correct.values, seed=7)
        same = (g.pred == base.pred.reindex(g.index)).astype(float)
        agree, alo, ahi = boot_ci(same.values, seed=8)
        rows.append({"arm": arm, "variant": v, "label": VAR_LABEL[v], "n": len(g),
                     "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "delta_acc_vs_intact": acc - base.correct.mean(),
                     "answer_agreement_with_intact": agree,
                     "agree_lo": alo, "agree_hi": ahi,
                     "mean_jsd_vs_intact": g.jsd_vs_intact.mean(),
                     "mean_p_gold": g.p_gold.mean(),
                     "mean_conf": g.confidence.mean()})
    w(pd.DataFrame(rows), arm, "e3_summary")
    return d


def e4(arm):
    c = ARMS[arm]
    d = pd.read_csv(c["dir"] / c["e4"] / "raw.csv")
    cf = "conf" if f"{c['e4_arms'][0]}_conf" in d.columns else "max_prob"
    rows = []
    for a in c["e4_arms"]:
        acc, lo, hi = boot_ci(d[f"{a}_correct"].values, seed=9)
        conf_col = f"{a}_{cf}"
        rows.append({"arm": arm, "condition": a, "label": E4_LABEL[a], "n": len(d),
                     "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "mean_p_gold": d[f"{a}_p_gold"].mean(),
                     "mean_conf": d[conf_col].mean() if conf_col in d else np.nan,
                     "ece": ece(d[conf_col], d[f"{a}_correct"]) if conf_col in d else np.nan})
    w(pd.DataFrame(rows), arm, "e4_arms")

    bn, en = ("bn_ml", "en_ml") if "bn_ml" in c["e4_arms"] else ("bn", "en")
    cell = np.select(
        [(d[f"{bn}_correct"] == 1) & (d[f"{en}_correct"] == 1),
         (d[f"{bn}_correct"] == 0) & (d[f"{en}_correct"] == 1),
         (d[f"{bn}_correct"] == 1) & (d[f"{en}_correct"] == 0)],
        ["both_right", "bangla_gap", "english_gap"], default="both_wrong")
    loc = pd.Series(cell).value_counts(normalize=True).rename("share").reset_index()
    loc.columns = ["cell", "share"]
    loc["n"] = pd.Series(cell).value_counts().values
    loc["arm"] = arm
    w(loc, arm, "e4_localisation")

    n01 = int(((d[f"{bn}_correct"] == 0) & (d[f"{en}_correct"] == 1)).sum())
    n10 = int(((d[f"{bn}_correct"] == 1) & (d[f"{en}_correct"] == 0)).sum())
    w(pd.DataFrame([{
        "arm": arm, "comparison": f"{en} vs {bn}", "n_paired": len(d),
        "bn_only_right": n10, "en_only_right": n01,
        "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue,
        "delta_accuracy": d[f"{en}_correct"].mean() - d[f"{bn}_correct"].mean(),
    }]), arm, "e4_mcnemar")
    return d


def e5(arm, d1):
    rows = []
    for fam in FAM_ORDER:
        g = d1[d1.family == fam]
        rows.append({"arm": arm, "family": fam, "label": FAM_LABEL[fam], "n": len(g),
                     "accuracy": g.correct.mean(), "mean_conf": g.confidence.mean(),
                     "overconfidence": g.confidence.mean() - g.correct.mean(),
                     "ece": ece(g.confidence, g.correct),
                     "aurc_conf": aurc(g.confidence.values, g.correct.values),
                     "aurc_random": 1 - g.correct.mean(),
                     "aurc_conf_head": aurc(np.nan_to_num(g.conf_head.values, nan=0.0),
                                            g.correct.values)
                     if g.conf_head.notna().any() else np.nan,
                     "ece_conf_head": ece(g.conf_head, g.correct)
                     if g.conf_head.notna().any() else np.nan})
    w(pd.DataFrame(rows), arm, "e5_calibration")

    lit = d1[d1.family.isin(["D-lit", "D-hard"]) & d1.lit_key.notna()]
    rows = []
    for name, sub in [("correct", lit[lit.correct == 1]),
                      ("chose_literal", lit[lit.chose_literal == 1]),
                      ("other_wrong", lit[(lit.correct == 0) & (lit.chose_literal == 0)])]:
        m, lo, hi = boot_ci(sub.confidence.values, seed=10)
        rows.append({"arm": arm, "outcome": name, "n": len(sub),
                     "share": len(sub) / len(lit), "mean_conf": m,
                     "conf_lo": lo, "conf_hi": hi,
                     "median_conf": sub.confidence.median(),
                     "mean_conf_head": sub.conf_head.mean(),
                     "mean_p_gold": sub.p_gold.mean(),
                     "mean_p_literal": sub.p_literal.mean()})
    w(pd.DataFrame(rows), arm, "e5_confidence_by_outcome")

    rows = []
    for fam in FAM_ORDER:
        g = d1[d1.family == fam]
        o = np.argsort(-g.confidence.values)
        corr = g.correct.values[o]
        cum = np.cumsum(corr) / np.arange(1, len(corr) + 1)
        for cov in np.arange(0.05, 1.001, 0.05):
            i = max(0, int(round(cov * len(corr))) - 1)
            rows.append({"arm": arm, "family": fam, "coverage": round(cov, 2),
                         "accuracy": cum[i], "threshold": g.confidence.values[o][i]})
    w(pd.DataFrame(rows), arm, "e5_risk_coverage")

    rows, edges = [], np.linspace(0, 1, 11)
    for fam in FAM_ORDER:
        g = d1[d1.family == fam]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (g.confidence > lo) & (g.confidence <= hi)
            if m.sum() < 10:
                continue
            rows.append({"arm": arm, "family": fam, "bin_lo": lo, "bin_hi": hi,
                         "n": int(m.sum()), "mean_conf": g.confidence[m].mean(),
                         "accuracy": g.correct[m].mean()})
    w(pd.DataFrame(rows), arm, "e5_reliability")


def e6(arm):
    c = ARMS[arm]
    d = pd.read_csv(c["dir"] / "e6_options" / "raw.csv")
    d["confidence"] = d[c["conf"]] if c["conf"] in d.columns else d["conf"]
    base = d[d.form == "O0_full_bn"].set_index("id")
    rows = []
    for f in FORM_LABEL:
        g = d[d.form == f].set_index("id")
        acc, lo, hi = boot_ci(g.correct.values, seed=11)
        same = (g.pred == base.pred.reindex(g.index)).astype(float)
        rows.append({"arm": arm, "form": f, "label": FORM_LABEL[f], "n": len(g),
                     "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "delta_vs_full": acc - base.correct.mean(),
                     "answer_agreement_with_full": same.mean(),
                     "mean_jsd_vs_full": g.jsd_vs_full.mean(),
                     "mean_p_gold": g.p_gold.mean(),
                     "mean_conf": g.confidence.mean(),
                     "mean_opt_tokens": g.opt_tokens.mean()})
    s = pd.DataFrame(rows)
    full_acc = s.loc[s.form == "O0_full_bn", "accuracy"].iat[0]
    null_acc = s.loc[s.form == "O4_labels", "accuracy"].iat[0]
    s["semantic_signal_retained"] = (s.accuracy - null_acc) / (full_acc - null_acc)
    w(s, arm, "e6_summary")
    return d


def run_arm(arm):
    print(f"\n=== {arm} : {ARMS[arm]['label']}")
    d1 = e1(arm)
    e7(arm)
    e2(arm)
    e3(arm)
    e4(arm)
    e5(arm, d1)
    e6(arm)


def main():
    for arm in ARMS:
        run_arm(arm)
    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()
