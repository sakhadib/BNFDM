#!/usr/bin/env python3
"""Jev 1.13 arm -> analysis/tables/jev-1.13/*.csv

Maps the Jev notebook's blocks onto the same table schema the Bangla arms use,
so figures.py can treat all three uniformly:

    j2_literal_en  -> e1_summary        (ENGLISH ladder; language='en')
    j4_bn_control  -> e1_bangla_control (BANGLA, D-rand only; language='bn')
    j1_order       -> e7_*
    j5_options_en  -> e6_summary
    j3_ceiling_en  -> e3_ceiling

Every emitted row carries a `language` column. Nothing here ever puts an
English Jev number on the same axis as a Bangla Laya/Lod number without it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bnfdm_style import boot_ci, ece, aurc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "JEV_RUN" / "results" / "jev-1.13"
ARM = "jev-1.13"
OUT = ROOT / "analysis" / "tables" / ARM
OUT.mkdir(parents=True, exist_ok=True)

K, CHANCE = 4, 0.25
OPT_KEYS = ["A", "B", "C", "D"]
FAM = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]
FAM_LABEL = {"D-rand": "random", "D-surf": "surface-matched", "D-sem": "tag-matched",
             "D-lit": "+ own literal gloss", "D-hard": "literal + surface + tag"}
FORM_LABEL = {"O0_full_en": "full English gloss", "O1_trunc6": "truncated to 6 words",
              "O2_scrambled": "word-scrambled", "O3_bangla": "Bangla gloss",
              "O4_labels": "semantics removed"}


def w(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  [tbl] {ARM}/{name}.csv ({len(df)})")
    return df


def _mark(d):
    d["correct"] = (d.pred == d.gold_key).astype(int)
    if "lit_key" in d.columns:
        d["chose_literal"] = ((d.lit_key.notna()) & (d.pred == d.lit_key)).astype(int)
        d["p_literal"] = d.apply(
            lambda r: r[f"p_{r.lit_key}"] if isinstance(r.lit_key, str) else np.nan,
            axis=1)
    d["p_gold"] = d.apply(lambda r: r[f"p_{r.gold_key}"], axis=1)
    d["confidence"] = d.max_prob
    return d


# ════════════════════════════════════════════════════════════════════
def ladder_en():
    d = _mark(pd.read_csv(RES / "j2_literal_en" / "raw.csv"))
    ceil = _mark(pd.read_csv(RES / "j3_ceiling_en" / "raw.csv"))
    w(d, "e1_raw_en")

    rows = []
    for fam in FAM:
        src = d if fam in d.family.unique() else ceil
        g = src[src.family == fam]
        if not len(g):
            continue
        # D-rand exists in both; prefer the in-pool version for comparability
        acc, lo, hi = boot_ci(g.correct.values, seed=1)
        lar = lar_lo = lar_hi = np.nan
        if "lit_key" in g.columns and g.lit_key.notna().any():
            gl = g[g.lit_key.notna()]
            lar, lar_lo, lar_hi = boot_ci(gl.chose_literal.values, seed=2)
        rows.append({
            "arm": ARM, "language": "en", "family": fam, "label": FAM_LABEL[fam],
            "n": len(g), "pool": "en_literal_3265" if src is d else "en_full_8761",
            "accuracy": acc, "acc_lo": lo, "acc_hi": hi, "chance": CHANCE,
            "below_chance": acc < CHANCE,
            "LAR": lar, "LAR_lo": lar_lo, "LAR_hi": lar_hi,
            "mean_p_gold": g.p_gold.mean(),
            "mean_p_literal": g.p_literal.mean() if "p_literal" in g else np.nan,
            "mean_conf": g.confidence.mean(), "mean_conf_head": g.conf_head.mean(),
            "ece": ece(g.confidence, g.correct),
            "aurc": aurc(g.confidence.values, g.correct.values),
            "aurc_random": 1 - g.correct.mean(),
        })
    w(pd.DataFrame(rows), "e1_summary")

    dec = []
    for fam in ["D-lit", "D-hard"]:
        g = d[(d.family == fam) & d.lit_key.notna()]
        dec.append({"arm": ARM, "language": "en", "family": fam, "n": len(g),
                    "chose_gold": (g.correct == 1).mean(),
                    "chose_literal": (g.chose_literal == 1).mean(),
                    "chose_other": ((g.correct == 0) & (g.chose_literal == 0)).mean()})
    w(pd.DataFrame(dec), "e1_outcome_decomposition")

    lit = d[d.family.isin(["D-lit", "D-hard"]) & d.lit_key.notna()]
    rows = []
    for name, sub in [("correct", lit[lit.correct == 1]),
                      ("chose_literal", lit[lit.chose_literal == 1]),
                      ("other_wrong", lit[(lit.correct == 0) & (lit.chose_literal == 0)])]:
        m, lo, hi = boot_ci(sub.confidence.values, seed=10) if len(sub) else (np.nan,)*3
        rows.append({"arm": ARM, "language": "en", "outcome": name, "n": len(sub),
                     "share": len(sub) / len(lit), "mean_conf": m,
                     "conf_lo": lo, "conf_hi": hi,
                     "mean_conf_head": sub.conf_head.mean(),
                     "mean_p_gold": sub.p_gold.mean(),
                     "mean_p_literal": sub.p_literal.mean()})
    w(pd.DataFrame(rows), "e5_confidence_by_outcome")

    rows, edges = [], np.linspace(0, 1, 11)
    for fam in ["D-rand", "D-lit", "D-hard"]:
        g = d[d.family == fam]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (g.confidence > lo) & (g.confidence <= hi)
            if m.sum() < 10:
                continue
            rows.append({"arm": ARM, "language": "en", "family": fam,
                         "bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                         "mean_conf": g.confidence[m].mean(),
                         "accuracy": g.correct[m].mean()})
    w(pd.DataFrame(rows), "e5_reliability")

    rows = []
    for fam in ["D-rand", "D-lit", "D-hard"]:
        g = d[d.family == fam]
        o = np.argsort(-g.confidence.values)
        corr = g.correct.values[o]
        cum = np.cumsum(corr) / np.arange(1, len(corr) + 1)
        for cov in np.arange(0.05, 1.001, 0.05):
            i = max(0, int(round(cov * len(corr))) - 1)
            rows.append({"arm": ARM, "language": "en", "family": fam,
                         "coverage": round(cov, 2), "accuracy": cum[i]})
    w(pd.DataFrame(rows), "e5_risk_coverage")

    rows = []
    for fam in ["D-rand", "D-lit", "D-hard"]:
        g = d[d.family == fam]
        rows.append({"arm": ARM, "language": "en", "family": fam, "n": len(g),
                     "accuracy": g.correct.mean(), "mean_conf": g.confidence.mean(),
                     "overconfidence": g.confidence.mean() - g.correct.mean(),
                     "ece": ece(g.confidence, g.correct),
                     "aurc_conf": aurc(g.confidence.values, g.correct.values),
                     "aurc_random": 1 - g.correct.mean(),
                     "ece_conf_head": ece(g.conf_head, g.correct),
                     "aurc_conf_head": aurc(g.conf_head.values, g.correct.values)})
    w(pd.DataFrame(rows), "e5_calibration")
    return d


def bangla_control():
    """J4: the fully-Bangla D-rand task. Directly comparable to Laya and Lod."""
    d = _mark(pd.read_csv(RES / "j4_bn_control" / "raw.csv").assign(family="D-rand"))
    acc, lo, hi = boot_ci(d.correct.values, seed=4)
    w(pd.DataFrame([{
        "arm": ARM, "language": "bn", "family": "D-rand", "n": len(d),
        "accuracy": acc, "acc_lo": lo, "acc_hi": hi, "chance": CHANCE,
        "mean_p_gold": d.p_gold.mean(), "mean_conf": d.confidence.mean(),
        "mean_conf_head": d.conf_head.mean(),
        "ece": ece(d.confidence, d.correct),
    }]), "e1_bangla_control")
    w(d, "e1_raw_bn")

    rows = []
    for pos in range(K):
        g = d[d.gold_pos == pos]
        a, l, h = boot_ci(g.correct.values, seed=5)
        rows.append({"arm": ARM, "language": "bn", "position": pos,
                     "key": OPT_KEYS[pos],
                     "gold_share": float((d.gold_pos == pos).mean()),
                     "selection_share": float((d.pred == OPT_KEYS[pos]).mean()),
                     "accuracy_when_gold_here": a, "acc_lo": l, "acc_hi": h,
                     "mean_p_at_position": float(d[f"p_{OPT_KEYS[pos]}"].mean())})
    w(pd.DataFrame(rows), "e1_position_bias")
    return d


def order():
    d = pd.read_csv(RES / "j1_order" / "raw.csv")
    d["perm"] = d.perm.astype(str).str.zfill(K)
    if "chosen_opt" not in d.columns:
        d["chosen_opt"] = [int(p[OPT_KEYS.index(pr)]) for p, pr in zip(d.perm, d.pred)]
    if "correct" not in d.columns:
        d["correct"] = (d.pred == d.gold_key).astype(int)
    if "p_gold" not in d.columns:
        d["p_gold"] = d.apply(lambda r: r[f"p_{r.gold_key}"], axis=1)

    g = d.groupby("id")
    per = pd.DataFrame({
        "n_distinct_option": g.chosen_opt.nunique(),
        "n_distinct_key": g.pred.nunique(),
        "accuracy": g.correct.mean(),
        "p_gold_mean": g.p_gold.mean(), "p_gold_std": g.p_gold.std(),
        "p_gold_range": g.p_gold.agg(lambda s: s.max() - s.min()),
        "conf_range": g.max_prob.agg(lambda s: s.max() - s.min()),
    }).reset_index()
    per["answer_unstable"] = per.n_distinct_option > 1
    per["correctness_varies"] = (per.accuracy > 0) & (per.accuracy < 1)
    per["arm"] = ARM
    w(per, "e7_per_item")

    hist = per.n_distinct_option.value_counts().sort_index().rename("items").reset_index()
    hist.columns = ["n_distinct_option", "items"]
    hist["share"] = hist["items"] / hist["items"].sum()
    hist["arm"] = ARM
    w(hist, "e7_instability_hist")

    w(pd.DataFrame([{
        "arm": ARM, "language": "en", "n_items": int(per.shape[0]),
        "n_permutations": int(d.perm_id.nunique()),
        "answer_unstable_share": float(per.answer_unstable.mean()),
        "correctness_varies_share": float(per.correctness_varies.mean()),
        "mean_distinct_options": float(per.n_distinct_option.mean()),
        "mean_distinct_keys": float(per.n_distinct_key.mean()),
        "mean_p_gold_std": float(per.p_gold_std.mean()),
        "mean_p_gold_range": float(per.p_gold_range.mean()),
        "max_p_gold_range": float(per.p_gold_range.max()),
        "mean_conf_range": float(per.conf_range.mean()),
        "overall_accuracy": float(d.correct.mean()),
    }]).round(6), "e7_headline")

    rows = []
    for pos in range(K):
        gg = d[d.gold_pos == pos]
        a, l, h = boot_ci(gg.correct.values, seed=6)
        rows.append({"arm": ARM, "position": pos, "key": OPT_KEYS[pos], "n": len(gg),
                     "accuracy_when_gold_here": a, "acc_lo": l, "acc_hi": h,
                     "mean_p_gold": gg.p_gold.mean(),
                     "selection_share": float((d.pred == OPT_KEYS[pos]).mean()),
                     "mean_p_at_position": float(d[f"p_{OPT_KEYS[pos]}"].mean())})
    w(pd.DataFrame(rows), "e7_position_bias")


def options():
    d = _mark(pd.read_csv(RES / "j5_options_en" / "raw.csv"))
    base = d[d.form == "O0_full_en"].set_index("id")
    rows = []
    for f in FORM_LABEL:
        g = d[d.form == f].set_index("id")
        if not len(g):
            continue
        acc, lo, hi = boot_ci(g.correct.values, seed=11)
        same = (g.pred == base.pred.reindex(g.index)).astype(float)
        rows.append({"arm": ARM, "language": "en", "form": f, "label": FORM_LABEL[f],
                     "n": len(g), "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "delta_vs_full": acc - base.correct.mean(),
                     "answer_agreement_with_full": same.mean(),
                     "mean_p_gold": g.p_gold.mean(),
                     "mean_conf": g.confidence.mean(),
                     "mean_conf_head": g.conf_head.mean(),
                     "ece": ece(g.confidence, g.correct),
                     "mean_opt_tokens": g.opt_tokens.mean()})
    s = pd.DataFrame(rows)
    fa = s.loc[s.form == "O0_full_en", "accuracy"].iat[0]
    na = s.loc[s.form == "O4_labels", "accuracy"].iat[0]
    s["semantic_signal_retained"] = (s.accuracy - na) / (fa - na)
    w(s, "e6_summary")


def main():
    print(f"=== {ARM}")
    ladder_en()
    bangla_control()
    order()
    options()
    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()
