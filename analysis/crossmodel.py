#!/usr/bin/env python3
"""Three-arm comparison. Writes analysis/tables/_cross/.

The central methodological constraint: Jev's ladder is ENGLISH while Laya's and
Lod's are BANGLA, so literal-capture numbers are not directly comparable across
all three. Exactly one condition is identical across all three models — the
fully-Bangla D-rand meaning-choice task (Laya/Lod e1 D-rand, Jev j4_bn_control)
— and that is the only place a three-way accuracy claim is made.

Everything else is reported as a 2-way paired test (Laya vs Lod, Bangla,
byte-identical questions) with Jev alongside and its language marked.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from bnfdm_style import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TBL = ROOT / "analysis" / "tables"
OUT = TBL / "_cross"
OUT.mkdir(parents=True, exist_ok=True)

LAYA = ROOT / "LAYA_RUN" / "results" / "laya-ml"
LOD = ROOT / "LOD_RUN" / "results" / "lod-lille-0.6b"
JEV = ROOT / "JEV_RUN" / "results" / "jev-1.13"
LBL = {"laya-ml": "Laya-ml", "lod-lille-0.6b": "Lod-lille", "jev-1.13": "Jev-1.13"}
FAM = ["D-rand", "D-surf", "D-sem", "D-lit", "D-hard"]


def w(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  [tbl] _cross/{name}.csv ({len(df)})")
    return df


def verify_pairing():
    """Laya and Lod must have seen byte-identical questions on the overlap."""
    L = pd.read_csv(LAYA / "e1_ladder" / "raw.csv")
    D = pd.read_csv(LOD / "e1_ladder" / "raw.csv")
    m = L[["id", "family", "gold_key", "gold_pos", "idiom", "lit_key"]].merge(
        D[["id", "family", "gold_key", "gold_pos", "idiom", "lit_key"]],
        on=["id", "family"], suffixes=("_L", "_D"))
    lk = m.dropna(subset=["lit_key_L", "lit_key_D"])
    r = w(pd.DataFrame([{
        "pair": "laya vs lod (Bangla)",
        "n_overlap_rows": len(m), "n_overlap_items": int(m.id.nunique()),
        "gold_key_match": float((m.gold_key_L == m.gold_key_D).mean()),
        "gold_pos_match": float((m.gold_pos_L == m.gold_pos_D).mean()),
        "idiom_match": float((m.idiom_L == m.idiom_D).mean()),
        "literal_key_match": float((lk.lit_key_L == lk.lit_key_D).mean()),
    }]).round(6), "pairing_verification")
    ok = all(r.iloc[0][c] == 1.0 for c in
             ["gold_key_match", "gold_pos_match", "idiom_match", "literal_key_match"])
    print(f"  pairing verified: {ok}")
    return ok, set(m.id.unique())


def three_way_bangla():
    """The one condition identical across all three models."""
    L = pd.read_csv(LAYA / "e1_ladder" / "raw.csv")
    L = L[L.family == "D-rand"]
    D = pd.read_csv(LOD / "e1_ladder" / "raw.csv")
    D = D[D.family == "D-rand"]
    J = pd.read_csv(JEV / "j4_bn_control" / "raw.csv")
    J["correct"] = (J.pred == J.gold_key).astype(int)
    J["confidence"] = J.max_prob

    rows = []
    for arm, g, conf in [("laya-ml", L, "conf"), ("lod-lille-0.6b", D, "max_prob"),
                         ("jev-1.13", J, "confidence")]:
        acc, lo, hi = boot_ci(g.correct.values, seed=1)
        rows.append({"arm": arm, "label": LBL[arm], "language": "bn",
                     "condition": "D-rand meaning choice", "n": len(g),
                     "accuracy": acc, "acc_lo": lo, "acc_hi": hi, "chance": 0.25,
                     "mean_conf": g[conf].mean()})
    t = w(pd.DataFrame(rows), "three_way_bangla_drand")

    # paired where item sets overlap
    pair = []
    for a, A, b, B in [("laya-ml", L, "jev-1.13", J),
                       ("lod-lille-0.6b", D, "jev-1.13", J),
                       ("laya-ml", L, "lod-lille-0.6b", D)]:
        M = (A[["id", "correct"]].rename(columns={"correct": "a"})
             .merge(B[["id", "correct"]].rename(columns={"correct": "b"}), on="id"))
        n01 = int(((M.a == 0) & (M.b == 1)).sum())
        n10 = int(((M.a == 1) & (M.b == 0)).sum())
        pair.append({"model_a": LBL[a], "model_b": LBL[b], "n_paired": len(M),
                     "acc_a": M.a.mean(), "acc_b": M.b.mean(),
                     "delta_b_minus_a": M.b.mean() - M.a.mean(),
                     "a_only_right": n10, "b_only_right": n01,
                     "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue
                     if (n01 + n10) else np.nan})
    w(pd.DataFrame(pair), "three_way_bangla_mcnemar")
    return t


def literal_capture_matrix(pool_ids):
    """LAR for every (model, language) cell we actually measured.

    Laya/Lod: Bangla, reported both full-set and restricted to the English-literal
    pool so the Jev row has a like-for-like reference population.
    Jev: English only. The empty cells are the experiments not run.
    """
    rows = []
    for arm, path, conf in [("laya-ml", LAYA, "conf"), ("lod-lille-0.6b", LOD, "max_prob")]:
        d = pd.read_csv(path / "e1_ladder" / "raw.csv")
        for fam in ["D-lit", "D-hard"]:
            g = d[d.family == fam]
            gi = g[g.id.isin(pool_ids)]
            for pop, gg in [("full", g), ("en_literal_pool", gi)]:
                if not len(gg):
                    continue
                lar, lo, hi = boot_ci(gg.chose_literal.values, seed=2)
                acc, alo, ahi = boot_ci(gg.correct.values, seed=3)
                rows.append({"arm": arm, "label": LBL[arm], "language": "bn",
                             "family": fam, "population": pop, "n": len(gg),
                             "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                             "LAR": lar, "LAR_lo": lo, "LAR_hi": hi,
                             "mean_conf": gg[conf].mean()})
    J = pd.read_csv(JEV / "j2_literal_en" / "raw.csv")
    J["correct"] = (J.pred == J.gold_key).astype(int)
    J["chose_literal"] = ((J.lit_key.notna()) & (J.pred == J.lit_key)).astype(int)
    for fam in ["D-lit", "D-hard"]:
        g = J[J.family == fam]
        lar, lo, hi = boot_ci(g.chose_literal.values, seed=2)
        acc, alo, ahi = boot_ci(g.correct.values, seed=3)
        rows.append({"arm": "jev-1.13", "label": "Jev-1.13", "language": "en",
                     "family": fam, "population": "en_literal_pool", "n": len(g),
                     "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                     "LAR": lar, "LAR_lo": lo, "LAR_hi": hi,
                     "mean_conf": g.max_prob.mean()})
    t = w(pd.DataFrame(rows), "literal_capture_matrix")

    w(pd.DataFrame([{
        "note": "Jev LAR is ENGLISH; Laya/Lod LAR is BANGLA. The populations are "
                "the same 3,265 items but the language is not, so the cross-model "
                "LAR difference is confounded with language. The missing cells "
                "(Jev on Bangla literal; Laya/Lod on English literal) are the "
                "experiments that would resolve it.",
        "laya_bn_lar_pool": float(t[(t.arm == "laya-ml") & (t.family == "D-lit") &
                                    (t.population == "en_literal_pool")].LAR.iat[0]),
        "lod_bn_lar_pool": float(t[(t.arm == "lod-lille-0.6b") & (t.family == "D-lit") &
                                   (t.population == "en_literal_pool")].LAR.iat[0]),
        "jev_en_lar": float(t[(t.arm == "jev-1.13") & (t.family == "D-lit")].LAR.iat[0]),
    }]), "literal_capture_caveat")
    return t


def no_signal_control():
    """O4_labels: options stripped of semantics. Accuracy must be chance by
    construction, so reported confidence there is a pure overconfidence probe —
    and it is the one option-ablation cell that is comparable across all three."""
    rows = []
    for arm, path, form, conf in [
            ("laya-ml", LAYA / "e6_options" / "raw.csv", "O4_labels", "conf"),
            ("lod-lille-0.6b", LOD / "e6_options" / "raw.csv", "O4_labels", "max_prob"),
            ("jev-1.13", JEV / "j5_options_en" / "raw.csv", "O4_labels", "max_prob")]:
        d = pd.read_csv(path)
        if "correct" not in d.columns:
            d["correct"] = (d.pred == d.gold_key).astype(int)
        g = d[d.form == form]
        full_form = "O0_full_bn" if arm != "jev-1.13" else "O0_full_en"
        f = d[d.form == full_form]
        acc, lo, hi = boot_ci(g.correct.values, seed=7)
        rows.append({"arm": arm, "label": LBL[arm],
                     "language": "bn" if arm != "jev-1.13" else "en",
                     "n": len(g), "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                     "chance": 0.25, "mean_conf": g[conf].mean(),
                     "overconfidence": g[conf].mean() - acc,
                     "accuracy_with_full_options": f.correct.mean(),
                     "conf_with_full_options": f[conf].mean()})
    return w(pd.DataFrame(rows), "no_signal_control")


def order_three_way():
    rows = []
    for arm in ["laya-ml", "lod-lille-0.6b", "jev-1.13"]:
        h = pd.read_csv(TBL / arm / "e7_headline.csv").iloc[0]
        p = pd.read_csv(TBL / arm / "e1_position_bias.csv")
        rows.append({"arm": arm, "label": LBL[arm],
                     "language": h.get("language", "bn"),
                     "n_items": int(h.n_items),
                     "answer_unstable_share": h.answer_unstable_share,
                     "correctness_varies_share": h.correctness_varies_share,
                     "mean_distinct_options": h.mean_distinct_options,
                     "mean_p_gold_range": h.mean_p_gold_range,
                     "mean_conf_range": h.mean_conf_range,
                     "selection_share_max": p.selection_share.max(),
                     "selection_share_min": p.selection_share.min(),
                     "selection_share_spread": p.selection_share.max() - p.selection_share.min()})
    return w(pd.DataFrame(rows), "order_three_way")


def paired_laya_lod(sub_ids):
    out = []
    for exp, keys in [("e1_ladder", ["id", "family"]), ("e3_unit", ["id", "variant"]),
                      ("e6_options", ["id", "form"])]:
        a, b = LAYA / exp / "raw.csv", LOD / exp / "raw.csv"
        if not (a.exists() and b.exists()):
            continue
        A = pd.read_csv(a)[keys + ["correct"]].rename(columns={"correct": "laya"})
        B = pd.read_csv(b)[keys + ["correct"]].rename(columns={"correct": "lod"})
        M = A.merge(B, on=keys, how="inner")
        for grp, g in M.groupby(keys[1]):
            n01 = int(((g.laya == 0) & (g.lod == 1)).sum())
            n10 = int(((g.laya == 1) & (g.lod == 0)).sum())
            out.append({"experiment": exp, "group": grp, "n_paired": len(g),
                        "laya_on_overlap": g.laya.mean(), "lod": g.lod.mean(),
                        "delta_lod_minus_laya": g.lod.mean() - g.laya.mean(),
                        "laya_only_right": n10, "lod_only_right": n01,
                        "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue
                        if (n01 + n10) else np.nan})
    return w(pd.DataFrame(out).round(6), "paired_mcnemar_laya_lod")


def subset_check(sub_ids):
    L = pd.read_csv(LAYA / "e1_ladder" / "raw.csv")
    rows = []
    for fam, g in L.groupby("family"):
        s = g[g.id.isin(sub_ids)]
        full, flo, fhi = boot_ci(g.correct.values, seed=20)
        sub, slo, shi = boot_ci(s.correct.values, seed=21)
        rows.append({"family": fam, "n_full": len(g), "n_subset": len(s),
                     "laya_full": full, "full_lo": flo, "full_hi": fhi,
                     "laya_subset": sub, "sub_lo": slo, "sub_hi": shi,
                     "drift": sub - full,
                     "subset_within_full_CI": bool(flo <= sub <= fhi)})
    return w(pd.DataFrame(rows).round(6), "subset_representativeness")


def main():
    print("Cross-model (three arms)")
    ok, _ = verify_pairing()
    if not ok:
        raise SystemExit("PAIRING FAILED")
    sub_ids = set(pd.read_json(ROOT / "LOD_RUN" / "data" / "splits" / "sub10.jsonl",
                               lines=True).id)
    pool_ids = set(pd.read_json(ROOT / "JEV_RUN" / "data" / "splits" /
                                "pool_en_literal.jsonl", lines=True).id)
    three_way_bangla()
    literal_capture_matrix(pool_ids)
    no_signal_control()
    order_three_way()
    paired_laya_lod(sub_ids)
    subset_check(sub_ids)
    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()
