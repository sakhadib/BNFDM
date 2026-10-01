#!/usr/bin/env python3
"""Paired cross-model comparison: Laya-multilingual vs Lod-lille.

The two arms were run independently (different Drive accounts, different
notebooks) from a hardcoded reproducibility contract: same dataset, same
dedup/filter pipeline, same k, option keys, instruction string, distractor
families and per-item seeds. Lod ran a fixed 10% subset.

`verify_pairing()` checks that contract empirically before any paired test is
reported. If the option sets were not byte-identical the McNemar tests below
would be meaningless, so a failure here is fatal rather than cosmetic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from bnfdm_style import boot_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "tables" / "_cross"
OUT.mkdir(parents=True, exist_ok=True)

A = ("laya-ml", ROOT / "LAYA_RUN" / "results" / "laya-ml")
B = ("lod-lille-0.6b", ROOT / "LOD_RUN" / "results" / "lod-lille-0.6b")
LBL = {A[0]: "Laya-ml", B[0]: "Lod-lille"}


def w(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    print(f"  [tbl] _cross/{name}.csv ({len(df)})")
    return df


def verify_pairing():
    """Both arms must have seen byte-identical questions on the overlap."""
    L = pd.read_csv(A[1] / "e1_ladder" / "raw.csv")
    D = pd.read_csv(B[1] / "e1_ladder" / "raw.csv")
    m = L[["id", "family", "gold_key", "gold_pos", "idiom", "lit_key"]].merge(
        D[["id", "family", "gold_key", "gold_pos", "idiom", "lit_key"]],
        on=["id", "family"], suffixes=("_L", "_D"))
    lk = m.dropna(subset=["lit_key_L", "lit_key_D"])
    rows = [{
        "n_overlap_rows": len(m), "n_overlap_items": int(m.id.nunique()),
        "gold_key_match": float((m.gold_key_L == m.gold_key_D).mean()),
        "gold_pos_match": float((m.gold_pos_L == m.gold_pos_D).mean()),
        "idiom_match": float((m.idiom_L == m.idiom_D).mean()),
        "literal_key_match": float((lk.lit_key_L == lk.lit_key_D).mean()),
        "n_literal_rows": len(lk),
    }]
    r = w(pd.DataFrame(rows).round(6), "pairing_verification")
    ok = all(r.iloc[0][c] == 1.0 for c in
             ["gold_key_match", "gold_pos_match", "idiom_match", "literal_key_match"])
    print(f"  pairing verified: {ok}")
    return ok, set(m.id.unique())


def subset_representativeness(sub_ids):
    """Does the 10% subset move Laya's own numbers? If not, it is benign."""
    L = pd.read_csv(A[1] / "e1_ladder" / "raw.csv")
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


def paired(sub_ids):
    """McNemar per group, on the overlap only."""
    out = []
    for exp, keys in [("e1_ladder", ["id", "family"]),
                      ("e3_unit", ["id", "variant"]),
                      ("e6_options", ["id", "form"])]:
        fa, fb = A[1] / exp / "raw.csv", B[1] / exp / "raw.csv"
        if not (fa.exists() and fb.exists()):
            continue
        L = pd.read_csv(fa)[keys + ["correct"]].rename(columns={"correct": "laya"})
        D = pd.read_csv(fb)[keys + ["correct"]].rename(columns={"correct": "lod"})
        M = L.merge(D, on=keys, how="inner")
        for grp, g in M.groupby(keys[1]):
            n01 = int(((g.laya == 0) & (g.lod == 1)).sum())
            n10 = int(((g.laya == 1) & (g.lod == 0)).sum())
            out.append({
                "experiment": exp, "group": grp, "n_paired": len(g),
                "laya_on_overlap": g.laya.mean(), "lod": g.lod.mean(),
                "delta_lod_minus_laya": g.lod.mean() - g.laya.mean(),
                "laya_only_right": n10, "lod_only_right": n01,
                "both_right": int(((g.laya == 1) & (g.lod == 1)).sum()),
                "both_wrong": int(((g.laya == 0) & (g.lod == 0)).sum()),
                "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue
                if (n01 + n10) else np.nan,
            })
    return w(pd.DataFrame(out).round(6), "paired_mcnemar")


def headline():
    """Side-by-side of the four claims, both arms."""
    rows = []
    for arm, _ in (A, B):
        t = ROOT / "analysis" / "tables" / arm
        e1 = pd.read_csv(t / "e1_summary.csv").set_index("family")
        e7 = pd.read_csv(t / "e7_headline.csv").iloc[0]
        e4 = pd.read_csv(t / "e4_arms.csv")
        e4l = pd.read_csv(t / "e4_localisation.csv").set_index("cell")
        e5 = pd.read_csv(t / "e5_confidence_by_outcome.csv").set_index("outcome")
        e6 = pd.read_csv(t / "e6_summary.csv").set_index("form")
        e3 = pd.read_csv(t / "e3_summary.csv").set_index("variant")
        e2 = pd.read_csv(t / "e2_summary.csv").iloc[0]
        bn = "bn_ml" if (e4.condition == "bn_ml").any() else "bn"
        en = "en_ml" if (e4.condition == "en_ml").any() else "en"
        e4i = e4.set_index("condition")
        rows.append({
            "arm": arm, "label": LBL[arm], "n_items": int(e1.loc["D-rand", "n"]),
            "acc_random_distractors": e1.loc["D-rand", "accuracy"],
            "acc_plus_literal": e1.loc["D-lit", "accuracy"],
            "literal_attraction_rate": e1.loc["D-lit", "LAR"],
            "below_chance_with_literal": bool(e1.loc["D-lit", "below_chance"]),
            "conf_when_correct": e5.loc["correct", "mean_conf"],
            "conf_when_captured": e5.loc["chose_literal", "mean_conf"],
            "confidence_inverted": bool(e5.loc["chose_literal", "mean_conf"]
                                        > e5.loc["correct", "mean_conf"]),
            "order_unstable_share": e7.answer_unstable_share,
            "mean_p_gold_range": e7.mean_p_gold_range,
            "acc_bangla": e4i.loc[bn, "accuracy"],
            "acc_english": e4i.loc[en, "accuracy"],
            "language_gap": e4i.loc[en, "accuracy"] - e4i.loc[bn, "accuracy"],
            "bangla_gap_share": e4l.loc["bangla_gap", "share"],
            "shuffle_delta": e3.loc["V1_shuffle", "delta_acc_vs_intact"],
            "option_scramble_delta": e6.loc["O2_scrambled", "delta_vs_full"],
            "option_null_accuracy": e6.loc["O4_labels", "accuracy"],
            "surface_invariance_agreement": e2.agreement,
        })
    return w(pd.DataFrame(rows).round(6), "headline_both_arms")


def main():
    print("Cross-model")
    ok, sub_ids = verify_pairing()
    if not ok:
        raise SystemExit("PAIRING FAILED — paired tests would be invalid")
    subset_representativeness(sub_ids)
    paired(sub_ids)
    headline()
    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()
