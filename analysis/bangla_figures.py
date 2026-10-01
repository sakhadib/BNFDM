#!/usr/bin/env python3
"""Vector figures carrying actual Bengali script.

pdflatex cannot shape Bengali, so these are rendered outside LaTeX by Chromium
(HarfBuzz) and included as PDFs. The output is true vector: embedded font
subsets, extractable text, no raster.

Every string here is read from the released artifacts. Option sets are
reconstructed with the same builder and seed the runs used, and the
reconstruction is checked against the recorded gold and literal keys before
anything is drawn.
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

K = 4
OPT = ["A", "B", "C", "D"]
SEED_M = 21000

CY, MG, YL = "#BEF9FC", "#FCCFFD", "#FCECB3"
CYD, MGD = "#27EBF5", "#F127F5"
GREY, CHAR = "#DCDCDC", "#4D4D4D"

CSS = """
@page { margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; padding: 10px 12px; background: #fff;
       font-family: 'Liberation Serif', 'DejaVu Serif', serif; color: #111; }
.bn { font-family: 'Noto Sans Bengali'; }
.lbl { font-size: 7.6pt; color: #666; }
.hd  { font-size: 8.2pt; color: #222; font-weight: 700; }
table { border-collapse: collapse; width: 100%; }
td, th { padding: 2.6px 5px; vertical-align: middle; }
th { font-size: 7.6pt; color: #555; font-weight: 600; text-align: left;
     border-bottom: 0.6px solid #bbb; }
tr.sep td { border-top: 0.5px solid #e6e6e6; }
.bar { height: 9px; border-radius: 1px; display: inline-block; vertical-align: middle; }
.num { font-size: 8pt; font-variant-numeric: tabular-nums; }
.pcell { white-space: nowrap; }
.tag { font-size: 6.8pt; padding: 0.6px 4px; border-radius: 2px; color: #222; }
.mono { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.2pt; }
"""


# ── data ──────────────────────────────────────────────────────────────
def load():
    dd = pd.read_json(ROOT / "JEV_RUN/data/splits/full.jsonl", lines=True)
    paren = re.compile(r"\(([^)]*)\)")

    def eng_literal(s):
        out = []
        for x in paren.findall(str(s)):
            x = x.strip()
            core = x.replace(" ", "")
            if not core:
                continue
            asc = sum(c.isascii() and c.isalpha() for c in core)
            if asc >= 3 and asc / len(core) >= 0.6:
                out.append(x)
        return " / ".join(out)

    dd["lit_en"] = dd.literal_meaning.map(eng_literal)
    return dd


def builder(dd):
    MEAN = {"bn": dict(zip(dd.id, dd.figurative_meaning_bn)),
            "en": dict(zip(dd.id, dd.figurative_meaning_en))}
    LIT = {"bn": dict(zip(dd.id, dd.literal_meaning)),
           "en": dict(zip(dd.id, dd.lit_en))}
    ALL = list(dd.id)

    def _fill(pool, need, ex, rng):
        pool = [i for i in pool if i not in ex]
        rng.shuffle(pool)
        out = pool[:need]
        while len(out) < need:
            c = rng.choice(ALL)
            if c not in ex and c not in out:
                out.append(c)
        return out

    def build(rid, lang):
        rng = random.Random(SEED_M + int(rid))
        pool = _fill(list(ALL), K - 2, {rid}, rng)
        res = {}
        for lg in ("bn", "en"):
            gold, lit = MEAN[lg][rid], LIT[lg][rid]
            opts = [gold, lit] + [MEAN[lg][i] for i in pool]
            opts = [o for o in opts if str(o).strip()]
            opts = list(dict.fromkeys(opts))[:K]
            while len(opts) < K:
                c = MEAN[lg][rng.choice(ALL)]
                if c and c not in opts:
                    opts.append(c)
            order = list(range(K))
            rng.shuffle(order)
            texts = [opts[i] for i in order]
            res[lg] = {"texts": texts,
                       "gold_key": OPT[texts.index(gold)],
                       "lit_key": OPT[texts.index(lit)] if lit in texts else None}
        return res[lang]
    return build


def render(html, path, w_in, h_in):
    full = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    tmp = Path("/tmp/_bnfig.html")
    tmp.write_text(full, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + str(tmp))
        pg.pdf(path=str(path), width=f"{w_in}in", height=f"{h_in}in",
               print_background=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    print("  [bn-fig]", Path(path).name)


# ── figure 1: one real item, as the model received it ─────────────────
def fig_item(dd, build):
    rid = 1246
    row = dd[dd.id == rid].iloc[0]
    laya = pd.read_csv(ROOT / "added/results/laya-ml/matrix_bn/raw.csv").set_index("id")
    jev = pd.read_csv(ROOT / "added/results/jev-1.13/matrix_bn/raw.csv").set_index("id")
    m = build(rid, "bn")
    rl, rj = laya.loc[rid], jev.loc[rid]
    assert m["gold_key"] == rl.gold_key and m["lit_key"] == rl.lit_key

    rows = ""
    for k, txt in zip(OPT, m["texts"]):
        role = ("gold" if k == m["gold_key"] else
                "literal" if k == m["lit_key"] else "distractor")
        tagbg = {"gold": CY, "literal": CHAR, "distractor": "#F0F0F0"}[role]
        tagfg = "#fff" if role == "literal" else "#222"
        pl, pj = float(rl[f"p_{k}"]), float(rj[f"p_{k}"])
        rows += f"""
        <tr class='sep'>
          <td class='mono' style='width:14px'>{k}</td>
          <td class='bn' style='font-size:9.2pt'>{txt}</td>
          <td style='width:62px'><span class='tag' style='background:{tagbg};color:{tagfg}'>{role}</span></td>
          <td class='pcell' style='width:104px'><span class='bar' style='width:{max(pl*58,1):.1f}px;background:{CYD}'></span><span class='num'> {pl:.2f}</span></td>
          <td class='pcell' style='width:104px'><span class='bar' style='width:{max(pj*58,1):.1f}px;background:#E8B400'></span><span class='num'> {pj:.2f}</span></td>
        </tr>"""

    html = f"""
    <div class='hd'>state sent to the model &nbsp;
      <span class='bn' style='font-size:11pt'>{row.idiom}</span>
      <span class='lbl'>&nbsp; ojon &nbsp;·&nbsp; literally {row.lit_en}
      &nbsp;·&nbsp; English equivalent <i>{row.sim_en}</i></span></div>
    <div class='lbl' style='margin:3px 0 5px 0'>instruction (identical for every
      item and every family): <span class='bn'>নিচের বাংলা বাগধারাটির প্রকৃত ভাবার্থ কোনটি?</span></div>
    <table>
      <tr><th style='width:14px'></th><th>option text, as supplied</th><th>role</th>
          <th>Laya-ml <span class='lbl'>p</span></th><th>Jev-1.13 <span class='lbl'>p</span></th></tr>
      {rows}
    </table>"""
    render(html, OUT / "fig_bn_item.pdf", 6.5, 2.05)


# ── figure 2: real capture examples ───────────────────────────────────
def fig_examples(dd):
    laya = pd.read_csv(ROOT / "added/results/laya-ml/matrix_bn/raw.csv").set_index("id")
    ids = [727, 384, 263, 47, 1246, 55]
    d = dd.set_index("id")
    rows = ""
    for rid in ids:
        if rid not in laya.index:
            continue
        r, s = laya.loc[rid], d.loc[rid]
        pl = float(r.p_literal) if pd.notna(r.p_literal) else 0.0
        rows += f"""
        <tr class='sep'>
          <td class='bn' style='font-size:9.4pt;width:118px'>{s.idiom}</td>
          <td class='bn' style='font-size:8.6pt'>{s.literal_meaning}</td>
          <td class='bn' style='font-size:8.6pt'>{s.figurative_meaning_bn}</td>
          <td class='lbl' style='width:112px'><i>{s.sim_en}</i></td>
          <td class='pcell' style='width:86px'><span class='bar' style='width:{max(pl*40,1):.1f}px;background:{CHAR}'></span><span class='num'> {pl:.2f}</span></td>
        </tr>"""
    html = f"""
    <div class='hd'>items where Laya-ml places its mass on the literal gloss</div>
    <div class='lbl' style='margin:2px 0 5px 0'>Both glosses below were in the
      option list. The last column is the probability the model assigned to the
      literal one.</div>
    <table>
      <tr><th style='width:118px'>idiom</th><th>expert literal gloss</th>
          <th>gold figurative gloss</th><th style='width:112px'>English equivalent</th>
          <th style='width:86px'>p(literal)</th></tr>
      {rows}
    </table>"""
    render(html, OUT / "fig_bn_examples.pdf", 6.5, 2.72)


# ── figure 3: what the manipulations do to Bangla text ────────────────
def fig_manipulations(dd):
    rid = 47
    e3 = pd.read_csv(ROOT / "LAYA_RUN/results/laya-ml/e3_unit/raw.csv")
    s = e3[e3.id == rid]
    d = dd.set_index("id")
    name = {"V0_intact": "intact", "V1_shuffle": "word shuffle",
            "V2_reverse": "reversed", "V3_delete1": "delete one word",
            "V4_substitute1": "substitute one word",
            "V5_in_sentence": "in its carrier sentence"}
    left = ""
    for v in ["V0_intact", "V1_shuffle", "V2_reverse", "V3_delete1",
              "V4_substitute1", "V5_in_sentence"]:
        g = s[s.variant == v]
        if not len(g):
            continue
        r = g.iloc[0]
        bg = CY if v == "V0_intact" else "transparent"
        left += (f"<tr class='sep'><td class='lbl' style='width:112px'>{name[v]}</td>"
                 f"<td class='bn' style='font-size:8.8pt;background:{bg}'>{r.text}</td>"
                 f"<td class='num' style='width:34px'>{float(r.p_gold):.2f}</td></tr>")

    gold = str(d.loc[rid, "figurative_meaning_bn"])
    w = gold.split()
    rng = random.Random(7)
    sc = w[:]
    rng.shuffle(sc)
    forms = [("full gloss", gold, CY),
             ("truncated to 6 words", " ".join(w[:6]), "transparent"),
             ("words scrambled", " ".join(sc), "transparent"),
             ("English gloss", str(d.loc[rid, "figurative_meaning_en"]), "transparent"),
             ("semantics removed", "A", GREY)]
    right = ""
    for nm, tx, bg in forms:
        cls = "bn" if nm not in ("English gloss", "semantics removed") else ""
        right += (f"<tr class='sep'><td class='lbl' style='width:118px'>{nm}</td>"
                  f"<td class='{cls}' style='font-size:8.6pt;background:{bg}'>{tx}</td></tr>")

    html = f"""
    <div class='hd'>what each manipulation does to real Bangla text &nbsp;
      <span class='lbl'>idiom <span class='bn'>{d.loc[rid, 'idiom']}</span>,
      English equivalent <i>{d.loc[rid, 'sim_en']}</i></span></div>
    <table style='margin-top:4px'>
      <tr><th style='width:112px'>E3, the state changes</th>
          <th>string sent as the state</th><th style='width:34px'>p(gold)</th></tr>
      {left}
    </table>
    <div style='height:7px'></div>
    <table>
      <tr><th style='width:118px'>E6, the options change</th>
          <th>the gold option, rewritten; the other three are rewritten the same way</th></tr>
      {right}
    </table>"""
    render(html, OUT / "fig_bn_manipulations.pdf", 6.5, 3.28)


if __name__ == "__main__":
    dd = load()
    build = builder(dd)
    fig_item(dd, build)
    fig_examples(dd)
    fig_manipulations(dd)
    print("done ->", OUT)
