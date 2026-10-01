"""BNFDM figure style.

Palette is used exactly as specified, undarkened:
    #27ebf5 cyan    Laya-ml
    #f127f5 magenta Lod-lille
    #f5c827 yellow  Jev-1.13
    black           reference lines
    greys           neutral / control categories

Figures carry no legends, no value labels, no in-axes annotation. Axis labels
and tick labels only. Every mark is documented in analysis/figures/CAPTIONS.md
so captions can be written from that file.

Bright fills get a thin black edge, which is what keeps them legible on white
without changing the colour.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CYAN, MAG, YEL = "#27EBF5", "#F127F5", "#F5C827"
K = "#000000"
G1, G2, G3 = "#4D4D4D", "#999999", "#CCCCCC"

ARM_COLOR = {"laya-ml": CYAN, "lod-lille-0.6b": MAG, "jev-1.13": YEL}
ARM_LABEL = {"laya-ml": "Laya-ml", "lod-lille-0.6b": "Lod-lille", "jev-1.13": "Jev-1.13"}
ARMS3 = ["laya-ml", "lod-lille-0.6b", "jev-1.13"]
ARMS2 = ["laya-ml", "lod-lille-0.6b"]

CMAP_SEQ = LinearSegmentedColormap.from_list("bnfdm_seq", ["#FFFFFF", YEL, MAG, K])

W1, W2 = 3.15, 6.50          # ACL single / double column, inches
EDGE = dict(edgecolor=K, linewidth=0.5)


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": K, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.45,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "lines.markersize": 3.0,
    })


use_style()


def chance(ax, y=0.25):
    ax.axhline(y, color=K, ls=(0, (4, 2)), lw=0.8, zorder=1)


def eb(ax, x, d, col, lo, hi):
    ax.errorbar(x, d[col], yerr=np.vstack([d[col] - d[lo], d[hi] - d[col]]),
                fmt="none", ecolor=K, elinewidth=0.7, capsize=1.6, zorder=3)


def save(fig, path_pdf, data=None, png=True):
    p = Path(path_pdf)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf")
    if png:
        fig.savefig(p.with_suffix(".png"), format="png", dpi=320)
    if data is not None:
        d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        d.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8")
    plt.close(fig)
    print(f"  [fig] {p.name}")
    return p


# ── statistics ─────────────────────────────────────────────────────────────
def boot_ci(x, stat=np.mean, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(x), size=(n, len(x)))
    bs = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(bs, 100 * alpha / 2)), \
        float(np.percentile(bs, 100 * (1 - alpha / 2)))


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    ok = ~np.isnan(conf)
    conf, correct = conf[ok], correct[ok]
    if len(conf) == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def aurc(score, correct):
    score, correct = np.asarray(score, float), np.asarray(correct, float)
    o = np.argsort(-score)
    y = correct[o]
    return float((1 - np.cumsum(y) / np.arange(1, len(y) + 1)).mean())
