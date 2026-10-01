"""BNFDM figure style: CMYK palette, white background, ACL-ready PDF output.

No titles on any figure (captions carry them in the paper). Every `save()` writes
a PDF, a PNG preview, and a sibling CSV holding exactly the data that was plotted.

Note on Bangla: matplotlib cannot shape Bengali (no HarfBuzz), and this machine has
no Bengali font installed, so NO figure here contains Bangla glyphs. Every figure is
quantitative; qualitative examples live in FINDINGS.md, which renders Bangla natively.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── palette: cyan / magenta / yellow / black, shades tuned for white paper ──
C = "#0F7F8C"   # cyan, deepened so it reads on white
M = "#B01C7E"   # magenta
Y = "#C8971B"   # yellow, darkened for legibility
K = "#1B1B1B"   # black
C_L, M_L, Y_L, K_L = "#7FBFC6", "#D98EBF", "#E4CB8D", "#9A9A9A"
GRID = "#DCDCDC"

PALETTE = [C, M, Y, K]
PALETTE_WIDE = [C, M, Y, K, C_L, M_L, Y_L, K_L]

CMAP_SEQ = LinearSegmentedColormap.from_list("cmyk_seq", ["#FFFFFF", Y, M, K])
CMAP_DIV = LinearSegmentedColormap.from_list("cmyk_div", [C, "#FFFFFF", M])

# two-column ACL page widths, inches
W1, W2 = 3.15, 6.50


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": K, "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.alpha": 1.0, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE_WIDE),
    })


use_style()


def save(fig, path_pdf, data=None, png=True):
    """PDF + PNG preview + sibling CSV of the plotted data."""
    p = Path(path_pdf)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf")
    if png:
        fig.savefig(p.with_suffix(".png"), format="png", dpi=300)
    if data is not None:
        d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        d.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8")
    plt.close(fig)
    print(f"  [fig] {p.name}")
    return p


# ── statistics helpers ─────────────────────────────────────────────────────
def boot_ci(x, stat=np.mean, n=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a 1-D sample."""
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
    """Area under the risk-coverage curve; lower is a better selective signal."""
    score, correct = np.asarray(score, float), np.asarray(correct, float)
    o = np.argsort(-score)
    y = correct[o]
    return float((1 - np.cumsum(y) / np.arange(1, len(y) + 1)).mean())


def fmt_ci(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
