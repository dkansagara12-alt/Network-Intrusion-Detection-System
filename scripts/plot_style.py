"""
Shared matplotlib style for the EDA and results figures.

Colours are the validated default data-viz palette (categorical slots 1-4
pass the colour-vision-deficiency checks as adjacent bars; slots 3-4 are
below 3:1 contrast on the surface, so every bar chart carries direct value
labels and the webpage shows the numbers in tables too).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#c3c2b7"       # de-emphasised marks (background points, reference bars)

# categorical slots, fixed order: model identity is always the same colour
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {
    "FT-Transformer": SERIES[0],
    "Logistic Regression": SERIES[1],
    "Random Forest": SERIES[2],
    "XGBoost": SERIES[3],
}

# sequential (magnitude) and diverging (polarity) ramps
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#a32a2a"])


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "system-ui", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")
