#!/usr/bin/env python3
"""Earlier TrajRisk analysis retained for source comparison.

See docs/workflow.md for the study workflow and docs/reproduction.md for
the commands used to reproduce the reported analyses.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path



# PATHS

BASE = Path(__file__).resolve().parents[3]
FIGS = BASE / "results" / "manuscript_figures"
FIGS.mkdir(parents=True, exist_ok=True)



# STYLE

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
})



# CONFIRMED CELL COUNTS

TIMEPOINTS = [
    "0 h",
    "2 h",
    "6 h",
    "24 h",
    "48 h",
    "72 h",
    "96 h",
    "APAP-ALF\n(n=10)",
    "Healthy\n(n=9)",
]

DATA = {
    "Walesky et al.\n(GSE136679)": {
        "role": "training",
        "cells": {
            "0 h": 3401,
            "2 h": None,
            "6 h": 419,
            "24 h": 1043,
            "48 h": 910,
            "72 h": None,
            "96 h": 809,
            "APAP-ALF\n(n=10)": None,
            "Healthy\n(n=9)": None,
        },
        "total": 6582,
        "platform": "Seq-Well",
    },
    "Ben-Moshe et al.\n(Zenodo 6035873)": {
        "role": "mouse_val",
        "cells": {
            "0 h": 1999,
            "2 h": None,
            "6 h": None,
            "24 h": None,
            "48 h": 532,
            "72 h": 239,
            "96 h": None,
            "APAP-ALF\n(n=10)": None,
            "Healthy\n(n=9)": None,
        },
        "total": 2770,
        "platform": "10x Chromium",
    },
    "Umbaugh et al.\n(GSE200771)": {
        "role": "mouse_val",
        "cells": {
            "0 h": 7907,
            "2 h": 13502,
            "6 h": None,
            "24 h": None,
            "48 h": None,
            "72 h": None,
            "96 h": None,
            "APAP-ALF\n(n=10)": None,
            "Healthy\n(n=9)": None,
        },
        "total": 21409,
        "platform": "10x Chromium",
    },
    "Umbaugh et al.\n(GSE255834)": {
        "role": "mouse_val",
        "cells": {
            "0 h": None,
            "2 h": None,
            "6 h": None,
            "24 h": None,
            "48 h": 2777,
            "72 h": 2529,
            "96 h": None,
            "APAP-ALF\n(n=10)": None,
            "Healthy\n(n=9)": None,
        },
        "total": 5306,
        "platform": "10x Chromium",
    },
    "Matchett et al.\n(GSE223581)": {
        "role": "human_val",
        "cells": {
            "0 h": None,
            "2 h": None,
            "6 h": None,
            "24 h": None,
            "48 h": None,
            "72 h": None,
            "96 h": None,
            "APAP-ALF\n(n=10)": 14247,
            "Healthy\n(n=9)": 35737,
        },
        "total": 49984,
        "platform": "10x Chromium\n(snRNA-seq)",
    },
}

DATASET_ORDER = [
    "Walesky et al.\n(GSE136679)",
    "Ben-Moshe et al.\n(Zenodo 6035873)",
    "Umbaugh et al.\n(GSE200771)",
    "Umbaugh et al.\n(GSE255834)",
    "Matchett et al.\n(GSE223581)",
]



# COLOURS

ROLE_COL = {
    "training": "#D9A514",
    "mouse_val": "#2FB36A",
    "human_val": "#9752B8",
}

ROLE_LABEL = {
    "training": "Training",
    "mouse_val": "Mouse validation",
    "human_val": "Human validation",
}



# HELPERS

def bubble_size(n_cells: int, maximum: int) -> float:
    """
    Compress the wide cell-count range while preserving visible differences.

    Matplotlib scatter's s parameter represents marker area. A square-root
    transformation prevents the largest human sample from overwhelming the
    smaller mouse samples.
    """
    minimum_area = 42.0
    additional_area = 360.0
    scaled = np.sqrt(n_cells / maximum)
    return minimum_area + additional_area * scaled


def format_cell_count(n_cells: int) -> str:
    """Compact label used inside bubbles."""
    if n_cells >= 10_000:
        return f"{n_cells / 1000:.1f}k"
    if n_cells >= 1_000:
        return f"{n_cells / 1000:.1f}k"
    return f"{n_cells:,}"



# BUILD FIGURE

fig, ax = plt.subplots(figsize=(7.2, 3.15))

fig.subplots_adjust(
    left=0.245,
    right=0.965,
    top=0.82,
    bottom=0.21,
)

n_datasets = len(DATASET_ORDER)
n_timepoints = len(TIMEPOINTS)

largest_observation = max(
    n
    for dataset in DATA.values()
    for n in dataset["cells"].values()
    if n is not None
)

# Metadata-column positions.
total_x = 9.30
platform_x = 10.45

# Limits include the aligned metadata columns but avoid uncontrolled overflow.
ax.set_xlim(-0.70, 11.30)
ax.set_ylim(-0.62, n_datasets - 0.38)


# -----------------------------------------------------------------------------
# Empty matrix positions
# -----------------------------------------------------------------------------
for x_index in range(n_timepoints):
    for y_index in range(n_datasets):
        ax.scatter(
            x_index,
            y_index,
            s=112,
            color="#F2F2F2",
            edgecolors="none",
            zorder=0,
        )


# -----------------------------------------------------------------------------
# Mouse–human separator
# -----------------------------------------------------------------------------
separator_x = 6.5

ax.axvline(
    separator_x,
    color="#C9C9C9",
    linewidth=0.75,
    linestyle=(0, (3, 2)),
    zorder=1,
)

header_y = n_datasets - 0.18

ax.text(
    3.0,
    header_y,
    "Mouse timepoints",
    ha="center",
    va="bottom",
    fontsize=6,
    color="#6F6F6F",
    fontstyle="italic",
    clip_on=False,
)

ax.text(
    7.5,
    header_y,
    "Human condition",
    ha="center",
    va="bottom",
    fontsize=6,
    color="#6F6F6F",
    fontstyle="italic",
    clip_on=False,
)


# -----------------------------------------------------------------------------
# Observed bubbles and cell-count labels
# -----------------------------------------------------------------------------
for y_index, dataset_name in enumerate(DATASET_ORDER):
    dataset = DATA[dataset_name]
    colour = ROLE_COL[dataset["role"]]

    for x_index, timepoint in enumerate(TIMEPOINTS):
        n_cells = dataset["cells"].get(timepoint)

        if n_cells is None or n_cells <= 0:
            continue

        size = bubble_size(n_cells, largest_observation)

        ax.scatter(
            x_index,
            y_index,
            s=size,
            color=colour,
            alpha=0.90,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )

        ax.text(
            x_index,
            y_index,
            format_cell_count(n_cells),
            ha="center",
            va="center",
            fontsize=4.3 if n_cells < 1000 else 4.5,
            color="white",
            fontweight="bold",
            zorder=4,
        )


# -----------------------------------------------------------------------------
# Exact totals and platform metadata
# -----------------------------------------------------------------------------
ax.text(
    total_x,
    header_y,
    "Total cells",
    ha="left",
    va="bottom",
    fontsize=5.8,
    color="#555555",
    fontweight="bold",
    clip_on=False,
)

ax.text(
    platform_x,
    header_y,
    "Platform",
    ha="left",
    va="bottom",
    fontsize=5.8,
    color="#555555",
    fontweight="bold",
    clip_on=False,
)

for y_index, dataset_name in enumerate(DATASET_ORDER):
    dataset = DATA[dataset_name]

    ax.text(
        total_x,
        y_index,
        f"{dataset['total']:,}",
        ha="left",
        va="center",
        fontsize=5.3,
        color=ROLE_COL[dataset["role"]],
        fontweight="bold",
    )

    ax.text(
        platform_x,
        y_index,
        dataset["platform"],
        ha="left",
        va="center",
        fontsize=4.7,
        color="#777777",
        linespacing=0.95,
    )


# -----------------------------------------------------------------------------
# Axes
# -----------------------------------------------------------------------------
ax.set_xticks(range(n_timepoints))
ax.set_xticklabels(TIMEPOINTS, fontsize=5.8)

ax.set_yticks(range(n_datasets))
ax.set_yticklabels(DATASET_ORDER, fontsize=5.8)

ax.set_xlabel("Timepoint / condition", labelpad=5)

ax.tick_params(axis="y", pad=5)


# -----------------------------------------------------------------------------
# Role legend
# -----------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(
        facecolor=ROLE_COL["training"],
        edgecolor="none",
        label=ROLE_LABEL["training"],
    ),
    mpatches.Patch(
        facecolor=ROLE_COL["mouse_val"],
        edgecolor="none",
        label=ROLE_LABEL["mouse_val"],
    ),
    mpatches.Patch(
        facecolor=ROLE_COL["human_val"],
        edgecolor="none",
        label=ROLE_LABEL["human_val"],
    ),
]

ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.50, 1.115),
    ncol=3,
    fontsize=5.8,
    handlelength=0.9,
    handleheight=0.9,
    handletextpad=0.35,
    columnspacing=1.1,
    borderaxespad=0,
)


# -----------------------------------------------------------------------------
# Panel label
# -----------------------------------------------------------------------------
ax.text(
    -0.285,
    1.12,
    "B",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
    ha="left",
    va="top",
)



# EXPORT

output_stem = FIGS / "figure_1b_dataset_summary"

for extension in [".pdf", ".svg", ".png"]:
    fig.savefig(output_stem.with_suffix(extension))

plt.close(fig)

print(
    "Saved: figure_1b_dataset_summary.pdf / "
    ".svg / .png"
)
