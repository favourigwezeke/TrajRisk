from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("TRAJRISK_ROOT", REPO_ROOT)).expanduser().resolve()

import numpy as np
import pandas as pd
from frozen_inputs import read_view, view_path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ============================================================
# PATHS
# ============================================================

DATA = INPUT_ROOT / "Data" / "processed"
OUT = REPO_ROOT / "build" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "A": {
        "title": "Ben-Moshe hepatocytes",
        "file": view_path(DATA / "benmoshe_hepatocytes.h5ad", "benmoshe_qc"),
        "group": "time_point",
    },
    "B": {
        "title": "Umbaugh 2 h hepatocytes",
        "file": view_path(DATA / "umbaugh_2h_filtered.h5ad", "umbaugh_2h_qc"),
        "group": "treatment",
    },
    "C": {
        "title": "Umbaugh 48/72 h hepatocytes",
        "file": view_path(DATA / "umbaugh_48h72h_hepatocytes.h5ad", "umbaugh_late_qc"),
        "group": "timepoint_treatment",
    },
    "D": {
        "title": "Matchett human hepatocyte nuclei",
        "file": view_path(DATA / "matchett_human_hepatocytes.h5ad", "matchett_human_qc"),
        "group": "condition",
    },
}

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "Control": "#4C78A8",
    "Healthy": "#4C78A8",
    "APAP": "#E45756",
    "APAP_ALF": "#E45756",
    "APAP-ALF": "#E45756",
    "0": "#4C78A8",
    "48": "#F2A541",
    "72": "#59A14F",
}

HIST_COLOR = "#7A6FAC"
MEDIAN_COLOR = "#D62728"


def comma(x, pos=None):
    return f"{x:,.0f}"


# ============================================================
# HELPERS
# ============================================================

def ensure_qc_columns(adata, name):
    """
    Final saved objects should already contain these columns.
    Fail loudly rather than silently recomputing differently.
    """
    required = ["n_genes_by_counts", "total_counts"]

    missing = [
        x for x in required
        if x not in adata.obs.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing QC columns {missing}. "
            "Do not silently recompute; inspect the saved object."
        )


def composition_plot(ax, adata, mode):
    obs = adata.obs.copy()

    # --------------------------------------------------------
    # Ben-Moshe
    # --------------------------------------------------------
    if mode == "time_point":
        values = pd.to_numeric(
            obs["time_point"],
            errors="coerce"
        )

        counts = (
            values
            .value_counts()
            .sort_index()
        )

        x = np.arange(len(counts))

        ax.bar(
            x,
            counts.values,
            width=0.65,
            color="#4C78A8"
        )

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"{int(v)} h"
            for v in counts.index
        ])

        title = "Cells per timepoint"

    # --------------------------------------------------------
    # Umbaugh 2 h
    # --------------------------------------------------------
    elif mode == "treatment":
        order = [
            x for x in ["Control", "APAP"]
            if x in obs["treatment"].unique()
        ]

        counts = (
            obs["treatment"]
            .value_counts()
            .reindex(order)
        )

        x = np.arange(len(order))

        ax.bar(
            x,
            counts.values,
            width=0.65,
            color=[
                COLORS[x]
                for x in order
            ]
        )

        ax.set_xticks(x)
        ax.set_xticklabels(order)

        title = "Cells per treatment"

    # --------------------------------------------------------
    # Matchett human
    # --------------------------------------------------------
    elif mode == "condition":
        raw = obs["condition"].astype(str)

        # Display label only; underlying metadata unchanged.
        pretty = raw.replace({
            "APAP_ALF": "APAP-ALF"
        })

        order = [
            x for x in ["Healthy", "APAP-ALF"]
            if x in pretty.unique()
        ]

        counts = (
            pretty
            .value_counts()
            .reindex(order)
        )

        x = np.arange(len(order))

        ax.bar(
            x,
            counts.values,
            width=0.65,
            color=[
                COLORS[x]
                for x in order
            ]
        )

        ax.set_xticks(x)
        ax.set_xticklabels(order)

        title = "Nuclei per condition"

    # --------------------------------------------------------
    # Umbaugh 48/72 h
    # --------------------------------------------------------
    elif mode == "timepoint_treatment":
        table = (
            obs
            .groupby(
                ["timepoint", "treatment"],
                observed=True
            )
            .size()
            .unstack(fill_value=0)
        )

        time_order = [
            x for x in ["48h", "72h"]
            if x in table.index
        ]

        treatment_order = [
            x for x in ["Control", "APAP"]
            if x in table.columns
        ]

        table = table.reindex(
            index=time_order,
            columns=treatment_order
        )

        x = np.arange(len(table.index))
        width = 0.34

        for j, treatment in enumerate(
            treatment_order
        ):
            offset = (
                j -
                (len(treatment_order) - 1) / 2
            ) * width

            ax.bar(
                x + offset,
                table[treatment].values,
                width=width,
                label=treatment,
                color=COLORS[treatment]
            )

        ax.set_xticks(x)

        ax.set_xticklabels([
            x.replace("h", " h")
            for x in table.index
        ])

        ax.legend(
            frameon=False,
            loc="upper left",
            fontsize=6.5
        )

        title = "Cells per timepoint and treatment"

    else:
        raise ValueError(mode)

    # --------------------------------------------------------
    # Count labels
    # --------------------------------------------------------
    for patch in ax.patches:
        h = patch.get_height()

        if h > 0:
            ax.text(
                patch.get_x()
                + patch.get_width() / 2,
                h,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=6.5
            )

    ax.set_title(title)
    ax.set_ylabel("Cells")

    ax.yaxis.set_major_formatter(
        FuncFormatter(comma)
    )

    ax.margins(y=0.15)


def histogram_plot(ax, values, metric):
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna().values

    # Median is calculated from the FULL dataset.
    median = np.median(values)

    # --------------------------------------------------------
    # DISPLAY RANGE
    #
    # UMI distributions contain extreme high-count tails.
    # For visualization only, truncate at the 99.5th
    # percentile. No observations are removed from analysis,
    # and the displayed median still uses all observations.
    # --------------------------------------------------------
    if metric == "umi":
        display_max = np.quantile(
            values,
            0.995
        )

        plot_values = values[
            values <= display_max
        ]

    else:
        display_max = None
        plot_values = values

    ax.hist(
        plot_values,
        bins=80,
        color=HIST_COLOR,
        alpha=0.85,
        edgecolor="none"
    )

    ax.axvline(
        median,
        linestyle="--",
        linewidth=1.5,
        color=MEDIAN_COLOR
    )

    if metric == "genes":
        ax.set_title("Genes detected")
        ax.set_xlabel(
            "Genes per cell/nucleus"
        )

    elif metric == "umi":
        ax.set_title("UMI counts")
        ax.set_xlabel(
            "UMIs per cell/nucleus"
        )

        ax.set_xlim(
            0,
            display_max
        )

    else:
        raise ValueError(metric)

    ax.set_ylabel("Cells/nuclei")

    ax.text(
        0.97,
        0.94,
        f"Median = {median:,.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7
    )


def scatter_plot(ax, adata):
    x = pd.to_numeric(
        adata.obs["total_counts"],
        errors="coerce"
    ).values

    y = pd.to_numeric(
        adata.obs["n_genes_by_counts"],
        errors="coerce"
    ).values

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    ax.scatter(
        x,
        y,
        s=2,
        alpha=0.20,
        linewidths=0,
        color="#4C78A8",
        rasterized=True
    )

    ax.set_title(
        "Genes detected vs UMI counts"
    )

    ax.set_xlabel(
        "UMIs per cell/nucleus"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.set_ylabel(
        "Genes per cell/nucleus"
    )


def plot_dataset(
    subfig,
    letter,
    cfg
):
    adata = read_view(
        cfg["file"]
    )

    ensure_qc_columns(
        adata,
        cfg["title"]
    )

    subfig.suptitle(
        (
            f"{letter}   "
            f"{cfg['title']}   "
            f"(n = {adata.n_obs:,})"
        ),
        x=0.02,
        y=0.995,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    axes = subfig.subplots(
        2,
        2
    )

    composition_plot(
        axes[0, 0],
        adata,
        cfg["group"]
    )

    histogram_plot(
        axes[0, 1],
        adata.obs[
            "n_genes_by_counts"
        ],
        "genes"
    )

    histogram_plot(
        axes[1, 0],
        adata.obs[
            "total_counts"
        ],
        "umi"
    )

    scatter_plot(
        axes[1, 1],
        adata
    )

    subfig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.10,
        top=0.88,
        wspace=0.36,
        hspace=0.42
    )


# ============================================================
# LOAD CHECK
# ============================================================

print(
    "===== FIGURE S1 INPUT AUDIT ====="
)

for letter, cfg in FILES.items():

    if not cfg["file"].exists():
        raise FileNotFoundError(
            cfg["file"]
        )

    adata = read_view(
        cfg["file"],
        backed="r"
    )

    print(
        f"{letter}: "
        f"{cfg['title']:<36} "
        f"{adata.n_obs:>7,} cells/nuclei | "
        f"{adata.n_vars:>6,} genes"
    )

    del adata


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(15.5, 11.5),
    facecolor="white"
)

subfigs = fig.subfigures(
    2,
    2,
    wspace=0.04,
    hspace=0.08
)

for subfig, (
    letter,
    cfg
) in zip(
    subfigs.flat,
    FILES.items()
):
    plot_dataset(
        subfig,
        letter,
        cfg
    )

# No global title:
# manuscript caption carries figure title.

png = (
    OUT
    / "Figure_S1_external_dataset_QC_FINAL.png"
)

pdf = (
    OUT
    / "Figure_S1_external_dataset_QC_FINAL.pdf"
)

fig.savefig(
    png,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    pdf,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("\nCreated:")
print(" ", png)
print(" ", pdf)
