#!/usr/bin/env python3
"""Earlier TrajRisk analysis retained for source comparison.

See docs/workflow.md for the study workflow and docs/reproduction.md for
the commands used to reproduce the reported analyses.
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# PATHS


BASE = Path(__file__).resolve().parents[2]
RES = BASE / "results"
FIGS = RES / "manuscript_figures"
FIGS.mkdir(parents=True, exist_ok=True)



# STYLE


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
})


SCALE_COL = {
    "Scale1_AcuteStress": "#C0392B",
    "Scale2_PeakInjury": "#E67E22",
    "Scale3_FateDecision": "#27AE60",
    "Scale4_Resolution": "#2980B9",
}

SCALE_ORDER = [
    "Scale1_AcuteStress",
    "Scale2_PeakInjury",
    "Scale3_FateDecision",
    "Scale4_Resolution",
]

SCALE_TICK = [
    "S1\n0–6 h",
    "S2\n6–24 h",
    "S3\n24–48 h",
    "S4\n48–96 h",
]

SCALE_LEGEND_LABEL = {
    "Scale1_AcuteStress": "S1: 0–6 h",
    "Scale2_PeakInjury": "S2: 6–24 h",
    "Scale3_FateDecision": "S3: 24–48 h",
    "Scale4_Resolution": "S4: 48–96 h",
}


def label_panel(
    ax: plt.Axes,
    letter: str,
    x: float = -0.15,
    y: float = 1.08,
) -> None:
    """Add a bold manuscript panel label."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save a figure as PDF, SVG, and PNG."""
    for extension in (".pdf", ".svg", ".png"):
        output_path = FIGS / f"{stem}{extension}"
        fig.savefig(output_path)
    print(f"  Saved: {stem}.pdf / .svg / .png")


def require_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    dataframe_name: str,
) -> None:
    """Fail early when a required plotting column is absent."""
    missing = [column for column in columns if column not in dataframe.columns]

    if missing:
        raise KeyError(
            f"{dataframe_name} is missing required columns: {missing}"
        )



# LOAD DATA


print("Loading data ...")

bm = pd.read_csv(
    RES / "week10_spatial_visium_benmoshe.csv"
)
mmv = pd.read_csv(
    RES / "week10_spatial_matchett_mouse_visium.csv"
)
mhv = pd.read_csv(
    RES / "week10_spatial_matchett_human_visium.csv"
)

with open(
    RES / "week5_transition_summary.json",
    encoding="utf-8",
) as file_handle:
    w5 = json.load(file_handle)


require_columns(
    bm,
    [
        "Fibrogenic_spot",
        "visium_coor_x",
        "visium_coor_y",
        "sig_net",
        "sig_risk",
        "Cyp2e1",
        "Glul",
    ],
    "Ben-Moshe spatial data",
)

require_columns(
    mmv,
    ["zone", "timepoint", "sig_net"],
    "Matchett mouse Visium data",
)

require_columns(
    mhv,
    ["zone", "condition", "sig_net"],
    "Matchett human Visium data",
)

print(
    f"  Ben-Moshe: {len(bm):,} spots | "
    f"Mouse Visium: {len(mmv):,} spots | "
    f"Human Visium: {len(mhv):,} spots"
)



# FIGURE 7 — SPATIAL VALIDATION


print("Building Figure 7 ...")

fig7 = plt.figure(figsize=(7.2, 6.55))

gs7 = gridspec.GridSpec(
    2,
    2,
    figure=fig7,
    left=0.09,
    right=0.965,
    top=0.925,
    bottom=0.085,
    hspace=0.60,
    wspace=0.46,
)

ax7A = fig7.add_subplot(gs7[0, 0])
ax7B = fig7.add_subplot(gs7[0, 1])
ax7C = fig7.add_subplot(gs7[1, 0])
ax7D = fig7.add_subplot(gs7[1, 1])


# -----------------------------------------------------------------------------
# Panel A — Ben-Moshe spatial map
# -----------------------------------------------------------------------------

fib_mask = bm["Fibrogenic_spot"].eq(1)
non_mask = bm["Fibrogenic_spot"].eq(0)

ax7A.scatter(
    bm.loc[non_mask, "visium_coor_x"],
    bm.loc[non_mask, "visium_coor_y"],
    c="#CCCCCC",
    s=0.7,
    linewidths=0,
    alpha=0.28,
    rasterized=True,
    zorder=1,
)

spatial_scatter = ax7A.scatter(
    bm.loc[fib_mask, "visium_coor_x"],
    bm.loc[fib_mask, "visium_coor_y"],
    c=bm.loc[fib_mask, "sig_net"],
    cmap="YlOrRd",
    s=2.2,
    linewidths=0,
    alpha=0.88,
    rasterized=True,
    zorder=2,
)

colorbar = fig7.colorbar(
    spatial_scatter,
    ax=ax7A,
    shrink=0.70,
    fraction=0.044,
    pad=0.025,
)

colorbar.set_label(
    "Net signature (CPM)",
    fontsize=6,
    labelpad=5,
)
colorbar.ax.tick_params(
    labelsize=5.5,
    width=0.6,
    length=2.5,
)

ax7A.set_xlabel("Spatial X")
ax7A.set_ylabel("Spatial Y")
ax7A.set_aspect("equal", adjustable="datalim")
ax7A.tick_params(labelsize=6)

legend_non = ax7A.scatter(
    [],
    [],
    c="#CCCCCC",
    s=10,
    label="Non-fibrogenic",
)
legend_fib = ax7A.scatter(
    [],
    [],
    c="#E67E22",
    s=10,
    label="Fibrogenic",
)

ax7A.legend(
    handles=[legend_non, legend_fib],
    fontsize=5.5,
    loc="lower left",
    bbox_to_anchor=(0.00, 1.02),
    ncol=2,
    handlelength=0.8,
    handletextpad=0.35,
    columnspacing=0.8,
    borderaxespad=0,
)

label_panel(ax7A, "A", x=-0.16, y=1.10)


# -----------------------------------------------------------------------------
# Panel B — Fibrogenic versus non-fibrogenic expression
# Descriptive means only. Inferential spot-level testing belongs in caption or
# supplement with the appropriate independence caveat.
# -----------------------------------------------------------------------------

expression_groups = [
    "sig_risk",
    "Cyp2e1",
    "Glul",
]

expression_labels = [
    "Risk signature",
    "Cyp2e1",
    "Glul",
]

fibrogenic_means = [
    bm.loc[fib_mask, group].mean()
    for group in expression_groups
]

nonfibrogenic_means = [
    bm.loc[non_mask, group].mean()
    for group in expression_groups
]

x_panel_b = np.arange(len(expression_groups))
bar_width_b = 0.36

ax7B.bar(
    x_panel_b - bar_width_b / 2,
    fibrogenic_means,
    width=bar_width_b,
    color="#E67E22",
    alpha=0.85,
    linewidth=0,
    label="Fibrogenic",
)

ax7B.bar(
    x_panel_b + bar_width_b / 2,
    nonfibrogenic_means,
    width=bar_width_b,
    color="#AAAAAA",
    alpha=0.85,
    linewidth=0,
    label="Non-fibrogenic",
)

ax7B.set_xticks(x_panel_b)
ax7B.set_xticklabels(
    expression_labels,
    fontsize=6.5,
)
ax7B.set_ylabel("Mean expression (CPM)")

ax7B.legend(
    fontsize=5.5,
    loc="lower left",
    bbox_to_anchor=(0.00, 1.02),
    ncol=2,
    handlelength=1.0,
    handletextpad=0.4,
    columnspacing=0.8,
    borderaxespad=0,
)

# Deliberately no spot-level p-value inside the panel.
label_panel(ax7B, "B", x=-0.16, y=1.10)


# -----------------------------------------------------------------------------
# Panel C — Matchett mouse spatial result
# -----------------------------------------------------------------------------

timepoint_order = [
    "healthy",
    "pod24h",
    "pod36h",
    "pod48h",
]

timepoint_labels = {
    "healthy": "Healthy",
    "pod24h": "24 h",
    "pod36h": "36 h",
    "pod48h": "48 h",
}

zone_colors = {
    "pericentral": "#D73027",
    "midlobular": "#FC8D59",
    "periportal": "#1A9850",
}

zone_order = [
    "pericentral",
    "midlobular",
    "periportal",
]

x_panel_c = np.arange(len(timepoint_order))
bar_width_c = 0.25

for zone_index, zone in enumerate(zone_order):
    means = [
        mmv.loc[
            mmv["zone"].eq(zone)
            & mmv["timepoint"].eq(timepoint),
            "sig_net",
        ].mean()
        for timepoint in timepoint_order
    ]

    ax7C.bar(
        x_panel_c + (zone_index - 1) * bar_width_c,
        means,
        width=bar_width_c,
        color=zone_colors[zone],
        alpha=0.80,
        linewidth=0,
        label=zone.capitalize(),
    )

ax7C.set_xticks(x_panel_c)
ax7C.set_xticklabels(
    [timepoint_labels[item] for item in timepoint_order],
    fontsize=6.5,
)
ax7C.set_ylabel("Net signature (CPM)")
ax7C.set_xlabel("Timepoint")

ax7C.legend(
    fontsize=5.5,
    loc="lower left",
    bbox_to_anchor=(0.00, 1.055),
    ncol=3,
    handlelength=0.9,
    handletextpad=0.35,
    columnspacing=0.7,
    borderaxespad=0,
)

ax7C.text(
    0.98,
    1.025,
    r"$p = 0.087$",
    transform=ax7C.transAxes,
    ha="right",
    va="bottom",
    fontsize=5.8,
    color="#555555",
    fontstyle="italic",
    clip_on=False,
)

label_panel(ax7C, "C", x=-0.16, y=1.10)


# -----------------------------------------------------------------------------
# Panel D — Matchett human spatial result
# Descriptive comparison only; independence caveat belongs in the caption.
# -----------------------------------------------------------------------------

condition_colors = {
    "APAP": "#D73027",
    "Healthy": "#4575B4",
}

x_panel_d = np.arange(len(zone_order))
bar_width_d = 0.38

for condition_index, condition in enumerate(["APAP", "Healthy"]):
    means = [
        mhv.loc[
            mhv["zone"].eq(zone)
            & mhv["condition"].eq(condition),
            "sig_net",
        ].mean()
        for zone in zone_order
    ]

    label = (
        "APAP-ALF"
        if condition == "APAP"
        else "Healthy"
    )

    ax7D.bar(
        x_panel_d + (condition_index - 0.5) * bar_width_d,
        means,
        width=bar_width_d,
        color=condition_colors[condition],
        alpha=0.82,
        linewidth=0,
        label=label,
    )

ax7D.set_xticks(x_panel_d)
ax7D.set_xticklabels(
    [zone.capitalize() for zone in zone_order],
    fontsize=6.5,
)
ax7D.set_ylabel("Net signature (CPM)")
ax7D.set_xlabel("Spatial zone")

ax7D.legend(
    fontsize=6,
    loc="lower left",
    bbox_to_anchor=(0.00, 1.02),
    ncol=2,
    handlelength=1.0,
    handletextpad=0.4,
    columnspacing=0.8,
    borderaxespad=0,
)

# Deliberately no "Descriptive" text inside the plotting region.
label_panel(ax7D, "D", x=-0.16, y=1.10)


save_fig(fig7, "figure_7_spatial")
plt.close(fig7)



# FIGURE 2B–C — RF PERFORMANCE AND GAUSSIAN COMPARISON


print("Building Figure 2B+C ...")

cv_means = [
    w5["scales"][scale]["cv_auroc_mean"]
    for scale in SCALE_ORDER
]

cv_stds = [
    w5["scales"][scale]["cv_auroc_std"]
    for scale in SCALE_ORDER
]

cv_folds = [
    w5["scales"][scale]["cv_auroc_folds"]
    for scale in SCALE_ORDER
]

oob_scores = [
    w5["scales"][scale]["oob_score"]
    for scale in SCALE_ORDER
]

gaussian_correlations = [
    w5["scales"][scale]["gaussian_comparison"][
        "mean_row_pearson_vs_gaussian"
    ]
    for scale in SCALE_ORDER
]


fig_2bc = plt.figure(figsize=(6.5, 2.65))

gs_2bc = gridspec.GridSpec(
    1,
    2,
    figure=fig_2bc,
    left=0.10,
    right=0.985,
    top=0.83,
    bottom=0.27,
    wspace=0.56,
)

ax2B = fig_2bc.add_subplot(gs_2bc[0, 0])
ax2C = fig_2bc.add_subplot(gs_2bc[0, 1])

x_scales = np.arange(len(SCALE_ORDER))


# -----------------------------------------------------------------------------
# Panel B — CV AUROC and OOB score
# -----------------------------------------------------------------------------

for scale_index, scale in enumerate(SCALE_ORDER):
    color = SCALE_COL[scale]

    ax2B.bar(
        scale_index,
        cv_means[scale_index],
        width=0.52,
        color=color,
        alpha=0.80,
        linewidth=0,
        zorder=1,
    )

    confidence_interval = (
        1.96
        * cv_stds[scale_index]
        / np.sqrt(len(cv_folds[scale_index]))
    )

    ax2B.errorbar(
        scale_index,
        cv_means[scale_index],
        yerr=confidence_interval,
        color="black",
        lw=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=4,
    )

    for fold_value in cv_folds[scale_index]:
        ax2B.scatter(
            scale_index,
            fold_value,
            color="white",
            edgecolors=color,
            s=10,
            zorder=5,
            linewidths=0.7,
        )

    ax2B.scatter(
        scale_index,
        oob_scores[scale_index],
        marker="D",
        color=color,
        s=16,
        zorder=6,
        edgecolors="black",
        linewidths=0.55,
    )

ax2B.set_xticks(x_scales)
ax2B.set_xticklabels(
    SCALE_TICK,
    fontsize=6.3,
)
ax2B.set_ylabel("Performance score")
ax2B.set_ylim(0.93, 1.002)

cv_handle = ax2B.scatter(
    [],
    [],
    marker="o",
    color="white",
    edgecolors="black",
    s=10,
    linewidths=0.7,
    label="CV fold AUROC",
)

oob_handle = ax2B.scatter(
    [],
    [],
    marker="D",
    color="grey",
    edgecolors="black",
    s=16,
    linewidths=0.55,
    label="OOB accuracy",
)

ax2B.legend(
    handles=[cv_handle, oob_handle],
    fontsize=5.5,
    loc="lower center",
    bbox_to_anchor=(0.52, 1.02),
    ncol=2,
    handlelength=0.9,
    handletextpad=0.4,
    columnspacing=0.9,
    borderaxespad=0,
)

label_panel(ax2B, "B", x=-0.16, y=1.16)


# -----------------------------------------------------------------------------
# Panel C — Mean Pearson correlation versus Gaussian transitions
# -----------------------------------------------------------------------------

for scale_index, scale in enumerate(SCALE_ORDER):
    color = SCALE_COL[scale]
    value = gaussian_correlations[scale_index]

    ax2C.bar(
        scale_index,
        value,
        width=0.52,
        color=color,
        alpha=0.80,
        linewidth=0,
    )

    ax2C.text(
        scale_index,
        value + 0.010,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=color,
    )

ax2C.set_xticks(x_scales)
ax2C.set_xticklabels(
    SCALE_TICK,
    fontsize=6.3,
)
ax2C.set_ylabel(
    "Mean Pearson $r$\n"
    "(learned vs. Gaussian)"
)
ax2C.set_ylim(0, 0.50)

label_panel(ax2C, "C", x=-0.16, y=1.16)


save_fig(
    fig_2bc,
    "figure_2bc_rf_performance",
)
plt.close(fig_2bc)



# SUPPLEMENTARY S14 — CROSS-SPECIES CONSERVATION


print("Building Supplementary Figure S14 ...")

s14_scale = pd.read_csv(
    RES / "week12_task1_scale_summary.csv"
)
s14_genes = pd.read_csv(
    RES / "week12_task2_conserved_genes.csv"
)

require_columns(
    s14_scale,
    [
        "scale",
        "n_human_tested",
        "n_sig_human",
        "n_concordant",
        "pct_conserved",
    ],
    "S14 scale summary",
)

require_columns(
    s14_genes,
    [
        "mouse_gene",
        "human_gene",
        "log2fc_human",
        "pval_human",
    ],
    "S14 conserved-gene data",
)


# -----------------------------------------------------------------------------
# Assign each conserved gene to a TrajRisk scale
# -----------------------------------------------------------------------------

scale_file_map = {
    "scale1_acutestress": "Scale1_AcuteStress",
    "scale2_peakinjury": "Scale2_PeakInjury",
    "scale3_fatedecision": "Scale3_FateDecision",
    "scale4_resolution": "Scale4_Resolution",
}

gene_to_scale: dict[str, str] = {}

for safe_scale_name, full_scale_name in scale_file_map.items():
    differential_expression_file = (
        RES / f"bottleneck_s3_de_{safe_scale_name}.csv"
    )

    if not differential_expression_file.exists():
        continue

    differential_expression = pd.read_csv(
        differential_expression_file
    )

    if "gene" not in differential_expression.columns:
        raise KeyError(
            f"{differential_expression_file.name} "
            "does not contain a 'gene' column."
        )

    for mouse_gene in differential_expression["gene"].dropna():
        gene_to_scale.setdefault(
            mouse_gene,
            full_scale_name,
        )

s14_genes["scale"] = s14_genes["mouse_gene"].map(
    gene_to_scale
)


# -----------------------------------------------------------------------------
# Select representative genes
# -----------------------------------------------------------------------------

priority_genes = {
    "Scale1_AcuteStress": [
        "HMOX1",
        "SRXN1",
    ],
    "Scale2_PeakInjury": [
        "HAL",
        "AACS",
    ],
    "Scale3_FateDecision": [
        "RPL7A",
        "SAMM50",
    ],
    "Scale4_Resolution": [
        "ABCA8",
        "AGMO",
    ],
}

selected_rows: list[dict] = []

for scale, genes in priority_genes.items():
    for human_gene in genes:
        matched_rows = s14_genes.loc[
            s14_genes["human_gene"].eq(human_gene)
        ]

        if not matched_rows.empty:
            selected_row = matched_rows.iloc[0].to_dict()
            selected_row["scale"] = scale
            selected_rows.append(selected_row)
            continue

        # Fallback only when a requested representative gene is absent.
        scale_subset = s14_genes.loc[
            s14_genes["scale"].eq(scale)
        ].copy()

        already_selected = {
            row["human_gene"]
            for row in selected_rows
            if row.get("scale") == scale
        }

        scale_subset = scale_subset.loc[
            ~scale_subset["human_gene"].isin(already_selected)
        ]

        if scale_subset.empty:
            continue

        scale_subset["absolute_log2fc"] = (
            scale_subset["log2fc_human"].abs()
        )

        fallback_row = (
            scale_subset
            .nlargest(1, "absolute_log2fc")
            .iloc[0]
            .to_dict()
        )

        fallback_row["scale"] = scale
        selected_rows.append(fallback_row)


plot_df = (
    pd.DataFrame(selected_rows)
    .drop_duplicates("human_gene")
    .copy()
)

if plot_df.empty:
    raise ValueError(
        "No genes were available for Supplementary Figure S14B."
    )

scale_rank = {
    scale: rank
    for rank, scale in enumerate(SCALE_ORDER)
}

plot_df["scale_rank"] = plot_df["scale"].map(
    scale_rank
)

plot_df = (
    plot_df
    .sort_values(
        ["scale_rank", "log2fc_human"],
        ascending=[True, True],
    )
    .reset_index(drop=True)
)

print(
    "  S14B genes:",
    plot_df["human_gene"].tolist(),
)


# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------

fig_s14 = plt.figure(figsize=(7.4, 4.65))

gs_s14 = gridspec.GridSpec(
    1,
    2,
    figure=fig_s14,
    left=0.08,
    right=0.97,
    top=0.865,
    bottom=0.13,
    wspace=0.48,
    width_ratios=[1.0, 1.32],
)

axS14A = fig_s14.add_subplot(gs_s14[0, 0])
axS14B = fig_s14.add_subplot(gs_s14[0, 1])


# -----------------------------------------------------------------------------
# Panel A — tested, significant, and directionally concordant genes
# -----------------------------------------------------------------------------

bar_width_s14 = 0.25
x_s14 = np.arange(len(SCALE_ORDER))

category_specification = [
    ("n_human_tested", 0.38),
    ("n_sig_human", 0.65),
    ("n_concordant", 0.92),
]

for category_index, (
    column_name,
    alpha_value,
) in enumerate(category_specification):

    values = [
        int(
            s14_scale.loc[
                s14_scale["scale"].eq(scale),
                column_name,
            ].iloc[0]
        )
        for scale in SCALE_ORDER
    ]

    axS14A.bar(
        x_s14
        + (category_index - 1) * bar_width_s14,
        values,
        width=bar_width_s14,
        color=[
            SCALE_COL[scale]
            for scale in SCALE_ORDER
        ],
        alpha=alpha_value,
        linewidth=0,
    )


for scale_index, scale in enumerate(SCALE_ORDER):
    scale_row = s14_scale.loc[
        s14_scale["scale"].eq(scale)
    ]

    n_significant = int(
        scale_row["n_sig_human"].iloc[0]
    )
    n_tested = int(
        scale_row["n_human_tested"].iloc[0]
    )
    percentage = float(
        scale_row["pct_conserved"].iloc[0]
    )

    axS14A.text(
        x_s14[scale_index],
        n_significant + max(15, n_tested * 0.012),
        f"{n_significant}/{n_tested}\n"
        f"({percentage:.0f}%)",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color=SCALE_COL[scale],
    )

axS14A.set_xticks(x_s14)
axS14A.set_xticklabels(
    SCALE_TICK,
    fontsize=6.5,
)
axS14A.set_ylabel("Number of genes")

category_handles = [
    mpatches.Patch(
        facecolor="#666666",
        alpha=0.38,
        label="Tested",
    ),
    mpatches.Patch(
        facecolor="#666666",
        alpha=0.65,
        label=r"Significant ($p \leq 0.05$)",
    ),
    mpatches.Patch(
        facecolor="#666666",
        alpha=0.92,
        label="Concordant direction",
    ),
]

axS14A.legend(
    handles=category_handles,
    fontsize=5.5,
    loc="lower left",
    bbox_to_anchor=(0.00, 1.02),
    ncol=3,
    handlelength=0.9,
    handletextpad=0.35,
    columnspacing=0.7,
    borderaxespad=0,
)

label_panel(
    axS14A,
    "A",
    x=-0.16,
    y=1.16,
)


# -----------------------------------------------------------------------------
# Panel B — human APAP-ALF versus healthy log2 fold change
# -----------------------------------------------------------------------------

y_positions = np.arange(len(plot_df))

log2_fold_changes = (
    plot_df["log2fc_human"]
    .astype(float)
    .to_numpy()
)

# Prefer adjusted p-values when the saved table contains them.
if "padj_human" in plot_df.columns:
    pvalue_column = "padj_human"
elif "adj_pval_human" in plot_df.columns:
    pvalue_column = "adj_pval_human"
else:
    pvalue_column = "pval_human"

plot_pvalues = (
    plot_df[pvalue_column]
    .astype(float)
    .clip(lower=1e-300)
    .to_numpy()
)

negative_log10_p = -np.log10(plot_pvalues)

dot_sizes = np.clip(
    20 + negative_log10_p * 9,
    24,
    100,
)

dot_colors = [
    SCALE_COL[scale]
    for scale in plot_df["scale"]
]


# Set limits before plotting annotations so no text relies on autoscale state.
minimum_lfc = float(np.nanmin(log2_fold_changes))
maximum_lfc = float(np.nanmax(log2_fold_changes))
lfc_range = maximum_lfc - minimum_lfc

if not np.isfinite(lfc_range) or lfc_range == 0:
    lfc_range = 1.0

left_limit = min(-0.5, minimum_lfc - 0.12 * lfc_range)
right_limit = max(0.5, maximum_lfc + 0.12 * lfc_range)

axS14B.set_xlim(
    left_limit,
    right_limit,
)

axS14B.axvline(
    0,
    color="#777777",
    lw=0.7,
    ls="--",
    zorder=1,
)

axS14B.scatter(
    log2_fold_changes,
    y_positions,
    s=dot_sizes,
    c=dot_colors,
    edgecolors="white",
    linewidths=0.45,
    alpha=0.92,
    zorder=3,
)

axS14B.set_yticks(y_positions)
axS14B.set_yticklabels(
    plot_df["human_gene"],
    fontsize=6.5,
)

# Show S1 at the top and S4 at the bottom.
axS14B.invert_yaxis()

axS14B.set_xlabel(
    "Human APAP-ALF vs. healthy\n"
    r"$\log_{2}$ fold change"
)

# Scale is encoded by dot colour. Do not place S1–S4 text over the data.
scale_handles = [
    mpatches.Patch(
        facecolor=SCALE_COL[scale],
        label=SCALE_LEGEND_LABEL[scale],
    )
    for scale in SCALE_ORDER
]

scale_legend = axS14B.legend(
    handles=scale_handles,
    fontsize=5.2,
    loc="lower left",
    bbox_to_anchor=(0.00, 1.02),
    ncol=4,
    handlelength=0.8,
    handletextpad=0.3,
    columnspacing=0.65,
    borderaxespad=0,
)

axS14B.add_artist(scale_legend)


# Significance-size legend outside the data field.
reference_pvalues = [
    0.05,
    0.001,
]

size_handles = []

for reference_pvalue in reference_pvalues:
    reference_size = np.clip(
        20
        + (-np.log10(reference_pvalue)) * 9,
        24,
        100,
    )

    label = (
        r"$p = 0.05$"
        if np.isclose(reference_pvalue, 0.05)
        else r"$p = 0.001$"
    )

    size_handles.append(
        axS14B.scatter(
            [],
            [],
            s=reference_size,
            color="#888888",
            alpha=0.75,
            edgecolors="none",
            label=label,
        )
    )

axS14B.legend(
    handles=size_handles,
    title=(
        r"$-\log_{10}(p)$"
        if pvalue_column == "pval_human"
        else r"$-\log_{10}(\mathrm{adjusted}\ p)$"
    ),
    title_fontsize=5.2,
    fontsize=5.2,
    loc="upper left",
    bbox_to_anchor=(1.01, 0.32),
    handletextpad=0.45,
    borderaxespad=0,
    scatterpoints=1,
)

label_panel(
    axS14B,
    "B",
    x=-0.14,
    y=1.16,
)


save_fig(
    fig_s14,
    "supp_s14_conservation",
)
plt.close(fig_s14)


print("\nAll scVI integration panels complete.")
print(f"Output directory: {FIGS}")
