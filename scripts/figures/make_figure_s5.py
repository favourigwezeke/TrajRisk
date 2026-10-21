from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("TRAJRISK_ROOT", REPO_ROOT)).expanduser().resolve()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# PATHS


RES = INPUT_ROOT / "results"
SUPP = REPO_ROOT / "manuscript"
TABLES = REPO_ROOT / "manuscript" / "tables" / "data"
OUT = REPO_ROOT / "build" / "figures" / "supplementary"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

S6D_FILE = (
    TABLES
    / "Table_S6D_scale3_cell_enrichment_FINAL.csv"
)

S6E_FILE = (
    TABLES
    / "Table_S6E_scale3_threshold_sensitivity_FINAL.csv"
)

S6F_FILE = (
    TABLES
    / "Table_S6F_scale3_quartile_perturbation_FINAL.csv"
)

RAW_QUARTILE_FILE = (
    RES
    / "bottleneck_s2_quartile_perturbation.csv"
)

PNG_OUT = (
    OUT
    / "Figure_S5_bottleneck_sensitivity_FINAL.png"
)

PDF_OUT = (
    OUT
    / "Figure_S5_bottleneck_sensitivity_FINAL.pdf"
)


# COLORS


THRESHOLD_COLOR = "#7A5195"
BACKGROUND_COLOR = "#7F7F7F"

MEAN_COLOR = "#4C78A8"
MAX_COLOR = "#D95F59"


# STYLE


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# HELPERS


def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            path
        )


def require_columns(
    df,
    cols,
    label,
):

    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{label} missing required columns: "
            f"{missing}"
        )


def assert_close(
    observed,
    expected,
    label,
    rtol=1e-9,
    atol=1e-12,
):

    if not np.isclose(
        float(observed),
        float(expected),
        rtol=rtol,
        atol=atol,
    ):

        raise ValueError(
            f"{label} mismatch: "
            f"observed={observed}, "
            f"expected={expected}"
        )


def add_panel_label(
    ax,
    label,
):

    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left",
    )



# LOAD


print(
    "===== FIGURE S5 INPUT VALIDATION ====="
)

for path in [
    S6D_FILE,
    S6E_FILE,
    S6F_FILE,
    RAW_QUARTILE_FILE,
]:
    require_file(
        path
    )

s6d = pd.read_csv(
    S6D_FILE
)

s6e = pd.read_csv(
    S6E_FILE
)

s6f = pd.read_csv(
    S6F_FILE
)

raw_q = pd.read_csv(
    RAW_QUARTILE_FILE
)

# The historical generator padded the displayed Q4 endpoint to 1.010.
# Normalize that display label in memory; all numeric fields remain unchanged.
q4_padding = raw_q["quartile"].eq("Q4_high_pt") & raw_q["pt_range"].eq("0.092–1.010")
raw_q.loc[q4_padding, "pt_range"] = "0.092–1.000"


# REQUIRED COLUMNS


require_columns(
    s6e,
    [
        "Flux_threshold_top_pct",
        "N_edges",
        "N_unique_source_cells",
        "N_high_pseudotime_sources",
        "High_pseudotime_sources_pct",
        "Background_high_pseudotime_pct",
        "Odds_ratio",
        "Fisher_p",
    ],
    "S6E",
)

require_columns(
    s6f,
    [
        "quartile",
        "pt_range",
        "edges_removed",
        "total_abs_delta",
        "mean_abs_delta_48h",
        "mean_abs_delta_96h",
        "max_abs_delta_48h",
        "max_abs_delta_96h",
    ],
    "S6F",
)


# VALIDATION 1 — THRESHOLD SENSITIVITY STRUCTURE


print(
    "\n===== S6E THRESHOLD-SENSITIVITY VALIDATION ====="
)

expected_thresholds = np.array(
    [
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    ]
)

observed_thresholds = (
    s6e[
        "Flux_threshold_top_pct"
    ]
    .astype(float)
    .values
)

if not np.array_equal(
    observed_thresholds,
    expected_thresholds,
):
    raise ValueError(
        "Unexpected S6E threshold ordering."
    )

background = (
    s6e[
        "Background_high_pseudotime_pct"
    ]
    .astype(float)
    .values
)

if not np.allclose(
    background,
    background[0],
    rtol=0,
    atol=1e-12,
):
    raise ValueError(
        "Background high-pseudotime prevalence "
        "is not constant across S6E thresholds."
    )

assert_close(
    background[0],
    25.023969,
    "S6E background prevalence",
    rtol=1e-7,
    atol=1e-6,
)

if not (
    s6e[
        "Odds_ratio"
    ]
    .astype(float)
    .gt(1.0)
    .all()
):
    raise ValueError(
        "At least one S6E odds ratio is not > 1."
    )

if not (
    s6e[
        "Fisher_p"
    ]
    .astype(float)
    .lt(0.05)
    .all()
):
    raise ValueError(
        "At least one S6E Fisher p-value is not < 0.05."
    )

print(
    "Threshold ordering: PASS"
)

print(
    "Background prevalence constant: PASS | "
    f"{background[0]:.6f}%"
)

print(
    "All threshold odds ratios > 1: PASS"
)

print(
    "All threshold Fisher p-values < 0.05: PASS"
)


# VALIDATION 2 — 1% THRESHOLD MUST MATCH reported S6D


print(
    "\n===== S6D ↔ S6E 1% CROSS-CHECK ====="
)

s6e_1 = s6e.loc[
    np.isclose(
        s6e[
            "Flux_threshold_top_pct"
        ].astype(float),
        1.0,
    )
]

if len(
    s6e_1
) != 1:
    raise ValueError(
        "Could not uniquely identify the 1% S6E row."
    )

s6e_1 = s6e_1.iloc[0]

s6d_source = s6d.loc[
    s6d[
        "Analysis"
    ]
    .astype(str)
    .eq(
        "Scale 3 source: high pseudotime"
    )
]

if len(
    s6d_source
) != 1:
    raise ValueError(
        "Could not uniquely identify the "
        "Scale 3 source high-pseudotime row in S6D."
    )

s6d_source = s6d_source.iloc[0]

if int(
    s6e_1[
        "N_unique_source_cells"
    ]
) != int(
    s6d_source[
        "N_bottleneck_cells"
    ]
):
    raise ValueError(
        "S6D/S6E 1% source-cell count mismatch."
    )

if int(
    s6e_1[
        "N_high_pseudotime_sources"
    ]
) != int(
    s6d_source[
        "N_bottleneck_high"
    ]
):
    raise ValueError(
        "S6D/S6E 1% high-pseudotime count mismatch."
    )

assert_close(
    s6e_1[
        "High_pseudotime_sources_pct"
    ],
    s6d_source[
        "Bottleneck_high_pct"
    ],
    "S6D/S6E 1% high-pseudotime percentage",
)

assert_close(
    s6e_1[
        "Background_high_pseudotime_pct"
    ],
    s6d_source[
        "Background_high_pct"
    ],
    "S6D/S6E 1% background percentage",
)

assert_close(
    s6e_1[
        "Odds_ratio"
    ],
    s6d_source[
        "Odds_ratio"
    ],
    "S6D/S6E 1% odds ratio",
)

assert_close(
    s6e_1[
        "Fisher_p"
    ],
    s6d_source[
        "Fisher_p"
    ],
    "S6D/S6E 1% Fisher p-value",
)

print(
    "1% threshold reproduces reported S6D: PASS"
)


# VALIDATION 3 — RAW QUARTILE FILE VS reported S6F


print(
    "\n===== RAW QUARTILE ↔ reported S6F VALIDATION ====="
)

if list(
    raw_q.columns
) != list(
    s6f.columns
):
    raise ValueError(
        "Raw quartile and reported S6F columns differ."
    )

if len(
    raw_q
) != len(
    s6f
):
    raise ValueError(
        "Raw quartile and reported S6F row counts differ."
    )

for col in s6f.columns:

    if pd.api.types.is_numeric_dtype(
        s6f[col]
    ):

        a = pd.to_numeric(
            raw_q[col],
            errors="coerce",
        ).values

        b = pd.to_numeric(
            s6f[col],
            errors="coerce",
        ).values

        if not np.allclose(
            a,
            b,
            rtol=0,
            atol=0,
            equal_nan=True,
        ):
            raise ValueError(
                f"Raw quartile differs from "
                f"reported S6F in numeric column: {col}"
            )

    else:

        if not np.array_equal(
            raw_q[col]
            .astype(str)
            .values,
            s6f[col]
            .astype(str)
            .values,
        ):
            raise ValueError(
                f"Raw quartile differs from "
                f"reported S6F in text column: {col}"
            )

print(
    "Raw quartile file exactly matches reported S6F: PASS"
)


# VALIDATION 4 — QUARTILE INTERPRETATION


print(
    "\n===== S6F PERTURBATION VALIDATION ====="
)

expected_quartiles = [
    "Q1_low_pt",
    "Q2_mid_lo_pt",
    "Q3_mid_hi_pt",
    "Q4_high_pt",
]

if (
    s6f[
        "quartile"
    ]
    .astype(str)
    .tolist()
    != expected_quartiles
):
    raise ValueError(
        "Unexpected S6F quartile ordering."
    )

mean_48 = (
    s6f[
        "mean_abs_delta_48h"
    ]
    .astype(float)
    .values
)

max_48 = (
    s6f[
        "max_abs_delta_48h"
    ]
    .astype(float)
    .values
)

mean_96 = (
    s6f[
        "mean_abs_delta_96h"
    ]
    .astype(float)
    .values
)

max_96 = (
    s6f[
        "max_abs_delta_96h"
    ]
    .astype(float)
    .values
)

if not np.all(
    mean_48 > 0
):
    raise ValueError(
        "Expected all 48 h mean perturbation "
        "effects to be positive."
    )

if not np.all(
    max_48 > 0
):
    raise ValueError(
        "Expected all 48 h maximum perturbation "
        "effects to be positive."
    )

if not np.allclose(
    mean_96,
    0.0,
    rtol=0,
    atol=0,
):
    raise ValueError(
        "Expected all 96 h mean effects to be zero."
    )

if not np.allclose(
    max_96,
    0.0,
    rtol=0,
    atol=0,
):
    raise ValueError(
        "Expected all 96 h maximum effects to be zero."
    )

q4_max = float(
    max_48[3]
)

assert_close(
    q4_max,
    0.097897,
    "Q4 maximum 48 h effect",
    rtol=1e-6,
    atol=1e-6,
)

if not (
    q4_max
    == np.max(
        max_48
    )
):
    raise ValueError(
        "Q4 is not the largest maximum 48 h effect."
    )

print(
    "48 h mean effects positive: PASS"
)

print(
    "48 h maximum effects positive: PASS"
)

print(
    "All 96 h mean effects = 0: PASS"
)

print(
    "All 96 h maximum effects = 0: PASS"
)

print(
    "Q4 has largest maximum 48 h effect: PASS | "
    f"{q4_max:.6f}"
)

print(
    "\nALL FIGURE S5 VALIDATIONS PASSED."
)


# FIGURE


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        10.8,
        4.4,
    ),
)

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.18,
    top=0.86,
    wspace=0.34,
)


# PANEL A — THRESHOLD SENSITIVITY


ax = axes[0]

x = np.arange(
    len(
        s6e
    )
)

threshold_pct = (
    s6e[
        "Flux_threshold_top_pct"
    ]
    .astype(float)
    .values
)

source_high_pct = (
    s6e[
        "High_pseudotime_sources_pct"
    ]
    .astype(float)
    .values
)

source_n = (
    s6e[
        "N_unique_source_cells"
    ]
    .astype(int)
    .values
)

ax.plot(
    x,
    source_high_pct,
    marker="o",
    markersize=6,
    linewidth=2.0,
    color=THRESHOLD_COLOR,
    label="Bottleneck source cells",
)

ax.axhline(
    background[0],
    linestyle="--",
    linewidth=1.4,
    color=BACKGROUND_COLOR,
    label=(
        "All 24 h cells "
        f"({background[0]:.1f}%)"
    ),
)

for xi, yi, n in zip(
    x,
    source_high_pct,
    source_n,
):

    ax.annotate(
        f"n={n}",
        xy=(
            xi,
            yi,
        ),
        xytext=(
            0,
            8,
        ),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        f"{v:g}%"
        for v in threshold_pct
    ]
)

ax.set_xlabel(
    "Top Scale 3 flux threshold"
)

ax.set_ylabel(
    "High-pseudotime source cells (%)"
)

ax.set_ylim(
    15,
    90,
)

ax.set_title(
    "Scale 3 bottleneck-threshold sensitivity"
)

ax.legend(
    frameon=False,
    loc="upper right",
)

ax.grid(
    axis="y",
    alpha=0.15,
    linewidth=0.5,
)

add_panel_label(
    ax,
    "A",
)


# PANEL B — QUARTILE PERTURBATION


ax = axes[1]

quartile_labels = [
    "Q1\n0–25%",
    "Q2\n25–50%",
    "Q3\n50–75%",
    "Q4\n75–100%",
]

xq = np.arange(
    4
)

offset = 0.09

# Means and maxima are deliberately shown as distinct
# marker series. No bars are used, avoiding ambiguity over
# whether bar height represents a mean or a maximum.
ax.scatter(
    xq - offset,
    mean_48,
    s=48,
    marker="o",
    color=MEAN_COLOR,
    label=r"Mean $|\Delta\mathrm{risk}|$",
    zorder=4,
)

ax.scatter(
    xq + offset,
    max_48,
    s=52,
    marker="D",
    color=MAX_COLOR,
    label=r"Maximum $|\Delta\mathrm{risk}|$",
    zorder=4,
)

# Thin vertical connectors emphasize the within-quartile
# separation between mean and maximum without implying that
# either series is continuous across quartiles.
for xi, mn, mx in zip(
    xq,
    mean_48,
    max_48,
):

    ax.plot(
        [
            xi,
            xi,
        ],
        [
            mn,
            mx,
        ],
        linewidth=0.7,
        alpha=0.35,
        color="#777777",
        zorder=1,
    )

ax.set_yscale(
    "log"
)

ax.set_xticks(
    xq
)

ax.set_xticklabels(
    quartile_labels
)

ax.set_xlabel(
    "24 h source-cell pseudotime quartile"
)

ax.set_ylabel(
    r"48 h absolute risk change, $|\Delta\mathrm{risk}|$"
)

ax.set_title(
    "Scale 3 source-quartile perturbation"
)

ax.legend(
    frameon=False,
    loc="upper left",
)

# Explicitly identify the Q4 maximum, which is the
# distinctive sensitivity result.
ax.annotate(
    f"Q4 max = {max_48[3]:.4f}",
    xy=(
        xq[3] + offset,
        max_48[3],
    ),
    xytext=(
        -8,
        -22,
    ),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=7,
)

# Zero values cannot be represented on a log axis.
ax.text(
    0.98,
    0.04,
    "96 h mean and maximum effects = 0\n"
    "for all quartiles",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7,
)

ax.grid(
    axis="y",
    which="both",
    alpha=0.15,
    linewidth=0.5,
)

add_panel_label(
    ax,
    "B",
)


# SAVE


fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(
    fig
)

print(
    "\nCreated:"
)

print(
    " ",
    PNG_OUT,
)

print(
    " ",
    PDF_OUT,
)
