from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("TRAJRISK_ROOT", REPO_ROOT)).expanduser().resolve()

import numpy as np
import pandas as pd
from frozen_inputs import read_view, view_path
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# ============================================================
# PATHS
# ============================================================

DATA = INPUT_ROOT / "Data" / "processed"
OUT = REPO_ROOT / "build" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

ADATA_FILE = view_path(DATA / "adata_integrated_with_trajectory.h5ad", "mouse_integration")
PRED_FILE = DATA / "week7_predictions.csv"
EMB_FILE = DATA / "week7_scale_embeddings.npy"

PNG_OUT = OUT / "Figure_S2_pseudotime_representation_FINAL.png"
PDF_OUT = OUT / "Figure_S2_pseudotime_representation_FINAL.pdf"

RANDOM_SEED = 42

TIMEPOINT_ORDER = [
    "0h",
    "6h",
    "24h",
    "48h",
    "96h",
]

TIMEPOINT_COLORS = {
    "0h": "#2E8B57",
    "6h": "#D55E4F",
    "24h": "#E6953B",
    "48h": "#8E63A9",
    "96h": "#4C8FC2",
}

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# LOAD
# ============================================================

print("===== FIGURE S2 INPUT AUDIT =====")

for f in [
    ADATA_FILE,
    PRED_FILE,
    EMB_FILE,
]:
    if not f.exists():
        raise FileNotFoundError(f)

adata = read_view(ADATA_FILE)

walesky_mask = (
    adata.obs["dataset"]
    .astype(str)
    .eq("Walesky")
)

walesky = adata[walesky_mask].copy()

pred = pd.read_csv(
    PRED_FILE,
    index_col=0
)

embeddings = np.load(
    EMB_FILE
)

print(
    f"Walesky AnnData cells: "
    f"{walesky.n_obs:,}"
)

print(
    f"Prediction rows: "
    f"{len(pred):,}"
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}"
)

# ============================================================
# BASIC INTEGRITY CHECKS
# ============================================================

if walesky.n_obs != 6582:
    raise ValueError(
        f"Expected 6,582 Walesky cells; "
        f"found {walesky.n_obs:,}"
    )

if len(pred) != 6582:
    raise ValueError(
        f"Expected 6,582 prediction rows; "
        f"found {len(pred):,}"
    )

if embeddings.shape != (6582, 4, 64):
    raise ValueError(
        "Expected embedding shape "
        f"(6582, 4, 64); found "
        f"{embeddings.shape}"
    )

if not pred.index.is_unique:
    raise ValueError(
        "week7_predictions.csv contains "
        "duplicate cell indices."
    )

if not walesky.obs_names.is_unique:
    raise ValueError(
        "Walesky AnnData contains "
        "duplicate cell indices."
    )

# ============================================================
# BARCODE ALIGNMENT CHECK
# ============================================================

pred_ids = set(
    pred.index.astype(str)
)

walesky_ids = set(
    walesky.obs_names.astype(str)
)

if pred_ids != walesky_ids:

    only_pred = pred_ids - walesky_ids
    only_adata = walesky_ids - pred_ids

    raise ValueError(
        "Prediction and Walesky AnnData "
        "cell IDs do not match.\n"
        f"Only in predictions: {len(only_pred)}\n"
        f"Only in AnnData: {len(only_adata)}"
    )

print(
    "Cell-ID set alignment: PASS"
)

# ============================================================
# REQUIRED PREDICTION COLUMNS
# ============================================================

required_pred_cols = [
    "timepoint_str",
    "risk_predicted",
]

missing = [
    c for c in required_pred_cols
    if c not in pred.columns
]

if missing:
    raise ValueError(
        "Missing required prediction columns: "
        f"{missing}"
    )

# ============================================================
# CHECK RISK SCORES AGAINST AnnData
# ============================================================

if "risk_predicted" not in walesky.obs.columns:
    raise ValueError(
        "risk_predicted is missing from "
        "Walesky AnnData."
    )

adata_risk = (
    walesky.obs["risk_predicted"]
    .reindex(pred.index)
    .astype(float)
    .values
)

pred_risk = (
    pred["risk_predicted"]
    .astype(float)
    .values
)

if not np.all(
    np.isfinite(pred_risk)
):
    raise ValueError(
        "Non-finite values found in "
        "week7_predictions.csv."
    )

if not np.allclose(
    pred_risk,
    adata_risk,
    rtol=1e-5,
    atol=1e-6,
    equal_nan=False,
):
    max_diff = np.max(
        np.abs(
            pred_risk - adata_risk
        )
    )

    raise ValueError(
        "Prediction values do not agree "
        "between CSV and AnnData.\n"
        f"Maximum difference: {max_diff}"
    )

print(
    "Prediction-to-AnnData alignment: PASS"
)

# ============================================================
# RAW MONOCLE PSEUDOTIME CHECK
# ============================================================

if "pseudotime" not in walesky.obs.columns:
    raise ValueError(
        "Raw Monocle pseudotime column "
        "not found in AnnData."
    )

if "timepoint_str" not in walesky.obs.columns:
    raise ValueError(
        "timepoint_str column not found "
        "in Walesky AnnData."
    )

raw_pt = pd.to_numeric(
    walesky.obs["pseudotime"],
    errors="coerce"
)

if raw_pt.isna().any():
    raise ValueError(
        f"{raw_pt.isna().sum()} Walesky cells "
        "have missing raw pseudotime."
    )

# ============================================================
# TIMEPOINT COUNTS
# ============================================================

print("\nRaw pseudotime cohort:")

for tp in TIMEPOINT_ORDER:

    n = (
        walesky.obs["timepoint_str"]
        .astype(str)
        .eq(tp)
        .sum()
    )

    print(
        f"  {tp:>3}: {n:,}"
    )

# ============================================================
# PCA FROM SAVED FINAL EMBEDDINGS
# ============================================================

# (6582, 4, 64) -> (6582, 256)
combined = embeddings.reshape(
    embeddings.shape[0],
    -1
)

pca = PCA(
    n_components=2,
    random_state=RANDOM_SEED
)

pca_2d = pca.fit_transform(
    combined
)

pc1_var = (
    pca.explained_variance_ratio_[0]
    * 100
)

pc2_var = (
    pca.explained_variance_ratio_[1]
    * 100
)

print(
    "\nPCA explained variance:"
)

print(
    f"  PC1: {pc1_var:.2f}%"
)

print(
    f"  PC2: {pc2_var:.2f}%"
)

# Original Week 7 figure was approximately:
# PC1 = 17.9%, PC2 = 10.9%.
# We reproduce these from the saved final embeddings
# rather than hard-coding the values.

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15.5, 4.7)
)

# ============================================================
# PANEL A — RAW MONOCLE PSEUDOTIME ECDF
# ============================================================

ax = axes[0]

for tp in TIMEPOINT_ORDER:

    mask = (
        walesky.obs["timepoint_str"]
        .astype(str)
        .eq(tp)
    )

    values = (
        pd.to_numeric(
            walesky.obs.loc[
                mask,
                "pseudotime"
            ],
            errors="coerce"
        )
        .dropna()
        .values
    )

    values = np.sort(
        values
    )

    ecdf = (
        np.arange(
            1,
            len(values) + 1
        )
        / len(values)
    )

    ax.plot(
        values,
        ecdf,
        linewidth=1.8,
        color=TIMEPOINT_COLORS[tp],
        label=f"{tp}  (n={len(values):,})"
    )

ax.set_xlabel(
    "Raw Monocle pseudotime"
)

ax.set_ylabel(
    "Cumulative fraction"
)

ax.set_title(
    "Raw Monocle pseudotime ECDFs"
)

# Position legend away from the 6 h ECDF.
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.58, 0.88),
    ncol=1
)

ax.grid(
    alpha=0.15,
    linewidth=0.5
)

# ============================================================
# PANEL B — EMBEDDING PCA BY TIMEPOINT
# ============================================================

ax = axes[1]

tp_values = (
    pred["timepoint_str"]
    .astype(str)
    .values
)

for tp in TIMEPOINT_ORDER:

    mask = (
        tp_values == tp
    )

    ax.scatter(
        pca_2d[mask, 0],
        pca_2d[mask, 1],
        s=7,
        alpha=0.40,
        linewidths=0,
        color=TIMEPOINT_COLORS[tp],
        label=tp,
        rasterized=True
    )

ax.set_xlabel(
    f"PC1 ({pc1_var:.1f}%)"
)

ax.set_ylabel(
    f"PC2 ({pc2_var:.1f}%)"
)

ax.set_title(
    "Multi-scale representation by timepoint"
)

ax.legend(
    frameon=False,
    markerscale=2,
    loc="upper right"
)

# ============================================================
# PANEL C — SAME PCA, COLORED BY RISK
# ============================================================

ax = axes[2]

scat = ax.scatter(
    pca_2d[:, 0],
    pca_2d[:, 1],
    c=pred_risk,
    cmap="plasma",
    s=7,
    alpha=0.48,
    linewidths=0,
    rasterized=True
)

ax.set_xlabel(
    f"PC1 ({pc1_var:.1f}%)"
)

ax.set_ylabel(
    f"PC2 ({pc2_var:.1f}%)"
)

ax.set_title(
    "Multi-scale representation by predicted risk"
)

cbar = fig.colorbar(
    scat,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar.set_label(
    "Predicted TrajRisk score",
    fontsize=8
)

cbar.ax.tick_params(
    labelsize=7
)

# ============================================================
# MATCH PCA AXES EXACTLY
# ============================================================

xmin = pca_2d[:, 0].min()
xmax = pca_2d[:, 0].max()

ymin = pca_2d[:, 1].min()
ymax = pca_2d[:, 1].max()

xpad = (
    xmax - xmin
) * 0.04

ypad = (
    ymax - ymin
) * 0.04

for ax in axes[1:]:
    ax.set_xlim(
        xmin - xpad,
        xmax + xpad
    )

    ax.set_ylim(
        ymin - ypad,
        ymax + ypad
    )

# ============================================================
# PANEL LABELS
# ============================================================

for label, ax in zip(
    ["A", "B", "C"],
    axes
):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left"
    )

# ============================================================
# FINISH
# ============================================================

fig.subplots_adjust(
    left=0.06,
    right=0.97,
    bottom=0.16,
    top=0.88,
    wspace=0.34
)

fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("\nCreated:")
print(" ", PNG_OUT)
print(" ", PDF_OUT)
