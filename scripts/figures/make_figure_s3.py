from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("TRAJRISK_ROOT", REPO_ROOT)).expanduser().resolve()

import numpy as np
import pandas as pd
from frozen_inputs import read_view, view_path
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from umap import UMAP


# PATHS


DATA = INPUT_ROOT / "Data" / "processed"
RES = INPUT_ROOT / "results"
OUT = REPO_ROOT / "build" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

COMBINED_FILE = view_path(DATA / "adata_cross_species_combined.h5ad", "cross_species")
MOUSE_PRED_FILE = DATA / "week7_predictions.csv"
HUMAN_RISK_FILE = RES / "week9b_cell_risk_scores.csv"

PNG_OUT = OUT / "Figure_S3_cross_species_diagnostics_FINAL.png"
PDF_OUT = OUT / "Figure_S3_cross_species_diagnostics_FINAL.pdf"

RANDOM_SEED = 42
KNN_K = 15


# COLORS


SPECIES_COLORS = {
    "mouse": "#4C78A8",
    "human": "#F28E2B",
}

GROUP_COLORS = {
    "Mouse Control": "#59A14F",
    "Mouse APAP": "#E15759",
    "Human Healthy": "#4C78A8",
    "Human APAP-ALF": "#F28E2B",
}

TIMEPOINT_ORDER = [
    "0h",
    "6h",
    "24h",
    "48h",
    "96h",
]

HUMAN_CONDITION_COLORS = {
    "Healthy": "#4C78A8",
    "APAP-ALF": "#E15759",
}


# STYLE


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# FILE CHECKS


print("===== FIGURE S3 INPUT VALIDATION =====")

for f in [
    COMBINED_FILE,
    MOUSE_PRED_FILE,
    HUMAN_RISK_FILE,
]:
    if not f.exists():
        raise FileNotFoundError(f)


# LOAD FINAL CROSS-SPECIES LATENT SPACE


adata = read_view(
    COMBINED_FILE,
    backed="r"
)

obs = adata.obs.copy()

if "X_scvi_cross" not in adata.obsm:
    raise ValueError(
        "X_scvi_cross is missing from "
        "adata_cross_species_combined.h5ad"
    )

z = np.asarray(
    adata.obsm["X_scvi_cross"][:],
    dtype=np.float32
)

obs_names = pd.Index(
    adata.obs_names.astype(str)
)

print(
    f"Combined cells/nuclei: {len(obs):,}"
)

print(
    f"Cross-species latent shape: {z.shape}"
)

# Close backed file after extracting what we need.
adata.file.close()


# EXPECTED STRUCTURE


species = (
    obs["species"]
    .astype(str)
    .values
)

mouse_mask = species == "mouse"
human_mask = species == "human"

n_mouse = int(mouse_mask.sum())
n_human = int(human_mask.sum())

print(
    f"Mouse hepatocytes: {n_mouse:,}"
)

print(
    f"Human hepatocyte nuclei: {n_human:,}"
)

if n_mouse != 6582:
    raise ValueError(
        f"Expected 6,582 mouse cells; "
        f"found {n_mouse:,}"
    )

if n_human != 49984:
    raise ValueError(
        f"Expected 49,984 human nuclei; "
        f"found {n_human:,}"
    )

if z.shape != (56566, 30):
    raise ValueError(
        "Expected X_scvi_cross shape "
        f"(56566, 30); found {z.shape}"
    )

# human cross-species analysis deliberately placed Walesky first and human second.
if not np.all(mouse_mask[:n_mouse]):
    raise ValueError(
        "Mouse cells are not the first 6,582 rows "
        "of the combined object."
    )

if not np.all(human_mask[n_mouse:]):
    raise ValueError(
        "Human nuclei are not the final 49,984 rows "
        "of the combined object."
    )

print(
    "Mouse-first / human-second ordering: PASS"
)


# NORMALIZE CONCATENATED CELL IDs


# human cross-species analysis used:
#
# sc.concat(
#     [aw, ah],
#     axis=0,
#     join="inner",
#     label="dataset_origin",
#     keys=["walesky", "human"],
#     index_unique="-",
# )
#
# Therefore sc.concat added a terminal:
#   "-walesky" to mouse IDs
#   "-human"   to human IDs
#
# These suffixes were not present in the original reference model /
# human cross-species analysis source objects. Remove ONLY the final concat suffix.

mouse_ids_combined = pd.Index(
    obs_names[mouse_mask].astype(str)
)

human_ids_combined = pd.Index(
    obs_names[human_mask].astype(str)
)

if not mouse_ids_combined.str.endswith(
    "-walesky"
).all():
    raise ValueError(
        "Not all cross-species mouse IDs end with "
        "'-walesky' as expected from sc.concat."
    )

if not human_ids_combined.str.endswith(
    "-human"
).all():
    raise ValueError(
        "Not all cross-species human IDs end with "
        "'-human' as expected from sc.concat."
    )

mouse_ids = mouse_ids_combined.str.replace(
    r"-walesky$",
    "",
    regex=True
)

human_ids = human_ids_combined.str.replace(
    r"-human$",
    "",
    regex=True
)

if not mouse_ids.is_unique:
    raise ValueError(
        "Mouse IDs are not unique after removing "
        "the terminal '-walesky' concat suffix."
    )

if not human_ids.is_unique:
    raise ValueError(
        "Human IDs are not unique after removing "
        "the terminal '-human' concat suffix."
    )

print(
    "Concat suffix normalization: PASS"
)

print(
    "Example mouse ID:"
)
print(
    "  combined:  ",
    mouse_ids_combined[0]
)
print(
    "  normalized:",
    mouse_ids[0]
)

print(
    "Example human ID:"
)
print(
    "  combined:  ",
    human_ids_combined[0]
)
print(
    "  normalized:",
    human_ids[0]
)


# LOAD / ALIGN FINAL reference model MOUSE METADATA


mouse_pred = pd.read_csv(
    MOUSE_PRED_FILE,
    index_col=0
)

mouse_pred.index = (
    mouse_pred.index.astype(str)
)

mouse_pred_ids = set(
    mouse_pred.index
)

cross_mouse_ids = set(
    mouse_ids
)

if mouse_pred_ids != cross_mouse_ids:

    only_pred = (
        mouse_pred_ids
        - cross_mouse_ids
    )

    only_cross = (
        cross_mouse_ids
        - mouse_pred_ids
    )

    raise ValueError(
        "reference model mouse prediction IDs still do not "
        "match cross-species mouse IDs after concat "
        "suffix normalization.\n"
        f"Only reference model: {len(only_pred)}\n"
        f"Only cross-species: {len(only_cross)}"
    )

# Reindex into EXACT cross-species mouse row order.
mouse_meta = mouse_pred.reindex(
    mouse_ids
)

required_mouse_cols = [
    "timepoint_str",
    "treatment_str",
]

missing_mouse = [
    c for c in required_mouse_cols
    if c not in mouse_meta.columns
]

if missing_mouse:
    raise ValueError(
        "Missing required reference model mouse columns: "
        f"{missing_mouse}"
    )

if mouse_meta[
    required_mouse_cols
].isna().any().any():
    raise ValueError(
        "Mouse metadata failed to align after "
        "concat-suffix normalization."
    )

print(
    "Mouse reference model metadata alignment: PASS"
)


# LOAD / ALIGN FINAL human cross-species analysis HUMAN RISK SCORES


human_risk = pd.read_csv(
    HUMAN_RISK_FILE,
    index_col=0
)

human_risk.index = (
    human_risk.index.astype(str)
)

human_risk_ids = set(
    human_risk.index
)

cross_human_ids = set(
    human_ids
)

if human_risk_ids != cross_human_ids:

    only_csv = (
        human_risk_ids
        - cross_human_ids
    )

    only_cross = (
        cross_human_ids
        - human_risk_ids
    )

    raise ValueError(
        "Human human cross-species analysis risk-score IDs do not match "
        "cross-species human IDs after concat suffix "
        "normalization.\n"
        f"Only risk CSV: {len(only_csv)}\n"
        f"Only cross-species: {len(only_cross)}"
    )

# Reindex into EXACT cross-species human row order.
human_risk = human_risk.reindex(
    human_ids
)

if "risk_2a" not in human_risk.columns:
    raise ValueError(
        "risk_2a is missing from "
        "week9b_cell_risk_scores.csv"
    )

if human_risk[
    "risk_2a"
].isna().any():
    raise ValueError(
        "Missing risk_2a values after "
        "human-ID alignment."
    )

risk_2a = (
    human_risk["risk_2a"]
    .astype(float)
    .values
)

print(
    "Human risk_2a alignment: PASS"
)

print(
    "Primary W_self risk range: "
    f"[{risk_2a.min():.4f}, "
    f"{risk_2a.max():.4f}]"
)


# BUILD SPECIES-AWARE BIOLOGICAL GROUP LABELS


group = np.empty(
    len(obs),
    dtype=object
)

# Mouse labels come from final reference model metadata.
mouse_treatment = (
    mouse_meta["treatment_str"]
    .astype(str)
    .values
)

group[
    np.where(mouse_mask)[0]
] = np.where(
    mouse_treatment == "Control",
    "Mouse Control",
    "Mouse APAP"
)

# Human labels come from final human cross-species analysis metadata.
human_condition = (
    obs.loc[
        human_mask,
        "condition"
    ]
    .astype(str)
    .replace({
        "APAP_ALF": "APAP-ALF"
    })
    .values
)

human_group = np.where(
    human_condition == "Healthy",
    "Human Healthy",
    "Human APAP-ALF"
)

group[
    np.where(human_mask)[0]
] = human_group

print(
    "\nBiological groups:"
)

for g in GROUP_COLORS:
    print(
        f"  {g:<18}: "
        f"{np.sum(group == g):,}"
    )


# REPRODUCIBLE UMAP FROM FINAL X_scvi_cross


print(
    "\nComputing UMAP from final X_scvi_cross "
    "(visualization only)..."
)

umap_model = UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.3,
    metric="euclidean",
    random_state=RANDOM_SEED,
    n_jobs=1,
    low_memory=True,
)

umap_2d = umap_model.fit_transform(
    z
)

print(
    f"UMAP shape: {umap_2d.shape}"
)


# RECOMPUTE EXACT human cross-species analysis REFERENCE-NEIGHBOUR COMPOSITION


print(
    "\nRecomputing 15-nearest-Walesky-neighbour "
    "composition..."
)

z_mouse = z[mouse_mask]
z_human = z[human_mask]

knn = NearestNeighbors(
    n_neighbors=KNN_K,
    metric="euclidean",
    n_jobs=-1,
)

knn.fit(
    z_mouse
)

_, local_idx = knn.kneighbors(
    z_human
)

# mouse_meta has already been reordered into z_mouse order.
mouse_tp = (
    mouse_meta["timepoint_str"]
    .astype(str)
    .values
)

nbr_tp = mouse_tp[
    local_idx
]

composition = {}

for tp in TIMEPOINT_ORDER:
    composition[tp] = (
        nbr_tp == tp
    ).mean(
        axis=1
    )

composition_df = pd.DataFrame(
    composition,
    index=human_ids
)

composition_df[
    "condition"
] = human_condition

# Same descriptive averaging logic as human cross-species analysis.
comp_summary = (
    composition_df
    .groupby(
        "condition",
        observed=True
    )[TIMEPOINT_ORDER]
    .mean()
)

print(
    "\nReference-neighbour composition "
    "(mean fraction per human nucleus):"
)

print(
    comp_summary.round(4)
)

# Save exact source summary used by Panel D.
source_out = OUT / "week9b_reference_neighbor_composition_summary.csv"

comp_summary.to_csv(
    source_out
)

print(
    f"\nSaved composition summary: "
    f"{source_out}"
)


# FIGURE


fig, axes = plt.subplots(
    1,
    4,
    figsize=(17.2, 4.6)
)

# Shared UMAP bounds
xmin = umap_2d[:, 0].min()
xmax = umap_2d[:, 0].max()

ymin = umap_2d[:, 1].min()
ymax = umap_2d[:, 1].max()

xpad = (
    xmax - xmin
) * 0.03

ypad = (
    ymax - ymin
) * 0.03


# PANEL A — SPECIES


ax = axes[0]

# Plot human first and mouse second so the smaller
# mouse population remains visible.
for sp in [
    "human",
    "mouse",
]:
    mask = species == sp

    alpha = (
        0.22
        if sp == "human"
        else 0.48
    )

    size = (
        2.2
        if sp == "human"
        else 3.2
    )

    ax.scatter(
        umap_2d[mask, 0],
        umap_2d[mask, 1],
        s=size,
        alpha=alpha,
        linewidths=0,
        color=SPECIES_COLORS[sp],
        label=(
            f"{sp.capitalize()} "
            f"(n={mask.sum():,})"
        ),
        rasterized=True
    )

ax.set_title(
    "Cross-species scVI UMAP by species"
)

ax.set_xlabel(
    "UMAP 1"
)

ax.set_ylabel(
    "UMAP 2"
)

ax.legend(
    frameon=False,
    markerscale=3,
    loc="best"
)


# PANEL B — SPECIES-AWARE BIOLOGICAL CONDITION


ax = axes[1]

for g in [
    "Mouse Control",
    "Mouse APAP",
    "Human Healthy",
    "Human APAP-ALF",
]:
    mask = group == g

    ax.scatter(
        umap_2d[mask, 0],
        umap_2d[mask, 1],
        s=2.6,
        alpha=0.30,
        linewidths=0,
        color=GROUP_COLORS[g],
        label=g,
        rasterized=True
    )

ax.set_title(
    "Cross-species scVI UMAP by condition"
)

ax.set_xlabel(
    "UMAP 1"
)

ax.set_ylabel(
    "UMAP 2"
)

ax.legend(
    frameon=False,
    markerscale=3,
    loc="best"
)


# PANEL C — PRIMARY W_self HUMAN RISK


ax = axes[2]

# Mouse cells provide only faint reference context.
ax.scatter(
    umap_2d[mouse_mask, 0],
    umap_2d[mouse_mask, 1],
    s=2.0,
    alpha=0.08,
    linewidths=0,
    color="#9E9E9E",
    rasterized=True
)

# Human nuclei colored by PRIMARY W_self score.
scat = ax.scatter(
    umap_2d[human_mask, 0],
    umap_2d[human_mask, 1],
    c=risk_2a,
    cmap="plasma",
    vmin=0.0,
    vmax=1.0,
    s=2.5,
    alpha=0.42,
    linewidths=0,
    rasterized=True
)

ax.set_title(
    r"Human nuclei by primary $W_{\mathrm{self}}$ risk"
)

ax.set_xlabel(
    "UMAP 1"
)

ax.set_ylabel(
    "UMAP 2"
)

cbar = fig.colorbar(
    scat,
    ax=ax,
    fraction=0.047,
    pad=0.035
)

cbar.set_label(
    "Predicted TrajRisk score",
    fontsize=8
)

cbar.ax.tick_params(
    labelsize=7
)


# PANEL D — REFERENCE-KNN NEIGHBOUR COMPOSITION


ax = axes[3]

x = np.arange(
    len(TIMEPOINT_ORDER)
)

width = 0.36

healthy = (
    comp_summary
    .loc[
        "Healthy",
        TIMEPOINT_ORDER
    ]
    .values
    * 100
)

alf = (
    comp_summary
    .loc[
        "APAP-ALF",
        TIMEPOINT_ORDER
    ]
    .values
    * 100
)

ax.bar(
    x - width / 2,
    healthy,
    width=width,
    color=HUMAN_CONDITION_COLORS[
        "Healthy"
    ],
    label="Healthy"
)

ax.bar(
    x + width / 2,
    alf,
    width=width,
    color=HUMAN_CONDITION_COLORS[
        "APAP-ALF"
    ],
    label="APAP-ALF"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    TIMEPOINT_ORDER
)

ax.set_xlabel(
    "Walesky reference timepoint"
)

ax.set_ylabel(
    "Mean fraction of 15 neighbors (%)"
)

ax.set_title(
    "Reference-neighbour composition\n"
    "(Reference-KNN sensitivity)"
)

ax.legend(
    frameon=False,
    loc="upper right"
)

ax.grid(
    alpha=0.15,
    axis="y",
    linewidth=0.5
)


# IDENTICAL UMAP LIMITS FOR A–C


for ax in axes[:3]:
    ax.set_xlim(
        xmin - xpad,
        xmax + xpad
    )

    ax.set_ylim(
        ymin - ypad,
        ymax + ypad
    )

    # UMAP coordinates are arbitrary.
    ax.set_xticks([])
    ax.set_yticks([])


# PANEL LETTERS


for label, ax in zip(
    ["A", "B", "C", "D"],
    axes
):
    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left"
    )


# FINISH


fig.subplots_adjust(
    left=0.045,
    right=0.985,
    bottom=0.16,
    top=0.86,
    wspace=0.32
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

print(
    "\nCreated:"
)

print(
    " ",
    PNG_OUT
)

print(
    " ",
    PDF_OUT
)
