#!/usr/bin/env python3
"""
Supplementary Figure S_ — Species Integration UMAP
====================================================
Diagnostic figure demonstrating cross-species scVI integration quality.
Panels:
  A — UMAP coloured by species (mouse / human)
  B — UMAP coloured by condition (APAP/Control/APAP-ALF/Healthy)
  C — UMAP coloured by W_self TrajRisk score

Checks for an existing X_umap embedding first.
If absent, computes neighbours (using X_scvi_cross) then UMAP.
The combined adata is NOT modified; embeddings are computed in memory only.
Subsamples to 20,000 cells for plotting speed without losing structure.

Usage:
    conda activate trajrisk
    cd /path/to/TrajRisk
    python scripts/generate_umap_figure.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import scanpy as sc

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "Data" / "processed"
RES  = BASE / "results"
FIGS = RES / "manuscript_figures"
FIGS.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================
plt.rcParams.update({
    "font.family":        "Arial",
    "font.size":          8,
    "axes.labelsize":     8,
    "axes.titlesize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.75,
    "xtick.major.width":  0.75,
    "ytick.major.width":  0.75,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

SUBSAMPLE_N = 20_000   # cells to plot
RANDOM_SEED = 42

# =============================================================================
# STEP 0 — File checks
# =============================================================================
combined_path = DATA / "adata_cross_species_combined.h5ad"
cell_scores    = RES  / "week9b_cell_risk_scores.csv"

for label, path in [("Combined adata", combined_path),
                    ("Cell risk scores", cell_scores)]:
    if not path.exists():
        print(f"ERROR: {label} not found at {path}")
        sys.exit(1)
    print(f"OK:  {path.name}  ({path.stat().st_size // (1024**2)} MB)")

# =============================================================================
# STEP 1 — Load combined adata
# =============================================================================
print("\nLoading combined adata (this may take a minute) ...")
adata = sc.read_h5ad(str(combined_path))
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"  obsm keys: {list(adata.obsm.keys())}")
print(f"  obs columns: {list(adata.obs.columns)}")

assert "X_scvi_cross" in adata.obsm, \
    "X_scvi_cross not found — was week9b run to completion?"

# =============================================================================
# STEP 2 — Compute UMAP if not already present
# =============================================================================
if "X_umap" in adata.obsm:
    print("\nX_umap already present — skipping computation.")
else:
    print("\nX_umap not found — computing neighbours + UMAP ...")
    print("  (Uses X_scvi_cross as latent representation)")
    sc.pp.neighbors(adata, use_rep="X_scvi_cross",
                    n_neighbors=15, random_state=RANDOM_SEED)
    sc.tl.umap(adata, min_dist=0.3, random_state=RANDOM_SEED)
    print(f"  UMAP computed: {adata.obsm['X_umap'].shape}")

umap_coords = adata.obsm["X_umap"]

# =============================================================================
# STEP 3 — Attach TrajRisk W_self scores to human cells
# =============================================================================
cell_df = pd.read_csv(str(cell_scores), index_col=0)

# Human cells are positions n_wal onward in the combined adata
n_wal  = int((adata.obs["species"] == "mouse").sum())
n_hum  = adata.n_obs - n_wal

risk_scores_full = np.full(adata.n_obs, np.nan)
risk_scores_full[n_wal:] = cell_df["risk_2a"].values

adata.obs["risk_2a"] = risk_scores_full
print(f"\nWalesky cells: {n_wal:,}   Human cells: {n_hum:,}")

# =============================================================================
# STEP 4 — Subsample for plotting
# =============================================================================
rng = np.random.default_rng(RANDOM_SEED)
idx = rng.choice(adata.n_obs, size=min(SUBSAMPLE_N, adata.n_obs), replace=False)
idx_sorted = np.sort(idx)

umap_sub = umap_coords[idx_sorted]
species_sub   = adata.obs["species"].values[idx_sorted]
condition_raw = adata.obs.get("condition", adata.obs.get("patient_type", None))
if condition_raw is None:
    condition_raw = np.where(adata.obs["species"] == "mouse", "mouse", "human")
else:
    condition_raw = condition_raw.values
condition_sub = condition_raw[idx_sorted]
risk_sub      = adata.obs["risk_2a"].values[idx_sorted]

print(f"\nPlotting {len(idx_sorted):,} cells (subsampled from {adata.n_obs:,})")

# =============================================================================
# STEP 5 — Build figure
# =============================================================================
print("Generating supplementary UMAP figure ...")

fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4),
                          gridspec_kw={"wspace": 0.35})
fig.subplots_adjust(left=0.06, right=0.93, top=0.88, bottom=0.12)

dot_s = 1.2
alpha = 0.30

# ── Panel A: species ──────────────────────────────────────────────────────────
ax = axes[0]
species_colours = {"human": "#1A6EAF", "mouse": "#E07B39"}
for sp, col in species_colours.items():
    mask = species_sub == sp
    ax.scatter(umap_sub[mask, 0], umap_sub[mask, 1],
               c=col, s=dot_s, alpha=alpha, linewidths=0,
               rasterized=True)
ax.set_xlabel("UMAP 1", fontsize=7)
ax.set_ylabel("UMAP 2", fontsize=7)
ax.set_title("Species", fontsize=8, pad=5)
ax.set_xticks([]); ax.set_yticks([])
handles_sp = [Line2D([0],[0], marker="o", color="w",
                      markerfacecolor=col, markersize=5, label=sp.capitalize())
              for sp, col in species_colours.items()]
ax.legend(handles=handles_sp, loc="lower left", fontsize=7,
          frameon=False, markerscale=1.2, handletextpad=0.3)
# ASW annotation
asw = adata.uns.get("integration_quality", {}).get("asw_species")
if asw is not None:
    ax.text(0.98, 0.02, f"ASW(species) = {asw:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
ax.text(-0.06, 1.08, "A", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top")

# ── Panel B: condition ────────────────────────────────────────────────────────
ax = axes[1]
cond_colours = {
    "APAP":     "#C0392B",
    "Control":  "#2E86C1",
    "APAP_ALF": "#C0392B",
    "Healthy":  "#2E86C1",
    "unknown":  "#BDBDBD",
    "mouse":    "#DDDDDD",
    "human":    "#DDDDDD",
}
# Plot in specific order so APAP/ALF is visible on top
order = ["Control", "Healthy", "unknown", "mouse", "human", "APAP", "APAP_ALF"]
conds_present = list(dict.fromkeys(
    [c for c in order if c in set(condition_sub)] +
    [c for c in set(condition_sub) if c not in order]
))
for cond in conds_present:
    mask = condition_sub == cond
    col  = cond_colours.get(cond, "#999999")
    ax.scatter(umap_sub[mask, 0], umap_sub[mask, 1],
               c=col, s=dot_s, alpha=alpha, linewidths=0,
               rasterized=True)
ax.set_xlabel("UMAP 1", fontsize=7)
ax.set_title("Condition", fontsize=8, pad=5)
ax.set_xticks([]); ax.set_yticks([])
handles_cond = [
    Line2D([0],[0], marker="o", color="w",
           markerfacecolor="#C0392B", markersize=5, label="APAP/APAP-ALF"),
    Line2D([0],[0], marker="o", color="w",
           markerfacecolor="#2E86C1", markersize=5, label="Control/Healthy"),
]
ax.legend(handles=handles_cond, loc="lower left", fontsize=7,
          frameon=False, markerscale=1.2, handletextpad=0.3)
ax.text(-0.06, 1.08, "B", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top")

# ── Panel C: TrajRisk score (human only, W_self) ──────────────────────────────
ax = axes[2]
# Background: all cells grey
ax.scatter(umap_sub[:, 0], umap_sub[:, 1],
           c="#E0E0E0", s=dot_s, alpha=0.20, linewidths=0, rasterized=True)
# Human cells with valid risk scores
hum_mask = ~np.isnan(risk_sub)
sc_plot = ax.scatter(
    umap_sub[hum_mask, 0], umap_sub[hum_mask, 1],
    c=risk_sub[hum_mask], cmap="RdYlBu_r",
    vmin=float(np.nanpercentile(risk_sub[hum_mask], 1)),
    vmax=float(np.nanpercentile(risk_sub[hum_mask], 99)),
    s=dot_s * 1.2, alpha=0.50, linewidths=0, rasterized=True
)
cbar = plt.colorbar(sc_plot, ax=ax, shrink=0.70, pad=0.02, aspect=18)
cbar.set_label("TrajRisk score", fontsize=7)
cbar.ax.tick_params(labelsize=6)
ax.set_xlabel("UMAP 1", fontsize=7)
ax.set_title("TrajRisk (human cells)", fontsize=8, pad=5)
ax.set_xticks([]); ax.set_yticks([])
ax.text(-0.06, 1.08, "C", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top")

# =============================================================================
# STEP 6 — Save
# =============================================================================
for ext in [".pdf", ".svg", ".png"]:
    fig.savefig(str(FIGS / ("supp_figure_umap" + ext)))
plt.close(fig)

print("\nSaved: supp_figure_umap.pdf / .svg / .png")
print(f"Location: {FIGS}")
print("\nSupplementary UMAP figure complete.")
