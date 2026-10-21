#!/usr/bin/env python3
"""Earlier TrajRisk analysis retained for source comparison.

See docs/workflow.md for the study workflow and docs/reproduction.md for
the commands used to reproduce the reported analyses.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from umap import UMAP
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
RES  = BASE / "results"
FIGS = RES / "manuscript_figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":"Arial","font.size":8,
    "axes.labelsize":8,"axes.titlesize":8,
    "xtick.labelsize":7,"ytick.labelsize":7,
    "axes.linewidth":0.75,
    "xtick.major.width":0.75,"ytick.major.width":0.75,
    "xtick.major.size":3,"ytick.major.size":3,
    "xtick.direction":"out","ytick.direction":"out",
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":300,"savefig.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.05,
    "pdf.fonttype":42,"ps.fonttype":42,
    "legend.frameon":False,
})

# Unified palette — identical hex codes across all panels
TP_COL = {0:"#4575B4", 6:"#D73027", 24:"#FC8D59", 48:"#74C476", 96:"#27AE60"}
TP_LABELS = {0:"0h\n(n=3,401)", 6:"6h\n(n=419)",
             24:"24h\n(n=1,043)", 48:"48h\n(n=910)", 96:"96h\n(n=809)"}
TP_ORDER = [0, 6, 24, 48, 96]

SCALE_COL = {
    "Scale1_AcuteStress":"#C0392B","Scale2_PeakInjury":"#E67E22",
    "Scale3_FateDecision":"#27AE60","Scale4_Resolution":"#2980B9",
}
SCALE_ORDER = ["Scale1_AcuteStress","Scale2_PeakInjury",
               "Scale3_FateDecision","Scale4_Resolution"]
SCALE_SHORT = {"Scale1_AcuteStress":"S1\n(0h→6h)","Scale2_PeakInjury":"S2\n(6h→24h)",
               "Scale3_FateDecision":"S3\n(24h→48h)","Scale4_Resolution":"S4\n(48h→96h)"}

# Fully opaque white box — prevents all colour bleeding
WBOX = dict(facecolor="white", edgecolor="none", alpha=1.0, pad=2.0)

def label_panel(ax, letter, x=-0.16, y=1.09):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

def save_fig(fig, stem):
    for ext in [".pdf", ".svg", ".png"]:
        fig.savefig(str(FIGS / (stem + ext)))
    print(f"  Saved: {stem}")

def draw_violin(ax, data_dict):
    """Draw one violin per timepoint. Returns dict of violin tops."""
    tops = {}
    for pos, tp in enumerate(TP_ORDER):
        vals = data_dict[tp]
        col  = TP_COL[tp]
        parts = ax.violinplot([vals], positions=[pos],
                              widths=0.65, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.72)
            pc.set_edgecolor("black"); pc.set_linewidth(0.4)
        med = float(np.median(vals))
        q1  = float(np.percentile(vals, 25))
        q3  = float(np.percentile(vals, 75))
        ax.vlines(pos, q1, q3, color="black", linewidth=1.5, zorder=5)
        ax.scatter([pos], [med], color="white", edgecolors="black",
                   s=18, zorder=6, linewidths=0.7)
        tops[tp] = float(np.percentile(vals, 75))
    ax.set_xticks(range(5))
    ax.set_xticklabels([TP_LABELS[tp] for tp in TP_ORDER], fontsize=6.5)
    return tops


# LOAD ADATA ONCE

print("Loading adata ...")
adata = sc.read_h5ad(
    str(BASE/"Data/processed/adata_integrated_with_trajectory.h5ad"), backed="r")
walesky_mask = adata.obs["dataset"] == "Walesky"
walesky_obs  = adata.obs[walesky_mask].copy()
walesky_scvi = adata.obsm["X_scvi"][walesky_mask.values]
adata.file.close()
print(f"  Walesky cells: {len(walesky_obs)}  |  X_scvi: {walesky_scvi.shape}")

ptime  = walesky_obs["pseudotime_norm"].values
tp_hrs = walesky_obs["timepoint_hours"].values


# FIGURE 1D — Walesky-only UMAP from X_scvi

print("Computing Walesky UMAP ...")
umap_coords = UMAP(n_components=2, n_neighbors=15, min_dist=0.3,
                   random_state=42, n_jobs=1).fit_transform(walesky_scvi)
umap1, umap2 = umap_coords[:,0], umap_coords[:,1]

# Sort low→high pseudotime so high-ptime cells plot on top
sort_idx = np.argsort(ptime)

fig_1d, ax_1d = plt.subplots(figsize=(3.4, 3.0))
fig_1d.subplots_adjust(left=0.12, right=0.82, top=0.90, bottom=0.12)

sc_plot = ax_1d.scatter(
    umap1[sort_idx], umap2[sort_idx],
    c=ptime[sort_idx], cmap="plasma",
    s=1.8, linewidths=0, alpha=0.82, rasterized=True)

cbar = plt.colorbar(sc_plot, ax=ax_1d, shrink=0.72, pad=0.03)
cbar.set_label("Pseudotime\n(normalised)", fontsize=6.5)
cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
cbar.ax.tick_params(labelsize=6)

ax_1d.set_xlabel("UMAP 1"); ax_1d.set_ylabel("UMAP 2")
ax_1d.tick_params(labelsize=6)
# title removed for CRM style
label_panel(ax_1d, "D", x=-0.14)
save_fig(fig_1d, "figure_1d_pseudotime_umap")
plt.close(fig_1d)


# FIGURE 2A — Clock-time vs pseudotime

print("Building Figure 2A ...")
ptime_by_tp = {tp: walesky_obs.loc[tp_hrs==float(tp), "pseudotime_norm"].values
               for tp in TP_ORDER}

fig_2a, ax_2a = plt.subplots(figsize=(3.8, 2.9))
fig_2a.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.22)
tops_2a = draw_violin(ax_2a, ptime_by_tp)

# Median labels — all except 6h (annotation replaces it)
for pos, tp in enumerate(TP_ORDER):
    if tp == 6:
        continue   # replaced by annotation below
    med    = float(np.median(ptime_by_tp[tp]))
    top    = tops_2a[tp]
    label_y = top + 0.018
    ax_2a.text(pos, label_y, f"{med:.2f}",
               ha="center", va="bottom", fontsize=5.5,
               color=TP_COL[tp], bbox=WBOX)

# 6h annotation — in empty white space to the RIGHT of 6h, pointing LEFT
# Text at x=2.3 (between 24h pos=2 and 48h pos=3), y=0.96
med_6h = float(np.median(ptime_by_tp[6]))
ax_2a.annotate(
    f"6h median = {med_6h:.2f}",    xy=(1, med_6h),          # arrow tip at 6h median dot
    xytext=(2.35, 0.96),     # text in empty white space right of 6h
    ha="center", va="bottom",
    fontsize=5.0, color=TP_COL[6],
    bbox=WBOX,
    arrowprops=dict(
        arrowstyle="-|>", color=TP_COL[6], lw=0.65,
        connectionstyle="arc3,rad=-0.30",
        shrinkA=4, shrinkB=6)
)

ax_2a.set_ylabel("Pseudotime (normalised)")
ax_2a.set_ylim(-0.08, 1.12)
# title removed for CRM style
label_panel(ax_2a, "A")
save_fig(fig_2a, "figure_2a_pseudotime_violin")
plt.close(fig_2a)


# FIGURE 3D — Risk violin by timepoint

print("Building Figure 3D ...")
emb1 = pd.read_csv(RES / "bottleneck_s5_emb_norms_scale1_acutestress.csv")
risk_by_tp = {tp: emb1.loc[emb1["timepoint_h"]==float(tp), "risk_predicted"].values
              for tp in TP_ORDER}
mean_vals = {0:0.0077, 6:0.9173, 24:0.0473, 48:0.0304, 96:0.0448}

fig_3d, ax_3d = plt.subplots(figsize=(3.8, 2.9))
fig_3d.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.22)
tops_3d = draw_violin(ax_3d, risk_by_tp)

for pos, tp in enumerate(TP_ORDER):
    mu     = mean_vals[tp]
    top    = tops_3d[tp]
    # For 0h (blue, very flat): position label clearly above violin top
    # For 6h (red, tall): position at 6h median level, to the right
    # For others: above violin top + clearance
    if tp == 0:
        label_y = top + 0.025
    elif tp == 6:
        label_y = 0.970
    else:
        label_y = top + 0.018
    ax_3d.text(pos, label_y,
               f"\u03bc={mu:.3f}",
               ha="center", va="bottom", fontsize=4.8,
               color=TP_COL[tp], bbox=WBOX)

ax_3d.set_ylabel("Predicted risk score")
ax_3d.set_ylim(-0.05, 1.12)
# title removed for CRM style
label_panel(ax_3d, "D")
save_fig(fig_3d, "figure_3d_risk_violin")
plt.close(fig_3d)


# FIGURE 3E — Scale importance

print("Building Figure 3E ...")
si = pd.read_csv(RES / "week7_scale_importance.csv")

fig_3e, ax_3e = plt.subplots(figsize=(3.2, 2.6))
fig_3e.subplots_adjust(left=0.22, right=0.95, top=0.90, bottom=0.22)

for i, s in enumerate(SCALE_ORDER):
    row = si[si["scale"] == s].iloc[0]
    rho = float(row["spearman_vs_risk"])
    ax_3e.bar(i, rho, width=0.58, color=SCALE_COL[s],
              alpha=0.85, edgecolor="white", linewidth=0.4)
    offset = 0.02 if rho >= 0 else -0.02
    va     = "bottom" if rho >= 0 else "top"
    # ALL labels black for readability regardless of bar colour
    ax_3e.text(i, rho + offset, f"\u03c1={rho:+.3f}",
               ha="center", va=va, fontsize=6.5, color="black")

ax_3e.axhline(0, color="black", lw=0.75, alpha=0.6)
ax_3e.set_xticks(range(4))
ax_3e.set_xticklabels([SCALE_SHORT[s] for s in SCALE_ORDER], fontsize=6.5)
ax_3e.set_ylabel("Spearman \u03c1 with risk score")
ax_3e.set_ylim(-0.50, 0.50)
# title removed for CRM style
label_panel(ax_3e, "E")
save_fig(fig_3e, "figure_3e_scale_importance")
plt.close(fig_3e)

print("\nAll preprocessing panels complete.")
