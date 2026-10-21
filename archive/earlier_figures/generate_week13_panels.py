#!/usr/bin/env python3
"""Earlier TrajRisk analysis retained for source comparison.

See docs/workflow.md for the study workflow and docs/reproduction.md for
the commands used to reproduce the reported analyses.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RES  = BASE / "results"
FIGS = RES / "manuscript_figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Shared style
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
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "legend.frameon":     False,
})

# Shared colour palette — identical hex codes used everywhere
COL = {
    "s1":      "#C0392B",   # Scale 1
    "s2":      "#E67E22",   # Scale 2
    "s3":      "#27AE60",   # Scale 3
    "s4":      "#2980B9",   # Scale 4
    "trajrisk":"#1F77B4",   # TrajRisk blue
    "baseline": "#9E9E9E",  # Baseline grey
    "ablation": "#C0603A",  # Ablation orange-red
    "seeder":   "#E67E22",  # Seeder (same as S2 orange)
}

SCALE_ORDER = ["Scale1_AcuteStress","Scale2_PeakInjury",
               "Scale3_FateDecision","Scale4_Resolution"]
SCALE_COL   = {
    "Scale1_AcuteStress":  COL["s1"],
    "Scale2_PeakInjury":   COL["s2"],
    "Scale3_FateDecision": COL["s3"],
    "Scale4_Resolution":   COL["s4"],
}
SCALE_SHORT = {
    "Scale1_AcuteStress":  "S1\n(0h\u21926h)",
    "Scale2_PeakInjury":   "S2\n(6h\u219224h)",
    "Scale3_FateDecision": "S3\n(24h\u219248h)",
    "Scale4_Resolution":   "S4\n(48h\u219296h)",
}

# Panel label — IDENTICAL across all figures
def label_panel(ax, letter, x=-0.16, y=1.09):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            va="top", ha="left", fontfamily="Arial")

def save_fig(fig, stem):
    for ext in [".pdf", ".svg", ".png"]:
        fig.savefig(str(FIGS / (stem + ext)))
    print(f"  Saved: {stem}.pdf / .svg / .png")


# LOAD DATA

with open(RES / "week8_summary.json") as f:
    w8 = json.load(f)
all_methods = pd.DataFrame(w8["results"])

supp = pd.read_csv(RES / "week8_supplementary_table.csv")
cons = pd.read_csv(RES / "week12_task1_scale_summary.csv")
ent  = pd.read_csv(RES / "bottleneck_s1_scale_summary.csv")

with open(RES / "week12_crossspecies_summary.json") as f:
    cs = json.load(f)
overall_pct = float(cs["pct_conserved_p05"])  # 40.9

# TrajRisk reference
TR_AUROC = float(all_methods[all_methods["Method"]=="TrajRisk (full)"]["AUROC"].values[0])
TR_CI_LO = float(all_methods[all_methods["Method"]=="TrajRisk (full)"]["AUROC_CI_lower"].values[0])
TR_CI_HI = float(all_methods[all_methods["Method"]=="TrajRisk (full)"]["AUROC_CI_upper"].values[0])

XLIM = (0.75, 1.005)

def draw_horizontal_bars(ax, method_names, y_labels, bar_col,
                         title, panel_letter, show_xlab=True):
    """
    Clean horizontal bar chart for one group (baselines OR ablations).
    TrajRisk reference line drawn at TR_AUROC.
    No floating group labels inside the plot.
    """
    n     = len(method_names)
    y_pos = np.arange(n)[::-1]  # top = index 0

    for i, mname in enumerate(method_names):
        row = all_methods[all_methods["Method"] == mname]
        if len(row) == 0:
            continue
        auroc = float(row["AUROC"].values[0])
        ci_lo = float(row["AUROC_CI_lower"].values[0])
        ci_hi = float(row["AUROC_CI_upper"].values[0])
        y     = y_pos[i]

        ax.barh(y, auroc, height=0.5,
                color=bar_col, alpha=0.72,
                edgecolor="white", linewidth=0.3)
        # CI whisker
        ax.plot([ci_lo, ci_hi], [y, y],
                color="#222222", lw=0.9, zorder=5,
                solid_capstyle="round")
        ax.scatter([ci_lo, ci_hi], [y, y],
                   color="#222222", s=5, zorder=6, linewidths=0)

    # TrajRisk reference — thick, clearly visible
    ax.axvline(TR_AUROC, color=COL["trajrisk"],
               lw=1.4, ls="--", alpha=0.85, zorder=0)
    # Shaded CI band
    ax.axvspan(TR_CI_LO, TR_CI_HI,
               color=COL["trajrisk"], alpha=0.10, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=6.5)
    ax.set_xlim(XLIM)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_title(title, fontsize=8, pad=6)
    if show_xlab:
        ax.set_xlabel("AUROC (validation, n=1,318)", fontsize=7.5)
    label_panel(ax, panel_letter)



# FIGURE 4 — Three panels: A (baselines), B (ablations), C (within-TP AUPRC)

print("Building Figure 4 ...")

BASELINE_METHODS = [
    "B1: XGBoost (CV-tuned)",
    "B2: Gaussian propagation",
    "B3/A2: Single-scale GNN (best)",
    "B4: Expression distance",
]
BASE_LABELS = [
    "B1: XGBoost\u00b9",        # superscript 1 for footnote
    "B2: Gaussian\npropagation",
    "B3: Single-scale\nGNN (best)",
    "B4: Expression\ndistance",
]

ABLATION_METHODS = [
    "A1: Gaussian transitions",
    "A3: RF matrix propagation",
    "A4: No XGBoost seed",
]
ABL_LABELS = [
    "A1: Gaussian\ntransitions",
    "A3: RF matrix\npropagation",
    "A4: No XGBoost\nseed",
]

SUBSET_ORDER = ["Non-6h", "24h", "48h", "96h"]
supp_filt    = supp[supp["subset"].isin(SUBSET_ORDER)].copy()

fig4 = plt.figure(figsize=(7.2, 3.8))
gs4  = gridspec.GridSpec(
    1, 3, figure=fig4,
    left=0.03, right=0.98,
    top=0.88, bottom=0.25,
    wspace=0.58,
    width_ratios=[1.25, 0.92, 1.35],
)
ax4A = fig4.add_subplot(gs4[0, 0])
ax4B = fig4.add_subplot(gs4[0, 1])
ax4C = fig4.add_subplot(gs4[0, 2])

# Panel A — Baselines
draw_horizontal_bars(ax4A, BASELINE_METHODS, BASE_LABELS,
                     COL["baseline"], "vs. Baselines", "A")

# B1 footnote — placed BELOW the axes, never overlapping bars
ax4A.annotate(
    "\u00b9 B1 AUROC reflects partial circularity: scVI features \u2248 pseudotime label",
    xy=(0, 0), xycoords="axes fraction",
    xytext=(0, -0.30), textcoords="axes fraction",
    ha="left", va="top", fontsize=4.8,
    color="#888888", style="italic",
)

# TrajRisk label on reference line — positioned above all bars
ax4A.text(TR_AUROC + 0.002, 3.38,
          f"TrajRisk = {TR_AUROC:.3f}",
          ha="left", va="top", fontsize=5.2,
          color=COL["trajrisk"], style="italic")

# Panel B — Ablations
draw_horizontal_bars(ax4B, ABLATION_METHODS, ABL_LABELS,
                     COL["ablation"], "vs. Ablations", "B")

ax4B.text(TR_AUROC + 0.002, 2.38,
          f"TrajRisk = {TR_AUROC:.3f}",
          ha="left", va="top", fontsize=5.2,
          color=COL["trajrisk"], style="italic")

# Panel C — Within-timepoint AUPRC
method_specs = {
    "TrajRisk":    (COL["trajrisk"], 0.85, "TrajRisk"),
    "Seeder (W6)": (COL["seeder"],   0.80, "Seeder"),
    "B1 XGBoost":  (COL["baseline"], 0.72, "XGBoost"),
}
bar_w = 0.22
x_sub = np.arange(len(SUBSET_ORDER))
x_labels_C = ["Non-6h\n(n=1,234)", "24h\n(n=209)", "48h\n(n=182)", "96h\n(n=162)"]

for si, subset in enumerate(SUBSET_ORDER):
    sub_data = supp_filt[supp_filt["subset"] == subset]
    rand_bl  = float(sub_data["auprc_random"].iloc[0])

    for mi, (mname, (col, alpha, _)) in enumerate(method_specs.items()):
        row_m = sub_data[sub_data["method"] == mname]
        if len(row_m) == 0:
            continue
        auprc  = float(row_m["auprc"].values[0])
        offset = (mi - 1) * bar_w
        bar    = ax4C.bar(si + offset, auprc, width=bar_w,
                          color=col, alpha=alpha, linewidth=0)
        # Value on top of each bar
        ax4C.text(si + offset, auprc + 0.018,
                  f"{auprc:.2f}",
                  ha="center", va="bottom",
                  fontsize=4.2, color="#333333")

    # Per-subset random baseline
    ax4C.plot([si - bar_w*1.9, si + bar_w*1.9],
              [rand_bl, rand_bl],
              color="black", lw=0.8, ls="--",
              alpha=0.55, zorder=3)

ax4C.set_xticks(x_sub)
ax4C.set_xticklabels(x_labels_C, fontsize=6.5)
ax4C.set_xlabel("Validation subset")
ax4C.set_ylabel("AUPRC")
ax4C.set_ylim(0, 0.75)
ax4C.set_title("Within-timepoint discrimination", fontsize=8, pad=6)

leg_C = [mpatches.Patch(color=c, alpha=a, label=lbl)
         for _, (c, a, lbl) in method_specs.items()]
leg_C.append(plt.Line2D([0],[0], color="black", lw=0.8,
                         ls="--", label="Random (prevalence)"))
ax4C.legend(handles=leg_C, fontsize=5.5, loc="upper left",
            handlelength=1.0, handletextpad=0.4, borderpad=0.4)
label_panel(ax4C, "C", x=-0.24)

save_fig(fig4, "figure_4_benchmarking")
plt.close(fig4)


# FIGURE 6E — Cross-species conservation (fixed label collision)

print("Building Figure 6E ...")

fig_e, ax_e = plt.subplots(figsize=(3.2, 2.8))
fig_e.subplots_adjust(left=0.22, right=0.88, top=0.89, bottom=0.22)

x_pos = np.arange(4)
pct_vals = []

for i, s in enumerate(SCALE_ORDER):
    row = cons[cons["scale"] == s].iloc[0]
    pct = float(row["pct_conserved"])
    n_t = int(row["n_human_tested"])
    pct_vals.append(pct)
    alpha = 0.60 if s == "Scale3_FateDecision" else 0.85

    ax_e.bar(i, pct, width=0.58,
             color=SCALE_COL[s], alpha=alpha,
             edgecolor="white", linewidth=0.4)

    # n= label — bold, positioned cleanly above bar
    ax_e.text(i, pct + 4.2,
              f"n={n_t:,}",
              ha="center", va="bottom",
              fontsize=5.0, fontweight="bold",
              color=SCALE_COL[s])

# Overall mean dashed line
ax_e.plot([-0.4, 3.35], [overall_pct, overall_pct],
          color="#555555", lw=0.75, ls="--", alpha=0.65, zorder=0)

# "40.9%" label — placed OUTSIDE the bar area to the right of the axis
ax_e.text(4.05, overall_pct,
          f"{overall_pct}%\n(mean)",
          ha="left", va="center",
          fontsize=5.2, color="#555555")

# Scale 3 annotation — arrow from text below to bar top, no collision
s3_pct = float(cons[cons["scale"]=="Scale3_FateDecision"]["pct_conserved"].values[0])
ax_e.annotate(
    "Temporal\nmismatch\n(expected)",
    xy=(2, s3_pct - 0.5),
    xytext=(1.5, 13),
    ha="center", va="top",
    fontsize=5.0, color=SCALE_COL["Scale3_FateDecision"],
    bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9),
    arrowprops=dict(
        arrowstyle="-|>",
        color=SCALE_COL["Scale3_FateDecision"],
        lw=0.65,
        shrinkA=2, shrinkB=2,
    ),
)

ax_e.set_xticks(x_pos)
ax_e.set_xticklabels([SCALE_SHORT[s] for s in SCALE_ORDER], fontsize=6.5)
ax_e.set_ylabel("Bottleneck genes\nsignificant in human (%)", fontsize=7.5)
ax_e.set_ylim(0, 66)
ax_e.set_xlim(-0.55, 3.8)
ax_e.set_title("Cross-species conservation", fontsize=8, pad=5)
label_panel(ax_e, "E")

save_fig(fig_e, "figure_6e_conservation")
plt.close(fig_e)


# FIGURE 2D — Entropy (minor fixes: consistent label size, value placement)

print("Building Figure 2D ...")

fig_d, ax_d = plt.subplots(figsize=(3.0, 2.6))
fig_d.subplots_adjust(left=0.26, right=0.95, top=0.89, bottom=0.22)

for i, s in enumerate(SCALE_ORDER):
    row = ent[ent["scale"] == s].iloc[0]
    val = float(row["entropy_nats"])

    ax_d.bar(i, val, width=0.58,
             color=SCALE_COL[s], alpha=0.85,
             edgecolor="white", linewidth=0.4)
    # Value label — font size matches tick labels (7pt)
    ax_d.text(i, val + 0.04, f"{val:.2f}",
              ha="center", va="bottom",
              fontsize=7, color=SCALE_COL[s])

ax_d.set_xticks(np.arange(4))
ax_d.set_xticklabels([SCALE_SHORT[s] for s in SCALE_ORDER], fontsize=6.5)
ax_d.set_ylabel("Transition entropy (nats)", fontsize=7.5)
ax_d.set_ylim(0, 5.65)
ax_d.set_title("Scale transition entropy", fontsize=8, pad=5)
label_panel(ax_d, "D")

save_fig(fig_d, "figure_2d_entropy")
plt.close(fig_d)

print("\nAll panels complete.")
print(f"Output: {FIGS}")
