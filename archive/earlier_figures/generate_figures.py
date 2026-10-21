#!/usr/bin/env python3
"""Earlier TrajRisk analysis retained for source comparison.

See docs/workflow.md for the study workflow and docs/reproduction.md for
the commands used to reproduce the reported analyses.
"""

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")


# PATHS

BASE = Path(__file__).resolve().parents[2]
RES  = BASE / "results"
FIGS = RES / "manuscript_figures"
FIGS.mkdir(parents=True, exist_ok=True)


# STYLE

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
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

COL = {
    "trajrisk_knn":   "#1A6EAF",
    "trajrisk_wself": "#6BAED6",
    "seeder":         "#E07B39",
    "b1_xgb":         "#9E9E9E",
    "APAP":           "#C0392B",
    "Control":        "#2E86C1",
    "APAP_ALF":       "#C0392B",
    "Healthy":        "#2E86C1",
}

# Method display labels — no bad unicode, no internal names
METHOD_LABELS = {
    "trajrisk_knn":   "TrajRisk (KNN)",
    "trajrisk_wself": "TrajRisk (self-only)",
    "seeder":         "Seeder",
    "b1_xgb":         "XGBoost",
}

# Methods for main figure (3 only) and supplementary (all 4)
METHODS_MAIN = ["trajrisk_knn", "seeder", "b1_xgb"]
METHODS_ALL  = ["trajrisk_knn", "trajrisk_wself", "seeder", "b1_xgb"]


def label_panel(ax, letter, x=-0.14, y=1.08, size=11):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="top", ha="left")


def fmt_p(p):
    if p < 0.001: return f"p = {p:.2e}"
    if p < 0.01:  return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def save_fig(fig, stem):
    for ext in [".pdf", ".svg", ".png"]:
        fig.savefig(str(FIGS / (stem + ext)))
    print(f"  Saved: {stem}.pdf / .svg / .png")



# FILE GUARD

REQUIRED = {
    "Mouse validation table":  RES / "week9a_mouse_validation_table.csv",
    "Mouse p21 analysis":      RES / "week9a_mouse_p21_analysis.csv",
    "Mouse cell scores":       RES / "manuscript_mouse_cell_scores.csv",
    "Human donor primary":     RES / "week9b_donor_primary.csv",
    "Human donor sensitivity": RES / "week9b_donor_sensitivity.csv",
    "Human composition":       RES / "manuscript_human_composition.csv",
    "Human summary JSON":      RES / "week9b_summary.json",
    "Human cell scores":       RES / "week9b_cell_risk_scores.csv",
    "Gene signature":          RES / "week9b_gene_signature.csv",
}
missing = [k for k, v in REQUIRED.items() if not v.exists()]
if missing:
    print("ERROR — required files missing:")
    for m in missing: print(f"  {m}")
    sys.exit(1)
print("All required files present.\n")


# LOAD DATA

val_tbl  = pd.read_csv(RES / "week9a_mouse_validation_table.csv")
p21_tbl  = pd.read_csv(RES / "week9a_mouse_p21_analysis.csv")
cell_df  = pd.read_csv(RES / "manuscript_mouse_cell_scores.csv")
don_pri  = pd.read_csv(RES / "week9b_donor_primary.csv")
don_sen  = pd.read_csv(RES / "week9b_donor_sensitivity.csv")
comp_df  = pd.read_csv(RES / "manuscript_human_composition.csv")
cell_hum = pd.read_csv(RES / "week9b_cell_risk_scores.csv")
sig_df   = pd.read_csv(RES / "week9b_gene_signature.csv")
with open(RES / "week9b_summary.json") as f:
    hum_sum = json.load(f)

# Clean display labels for test sets — no internal codes in axes
def display_test(t):
    t = str(t)
    if "Walesky"  in t:                          return "Walesky\n(internal ref.)"
    if "Ben-Moshe" in t:                          return "Ben-Moshe\n(recovery)"
    if "Umbaugh 2h" in t:                         return "Umbaugh 2h\n(early injury)"
    if "Cdkn1a"   in t:                           return "Cdkn1a\n(late cohort)"
    if "48h/72h"  in t and "APAP vs" in t:        return "Umbaugh\n48–72 h"
    return t[:30]

val_tbl["test_label"] = val_tbl["test"].apply(display_test)
tests_ordered = val_tbl["test_label"].unique().tolist()
n_tests       = len(tests_ordered)
x_pos         = np.arange(n_tests)
bar_w         = 0.18     # kept for supplementary compatibility
n_methods     = len(METHODS_ALL)



# FIGURE 1 — MOUSE EXTERNAL VALIDATION (3 panels)

print("Generating Figure 1 — Mouse external validation ...")

fig1 = plt.figure(figsize=(7.5, 5.6))
gs1  = GridSpec(2, 2, figure=fig1,
                height_ratios=[1.15, 1.0],
                hspace=0.52, wspace=0.42,
                left=0.10, right=0.97, top=0.86, bottom=0.10)

ax1A = fig1.add_subplot(gs1[0, :])    # full-width top
ax1B = fig1.add_subplot(gs1[1, 0])    # bottom-left
ax1C = fig1.add_subplot(gs1[1, 1])    # bottom-right

# Panel A: AUROC (3 methods, cleaner)
n_main  = len(METHODS_MAIN)
bar_w_m = 0.22
for mi, method in enumerate(METHODS_MAIN):
    sub = val_tbl[val_tbl["method"] == method]
    if sub.empty:
        continue
    sub    = sub.set_index("test_label").reindex(tests_ordered)
    auroc  = sub["auroc"].values.astype(float)
    err_lo = auroc - sub["auroc_ci_lo"].values.astype(float)
    err_hi = sub["auroc_ci_hi"].values.astype(float) - auroc
    offset = (mi - (n_main - 1) / 2) * bar_w_m
    ax1A.bar(x_pos + offset, auroc,
             width=bar_w_m * 0.85,
             color=COL.get(method, "#999"),
             label=METHOD_LABELS[method],
             zorder=3)
    ax1A.errorbar(x_pos + offset, auroc,
                  yerr=[err_lo, err_hi],
                  fmt="none", color="black",
                  linewidth=0.6, capsize=2.5, capthick=0.7, zorder=4)

ax1A.set_xticks(x_pos)
ax1A.set_xticklabels(tests_ordered, fontsize=7)
ax1A.set_ylabel("AUROC")
ax1A.set_ylim(0.55, 1.03)
ax1A.axhline(0.5, color="black", lw=0.5, ls="--", alpha=0.35, zorder=2)
ax1A.set_title("External validation performance", fontsize=8, pad=6)
ax1A.legend(loc="lower center", bbox_to_anchor=(0.5, 1.22), ncol=3,
            fontsize=7, frameon=False, handlelength=1.2,
            handletextpad=0.4, columnspacing=0.8)
label_panel(ax1A, "A", x=-0.05, y=1.10)

# Panel B: Umbaugh 2h score distributions
u2h = cell_df[cell_df["dataset"] == "Umbaugh_2h"].copy()
for i, grp in enumerate(["Control", "APAP"]):
    scores = u2h.loc[u2h["treatment"] == grp, "trajrisk_knn"].values
    parts  = ax1B.violinplot([scores], positions=[i],
                              widths=0.52, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(COL.get(grp, "#999"))
        pc.set_alpha(0.60)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    med = float(np.median(scores))
    q1  = float(np.percentile(scores, 25))
    q3  = float(np.percentile(scores, 75))
    ax1B.vlines(i, q1, q3, color="black", linewidth=1.8, zorder=4)
    ax1B.scatter([i], [med], color="white", edgecolors="black",
                 s=20, zorder=5, linewidths=0.8)

# AUROC annotation from official table
u2h_row = val_tbl[(val_tbl["method"] == "trajrisk_knn") &
                   val_tbl["test"].str.contains("Umbaugh 2h", na=False)]
if not u2h_row.empty:
    ax1B.text(0.98, 0.98,
              f"AUROC = {u2h_row['auroc'].values[0]:.3f}",
              transform=ax1B.transAxes, ha="right", va="top",
              fontsize=7, color=COL["trajrisk_knn"],
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

ax1B.set_xticks([0, 1])
ax1B.set_xticklabels(["Control", "APAP"])
ax1B.set_ylabel("TrajRisk score (KNN)")
ax1B.set_title("Early-injury score distribution", fontsize=8, pad=6)
ax1B.set_xlim(-0.6, 1.6)
label_panel(ax1B, "B")

# Panel C: Cdkn1a vs TrajRisk
u72 = cell_df[cell_df["dataset"] == "Umbaugh_48h72h"].copy()
u72 = u72.dropna(subset=["cdkn1a_expression", "trajrisk_wself"])
expr = u72["cdkn1a_expression"] > 0

ax1C.scatter(u72.loc[~expr, "cdkn1a_expression"],
             u72.loc[~expr, "trajrisk_wself"],
             c="#BDBDBD", s=1.5, alpha=0.20, linewidths=0,
             label="Cdkn1a = 0", rasterized=True)
ax1C.scatter(u72.loc[expr, "cdkn1a_expression"],
             u72.loc[expr, "trajrisk_wself"],
             c=COL["APAP"], s=1.5, alpha=0.15, linewidths=0,
             label="Cdkn1a > 0", rasterized=True)

# ρ from official p21 file
p21_ws = p21_tbl[p21_tbl["method"].str.lower().str.contains("wself|w_self")]
if not p21_ws.empty:
    rho_v = float(p21_ws["spearman_rho_cdkn1a"].values[0])
    ax1C.text(0.96, 0.95, f"\u03C1 = {rho_v:+.3f}",
              transform=ax1C.transAxes, ha="right", va="top", fontsize=7.5)

ax1C.set_xlabel("Cdkn1a expression")
ax1C.set_ylabel("Trajectory risk score")
ax1C.set_title("p21/Cdkn1a mechanistic validation", fontsize=8, pad=6)
ax1C.legend(loc="upper left", fontsize=6.5, markerscale=3,
            frameon=False, handletextpad=0.3)
label_panel(ax1C, "C")

save_fig(fig1, "figure_mouse_validation")
plt.close(fig1)



# FIGURE 2 — HUMAN CROSS-SPECIES VALIDATION (2-row layout)

print("Generating Figure 2 — Human cross-species validation ...")

fig2 = plt.figure(figsize=(7.5, 6.2))
gs2  = GridSpec(2, 3, figure=fig2,
                height_ratios=[0.68, 1.35],
                hspace=0.55, wspace=0.45,
                left=0.09, right=0.90, top=0.94, bottom=0.09)

ax2A = fig2.add_subplot(gs2[0, :])    # Panel A — full-width top row
ax2B = fig2.add_subplot(gs2[1, 0])    # Panel B
ax2C = fig2.add_subplot(gs2[1, 1])    # Panel C
ax2D = fig2.add_subplot(gs2[1, 2])    # Panel D

# Panel A: horizontal workflow schematic
ax2A.set_xlim(0, 100)
ax2A.set_ylim(0, 10)
ax2A.axis("off")

# Five boxes arranged left-to-right
box_x = [4, 24, 44, 64, 84]
box_y = 5.0
box_w = 14
box_h = 5.5
box_texts = [
    "Mouse\nhepatocytes",
    "Human hepatocyte\nnuclei",
    "Cross-species\nscVI integration",
    "TrajRisk\ntransfer",
    "Donor-level\nvalidation",
]
for bx, btxt in zip(box_x, box_texts):
    ax2A.text(bx + box_w/2, box_y, btxt,
              ha="center", va="center", fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.40",
                        facecolor="#EBF5FB", edgecolor="#2E86C1",
                        linewidth=0.80))

# Arrows between boxes
for i in range(len(box_x) - 1):
    x_start = box_x[i] + box_w + 0.3
    x_end   = box_x[i+1] - 0.8
    ax2A.annotate("", xy=(x_end, box_y), xytext=(x_start, box_y),
                  arrowprops=dict(arrowstyle="-|>", color="#444444",
                                  lw=1.2, mutation_scale=10))

ax2A.set_title("Cross-species validation workflow", fontsize=8, pad=6)
label_panel(ax2A, "A", x=-0.02, y=1.18)

# Shared donor strip helper
def donor_strip(ax, df, score_col, title, json_scope):
    for ci, cond in enumerate(["Healthy", "APAP_ALF"]):
        sub = df[df["condition"] == cond][score_col].values
        rng = np.random.default_rng(seed=42 + ci)
        jit = rng.uniform(-0.10, 0.10, size=len(sub))
        ax.scatter(np.full(len(sub), ci) + jit, sub,
                   c=COL[cond], s=30, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=0.5)
        q1, med, q3 = (float(np.percentile(sub, q)) for q in [25, 50, 75])
        ax.vlines(ci, q1, q3, color=COL[cond], linewidth=3.0, zorder=3)
        ax.scatter([ci], [med], color="white", edgecolors=COL[cond],
                   s=26, zorder=5, linewidths=1.4)

    # Significance annotation — upper left, white background, no bracket
    stats = hum_sum.get(json_scope, {})
    pval  = stats.get(f"{score_col}_mw_pval")
    auroc = stats.get(f"{score_col}_auroc")
    if pval is not None and auroc is not None:
        ax.text(0.04, 0.88,
                f"AUROC = {auroc:.3f}\n{fmt_p(pval)}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, linespacing=1.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Healthy", "APAP-ALF"], fontsize=7.5)
    ax.set_ylabel("Donor mean risk score")
    ax.set_title(title, fontsize=8, pad=6)
    ax.set_xlim(-0.55, 1.55)

# Panel B: primary donor analysis
donor_strip(ax2B, don_pri, "risk_2a", "Primary analysis", "primary")
label_panel(ax2B, "B")

# Panel C: sensitivity analysis
donor_strip(ax2C, don_sen, "risk_2a", "Sensitivity analysis", "sensitivity")
label_panel(ax2C, "C")

# Sync y-limits so B and C are visually comparable
_ylo = min(ax2B.get_ylim()[0], ax2C.get_ylim()[0])
_yhi = max(ax2B.get_ylim()[1], ax2C.get_ylim()[1])
ax2B.set_ylim(_ylo, _yhi)
ax2C.set_ylim(_ylo, _yhi)

# Panel D: convergence scatter
for cond in ["Healthy", "APAP_ALF"]:
    sub = don_pri[don_pri["condition"] == cond]
    ax2D.scatter(sub["sig_net"], sub["risk_2b"],
                 c=COL[cond], s=34, zorder=4,
                 label=cond.replace("_", "-"),
                 edgecolors="white", linewidths=0.5)

conv_rho = hum_sum.get("primary", {}).get("convergence_rho")
conv_p   = hum_sum.get("primary", {}).get("convergence_p")
if conv_rho is not None and conv_p is not None:
    ax2D.text(0.05, 0.96,
              f"\u03C1 = {conv_rho:+.3f}\n{fmt_p(conv_p)}",
              transform=ax2D.transAxes, ha="left", va="top",
              fontsize=7, linespacing=1.5)

ax2D.set_xlabel("Net signature score\n(risk \u2212 recovery)")
ax2D.set_ylabel("TrajRisk score\n(KNN, donor mean)")
ax2D.set_title("Risk vs gene signature", fontsize=8, pad=6)
ax2D.legend(loc="upper left", bbox_to_anchor=(1.02, 0.92),
            fontsize=6.5, frameon=False,
            markerscale=1.0, handletextpad=0.3, borderaxespad=0)
label_panel(ax2D, "D")

save_fig(fig2, "figure_human_validation")
plt.close(fig2)



# SUPPLEMENTARY S1 — FULL FOUR-METHOD AUROC

print("Generating Supp S1 — Full four-method AUROC ...")

fig_s1, ax_s1 = plt.subplots(figsize=(7.5, 3.8))
fig_s1.subplots_adjust(left=0.10, right=0.97, top=0.78, bottom=0.18)

for mi, method in enumerate(METHODS_ALL):
    sub = val_tbl[val_tbl["method"] == method]
    if sub.empty: continue
    sub    = sub.set_index("test_label").reindex(tests_ordered)
    auroc  = sub["auroc"].values.astype(float)
    err_lo = auroc - sub["auroc_ci_lo"].values.astype(float)
    err_hi = sub["auroc_ci_hi"].values.astype(float) - auroc
    offset = (mi - (n_methods - 1) / 2) * bar_w
    ax_s1.bar(x_pos + offset, auroc,
              width=bar_w * 0.85,
              color=COL.get(method, "#999"),
              label=METHOD_LABELS[method], zorder=3)
    ax_s1.errorbar(x_pos + offset, auroc,
                   yerr=[err_lo, err_hi],
                   fmt="none", color="black",
                   linewidth=0.6, capsize=2, capthick=0.6, zorder=4)

ax_s1.set_xticks(x_pos)
ax_s1.set_xticklabels(tests_ordered, fontsize=7)
ax_s1.set_ylabel("AUROC")
ax_s1.set_ylim(0.55, 1.03)
ax_s1.axhline(0.5, color="black", lw=0.5, ls="--", alpha=0.35)
ax_s1.set_title("Cross-dataset validation — AUROC (all methods)", fontsize=8)
ax_s1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.30), ncol=2,
             fontsize=7, frameon=False, handlelength=1.2, columnspacing=0.8)

save_fig(fig_s1, "supp_figure_auroc_full")
plt.close(fig_s1)



# SUPPLEMENTARY S2 — AUPRC SUMMARY

print("Generating Supp S2 — AUPRC ...")

fig_s2, ax_s2 = plt.subplots(figsize=(7.5, 3.8))
fig_s2.subplots_adjust(left=0.10, right=0.97, top=0.78, bottom=0.18)

for mi, method in enumerate(METHODS_ALL):
    sub = val_tbl[val_tbl["method"] == method]
    if sub.empty: continue
    sub    = sub.set_index("test_label").reindex(tests_ordered)
    auprc  = sub["auprc"].values.astype(float)
    offset = (mi - (n_methods - 1) / 2) * bar_w
    ax_s2.bar(x_pos + offset, auprc,
              width=bar_w * 0.85,
              color=COL.get(method, "#999"),
              label=METHOD_LABELS[method], zorder=3, alpha=0.90)

rand_vals = val_tbl.groupby("test_label")["auprc_rand"].first().reindex(tests_ordered)
ax_s2.scatter(x_pos, rand_vals.values, marker="D",
              c="black", s=16, zorder=5, label="Random baseline", clip_on=False)

ax_s2.set_xticks(x_pos)
ax_s2.set_xticklabels(tests_ordered, fontsize=7)
ax_s2.set_ylabel("AUPRC")
ax_s2.set_ylim(0, 1.05)
ax_s2.set_title("Cross-dataset validation — AUPRC (all methods)", fontsize=8)
ax_s2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.30), ncol=3,
             fontsize=7, frameon=False, handlelength=1.2, columnspacing=0.8)

save_fig(fig_s2, "supp_figure_auprc")
plt.close(fig_s2)



# SUPPLEMENTARY S3 — NEIGHBOUR COMPOSITION

print("Generating Supp S3 — Neighbour composition ...")

fig_s3, ax_s3 = plt.subplots(figsize=(4.0, 3.4))
fig_s3.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.15)

tp_cols  = ["pct_0h", "pct_6h", "pct_24h", "pct_48h", "pct_96h"]
tp_labels = ["0h", "6h", "24h", "48h", "96h"]
bar_w3   = 0.33
x3       = np.arange(len(tp_labels))

cond_vals = {}
for cond in ["APAP_ALF", "Healthy"]:
    sub = comp_df[comp_df["condition"] == cond]
    cond_vals[cond] = sub[tp_cols].mean().values

offsets3 = {"APAP_ALF": -bar_w3/2, "Healthy": bar_w3/2}
clabels3 = {"APAP_ALF": "APAP-ALF", "Healthy": "Healthy"}

for cond, vals in cond_vals.items():
    ax_s3.bar(x3 + offsets3[cond], vals,
              width=bar_w3 * 0.88,
              color=COL[cond], label=clabels3[cond],
              alpha=0.85, edgecolor="white", linewidth=0.3)

ax_s3.set_xticks(x3)
ax_s3.set_xticklabels(tp_labels)
ax_s3.set_xlabel("Nearest Walesky-neighbour timepoint")
ax_s3.set_ylabel("Mean fraction of 15 neighbours")
ax_s3.set_title("Walesky-neighbour composition", fontsize=8)
ax_s3.legend(fontsize=7, frameon=False, loc="upper right")
ax_s3.set_ylim(0, 0.50)

save_fig(fig_s3, "supp_figure_composition")
plt.close(fig_s3)



# SUPPLEMENTARY S4 — GENE SIGNATURE ANALYSIS

print("Generating Supp S4 — Gene signature analysis ...")

fig_s4, axes_s4 = plt.subplots(1, 2, figsize=(8.2, 3.8),
                                 gridspec_kw={"wspace": 0.85})
fig_s4.subplots_adjust(left=0.12, right=0.92, top=0.90, bottom=0.12)

# Left: top risk and recovery genes
ax_s4l = axes_s4[0]
risk_g  = sig_df[sig_df["signature"] == "risk"].nlargest(10, "spearman_rho")
recov_g = sig_df[sig_df["signature"] == "recovery"].nsmallest(10, "spearman_rho")
genes_all = pd.concat([risk_g[["human_gene","spearman_rho"]],
                        recov_g[["human_gene","spearman_rho"]].iloc[::-1]],
                       ignore_index=True)
colours_g = [COL["APAP"] if r > 0 else COL["Healthy"]
             for r in genes_all["spearman_rho"].values]
yg = np.arange(len(genes_all))
ax_s4l.barh(yg, genes_all["spearman_rho"].values,
            color=colours_g, height=0.70,
            edgecolor="white", linewidth=0.25)
ax_s4l.set_yticks(yg)
ax_s4l.set_yticklabels(genes_all["human_gene"].values, fontsize=6.5)
ax_s4l.axvline(0, color="black", lw=0.6)
ax_s4l.set_xlabel("Spearman \u03C1 (pseudotime correlation)")
ax_s4l.set_title("Risk and recovery genes", fontsize=8)
ax_s4l.legend(handles=[
    mpatches.Patch(color=COL["APAP"],    label="Risk (positive \u03C1)"),
    mpatches.Patch(color=COL["Healthy"], label="Recovery (negative \u03C1)"),
], fontsize=6.5, frameon=False,
    loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2)
ax_s4l.text(-0.22, 1.06, "A", transform=ax_s4l.transAxes,
            fontsize=11, fontweight="bold", va="top")

# Right: donor sig_risk (honest display of direction)
ax_s4r = axes_s4[1]
for ci, cond in enumerate(["Healthy", "APAP_ALF"]):
    sub = don_pri[don_pri["condition"] == cond]["sig_risk"].values
    rng = np.random.default_rng(seed=7 + ci)
    jit = rng.uniform(-0.09, 0.09, size=len(sub))
    ax_s4r.scatter(np.full(len(sub), ci) + jit, sub,
                   c=COL[cond], s=30, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=0.5)
    q1, med, q3 = (float(np.percentile(sub, q)) for q in [25, 50, 75])
    ax_s4r.vlines(ci, q1, q3, color=COL[cond], linewidth=3.0, zorder=3)
    ax_s4r.scatter([ci], [med], color="white", edgecolors=COL[cond],
                   s=26, zorder=5, linewidths=1.4)
ax_s4r.set_xticks([0, 1])
ax_s4r.set_xticklabels(["Healthy", "APAP-ALF"])
ax_s4r.set_ylabel("Risk signature score")
ax_s4r.set_title("Risk signature by condition\n(includes hepatocyte identity genes)", fontsize=7.5)
ax_s4r.text(-0.22, 1.06, "B", transform=ax_s4r.transAxes,
            fontsize=11, fontweight="bold", va="top")

save_fig(fig_s4, "supp_figure_signature")
plt.close(fig_s4)



# SUPPLEMENTARY S5 — CELL-LEVEL HUMAN DISTRIBUTIONS

print("Generating Supp S5 — Cell-level distributions ...")

fig_s5, axes_s5 = plt.subplots(1, 2, figsize=(5.5, 3.8),
                                 gridspec_kw={"wspace": 0.48})
fig_s5.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)

score_panels = [("risk_2a", "Self-only transfer", "A"),
                ("risk_2b", "Reference-KNN transfer", "B")]

for ax_s5, (col, subtitle, plabel) in zip(axes_s5, score_panels):
    conds     = ["Healthy", "APAP_ALF"]
    data_list = [cell_hum.loc[cell_hum["condition"] == c, col].values for c in conds]
    parts     = ax_s5.violinplot(data_list, positions=[0, 1],
                                  widths=0.55, showmedians=False, showextrema=False)
    for pi, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COL[conds[pi]])
        pc.set_alpha(0.58)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.6)
    for pi, vals in enumerate(data_list):
        med = float(np.median(vals))
        q1  = float(np.percentile(vals, 25))
        q3  = float(np.percentile(vals, 75))
        ax_s5.vlines(pi, q1, q3, color="black", linewidth=1.5, zorder=4)
        ax_s5.scatter([pi], [med], color="white", edgecolors="black",
                      s=16, zorder=5, linewidths=0.8)
    n_h = int((cell_hum["condition"] == "Healthy").sum())
    n_a = int((cell_hum["condition"] == "APAP_ALF").sum())
    ax_s5.set_xticks([0, 1])
    ax_s5.set_xticklabels([f"Healthy\n(n={n_h:,})", f"APAP-ALF\n(n={n_a:,})"], fontsize=6.5)
    ax_s5.set_ylabel("Cell trajectory risk score")
    ax_s5.set_title(subtitle, fontsize=8)
    ax_s5.text(-0.22, 1.06, plabel, transform=ax_s5.transAxes,
               fontsize=11, fontweight="bold", va="top")

fig_s5.text(0.50, -0.02,
            "Cells from the same donor are not independent observations. "
            "Donor-level statistics are in the main text.",
            ha="center", va="top", fontsize=6.5, color="#555", style="italic")

save_fig(fig_s5, "supp_figure_celllevel")
plt.close(fig_s5)



# DONE

print("\n" + "=" * 60)
print("All figures saved to:")
print(f"  {FIGS}")
print()
for f in sorted(FIGS.glob("*")):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
