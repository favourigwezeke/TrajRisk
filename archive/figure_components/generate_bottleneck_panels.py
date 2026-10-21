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
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
import scanpy as sc
from scipy.stats import fisher_exact, gaussian_kde
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
RES  = BASE / "results"
FIGS = RES / "manuscript_figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75, "ytick.major.width": 0.75,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False,
})

SCALE_COL = {
    "Scale1_AcuteStress":   "#C0392B",
    "Scale2_PeakInjury":    "#E67E22",
    "Scale3_FateDecision":  "#27AE60",
    "Scale4_Resolution":    "#2980B9",
}
SCALE_SHORT = {
    "Scale1_AcuteStress":  "Scale 1 (0h->6h)",
    "Scale2_PeakInjury":   "Scale 2 (6h->24h)",
    "Scale3_FateDecision": "Scale 3 (24h->48h)",
    "Scale4_Resolution":   "Scale 4 (48h->96h)",
}
SCALE_NAMES = list(SCALE_COL.keys())

def label_panel(ax, letter, x=-0.14, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

def annot_box(ax, text, loc="upper right", fontsize=7):
    xp = 0.96 if "right" in loc else 0.04
    ha = "right" if "right" in loc else "left"
    yp = 0.96 if "upper" in loc else 0.04
    va = "top" if "upper" in loc else "bottom"
    ax.text(xp, yp, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))

def save_fig(fig, stem):
    for ext in [".pdf", ".svg", ".png"]:
        fig.savefig(str(FIGS / (stem + ext)))
    print(f"  Saved: {stem}.pdf / .svg / .png")

# Load
print("Loading data ...")
edges    = pd.read_csv(RES / "bottleneck_s1_all_edges.csv")
ablation = pd.read_csv(RES / "bottleneck_s2_scale_ablation.csv")
quartile = pd.read_csv(RES / "bottleneck_s2_quartile_perturbation.csv")
shift_df = pd.read_csv(RES / "bottleneck_s5_all_edges_with_shift.csv")

adata       = sc.read_h5ad(
    str(BASE / "Data/processed/adata_integrated_with_trajectory.h5ad"), backed="r")
walesky_obs = adata.obs[adata.obs["dataset"] == "Walesky"].copy()
adata.file.close()
ptime     = walesky_obs["pseudotime_norm"].values.astype(np.float32)
risk      = walesky_obs["risk_predicted"].values.astype(np.float32)
timepoint = walesky_obs["timepoint_hours"].values.astype(np.float32)
print(f"  Edges: {len(edges):,}")

# Figure layout
print("Building figure ...")
fig = plt.figure(figsize=(7.2, 7.8))

# Outer 3x2 grid
outer = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.54, wspace=0.42,
                           left=0.10, right=0.97,
                           top=0.94, bottom=0.07)

ax_A = fig.add_subplot(outer[0, 0])

# Panel B: PROPER ridge plot — 4 real axes from inner GridSpec
inner_B = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0, 1], hspace=0.06)
r_ax = [fig.add_subplot(inner_B[i]) for i in range(4)]
# r_ax[0] = top, r_ax[3] = bottom (GridSpec top-to-bottom)
# Biological order top-to-bottom: S1, S2, S3, S4
RIDGE_SCALES = ["Scale1_AcuteStress","Scale2_PeakInjury",
                "Scale3_FateDecision","Scale4_Resolution"]

ax_C = fig.add_subplot(outer[1, 0])
ax_D = fig.add_subplot(outer[1, 1])
ax_E = fig.add_subplot(outer[2, 0])
ax_F = fig.add_subplot(outer[2, 1])

# A: Risk flux histogram
for s_name in SCALE_NAMES:
    vals = edges[edges["scale"] == s_name]["flux"].values
    vals = vals[vals > 0]
    ax_A.hist(np.log10(vals), bins=80, density=True,
              color=SCALE_COL[s_name], alpha=0.55, linewidth=0)
handles_A = [mpatches.Patch(color=SCALE_COL[s], alpha=0.75, label=SCALE_SHORT[s])
             for s in SCALE_NAMES]
ax_A.legend(handles=handles_A, fontsize=6, loc="upper left",
            handlelength=1.0, handletextpad=0.3)
ax_A.set_xlabel("log10(risk flux)")
ax_A.set_ylabel("Density")
ax_A.set_title("Risk flux distribution", fontsize=8, pad=5)
label_panel(ax_A, "A")

# B: Ridge plot — 4 proper axes
BW = 0.30
pt_grid = np.linspace(-0.02, 1.02, 400)

for ri, s_name in enumerate(RIDGE_SCALES):
    ax_r = r_ax[ri]
    sub  = edges[(edges["scale"] == s_name) &
                 (edges["flux_pct_rank"] >= 99.0)]["pt_source"].values
    col  = SCALE_COL[s_name]

    if len(sub) >= 5:
        try:
            kde  = gaussian_kde(sub, bw_method=BW)
            dens = kde(pt_grid)
            dens = dens / dens.max()
        except Exception:
            dens = np.zeros_like(pt_grid)
        ax_r.fill_between(pt_grid, dens, alpha=0.50, color=col)
        ax_r.plot(pt_grid, dens, color=col, lw=1.5)

    ax_r.set_xlim(-0.02, 1.02)
    ax_r.set_ylim(0, 1.30)
    ax_r.set_yticks([])
    ax_r.spines["left"].set_visible(False)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    # Scale label on right side
    s_label = "S" + s_name.replace("Scale","").split("_")[0]
    ax_r.text(1.01, 0.5, s_label, transform=ax_r.transAxes,
              ha="left", va="center", fontsize=6.5,
              color=col, fontweight="bold")

    if ri < 3:  # only bottom axis gets x-axis
        ax_r.set_xticks([])
    else:
        ax_r.set_xlabel("Source cell pseudotime (normalised)", fontsize=7)
        ax_r.tick_params(axis="x", labelsize=6)

# Panel B label on top subplot
label_panel(r_ax[0], "B", x=-0.08, y=1.14)
r_ax[0].set_title("Pseudotime of top-1% flux source cells",
                   fontsize=8, pad=4)

# C: Fate commitment test
sub3 = edges[edges["scale"] == "Scale3_FateDecision"]
top3 = sub3[sub3["flux_pct_rank"] >= 99.0]
src3 = top3["source_node"].unique().astype(int)
tgt3 = top3["target_node"].unique().astype(int)
mask_24h = timepoint == 24.0
mask_48h = timepoint == 48.0
all_24h  = np.where(mask_24h)[0]
all_48h  = np.where(mask_48h)[0]
q75_24h  = float(np.percentile(ptime[mask_24h], 75))
q75_48h  = float(np.percentile(ptime[mask_48h], 75))

pct_src  = float((ptime[src3] >= q75_24h).mean() * 100)
pct_tgt  = float((ptime[tgt3] >= q75_48h).mean() * 100)

a = int((ptime[src3] >= q75_24h).sum())
b_val = len(src3) - a
c = max(int((ptime[all_24h] >= q75_24h).sum()) - a, 0)
d_val = len(all_24h) - len(src3) - c
_, p_src = fisher_exact([[a, b_val], [c, d_val]], alternative="greater")
or_denom = max(c / max(d_val, 1), 1e-9)
or_src   = (a / max(b_val, 1)) / or_denom if b_val > 0 else float('inf')

ax_C.bar([0, 1, 2], [pct_src, 25.0, pct_tgt], width=0.55,
         color=[SCALE_COL["Scale3_FateDecision"], "#BBBBBB", "#2980B9"],
         alpha=0.82, edgecolor="white", linewidth=0.4)
ax_C.axhline(25.0, color="black", lw=0.6, ls="--", alpha=0.4)
ax_C.set_xticks([0, 1, 2])
ax_C.set_xticklabels([f"Sources\n(n={len(src3)})",
                      "Null\nexpectation",
                      f"Targets\n(n={len(tgt3)})"], fontsize=6.5)
ax_C.set_ylabel("% high-pseudotime cells")
ax_C.set_ylim(0, 100)

# Bracket: sources vs null
y_br = 84
ax_C.plot([0, 0, 1, 1], [y_br-2, y_br, y_br, y_br-2], color="black", lw=0.7)
ax_C.text(0.5, y_br + 1, "****", ha="center", va="bottom", fontsize=9)

# Single annotation box: OR and p-value
or_str  = f"{or_src:.1f}" if or_src != float('inf') else "inf"
annot_box(ax_C, f"OR = {or_str}\np = {p_src:.1e}", loc="upper right")

# depleted arrow on target bar
ax_C.annotate("depleted", xy=(2, pct_tgt + 1),
              xytext=(2, pct_tgt + 10),
              ha="center", fontsize=6, color="#2980B9",
              arrowprops=dict(arrowstyle="-|>", color="#2980B9", lw=0.6))

ax_C.set_title("Scale 3 fate commitment", fontsize=8, pad=5)
label_panel(ax_C, "C")

# D: Scale ablation + inset
bio_order = ["Scale1_AcuteStress","Scale2_PeakInjury",
             "Scale3_FateDecision","Scale4_Resolution"]
abl_bio   = ablation.set_index("ablated_scale").reindex(bio_order).reset_index()
colors_D  = [SCALE_COL[s] for s in bio_order]

ax_D.barh(range(4), abl_bio["total_abs_risk_change"].values,
          color=colors_D, alpha=0.82, height=0.55,
          edgecolor="white", linewidth=0.4)
ax_D.set_yticks(range(4))
ax_D.set_yticklabels(["S1","S2","S3","S4"], fontsize=7)
ax_D.set_xlabel("Total |Drisk| (all 6,582 cells)")
ax_D.set_title("Scale structural necessity", fontsize=8, pad=5)

# Annotate S1 bar at y=0 (bottom) — S1 has the largest value
max_v   = float(abl_bio["total_abs_risk_change"].max())
s1_val  = float(abl_bio.loc[0, "total_abs_risk_change"])  # S1 at position 0
ax_D.text(s1_val + 0.2, 0, f"{s1_val:.1f}",
          va="center", fontsize=6.5,
          color=SCALE_COL["Scale1_AcuteStress"])

# Inset heatmap — biological order matches main bar chart
ax_ins = ax_D.inset_axes([0.45, 0.06, 0.52, 0.60])
tp_cols = ["tp_0h_mean_abs","tp_6h_mean_abs","tp_24h_mean_abs",
           "tp_48h_mean_abs","tp_96h_mean_abs"]
heat = np.where(abl_bio[tp_cols].values < 1e-5, 0,
                abl_bio[tp_cols].values)
ax_ins.imshow(heat, aspect="auto", cmap="Reds",
              interpolation="nearest", vmin=0)
ax_ins.set_xticks(range(5))
ax_ins.set_xticklabels(["0h","6h","24h","48h","96h"], fontsize=4.0)
ax_ins.set_yticks(range(4))
ax_ins.set_yticklabels(["S1","S2","S3","S4"], fontsize=4.0)
ax_ins.set_title("Mean |Drisk| per TP", fontsize=4.5, pad=2)
for sp in ax_ins.spines.values():
    sp.set_linewidth(0.4)
ax_ins.tick_params(width=0.4, length=1.5)
label_panel(ax_D, "D")

# E: Embedding shift
x_pos     = 0.0
xtick_pos = []
xtick_lab = []

for si, s_name in enumerate(SCALE_NAMES):
    sub  = shift_df[shift_df["scale"] == s_name]
    top1 = sub[sub["flux_pct_rank"] >= 99.0]["embedding_shift"].values
    bgnd = sub[sub["flux_pct_rank"] <  99.0]["embedding_shift"].values

    pts_bg = ax_E.violinplot([bgnd], positions=[x_pos], widths=0.36,
                              showmedians=False, showextrema=False)
    for pc in pts_bg["bodies"]:
        pc.set_facecolor("#CCCCCC"); pc.set_alpha(0.28)
        pc.set_edgecolor("none")

    pts_t1 = ax_E.violinplot([top1], positions=[x_pos+0.44], widths=0.36,
                              showmedians=False, showextrema=False)
    for pc in pts_t1["bodies"]:
        pc.set_facecolor(SCALE_COL[s_name]); pc.set_alpha(0.82)
        pc.set_edgecolor("black"); pc.set_linewidth(0.4)

    for xp, vals in [(x_pos, bgnd), (x_pos+0.44, top1)]:
        med = float(np.median(vals))
        q1  = float(np.percentile(vals, 25))
        q3  = float(np.percentile(vals, 75))
        ax_E.vlines(xp, q1, q3, color="black", linewidth=1.5, zorder=5)
        ax_E.scatter([xp], [med], color="white", edgecolors="black",
                     s=14, zorder=6, linewidths=0.7)

    xtick_pos.append(x_pos + 0.22)
    xtick_lab.append(f"S{si+1}")
    x_pos += 1.30

ax_E.set_xticks(xtick_pos)
ax_E.set_xticklabels(xtick_lab, fontsize=7)
ax_E.set_ylabel("Embedding shift (L2 norm)")
ax_E.set_title("Latent representation shift", fontsize=8, pad=5)

leg_E = [mpatches.Patch(facecolor="#CCCCCC", alpha=0.50, label="Background"),
         mpatches.Patch(facecolor="#666666", alpha=0.82, label="Top 1% flux")]
ax_E.legend(handles=leg_E, fontsize=6, loc="upper right",
            handlelength=1.0, handletextpad=0.4)

# Fix ylim BEFORE placing significance markers
ax_E.set_ylim(0, 78)
y_sig = 73.0
s3_x = 1.30*2 + 0.44   # S3 top-1% position
s4_x = 1.30*3 + 0.44   # S4 top-1% position
ax_E.text(s3_x, y_sig, "****", ha="center", va="bottom", fontsize=9,
          color=SCALE_COL["Scale3_FateDecision"])
ax_E.text(s4_x, y_sig, "n.s.", ha="center", va="bottom", fontsize=7,
          color=SCALE_COL["Scale4_Resolution"])
label_panel(ax_E, "E")

# F: Pathway enrichment dotplot
EXCLUDE = [
    "Parkinson","Alzheimer","Huntington","Prion","trypanosomiasis",
    "Malaria","Kaposi","sarcoma","Influenza","Epstein","Hepatitis",
    "Chagas","HTLV","Leishmaniasis","Pathways in cancer","MicroRNAs",
    "Transcriptional misreg","African","Toxoplasma","Salmonella",
    "Legionella","Shigellosis","Tuberculosis","COVID","Hippo","Ras ",
    "MAPK","Viral protein","Human papillo","Thyroid","small cell",
    "Non-small cell",
]
TOP_N = 2

records = []
for s_safe, s_name in [("scale1_acutestress","Scale1_AcuteStress"),
                        ("scale2_peakinjury","Scale2_PeakInjury"),
                        ("scale3_fatedecision","Scale3_FateDecision"),
                        ("scale4_resolution","Scale4_Resolution")]:
    for direction in ["up","down"]:
        path = RES / f"bottleneck_s4_{s_safe}_{direction}_kegg_2019_mouse.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[~df["Term"].str.contains(
            "|".join(EXCLUDE), case=False, na=False, regex=True)]
        if len(df) == 0:
            continue
        for _, row in df.nsmallest(TOP_N, "Adjusted P-value").iterrows():
            term = row["Term"]
            if " - " in term:
                term = term.split(" - ")[0].strip()
            term = term[:26] + ".." if len(term) > 28 else term
            records.append({"scale": s_name, "direction": direction.upper(),
                            "term": term,
                            "neg_log_p": float(-np.log10(
                                max(row["Adjusted P-value"], 1e-30)))})

if records:
    enr_df  = pd.DataFrame(records)
    dn_trms = sorted(enr_df[enr_df["direction"]=="DOWN"]["term"].unique())
    up_trms = sorted(enr_df[enr_df["direction"]=="UP"]["term"].unique())
    all_t   = dn_trms + [""] + up_trms
    term_y  = {t: i for i, t in enumerate(all_t)}
    scale_x = {s: i for i, s in enumerate(SCALE_NAMES)}

    for _, r in enr_df.iterrows():
        y = term_y.get(r["term"])
        if y is None:
            continue
        size = float(np.clip(r["neg_log_p"] * 12, 8, 100))
        mark = "v" if r["direction"] == "DOWN" else "^"
        ax_F.scatter([scale_x[r["scale"]]], [y], s=size,
                     c=SCALE_COL[r["scale"]], marker=mark,
                     alpha=0.85, linewidths=0.3,
                     edgecolors="white", zorder=4)

    # Scale 3 UP: check if empty and annotate
    s3_up = enr_df[(enr_df["scale"]=="Scale3_FateDecision") &
                   (enr_df["direction"]=="UP")]
    if len(s3_up) == 0 and up_trms:
        mid_up = float(np.mean([term_y[t] for t in up_trms]))
        ax_F.text(scale_x["Scale3_FateDecision"], mid_up,
                  "No sig.\nupregulated",
                  ha="center", va="center", fontsize=5.0,
                  color=SCALE_COL["Scale3_FateDecision"], style="italic")

    ax_F.set_yticks([term_y[t] for t in all_t if t != ""])
    ax_F.set_yticklabels([t for t in all_t if t != ""], fontsize=5.5)
    ax_F.set_xticks(range(4))
    ax_F.set_xticklabels(["S1","S2","S3","S4"], fontsize=7)
    ax_F.set_xlim(-0.7, 3.7)
    ax_F.set_ylim(-1, len(all_t))

    n_dn  = len(dn_trms)
    sep_y = n_dn - 0.5
    ax_F.axhline(sep_y, color="black", lw=0.5, ls="--", alpha=0.4)
    ax_F.text(-0.65, sep_y + 0.3, "DOWN", fontsize=6.0, color="#444")
    ax_F.text(-0.65, sep_y - 0.7, "UP",   fontsize=6.0, color="#444")

    for val, lab in [(10,"p=0.05"),(30,"p<0.001"),(60,"p<1e-6")]:
        ax_F.scatter([], [], s=val, c="grey", marker="o",
                     label=lab, alpha=0.75, linewidths=0)
    ax_F.legend(title="-log10(adj p)", title_fontsize=5.5,
                fontsize=5.5, loc="lower right",
                handletextpad=0.2, scatterpoints=1)
else:
    ax_F.text(0.5, 0.5, "Run Script 4 first.",
              ha="center", va="center", transform=ax_F.transAxes,
              fontsize=7, color="#888888")
    ax_F.set_xticks([]); ax_F.set_yticks([])

ax_F.set_xlabel("Scale")
ax_F.set_title("KEGG pathway enrichment", fontsize=8, pad=5)
label_panel(ax_F, "F", x=-0.10)

save_fig(fig, "figure_bottleneck")
plt.close(fig)

# Supplementary
print("Generating supplementary figure ...")
fig_s, ax_s = plt.subplots(figsize=(3.5, 3.0))
fig_s.subplots_adjust(left=0.18, right=0.92, top=0.92, bottom=0.22)

x_q      = np.arange(4)
max_vals = quartile["max_abs_delta_48h"].values
mn_vals  = quartile["mean_abs_delta_48h"].values
col_s    = SCALE_COL["Scale3_FateDecision"]

bars_s = ax_s.bar(x_q, max_vals, width=0.55, color=col_s,
                   edgecolor="white", linewidth=0.4)
for bar_s, a_s in zip(bars_s, [0.35, 0.52, 0.65, 0.90]):
    bar_s.set_alpha(a_s)

# Mean dots — no connecting line (categories are not continuous)
ax_s.scatter(x_q, mn_vals, color="black", s=22, zorder=5,
             edgecolors="white", linewidths=0.5, label="Mean")

ax_s.set_xticks(x_q)
ax_s.set_xticklabels([
    "Q1\n(0-25th pct)","Q2\n(25-50th pct)",
    "Q3\n(50-75th pct)","Q4\n(75-100th pct)"
], fontsize=5.5)
ax_s.set_xlabel("24h source cell pseudotime quartile")
ax_s.set_ylabel("|Delta risk| in 48h cells")
ax_s.legend(fontsize=6.5, loc="upper left", handlelength=1.2)

# Annotate Q4 above the bar with fold change vs Q1
fold = max_vals[3] / max(max_vals[0], 1e-9)
ax_s.text(3, max_vals[3] + 0.003,
          f"{max_vals[3]:.3f}\n({fold:.1f}x Q1)",
          ha="center", va="bottom", fontsize=6.0, color=col_s)

save_fig(fig_s, "supp_figure_bottleneck_quartile")
plt.close(fig_s)

print("\nAll figures saved to:", FIGS)
