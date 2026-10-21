#!/usr/bin/env python3
"""Evaluate reference predictions within sampling-time subsets.

Compare TrajRisk, the initial seed and the expression-only XGBoost comparator
using the recorded split and normalized-pseudotime threshold.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrajRisk-W8-Supp")


# CONFIGURATION


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "Data" / "processed"
RES_DIR  = BASE / "results"
FIG_DIR  = BASE / "results" / "figures"

HIGH_RISK_THRESHOLD = 0.10   # fixed in graph inputs, never adjusted
RANDOM_SEED         = 42

# Best B1 XGBoost hyperparameters from baseline and ablation analysis 5-fold CV
# (logged as: "Best XGB params: n_estimators=500 max_depth=6 lr=0.05")
XGB_BEST_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbosity=0, n_jobs=1,
)

# Minimum positives required to report a metric. Below this, the estimate
# is unreliable and we report "n_pos too low" instead of a number.
MIN_POSITIVES = 5



# DATA LOADING


def load_inputs() -> Dict:
    log.info("Loading graph inputs-7 outputs...")

    import torch
    td       = torch.load(DATA_DIR / "week6_training_data.pt", map_location="cpu")
    train_idx = td["train_idx"].numpy()
    val_idx   = td["val_idx"].numpy()
    train_y   = td["train_y"].numpy()
    val_y     = td["val_y"].numpy()
    log.info(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    x = torch.load(DATA_DIR / "week6_node_features.pt",
                   map_location="cpu").numpy()
    log.info(f"  Node features: {x.shape}")

    seeder = np.load(DATA_DIR / "week6_xgboost_risk.npy")
    log.info(f"  Seeder scores: {seeder.shape} | "
             f"val range [{seeder[val_idx].min():.3f}, {seeder[val_idx].max():.3f}]")

    preds_w7 = pd.read_csv(DATA_DIR / "week7_predictions.csv", index_col=0)
    trajrisk  = preds_w7["risk_predicted"].values
    log.info(f"  TrajRisk scores: {trajrisk.shape}")

    cell_index = pd.read_csv(DATA_DIR / "walesky_cell_index.csv", index_col=0)
    tp_col = "timepoint_str" if "timepoint_str" in cell_index.columns else "timepoint"
    log.info(f"  Cell index: {len(cell_index):,} cells | timepoint col = '{tp_col}'")

    # Timepoint for every val cell
    val_timepoints = cell_index.iloc[val_idx][tp_col].values
    unique_tp, counts = np.unique(val_timepoints, return_counts=True)
    for tp, ct in zip(unique_tp, counts):
        n_pos = (val_y[val_timepoints == tp] > HIGH_RISK_THRESHOLD).sum()
        log.info(f"  Val / {tp}: {ct} cells | {n_pos} high-risk ({100*n_pos/ct:.1f}%)")

    return {
        "x": x, "train_idx": train_idx, "val_idx": val_idx,
        "train_y": train_y, "val_y": val_y,
        "seeder": seeder, "trajrisk": trajrisk,
        "val_timepoints": val_timepoints, "tp_col": tp_col,
    }



# B1 XGBOOST REFIT


def refit_b1_xgboost(data: Dict) -> np.ndarray:
    """
    Refit the B1 XGBoost using the best hyperparameters from baseline and ablation analysis CV.
    Trains on the full train_idx (identical to baseline and ablation analysis procedure).
    Returns risk scores for all 6582 cells.
    """
    log.info("Refitting B1 XGBoost (best baseline and ablation analysis CV params, train on full train_idx)...")
    X_all   = data["x"][:, :30]     # scVI 30D only (no XGBoost risk column)
    X_train = X_all[data["train_idx"]]
    y_train = data["train_y"]

    xgb = XGBRegressor(**XGB_BEST_PARAMS)
    xgb.fit(X_train, y_train)

    scores  = np.clip(xgb.predict(X_all).astype(float), 0.0, 1.0)
    vi      = data["val_idx"]
    vb      = (data["val_y"] > HIGH_RISK_THRESHOLD).astype(int)
    log.info(f"  B1 val AUROC (all cells): {roc_auc_score(vb, scores[vi]):.4f}")
    return scores



# METRIC COMPUTATION


def subset_metrics(
    y_cont:  np.ndarray,
    scores:  np.ndarray,
    label:   str,
    subset:  str,
) -> Dict:
    """
    Compute AUROC and AUPRC for a subset of val cells.
    Returns NaN gracefully when there are too few positives.
    """
    y_bin = (y_cont > HIGH_RISK_THRESHOLD).astype(int)
    n_total = len(y_bin)
    n_pos   = int(y_bin.sum())

    if n_pos < MIN_POSITIVES:
        log.warning(f"  {subset} / {label}: only {n_pos} positives — metric unreliable")
        return {
            "method": label, "subset": subset,
            "n_total": n_total, "n_pos": n_pos, "prevalence": n_pos / n_total,
            "auroc": float("nan"), "auprc": float("nan"),
            "auprc_random": n_pos / n_total,
        }

    try:
        auroc = float(roc_auc_score(y_bin, scores))
    except Exception:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_bin, scores))
    except Exception:
        auprc = float("nan")

    return {
        "method":       label,
        "subset":       subset,
        "n_total":      n_total,
        "n_pos":        n_pos,
        "prevalence":   round(n_pos / n_total, 4),
        "auroc":        round(auroc, 4),
        "auprc":        round(auprc, 4),
        "auprc_random": round(n_pos / n_total, 4),  # baseline for AUPRC
    }



# MAIN ANALYSIS


def run_analysis(data: Dict, b1_scores: np.ndarray) -> pd.DataFrame:
    vi  = data["val_idx"]
    val_y   = data["val_y"]
    val_tp  = data["val_timepoints"]

    methods = {
        "TrajRisk":      data["trajrisk"][vi],
        "Seeder (W6)":   data["seeder"][vi],
        "B1 XGBoost":    b1_scores[vi],
    }

    rows: List[Dict] = []

    # Global val (all timepoints)
    log.info("\n[A] Global val (all timepoints):")
    for mname, scores in methods.items():
        r = subset_metrics(val_y, scores, mname, "All val")
        rows.append(r)
        log.info(f"  {mname:<20} AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}")

    # Non-6h val (exclude 6h cells)
    log.info("\n[B] Non-6h val (0h + 24h + 48h + 96h only):")
    mask_non6h = val_tp != "6h"
    y_non6h    = val_y[mask_non6h]
    log.info(f"  Cells: {mask_non6h.sum()} | "
             f"High-risk: {(y_non6h > HIGH_RISK_THRESHOLD).sum()}")
    for mname, scores in methods.items():
        r = subset_metrics(y_non6h, scores[mask_non6h], mname, "Non-6h")
        rows.append(r)
        log.info(f"  {mname:<20} AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}")

    # Per-timepoint analysis
    for tp in ["0h", "24h", "48h", "96h"]:
        mask   = val_tp == tp
        y_tp   = val_y[mask]
        n_pos  = int((y_tp > HIGH_RISK_THRESHOLD).sum())
        n_tot  = int(mask.sum())
        log.info(f"\n[C] {tp} only: {n_tot} cells | {n_pos} high-risk "
                 f"({100*n_pos/max(n_tot,1):.1f}%) | "
                 f"random AUPRC baseline = {n_pos/max(n_tot,1):.3f}")
        for mname, scores in methods.items():
            r = subset_metrics(y_tp, scores[mask], mname, tp)
            rows.append(r)
            if not np.isnan(r["auroc"]):
                log.info(f"  {mname:<20} AUROC={r['auroc']:.4f}  "
                         f"AUPRC={r['auprc']:.4f}  "
                         f"(random={r['auprc_random']:.3f})")
            else:
                log.info(f"  {mname:<20} n_pos={n_pos} — below threshold, not reported")

    return pd.DataFrame(rows)



# FIGURE


def make_within_timepoint_figure(df: pd.DataFrame) -> None:
    """
    AUPRC comparison across within-timepoint subsets.
    The random AUPRC baseline (= prevalence) is shown as a dashed line
    for each panel. This is the correct reference for AUPRC interpretation.
    """
    tp_subsets = ["Non-6h", "24h", "48h", "96h"]
    method_colors = {
        "TrajRisk":    "#2ecc71",
        "Seeder (W6)": "#e67e22",
        "B1 XGBoost":  "#3498db",
    }
    methods = list(method_colors.keys())

    fig, axes = plt.subplots(1, len(tp_subsets), figsize=(14, 5), sharey=False)
    fig.suptitle(
        "TrajRisk vs Expression-only Methods — Within-Timepoint AUPRC\n"
        "(dashed line = random baseline for each subset; "
        f"high-risk threshold: pseudotime_norm > {HIGH_RISK_THRESHOLD})",
        fontsize=11, fontweight="bold",
    )

    for ax, tp in zip(axes, tp_subsets):
        sub = df[df["subset"] == tp].set_index("method")
        x_pos = np.arange(len(methods))
        auprc_vals = []
        colors     = []
        for m in methods:
            if m in sub.index and not np.isnan(sub.loc[m, "auprc"]):
                auprc_vals.append(sub.loc[m, "auprc"])
            else:
                auprc_vals.append(0.0)
            colors.append(method_colors[m])

        ax.bar(x_pos, auprc_vals, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=9)
        ax.set_title(f"{tp}", fontweight="bold", fontsize=11)
        ax.set_ylabel("AUPRC" if tp == tp_subsets[0] else "")
        ax.set_ylim(0, min(1.0, max(auprc_vals) * 1.4 + 0.05))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Random baseline
        if tp in sub.index and not np.isnan(sub.iloc[0]["auprc_random"]):
            rand_auprc = sub.iloc[0]["auprc_random"]
            ax.axhline(rand_auprc, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.7,
                       label=f"Random ({rand_auprc:.3f})")
            ax.legend(fontsize=8, loc="upper right")

        # Value labels
        for xp, val in zip(x_pos, auprc_vals):
            if val > 0:
                ax.text(xp, val + 0.005, f"{val:.3f}", ha="center",
                        va="bottom", fontsize=8)

    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=m) for m, c in method_colors.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIG_DIR / "week8_within_timepoint_auprc.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out.name}")



# PRINT SUMMARY TABLE


def print_summary_table(df: pd.DataFrame) -> None:
    log.info("\n" + "=" * 72)
    log.info("SUPPLEMENTARY RESULTS SUMMARY")
    log.info("=" * 72)

    subsets = ["All val", "Non-6h", "24h", "48h", "96h"]
    metrics = ["TrajRisk", "Seeder (W6)", "B1 XGBoost"]

    log.info(f"\n{'Subset':<12} {'Method':<22} {'n_total':>8} {'n_pos':>7} "
             f"{'AUROC':>7} {'AUPRC':>7} {'AUPRC_rand':>11}")
    log.info("  " + "─" * 74)

    for subset in subsets:
        sub = df[df["subset"] == subset]
        for _, row in sub.iterrows():
            auroc = f"{row['auroc']:.4f}" if not np.isnan(row["auroc"]) else "n/a"
            auprc = f"{row['auprc']:.4f}" if not np.isnan(row["auprc"]) else "n/a"
            rand  = f"{row['auprc_random']:.4f}"
            log.info(f"  {subset:<12} {row['method']:<22} {int(row['n_total']):>8} "
                     f"{int(row['n_pos']):>7} {auroc:>7} {auprc:>7} {rand:>11}")
        log.info("  " + "─" * 74)

    # Key comparisons
    log.info("\nKEY COMPARISONS:")
    for subset in ["Non-6h", "24h", "96h"]:
        sub = df[df["subset"] == subset].set_index("method")
        if "TrajRisk" in sub.index and "Seeder (W6)" in sub.index:
            tr = sub.loc["TrajRisk", "auprc"]
            sd = sub.loc["Seeder (W6)", "auprc"]
            b1 = sub.loc["B1 XGBoost", "auprc"] if "B1 XGBoost" in sub.index else float("nan")
            rand = sub.loc["TrajRisk", "auprc_random"]
            if not (np.isnan(tr) or np.isnan(sd)):
                gain   = (tr - sd) * 100
                fold   = tr / max(rand, 1e-6)
                b1_str = f"{b1:.4f}" if not np.isnan(b1) else "n/a"
                log.info(f"  {subset}: TrajRisk={tr:.4f} | Seeder={sd:.4f} | "
                         f"B1={b1_str} | "
                         f"random={rand:.4f} | "
                         f"TrajRisk gain over seeder: {gain:+.2f}pp | "
                         f"TrajRisk fold over random: {fold:.1f}x")



# MAIN


def main() -> None:
    log.info("=" * 72)
    log.info("TRAJRISK — baseline and ablation analysis SUPPLEMENTARY ANALYSIS")
    log.info("=" * 72)
    log.info(f"Threshold: pseudotime_norm > {HIGH_RISK_THRESHOLD}")
    log.info(f"Min positives to report: {MIN_POSITIVES}")

    data     = load_inputs()
    b1_scores = refit_b1_xgboost(data)

    log.info("\nRunning within-timepoint analysis...")
    results  = run_analysis(data, b1_scores)

    print_summary_table(results)

    log.info("\nSaving outputs...")
    out_csv = RES_DIR / "week8_supplementary_table.csv"
    results.to_csv(out_csv, index=False, float_format="%.6f")
    log.info(f"  Saved: {out_csv.name}")

    log.info("Generating figure...")
    make_within_timepoint_figure(results)

    log.info("\n" + "=" * 72)
    log.info("baseline and ablation analysis SUPPLEMENTARY COMPLETE")
    log.info("=" * 72)
    log.info("")
    log.info("Interpretation guide:")
    log.info("  Non-6h AUROC:  tests risk discrimination after removing the trivial")
    log.info("                 6h vs non-6h signal.")
    log.info("  24h/96h AUPRC: tests identification of functional compensation cells.")
    log.info("                 Compare against random baseline (= prevalence).")
    log.info("  If TrajRisk > Seeder on these metrics: graph propagation adds value")
    log.info("    beyond the initial expression-based risk seed.")
    log.info("  If XGBoost wins here too: the within-timepoint claim needs revision")
    log.info("    and the paper argument shifts entirely to external validation generalization.")
    log.info("")
    log.info("→ Freeze baseline and ablation analysis. Move to external validation: Cross-Dataset Validation.")


if __name__ == "__main__":
    main()
