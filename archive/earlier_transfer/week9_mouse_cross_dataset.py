#!/usr/bin/env python3
"""
TrajRisk — Week 9 Script 1: Mouse Cross-Dataset Validation
===========================================================
Project : TrajRisk — Trajectory-Informed Risk in APAP Hepatotoxicity
Journal : Cell Reports Methods

What this script does
---------------------
Applies the Walesky-trained TrajRisk model to three independent mouse datasets
to test whether the learned risk representation generalises beyond the training
data. No re-training occurs. No pseudotime inference is re-run.

Validation tests
----------------
T0  Walesky held-out validation (Week 7/8 reference numbers, reproduced here
    for direct comparison with T1-T3 in the same table)

T1  Ben-Moshe (Zenodo 6035873) — different laboratory, 10x Chromium platform
    Label: treatment == 'APAP' (771 cells at 48h/72h) vs
           treatment == 'Control' (1,999 cells at 0h)
    Question: does TrajRisk generalise across labs and sequencing platforms?

T2  Umbaugh 2h (GSE200771) — timepoint never seen during training
    Label: treatment == 'APAP' (13,502 cells at 2h) vs
           treatment == 'Control' (7,907 cells at 0h)
    Question: can TrajRisk detect early injury at 2h before peak activation?

T3  Umbaugh 48h/72h (GSE255834) — p21-sorted hepatocytes (mechanistic)
    Label A: Cdkn1a expression > median vs <= median (n=5,306 total)
    Label B: treatment == 'APAP' (2,978) vs treatment == 'Control' (2,328)
    Question: do p21/Cdkn1a-high cells receive elevated trajectory risk?

All column names, treatment values, and dataset labels were confirmed by a
pre-run diagnostic before writing this script:
    treatment values : 'APAP', 'Control'  (exact strings, no variants)
    dataset labels   : 'Ben_Moshe', 'Umbaugh_2h', 'Umbaugh_48h72h', 'Walesky'
    cell_index cols  : 'treatment_str', 'timepoint_str', 'hours', 'global_idx'

How non-Walesky cells are processed by TrajRiskGNN
---------------------------------------------------
The Walesky transition graph connects cells at positions 0-6581 in the node
tensor.  Validation cells are appended at positions 6582+.  Since no edge in
any scale points TO a validation cell, TrajRiskConv reduces to:

    agg_i = 0   (no incoming messages)
    h_i    = W_nbr(0) + W_self(x_i)  =  W_self(x_i)   [W_nbr has bias=False]

This is the W_self feature-to-risk pathway.  BatchNorm in eval() mode uses
running statistics from Walesky training.  This is described in the Methods
as "cross-dataset feature-space transfer", NOT "trajectory graph propagation".

Methods compared
----------------
    TrajRisk   — AUROC-best Week 7 checkpoint (W_self path for external cells)
    Seeder     — Week 6 XGBClassifier (0h vs 6h only), no graph, no trajectory
    B1 XGBoost — Week 8 XGBRegressor (scVI → pseudotime_norm, expression-only)

Usage
-----
    conda activate trajrisk
    cd /path/to/TrajRisk
    python scripts/week9_mouse_cross_dataset.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

import scanpy as sc
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
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
log = logging.getLogger("W9-Mouse")


# =============================================================================
# 0.  PATHS AND CONSTANTS
# =============================================================================

BASE = Path(__file__).resolve().parents[2]
DATA     = BASE / "Data" / "processed"
MODELS   = BASE / "models"
RESULTS  = BASE / "results"
FIGS     = RESULTS / "figures"

for _d in [RESULTS, FIGS]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE         = torch.device("cpu")
RANDOM_SEED    = 42
N_BOOTSTRAP    = 1_000
MIN_POS        = 5          # minimum positives required to report a metric
HIGH_RISK_THR  = 0.10       # pseudotime_norm threshold for Walesky binary label
CDKN1A_GENE    = "Cdkn1a"  # confirmed present in var_names by diagnostic

# B1 XGBoost best params from Week 8 5-fold CV
XGB_B1_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbosity=0, n_jobs=1,
)

# Exact scale names — must match keys in multiscale_graphs.pt
SCALE_NAMES = [
    "Scale1_AcuteStress",
    "Scale2_PeakInjury",
    "Scale3_FateDecision",
    "Scale4_Resolution",
]

# Architecture constants — must match Week 7 exactly
NODE_DIM   = 31
HIDDEN_DIM = 64
N_SCALES   = 4
DROPOUT    = 0.3

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# 1.  MODEL CLASSES  (exact copies from Week 7 — do not modify)
# =============================================================================

class TrajRiskConv(MessagePassing):
    """Directed, edge-weighted graph convolution with residual self-loop."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.W_nbr  = nn.Linear(in_channels, out_channels, bias=False)
        self.W_self = nn.Linear(in_channels, out_channels, bias=True)
        nn.init.xavier_uniform_(self.W_nbr.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)

    def forward(self, x, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.W_nbr(agg) + self.W_self(x)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j


class TrajRiskGNN(nn.Module):
    """Multi-scale trajectory risk GNN — 4 parallel scale branches + fusion MLP."""

    def __init__(self, node_dim=NODE_DIM, hidden_dim=HIDDEN_DIM,
                 n_scales=N_SCALES, dropout=DROPOUT):
        super().__init__()
        self.n_scales    = n_scales
        self.scale_convs = nn.ModuleList()
        self.scale_bns   = nn.ModuleList()
        for _ in range(n_scales):
            self.scale_convs.append(nn.ModuleList([
                TrajRiskConv(node_dim,   hidden_dim),
                TrajRiskConv(hidden_dim, hidden_dim),
            ]))
            self.scale_bns.append(nn.ModuleList([
                nn.BatchNorm1d(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ]))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(n_scales * hidden_dim, 128), nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid(),
        )

    def forward(self, x, edge_indices, edge_weights):
        parts = []
        for s in range(self.n_scales):
            h = x
            for layer in range(2):
                h = self.scale_convs[s][layer](h, edge_indices[s], edge_weights[s])
                h = self.scale_bns[s][layer](h)
                h = torch.relu(h)
            parts.append(h)
        return self.fusion_mlp(torch.cat(parts, dim=1)).squeeze(-1)


# =============================================================================
# 2.  LOAD AND VERIFY ALL DATA
# =============================================================================

def load_and_verify() -> Dict:
    """
    Load every file needed for validation.
    Performs sanity checks so any misalignment is caught before computation.
    """
    log.info("=" * 68)
    log.info("LOADING AND VERIFYING ALL DATA")
    log.info("=" * 68)

    # ── Integrated AnnData ───────────────────────────────────────────────────
    h5ad = DATA / "adata_integrated_with_trajectory.h5ad"
    log.info(f"  Loading {h5ad.name} ...")
    adata = sc.read_h5ad(str(h5ad))
    assert adata.n_obs == 36_067, f"Expected 36,067 cells, got {adata.n_obs}"
    assert "X_scvi" in adata.obsm,   "X_scvi missing from adata.obsm"
    assert adata.obsm["X_scvi"].shape == (36_067, 30), \
        f"X_scvi shape mismatch: {adata.obsm['X_scvi'].shape}"
    log.info(f"  adata: {adata.n_obs:,} cells × {adata.n_vars:,} genes  ✓")
    log.info(f"  X_scvi: {adata.obsm['X_scvi'].shape}  ✓")
    assert CDKN1A_GENE in adata.var_names, f"{CDKN1A_GENE} not in var_names"
    log.info(f"  {CDKN1A_GENE} in var_names  ✓")

    # ── Dataset counts (confirm match with diagnostic) ───────────────────────
    ds_counts = adata.obs["dataset"].value_counts()
    log.info(f"  Dataset counts:\n{ds_counts.to_string()}")
    assert ds_counts["Walesky"]        == 6_582
    assert ds_counts["Umbaugh_2h"]     == 21_409
    assert ds_counts["Umbaugh_48h72h"] == 5_306
    assert ds_counts["Ben_Moshe"]      == 2_770

    # ── Treatment value cross-check per dataset ──────────────────────────────
    for ds in ["Ben_Moshe", "Umbaugh_2h", "Umbaugh_48h72h"]:
        vals = set(adata.obs.loc[adata.obs["dataset"] == ds, "treatment"].unique())
        assert vals == {"APAP", "Control"}, \
            f"Unexpected treatment values in {ds}: {vals}"
    log.info("  Treatment values {'APAP', 'Control'} confirmed for all validation datasets  ✓")

    # ── Walesky cell index ────────────────────────────────────────────────────
    ci_path = DATA / "walesky_cell_index.csv"
    cell_index = pd.read_csv(ci_path, index_col=0)
    assert len(cell_index) == 6_582, \
        f"cell_index expected 6,582 rows, got {len(cell_index)}"
    required_cols = {"global_idx", "timepoint_str", "hours", "treatment_str"}
    missing = required_cols - set(cell_index.columns)
    assert not missing, f"cell_index missing columns: {missing}"
    log.info(f"  walesky_cell_index.csv: {len(cell_index):,} rows  ✓")

    # ── Walesky node features ─────────────────────────────────────────────────
    x_walesky = torch.load(DATA / "week6_node_features.pt", map_location="cpu")
    assert x_walesky.shape == (6_582, 31), \
        f"x_walesky shape mismatch: {x_walesky.shape}"
    log.info(f"  x_walesky: {tuple(x_walesky.shape)}  ✓")

    # ── Alignment check: scVI in x_walesky vs adata.obsm['X_scvi'] ───────────
    # We verify the first 5 Walesky barcodes match between cell_index and adata
    adata_wal_mask = adata.obs["dataset"] == "Walesky"
    adata_wal_barcodes = set(adata.obs_names[adata_wal_mask])
    ci_barcodes = set(cell_index.index[:50])
    overlap = ci_barcodes & adata_wal_barcodes
    assert len(overlap) >= 40, \
        f"cell_index barcodes not found in adata (only {len(overlap)}/50 match)"
    log.info(f"  scVI barcode alignment: {len(overlap)}/50 cell_index barcodes in adata  ✓")

    # ── Transition graphs ─────────────────────────────────────────────────────
    graphs   = torch.load(DATA / "multiscale_graphs.pt", map_location="cpu")
    for sn in SCALE_NAMES:
        assert sn in graphs, f"Scale {sn} missing from multiscale_graphs.pt"
    edge_indices = [graphs[sn]["edge_index"] for sn in SCALE_NAMES]
    edge_weights = [graphs[sn]["edge_weight"] for sn in SCALE_NAMES]
    # All edge indices must reference positions in [0, 6581]
    for s, ei in enumerate(edge_indices):
        assert ei.max().item() <= 6_581, \
            f"Scale {s} edge_index out of range: max={ei.max().item()}"
    total_edges = sum(e.shape[1] for e in edge_indices)
    log.info(f"  Transition graphs: 4 scales, {total_edges:,} total edges  ✓")

    # ── TrajRisk model checkpoint ─────────────────────────────────────────────
    ckpt_path = MODELS / "week7_trajrisk_FINAL.pt"
    if not ckpt_path.exists():
        ckpt_path = MODELS / "week7_trajrisk_BEST_AUROC.pt"
    assert ckpt_path.exists(), f"No TrajRisk checkpoint found at {ckpt_path}"
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = TrajRiskGNN().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"  TrajRisk model: {ckpt_path.name}  ✓")

    # ── Seeder (Week 6 XGBClassifier) ────────────────────────────────────────
    seeder = joblib.load(MODELS / "week6_xgboost_model.pkl")
    log.info("  Seeder (Week 6 XGBClassifier)  ✓")

    # ── Seeder risk scores for Walesky cells ──────────────────────────────────
    seeder_walesky = np.load(DATA / "week6_xgboost_risk.npy")
    assert seeder_walesky.shape == (6_582,), \
        f"seeder_walesky shape: {seeder_walesky.shape}"
    log.info(f"  seeder_walesky: {seeder_walesky.shape}  ✓")

    # ── Week 7 TrajRisk predictions for Walesky cells ────────────────────────
    preds_w7 = pd.read_csv(DATA / "week7_predictions.csv", index_col=0)
    assert len(preds_w7) == 6_582, \
        f"week7_predictions expected 6,582 rows, got {len(preds_w7)}"
    assert "risk_predicted" in preds_w7.columns
    trajrisk_walesky = preds_w7["risk_predicted"].values.astype(np.float32)
    log.info(f"  week7_predictions: {len(preds_w7):,} rows  ✓")

    # ── Training split (for T0 reference and B1 XGBoost fitting) ─────────────
    td = torch.load(DATA / "week6_training_data.pt", map_location="cpu")
    train_idx = td["train_idx"].numpy()
    val_idx   = td["val_idx"].numpy()
    train_y   = td["train_y"].numpy()
    val_y     = td["val_y"].numpy()
    log.info(f"  Training split: {len(train_idx):,} train, {len(val_idx):,} val  ✓")

    log.info("  All files loaded and verified.\n")

    return {
        "adata":            adata,
        "cell_index":       cell_index,
        "x_walesky":        x_walesky,
        "edge_indices":     edge_indices,
        "edge_weights":     edge_weights,
        "model":            model,
        "seeder":           seeder,
        "seeder_walesky":   seeder_walesky,
        "trajrisk_walesky": trajrisk_walesky,
        "train_idx":        train_idx,
        "val_idx":          val_idx,
        "train_y":          train_y,
        "val_y":            val_y,
    }


# =============================================================================
# 3.  FIT B1 XGBOOST  (expression-only baseline from Week 8)
# =============================================================================

def fit_b1_xgboost(data: Dict) -> Tuple[XGBRegressor, np.ndarray]:
    """
    Fit the Week 8 expression-only XGBoost on Walesky training data.
    Input:  30D scVI features
    Target: pseudotime_norm
    Returns the fitted model and its predictions on ALL 6,582 Walesky cells.
    """
    log.info("Fitting B1 XGBoost (scVI → pseudotime_norm) ...")
    x_scvi_wal = data["x_walesky"][:, :30].numpy()   # (6582, 30)
    b1 = XGBRegressor(**XGB_B1_PARAMS)
    b1.fit(x_scvi_wal[data["train_idx"]], data["train_y"])
    b1_walesky = np.clip(b1.predict(x_scvi_wal).astype(float), 0.0, 1.0).astype(np.float32)
    log.info(f"  B1 XGBoost fitted  — Walesky range [{b1_walesky.min():.3f}, {b1_walesky.max():.3f}]  ✓\n")
    return b1, b1_walesky


# =============================================================================
# 4.  COMPUTE RISK SCORES FOR VALIDATION CELLS
# =============================================================================

def compute_scores(data: Dict, b1_xgb: XGBRegressor,
                   z_val: np.ndarray, label: str) -> Dict[str, np.ndarray]:
    """
    Compute TrajRisk, Seeder, and B1 XGBoost risk scores for a set of
    validation cells given their 30D scVI embeddings.

    TrajRisk pathway for validation cells:
        x_all = cat([x_walesky (6582 cells), x_val (n_val cells)])
        Edge indices reference only positions 0-6581 (Walesky).
        Validation cells at 6582+ receive zero aggregated messages.
        TrajRiskConv reduces to:  h_i = W_self @ x_i
        This is the W_self feature-to-risk pathway (no graph propagation).

    Parameters
    ----------
    z_val  : (n_val, 30)  scVI features for validation cells
    label  : string for logging

    Returns
    -------
    dict with keys 'trajrisk', 'seeder', 'b1_xgb'  — each shape (n_val,)
    """
    n_val = z_val.shape[0]
    n_wal = data["x_walesky"].shape[0]   # 6582
    log.info(f"  [{label}] n={n_val:,} cells — computing risk scores ...")

    # ── Seeder: XGBClassifier predict_proba → P(APAP/6h-like) ───────────────
    seeder_val = data["seeder"].predict_proba(z_val)[:, 1].astype(np.float32)

    # ── B1 XGBoost: predict pseudotime_norm from scVI ────────────────────────
    b1_val = np.clip(b1_xgb.predict(z_val).astype(float), 0.0, 1.0).astype(np.float32)

    # ── TrajRisk: append validation cells to Walesky tensor ──────────────────
    # Column 30 of x_val = seeder initial risk (r_i)
    r_val  = seeder_val.reshape(-1, 1)                       # (n_val, 1)
    x_val  = np.hstack([z_val, r_val]).astype(np.float32)    # (n_val, 31)

    x_all  = torch.tensor(
        np.vstack([data["x_walesky"].numpy(), x_val]),
        dtype=torch.float32,
    )   # (6582 + n_val, 31)
    # Positions 0–6581 : Walesky cells (connected by transition edges)
    # Positions 6582+  : Validation cells (no edges — W_self path only)

    with torch.no_grad():
        data["model"].eval()
        risk_all = data["model"](
            x_all, data["edge_indices"], data["edge_weights"]
        ).numpy()

    trajrisk_val = risk_all[n_wal:].astype(np.float32)   # (n_val,)

    log.info(f"    TrajRisk range : [{trajrisk_val.min():.4f}, {trajrisk_val.max():.4f}]")
    log.info(f"    Seeder  range  : [{seeder_val.min():.4f}, {seeder_val.max():.4f}]")
    log.info(f"    B1 XGB  range  : [{b1_val.min():.4f}, {b1_val.max():.4f}]")

    return {"trajrisk": trajrisk_val, "seeder": seeder_val, "b1_xgb": b1_val}


# =============================================================================
# 5.  METRICS
# =============================================================================

def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() < MIN_POS or (1 - y).sum() < MIN_POS:
        return float("nan")
    return float(roc_auc_score(y, scores))


def safe_auprc(y: np.ndarray, scores: np.ndarray) -> float:
    if y.sum() < MIN_POS:
        return float("nan")
    return float(average_precision_score(y, scores))


def bootstrap_auroc_ci(y: np.ndarray, scores: np.ndarray,
                        n: int = N_BOOTSTRAP, seed: int = RANDOM_SEED
                        ) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        ys  = y[idx]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        try:
            aurocs.append(roc_auc_score(ys, scores[idx]))
        except Exception:
            continue
    if not aurocs:
        return float("nan"), float("nan")
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def report_metrics(test_name: str, y: np.ndarray,
                   scores: Dict[str, np.ndarray]) -> List[Dict]:
    """Compute and log AUROC / AUPRC for each method. Return list of row dicts."""
    n_total = int(len(y))
    n_pos   = int(y.sum())
    prev    = n_pos / max(n_total, 1)

    log.info(f"\n  {test_name}")
    log.info(f"  n={n_total:,}  positives={n_pos:,}  prevalence={prev:.3f}")
    log.info(f"  {'Method':<20} {'AUROC':>7} {'95% CI':>16} {'AUPRC':>8} {'AUPRC_rand':>11}")
    log.info("  " + "─" * 66)

    rows = []
    for mname, sc in scores.items():
        auroc          = safe_auroc(y, sc)
        auprc          = safe_auprc(y, sc)
        ci_lo, ci_hi   = (bootstrap_auroc_ci(y, sc) if not np.isnan(auroc)
                          else (float("nan"), float("nan")))

        ci_str = (f"({ci_lo:.3f}–{ci_hi:.3f})" if not np.isnan(ci_lo) else "(—)")
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "n/a"
        auprc_str = f"{auprc:.4f}" if not np.isnan(auprc) else "n/a"

        log.info(f"  {mname:<20} {auroc_str:>7} {ci_str:>16} {auprc_str:>8} {prev:>11.4f}")

        rows.append({
            "test":       test_name,
            "method":     mname,
            "n_total":    n_total,
            "n_pos":      n_pos,
            "prevalence": round(prev, 4),
            "auroc":      round(auroc, 4) if not np.isnan(auroc) else float("nan"),
            "auroc_ci_lo":round(ci_lo,  4) if not np.isnan(ci_lo)  else float("nan"),
            "auroc_ci_hi":round(ci_hi,  4) if not np.isnan(ci_hi)  else float("nan"),
            "auprc":      round(auprc, 4) if not np.isnan(auprc) else float("nan"),
            "auprc_rand": round(prev,  4),
        })
    return rows


# =============================================================================
# 6.  T0 — WALESKY HELD-OUT VALIDATION (reference)
# =============================================================================

def run_t0_walesky(data: Dict, b1_walesky: np.ndarray) -> List[Dict]:
    """
    Reproduce the Week 7/8 Walesky validation AUROC using pre-computed scores.
    This row is the internal reference for comparing T1-T3.
    """
    log.info("\n" + "=" * 68)
    log.info("T0: Walesky held-out validation (Week 7/8 reference)")
    log.info("=" * 68)

    val_idx = data["val_idx"]
    val_y   = data["val_y"]
    y_bin   = (val_y > HIGH_RISK_THR).astype(int)

    scores = {
        "trajrisk": data["trajrisk_walesky"][val_idx],
        "seeder":   data["seeder_walesky"][val_idx],
        "b1_xgb":   b1_walesky[val_idx],
    }
    rows = report_metrics("T0: Walesky val (internal reference)", y_bin, scores)
    return rows


# =============================================================================
# 7.  T1 — BEN-MOSHE  (cross-dataset, different lab / platform)
# =============================================================================

def run_t1_benmoshe(data: Dict, b1_xgb: XGBRegressor) -> List[Dict]:
    """
    Ben-Moshe dataset: 1,999 Control (0h) + 771 APAP (48h/72h).
    Binary label: treatment == 'APAP' → 1, treatment == 'Control' → 0.
    Same column confirmed by diagnostic ('treatment', values 'APAP'/'Control').
    """
    log.info("\n" + "=" * 68)
    log.info("T1: Ben-Moshe — cross-dataset validation (10x vs Seq-Well)")
    log.info("=" * 68)

    adata    = data["adata"]
    bm_mask  = adata.obs["dataset"] == "Ben_Moshe"
    adata_bm = adata[bm_mask]

    y_bm   = (adata_bm.obs["treatment"] == "APAP").values.astype(int)
    z_bm   = adata_bm.obsm["X_scvi"].astype(np.float32)           # (2770, 30)

    log.info(f"  APAP: {y_bm.sum():,} | Control: {(1-y_bm).sum():,}")
    log.info(f"  Timepoints: {sorted(adata_bm.obs['timepoint_hours'].unique().tolist())}")

    scores = compute_scores(data, b1_xgb, z_bm, "Ben_Moshe")
    rows   = report_metrics("T1: Ben-Moshe (APAP vs Control)", y_bm, scores)
    return rows


# =============================================================================
# 8.  T2 — UMBAUGH 2h  (early detection, novel timepoint)
# =============================================================================

def run_t2_umbaugh2h(data: Dict, b1_xgb: XGBRegressor) -> List[Dict]:
    """
    Umbaugh 2h: 13,502 APAP cells (2h) + 7,907 Control cells (0h).
    Binary: treatment == 'APAP' → 1, treatment == 'Control' → 0.
    The model was trained on timepoints 0h, 6h, 24h, 48h, 96h.
    2h is a timepoint the model has NEVER seen.
    """
    log.info("\n" + "=" * 68)
    log.info("T2: Umbaugh 2h — early detection (2h APAP vs 0h Control)")
    log.info("=" * 68)

    adata      = data["adata"]
    u2_mask    = adata.obs["dataset"] == "Umbaugh_2h"
    adata_u2   = adata[u2_mask]

    y_u2   = (adata_u2.obs["treatment"] == "APAP").values.astype(int)
    z_u2   = adata_u2.obsm["X_scvi"].astype(np.float32)           # (21409, 30)

    # Cross-tabulate treatment × timepoint to confirm data structure
    ctab = pd.crosstab(
        adata_u2.obs["treatment"],
        adata_u2.obs["timepoint_hours"].round(0)
    )
    log.info(f"  Treatment × Timepoint cross-tabulation:\n{ctab.to_string()}")
    log.info(f"  APAP: {y_u2.sum():,} | Control: {(1-y_u2).sum():,}")

    scores = compute_scores(data, b1_xgb, z_u2, "Umbaugh_2h")
    rows   = report_metrics("T2: Umbaugh 2h (APAP vs Control)", y_u2, scores)
    return rows


# =============================================================================
# 9.  T3 — UMBAUGH 48h/72h  (p21/Cdkn1a mechanistic validation)
# =============================================================================

def run_t3_umbaugh_p21(data: Dict, b1_xgb: XGBRegressor
                        ) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Umbaugh 48h/72h: 2,978 APAP + 2,328 Control at 48h and 72h.
    Cdkn1a confirmed present in var_names.

    Two analyses:
      T3a  Cdkn1a median split:  Cdkn1a-high vs Cdkn1a-low (mechanistic)
           + Spearman correlation between Cdkn1a expression and risk score
      T3b  APAP vs Control:      standard treatment comparison

    p21/Cdkn1a marks cell-cycle-arrested hepatocytes that delay regeneration.
    Hypothesis: Cdkn1a-high cells occupy higher trajectory risk positions.
    """
    log.info("\n" + "=" * 68)
    log.info("T3: Umbaugh 48h/72h — p21 (Cdkn1a) mechanistic validation")
    log.info("=" * 68)

    adata     = data["adata"]
    u48_mask  = adata.obs["dataset"] == "Umbaugh_48h72h"
    adata_u48 = adata[u48_mask]

    z_u48 = adata_u48.obsm["X_scvi"].astype(np.float32)           # (5306, 30)
    log.info(f"  n={adata_u48.n_obs:,}  timepoints={sorted(adata_u48.obs['timepoint_hours'].unique().tolist())}")
    log.info(f"  APAP: {(adata_u48.obs['treatment']=='APAP').sum():,} | "
             f"Control: {(adata_u48.obs['treatment']=='Control').sum():,}")

    # ── Extract Cdkn1a expression ─────────────────────────────────────────────
    cdkn1a_col = list(adata_u48.var_names).index(CDKN1A_GENE)
    X = adata_u48.X
    if hasattr(X, "toarray"):
        cdkn1a_expr = np.asarray(X[:, cdkn1a_col].toarray()).flatten()
    else:
        cdkn1a_expr = np.asarray(X[:, cdkn1a_col]).flatten()

    log.info(f"  Cdkn1a expression — mean={cdkn1a_expr.mean():.4f}  "
             f"median={np.median(cdkn1a_expr):.4f}  "
             f"max={cdkn1a_expr.max():.4f}  "
             f"% expressed={100*(cdkn1a_expr>0).mean():.1f}%")

    # ── Compute risk scores for all Umbaugh 48h/72h cells ────────────────────
    scores = compute_scores(data, b1_xgb, z_u48, "Umbaugh_48h72h")

    # ── T3a: Cdkn1a median split ──────────────────────────────────────────────
    median_cdkn1a = np.median(cdkn1a_expr)
    y_cdkn1a      = (cdkn1a_expr > median_cdkn1a).astype(int)
    n_high        = int(y_cdkn1a.sum())
    n_low         = int((1 - y_cdkn1a).sum())
    log.info(f"\n  T3a — Cdkn1a median split (threshold={median_cdkn1a:.4f}):  "
             f"high={n_high:,}  low={n_low:,}")
    rows_t3a = report_metrics(
        "T3a: Umbaugh 48h/72h (Cdkn1a-high vs -low)", y_cdkn1a, scores
    )

    # ── Spearman correlation: Cdkn1a vs risk ──────────────────────────────────
    log.info("\n  T3a — Spearman ρ (Cdkn1a expression vs risk score):")
    spearman_rows = []
    for mname, sc in scores.items():
        rho, pval = spearmanr(cdkn1a_expr, sc)
        log.info(f"    {mname:<20} ρ={rho:+.4f}  p={pval:.2e}")
        spearman_rows.append({
            "method": mname, "spearman_rho": round(float(rho), 4),
            "pvalue": float(pval)
        })

    # Mann-Whitney: Cdkn1a-high vs Cdkn1a-low risk scores
    log.info("\n  T3a — Mann-Whitney U (Cdkn1a-high > Cdkn1a-low, one-sided):")
    mw_rows = []
    for mname, sc in scores.items():
        high_scores = sc[y_cdkn1a == 1]
        low_scores  = sc[y_cdkn1a == 0]
        stat, pval  = mannwhitneyu(high_scores, low_scores, alternative="greater")
        log.info(f"    {mname:<20} "
                 f"mean_high={high_scores.mean():.4f}  "
                 f"mean_low={low_scores.mean():.4f}  "
                 f"p={pval:.2e}")
        mw_rows.append({
            "method":    mname,
            "mean_high": round(float(high_scores.mean()), 4),
            "mean_low":  round(float(low_scores.mean()),  4),
            "mw_pval":   float(pval),
        })
    mw_df = pd.DataFrame(mw_rows)

    # Combine Spearman + MW for p21 output file
    p21_df = pd.merge(
        pd.DataFrame(spearman_rows),
        mw_df,
        on="method"
    )

    # ── T3b: APAP vs Control ──────────────────────────────────────────────────
    y_apap = (adata_u48.obs["treatment"] == "APAP").values.astype(int)
    log.info(f"\n  T3b — APAP vs Control:  APAP={y_apap.sum():,}  Control={(1-y_apap).sum():,}")
    rows_t3b = report_metrics(
        "T3b: Umbaugh 48h/72h (APAP vs Control)", y_apap, scores
    )

    return rows_t3a + rows_t3b, p21_df


# =============================================================================
# 10.  FIGURES
# =============================================================================

def make_figures(df: pd.DataFrame) -> None:
    """
    Figure A: Grouped bar chart of AUROC per test × method with 95% CI.
    Figure B: AUPRC per test × method (validation tests only, T1-T3).
    """
    methods = ["trajrisk", "seeder", "b1_xgb"]
    labels  = {"trajrisk": "TrajRisk", "seeder": "Seeder (W6)",
               "b1_xgb": "B1 XGBoost"}
    colors  = {"trajrisk": "#27ae60", "seeder": "#e67e22", "b1_xgb": "#2980b9"}

    # ── Figure A: AUROC ───────────────────────────────────────────────────────
    tests  = df["test"].unique()
    n_t    = len(tests)
    x      = np.arange(n_t)
    w      = 0.24
    offs   = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(max(10, n_t * 2.2), 5.5))
    fig.suptitle(
        "TrajRisk Mouse Cross-Dataset Validation — AUROC\n"
        "(T0 = Walesky internal reference  |  T1–T3b = cross-dataset tests)",
        fontsize=11, fontweight="bold"
    )

    for mname, off in zip(methods, offs):
        aurocs, los, his = [], [], []
        for test in tests:
            row = df[(df["test"] == test) & (df["method"] == mname)]
            if len(row) == 0 or np.isnan(row["auroc"].values[0]):
                aurocs.append(0); los.append(0); his.append(0)
            else:
                a = row["auroc"].values[0]
                lo = row["auroc_ci_lo"].values[0]
                hi = row["auroc_ci_hi"].values[0]
                aurocs.append(a)
                los.append(a - lo if not np.isnan(lo) else 0)
                his.append(hi - a if not np.isnan(hi) else 0)

        ax.bar(x + off, aurocs, w, label=labels[mname],
               color=colors[mname], alpha=0.85,
               edgecolor="black", linewidth=0.6,
               yerr=[los, his],
               error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"})

    short = [t.split(":")[0].strip() for t in tests]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="red", lw=1, ls="--", alpha=0.35, label="Random (0.5)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGS / "week9_mouse_auroc.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out.name}")

    # ── Figure B: AUPRC (validation tests only) ───────────────────────────────
    val_tests = [t for t in tests if "T0" not in t]
    if not val_tests:
        return

    n_vt = len(val_tests)
    xv   = np.arange(n_vt)
    fig, ax = plt.subplots(figsize=(max(9, n_vt * 2.2), 5))
    fig.suptitle(
        "TrajRisk Mouse Cross-Dataset Validation — AUPRC\n"
        "(dashed line = random baseline per test)",
        fontsize=11, fontweight="bold"
    )

    for mname, off in zip(methods, offs):
        auprcs = []
        for test in val_tests:
            row = df[(df["test"] == test) & (df["method"] == mname)]
            auprcs.append(row["auprc"].values[0] if len(row) and not np.isnan(row["auprc"].values[0]) else 0)
        ax.bar(xv + off, auprcs, w, label=labels[mname],
               color=colors[mname], alpha=0.85, edgecolor="black", linewidth=0.6)

    for j, test in enumerate(val_tests):
        row = df[(df["test"] == test) & (df["method"] == "trajrisk")]
        if len(row):
            rand = row["auprc_rand"].values[0]
            ax.plot([j - w * 1.6, j + w * 1.6], [rand, rand],
                    "r--", alpha=0.55, lw=1.5)

    short_v = [t.split(":")[0].strip() for t in val_tests]
    ax.set_xticks(xv)
    ax.set_xticklabels(short_v, fontsize=9)
    ax.set_ylabel("AUPRC", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGS / "week9_mouse_auprc.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out.name}")


# =============================================================================
# 11.  MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()
    log.info("=" * 68)
    log.info("TRAJRISK — WEEK 9 SCRIPT 1: MOUSE CROSS-DATASET VALIDATION")
    log.info("=" * 68)

    # ── Load and verify ───────────────────────────────────────────────────────
    data = load_and_verify()

    # ── Fit B1 XGBoost ────────────────────────────────────────────────────────
    b1_xgb, b1_walesky = fit_b1_xgboost(data)
    data["b1_walesky"] = b1_walesky

    # ── Run all tests ─────────────────────────────────────────────────────────
    all_rows: List[Dict] = []

    rows_t0            = run_t0_walesky(data, b1_walesky)
    rows_t1            = run_t1_benmoshe(data, b1_xgb)
    rows_t2            = run_t2_umbaugh2h(data, b1_xgb)
    rows_t3, p21_df    = run_t3_umbaugh_p21(data, b1_xgb)

    all_rows.extend(rows_t0)
    all_rows.extend(rows_t1)
    all_rows.extend(rows_t2)
    all_rows.extend(rows_t3)

    # ── Build results dataframe ───────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("FINAL SUMMARY — TrajRisk vs Seeder vs B1 XGBoost")
    log.info("=" * 68)
    log.info(f"  {'Test':<45} {'TrajRisk':>9} {'Seeder':>9} {'B1 XGB':>9} "
             f"{'ΔTR-Seed':>10} {'ΔTR-B1':>8}")
    log.info("  " + "─" * 95)

    summary_rows = []
    for test in df["test"].unique():
        sub = df[df["test"] == test]
        tr_row = sub[sub["method"] == "trajrisk"]
        sd_row = sub[sub["method"] == "seeder"]
        b1_row = sub[sub["method"] == "b1_xgb"]

        tr_auroc = tr_row["auroc"].values[0] if len(tr_row) else float("nan")
        sd_auroc = sd_row["auroc"].values[0] if len(sd_row) else float("nan")
        b1_auroc = b1_row["auroc"].values[0] if len(b1_row) else float("nan")

        d_sd = round((tr_auroc - sd_auroc) * 100, 2) if not (np.isnan(tr_auroc) or np.isnan(sd_auroc)) else float("nan")
        d_b1 = round((tr_auroc - b1_auroc) * 100, 2) if not (np.isnan(tr_auroc) or np.isnan(b1_auroc)) else float("nan")

        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "n/a"
        def fmd(v): return f"{v:+.2f}pp" if not np.isnan(v) else "n/a"

        log.info(f"  {test:<45} {fmt(tr_auroc):>9} {fmt(sd_auroc):>9} "
                 f"{fmt(b1_auroc):>9} {fmd(d_sd):>10} {fmd(d_b1):>8}")

        summary_rows.append({
            "test": test,
            "trajrisk_auroc": tr_auroc, "seeder_auroc": sd_auroc,
            "b1_xgb_auroc": b1_auroc,
            "delta_trajrisk_vs_seeder_pp": d_sd,
            "delta_trajrisk_vs_b1_pp": d_b1,
        })

    log.info("\n  Interpretation:")
    log.info("    ΔTR-Seed > 0  →  TrajRisk adds value over expression-only seeder")
    log.info("    ΔTR-B1  improves (less negative) T1-T3 vs T0")
    log.info("    → circularity broke on independent data → generalisation claim holds")

    # ── Save all outputs ──────────────────────────────────────────────────────
    log.info("\nSaving outputs ...")
    out_table = RESULTS / "week9_validation_table.csv"
    df.to_csv(out_table, index=False, float_format="%.6f")
    log.info(f"  {out_table.name}")

    out_p21 = RESULTS / "week9_p21_analysis.csv"
    p21_df.to_csv(out_p21, index=False, float_format="%.6f")
    log.info(f"  {out_p21.name}")

    out_sum = RESULTS / "week9_summary.json"
    with open(out_sum, "w") as f:
        json.dump({"week": 9, "script": "mouse_cross_dataset",
                   "tests": summary_rows}, f, indent=2, default=str)
    log.info(f"  {out_sum.name}")

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("\nGenerating figures ...")
    make_figures(df)

    elapsed = time.time() - t_start
    log.info(f"\nScript completed in {elapsed/60:.1f} minutes.")
    log.info("Next: Script 2 — week9_human_cross_species.py")


if __name__ == "__main__":
    main()
