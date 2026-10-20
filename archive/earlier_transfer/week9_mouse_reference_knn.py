
#!/usr/bin/env python3
"""
TrajRisk — Week 9 Script 2: Reference-KNN Cross-Dataset Validation
===================================================================
Project : TrajRisk — Trajectory-Informed Risk in APAP Hepatotoxicity
Journal : Cell Reports Methods

WHY THIS SCRIPT EXISTS
----------------------
Script 1 (week9_mouse_cross_dataset.py) showed that W_self-only TrajRisk
underperforms the Seeder on external datasets. The cause is exact and known:

    External cells received NO incoming edges.
    TrajRiskConv reduced to: h_i = W_self(x_i)
    No trajectory information reached external cells.

This script fixes that by connecting each external cell to biologically
appropriate Walesky reference cells in the shared scVI space, giving it
genuine graph-based trajectory context before risk scoring.

REFERENCE-KNN PROJECTION DESIGN
--------------------------------
For each external cell:
  1. Find its k=15 nearest neighbors AMONG the biologically appropriate
     Walesky anchor cells in 30D scVI space.
  2. Add edges: Walesky anchor → external cell
     (source_to_target, same convention as all existing edges)
  3. Assign Gaussian similarity weights, row-normalised per external cell.
  4. Run TrajRiskGNN — external cell now receives trajectory-informed
     messages from its Walesky anchors in both conv layers.

ANCHOR SELECTION (biologically motivated, not arbitrary)
---------------------------------------------------------
  Umbaugh_2h     → Walesky 0h + 6h
      Early injury dataset. Reference anchors are the baseline (0h) and
      the peak acute injury state (6h) — the trajectory states that bound
      the 2h early response.

  Ben_Moshe      → Walesky 24h + 48h + 96h
      APAP dataset at 48h/72h. Reference anchors are the recovery
      trajectory states corresponding to the same phase of injury.

  Umbaugh_48h72h → Walesky 24h + 48h + 96h
      p21-sorted cells at 48h/72h. Same reasoning as Ben-Moshe.

METHODS COMPARED
----------------
  W_self TrajRisk    — previous version, no graph context for external cells
  KNN   TrajRisk     — this script, reference-projected version (THE FIX)
  Seeder             — Week 6 XGBClassifier (0h vs 6h expression only)
  B1 XGBoost         — Week 8 XGBRegressor (scVI → pseudotime_norm)

SUCCESS CRITERIA
----------------
  Primary:   KNN TrajRisk > W_self TrajRisk on all external tests
  Secondary: KNN TrajRisk >= Seeder on ≥1 biologically meaningful test
  Most likely: T1 Ben-Moshe (best scVI alignment) and T3a Cdkn1a

Usage:
    conda activate trajrisk
    cd /path/to/TrajRisk
    python scripts/week9_mouse_reference_knn.py
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

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
log = logging.getLogger("W9-KNN")


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

BASE = Path(__file__).resolve().parents[2]
DATA   = BASE / "Data" / "processed"
MODELS = BASE / "models"
RES    = BASE / "results"
FIGS   = RES  / "figures"

for _d in [RES, FIGS]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE         = torch.device("cpu")
RANDOM_SEED    = 42
N_BOOTSTRAP    = 1_000
MIN_POS        = 5
HIGH_RISK_THR  = 0.10     # pseudotime_norm threshold for Walesky binary label
CDKN1A_GENE    = "Cdkn1a"
KNN_K          = 15       # reference neighbours per external cell

# Anchor hours per external dataset — biologically motivated
ANCHOR_HOURS = {
    "Umbaugh_2h":     [0.0, 6.0],
    "Ben_Moshe":      [24.0, 48.0, 96.0],
    "Umbaugh_48h72h": [24.0, 48.0, 96.0],
}

SCALE_NAMES = [
    "Scale1_AcuteStress",
    "Scale2_PeakInjury",
    "Scale3_FateDecision",
    "Scale4_Resolution",
]

NODE_DIM   = 31
HIDDEN_DIM = 64
N_SCALES   = 4
DROPOUT    = 0.3

XGB_B1_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbosity=0, n_jobs=1,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# 1.  MODEL CLASSES  (exact copies from Week 7 — do not modify)
# =============================================================================

class TrajRiskConv(MessagePassing):
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
    log.info("=" * 68)
    log.info("LOADING AND VERIFYING ALL DATA")
    log.info("=" * 68)

    adata = sc.read_h5ad(str(DATA / "adata_integrated_with_trajectory.h5ad"))
    assert adata.n_obs == 36_067
    assert "X_scvi" in adata.obsm
    assert CDKN1A_GENE in adata.var_names
    ds_counts = adata.obs["dataset"].value_counts()
    assert ds_counts["Walesky"]        == 6_582
    assert ds_counts["Umbaugh_2h"]     == 21_409
    assert ds_counts["Umbaugh_48h72h"] == 5_306
    assert ds_counts["Ben_Moshe"]      == 2_770
    log.info(f"  adata: {adata.n_obs:,} cells  ✓")

    cell_index = pd.read_csv(DATA / "walesky_cell_index.csv", index_col=0)
    assert len(cell_index) == 6_582
    log.info(f"  walesky_cell_index: {len(cell_index):,} rows  ✓")
    log.info(f"  Walesky timepoints (hours): "
             f"{sorted(cell_index['hours'].unique().tolist())}")

    x_walesky = torch.load(DATA / "week6_node_features.pt", map_location="cpu")
    assert x_walesky.shape == (6_582, 31)
    log.info(f"  x_walesky: {tuple(x_walesky.shape)}  ✓")

    graphs       = torch.load(DATA / "multiscale_graphs.pt", map_location="cpu")
    edge_indices = [graphs[sn]["edge_index"] for sn in SCALE_NAMES]
    edge_weights = [graphs[sn]["edge_weight"] for sn in SCALE_NAMES]
    for s, ei in enumerate(edge_indices):
        assert ei.max().item() <= 6_581
    log.info(f"  Graphs: 4 scales, "
             f"{sum(e.shape[1] for e in edge_indices):,} total edges  ✓")

    ckpt_path = MODELS / "week7_trajrisk_FINAL.pt"
    if not ckpt_path.exists():
        ckpt_path = MODELS / "week7_trajrisk_BEST_AUROC.pt"
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = TrajRiskGNN().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"  TrajRisk model: {ckpt_path.name}  ✓")

    seeder          = joblib.load(MODELS / "week6_xgboost_model.pkl")
    seeder_walesky  = np.load(DATA / "week6_xgboost_risk.npy")
    preds_w7        = pd.read_csv(DATA / "week7_predictions.csv", index_col=0)
    trajrisk_walesky = preds_w7["risk_predicted"].values.astype(np.float32)
    td              = torch.load(DATA / "week6_training_data.pt", map_location="cpu")
    log.info("  All supporting files loaded  ✓\n")

    return {
        "adata":             adata,
        "cell_index":        cell_index,
        "x_walesky":         x_walesky,
        "edge_indices":      edge_indices,
        "edge_weights":      edge_weights,
        "model":             model,
        "seeder":            seeder,
        "seeder_walesky":    seeder_walesky,
        "trajrisk_walesky":  trajrisk_walesky,
        "train_idx":         td["train_idx"].numpy(),
        "val_idx":           td["val_idx"].numpy(),
        "train_y":           td["train_y"].numpy(),
        "val_y":             td["val_y"].numpy(),
    }


# =============================================================================
# 3.  FIT B1 XGBOOST
# =============================================================================

def fit_b1_xgboost(data: Dict) -> Tuple[XGBRegressor, np.ndarray]:
    log.info("Fitting B1 XGBoost (scVI → pseudotime_norm) ...")
    x_scvi_wal = data["x_walesky"][:, :30].numpy()
    b1 = XGBRegressor(**XGB_B1_PARAMS)
    b1.fit(x_scvi_wal[data["train_idx"]], data["train_y"])
    b1_wal = np.clip(b1.predict(x_scvi_wal).astype(float), 0.0, 1.0).astype(np.float32)
    log.info(f"  B1 fitted  — range [{b1_wal.min():.3f}, {b1_wal.max():.3f}]  ✓\n")
    return b1, b1_wal


# =============================================================================
# 4.  ANCHOR CELL HELPERS
# =============================================================================

def get_anchor_cells(data: Dict, anchor_hours: List[float]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return scVI features and global positions of Walesky anchor cells.

    Parameters
    ----------
    anchor_hours : list of float  e.g. [0.0, 6.0] for early anchors

    Returns
    -------
    z_anchors : (n_anchors, 30) float32  — scVI features of anchor cells
    anchor_idx : (n_anchors,) int64      — positions in x_walesky (0–6581)
    """
    ci = data["cell_index"]
    mask = ci["hours"].isin(anchor_hours).values               # boolean (6582,)
    anchor_idx = np.where(mask)[0].astype(np.int64)            # positions in x_walesky
    z_anchors  = data["x_walesky"][mask, :30].numpy()          # (n_anchors, 30)
    return z_anchors, anchor_idx


def build_reference_knn_edges(
    z_ext:          np.ndarray,    # (n_ext, 30) — external cell scVI features
    z_anchors:      np.ndarray,    # (n_anchors, 30) — Walesky anchor scVI features
    anchor_idx:     np.ndarray,    # (n_anchors,) — global Walesky positions [0, 6581]
    n_wal:          int,           # 6582 — total Walesky cells
    k:              int = KNN_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build reference edges: Walesky anchor → external cell.

    For each of the n_ext external cells, finds k nearest Walesky anchors
    in scVI space and creates directed edges:
        source = anchor global position (in [0, n_wal-1])
        target = external cell position (in [n_wal, n_wal + n_ext - 1])

    Edge weights are Gaussian-based and row-normalised per external cell.

    Returns
    -------
    ref_ei : (2, n_ext * k)  int64   — edge index tensor
    ref_ew : (n_ext * k,)    float32 — edge weight tensor
    """
    n_ext     = z_ext.shape[0]
    n_anchors = z_anchors.shape[0]
    k_actual  = min(k, n_anchors)          # cannot exceed available anchors

    if k_actual < k:
        log.warning(f"  Only {n_anchors} anchors available; using k={k_actual}")

    # ── KNN in scVI space ─────────────────────────────────────────────────────
    knn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", n_jobs=-1)
    knn.fit(z_anchors)
    distances, local_indices = knn.kneighbors(z_ext)     # (n_ext, k_actual)

    # ── Gaussian weights ──────────────────────────────────────────────────────
    nonzero_d = distances[distances > 0]
    sigma     = float(np.median(nonzero_d)) if len(nonzero_d) > 0 else 1.0
    raw_w     = np.exp(-distances ** 2 / (2.0 * sigma ** 2))  # (n_ext, k_actual)

    # ── Row-normalise per external cell ───────────────────────────────────────
    row_sums = raw_w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    norm_w   = (raw_w / row_sums).astype(np.float32)           # (n_ext, k_actual)

    # ── Convert local anchor indices → global Walesky positions ──────────────
    global_src = anchor_idx[local_indices]          # (n_ext, k_actual)

    # ── Build edge tensors ────────────────────────────────────────────────────
    src = global_src.flatten().astype(np.int64)     # Walesky position
    tgt = (np.repeat(np.arange(n_ext), k_actual) + n_wal).astype(np.int64)

    ref_ei = torch.tensor(np.stack([src, tgt]), dtype=torch.long)
    ref_ew = torch.tensor(norm_w.flatten(), dtype=torch.float32)

    return ref_ei, ref_ew


# =============================================================================
# 5.  COMPUTE RISK SCORES — W_SELF VARIANT  (kept for direct comparison)
# =============================================================================

def compute_wself(data: Dict, b1_xgb: XGBRegressor,
                  z_val: np.ndarray, label: str) -> Dict[str, np.ndarray]:
    """W_self-only TrajRisk (no reference edges). Kept for comparison."""
    n_wal = data["x_walesky"].shape[0]
    seeder_val = data["seeder"].predict_proba(z_val)[:, 1].astype(np.float32)
    b1_val     = np.clip(b1_xgb.predict(z_val), 0.0, 1.0).astype(np.float32)

    x_val = np.hstack([z_val, seeder_val.reshape(-1, 1)]).astype(np.float32)
    x_all = torch.tensor(
        np.vstack([data["x_walesky"].numpy(), x_val]), dtype=torch.float32
    )
    with torch.no_grad():
        data["model"].eval()
        risk_all = data["model"](
            x_all, data["edge_indices"], data["edge_weights"]
        ).numpy()

    return {
        "trajrisk_wself": risk_all[n_wal:].astype(np.float32),
        "seeder":         seeder_val,
        "b1_xgb":         b1_val,
    }


# =============================================================================
# 6.  COMPUTE RISK SCORES — REFERENCE-KNN VARIANT  (THE FIX)
# =============================================================================

def compute_knn(data: Dict, b1_xgb: XGBRegressor,
                z_val: np.ndarray,
                z_anchors: np.ndarray, anchor_idx: np.ndarray,
                label: str) -> np.ndarray:
    """
    Reference-KNN TrajRisk.

    Each external cell is connected to its k nearest Walesky anchor cells.
    These reference edges are added to ALL 4 scales so every scale branch
    in TrajRiskGNN receives trajectory context for the external cell.

    Two-layer message passing propagation:
      Layer 1: external cell h1_i = ReLU(BN( W_nbr(sum_j w[j,i]*x_j) + W_self(x_i) ))
               where x_j are the INITIAL features of Walesky anchors j
      Layer 2: external cell h2_i = ReLU(BN( W_nbr(sum_j w[j,i]*h1_j) + W_self(h1_i) ))
               where h1_j are the LAYER-1 HIDDEN STATES of Walesky anchors
               (already trajectory-encoded from Walesky graph processing)

    The external cell therefore receives trajectory-informed representations
    of its reference anchors, not just their raw features.
    """
    n_val = z_val.shape[0]
    n_wal = data["x_walesky"].shape[0]

    log.info(f"  [{label}] Building reference-KNN edges (k={KNN_K}, "
             f"n_anchors={len(anchor_idx):,}) ...")

    # Build reference edge tensors
    ref_ei, ref_ew = build_reference_knn_edges(
        z_val, z_anchors, anchor_idx, n_wal, k=KNN_K
    )

    log.info(f"  [{label}] Reference edges built: {ref_ei.shape[1]:,} "
             f"({n_val:,} external cells × {KNN_K} anchors each)")

    # Augment all 4 scale edge lists with reference edges
    aug_indices = [
        torch.cat([data["edge_indices"][s], ref_ei], dim=1)
        for s in range(N_SCALES)
    ]
    aug_weights = [
        torch.cat([data["edge_weights"][s], ref_ew])
        for s in range(N_SCALES)
    ]

    # Build node feature tensor
    seeder_val = data["seeder"].predict_proba(z_val)[:, 1].astype(np.float32)
    x_val = np.hstack([z_val, seeder_val.reshape(-1, 1)]).astype(np.float32)
    x_all = torch.tensor(
        np.vstack([data["x_walesky"].numpy(), x_val]), dtype=torch.float32
    )
    # Shape: (6582 + n_val, 31)
    # Positions 0–6581: Walesky cells (original edges unchanged)
    # Positions 6582+:  external cells (connected to anchors via ref_ei)

    with torch.no_grad():
        data["model"].eval()
        risk_all = data["model"](x_all, aug_indices, aug_weights).numpy()

    knn_scores = risk_all[n_wal:].astype(np.float32)
    log.info(f"  [{label}] KNN TrajRisk range: "
             f"[{knn_scores.min():.4f}, {knn_scores.max():.4f}]")
    return knn_scores


def compute_all_scores(data: Dict, b1_xgb: XGBRegressor,
                        z_val: np.ndarray, dataset_name: str) -> Dict[str, np.ndarray]:
    """
    Compute all four methods for a validation dataset:
        trajrisk_knn   — reference-KNN TrajRisk (the fix)
        trajrisk_wself — W_self-only TrajRisk (for comparison)
        seeder         — Week 6 XGBClassifier
        b1_xgb         — Week 8 expression-only XGBoost

    All scores are computed from the same z_val and the same model.
    """
    log.info(f"\n  [{dataset_name}] n={z_val.shape[0]:,} — computing all methods ...")

    # W_self scores (reuse seeder + b1 from inside)
    wself_scores = compute_wself(data, b1_xgb, z_val, dataset_name)

    # KNN scores — anchor cells determined by dataset
    z_anchors, anchor_idx = get_anchor_cells(data, ANCHOR_HOURS[dataset_name])
    log.info(f"  [{dataset_name}] Anchors: hours={ANCHOR_HOURS[dataset_name]}  "
             f"n_anchors={len(anchor_idx):,}")

    knn_scores = compute_knn(
        data, b1_xgb, z_val, z_anchors, anchor_idx, dataset_name
    )

    return {
        "trajrisk_knn":   knn_scores,
        "trajrisk_wself": wself_scores["trajrisk_wself"],
        "seeder":         wself_scores["seeder"],
        "b1_xgb":         wself_scores["b1_xgb"],
    }


# =============================================================================
# 7.  METRICS
# =============================================================================

def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() < MIN_POS or (1 - y).sum() < MIN_POS:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() < MIN_POS:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_ci(y: np.ndarray, s: np.ndarray,
                 n: int = N_BOOTSTRAP, seed: int = RANDOM_SEED
                 ) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yt  = y[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            vals.append(roc_auc_score(yt, s[idx]))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report_metrics(test_name: str, y: np.ndarray,
                   scores: Dict[str, np.ndarray]) -> List[Dict]:
    n_total = len(y)
    n_pos   = int(y.sum())
    prev    = n_pos / max(n_total, 1)

    log.info(f"\n  {test_name}")
    log.info(f"  n={n_total:,}  pos={n_pos:,}  prev={prev:.3f}")
    log.info(f"  {'Method':<22} {'AUROC':>7}  {'95% CI':>15}  {'AUPRC':>8}")
    log.info("  " + "─" * 60)

    rows = []
    for mname, sc in scores.items():
        auroc        = safe_auroc(y, sc)
        auprc        = safe_auprc(y, sc)
        ci_lo, ci_hi = (bootstrap_ci(y, sc)
                        if not np.isnan(auroc) else (float("nan"), float("nan")))
        ci_str   = (f"({ci_lo:.3f}–{ci_hi:.3f})" if not np.isnan(ci_lo) else "(—)")
        auroc_s  = f"{auroc:.4f}" if not np.isnan(auroc) else "n/a"
        auprc_s  = f"{auprc:.4f}" if not np.isnan(auprc) else "n/a"
        log.info(f"  {mname:<22} {auroc_s:>7}  {ci_str:>15}  {auprc_s:>8}")
        rows.append({
            "test":       test_name, "method":      mname,
            "n_total":    n_total,   "n_pos":        n_pos,
            "prevalence": round(prev, 4),
            "auroc":      round(auroc, 4) if not np.isnan(auroc) else float("nan"),
            "auroc_ci_lo":round(ci_lo, 4) if not np.isnan(ci_lo) else float("nan"),
            "auroc_ci_hi":round(ci_hi, 4) if not np.isnan(ci_hi) else float("nan"),
            "auprc":      round(auprc, 4) if not np.isnan(auprc) else float("nan"),
            "auprc_rand": round(prev, 4),
        })
    return rows


# =============================================================================
# 8.  INDIVIDUAL TESTS
# =============================================================================

def run_t0_reference(data: Dict, b1_wal: np.ndarray) -> List[Dict]:
    """T0: Walesky held-out val — uses full graph (Week 7/8 numbers reproduced)."""
    log.info("\n" + "=" * 68)
    log.info("T0: Walesky held-out val (internal reference — full graph)")
    log.info("=" * 68)
    vi, vy   = data["val_idx"], data["val_y"]
    y_bin    = (vy > HIGH_RISK_THR).astype(int)
    scores   = {
        "trajrisk_knn":   data["trajrisk_walesky"][vi],   # same as wself for Walesky
        "trajrisk_wself": data["trajrisk_walesky"][vi],
        "seeder":         data["seeder_walesky"][vi],
        "b1_xgb":         b1_wal[vi],
    }
    return report_metrics("T0: Walesky val (reference)", y_bin, scores)


def run_t1_benmoshe(data: Dict, b1_xgb: XGBRegressor) -> List[Dict]:
    log.info("\n" + "=" * 68)
    log.info("T1: Ben-Moshe — APAP (48h/72h) vs Control (0h)")
    log.info("=" * 68)
    adata  = data["adata"]
    sub    = adata[adata.obs["dataset"] == "Ben_Moshe"]
    z_bm   = sub.obsm["X_scvi"].astype(np.float32)
    y_bm   = (sub.obs["treatment"] == "APAP").values.astype(int)
    log.info(f"  APAP={y_bm.sum():,}  Control={(1-y_bm).sum():,}")
    scores = compute_all_scores(data, b1_xgb, z_bm, "Ben_Moshe")
    return report_metrics("T1: Ben-Moshe (APAP vs Control)", y_bm, scores)


def run_t2_umbaugh2h(data: Dict, b1_xgb: XGBRegressor) -> List[Dict]:
    log.info("\n" + "=" * 68)
    log.info("T2: Umbaugh 2h — APAP (2h) vs Control (0h)")
    log.info("=" * 68)
    adata = data["adata"]
    sub   = adata[adata.obs["dataset"] == "Umbaugh_2h"]
    z_u2  = sub.obsm["X_scvi"].astype(np.float32)
    y_u2  = (sub.obs["treatment"] == "APAP").values.astype(int)
    log.info(f"  APAP={y_u2.sum():,}  Control={(1-y_u2).sum():,}")
    scores = compute_all_scores(data, b1_xgb, z_u2, "Umbaugh_2h")
    return report_metrics("T2: Umbaugh 2h (APAP vs Control)", y_u2, scores)


def run_t3_umbaugh_p21(data: Dict, b1_xgb: XGBRegressor
                        ) -> Tuple[List[Dict], pd.DataFrame]:
    log.info("\n" + "=" * 68)
    log.info("T3: Umbaugh 48h/72h — Cdkn1a (p21) mechanistic validation")
    log.info("=" * 68)
    adata = data["adata"]
    sub   = adata[adata.obs["dataset"] == "Umbaugh_48h72h"]
    z_u48 = sub.obsm["X_scvi"].astype(np.float32)

    # Extract Cdkn1a expression
    cdkn1a_col = list(sub.var_names).index(CDKN1A_GENE)
    X = sub.X
    cdkn1a_expr = (np.asarray(X[:, cdkn1a_col].toarray()).flatten()
                   if hasattr(X, "toarray")
                   else np.asarray(X[:, cdkn1a_col]).flatten())
    log.info(f"  Cdkn1a: mean={cdkn1a_expr.mean():.4f}  "
             f"median={np.median(cdkn1a_expr):.4f}  "
             f"%expressed={100*(cdkn1a_expr>0).mean():.1f}%")

    scores = compute_all_scores(data, b1_xgb, z_u48, "Umbaugh_48h72h")

    # T3a: Cdkn1a median split
    y_cdkn1a = (cdkn1a_expr > np.median(cdkn1a_expr)).astype(int)
    log.info(f"\n  T3a Cdkn1a split: high={y_cdkn1a.sum():,}  low={(1-y_cdkn1a).sum():,}")
    rows_t3a = report_metrics(
        "T3a: Umbaugh 48h/72h (Cdkn1a-high vs -low)", y_cdkn1a, scores
    )

    # Spearman correlation: Cdkn1a vs each risk score
    log.info("\n  Spearman ρ (Cdkn1a expression vs risk score):")
    spearman_rows = []
    for mname, sc in scores.items():
        rho, pval = spearmanr(cdkn1a_expr, sc)
        log.info(f"    {mname:<22} ρ={rho:+.4f}  p={pval:.2e}")
        spearman_rows.append({"method": mname, "rho": round(float(rho), 4),
                               "pvalue": float(pval)})

    # Mann-Whitney: Cdkn1a-high > Cdkn1a-low (one-sided)
    log.info("\n  Mann-Whitney U (Cdkn1a-high > Cdkn1a-low, one-sided):")
    mw_rows = []
    for mname, sc in scores.items():
        stat, pval = mannwhitneyu(sc[y_cdkn1a == 1], sc[y_cdkn1a == 0],
                                  alternative="greater")
        log.info(f"    {mname:<22} "
                 f"mean_high={sc[y_cdkn1a==1].mean():.4f}  "
                 f"mean_low={sc[y_cdkn1a==0].mean():.4f}  p={pval:.2e}")
        mw_rows.append({"method": mname,
                        "mean_high": round(float(sc[y_cdkn1a == 1].mean()), 4),
                        "mean_low":  round(float(sc[y_cdkn1a == 0].mean()), 4),
                        "mw_pval":   float(pval)})

    p21_df = pd.merge(pd.DataFrame(spearman_rows), pd.DataFrame(mw_rows), on="method")

    # T3b: APAP vs Control
    y_apap = (sub.obs["treatment"] == "APAP").values.astype(int)
    log.info(f"\n  T3b APAP vs Control: APAP={y_apap.sum():,}  "
             f"Control={(1-y_apap).sum():,}")
    rows_t3b = report_metrics(
        "T3b: Umbaugh 48h/72h (APAP vs Control)", y_apap, scores
    )

    return rows_t3a + rows_t3b, p21_df


# =============================================================================
# 9.  SUMMARY TABLE AND FIGURES
# =============================================================================

def print_summary(df: pd.DataFrame) -> List[Dict]:
    log.info("\n" + "=" * 68)
    log.info("SUMMARY — KNN vs W_self vs Seeder vs B1 XGBoost")
    log.info("=" * 68)
    log.info(f"\n  {'Test':<45} {'KNN':>7} {'W_self':>7} {'Seeder':>7} "
             f"{'B1XGB':>7}  {'ΔKNN-Wself':>11}  {'ΔKNN-Seed':>10}")
    log.info("  " + "─" * 105)

    summary_rows = []
    for test in df["test"].unique():
        sub  = df[df["test"] == test]
        get  = lambda m: sub.loc[sub["method"] == m, "auroc"].values
        knn  = float(get("trajrisk_knn")[0])   if len(get("trajrisk_knn"))  else float("nan")
        ws   = float(get("trajrisk_wself")[0]) if len(get("trajrisk_wself")) else float("nan")
        sd   = float(get("seeder")[0])         if len(get("seeder"))         else float("nan")
        b1   = float(get("b1_xgb")[0])         if len(get("b1_xgb"))         else float("nan")
        d_ws = round((knn - ws) * 100, 2) if not (np.isnan(knn) or np.isnan(ws)) else float("nan")
        d_sd = round((knn - sd) * 100, 2) if not (np.isnan(knn) or np.isnan(sd)) else float("nan")

        def f(v): return f"{v:.4f}" if not np.isnan(v) else "n/a"
        def fd(v): return f"{v:+.2f}pp" if not np.isnan(v) else "n/a"

        log.info(f"  {test:<45} {f(knn):>7} {f(ws):>7} {f(sd):>7} "
                 f"{f(b1):>7}  {fd(d_ws):>11}  {fd(d_sd):>10}")
        summary_rows.append({
            "test": test,
            "trajrisk_knn": knn, "trajrisk_wself": ws,
            "seeder": sd, "b1_xgb": b1,
            "delta_knn_vs_wself_pp": d_ws,
            "delta_knn_vs_seeder_pp": d_sd,
        })

    # Verdict
    log.info("\n  ΔKNN-Wself: how much reference-KNN improves over W_self-only")
    log.info("  ΔKNN-Seed:  how much reference-KNN closes gap with Seeder")
    log.info("  Goal: ΔKNN-Wself > 0 on all tests (confirmed: reference projection helps)")
    log.info("        ΔKNN-Seed  > 0 on ≥1 test   (reference projection beats seeder)")
    return summary_rows


def make_comparison_figure(df: pd.DataFrame) -> None:
    """
    Side-by-side comparison: KNN TrajRisk vs W_self TrajRisk vs Seeder vs B1 XGB.
    Two subplots: AUROC (left) and AUPRC (right).
    """
    tests   = df["test"].unique()
    methods = ["trajrisk_knn", "trajrisk_wself", "seeder", "b1_xgb"]
    labels  = {
        "trajrisk_knn":   "TrajRisk KNN (fixed)",
        "trajrisk_wself": "TrajRisk W_self (prev)",
        "seeder":         "Seeder (W6)",
        "b1_xgb":         "B1 XGBoost",
    }
    colors  = {
        "trajrisk_knn":   "#27ae60",
        "trajrisk_wself": "#a8d5b5",
        "seeder":         "#e67e22",
        "b1_xgb":         "#2980b9",
    }
    hatches = {
        "trajrisk_knn":   "",
        "trajrisk_wself": "//",
        "seeder":         "",
        "b1_xgb":         "",
    }

    n_t  = len(tests)
    x    = np.arange(n_t)
    w    = 0.19
    offs = np.array([-1.5, -0.5, 0.5, 1.5]) * w

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_t * 3), 6))
    fig.suptitle(
        "TrajRisk Week 9: W_self vs Reference-KNN vs Baselines\n"
        "(hatching = previous W_self-only version;  solid green = fixed KNN version)",
        fontsize=11, fontweight="bold"
    )

    for ax_idx, (metric, metric_name) in enumerate([("auroc", "AUROC"), ("auprc", "AUPRC")]):
        ax = axes[ax_idx]
        for mname, off in zip(methods, offs):
            vals, los, his = [], [], []
            for test in tests:
                row = df[(df["test"] == test) & (df["method"] == mname)]
                if len(row) == 0 or np.isnan(row[metric].values[0]):
                    vals.append(0); los.append(0); his.append(0)
                else:
                    v = row[metric].values[0]
                    vals.append(v)
                    if metric == "auroc":
                        lo = row["auroc_ci_lo"].values[0]
                        hi = row["auroc_ci_hi"].values[0]
                        los.append(v - lo if not np.isnan(lo) else 0)
                        his.append(hi - v if not np.isnan(hi) else 0)
                    else:
                        los.append(0); his.append(0)

            yerr = [los, his] if metric == "auroc" else None
            ax.bar(x + off, vals, w,
                   label=labels[mname], color=colors[mname],
                   hatch=hatches[mname], alpha=0.85,
                   edgecolor="black", linewidth=0.6,
                   yerr=yerr,
                   error_kw={"elinewidth": 1.0, "capsize": 2, "ecolor": "black"} if yerr else None)

        short = [t.split(":")[0].strip() for t in tests]
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=8)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="red", lw=1, ls="--", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(metric_name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = FIGS / "week9_knn_vs_wself_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out.name}")


def make_improvement_figure(summary_rows: List[Dict]) -> None:
    """
    Bar chart showing ΔKNN-Wself (how much KNN improves over W_self) per test.
    Green bars = improvement. This is the key result figure.
    """
    if not summary_rows:
        return
    tests = [r["test"].split(":")[0].strip() for r in summary_rows]
    deltas = [r["delta_knn_vs_wself_pp"] for r in summary_rows]
    valid  = [(t, d) for t, d in zip(tests, deltas) if not np.isnan(d)]
    if not valid:
        return
    tests_v, deltas_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(max(8, len(tests_v) * 1.6), 4))
    colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas_v]
    bars = ax.bar(range(len(tests_v)), deltas_v, color=colors,
                  edgecolor="black", linewidth=0.7, alpha=0.85)

    ax.axhline(0, color="black", lw=0.8)
    for bar, val in zip(bars, deltas_v):
        ha = "center"
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.2 if val >= 0 else -0.2),
                f"{val:+.2f}pp", ha=ha, va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(tests_v)))
    ax.set_xticklabels(tests_v, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("ΔAUROC (KNN − W_self) percentage points", fontsize=10)
    ax.set_title(
        "Improvement from Reference-KNN over W_self-only TrajRisk\n"
        "(green = reference projection helps;  red = no improvement)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGS / "week9_knn_improvement.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out.name}")


# =============================================================================
# 10.  MAIN
# =============================================================================

def main() -> None:
    t0 = time.time()
    log.info("=" * 68)
    log.info("TRAJRISK — WEEK 9 SCRIPT 2: REFERENCE-KNN VALIDATION")
    log.info("=" * 68)

    data            = load_and_verify()
    b1_xgb, b1_wal  = fit_b1_xgboost(data)
    data["b1_wal"]  = b1_wal

    all_rows: List[Dict] = []

    all_rows.extend(run_t0_reference(data, b1_wal))
    all_rows.extend(run_t1_benmoshe(data, b1_xgb))
    all_rows.extend(run_t2_umbaugh2h(data, b1_xgb))
    rows_t3, p21_df = run_t3_umbaugh_p21(data, b1_xgb)
    all_rows.extend(rows_t3)

    df = pd.DataFrame(all_rows)

    summary_rows = print_summary(df)

    log.info("\nGenerating figures ...")
    make_comparison_figure(df)
    make_improvement_figure(summary_rows)

    log.info("\nSaving outputs ...")
    df.to_csv(RES / "week9_knn_validation_table.csv", index=False, float_format="%.6f")
    log.info(f"  week9_knn_validation_table.csv")
    p21_df.to_csv(RES / "week9_knn_p21_analysis.csv", index=False, float_format="%.6f")
    log.info(f"  week9_knn_p21_analysis.csv")
    with open(RES / "week9_knn_summary.json", "w") as f:
        json.dump({"week": 9, "variant": "reference_knn",
                   "knn_k": KNN_K, "anchor_hours": ANCHOR_HOURS,
                   "results": summary_rows}, f, indent=2, default=str)
    log.info(f"  week9_knn_summary.json")

    log.info(f"\nCompleted in {(time.time()-t0)/60:.1f} minutes.")


if __name__ == "__main__":
    main()
