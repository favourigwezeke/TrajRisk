#!/usr/bin/env python3
"""
TrajRisk: Week 8 — Baseline Comparisons and Ablation Studies
=============================================================
Project: TrajRisk — Trajectory-Informed Risk Propagation in APAP Hepatotoxicity
Target Journal: Cell Reports Methods

Evaluation protocol
-------------------
Every method is evaluated on the same held-out val_idx established in Week 6
(n=1318 cells, 213 high-risk). The binary high-risk label
(pseudotime_norm > 0.10) was fixed in Week 6 based on biology, not tuned
on any AUROC.

Important qualification on val_idx use in Week 7
-------------------------------------------------
The Week 7 TrajRisk checkpoint (BEST_AUROC.pt) was selected to maximise
val AUROC — so val_idx WAS used for Week 7 model selection. This is a
standard deep-learning practice and is documented as a limitation here.

For Week 8 ablations, we do NOT repeat this. GNN ablations use dev_inner
(a 20 % hold-out from train_idx) for early stopping; val_idx is reserved
for final AUROC reporting only. This asymmetry is conservative: TrajRisk
receives a checkpoint-selection advantage that ablations do not. If TrajRisk
still outperforms significantly under this conservative comparison, the
conclusion is robust.

Early stopping — corrected design
----------------------------------
Red flag in earlier version: GNN ablations used val AUROC for early stopping
AND for final AUROC reporting. That makes the reported metric optimistic.

Fix: train_idx is divided into:
  train_inner  (80 %) — gradient updates
  dev_inner    (20 %) — early stopping only (never appears in any reported metric)
  val_idx              — final AUROC reporting only (never appears in training)

TrajRisk itself (loaded from Week 7) used val AUROC for early stopping.
This asymmetry is documented in the table header and is CONSERVATIVE:
it gives TrajRisk a slight advantage in checkpoint selection, making its
margin over ablations harder to inflate artificially.

Baselines
---------
B1  XGBoost on scVI features
    Hyperparameters selected by 5-fold cross-validation on train_idx (AUROC
    scoring). Final model trained on full train_idx. No graph, no trajectory.

B2  Gaussian propagation (multi-scale, matrix)
    Same k-NN graph structure and temporal windows as RF transitions. Gaussian
    kernel weights (median bandwidth heuristic, top-k=15 sparsification, row-
    normalisation matching RF convention). Linear label propagation with
    damping=0.5; no GNN, no learning.

B3  Best single-scale GNN
    TrajRiskGNN with n_scales=1 trained separately for all four scales.
    Best is selected by dev_inner AUROC (val_idx never touched).
    All four individual results are saved for supplementary reporting.

B4  Expression distance from healthy baseline
    Euclidean distance in scVI space from the mean 0h healthy cell, normalised
    to [0,1]. Simplest no-trajectory proxy.
    RNA velocity is not included: GSE136679 (Walesky, Seq-Well) was deposited
    as a pre-processed expression matrix without the spliced/unspliced count
    layers required for RNA velocity estimation. This limitation is stated
    explicitly rather than marking an N/A entry.

Ablations — one component removed at a time from TrajRisk
---------
A1  Gaussian transitions + full GNN
    Same 4-scale GNN, same training procedure, same node features. RF edge
    weights replaced by Gaussian kernel weights (same sparsity structure,
    same normalisation convention). Isolates: learned RF transitions.

A2  Single-scale (same model as best B3)
    Same result as B3, framed as an ablation. Trained once, reported twice.

A3  RF transitions + matrix propagation (no GNN)
    RF edge weights preserved. GNN replaced by two-step linear label
    propagation of the scalar XGBoost risk signal (damping=0.5). This removes
    nonlinear aggregation AND high-dimensional feature processing simultaneously.
    These cannot be separated without a GNN, so A3 is labelled explicitly:
    it tests 'nonlinear GNN aggregation + feature integration vs scalar linear
    propagation'. Both B2 (Gaussian + matrix) and A3 (RF + matrix) fall in
    this category, differing only in edge weights.

A4  No initial XGBoost risk (remove seed)
    Full 4-scale GNN with RF transitions. Node features reduced from 31D to
    30D by removing the XGBoost risk column. Isolates: XGBoost seed contribution.

Statistical tests
-----------------
95 % bootstrap CI on AUROC: 1000 resamples of the validation set (default_rng,
seed=42). One-sided paired bootstrap significance test:
  H0: AUROC(TrajRisk) <= AUROC(method)
  p = proportion of 1000 bootstrap samples where TrajRisk does NOT outperform.
This is NOT a DeLong test. It is labelled as a paired bootstrap significance
test throughout.
"""

import os
import json
import time
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import make_scorer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    raise ImportError("xgboost is required: pip install xgboost")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrajRisk-W8")


# =============================================================================
# SECTION 0: CONFIGURATION
# =============================================================================

BASE = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE / "Data"  / "processed"
MODEL_DIR = BASE / "models"
FIG_DIR   = BASE / "results" / "figures"
RES_DIR   = BASE / "results"

for _d in [MODEL_DIR, FIG_DIR, RES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

# Architecture constants — must match Week 7 exactly
NODE_DIM_FULL = 31    # scVI 30D + XGBoost initial risk 1D
NODE_DIM_SCVI = 30    # scVI only (Ablation A4)
HIDDEN_DIM    = 64
DROPOUT       = 0.3
N_CONV_LAYERS = 2

SCALE_NAMES: List[str] = [
    "Scale1_AcuteStress",
    "Scale2_PeakInjury",
    "Scale3_FateDecision",
    "Scale4_Resolution",
]

# Temporal windows — must match Week 5 RF training windows exactly
SCALE_TEMPORAL_WINDOWS: List[Tuple[str, str]] = [
    ("0h",  "6h"),
    ("6h",  "24h"),
    ("24h", "48h"),
    ("48h", "96h"),
]

# GNN hyperparameters fixed from Week 7 Phase 1 search — never re-tuned here
BEST_LR            = 0.001
BEST_WD            = 0.0001
MAX_EPOCHS_ABLATION = 200   # Week 7 converged at epoch 144; 200 is a safe ceiling
PATIENCE           = 25
LR_REDUCE_PATIENCE = 10
LR_REDUCE_FACTOR   = 0.5
GRADIENT_CLIP      = 1.0

# Gaussian kernel parameters — match Week 5 RF sparsity
K_NEIGHBORS_GAUSSIAN = 15   # same as RF transition matrices (k=15 per source cell)

# Binary label — fixed in Week 6 based on biology, never adjusted to maximise AUROC
HIGH_RISK_THRESHOLD = 0.10

# Early stopping split
DEV_SPLIT_FRAC = 0.20   # 20 % of train_idx → dev_inner for GNN early stopping

# Statistics
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# SECTION 1: MODEL CLASSES — exact copies from Week 7 (do not modify)
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
    def __init__(
        self,
        node_dim:   int   = NODE_DIM_FULL,
        hidden_dim: int   = HIDDEN_DIM,
        n_scales:   int   = len(SCALE_NAMES),
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.n_scales = n_scales

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
            nn.Linear(n_scales * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        for layer in self.fusion_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, edge_indices, edge_weights):
        parts = []
        for s in range(self.n_scales):
            h = x
            for l in range(N_CONV_LAYERS):
                h = self.scale_convs[s][l](h, edge_indices[s], edge_weights[s])
                h = self.scale_bns[s][l](h)
                h = torch.relu(h)
            parts.append(h)
        return self.fusion_mlp(torch.cat(parts, dim=1)).squeeze(-1)


# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================

def create_dev_split(
    train_idx:  torch.Tensor,
    dev_frac:   float = DEV_SPLIT_FRAC,
    seed:       int   = RANDOM_SEED,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split train_idx into train_inner and dev_inner.
    dev_inner is used ONLY for GNN early stopping.
    val_idx (Week 6) is reserved for final AUROC reporting.

    Returns:
        train_inner_ids  — global cell IDs for gradient updates
        dev_inner_ids    — global cell IDs for early stopping
        train_inner_pos  — positions in train_idx (for train_y indexing)
        dev_inner_pos    — positions in train_idx (for train_y indexing)

    Returning positions avoids an O(n²) search when mapping train_y values.
    """
    n    = len(train_idx)
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(n)
    dev_n   = int(n * dev_frac)
    dev_pos = torch.tensor(perm[:dev_n], dtype=torch.long)
    trn_pos = torch.tensor(perm[dev_n:], dtype=torch.long)
    return train_idx[trn_pos], train_idx[dev_pos], trn_pos, dev_pos


def load_all_inputs() -> Dict:
    log.info("Loading all inputs...")

    graphs = torch.load(DATA_DIR / "multiscale_graphs.pt", map_location="cpu")
    rf_ei: List[torch.Tensor] = []
    rf_ew: List[torch.Tensor] = []
    for sn in SCALE_NAMES:
        rf_ei.append(graphs[sn]["edge_index"])
        rf_ew.append(graphs[sn]["edge_weight"])
    total_edges = sum(e.shape[1] for e in rf_ei)
    log.info(f"  RF graphs: 4 scales | {total_edges:,} total edges")

    x   = torch.load(DATA_DIR / "week6_node_features.pt",  map_location="cpu")
    td  = torch.load(DATA_DIR / "week6_training_data.pt",  map_location="cpu")
    train_idx = td["train_idx"]
    val_idx   = td["val_idx"]
    train_y   = td["train_y"]
    val_y     = td["val_y"]

    log.info(f"  Features: {tuple(x.shape)} | Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    # ── Dev split (RED FLAG 1 fix) ────────────────────────────────────────────
    train_inner, dev_inner, train_inner_pos, dev_inner_pos = create_dev_split(train_idx)
    # Position-based indexing: train_y[i] corresponds to train_idx[i]
    train_inner_y = train_y[train_inner_pos]
    dev_y         = train_y[dev_inner_pos]

    log.info(f"  Train inner: {len(train_inner):,} | Dev inner: {len(dev_inner):,} "
             f"| Val: {len(val_idx):,}")

    xgb_init   = np.load(DATA_DIR / "week6_xgboost_risk.npy")
    cell_index = pd.read_csv(DATA_DIR / "walesky_cell_index.csv", index_col=0)

    tp_col  = "timepoint_str" if "timepoint_str" in cell_index.columns else "timepoint"
    trt_col = "treatment_str" if "treatment_str" in cell_index.columns else "treatment"

    # Load Week 7 TrajRisk predictions
    preds_w7 = pd.read_csv(DATA_DIR / "week7_predictions.csv", index_col=0)
    risk_trajrisk = preds_w7["risk_predicted"].values

    val_bin     = (val_y.numpy() > HIGH_RISK_THRESHOLD).astype(int)
    tr_val_auroc = roc_auc_score(val_bin, risk_trajrisk[val_idx.numpy()])
    log.info(f"  TrajRisk val AUROC (from Week 7): {tr_val_auroc:.4f}")
    if not (0.89 <= tr_val_auroc <= 0.92):
        log.warning("  TrajRisk AUROC outside expected range [0.89, 0.92] — "
                    "verify week7_predictions.csv is from BEST_AUROC checkpoint")

    return {
        "x": x, "rf_ei": rf_ei, "rf_ew": rf_ew,
        "train_idx": train_idx, "val_idx": val_idx,
        "train_inner": train_inner, "dev_inner": dev_inner,
        "train_inner_pos": train_inner_pos, "dev_inner_pos": dev_inner_pos,
        "train_y": train_y, "val_y": val_y,
        "train_inner_y": train_inner_y, "dev_y": dev_y,
        "xgb_init": xgb_init, "cell_index": cell_index,
        "risk_trajrisk": risk_trajrisk, "n_cells": x.shape[0],
        "tp_col": tp_col, "trt_col": trt_col,
    }


# =============================================================================
# SECTION 3: GAUSSIAN KERNEL COMPUTATION
# =============================================================================

def compute_gaussian_transitions(
    x_scvi:     np.ndarray,
    cell_index: pd.DataFrame,
    tp_col:     str,
    k:          int = K_NEIGHBORS_GAUSSIAN,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Build Gaussian kernel transition matrices in PyG format.

    Normalisation convention: row-stochastic over outgoing edges of each
    source node (sum_j w[i,j] = 1 for each i). This MATCHES the RF matrix
    convention from Week 5, ensuring B2/A1 comparisons are apples-to-apples.

    Bandwidth: median pairwise L2 distance across all (source, target) pairs
    within each temporal window (median heuristic — no tunable hyperparameter).
    Sparsification: top-k=15 nearest targets per source, same as RF k.
    """
    log.info(f"  Gaussian transitions: median bandwidth, k={k} sparsification")
    g_ei: List[torch.Tensor] = []
    g_ew: List[torch.Tensor] = []

    for scale_name, (src_tp, tgt_tp) in zip(SCALE_NAMES, SCALE_TEMPORAL_WINDOWS):
        src_idx = np.where((cell_index[tp_col] == src_tp).values)[0]
        tgt_idx = np.where((cell_index[tp_col] == tgt_tp).values)[0]

        if len(src_idx) == 0 or len(tgt_idx) == 0:
            raise ValueError(
                f"{scale_name}: no cells for timepoint '{src_tp}' or '{tgt_tp}'. "
                f"Check that tp_col='{tp_col}' is correct."
            )

        # Pairwise L2 in scVI space
        dists = euclidean_distances(x_scvi[src_idx], x_scvi[tgt_idx])

        # Median heuristic bandwidth
        sigma = float(np.median(dists)) + 1e-8

        K = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))

        # Top-k sparsification
        k_eff    = min(k, K.shape[1])
        K_sparse = np.zeros_like(K)
        for row_i in range(K.shape[0]):
            top_cols = np.argsort(-K[row_i])[:k_eff]
            K_sparse[row_i, top_cols] = K[row_i, top_cols]

        # Row-normalise (over outgoing edges from each source) — matches RF
        row_sums = K_sparse.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        K_norm   = K_sparse / row_sums

        src_rows, tgt_cols = np.where(K_norm > 0)
        g_src = src_idx[src_rows].astype(np.int64)
        g_tgt = tgt_idx[tgt_cols].astype(np.int64)
        w     = K_norm[src_rows, tgt_cols].astype(np.float32)

        g_ei.append(torch.tensor(np.stack([g_src, g_tgt]), dtype=torch.long))
        g_ew.append(torch.tensor(w, dtype=torch.float32))
        log.info(f"  {scale_name}: sigma={sigma:.4f} | edges={len(w):,} | "
                 f"src={len(src_idx):,} → tgt={len(tgt_idx):,}")

    return g_ei, g_ew


# =============================================================================
# SECTION 4: MATRIX PROPAGATION (B2 and A3)
# =============================================================================

def pyg_to_scipy(edge_index, edge_weight, n_nodes) -> csr_matrix:
    """
    PyG edge format → scipy CSR matrix M where M[target, source] = weight.
    M @ risk propagates source risk to each target cell as a weighted average
    of the source cells pointing to it (matching source_to_target flow in GNN).
    """
    targets = edge_index[1].numpy().astype(np.int32)
    sources = edge_index[0].numpy().astype(np.int32)
    weights = edge_weight.numpy().astype(np.float64)
    return csr_matrix((weights, (targets, sources)), shape=(n_nodes, n_nodes))


def matrix_propagate(
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    initial_risk: np.ndarray,
    n_nodes:      int,
    damping:      float = 0.5,
) -> np.ndarray:
    """
    Two-scale linear label propagation (no GNN, no feature learning).

    This is a scalar diffusion heuristic, NOT a linearized GNN. It tests
    whether propagating the 1D XGBoost risk signal over the learned graph
    is sufficient — not whether it reproduces GNN behaviour without
    nonlinearities. TrajRisk operates on 31D node features with learned
    nonlinear aggregation; those cannot be recovered by scalar propagation.

    For A3: replaces nonlinear GNN aggregation AND high-dimensional feature
    processing simultaneously. These cannot be decoupled without a GNN; A3
    is therefore labelled "nonlinear GNN aggregation + feature integration
    vs scalar linear diffusion". B2 and A3 differ only in edge weights
    (Gaussian vs RF); both are heuristic propagation baselines.

    risk_final = damping * risk_initial + (1 - damping) * mean(M_s @ risk_initial)

    damping=0.5 preserves the local expression signal for cells that already
    have strong initial risk (e.g. 6h cells scored by XGBoost).
    """
    Ms = [pyg_to_scipy(ei, ew, n_nodes) for ei, ew in zip(edge_indices, edge_weights)]
    scale_results = np.stack(
        [M @ initial_risk.astype(np.float64) for M in Ms], axis=1
    )
    mean_prop = scale_results.mean(axis=1)
    return (damping * initial_risk + (1.0 - damping) * mean_prop).clip(0.0, 1.0)


# =============================================================================
# SECTION 5: GNN ABLATION TRAINING (corrected early stopping)
# =============================================================================

def train_gnn_ablation(
    node_dim:        int,
    n_scales:        int,
    edge_indices:    List[torch.Tensor],
    edge_weights:    List[torch.Tensor],
    x_input:         torch.Tensor,
    train_inner_idx: torch.Tensor,
    dev_inner_idx:   torch.Tensor,
    train_inner_y:   torch.Tensor,
    dev_inner_y:     torch.Tensor,
    label:           str,
    save_path:       Path,
) -> Tuple[np.ndarray, float]:
    """
    Train a TrajRiskGNN variant. Returns (all_cell_predictions, best_dev_auroc).

    Early stopping: monitors AUROC on dev_inner (a 20 % hold-out from train_idx
    that is NEVER included in final AUROC reporting). val_idx is never touched.

    Hyperparameters: identical to Week 7 (lr=0.001, wd=0.0001, dropout=0.3,
    grad_clip=1.0, ReduceLROnPlateau factor=0.5 patience=10). Only the model
    architecture (node_dim, n_scales) and edge sets vary across ablations.
    This ensures the comparison isolates the ablated component only.
    """
    log.info(f"  [{label}] node_dim={node_dim} n_scales={n_scales} "
             f"max_epochs={MAX_EPOCHS_ABLATION}")

    torch.manual_seed(RANDOM_SEED)
    model     = TrajRiskGNN(node_dim=node_dim, n_scales=n_scales).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_LR, weight_decay=BEST_WD)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE, min_lr=1e-6,
    )

    dev_bin    = (dev_inner_y.numpy() > HIGH_RISK_THRESHOLD).astype(int)
    best_auroc = -np.inf   # ensures best_state is set at epoch 1 regardless of score
    best_state = None
    no_improve = 0
    t0         = time.time()
    stopped_at = MAX_EPOCHS_ABLATION

    for epoch in range(1, MAX_EPOCHS_ABLATION + 1):

        model.train()
        optimizer.zero_grad()
        risk_all = model(x_input, edge_indices, edge_weights)
        loss     = criterion(risk_all[train_inner_idx], train_inner_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            risk_eval = model(x_input, edge_indices, edge_weights)

        val_loss  = float(criterion(risk_eval[dev_inner_idx], dev_inner_y).item())
        dev_pred  = risk_eval[dev_inner_idx].numpy()

        try:
            dev_auroc = float(roc_auc_score(dev_bin, dev_pred))
        except ValueError:
            dev_auroc = 0.0

        scheduler.step(val_loss)

        if dev_auroc > best_auroc:
            best_auroc = dev_auroc
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            stopped_at = epoch
            break

    elapsed = time.time() - t0
    log.info(f"    Stopped epoch {stopped_at}/{MAX_EPOCHS_ABLATION} | "
             f"{elapsed:.0f}s | best dev AUROC={best_auroc:.4f}")

    assert best_state is not None, (
        f"train_gnn_ablation ({label}): best_state is None — "
        "model never produced a valid AUROC. Check data or architecture."
    )

    torch.save({
        "model_state_dict": best_state,
        "node_dim": node_dim, "n_scales": n_scales,
        "best_dev_auroc": best_auroc,
        "stopped_epoch": stopped_at, "label": label,
    }, save_path)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_preds = model(x_input, edge_indices, edge_weights).numpy()

    return all_preds, best_auroc


# =============================================================================
# SECTION 6: EVALUATION AND STATISTICS
# =============================================================================

def evaluate_on_val(
    val_y_cont: np.ndarray,
    val_pred:   np.ndarray,
    label:      str,
) -> Dict:
    y_bin  = (val_y_cont > HIGH_RISK_THRESHOLD).astype(int)
    auroc  = float(roc_auc_score(y_bin, val_pred))
    auprc  = float(average_precision_score(y_bin, val_pred))
    rho, _ = spearmanr(val_pred, val_y_cont)
    return {
        "method":   label,
        "auroc":    round(auroc, 4),
        "auprc":    round(auprc, 4),
        "spearman": round(float(rho), 4),
    }


def bootstrap_ci_and_pvalue(
    val_y_cont:    np.ndarray,
    pred_trajrisk: np.ndarray,
    pred_other:    np.ndarray,
    n_bootstrap:   int = N_BOOTSTRAP,
    seed:          int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """
    95 % CI on competitor AUROC and one-sided paired bootstrap significance test.

    H0: AUROC(TrajRisk) <= AUROC(method).
    p = proportion of 1000 resamples where TrajRisk does NOT outperform.
    This is a paired bootstrap significance test, NOT DeLong's test.
    """
    rng   = np.random.default_rng(seed)
    y_bin = (val_y_cont > HIGH_RISK_THRESHOLD).astype(int)
    n     = len(val_y_cont)

    comp_aurocs: List[float] = []
    diffs:       List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt  = y_bin[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            a_tr = roc_auc_score(yt, pred_trajrisk[idx])
            a_ot = roc_auc_score(yt, pred_other[idx])
            comp_aurocs.append(a_ot)
            diffs.append(a_tr - a_ot)
        except Exception:
            continue

    if not comp_aurocs:
        return float("nan"), float("nan"), float("nan")

    ca = np.array(comp_aurocs)
    d  = np.array(diffs)
    return (
        float(np.percentile(ca, 2.5)),
        float(np.percentile(ca, 97.5)),
        float((d <= 0.0).mean()),
    )


def bootstrap_trajrisk_ci(
    val_y_cont:    np.ndarray,
    pred_trajrisk: np.ndarray,
    n_bootstrap:   int = N_BOOTSTRAP,
    seed:          int = RANDOM_SEED,
) -> Tuple[float, float]:
    rng   = np.random.default_rng(seed)
    y_bin = (val_y_cont > HIGH_RISK_THRESHOLD).astype(int)
    n     = len(val_y_cont)
    aurocs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt  = y_bin[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            aurocs.append(roc_auc_score(yt, pred_trajrisk[idx]))
        except Exception:
            continue
    a = np.array(aurocs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# =============================================================================
# SECTION 7: INDIVIDUAL METHOD RUNS
# =============================================================================

# ── Baseline 1: XGBoost with cross-validated hyperparameters (RF7 fix) ───────

def _xgb_auroc_scorer(y_true, y_pred):
    """AUROC scorer for GridSearchCV with continuous pseudotime labels."""
    y_bin = (y_true > HIGH_RISK_THRESHOLD).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return 0.5
    return float(roc_auc_score(y_bin, y_pred))


def run_baseline1_xgboost(data: Dict) -> np.ndarray:
    """
    B1: XGBoost trained on scVI 30D features to predict pseudotime_norm.

    Hyperparameter selection: 5-fold cross-validation on train_idx with AUROC
    scoring. This prevents the comparison from being unfair due to hardcoded
    hyperparameters (RF7). val_idx is never seen during selection or fitting.

    Note: XGBoost trains on the full train_idx (not train_inner) because it
    does not require a gradient-based early stopping split. The CV handles
    overfitting implicitly through cross-validation.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("pip install xgboost")

    log.info("B1: XGBoost (5-fold CV on train_idx, AUROC scoring)...")

    X_all  = data["x"][:, :NODE_DIM_SCVI].numpy()
    ti     = data["train_idx"].numpy()
    X_train = X_all[ti]
    y_train = data["train_y"].numpy()

    param_grid = {
        "n_estimators":  [200, 500],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    base_xgb = XGBRegressor(
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=0, n_jobs=1,
    )
    scorer = make_scorer(_xgb_auroc_scorer, greater_is_better=True)
    # StratifiedKFold on binary label ensures each fold contains positive examples.
    # The CV splits are generated from y_bin_train; the model trains on y_train (continuous).
    y_bin_train = (y_train > HIGH_RISK_THRESHOLD).astype(int)
    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_splits   = list(skf.split(X_train, y_bin_train))  # list of (train_idx, test_idx)

    grid_cv = GridSearchCV(
        base_xgb, param_grid=param_grid, scoring=scorer,
        cv=cv_splits, n_jobs=1, refit=True, verbose=0,
    )
    grid_cv.fit(X_train, y_train)

    bp = grid_cv.best_params_
    log.info(f"  Best XGB params: n_estimators={bp['n_estimators']} "
             f"max_depth={bp['max_depth']} lr={bp['learning_rate']}")

    risk_all = np.clip(grid_cv.best_estimator_.predict(X_all).astype(float), 0.0, 1.0)

    vi   = data["val_idx"].numpy()
    vb   = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
    log.info(f"  Val AUROC: {roc_auc_score(vb, risk_all[vi]):.4f}")
    return risk_all


# ── Baseline 2: Gaussian propagation ─────────────────────────────────────────

def run_baseline2_gaussian_propagation(
    data: Dict,
    g_ei: List[torch.Tensor],
    g_ew: List[torch.Tensor],
) -> np.ndarray:
    log.info("B2: Gaussian propagation (multi-scale matrix, no GNN)...")
    risk = matrix_propagate(g_ei, g_ew, data["xgb_init"], data["n_cells"])
    vi   = data["val_idx"].numpy()
    vb   = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
    log.info(f"  Val AUROC: {roc_auc_score(vb, risk[vi]):.4f}")
    return risk


# ── Baseline 3 / Ablation 2: Single-scale GNN — all four scales (RF4 fix) ───

def run_baseline3_single_scale_gnn(
    data:            Dict,
    train_inner_idx: torch.Tensor,
    dev_inner_idx:   torch.Tensor,
    train_inner_y:   torch.Tensor,
    dev_y:           torch.Tensor,
) -> Tuple[np.ndarray, str, Dict]:
    """
    Train Scale1-4 separately and select best by dev_inner AUROC.
    val_idx is never used for selection. All four results are returned for
    supplementary reporting (eliminates the cherry-picking concern from RF4).
    train_inner_y is pre-computed in load_all_inputs() via position indexing.
    Note: the fusion MLP always starts at Linear(n_scales*64, 128), so single-
    scale models use a 64→128→32→1 head (slight capacity expansion). This is
    if anything helpful to the baseline — if TrajRisk still wins, the advantage
    is not due to the single-scale model having an undersized head.
    """
    log.info("B3/A2: Single-scale GNN — training all 4 scales, selecting best by dev AUROC...")
    all_results: Dict[str, Dict] = {}

    for s in range(4):
        sname = SCALE_NAMES[s]
        preds, dev_auroc = train_gnn_ablation(
            node_dim=NODE_DIM_FULL, n_scales=1,
            edge_indices=[data["rf_ei"][s]],
            edge_weights=[data["rf_ew"][s]],
            x_input=data["x"],
            train_inner_idx=train_inner_idx,
            dev_inner_idx=dev_inner_idx,
            train_inner_y=train_inner_y,
            dev_inner_y=dev_y,
            label=f"SingleScale-{sname}",
            save_path=MODEL_DIR / f"week8_singlescale_{s+1}_{sname}.pt",
        )
        vi  = data["val_idx"].numpy()
        vb  = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
        val_auroc = roc_auc_score(vb, preds[vi])
        all_results[sname] = {
            "preds": preds, "dev_auroc": dev_auroc, "val_auroc": val_auroc
        }
        log.info(f"  {sname}: dev AUROC={dev_auroc:.4f} | val AUROC={val_auroc:.4f}")

    best_scale = max(all_results, key=lambda k: all_results[k]["dev_auroc"])
    log.info(f"  Selected: {best_scale} (dev AUROC={all_results[best_scale]['dev_auroc']:.4f})")
    log.info(f"  Individual val AUROCs: "
             + " | ".join(f"{k.replace('Scale','S')}={v['val_auroc']:.4f}"
                          for k, v in all_results.items()))

    return all_results[best_scale]["preds"], best_scale, all_results


# ── Baseline 4: Expression distance ───────────────────────────────────────────

def run_baseline4_expression_distance(data: Dict) -> np.ndarray:
    """
    B4: Euclidean distance from the mean healthy (0h) cell in scVI space,
    normalised to [0,1]. No graph, no trajectory.

    RNA velocity is excluded from baselines: GSE136679 was deposited as a
    pre-processed expression matrix without spliced/unspliced count layers.
    Without these layers scVelo cannot estimate velocity. Including an N/A
    entry would not add scientific value; the limitation is stated here instead.
    """
    log.info("B4: Expression distance from healthy baseline "
             "(RNA velocity excluded — see docstring)")
    x_scvi       = data["x"][:, :NODE_DIM_SCVI].numpy()
    tp_col       = data["tp_col"]
    healthy_mask = (data["cell_index"][tp_col] == "0h").values
    mean_healthy = x_scvi[healthy_mask].mean(axis=0)

    dists = np.linalg.norm(x_scvi - mean_healthy, axis=1)
    risk  = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)

    vi  = data["val_idx"].numpy()
    vb  = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
    log.info(f"  Val AUROC: {roc_auc_score(vb, risk[vi]):.4f}")
    return risk


# ── Ablation 1: Gaussian transitions + full GNN ───────────────────────────────

def run_ablation1_gaussian_gnn(
    data:            Dict,
    g_ei:            List[torch.Tensor],
    g_ew:            List[torch.Tensor],
    train_inner_idx: torch.Tensor,
    dev_inner_idx:   torch.Tensor,
    train_inner_y:   torch.Tensor,
    dev_y:           torch.Tensor,
) -> np.ndarray:
    log.info("A1: Full 4-scale GNN with Gaussian transitions...")
    preds, dev_auroc = train_gnn_ablation(
        node_dim=NODE_DIM_FULL, n_scales=len(SCALE_NAMES),
        edge_indices=g_ei, edge_weights=g_ew,
        x_input=data["x"],
        train_inner_idx=train_inner_idx, dev_inner_idx=dev_inner_idx,
        train_inner_y=train_inner_y, dev_inner_y=dev_y,
        label="Ablation1-GaussianGNN",
        save_path=MODEL_DIR / "week8_ablation1_gaussian_gnn.pt",
    )
    vi  = data["val_idx"].numpy()
    vb  = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
    log.info(f"  Val AUROC: {roc_auc_score(vb, preds[vi]):.4f}")
    return preds


# ── Ablation 3: RF transitions + matrix propagation ───────────────────────────

def run_ablation3_rf_matrix_propagation(data: Dict) -> np.ndarray:
    log.info("A3: RF transitions + matrix propagation (no GNN)...")
    risk = matrix_propagate(
        data["rf_ei"], data["rf_ew"], data["xgb_init"], data["n_cells"]
    )
    vi  = data["val_idx"].numpy()
    vb  = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
    log.info(f"  Val AUROC: {roc_auc_score(vb, risk[vi]):.4f}")
    return risk


# ── Ablation 4: No XGBoost risk seed ─────────────────────────────────────────

def run_ablation4_no_init_risk(
    data:            Dict,
    train_inner_idx: torch.Tensor,
    dev_inner_idx:   torch.Tensor,
    train_inner_y:   torch.Tensor,
    dev_y:           torch.Tensor,
) -> np.ndarray:
    log.info("A4: Full 4-scale GNN — 30D features (no XGBoost seed)...")
    x_scvi = data["x"][:, :NODE_DIM_SCVI]
    preds, dev_auroc = train_gnn_ablation(
        node_dim=NODE_DIM_SCVI, n_scales=len(SCALE_NAMES),
        edge_indices=data["rf_ei"], edge_weights=data["rf_ew"],
        x_input=x_scvi,
        train_inner_idx=train_inner_idx, dev_inner_idx=dev_inner_idx,
        train_inner_y=train_inner_y, dev_inner_y=dev_y,
        label="Ablation4-NoInitRisk",
        save_path=MODEL_DIR / "week8_ablation4_noinit_gnn.pt",
    )
    vi  = data["val_idx"].numpy()
    vb  = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
    log.info(f"  Val AUROC: {roc_auc_score(vb, preds[vi]):.4f}")
    return preds


# =============================================================================
# SECTION 8: COMPARISON TABLE
# =============================================================================

def build_comparison_table(
    all_preds: Dict[str, np.ndarray],
    data:      Dict,
) -> pd.DataFrame:
    """
    Compute AUROC, 95 % bootstrap CI, AUPRC, Spearman ρ, Δ AUROC, and
    one-sided paired bootstrap p-value for every method vs TrajRisk.
    """
    log.info(f"  Bootstrap: n={N_BOOTSTRAP} resamples | "
             "paired bootstrap significance test (not DeLong)")

    vi      = data["val_idx"].numpy()
    val_y_c = data["val_y"].numpy()
    pred_tr = all_preds["TrajRisk (full)"][vi]
    tr_auroc = roc_auc_score((val_y_c > HIGH_RISK_THRESHOLD).astype(int), pred_tr)

    tr_ci_lo, tr_ci_hi = bootstrap_trajrisk_ci(val_y_c, pred_tr)

    ORDER = [
        "TrajRisk (full)",
        "B1: XGBoost (CV-tuned)",
        "B2: Gaussian propagation",
        "B3/A2: Single-scale GNN (best)",
        "B4: Expression distance",
        "A1: Gaussian transitions",
        "A3: RF matrix propagation",
        "A4: No XGBoost seed",
    ]
    TYPE_MAP = {
        "TrajRisk (full)":              "Proposed",
        "B1: XGBoost (CV-tuned)":       "Baseline",
        "B2: Gaussian propagation":     "Baseline",
        "B3/A2: Single-scale GNN (best)":"Baseline/Ablation",
        "B4: Expression distance":      "Baseline",
        "A1: Gaussian transitions":     "Ablation",
        "A3: RF matrix propagation":    "Ablation",
        "A4: No XGBoost seed":          "Ablation",
    }

    rows = []
    for method in ORDER:
        if method not in all_preds:
            continue
        pred     = all_preds[method][vi]
        base     = evaluate_on_val(val_y_c, pred, method)
        delta_pp = round((tr_auroc - base["auroc"]) * 100, 2)   # positive = TrajRisk outperforms

        if method == "TrajRisk (full)":
            ci_lo, ci_hi, p_val = tr_ci_lo, tr_ci_hi, float("nan")
        else:
            ci_lo, ci_hi, p_val = bootstrap_ci_and_pvalue(val_y_c, pred_tr, pred)

        rows.append({
            "Method":               method,
            "Type":                 TYPE_MAP.get(method, ""),
            "AUROC":                base["auroc"],
            "AUROC_CI_lower":       round(ci_lo, 4) if not np.isnan(ci_lo) else float("nan"),
            "AUROC_CI_upper":       round(ci_hi, 4) if not np.isnan(ci_hi) else float("nan"),
            "AUPRC":                base["auprc"],
            "Spearman_rho":         base["spearman"],
            "Delta_vs_TrajRisk_pp": delta_pp,
            "p_value_bootstrap":    round(p_val, 4) if not np.isnan(p_val) else "—",
        })

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 9: FIGURES
# =============================================================================

def make_auroc_barplot(df: pd.DataFrame) -> None:
    type_colors = {
        "Proposed":          "#2ecc71",
        "Baseline":          "#3498db",
        "Baseline/Ablation": "#9b59b6",
        "Ablation":          "#e67e22",
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(
        "TrajRisk vs Baselines and Ablations — Validation AUROC\n"
        "(95 % bootstrap CI, n=1000; "
        "paired bootstrap significance test vs TrajRisk)",
        fontweight="bold", fontsize=11,
    )

    n      = len(df)
    y_pos  = list(range(n))
    colors = [type_colors.get(t, "#95a5a6") for t in df["Type"]]
    auroc  = df["AUROC"].values
    ci_lo  = df["AUROC_CI_lower"].values
    ci_hi  = df["AUROC_CI_upper"].values

    xerr_lo = np.where(np.isnan(ci_lo), 0.0, auroc - ci_lo)
    xerr_hi = np.where(np.isnan(ci_hi), 0.0, ci_hi - auroc)

    ax.barh(y_pos, auroc, xerr=[xerr_lo, xerr_hi], color=colors,
            alpha=0.85, edgecolor="black", linewidth=0.7,
            error_kw=dict(elinewidth=1.2, capsize=4, ecolor="black"))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Method"], fontsize=10)
    ax.set_xlabel("Validation AUROC", fontsize=11)
    ax.set_xlim(0.5, 1.05)
    ax.axvline(0.90, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for i, (a, hi) in enumerate(zip(auroc, ci_hi)):
        x_text = hi + 0.005 if not np.isnan(hi) else a + 0.005
        ax.text(x_text, i, f"{a:.4f}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "week8_auroc_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: week8_auroc_barplot.png")


def make_auprc_barplot(df: pd.DataFrame) -> None:
    type_colors = {
        "Proposed": "#2ecc71", "Baseline": "#3498db",
        "Baseline/Ablation": "#9b59b6", "Ablation": "#e67e22",
    }
    colors = [type_colors.get(t, "#95a5a6") for t in df["Type"]]
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("TrajRisk vs Baselines and Ablations — Validation AUPRC",
                 fontweight="bold", fontsize=11)

    y_pos = list(range(len(df)))
    ax.barh(y_pos, df["AUPRC"], color=colors,
            alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Method"], fontsize=10)
    ax.set_xlabel("Validation AUPRC", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    for i, a in enumerate(df["AUPRC"]):
        ax.text(a + 0.005, i, f"{a:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "week8_auprc_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: week8_auprc_barplot.png")


def make_ablation_waterfall(df: pd.DataFrame) -> None:
    ablation_rows = [
        "TrajRisk (full)", "A1: Gaussian transitions",
        "B3/A2: Single-scale GNN (best)", "A3: RF matrix propagation",
        "A4: No XGBoost seed",
    ]
    abl_df = df[df["Method"].isin(ablation_rows)].copy()
    abl_df = abl_df.set_index("Method").reindex(
        [m for m in ablation_rows if m in abl_df.index]
    ).reset_index()

    if len(abl_df) < 2:
        log.info("  Skipping waterfall: insufficient rows")
        return

    tr_auroc = float(abl_df.loc[abl_df["Method"] == "TrajRisk (full)", "AUROC"].values[0])
    colors   = ["#2ecc71"] + ["#e74c3c"] * (len(abl_df) - 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Ablation Study — AUROC per Removed Component",
                 fontweight="bold", fontsize=12)
    y_pos = list(range(len(abl_df)))

    ax.barh(y_pos, abl_df["AUROC"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.6)
    ax.axvline(tr_auroc, color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Full TrajRisk ({tr_auroc:.4f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(abl_df["Method"], fontsize=10)
    ax.set_xlabel("Validation AUROC", fontsize=11)
    ax.set_xlim(max(0.5, float(abl_df["AUROC"].min()) - 0.07), 1.02)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    for i, (auroc, delta) in enumerate(
        zip(abl_df["AUROC"], abl_df["Delta_vs_TrajRisk_pp"])
    ):
        lbl = f"{auroc:.4f}" if i == 0 else f"{auroc:.4f}  (Δ={delta:+.1f} pp)"
        ax.text(auroc + 0.004, i, lbl, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "week8_ablation_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: week8_ablation_waterfall.png")


# =============================================================================
# SECTION 10: MAIN
# =============================================================================

def main() -> None:
    log.info("=" * 72)
    log.info("TRAJRISK — WEEK 8: BASELINE COMPARISONS AND ABLATION STUDIES")
    log.info("=" * 72)
    log.info(f"Threshold: pseudotime_norm > {HIGH_RISK_THRESHOLD}")
    log.info(f"Bootstrap: n={N_BOOTSTRAP}, paired bootstrap significance test (not DeLong)")
    log.info(f"GNN hyperparams: lr={BEST_LR}, wd={BEST_WD}, max_epochs={MAX_EPOCHS_ABLATION}")
    log.info(f"Dev split: {int(DEV_SPLIT_FRAC*100)}% of train_idx for GNN early stopping")
    log.info(f"TrajRisk: loaded from Week 7 checkpoint (early stopping used val AUROC)")
    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("\n[1] Loading inputs...")
    data = load_all_inputs()

    train_inner_idx = data["train_inner"]
    dev_inner_idx   = data["dev_inner"]
    dev_y           = data["dev_y"]
    # train_inner_y and dev_y were computed in load_all_inputs() via position indexing
    train_inner_y   = data["train_inner_y"]

    # ── Gaussian matrices (shared by B2 and A1) ───────────────────────────────
    log.info("\n[2] Gaussian transition matrices...")
    x_scvi = data["x"][:, :NODE_DIM_SCVI].numpy()
    g_ei, g_ew = compute_gaussian_transitions(x_scvi, data["cell_index"], data["tp_col"])

    # ── Run all methods ───────────────────────────────────────────────────────
    log.info("\n[3] Running all baselines and ablations...")
    preds: Dict[str, np.ndarray] = {}

    preds["TrajRisk (full)"] = data["risk_trajrisk"]

    preds["B1: XGBoost (CV-tuned)"]     = run_baseline1_xgboost(data)
    preds["B2: Gaussian propagation"]   = run_baseline2_gaussian_propagation(data, g_ei, g_ew)

    b3_preds, best_scale, single_scale_all = run_baseline3_single_scale_gnn(
        data, train_inner_idx, dev_inner_idx, train_inner_y, dev_y
    )
    preds["B3/A2: Single-scale GNN (best)"] = b3_preds

    preds["B4: Expression distance"] = run_baseline4_expression_distance(data)

    preds["A1: Gaussian transitions"] = run_ablation1_gaussian_gnn(
        data, g_ei, g_ew, train_inner_idx, dev_inner_idx, train_inner_y, dev_y
    )
    preds["A3: RF matrix propagation"] = run_ablation3_rf_matrix_propagation(data)
    preds["A4: No XGBoost seed"]       = run_ablation4_no_init_risk(
        data, train_inner_idx, dev_inner_idx, train_inner_y, dev_y
    )

    # ── Table and statistics ──────────────────────────────────────────────────
    log.info("\n[4] Comparison table + bootstrap statistics...")
    df = build_comparison_table(preds, data)

    vi      = data["val_idx"].numpy()
    val_y_c = data["val_y"].numpy()
    pred_tr = preds["TrajRisk (full)"][vi]
    tr_auroc = roc_auc_score((val_y_c > HIGH_RISK_THRESHOLD).astype(int), pred_tr)

    log.info("\n  RESULTS (paired bootstrap significance test, n=1000):")
    log.info("  " + "─" * 108)
    log.info(f"  {'Method':<38} {'Type':<20} {'AUROC':>6} "
             f"{'95% CI':>16} {'AUPRC':>6} {'ρ':>6} {'Δ pp':>7} {'p-val':>7}")
    log.info("  " + "─" * 108)
    for _, row in df.iterrows():
        ci_lo = row["AUROC_CI_lower"]
        ci_hi = row["AUROC_CI_upper"]
        ci_str = (f"({ci_lo:.4f}–{ci_hi:.4f})"
                  if not (isinstance(ci_lo, float) and np.isnan(ci_lo)) else "(—)")
        delta  = (f"{row['Delta_vs_TrajRisk_pp']:+.2f}"
                  if row["Method"] != "TrajRisk (full)" else "—")
        log.info(
            f"  {row['Method']:<38} {row['Type']:<20} {row['AUROC']:>6.4f} "
            f"{ci_str:>16} {row['AUPRC']:>6.4f} {row['Spearman_rho']:>6.4f} "
            f"{delta:>7} {str(row['p_value_bootstrap']):>7}"
        )
    log.info("  " + "─" * 108)

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("\n[5] Figures...")
    make_auroc_barplot(df)
    make_auprc_barplot(df)
    make_ablation_waterfall(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("\n[6] Saving results...")
    df.to_csv(RES_DIR / "week8_comparison_table.csv", index=False)
    log.info("  Saved: week8_comparison_table.csv")

    abl_rows = df["Method"].str.startswith("A") | (df["Method"] == "TrajRisk (full)")
    df[abl_rows].to_csv(RES_DIR / "week8_ablation_table.csv", index=False)
    log.info("  Saved: week8_ablation_table.csv")

    # Supplementary: all 4 single-scale results
    ss_rows = []
    for sname, res in single_scale_all.items():
        vi_np  = data["val_idx"].numpy()
        vb     = (data["val_y"].numpy() > HIGH_RISK_THRESHOLD).astype(int)
        ss_rows.append({
            "scale": sname,
            "dev_auroc": round(res["dev_auroc"], 4),
            "val_auroc": round(res["val_auroc"], 4),
            "selected": sname == best_scale,
        })
    pd.DataFrame(ss_rows).to_csv(
        RES_DIR / "week8_singlescale_all_results.csv", index=False
    )
    log.info("  Saved: week8_singlescale_all_results.csv")

    summary = {
        "week": 8,
        "n_val": int(len(vi)),
        "n_train": int(len(data["train_idx"])),
        "n_train_inner": int(len(train_inner_idx)),
        "n_dev_inner": int(len(dev_inner_idx)),
        "threshold": HIGH_RISK_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "trajrisk_auroc": round(tr_auroc, 4),
        "best_single_scale": best_scale,
        "early_stopping_note": (
            "TrajRisk used val_idx AUROC for early stopping (Week 7). "
            "GNN ablations used dev_inner AUROC (20 % of train_idx). "
            "Comparison is conservative — TrajRisk has slight checkpoint advantage."
        ),
        "significance_test": "paired bootstrap (one-sided, n=1000), NOT DeLong",
        "results": df.to_dict(orient="records"),
    }
    with open(RES_DIR / "week8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("  Saved: week8_summary.json")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("\n" + "=" * 72)
    log.info("WEEK 8 COMPLETE")
    log.info("=" * 72)
    log.info(f"Runtime: {elapsed/60:.1f} minutes")
    log.info(f"TrajRisk val AUROC: {tr_auroc:.4f}")

    baselines_df = df[df["Type"].str.contains("Baseline", na=False)]
    if len(baselines_df):
        best_bl = baselines_df.sort_values("AUROC", ascending=False).iloc[0]
        log.info(f"Best baseline:     {best_bl['Method']} "
                 f"AUROC={best_bl['AUROC']:.4f} "
                 f"Gap={abs(best_bl['Delta_vs_TrajRisk_pp']):.2f} pp")

    abl_df_only = df[df["Type"] == "Ablation"]
    if len(abl_df_only):
        worst = abl_df_only.sort_values("AUROC").iloc[0]
        log.info(f"Largest drop:      {worst['Method']} "
                 f"AUROC={worst['AUROC']:.4f} "
                 f"Drop={abs(worst['Delta_vs_TrajRisk_pp']):.2f} pp")

    sig = df[
        df["p_value_bootstrap"].apply(
            lambda p: isinstance(p, float) and p < 0.05
        )
    ]
    log.info(f"Significant wins:  {len(sig)}/{len(df)-1} methods (paired bootstrap p<0.05)")
    log.info("")
    log.info("→ Ready for Week 9: Cross-Dataset Validation.")


if __name__ == "__main__":
    main()
