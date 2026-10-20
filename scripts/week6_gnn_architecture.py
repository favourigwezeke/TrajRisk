#!/usr/bin/env python3
"""
TrajRisk: Week 6 — Graph Neural Network Architecture
=====================================================
Project: TrajRisk — Trajectory-Informed Risk Propagation in APAP Hepatotoxicity
Target Journal: Cell Reports Methods

Overview
--------
This script implements and validates the TrajRiskGNN architecture: a multi-scale
graph neural network that propagates risk scores through the APAP injury trajectory.

The architecture rests on three Week-5 outputs:
  - multiscale_graphs.pt      : 4 directed transition graphs (PyG edge format)
  - walesky_cell_index.csv    : cell metadata and global index mapping
  - adata_integrated_with_trajectory.h5ad : scVI latent + pseudotime_norm

Architecture Summary
--------------------
Node features (31D per cell):
    z_i  (30D): scVI latent representation (from Week 3)
    r_i   (1D): XGBoost initial risk trained on 0h Control vs 6h APAP cells

Graph: 4 directed temporal scales (0h→6h, 6h→24h, 24h→48h, 48h→96h)
    Each scale provides a directed edge set with learned RF transition probabilities
    as edge weights (row-stochastic per source cell, from Week 5 Random Forests).

GNN per scale:
    TrajRiskConv Layer 1: 31D → 64D  (aggregate source features weighted by T_s)
    TrajRiskConv Layer 2: 64D → 64D  (refine aggregated representation)
    Residual self-connection in both layers (prevents dead nodes with no in-edges)

Fusion MLP:
    Concat all 4 scale outputs → 256D
    Linear(256 → 128) → ReLU → Dropout(0.3)
    Linear(128 → 32)  → ReLU
    Linear(32 → 1)    → Sigmoid
    Output: risk_score ∈ [0, 1]

Training objective:
    Continuous regression on pseudotime_norm (0 → 1), not binary APAP/Control.
    MSE loss forces the GNN to distinguish the gradient between recovered cells
    (pseudotime_norm ≈ 0.02) and compensation-mode cells (pseudotime_norm ≈ 0.89),
    which is the biologically meaningful risk signal identified in Week 5.

Week 6 scope: architecture implementation + quick 10-epoch diagnostic run.
Full training (200 epochs with early stopping) is performed in Week 7.

Outputs
-------
Data/processed/week6_node_features.pt      — node feature tensor (6582, 31)
Data/processed/week6_training_data.pt      — train/val indices and labels
Data/processed/week6_xgboost_risk.npy      — XGBoost initial risk scores
models/week6_trajrisk_test.pt              — model checkpoint (10-epoch test)
models/week6_xgboost_model.pkl             — trained XGBoost scorer
results/week6_diagnostics.json             — architecture and training metrics
results/figures/week6_*.png                — diagnostic figures

Usage
-----
    conda activate trajrisk
    python scripts/week6_gnn_architecture.py
"""

import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import scanpy as sc
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrajRisk-W6")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0: CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE / "Data"  / "processed"
MODEL_DIR = BASE / "models"
FIG_DIR   = BASE / "results" / "figures"
RES_DIR   = BASE / "results"

for _d in [DATA_DIR, MODEL_DIR, FIG_DIR, RES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Node feature dimensions
SCVI_DIM     = 30   # scVI latent space (fixed, from Week 3)
INIT_RISK_DIM = 1   # XGBoost initial risk score
NODE_DIM     = SCVI_DIM + INIT_RISK_DIM  # 31D total

# GNN hyperparameters (locked for Week 6 architecture; tuned in Week 7)
HIDDEN_DIM   = 64    # output dimension of each TrajRiskConv layer
N_CONV_LAYERS = 2    # conv layers per scale
N_SCALES     = 4     # temporal scales from Week 5
FUSION_DIM   = 256   # N_SCALES * HIDDEN_DIM (derived, do not change independently)
DROPOUT      = 0.3

# XGBoost configuration — trained only on 0h and 6h cells (maximum biological contrast)
XGB_PARAMS: Dict[str, Any] = {
    "n_estimators":    300,
    "max_depth":        6,
    "learning_rate":  0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "random_state":    42,
    "n_jobs":          -1,
    "eval_metric":   "auc",
}

# Training (Week 6 = quick diagnostic; Week 7 = full)
N_EPOCHS_TEST = 10        # quick test to verify pipeline
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
VAL_FRACTION  = 0.20      # 20% of each timepoint → validation
RANDOM_SEED   = 42

# Scale names matching Week 5 output keys
SCALE_NAMES: List[str] = [
    "Scale1_AcuteStress",
    "Scale2_PeakInjury",
    "Scale3_FateDecision",
    "Scale4_Resolution",
]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD WEEK 5 OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def load_week5_outputs() -> Tuple[Dict, pd.DataFrame, sc.AnnData]:
    """
    Load all outputs produced by Week 5 (transition matrices, cell index,
    updated AnnData with pseudotime_norm).

    Returns
    -------
    graphs     : dict — PyG graphs keyed by scale name
    cell_index : DataFrame — global index mapping with pseudotime_norm
    walesky    : AnnData — 6,582 Walesky cells with scVI latent
    """
    log.info("Loading Week 5 outputs...")

    # PyTorch Geometric multi-scale graphs
    pyg_path = DATA_DIR / "multiscale_graphs.pt"
    if not pyg_path.exists():
        raise FileNotFoundError(f"Week 5 output not found: {pyg_path}. Run week5 first.")
    graphs = torch.load(pyg_path, map_location="cpu")
    log.info(f"  Loaded {len(graphs)} scale graphs:")
    for sn, g in graphs.items():
        log.info(f"    {sn}: {g['n_edges']:,} edges | {g['n_nodes']} nodes")

    # Cell index
    cell_index = pd.read_csv(DATA_DIR / "walesky_cell_index.csv", index_col=0)
    log.info(f"  Cell index: {len(cell_index):,} cells")

    if "pseudotime_norm" not in cell_index.columns:
        raise ValueError(
            "pseudotime_norm not in cell_index. "
            "Ensure week5_transition_matrices.py completed successfully."
        )

    # Walesky AnnData
    adata = sc.read_h5ad(DATA_DIR / "adata_integrated_with_trajectory.h5ad")
    walesky = adata[adata.obs["dataset"] == "Walesky"].copy()
    log.info(f"  Walesky AnnData: {walesky.n_obs:,} cells, scVI shape {walesky.obsm['X_scvi'].shape}")

    # Sync pseudotime_norm from cell_index into walesky.obs
    walesky.obs["pseudotime_norm"] = cell_index.loc[walesky.obs.index, "pseudotime_norm"].values

    return graphs, cell_index, walesky


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: XGBOOST INITIAL RISK SCORER
# ═════════════════════════════════════════════════════════════════════════════

def train_xgboost_risk(
    walesky_scvi: np.ndarray,
    cell_index: pd.DataFrame,
) -> Tuple[XGBClassifier, np.ndarray]:
    """
    Train an XGBoost classifier to distinguish 0h control cells from
    6h APAP cells, then apply it to all 6,582 Walesky cells to generate
    an initial, graph-agnostic risk score for each cell.

    Design rationale
    ----------------
    We train exclusively on 0h (healthy baseline) vs 6h (peak acute stress)
    because this pair represents the maximum biological contrast in the dataset.
    0h cells are the trajectory root; 6h cells are the furthest departure from
    baseline due to functional compensation. An XGBoost trained on these two
    extremes produces a scalar 'initial risk' for each cell that reflects how
    transcriptionally APAP-stress-like the cell is, without any graph context.

    The GNN then takes this initial risk as a node feature and propagates and
    refines it through the trajectory graph, incorporating neighbourhood
    information from connected cells at adjacent timepoints.

    Crucially, 24h, 48h, and 96h cells are NOT included in XGBoost training.
    They are scored AFTER training, so their initial risk reflects their
    similarity to the 0h/6h endpoints without label contamination.

    Class imbalance: 0h = 3,401 cells, 6h = 419 cells (≈8:1 ratio).
    We use scale_pos_weight = n_neg / n_pos to correct for this.

    Parameters
    ----------
    walesky_scvi : (6582, 30) scVI latent for all Walesky cells
    cell_index   : DataFrame with timepoint_str, global_idx

    Returns
    -------
    xgb   : trained XGBClassifier
    risk  : (6582,) float array, P(being 6h-APAP-like) for every cell
    """
    # Select cells for training: only 0h Control and 6h APAP
    mask_0h = cell_index["timepoint_str"] == "0h"
    mask_6h = cell_index["timepoint_str"] == "6h"

    train_mask = mask_0h | mask_6h
    train_idx  = cell_index.loc[train_mask, "global_idx"].values

    X_train = walesky_scvi[train_idx]
    y_train = (cell_index.loc[train_mask, "timepoint_str"] == "6h").astype(int).values

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    log.info(f"  XGBoost training: {n_pos} (6h APAP) vs {n_neg} (0h Control)")
    log.info(f"  Class imbalance ratio: {n_neg/n_pos:.1f}:1 → scale_pos_weight={n_neg/n_pos:.2f}")

    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=n_neg / n_pos,
    )

    # Hold out 20% of training cells for XGBoost internal validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_SEED
    )

    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_auroc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:, 1])
    log.info(f"  XGBoost validation AUROC (0h vs 6h): {val_auroc:.4f}")

    if val_auroc < 0.80:
        log.warning("  ⚠ XGBoost AUROC < 0.80. scVI may not cleanly separate 0h from 6h.")

    # Score ALL 6,582 Walesky cells
    risk_scores = xgb.predict_proba(walesky_scvi)[:, 1].astype(np.float32)

    # Log expected patterns — biological sanity check
    for tp in ["0h", "6h", "24h", "48h", "96h"]:
        mask = cell_index["timepoint_str"] == tp
        tp_idx = cell_index.loc[mask, "global_idx"].values
        tp_risk = risk_scores[tp_idx]
        log.info(f"  Risk score at {tp:>4s}: mean={tp_risk.mean():.3f} | "
                 f"median={np.median(tp_risk):.3f} | max={tp_risk.max():.3f}")

    return xgb, risk_scores


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: NODE FEATURE ASSEMBLY
# ═════════════════════════════════════════════════════════════════════════════

def assemble_node_features(
    walesky_scvi: np.ndarray,
    xgb_risk: np.ndarray,
) -> torch.Tensor:
    """
    Assemble the 31-dimensional node feature matrix.

    Features (per cell, ordered):
        Dims 0-29  : scVI latent representation (from Week 3 scVI model)
        Dim  30    : XGBoost initial risk score (trained above)

    Note: pseudotime_norm is intentionally EXCLUDED from node features.
    It is the training label. Including it as input would trivially inflate
    performance and make the GNN's task meaningless.

    Note: treatment indicator (APAP/Control) is intentionally EXCLUDED.
    It is already implicitly encoded in the XGBoost risk score: control
    cells score low (the model was trained to assign low risk to 0h cells)
    and APAP cells score high or intermediate depending on their state.
    Adding treatment as an explicit feature would create redundancy and
    risk teaching the GNN to rely on it rather than graph structure.

    Parameters
    ----------
    walesky_scvi : (6582, 30) scVI latent
    xgb_risk     : (6582,)   initial risk from XGBoost

    Returns
    -------
    x : FloatTensor (6582, 31) on CPU
    """
    x = np.concatenate([
        walesky_scvi,
        xgb_risk.reshape(-1, 1),
    ], axis=1).astype(np.float32)

    assert x.shape == (len(xgb_risk), NODE_DIM), \
        f"Expected ({len(xgb_risk)}, {NODE_DIM}), got {x.shape}"

    log.info(f"  Node features assembled: {x.shape[0]:,} cells × {x.shape[1]} dims")
    log.info(f"  Feature range: [{x.min():.3f}, {x.max():.3f}]")

    return torch.tensor(x, dtype=torch.float32)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING / VALIDATION SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def make_train_val_split(
    cell_index: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/validation splits stratified by timepoint.

    Stratification ensures all five timepoints (0h, 6h, 24h, 48h, 96h) are
    represented in both splits at the specified fraction. This prevents the
    model from training on a subset of timepoints and validating on another,
    which would conflate temporal generalisation with hyperparameter quality.

    Returns
    -------
    train_idx : LongTensor — global indices of training cells
    val_idx   : LongTensor — global indices of validation cells
    train_y   : FloatTensor — pseudotime_norm labels for training cells
    val_y     : FloatTensor — pseudotime_norm labels for validation cells
    """
    all_idx  = cell_index["global_idx"].values
    all_y    = cell_index["pseudotime_norm"].values.astype(np.float32)
    all_tp   = cell_index["timepoint_str"].values

    train_indices, val_indices = [], []

    for tp in ["0h", "6h", "24h", "48h", "96h"]:
        tp_mask = all_tp == tp
        tp_idx  = all_idx[tp_mask]
        tp_y    = all_y[tp_mask]

        if len(tp_idx) < 5:
            # Tiny group: add all to training
            train_indices.append(tp_idx)
            continue

        tr, va = train_test_split(
            tp_idx, test_size=val_fraction,
            random_state=RANDOM_SEED,
        )
        train_indices.append(tr)
        val_indices.append(va)

    train_idx = np.concatenate(train_indices)
    val_idx   = np.concatenate(val_indices)

    train_y = all_y[train_idx]
    val_y   = all_y[val_idx]

    log.info(f"  Split — Train: {len(train_idx):,} cells | Val: {len(val_idx):,} cells")
    log.info(f"  Train pseudotime_norm: mean={train_y.mean():.4f} | max={train_y.max():.4f}")
    log.info(f"  Val   pseudotime_norm: mean={val_y.mean():.4f} | max={val_y.max():.4f}")

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx,   dtype=torch.long),
        torch.tensor(train_y,   dtype=torch.float32),
        torch.tensor(val_y,     dtype=torch.float32),
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: TRAJRISKCONV — CUSTOM MESSAGE PASSING LAYER
# ═════════════════════════════════════════════════════════════════════════════

class TrajRiskConv(MessagePassing):
    """
    Custom graph convolution layer for trajectory-aware risk propagation.

    Standard GCNConv (Kipf & Welling 2017) applies symmetric degree
    normalisation: each edge contribution is scaled by
    1/sqrt(d_i) * 1/sqrt(d_j), where d_i and d_j are node degrees.
    This is appropriate for undirected graphs with uniform edge weights,
    but is incorrect here for two reasons:

    (1) Our graphs are directed: 0h → 6h, 6h → 24h, etc. Symmetric
        normalisation assumes undirected connectivity and would distort
        the asymmetric risk propagation direction.

    (2) Our edge weights are row-stochastic transition probabilities
        produced by the Week 5 Random Forest. They are already normalised
        such that sum_j T[i, j] = 1 for each source cell i. Applying
        GCN's additional degree normalisation would corrupt these weights.

    TrajRiskConv instead computes:

        agg_i  = sum_{j : (j → i) ∈ E_s}  T_s[j, i]  *  W_nbr  @ h_j
        h_i^l  = ReLU( agg_i  +  W_self @ h_i )

    where T_s[j, i] is the transition probability from source cell j to
    target cell i (pre-learned by the Week 5 RF), W_nbr and W_self are
    learnable weight matrices, and h_j is the feature vector of cell j.

    The residual term W_self @ h_i prevents information loss for cells
    with no incoming edges in a given scale (e.g., 0h cells in Scales
    2–4, which have no predecessors at those temporal scales).

    Parameters
    ----------
    in_channels  : int — input feature dimension
    out_channels : int — output embedding dimension
    """

    def __init__(self, in_channels: int, out_channels: int):
        # aggr='add' sums incoming messages (the sum is correctly weighted
        # because each message is multiplied by the edge weight T_s[j,i]).
        # flow='source_to_target' means: target node i aggregates from
        # all source nodes j that have edges pointing to i, which is the
        # forward direction of risk propagation along the trajectory.
        super().__init__(aggr="add", flow="source_to_target")

        self.in_channels  = in_channels
        self.out_channels = out_channels

        # Separate weight matrices for neighbour aggregation and self-loop.
        # Using two separate matrices (rather than a single matrix applied
        # to concatenated features) follows the design in Hamilton et al.
        # (GraphSAGE, 2017), which improves expressivity for directed graphs.
        self.W_nbr  = nn.Linear(in_channels,  out_channels, bias=False)
        self.W_self = nn.Linear(in_channels,  out_channels, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        # Glorot (Xavier) uniform initialisation — standard for GNN layers.
        nn.init.xavier_uniform_(self.W_nbr.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : FloatTensor (N, in_channels) — all node features
        edge_index  : LongTensor  (2, E)            — [source_idx; target_idx]
        edge_weight : FloatTensor (E,)              — T_s values (row-normalised)

        Returns
        -------
        h : FloatTensor (N, out_channels)
        """
        # Aggregate weighted source features into target nodes
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)  # (N, in_channels)

        # Combine neighbour aggregation with self-loop (residual)
        return self.W_nbr(agg) + self.W_self(x)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute the message from source node j to target node i.

        x_j         : (E, in_channels) — features of source nodes
        edge_weight : (E,)             — T_s[source_j → target_i]

        The message is simply the source cell's features scaled by the
        transition probability, which determines how much of the source
        cell's risk signature flows into the target cell.
        """
        return edge_weight.unsqueeze(-1) * x_j


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAJRISKGNN — FULL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

class TrajRiskGNN(nn.Module):
    """
    Multi-scale Graph Neural Network for trajectory-informed risk propagation.

    Architecture
    ------------
    Four parallel scale-specific GNN branches (one per temporal scale),
    each consisting of two TrajRiskConv layers. Scale branches are
    independent: they process different edge sets corresponding to
    different phases of the APAP injury response. This parallelism allows
    the model to learn scale-specific aggregation patterns — for example,
    the 0h → 6h scale captures which baseline hepatocytes transition into
    the acute stress state, while the 24h → 48h scale captures the fate
    decision between recovery and persistent compensation.

    After both layers of each branch, the resulting embeddings are
    concatenated across all four scales (256D total) and passed through
    a three-layer MLP with a Sigmoid final activation to produce a
    per-cell risk score in [0, 1].

    The output risk score is regressed against pseudotime_norm (0 → 1)
    using mean squared error loss. The continuity of the target ensures
    that the model learns the fine-grained gradient between recovered
    (low pseudotime_norm ≈ 0.02) and compensation-committed cells
    (high pseudotime_norm ≈ 0.89), which is the biologically meaningful
    signal that cannot be recovered from binary APAP/Control labels alone.

    Parameters
    ----------
    node_dim    : int — input node feature dimension (default 31)
    hidden_dim  : int — output dimension of each TrajRiskConv layer (default 64)
    n_scales    : int — number of temporal scales (default 4)
    dropout     : float — dropout probability in the fusion MLP (default 0.3)
    """

    def __init__(
        self,
        node_dim:   int   = NODE_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        n_scales:   int   = N_SCALES,
        dropout:    float = DROPOUT,
    ):
        super().__init__()

        self.node_dim   = node_dim
        self.hidden_dim = hidden_dim
        self.n_scales   = n_scales

        # Scale-specific GNN branches — one ModuleList per scale,
        # each containing N_CONV_LAYERS TrajRiskConv layers.
        self.scale_convs = nn.ModuleList()
        for _ in range(n_scales):
            branch = nn.ModuleList([
                TrajRiskConv(node_dim,   hidden_dim),
                TrajRiskConv(hidden_dim, hidden_dim),
            ])
            self.scale_convs.append(branch)

        # Batch normalisation after each conv layer for training stability.
        # Applied per scale, per layer.
        self.scale_bns = nn.ModuleList()
        for _ in range(n_scales):
            bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ])
            self.scale_bns.append(bns)

        # Fusion MLP: takes concatenated scale embeddings → scalar risk score.
        fusion_input_dim = n_scales * hidden_dim  # 256D

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_fusion_weights()

    def _init_fusion_weights(self):
        for layer in self.fusion_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x:            torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Full forward pass.

        Parameters
        ----------
        x            : FloatTensor (N, node_dim) — node features for all cells
        edge_indices : list of 4 LongTensors, each (2, E_s) — one per scale
        edge_weights : list of 4 FloatTensors, each (E_s,)  — one per scale

        Returns
        -------
        risk : FloatTensor (N,) — risk scores in [0, 1] for all N cells
        """
        scale_embeddings = []

        for s in range(self.n_scales):
            ei = edge_indices[s]  # (2, E_s)
            ew = edge_weights[s]  # (E_s,)
            h  = x                # start fresh from input features each scale

            for layer_idx in range(N_CONV_LAYERS):
                conv = self.scale_convs[s][layer_idx]
                bn   = self.scale_bns[s][layer_idx]

                h = conv(h, ei, ew)  # aggregate from trajectory neighbours
                h = bn(h)            # stabilise activations across the batch
                h = torch.relu(h)

            scale_embeddings.append(h)  # (N, hidden_dim)

        # Concatenate all scale embeddings into a single representation per cell
        h_cat = torch.cat(scale_embeddings, dim=1)  # (N, n_scales * hidden_dim)

        # Risk score via MLP with final Sigmoid
        risk = self.fusion_mlp(h_cat).squeeze(-1)  # (N,)

        return risk

    def get_scale_embeddings(
        self,
        x:            torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Return per-scale embeddings before fusion (for interpretability analyses
        and scale-importance assessment in Week 7/8 ablation studies).

        Returns
        -------
        embeddings : FloatTensor (N, n_scales, hidden_dim)
        """
        scale_embeddings = []

        for s in range(self.n_scales):
            ei = edge_indices[s]
            ew = edge_weights[s]
            h  = x

            for layer_idx in range(N_CONV_LAYERS):
                h = self.scale_convs[s][layer_idx](h, ei, ew)
                h = self.scale_bns[s][layer_idx](h)
                h = torch.relu(h)

            scale_embeddings.append(h.unsqueeze(1))  # (N, 1, hidden_dim)

        return torch.cat(scale_embeddings, dim=1)  # (N, n_scales, hidden_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def prepare_graph_tensors(
    graphs: Dict,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Move all scale graphs to the target device and extract edge tensors.

    Returns
    -------
    edge_indices : list of LongTensors  on DEVICE
    edge_weights : list of FloatTensors on DEVICE
    """
    edge_indices, edge_weights = [], []

    for sn in SCALE_NAMES:
        g  = graphs[sn]
        ei = g["edge_index"].to(DEVICE)
        ew = g["edge_weight"].to(DEVICE)
        edge_indices.append(ei)
        edge_weights.append(ew)

    return edge_indices, edge_weights


def compute_metrics(
    y_pred:      np.ndarray,
    y_true:      np.ndarray,
    cell_index:  pd.DataFrame,
    split_idx:   np.ndarray,
    threshold:   float = 0.10,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a set of cells.

    Metrics:
    - MSE: primary training loss (regression on pseudotime_norm)
    - Spearman ρ: rank correlation between predicted risk and pseudotime_norm
    - AUROC: binary discrimination (high-risk: pseudotime_norm > threshold)
    - AUPRC: area under precision-recall curve (informative for imbalanced tasks)

    The threshold of 0.10 was chosen to separate cells clearly in the
    compensation mode from those that have recovered. At pseudotime_norm > 0.10:
      - Virtually all 6h cells (median 0.895) are captured
      - A minority of 24h/48h/96h cells still in compensation state
    """
    mse  = float(np.mean((y_pred - y_true) ** 2))
    rho, p_spearman = spearmanr(y_pred, y_true)

    # Binary labels for AUROC: high-risk (compensation mode) vs low-risk (recovered)
    y_binary = (y_true > threshold).astype(int)

    if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
        auroc = float(roc_auc_score(y_binary, y_pred))
        auprc = float(average_precision_score(y_binary, y_pred))
    else:
        auroc = auprc = float("nan")

    return {
        "mse":     mse,
        "rmse":    float(np.sqrt(mse)),
        "spearman_rho": float(rho),
        "spearman_p":   float(p_spearman),
        "auroc":   auroc,
        "auprc":   auprc,
        "n_high_risk": int(y_binary.sum()),
        "n_low_risk":  int((1 - y_binary).sum()),
    }


def train_one_epoch(
    model:        TrajRiskGNN,
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    train_idx:    torch.Tensor,
    train_y:      torch.Tensor,
    optimizer:    optim.Optimizer,
    criterion:    nn.Module,
) -> float:
    """Single training epoch. Returns training MSE loss."""
    model.train()
    optimizer.zero_grad()

    # Full-batch forward pass (all 6,582 nodes; only train subset for loss)
    risk_all = model(x, edge_indices, edge_weights)

    # Loss on training cells only
    loss = criterion(risk_all[train_idx], train_y.to(DEVICE))
    loss.backward()

    # Gradient clipping to prevent exploding gradients in early epochs
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate(
    model:        TrajRiskGNN,
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    val_idx:      torch.Tensor,
    val_y:        torch.Tensor,
    criterion:    nn.Module,
    cell_index:   pd.DataFrame,
) -> Tuple[float, Dict]:
    """Evaluate model on the validation set. Returns val loss and metrics."""
    model.eval()

    risk_all  = model(x, edge_indices, edge_weights)
    val_loss  = float(criterion(risk_all[val_idx], val_y.to(DEVICE)).item())

    y_pred    = risk_all[val_idx].cpu().numpy()
    y_true    = val_y.numpy()
    val_idx_np = val_idx.numpy()

    metrics = compute_metrics(y_pred, y_true, cell_index, val_idx_np)

    return val_loss, metrics


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: QUICK DIAGNOSTIC RUN (10 EPOCHS)
# ═════════════════════════════════════════════════════════════════════════════

def run_diagnostic_training(
    model:        TrajRiskGNN,
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    train_idx:    torch.Tensor,
    val_idx:      torch.Tensor,
    train_y:      torch.Tensor,
    val_y:        torch.Tensor,
    cell_index:   pd.DataFrame,
    n_epochs:     int = N_EPOCHS_TEST,
) -> Dict:
    """
    Short diagnostic training run to verify that:
    (1) The model forward pass runs without error
    (2) Loss decreases over epochs (the model is learning)
    (3) Predicted risks are non-degenerate (not all 0.5)
    (4) Spearman correlation improves above chance

    Full training (200 epochs, early stopping) is done in Week 7.

    Returns a history dictionary for logging and figure generation.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history = {
        "train_loss": [], "val_loss": [],
        "val_spearman": [], "val_auroc": [], "val_rmse": [],
    }

    log.info(f"  Running {n_epochs}-epoch diagnostic training on {DEVICE}...")
    log.info(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Val ρ':>8}  {'Val AUROC':>10}")
    log.info("  " + "─" * 55)

    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model, x, edge_indices, edge_weights,
            train_idx, train_y, optimizer, criterion,
        )
        val_loss, metrics = evaluate(
            model, x, edge_indices, edge_weights,
            val_idx, val_y, criterion, cell_index,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(metrics["spearman_rho"])
        history["val_auroc"].append(metrics["auroc"])
        history["val_rmse"].append(metrics["rmse"])

        log.info(
            f"  {epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
            f"{metrics['spearman_rho']:>8.4f}  {metrics['auroc']:>10.4f}"
        )

    elapsed = time.time() - t0
    log.info(f"  {n_epochs} epochs in {elapsed:.1f}s ({elapsed/n_epochs:.2f}s/epoch)")

    # Diagnostic checks
    loss_decreased = history["train_loss"][-1] < history["train_loss"][0]
    spearman_positive = history["val_spearman"][-1] > 0.10

    log.info(f"  Loss decreased over {n_epochs} epochs: {'✓' if loss_decreased else '✗'}")
    log.info(f"  Spearman ρ > 0.10 at epoch {n_epochs}: {'✓' if spearman_positive else '✗'}")

    if not loss_decreased:
        log.warning(
            "  ⚠ Loss did not decrease. Check learning rate, "
            "node features, and label scaling."
        )

    return history


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: VALIDATION FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def make_diagnostic_figures(
    history:         Dict,
    model:           TrajRiskGNN,
    x:               torch.Tensor,
    edge_indices:    List[torch.Tensor],
    edge_weights:    List[torch.Tensor],
    cell_index:      pd.DataFrame,
    xgb_risk:        np.ndarray,
) -> None:
    """
    Three-panel diagnostic figure:
    Panel A: Training and validation MSE loss over 10 epochs
    Panel B: Predicted risk vs pseudotime_norm (scatter, coloured by timepoint)
    Panel C: XGBoost initial risk distribution by timepoint (baseline comparison)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("TrajRisk Week 6 — Architecture Diagnostics (10-epoch test)", fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)
    colors_tp = {"0h": "#2ecc71", "6h": "#e74c3c", "24h": "#e67e22", "48h": "#9b59b6", "96h": "#3498db"}

    # ── A: Loss curves ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train MSE", color="#3498db", lw=2)
    ax.plot(epochs, history["val_loss"],   label="Val MSE",   color="#e74c3c", lw=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("A  Training and Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # ── B: Predicted risk vs pseudotime_norm ──────────────────────────────
    ax = axes[1]
    model.eval()
    with torch.no_grad():
        risk_all = model(x, edge_indices, edge_weights).cpu().numpy()

    pt_norm   = cell_index["pseudotime_norm"].values
    timepoints = cell_index["timepoint_str"].values

    for tp, col in colors_tp.items():
        mask = timepoints == tp
        ax.scatter(pt_norm[mask], risk_all[mask], c=col, alpha=0.4, s=8, label=tp)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="y = x")
    ax.set_xlabel("pseudotime_norm (true label)")
    ax.set_ylabel("Predicted risk score")
    ax.set_title("B  Predicted Risk vs Ground Truth\n(after 10 epochs — expect loose correlation)")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(alpha=0.3)

    rho, _ = spearmanr(risk_all, pt_norm)
    ax.text(0.05, 0.92, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes, fontsize=9)

    # ── C: XGBoost initial risk by timepoint ──────────────────────────────
    ax = axes[2]
    import seaborn as sns
    df_plot = cell_index.copy()
    df_plot["xgb_risk"] = xgb_risk[cell_index["global_idx"].values]
    tp_order = ["0h", "6h", "24h", "48h", "96h"]

    sns.violinplot(
        data=df_plot, x="timepoint_str", y="xgb_risk",
        order=tp_order, palette=[colors_tp[t] for t in tp_order],
        ax=ax, inner="box", linewidth=0.8,
    )
    ax.set_xlabel("Timepoint"); ax.set_ylabel("XGBoost Initial Risk Score")
    ax.set_title("C  XGBoost Initial Risk by Timepoint\n(baseline node feature, before graph propagation)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "week6_architecture_diagnostics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Figure saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: SAVE ALL OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def save_week6_outputs(
    model:       TrajRiskGNN,
    xgb_model:   XGBClassifier,
    xgb_risk:    np.ndarray,
    x:           torch.Tensor,
    train_idx:   torch.Tensor,
    val_idx:     torch.Tensor,
    train_y:     torch.Tensor,
    val_y:       torch.Tensor,
    history:     Dict,
    metrics:     Dict,
) -> None:
    """Save all Week 6 outputs needed for Week 7 (full training)."""

    # 1. Test model checkpoint (10-epoch; Week 7 will overwrite with final)
    torch.save({
        "model_state_dict":  model.state_dict(),
        "architecture": {
            "node_dim":   model.node_dim,
            "hidden_dim": model.hidden_dim,
            "n_scales":   model.n_scales,
            "dropout":    DROPOUT,
            "n_parameters": model.count_parameters(),
        },
        "n_epochs_trained": N_EPOCHS_TEST,
        "final_val_metrics": metrics,
    }, MODEL_DIR / "week6_trajrisk_test.pt")
    log.info("  Saved test model checkpoint: week6_trajrisk_test.pt")

    # 2. XGBoost scorer
    joblib.dump(xgb_model, MODEL_DIR / "week6_xgboost_model.pkl")
    log.info("  Saved XGBoost model: week6_xgboost_model.pkl")

    # 3. Node features (needed for Week 7 training without re-running everything)
    torch.save(x, DATA_DIR / "week6_node_features.pt")
    log.info(f"  Saved node features: {tuple(x.shape)} → week6_node_features.pt")

    # 4. XGBoost initial risk scores
    np.save(DATA_DIR / "week6_xgboost_risk.npy", xgb_risk)
    log.info("  Saved XGBoost risk scores: week6_xgboost_risk.npy")

    # 5. Training data (indices and labels for Week 7)
    torch.save({
        "train_idx": train_idx,
        "val_idx":   val_idx,
        "train_y":   train_y,
        "val_y":     val_y,
    }, DATA_DIR / "week6_training_data.pt")
    log.info("  Saved train/val split: week6_training_data.pt")

    # 6. Diagnostics JSON
    summary = {
        "week": 6,
        "architecture": {
            "node_dim":        NODE_DIM,
            "hidden_dim":      HIDDEN_DIM,
            "n_scales":        N_SCALES,
            "conv_layers_per_scale": N_CONV_LAYERS,
            "dropout":         DROPOUT,
            "n_parameters":    model.count_parameters(),
            "fusion_mlp_dims": "256→128→32→1",
        },
        "xgboost": {
            "training_cells":  "0h Control + 6h APAP only",
            "val_auroc_0h_vs_6h": metrics.get("xgb_val_auroc", "see log"),
        },
        "training": {
            "n_cells":      6582,
            "n_train":      int(len(train_idx)),
            "n_val":        int(len(val_idx)),
            "n_epochs_test": N_EPOCHS_TEST,
            "learning_rate": LEARNING_RATE,
            "weight_decay":  WEIGHT_DECAY,
            "device":        str(DEVICE),
            "loss_function": "MSE on pseudotime_norm",
        },
        "test_run_history": {k: [round(v, 6) for v in vlist] for k, vlist in history.items()},
        "final_val_metrics": {k: round(v, 6) if isinstance(v, float) else v
                              for k, v in metrics.items()},
    }

    with open(RES_DIR / "week6_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  Saved diagnostics: week6_diagnostics.json")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 70)
    log.info("TRAJRISK — WEEK 6: GNN ARCHITECTURE IMPLEMENTATION")
    log.info("=" * 70)
    log.info(f"Device: {DEVICE}")
    log.info(f"Node features: {NODE_DIM}D (scVI {SCVI_DIM}D + XGBoost {INIT_RISK_DIM}D)")
    log.info(f"Architecture: {N_SCALES} scales × {N_CONV_LAYERS} TrajRiskConv layers → {HIDDEN_DIM}D → MLP → risk")
    log.info(f"Training objective: MSE regression on pseudotime_norm")
    t_start = time.time()

    # ── 1. Load Week 5 outputs ─────────────────────────────────────────────
    log.info("\n[1/9] Loading Week 5 outputs...")
    graphs, cell_index, walesky = load_week5_outputs()
    walesky_scvi = walesky.obsm["X_scvi"].astype(np.float32)

    # ── 2. XGBoost initial risk ────────────────────────────────────────────
    log.info("\n[2/9] Training XGBoost initial risk scorer (0h vs 6h)...")
    xgb_model, xgb_risk = train_xgboost_risk(walesky_scvi, cell_index)

    # ── 3. Node feature assembly ───────────────────────────────────────────
    log.info("\n[3/9] Assembling node features (31D per cell)...")
    x = assemble_node_features(walesky_scvi, xgb_risk).to(DEVICE)

    # ── 4. Train/val split ─────────────────────────────────────────────────
    log.info("\n[4/9] Creating stratified train/validation split...")
    train_idx, val_idx, train_y, val_y = make_train_val_split(cell_index)

    # ── 5. Prepare graph tensors ───────────────────────────────────────────
    log.info("\n[5/9] Preparing graph tensors on device...")
    edge_indices, edge_weights = prepare_graph_tensors(graphs)

    for s, sn in enumerate(SCALE_NAMES):
        log.info(f"  {sn}: {edge_indices[s].shape[1]:,} edges on {DEVICE}")

    # ── 6. Instantiate model ───────────────────────────────────────────────
    log.info("\n[6/9] Instantiating TrajRiskGNN...")
    model = TrajRiskGNN(
        node_dim=NODE_DIM, hidden_dim=HIDDEN_DIM,
        n_scales=N_SCALES, dropout=DROPOUT,
    ).to(DEVICE)

    log.info(f"  Parameters: {model.count_parameters():,}")
    log.info(f"  Scale branches: {N_SCALES} × {N_CONV_LAYERS} TrajRiskConv layers")
    log.info(f"  Fusion MLP: {N_SCALES*HIDDEN_DIM}D → 128D → 32D → 1D")

    # Verify architecture with a single forward pass before training
    log.info("  Verifying forward pass...")
    model.eval()
    with torch.no_grad():
        test_out = model(x, edge_indices, edge_weights)

    assert test_out.shape == (walesky.n_obs,), \
        f"Expected output shape ({walesky.n_obs},), got {test_out.shape}"
    assert (test_out >= 0).all() and (test_out <= 1).all(), \
        "Output risk scores not in [0, 1]. Check Sigmoid activation."

    log.info(f"  ✓ Forward pass successful: output shape {tuple(test_out.shape)}")
    log.info(f"  ✓ Risk score range after random init: [{test_out.min():.4f}, {test_out.max():.4f}]")

    # ── 7. Diagnostic training (10 epochs) ────────────────────────────────
    log.info(f"\n[7/9] Running {N_EPOCHS_TEST}-epoch diagnostic training...")
    history = run_diagnostic_training(
        model, x, edge_indices, edge_weights,
        train_idx, val_idx, train_y, val_y, cell_index,
    )

    # Final evaluation
    criterion = nn.MSELoss()
    final_val_loss, final_metrics = evaluate(
        model, x, edge_indices, edge_weights,
        val_idx, val_y, criterion, cell_index,
    )
    final_metrics["xgb_val_auroc"] = "see log"

    log.info(f"\n  After {N_EPOCHS_TEST} epochs:")
    log.info(f"    Val MSE:      {final_val_loss:.6f}")
    log.info(f"    Val RMSE:     {final_metrics['rmse']:.6f}")
    log.info(f"    Val Spearman: {final_metrics['spearman_rho']:.4f}")
    log.info(f"    Val AUROC:    {final_metrics['auroc']:.4f}")
    log.info(f"    High-risk cells (pt_norm > 0.10): {final_metrics['n_high_risk']:,}")

    # ── 8. Figures ─────────────────────────────────────────────────────────
    log.info("\n[8/9] Generating diagnostic figures...")
    make_diagnostic_figures(
        history, model, x, edge_indices, edge_weights, cell_index, xgb_risk,
    )

    # ── 9. Save all outputs ────────────────────────────────────────────────
    log.info("\n[9/9] Saving outputs...")
    save_week6_outputs(
        model, xgb_model, xgb_risk, x,
        train_idx, val_idx, train_y, val_y,
        history, final_metrics,
    )

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info("WEEK 6 COMPLETE")
    log.info("=" * 70)
    log.info(f"Total runtime: {elapsed/60:.1f} minutes")
    log.info("")
    log.info("Architecture confirmed:")
    log.info(f"  TrajRiskGNN  — {model.count_parameters():,} parameters")
    log.info(f"  Node features — {NODE_DIM}D (scVI {SCVI_DIM}D + XGBoost {INIT_RISK_DIM}D)")
    log.info(f"  Graph         — {N_SCALES} directed scales, {sum(ei.shape[1] for ei in edge_indices):,} total edges")
    log.info(f"  Loss function — MSE on pseudotime_norm (continuous 0→1)")
    log.info("")
    log.info("Key diagnostic results:")
    log.info(f"  Final val MSE:       {final_val_loss:.6f}")
    log.info(f"  Final val Spearman:  {final_metrics['spearman_rho']:.4f} (expect low at 10 epochs; >0.60 after full training)")
    log.info(f"  Final val AUROC:     {final_metrics['auroc']:.4f}")
    log.info("")
    log.info("Files written:")
    log.info(f"  {MODEL_DIR}/week6_trajrisk_test.pt")
    log.info(f"  {MODEL_DIR}/week6_xgboost_model.pkl")
    log.info(f"  {DATA_DIR}/week6_node_features.pt")
    log.info(f"  {DATA_DIR}/week6_xgboost_risk.npy")
    log.info(f"  {DATA_DIR}/week6_training_data.pt")
    log.info(f"  {RES_DIR}/week6_diagnostics.json")
    log.info(f"  {FIG_DIR}/week6_architecture_diagnostics.png")
    log.info("")
    log.info("→ Week 6 complete. Ready for Week 7: Full Model Training (200 epochs).")
    log.info("  Load week6_node_features.pt, week6_training_data.pt, multiscale_graphs.pt")


if __name__ == "__main__":
    main()
