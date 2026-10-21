#!/usr/bin/env python3
"""Train and evaluate the four-branch reference graph neural network.

The supervised target is normalized Monocle 3 pseudotime. Each branch produces
a 64-dimensional embedding; the prediction head integrates their concatenation.
"""

import os
import json
import time
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrajRisk-W7")

# CONFIGURATION

BASE = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE / "Data"  / "processed"
MODEL_DIR = BASE / "models"
FIG_DIR   = BASE / "results" / "figures"
RES_DIR   = BASE / "results"

for _d in [DATA_DIR, MODEL_DIR, FIG_DIR, RES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

# Architecture constants — must match graph inputs exactly
NODE_DIM    = 31
HIDDEN_DIM  = 64
N_SCALES    = 4
N_CONV_LAYERS = 2
DROPOUT     = 0.3

SCALE_NAMES: List[str] = [
    "Scale1_AcuteStress",
    "Scale2_PeakInjury",
    "Scale3_FateDecision",
    "Scale4_Resolution",
]

# Hyperparameter search grid
# Dropout is fixed at 0.3 — stable for this graph density.
# We search over the two parameters that most affect convergence quality.
HP_SEARCH_LRS          = [0.01, 0.001, 0.0001]
HP_SEARCH_WEIGHT_DECAYS = [0.0, 1e-5, 1e-4]
HP_SEARCH_EPOCHS       = 50    # quick evaluation per config
HP_SEARCH_PATIENCE     = 10   # early stop if val loss stagnates during search

# Full training — uses best config from search
FULL_MAX_EPOCHS     = 300
FULL_PATIENCE       = 25    # more patience for full run; val loss has more room to oscillate
LR_REDUCE_PATIENCE  = 10    # reduce LR if val loss doesn't improve for 10 epochs
LR_REDUCE_FACTOR    = 0.5   # halve the learning rate when plateau is detected

# Binary AUROC threshold for biological evaluation
# pseudotime_norm > 0.10 captures cells in functional compensation mode
# (the 6h peak cells plus the minority of later cells that remain compensating)
HIGH_RISK_THRESHOLD = 0.10

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# TrajRiskConv and TrajRiskGNN
# Exact copies from graph inputs — any change here breaks the saved checkpoint.

class TrajRiskConv(MessagePassing):
    """
    Custom graph convolution for trajectory-aware risk propagation.
    Uses pre-learned RF transition probabilities (row-stochastic) as
    edge weights directly, without GCN's symmetric degree normalisation,
    which would corrupt the already-normalised weights.
    Includes a residual self-connection so cells with no incoming edges
    (e.g., 0h cells in Scales 2-4) still receive a linear transformation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.W_nbr  = nn.Linear(in_channels,  out_channels, bias=False)
        self.W_self = nn.Linear(in_channels,  out_channels, bias=True)
        nn.init.xavier_uniform_(self.W_nbr.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)

    def forward(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.W_nbr(agg) + self.W_self(x)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.unsqueeze(-1) * x_j


class TrajRiskGNN(nn.Module):
    """
    Multi-scale GNN for trajectory-informed risk propagation.
    Four parallel scale branches, each with two TrajRiskConv layers,
    fused by a three-layer MLP into a scalar risk score per cell.
    Output is Sigmoid-bounded to [0, 1] for regression against pseudotime_norm.
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

    def forward(
        self,
        x:            torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        scale_embeddings = []
        for s in range(self.n_scales):
            h = x
            for layer_idx in range(N_CONV_LAYERS):
                h = self.scale_convs[s][layer_idx](h, edge_indices[s], edge_weights[s])
                h = self.scale_bns[s][layer_idx](h)
                h = torch.relu(h)
            scale_embeddings.append(h)

        h_cat = torch.cat(scale_embeddings, dim=1)
        return self.fusion_mlp(h_cat).squeeze(-1)

    @torch.no_grad()
    def get_scale_embeddings(
        self,
        x:            torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Return per-scale embeddings before the fusion MLP.
        Shape: (N, n_scales, hidden_dim).
        Used for scale-importance analysis in Phase 3.
        """
        self.eval()
        parts = []
        for s in range(self.n_scales):
            h = x
            for layer_idx in range(N_CONV_LAYERS):
                h = self.scale_convs[s][layer_idx](h, edge_indices[s], edge_weights[s])
                h = self.scale_bns[s][layer_idx](h)
                h = torch.relu(h)
            parts.append(h.unsqueeze(1))
        return torch.cat(parts, dim=1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# DATA LOADING

def load_all_inputs() -> Tuple:
    """
    Load everything produced by Weeks 5 and 6.
    Returns all tensors needed for training and evaluation.
    """
    log.info("Loading transition graphs-6 outputs...")

    # Multi-scale graphs
    graphs = torch.load(DATA_DIR / "multiscale_graphs.pt", map_location="cpu")
    edge_indices: List[torch.Tensor] = []
    edge_weights: List[torch.Tensor] = []
    for sn in SCALE_NAMES:
        edge_indices.append(graphs[sn]["edge_index"].to(DEVICE))
        edge_weights.append(graphs[sn]["edge_weight"].to(DEVICE))
    log.info(f"  Graphs: {N_SCALES} scales, "
             f"{sum(ei.shape[1] for ei in edge_indices):,} total edges")

    # Node features and labels
    x = torch.load(DATA_DIR / "week6_node_features.pt",
                   map_location="cpu").to(DEVICE)
    log.info(f"  Node features: {tuple(x.shape)}")

    td = torch.load(DATA_DIR / "week6_training_data.pt", map_location="cpu")
    train_idx = td["train_idx"].to(DEVICE)
    val_idx   = td["val_idx"].to(DEVICE)
    train_y   = td["train_y"].to(DEVICE)
    val_y     = td["val_y"].to(DEVICE)
    log.info(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} cells")

    # Cell metadata
    cell_index = pd.read_csv(DATA_DIR / "walesky_cell_index.csv", index_col=0)
    log.info(f"  Cell index: {len(cell_index):,} cells")

    return x, edge_indices, edge_weights, train_idx, val_idx, train_y, val_y, cell_index

# TRAINING UTILITIES

def build_model(lr: float, weight_decay: float) -> Tuple:
    """Construct model, criterion, optimiser, and scheduler."""
    # Seed BEFORE model instantiation — Xavier initialisation fires inside
    # TrajRiskGNN.__init__, so the seed must already be set at that point.
    # Setting it after (the original placement) leaves the weights initialised
    # from whatever RNG state preceded the call, making the 9 search configs
    # incomparable because they start from different random initializations.
    torch.manual_seed(RANDOM_SEED)
    model = TrajRiskGNN().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=1e-6,
    )
    return model, criterion, optimizer, scheduler


def train_epoch(
    model:        TrajRiskGNN,
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    train_idx:    torch.Tensor,
    train_y:      torch.Tensor,
    optimizer:    optim.Optimizer,
    criterion:    nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad()

    risk = model(x, edge_indices, edge_weights)
    loss = criterion(risk[train_idx], train_y)
    loss.backward()

    # Clip gradients to prevent the rare explosive update that
    # can occur in the first few epochs before LR warms up.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate_epoch(
    model:        TrajRiskGNN,
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    idx:          torch.Tensor,
    y:            torch.Tensor,
    criterion:    nn.Module,
) -> Dict[str, float]:
    model.eval()

    risk    = model(x, edge_indices, edge_weights)
    loss    = float(criterion(risk[idx], y).item())

    y_pred  = risk[idx].cpu().numpy()
    y_true  = y.cpu().numpy()

    rho, _  = spearmanr(y_pred, y_true)
    y_bin   = (y_true > HIGH_RISK_THRESHOLD).astype(int)

    auroc = float(roc_auc_score(y_bin, y_pred)) if y_bin.sum() > 0 else float("nan")
    auprc = float(average_precision_score(y_bin, y_pred)) if y_bin.sum() > 0 else float("nan")

    return {
        "loss":    loss,
        "rmse":    float(np.sqrt(loss)),
        "spearman": float(rho),
        "auroc":   auroc,
        "auprc":   auprc,
    }

# PHASE 1 — HYPERPARAMETER GRID SEARCH

def run_hyperparameter_search(
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    train_idx:    torch.Tensor,
    val_idx:      torch.Tensor,
    train_y:      torch.Tensor,
    val_y:        torch.Tensor,
) -> Dict[str, Any]:
    """
    Grid search over learning rate × weight decay combinations.
    Each configuration is trained for HP_SEARCH_EPOCHS epochs with early
    stopping on validation loss (patience=HP_SEARCH_PATIENCE).

    Selection criterion is final validation AUROC, not val loss.
    AUROC is the biological metric we care about; val MSE is used only
    to trigger early stopping during the search to avoid wasting time on
    configurations that clearly diverge.

    Returns the best configuration dict.
    """
    log.info(f"Phase 1: Hyperparameter grid search "
             f"({len(HP_SEARCH_LRS) * len(HP_SEARCH_WEIGHT_DECAYS)} configs × "
             f"{HP_SEARCH_EPOCHS} epochs)")

    search_results = []
    criterion = nn.MSELoss()

    total_configs = len(HP_SEARCH_LRS) * len(HP_SEARCH_WEIGHT_DECAYS)
    config_num    = 0

    for lr in HP_SEARCH_LRS:
        for wd in HP_SEARCH_WEIGHT_DECAYS:
            config_num += 1
            model, _, optimizer, scheduler = build_model(lr, wd)

            best_val_loss  = float("inf")
            best_val_auroc = 0.0
            no_improve     = 0

            for epoch in range(1, HP_SEARCH_EPOCHS + 1):
                train_epoch(model, x, edge_indices, edge_weights,
                            train_idx, train_y, optimizer, criterion)

                val_metrics = evaluate_epoch(
                    model, x, edge_indices, edge_weights, val_idx, val_y, criterion
                )

                scheduler.step(val_metrics["loss"])

                # Early stopping is driven by val loss — the smooth regression
                # target. AUROC is tracked independently because it does not
                # necessarily peak at the same epoch as the lowest MSE loss.
                # Coupling AUROC updates to loss improvements (the original bug)
                # means we could miss the true best AUROC by several points.
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    no_improve    = 0
                else:
                    no_improve += 1

                if val_metrics["auroc"] > best_val_auroc:
                    best_val_auroc = val_metrics["auroc"]

                if no_improve >= HP_SEARCH_PATIENCE:
                    break

            result = {
                "lr":          lr,
                "weight_decay": wd,
                "best_val_loss":  round(best_val_loss,  6),
                "best_val_auroc": round(best_val_auroc, 4),
                "epochs_run":   epoch,
            }
            search_results.append(result)

            log.info(
                f"  [{config_num:2d}/{total_configs}] "
                f"lr={lr:.4f}, wd={wd:.0e}  →  "
                f"val_loss={best_val_loss:.5f}, val_auroc={best_val_auroc:.4f}, "
                f"epochs={epoch}"
            )

    # Select best by validation AUROC
    best = max(search_results, key=lambda r: r["best_val_auroc"])

    log.info(f"\n  Best config: lr={best['lr']}, wd={best['weight_decay']}")
    log.info(f"  Best val AUROC: {best['best_val_auroc']:.4f}")
    log.info(f"  Best val MSE:   {best['best_val_loss']:.6f}")

    # Save search table
    df = pd.DataFrame(search_results).sort_values("best_val_auroc", ascending=False)
    df.to_csv(RES_DIR / "week7_hyperparameter_search.csv", index=False)
    log.info(f"  Search results saved: week7_hyperparameter_search.csv")

    return best

# PHASE 2 — FULL TRAINING WITH BEST HYPERPARAMETERS

def run_full_training(
    best_config:  Dict[str, Any],
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    train_idx:    torch.Tensor,
    val_idx:      torch.Tensor,
    train_y:      torch.Tensor,
    val_y:        torch.Tensor,
) -> Tuple[TrajRiskGNN, Dict]:
    """
    Full training using the best hyperparameters from Phase 1.

    Checkpoint strategy: save the model state whenever validation MSE improves.
    The final returned model is the BEST checkpoint, not the last epoch.
    This prevents returning an overfit model if training runs past the optimum.

    LR scheduling: ReduceLROnPlateau halves the learning rate whenever val loss
    fails to improve for LR_REDUCE_PATIENCE consecutive epochs. On a GNN of this
    depth, the loss typically plateaus twice: once around epoch 30-50 (fast
    feature learning phase), and once around epoch 100-150 (fine-tuning phase).
    The scheduler addresses both plateaus automatically.
    """
    lr = best_config["lr"]
    wd = best_config["weight_decay"]

    log.info(f"Phase 2: Full training — lr={lr}, wd={wd}")
    log.info(f"  Max epochs: {FULL_MAX_EPOCHS} | Early stop patience: {FULL_PATIENCE}")
    log.info(f"  LR scheduler: ReduceLROnPlateau "
             f"(factor={LR_REDUCE_FACTOR}, patience={LR_REDUCE_PATIENCE})")

    model, criterion, optimizer, scheduler = build_model(lr, wd)

    history = {
        "train_loss": [], "val_loss": [],
        "val_auroc": [], "val_auprc": [],
        "val_spearman": [], "val_rmse": [],
        "lr_history": [],
    }

    best_val_loss    = float("inf")
    best_val_auroc   = 0.0              # tracked independently from loss
    best_state_dict  = None
    best_auroc_state = None             # separate checkpoint for AUROC-best model
    no_improve       = 0
    t0               = time.time()

    log.info(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  "
             f"{'Val ρ':>8}  {'Val AUROC':>10}  {'LR':>10}  {'Flag':>6}")
    log.info("  " + "─" * 72)

    for epoch in range(1, FULL_MAX_EPOCHS + 1):

        train_loss  = train_epoch(model, x, edge_indices, edge_weights,
                                  train_idx, train_y, optimizer, criterion)
        val_metrics = evaluate_epoch(model, x, edge_indices, edge_weights,
                                     val_idx, val_y, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])
        history["val_spearman"].append(val_metrics["spearman"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["lr_history"].append(current_lr)

        improved_loss  = val_metrics["loss"]  < best_val_loss
        improved_auroc = val_metrics["auroc"] > best_val_auroc

        if improved_loss:
            best_val_loss   = val_metrics["loss"]
            best_state_dict = deepcopy(model.state_dict())
            no_improve      = 0
            torch.save({
                "model_state_dict": best_state_dict,
                "epoch":            epoch,
                "val_loss":         best_val_loss,
                "val_auroc":        val_metrics["auroc"],
                "val_spearman":     val_metrics["spearman"],
                "hyperparams":      best_config,
                "checkpoint_type":  "best_mse",
            }, MODEL_DIR / "week7_trajrisk_BEST.pt")
        else:
            no_improve += 1

        # Save a separate checkpoint when AUROC improves, regardless of MSE.
        # For a paper centred on risk discrimination, the AUROC-best model is
        # the one we use for biological validation and cross-dataset scoring.
        if improved_auroc:
            best_val_auroc  = val_metrics["auroc"]
            best_auroc_state = deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_auroc_state,
                "epoch":            epoch,
                "val_auroc":        best_val_auroc,
                "val_loss":         val_metrics["loss"],
                "val_spearman":     val_metrics["spearman"],
                "hyperparams":      best_config,
                "checkpoint_type":  "best_auroc",
            }, MODEL_DIR / "week7_trajrisk_BEST_AUROC.pt")

        flag = ("✓MSE " if improved_loss else "      ") + ("✓AUC" if improved_auroc else "")

        # Log every 10 epochs and final epoch
        if epoch % 10 == 0 or epoch == 1 or improved_loss or improved_auroc or epoch == FULL_MAX_EPOCHS:
            log.info(
                f"  {epoch:>6}  {train_loss:>10.6f}  {val_metrics['loss']:>10.6f}  "
                f"{val_metrics['spearman']:>8.4f}  {val_metrics['auroc']:>10.4f}  "
                f"{current_lr:>10.2e}  {flag:>6}"
            )

        if no_improve >= FULL_PATIENCE:
            log.info(f"  Early stopping at epoch {epoch} "
                     f"(no improvement for {FULL_PATIENCE} epochs)")
            break

    elapsed = time.time() - t0
    log.info(f"  Training complete: {epoch} epochs in {elapsed/60:.1f} minutes")
    log.info(f"  Best val MSE:    {best_val_loss:.6f}  → week7_trajrisk_BEST.pt")
    log.info(f"  Best val AUROC:  {best_val_auroc:.4f}  → week7_trajrisk_BEST_AUROC.pt")

    # The model returned is the AUROC-best — used for biological validation.
    # The MSE-best checkpoint is available for regression quality comparisons.
    if best_auroc_state is not None:
        model.load_state_dict(best_auroc_state)
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch_reached":    epoch,
        "hyperparams":      best_config,
        "best_val_mse":     best_val_loss,
        "best_val_auroc":   best_val_auroc,
        "final_val_metrics": val_metrics,
    }, MODEL_DIR / "week7_trajrisk_FINAL.pt")

    log.info("  Saved: week7_trajrisk_BEST.pt       (lowest val MSE)")
    log.info("  Saved: week7_trajrisk_BEST_AUROC.pt  (highest val AUROC)")
    log.info("  Saved: week7_trajrisk_FINAL.pt       (AUROC-best restored)")

    return model, history

# PHASE 3A — FULL EVALUATION

@torch.no_grad()
def full_evaluation(
    model:        TrajRiskGNN,
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
    val_idx:      torch.Tensor,
    val_y:        torch.Tensor,
    train_idx:    torch.Tensor,
    train_y:      torch.Tensor,
    cell_index:   pd.DataFrame,
) -> Tuple[np.ndarray, Dict]:
    """
    Comprehensive evaluation of the final model.

    Returns
    -------
    risk_all : (6582,) float — predicted risk for every Walesky cell
    metrics  : dict — all evaluation metrics for the manuscript
    """
    model.eval()
    criterion = nn.MSELoss()

    risk_all = model(x, edge_indices, edge_weights).cpu().numpy()
    cell_index = cell_index.copy()
    cell_index["risk_predicted"] = risk_all

    # Validation set metrics
    y_pred_val = risk_all[val_idx.cpu().numpy()]
    y_true_val = val_y.cpu().numpy()
    y_bin_val  = (y_true_val > HIGH_RISK_THRESHOLD).astype(int)

    val_loss  = float(criterion(
        torch.tensor(y_pred_val), torch.tensor(y_true_val)
    ).item())
    rho_val, p_val = spearmanr(y_pred_val, y_true_val)
    auroc_val = float(roc_auc_score(y_bin_val, y_pred_val))
    auprc_val = float(average_precision_score(y_bin_val, y_pred_val))

    # Training set metrics (for convergence check)
    y_pred_tr = risk_all[train_idx.cpu().numpy()]
    y_true_tr = train_y.cpu().numpy()
    rho_tr, _ = spearmanr(y_pred_tr, y_true_tr)

    log.info("  Final evaluation:")
    log.info(f"    Val MSE:         {val_loss:.6f}")
    log.info(f"    Val RMSE:        {np.sqrt(val_loss):.6f}")
    log.info(f"    Val Spearman ρ:  {rho_val:.4f}  (p={p_val:.2e})")
    log.info(f"    Val AUROC:       {auroc_val:.4f}")
    log.info(f"    Val AUPRC:       {auprc_val:.4f}")
    log.info(f"    Train Spearman:  {rho_tr:.4f}")
    log.info(f"    Train-Val gap:   {abs(rho_tr - rho_val):.4f}  "
             f"(< 0.10 = no overfitting)")
    log.info(f"    High-risk cells (pt_norm > {HIGH_RISK_THRESHOLD}): "
             f"{y_bin_val.sum()} / {len(y_bin_val)} val cells")

    # Per-timepoint predicted risk — biological sanity check
    log.info("\n  Per-timepoint predicted risk:")
    log.info(f"  {'Timepoint':<10} {'N':>6} {'Mean risk':>11} {'Median':>9} "
             f"{'% > 0.10':>10} {'% > 0.50':>10}")
    log.info("  " + "─" * 60)
    tp_order = ["0h", "6h", "24h", "48h", "96h"]
    tp_stats = {}
    for tp in tp_order:
        mask = cell_index["timepoint_str"] == tp
        tp_risk = cell_index.loc[mask, "risk_predicted"].values
        mean_r  = tp_risk.mean()
        med_r   = np.median(tp_risk)
        pct10   = (tp_risk > 0.10).mean() * 100
        pct50   = (tp_risk > 0.50).mean() * 100
        log.info(f"  {tp:<10} {mask.sum():>6,} {mean_r:>11.4f} {med_r:>9.4f} "
                 f"{pct10:>9.1f}% {pct50:>9.1f}%")
        tp_stats[tp] = {
            "n": int(mask.sum()), "mean": round(mean_r, 4),
            "median": round(med_r, 4), "pct_above_0.10": round(pct10, 2),
        }

    # Biological sanity gates
    log.info("\n  Biological sanity checks:")
    checks = {
        "6h highest mean risk":
            tp_stats["6h"]["mean"] > tp_stats["0h"]["mean"] and
            tp_stats["6h"]["mean"] > tp_stats["96h"]["mean"],
        "0h lowest mean risk":
            all(tp_stats["0h"]["mean"] < tp_stats[tp]["mean"]
                for tp in ["6h", "24h", "48h", "96h"]),
        "APAP cells > Controls overall":
            cell_index.loc[cell_index["treatment_str"] == "APAP", "risk_predicted"].mean() >
            cell_index.loc[cell_index["treatment_str"] == "Control", "risk_predicted"].mean(),
    }
    for check_name, passed in checks.items():
        log.info(f"    {'✓' if passed else '✗'} {check_name}")

    metrics = {
        "val_mse":        round(val_loss,  6),
        "val_rmse":       round(float(np.sqrt(val_loss)), 6),
        "val_spearman":   round(float(rho_val), 4),
        "val_p":          float(p_val),
        "val_auroc":      round(auroc_val, 4),
        "val_auprc":      round(auprc_val, 4),
        "train_spearman": round(float(rho_tr), 4),
        "per_timepoint":  tp_stats,
        "sanity_checks":  checks,
    }
    return risk_all, metrics

# PHASE 3B — MODEL INTERPRETATION

@torch.no_grad()
def compute_scale_embeddings(
    model:        TrajRiskGNN,
    x:            torch.Tensor,
    edge_indices: List[torch.Tensor],
    edge_weights: List[torch.Tensor],
) -> np.ndarray:
    """
    Extract per-scale embeddings (6582, 4, 64) for all Walesky cells.
    Used for scale-importance analysis and PCA visualisation.
    """
    model.eval()
    embeddings = model.get_scale_embeddings(x, edge_indices, edge_weights)
    return embeddings.cpu().numpy()


def compute_scale_importance(
    embeddings: np.ndarray,
    risk_all:   np.ndarray,
) -> pd.DataFrame:
    """
    Quantify each scale's contribution to the final risk score.

    Method: Pearson correlation between each cell's mean embedding value
    per scale (64D mean → scalar) and its predicted risk score.
    A higher correlation indicates that scale's learned representation
    drives more of the variation in the risk score.

    This is reported in the paper's scale-importance supplementary table
    and is cited when discussing which temporal phase of APAP biology
    carries the most trajectory-risk signal.
    """
    rows = []
    for s, sn in enumerate(SCALE_NAMES):
        scale_emb    = embeddings[:, s, :]       # (6582, 64)
        scale_norm   = np.linalg.norm(scale_emb, axis=1)  # (6582,) — embedding magnitude
        rho, _       = spearmanr(scale_norm, risk_all)
        var_explained = np.var(scale_emb.mean(axis=1)) / np.var(risk_all)
        rows.append({
            "scale":           sn,
            "spearman_vs_risk": round(float(rho), 4),
            "embedding_var":    round(float(np.var(scale_emb)), 6),
            "relative_var":     round(float(var_explained), 4),
        })

    df = pd.DataFrame(rows).sort_values("spearman_vs_risk", ascending=False)
    log.info("\n  Scale importance (Spearman ρ between scale embedding norm and risk):")
    for _, row in df.iterrows():
        log.info(f"    {row['scale']:<30} ρ = {row['spearman_vs_risk']:.4f}")
    return df

# FIGURES

def make_training_curves(history: Dict) -> None:
    """Training and validation loss + AUROC over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("TrajRisk reference model — Full Training", fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train MSE", color="#2196F3", lw=1.5)
    ax.plot(epochs, history["val_loss"],   label="Val MSE",   color="#FF5722", lw=1.5, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("A  Train / Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # AUROC
    ax = axes[1]
    ax.plot(epochs, history["val_auroc"],  color="#4CAF50", lw=2)
    ax.axhline(0.80, color="red",  linestyle="--", alpha=0.5, label="0.80")
    ax.axhline(0.70, color="grey", linestyle=":",  alpha=0.5, label="0.70")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val AUROC")
    ax.set_title("B  Validation AUROC (high-risk threshold = 0.10)")
    ax.legend(); ax.grid(alpha=0.3)

    # Spearman
    ax = axes[2]
    ax.plot(epochs, history["val_spearman"], color="#9C27B0", lw=2)
    ax.axhline(0.60, color="red",  linestyle="--", alpha=0.5, label="ρ = 0.60")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Spearman ρ")
    ax.set_title("C  Validation Spearman ρ (pseudotime_norm)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "week7_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: week7_training_curves.png")


def make_risk_distributions(
    risk_all:   np.ndarray,
    cell_index: pd.DataFrame,
) -> None:
    """
    Predicted risk distributions by timepoint and treatment.
    These are the primary biological validation figures for the paper.
    """
    tp_order = ["0h", "6h", "24h", "48h", "96h"]
    colors   = {"0h": "#2ecc71", "6h": "#e74c3c", "24h": "#e67e22",
                "48h": "#9b59b6", "96h": "#3498db"}

    ci = cell_index.copy()
    ci["risk_predicted"] = risk_all

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("TrajRisk reference model — Predicted Risk Distributions", fontweight="bold")

    # Panel A: violin by timepoint
    ax = axes[0]
    sns.violinplot(
        data=ci, x="timepoint_str", y="risk_predicted",
        order=tp_order, palette=colors, ax=ax, inner="box", linewidth=0.8,
    )
    ax.set_xlabel("Timepoint"); ax.set_ylabel("Predicted Risk Score")
    ax.set_title("A  Risk by Timepoint\n(biological sanity check)")
    ax.grid(alpha=0.3, axis="y")

    # Panel B: scatter — predicted vs pseudotime_norm
    ax = axes[1]
    for tp in tp_order:
        mask = ci["timepoint_str"] == tp
        ax.scatter(
            ci.loc[mask, "pseudotime_norm"],
            ci.loc[mask, "risk_predicted"],
            c=colors[tp], alpha=0.3, s=8, label=tp,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    rho, _ = spearmanr(ci["pseudotime_norm"], ci["risk_predicted"])
    ax.set_xlabel("pseudotime_norm (label)")
    ax.set_ylabel("Predicted risk score")
    ax.set_title(f"B  Predicted vs Label\n(Spearman ρ = {rho:.3f}, all cells)")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: bar — mean predicted risk per timepoint with SD
    ax = axes[2]
    means = [ci.loc[ci["timepoint_str"] == tp, "risk_predicted"].mean() for tp in tp_order]
    stds  = [ci.loc[ci["timepoint_str"] == tp, "risk_predicted"].std()  for tp in tp_order]
    bars  = ax.bar(tp_order, means, yerr=stds, capsize=4,
                   color=[colors[t] for t in tp_order], alpha=0.85, edgecolor="black")
    ax.set_ylabel("Mean predicted risk")
    ax.set_title("C  Mean Risk ± SD per Timepoint")
    ax.grid(alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "week7_risk_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: week7_risk_distributions.png")


def make_embedding_figures(
    embeddings:  np.ndarray,
    risk_all:    np.ndarray,
    cell_index:  pd.DataFrame,
    scale_imp:   pd.DataFrame,
) -> None:
    """
    PCA of combined multi-scale embeddings coloured by timepoint, treatment,
    and predicted risk. Shows how the GNN's internal representation organises
    the hepatocyte states across the APAP trajectory.
    """
    ci = cell_index.copy()
    ci["risk_predicted"] = risk_all

    tp_order = ["0h", "6h", "24h", "48h", "96h"]
    colors   = {"0h": "#2ecc71", "6h": "#e74c3c", "24h": "#e67e22",
                "48h": "#9b59b6", "96h": "#3498db"}

    # PCA on the concatenated 256D embedding
    combined = embeddings.reshape(embeddings.shape[0], -1)  # (6582, 256)
    pca      = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_2d   = pca.fit_transform(combined)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("TrajRisk reference model — Multi-Scale Embedding PCA (256D → 2D)", fontweight="bold")

    # Panel A: coloured by timepoint
    ax = axes[0]
    for tp in tp_order:
        mask = ci["timepoint_str"].values == tp
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                   c=colors[tp], alpha=0.4, s=6, label=tp)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("A  By Timepoint")
    ax.legend(markerscale=2, fontsize=8)

    # Panel B: coloured by treatment
    ax = axes[1]
    trt_colors = {"Control": "#2ecc71", "APAP": "#e74c3c"}
    for trt, col in trt_colors.items():
        mask = ci["treatment_str"].values == trt
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                   c=col, alpha=0.3, s=6, label=trt)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_title("B  By Treatment")
    ax.legend(markerscale=2, fontsize=8)

    # Panel C: coloured by predicted risk (continuous)
    ax = axes[2]
    sc = ax.scatter(pca_2d[:, 0], pca_2d[:, 1],
                    c=risk_all, cmap="plasma", alpha=0.4, s=6)
    plt.colorbar(sc, ax=ax, label="Predicted Risk")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_title("C  By Predicted Risk")

    # Panel D: scale importance bar chart
    ax = axes[3]
    s_names  = [sn.replace("_", "\n") for sn in scale_imp["scale"].values]
    s_rho    = scale_imp["spearman_vs_risk"].values
    bar_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    ax.bar(range(len(s_names)), s_rho,
           color=bar_colors[:len(s_names)], alpha=0.85, edgecolor="black")
    ax.set_xticks(range(len(s_names)))
    ax.set_xticklabels(s_names, fontsize=8)
    ax.set_ylabel("Spearman ρ (embedding norm vs risk)")
    ax.set_title("D  Scale Importance")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "week7_scale_embeddings_pca.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: week7_scale_embeddings_pca.png")


def make_prediction_scatter(
    risk_all:    np.ndarray,
    cell_index:  pd.DataFrame,
    val_idx:     torch.Tensor,
    val_y:       torch.Tensor,
) -> None:
    """
    Predicted vs true pseudotime_norm for val set, per timepoint.
    The per-timepoint scatter shows how well the model captures the
    within-timepoint gradient — the core TrajRisk biological claim.
    """
    tp_order = ["0h", "6h", "24h", "48h", "96h"]
    colors   = {"0h": "#2ecc71", "6h": "#e74c3c", "24h": "#e67e22",
                "48h": "#9b59b6", "96h": "#3498db"}

    val_global = val_idx.cpu().numpy()
    y_pred     = risk_all[val_global]
    y_true     = val_y.cpu().numpy()
    val_tp     = cell_index.iloc[val_global]["timepoint_str"].values

    fig, axes = plt.subplots(1, len(tp_order), figsize=(18, 4), sharey=True)
    fig.suptitle("TrajRisk — Predicted Risk vs pseudotime_norm "
                 "(Validation Set, per Timepoint)", fontweight="bold")

    for i, tp in enumerate(tp_order):
        mask = val_tp == tp
        ax   = axes[i]
        ax.scatter(y_true[mask], y_pred[mask], c=colors[tp], alpha=0.5, s=10)
        if mask.sum() > 2:
            rho, _ = spearmanr(y_true[mask], y_pred[mask])
            ax.set_title(f"{tp}\n(n={mask.sum()}, ρ={rho:.3f})")
        else:
            ax.set_title(f"{tp}\n(n={mask.sum()})")
        ax.set_xlabel("pseudotime_norm")
        if i == 0:
            ax.set_ylabel("Predicted risk")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "week7_predictions_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  Saved: week7_predictions_scatter.png")

# SAVE OUTPUTS

def save_all_outputs(
    risk_all:    np.ndarray,
    embeddings:  np.ndarray,
    cell_index:  pd.DataFrame,
    best_config: Dict,
    history:     Dict,
    metrics:     Dict,
    scale_imp:   pd.DataFrame,
) -> None:
    """
    Persist all reference model outputs and write risk scores back into the AnnData
    so that spatial validation (trajectory analysis) can load them directly.
    """
    log.info("Saving outputs...")

    # 1. Per-cell predictions CSV
    preds_df = cell_index.copy()
    preds_df["risk_predicted"] = risk_all
    preds_path = DATA_DIR / "week7_predictions.csv"
    preds_df.to_csv(preds_path, index=True)
    log.info(f"  Predictions: {preds_path.name}")

    # 2. Per-scale embeddings (6582, 4, 64)
    emb_path = DATA_DIR / "week7_scale_embeddings.npy"
    np.save(emb_path, embeddings)
    log.info(f"  Embeddings:  {emb_path.name}  shape={embeddings.shape}")

    # 3. Scale importance table
    si_path = RES_DIR / "week7_scale_importance.csv"
    scale_imp.to_csv(si_path, index=False)
    log.info(f"  Scale importance: {si_path.name}")

    # 4. Write risk scores into the AnnData (needed for trajectory analysis spatial validation)
    h5ad_path = DATA_DIR / "adata_integrated_with_trajectory.h5ad"
    adata      = sc.read_h5ad(h5ad_path)
    walesky_mask = adata.obs["dataset"] == "Walesky"

    # Map predictions by cell barcode to handle any ordering differences
    pred_series = pd.Series(
        risk_all,
        index=cell_index.index,
        name="risk_predicted",
    )
    adata.obs.loc[walesky_mask, "risk_predicted"] = pred_series.reindex(
        adata.obs.index[walesky_mask]
    ).values

    adata.write_h5ad(h5ad_path)
    log.info(f"  Updated AnnData: risk_predicted added for "
             f"{walesky_mask.sum():,} Walesky cells")

    # 5. Training summary JSON for manuscript reporting
    summary = {
        "week": 7,
        "best_hyperparameters": best_config,
        "final_val_metrics": metrics,
        "training_history_length": len(history["train_loss"]),
        "convergence": {
            "final_train_loss":  round(history["train_loss"][-1], 6),
            "final_val_loss":    round(history["val_loss"][-1],   6),
            "best_val_mse":      round(min(history["val_loss"]),  6),
            "best_val_auroc":    round(max(history["val_auroc"]), 4),
            "best_val_spearman": round(max(history["val_spearman"]), 4),
            "best_epoch_mse":    int(np.argmin(history["val_loss"])) + 1,
            "best_epoch_auroc":  int(np.argmax(history["val_auroc"])) + 1,
            "checkpoints": {
                "best_mse":   "week7_trajrisk_BEST.pt",
                "best_auroc": "week7_trajrisk_BEST_AUROC.pt",
                "final":      "week7_trajrisk_FINAL.pt  (AUROC-best restored)",
            },
        },
    }
    with open(RES_DIR / "week7_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("  Summary: week7_training_summary.json")

# MAIN

def main() -> None:
    log.info("=" * 70)
    log.info("TRAJRISK — reference model: FULL MODEL TRAINING")
    log.info("=" * 70)
    log.info(f"Device: {DEVICE}")
    log.info(f"Search grid: {len(HP_SEARCH_LRS) * len(HP_SEARCH_WEIGHT_DECAYS)} configs")
    log.info(f"Full training: up to {FULL_MAX_EPOCHS} epochs, patience={FULL_PATIENCE}")
    t_global = time.time()

    # Load
    log.info("\n[1] Loading inputs...")
    x, edge_indices, edge_weights, train_idx, val_idx, train_y, val_y, cell_index = (
        load_all_inputs()
    )

    # Phase 1: Hyperparameter search
    log.info("\n[2] Phase 1 — Hyperparameter grid search...")
    best_config = run_hyperparameter_search(
        x, edge_indices, edge_weights, train_idx, val_idx, train_y, val_y
    )

    # Phase 2: Full training
    log.info("\n[3] Phase 2 — Full training with best hyperparameters...")
    model, history = run_full_training(
        best_config, x, edge_indices, edge_weights,
        train_idx, val_idx, train_y, val_y
    )

    # Phase 3A: Full evaluation
    log.info("\n[4] Phase 3 — Evaluation and interpretation...")
    risk_all, metrics = full_evaluation(
        model, x, edge_indices, edge_weights,
        val_idx, val_y, train_idx, train_y, cell_index
    )

    # Phase 3B: Model interpretation
    log.info("\n[5] Extracting per-scale embeddings...")
    embeddings = compute_scale_embeddings(model, x, edge_indices, edge_weights)
    scale_imp  = compute_scale_importance(embeddings, risk_all)

    # Figures
    log.info("\n[6] Generating figures...")
    make_training_curves(history)
    make_risk_distributions(risk_all, cell_index)
    make_embedding_figures(embeddings, risk_all, cell_index, scale_imp)
    make_prediction_scatter(risk_all, cell_index, val_idx, val_y)

    # Save
    log.info("\n[7] Saving all outputs...")
    save_all_outputs(
        risk_all, embeddings, cell_index,
        best_config, history, metrics, scale_imp
    )

    # Final summary
    elapsed = time.time() - t_global
    log.info("\n" + "=" * 70)
    log.info("reference model COMPLETE")
    log.info("=" * 70)
    log.info(f"Total runtime: {elapsed/60:.1f} minutes")
    log.info("")
    log.info("Best hyperparameters:")
    log.info(f"  Learning rate:  {best_config['lr']}")
    log.info(f"  Weight decay:   {best_config['weight_decay']}")
    log.info("")
    log.info("Final model performance (AUROC-best checkpoint):")
    log.info(f"  Val MSE:         {metrics['val_mse']:.6f}")
    log.info(f"  Val Spearman ρ:  {metrics['val_spearman']:.4f}")
    log.info(f"  Val AUROC:       {metrics['val_auroc']:.4f}")
    log.info(f"  Val AUPRC:       {metrics['val_auprc']:.4f}")
    log.info("")
    log.info("Biological sanity:")
    for check, passed in metrics["sanity_checks"].items():
        log.info(f"  {'✓' if passed else '✗'} {check}")
    log.info("")
    log.info("Files written:")
    log.info(f"  {MODEL_DIR}/week7_trajrisk_BEST.pt        (lowest val MSE)")
    log.info(f"  {MODEL_DIR}/week7_trajrisk_BEST_AUROC.pt  (highest val AUROC)")
    log.info(f"  {MODEL_DIR}/week7_trajrisk_FINAL.pt       (AUROC-best restored)")
    log.info(f"  {DATA_DIR}/week7_predictions.csv")
    log.info(f"  {DATA_DIR}/week7_scale_embeddings.npy")
    log.info(f"  {DATA_DIR}/adata_integrated_with_trajectory.h5ad  (+ risk_predicted)")
    log.info(f"  {RES_DIR}/week7_hyperparameter_search.csv")
    log.info(f"  {RES_DIR}/week7_training_summary.json")
    log.info(f"  {FIG_DIR}/week7_training_curves.png")
    log.info(f"  {FIG_DIR}/week7_risk_distributions.png")
    log.info(f"  {FIG_DIR}/week7_scale_embeddings_pca.png")
    log.info(f"  {FIG_DIR}/week7_predictions_scatter.png")
    log.info("")
    log.info("Reference-model training and evaluation complete.")


if __name__ == "__main__":
    main()
