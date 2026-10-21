#!/usr/bin/env python3
"""Fit cross-species coordinates and evaluate human donor-level transfer.

Supervised graph-model training uses Walesky mouse cells in shared coordinates.
W_self-only is the primary human transfer mode; Reference-KNN is a sensitivity
analysis. Human nuclei are summarized over 10 APAP-ALF and 9 healthy donors.
"""

import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

import scanpy as sc
import scvi
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.nn import MessagePassing
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrajRisk-W9B")


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

BASE = Path(__file__).resolve().parents[1]
DATA   = BASE / "Data" / "processed"
MODELS = BASE / "models"
RES    = BASE / "results"
for d in [RES, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED      = 42
N_SIG_GENES      = 50      # genes per signature direction
KNN_K            = 15      # reference neighbours per human cell
LOW_CELL_THRESH  = 300     # patients below this excluded in sensitivity

# GNN — must match Week 7 architecture exactly
NODE_DIM   = 31
HIDDEN_DIM = 64
N_SCALES   = 4
DROPOUT    = 0.3
SCALE_NAMES = [
    "Scale1_AcuteStress",
    "Scale2_PeakInjury",
    "Scale3_FateDecision",
    "Scale4_Resolution",
]

# Cross-species TrajRisk training hyperparameters
# (same as Week 7 best configuration: lr=0.001, wd=1e-4)
CS_LR          = 1e-3
CS_WD          = 1e-4
CS_MAX_EPOCHS  = 200
CS_PATIENCE    = 25
CS_RISK_THRESH = 0.10

# scVI cross-species integration
SCVI_N_LATENT     = 30
SCVI_N_LAYERS     = 2
SCVI_N_HIDDEN     = 128
SCVI_DROPOUT      = 0.1
SCVI_MAX_EPOCHS   = 400
SCVI_ES_PATIENCE  = 20

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
scvi.settings.seed = RANDOM_SEED


# =============================================================================
# 1.  MODEL CLASSES  (exact copies from Week 7)
# =============================================================================

class TrajRiskConv(MessagePassing):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.W_nbr  = nn.Linear(in_ch, out_ch, bias=False)
        self.W_self = nn.Linear(in_ch, out_ch, bias=True)
        nn.init.xavier_uniform_(self.W_nbr.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)

    def forward(self, x, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.W_nbr(agg) + self.W_self(x)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j


class TrajRiskGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_scales    = N_SCALES
        self.scale_convs = nn.ModuleList()
        self.scale_bns   = nn.ModuleList()
        for _ in range(N_SCALES):
            self.scale_convs.append(nn.ModuleList([
                TrajRiskConv(NODE_DIM,   HIDDEN_DIM),
                TrajRiskConv(HIDDEN_DIM, HIDDEN_DIM),
            ]))
            self.scale_bns.append(nn.ModuleList([
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
            ]))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(N_SCALES * HIDDEN_DIM, 128), nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
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
# 2.  LOAD ALL DATA
# =============================================================================

def load_all_data() -> Dict:
    log.info("=" * 68)
    log.info("LOADING AND VERIFYING ALL DATA")
    log.info("=" * 68)

    # ── Human ─────────────────────────────────────────────────────────────────
    human = sc.read_h5ad(str(DATA / "matchett_human_hepatocytes.h5ad"))
    assert human.n_obs == 49_984
    assert {"condition", "sample"}.issubset(human.obs.columns)

    def extract_pid(s):
        m = re.search(r'(apap\d+[AB]?|hea\d+)', s, re.IGNORECASE)
        return m.group(1).lower() if m else s

    human.obs["patient_id"] = (
        human.obs["sample"].apply(extract_pid)
        .str.replace(r'apap09[ab]', 'apap09', regex=True, case=False)
    )

    n_apap    = human.obs.loc[human.obs["condition"]=="APAP_ALF","patient_id"].nunique()
    n_healthy = human.obs.loc[human.obs["condition"]=="Healthy",  "patient_id"].nunique()
    assert n_apap == 10 and n_healthy == 9
    log.info(f"  Human: {human.n_obs:,} cells | APAP={n_apap} | Healthy={n_healthy}")

    cells_per_patient = human.obs.groupby("patient_id").size()
    low_cell = cells_per_patient[
        cells_per_patient.lt(LOW_CELL_THRESH) &
        cells_per_patient.index.str.startswith("apap")
    ].index.tolist()
    log.info(f"  Low-cell patients (< {LOW_CELL_THRESH} cells): {low_cell}")

    # ── Walesky from integrated mouse adata ───────────────────────────────────
    mouse_adata   = sc.read_h5ad(str(DATA / "adata_integrated_with_trajectory.h5ad"))
    adata_walesky = mouse_adata[mouse_adata.obs["dataset"] == "Walesky"].copy()
    assert adata_walesky.n_obs == 6_582
    adata_walesky.obs["species"]    = "mouse"
    adata_walesky.obs["patient_id"] = "walesky"
    log.info(f"  Walesky: {adata_walesky.n_obs:,} cells")

    # ── Walesky cell index (timepoints for KNN scale assignment) ──────────────
    cell_index    = pd.read_csv(DATA / "walesky_cell_index.csv", index_col=0)
    walesky_hours = cell_index["hours"].values.astype(float)
    assert len(walesky_hours) == 6_582

    # ── Transition graphs ─────────────────────────────────────────────────────
    graphs       = torch.load(DATA / "multiscale_graphs.pt", map_location="cpu")
    edge_indices = [graphs[sn]["edge_index"] for sn in SCALE_NAMES]
    edge_weights = [graphs[sn]["edge_weight"] for sn in SCALE_NAMES]
    for s, ei in enumerate(edge_indices):
        assert ei.max().item() <= 6_581

    # ── Week 7 predictions (pseudotime_norm + risk for gene signature) ────────
    preds_w7 = pd.read_csv(DATA / "week7_predictions.csv", index_col=0)
    assert "pseudotime_norm" in preds_w7.columns
    assert set(adata_walesky.obs_names).issubset(set(preds_w7.index)), \
        "Walesky barcodes in adata not all present in week7_predictions.csv"

    # ── Training split ────────────────────────────────────────────────────────
    td = torch.load(DATA / "week6_training_data.pt", map_location="cpu")

    # ── Ortholog table ────────────────────────────────────────────────────────
    orthologs = pd.read_csv(DATA / "mouse_human_orthologs_biomart.csv")

    log.info("  All data verified.\n")

    return {
        "human":           human,
        "adata_walesky":   adata_walesky,
        "mouse_adata":     mouse_adata,
        "cell_index":      cell_index,
        "walesky_hours":   walesky_hours,
        "edge_indices":    edge_indices,
        "edge_weights":    edge_weights,
        "preds_w7":        preds_w7,
        "orthologs":       orthologs,
        "train_idx":       td["train_idx"].numpy(),
        "val_idx":         td["val_idx"].numpy(),
        "train_y":         td["train_y"],   # tensor
        "val_y":           td["val_y"],     # tensor
        "low_cell":        low_cell,
    }


# =============================================================================
# 3.  ORTHOLOG GENE SPACE
# =============================================================================

def prepare_ortholog_gene_space(data: Dict) -> Dict:
    log.info("Preparing strict 1:1 ortholog gene space ...")
    o  = data["orthologs"]
    mc = o["mouse_gene_name"].value_counts()
    hc = o["human_gene_name"].value_counts()
    o1 = o[o["mouse_gene_name"].isin(mc[mc==1].index) &
            o["human_gene_name"].isin(hc[hc==1].index)].copy().reset_index(drop=True)

    wg = set(data["adata_walesky"].var_names)
    hg = set(data["human"].var_names)
    ou = o1[o1["mouse_gene_name"].isin(wg) &
             o1["human_gene_name"].isin(hg)].copy().reset_index(drop=True)

    assert len(ou) > 10_000, f"Too few usable orthologs: {len(ou)}"
    log.info(f"  Strict 1:1 pairs in both adatas: {len(ou):,}\n")

    return {
        "orth_usable":    ou,
        "mouse_to_human": dict(zip(ou["mouse_gene_name"], ou["human_gene_name"])),
        "shared_mouse":   ou["mouse_gene_name"].tolist(),
        "shared_human":   ou["human_gene_name"].tolist(),
    }


# =============================================================================
# 4.  BUILD COMBINED AnnData
# =============================================================================

def _check_and_normalize(adata: sc.AnnData, label: str) -> sc.AnnData:
    """Check normalization state; apply normalize_total + log1p if needed."""
    x10 = adata.X[:10]
    if hasattr(x10, "toarray"):
        x10 = x10.toarray()
    x10 = np.asarray(x10, dtype=np.float32)
    if float(x10.max()) > 50.0:
        log.info(f"  {label}: raw counts detected — normalising (target_sum=1e4, log1p)")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        log.info(f"  {label}: already log-normalised — using as-is")
    return adata


def build_combined_adata(data: Dict, gi: Dict) -> Tuple[sc.AnnData, int]:
    """
    Build joint AnnData for cross-species scVI.
    Walesky cells are placed FIRST (positions 0–6581) to preserve alignment
    with the Walesky transition graph edge indices.
    """
    log.info("Building combined AnnData (Walesky first, then human) ...")

    aw = data["adata_walesky"][:, gi["shared_mouse"]].copy()
    aw.var_names = [gi["mouse_to_human"][g] for g in aw.var_names]
    aw.obs["species"] = "mouse"
    aw.obs["sample"]  = "walesky"
    aw = _check_and_normalize(aw, "Walesky")

    ah = data["human"][:, gi["shared_human"]].copy()
    ah.obs["species"] = "human"
    ah = _check_and_normalize(ah, "Human")

    # ── Fix 1: ensure shared metadata columns survive sc.concat ──────────────
    # sc.concat may drop columns with Categorical dtype conflicts between adatas.
    # Force both columns to plain str before concatenation.
    if "treatment" in aw.obs.columns:
        aw.obs["condition"] = aw.obs["treatment"].map(
            {"Control": "Healthy", "APAP": "APAP"}
        ).astype(str)
    elif "condition" in aw.obs.columns:
        aw.obs["condition"] = aw.obs["condition"].astype(str)
    else:
        aw.obs["condition"] = "unknown"
    aw.obs["patient_type"] = aw.obs["condition"].astype(str)

    ah.obs["condition"] = ah.obs["condition"].astype(str)
    if "patient_type" in ah.obs.columns:
        ah.obs["patient_type"] = ah.obs["patient_type"].astype(str)
    else:
        ah.obs["patient_type"] = ah.obs["condition"].astype(str)

    combined = sc.concat(
        [aw, ah], axis=0, join="inner",
        label="dataset_origin", keys=["walesky","human"], index_unique="-",
    )
    combined.obs_names_make_unique()
    log.info(f"  Combined obs columns: {combined.obs.columns.tolist()}")

    n_wal = aw.n_obs
    assert combined.n_obs == n_wal + ah.n_obs
    assert combined.obs["species"].iloc[:n_wal].eq("mouse").all(), \
        "Walesky cells must occupy positions 0 – n_wal-1"

    log.info(f"  Combined: {combined.n_obs:,} cells × {combined.n_vars:,} genes")
    log.info(f"  Walesky: 0 – {n_wal-1}  |  Human: {n_wal} – {combined.n_obs-1}\n")
    return combined, n_wal


# =============================================================================
# 5.  CROSS-SPECIES scVI
# =============================================================================

def train_cross_species_scvi(combined: sc.AnnData, n_wal: int) -> sc.AnnData:
    combined_path = DATA / "adata_cross_species_combined.h5ad"
    model_path    = MODELS / "week9b_scvi_cross_species"

    if combined_path.exists() and model_path.exists():
        log.info("Loading existing cross-species scVI (checkpoint found) ...")
        combined = sc.read_h5ad(str(combined_path))
        assert "X_scvi_cross" in combined.obsm
        log.info("  Loaded from checkpoint.\n")
        return combined

    log.info("=" * 68)
    log.info("CROSS-SPECIES scVI INTEGRATION")
    log.info("=" * 68)

    try:
        scvi.model.SCVI.setup_anndata(
            combined, layer=None, batch_key="sample",
            categorical_covariate_keys=["species"],
        )
    except TypeError:
        log.warning("  Falling back: categorical_covariate_keys not supported")
        scvi.model.SCVI.setup_anndata(combined, layer=None, batch_key="sample")

    model = scvi.model.SCVI(
        combined, n_latent=SCVI_N_LATENT, n_layers=SCVI_N_LAYERS,
        n_hidden=SCVI_N_HIDDEN, dropout_rate=SCVI_DROPOUT,
        gene_likelihood="normal",
    )

    log.info(f"  Training ({combined.n_obs:,} cells × {combined.n_vars:,} genes) ...")
    log.info("  Expected runtime: 30-60 minutes on CPU.")
    model.train(
        max_epochs=SCVI_MAX_EPOCHS,
        early_stopping=True,
        early_stopping_patience=SCVI_ES_PATIENCE,
        plan_kwargs={"lr": 1e-3},
    )

    combined.obsm["X_scvi_cross"] = model.get_latent_representation()

    # Integration quality
    X = combined.obsm["X_scvi_cross"]
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), min(10_000, len(X)), replace=False)
    sp  = combined.obs["species"].values[idx]
    asw_species = float(silhouette_score(X[idx], sp))
    log.info(f"  ASW (species): {asw_species:.4f}  [target < 0.10]")
    # 'condition' may be dropped by sc.concat if Categorical dtype conflicts occur.
    # Fall back to 'patient_type' if needed (same information, present in human adata).
    cond_col = next(
        (c for c in ["condition", "patient_type"] if c in combined.obs.columns), None
    )
    if cond_col:
        cond = combined.obs[cond_col].fillna("unknown").astype(str).values[idx]
        if len(set(cond)) >= 2:
            asw_cond = float(silhouette_score(X[idx], cond))
            log.info(f"  ASW ({cond_col}): {asw_cond:.4f}  [positive = biology preserved]")
        else:
            asw_cond = float("nan")
            log.info("    ASW (condition): insufficient groups in sample")
    else:
        asw_cond = float("nan")
        log.info("    ASW (condition): column unavailable after concat — skipping")
    combined.uns["integration_quality"] = {
        "asw_species": asw_species, "asw_condition": asw_cond
    }
    if asw_species > 0.30:
        log.warning("  ASW_species > 0.30 — species mixing may be insufficient")

    model.save(str(model_path), overwrite=True)
    combined.write_h5ad(str(combined_path))
    log.info(f"  Saved combined adata and scVI model.\n")
    return combined


# =============================================================================
# 6.  CROSS-SPECIES SEEDER  (new, applied to human cells only)
# =============================================================================

def train_cross_species_seeder(
    combined: sc.AnnData, data: Dict, n_wal: int
) -> XGBClassifier:
    ckpt = MODELS / "week9b_cross_species_seeder.pkl"
    if ckpt.exists():
        log.info("Loading existing cross-species seeder ...")
        return joblib.load(str(ckpt))

    log.info("Training cross-species seeder (Walesky 0h-Control vs 6h-APAP) ...")
    ci  = data["cell_index"]
    hrs = ci["hours"].values.astype(float)
    trt = ci["treatment_str"].values.astype(str)

    mask0 = (hrs == 0.0) & (trt == "Control")
    mask6 = (hrs == 6.0) & (trt == "APAP")

    z_cross = combined.obsm["X_scvi_cross"][:n_wal].astype(np.float32)
    X_tr = np.vstack([z_cross[mask0], z_cross[mask6]])
    y_tr = np.array([0]*mask0.sum() + [1]*mask6.sum())

    spw = float(mask0.sum()) / float(mask6.sum())
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0,
    )
    clf.fit(X_tr, y_tr)

    auroc = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])
    log.info(f"  Training AUROC: {auroc:.4f}  "
             f"(0h: {mask0.sum():,} | 6h: {mask6.sum():,})")
    joblib.dump(clf, str(ckpt))
    log.info("  Saved: week9b_cross_species_seeder.pkl\n")
    return clf


# =============================================================================
# 7.  RETRAIN TrajRisk IN CROSS-SPECIES COORDINATE SYSTEM
# =============================================================================

def retrain_trajrisk_cross_species(
    combined:      sc.AnnData,
    data:          Dict,
    cross_seeder:  XGBClassifier,
    n_wal:         int,
) -> TrajRiskGNN:
    """
    Retrain TrajRiskGNN using Walesky cells' CROSS-SPECIES scVI latent
    coordinates.  This ensures coordinate-system coherence when the same
    model is applied to human cells.

    What stays the same as Week 7:
      - Architecture (TrajRiskGNN, 87,233 parameters)
      - Transition graphs (biological transitions are independent of latent space)
      - Pseudotime_norm labels
      - Train/val split (80/20 by timepoint)
      - Hyperparameters (lr=0.001, wd=1e-4, patience=25)

    What changes:
      - Node features: x = [z_cross_species; r_cross_seeder(z_cross_species)]
        instead of [z_original_mouse; r_original_seeder]

    This new model is saved as week9b_trajrisk_cross_species.pt and is used
    ONLY for Week 9B human validation.  The original Week 7 model is untouched.
    """
    ckpt_path = MODELS / "week9b_trajrisk_cross_species.pt"

    if ckpt_path.exists():
        log.info("Loading existing cross-species TrajRisk checkpoint ...")
        model = TrajRiskGNN()
        ckpt  = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        saved_auroc = ckpt.get("best_val_auroc", float("nan"))
        log.info(f"  Best val AUROC from checkpoint: {saved_auroc}")
        return model, saved_auroc

    log.info("=" * 68)
    log.info("RETRAINING TrajRisk IN CROSS-SPECIES scVI COORDINATE SYSTEM")
    log.info("=" * 68)
    log.info("  Architecture, graphs, labels, and split identical to Week 7.")
    log.info("  Only the node features change (cross-species scVI coordinates).")

    # Build cross-species node features for all 6,582 Walesky cells
    z_cross = combined.obsm["X_scvi_cross"][:n_wal].astype(np.float32)
    r_cross = cross_seeder.predict_proba(z_cross)[:, 1].astype(np.float32)
    x_cross = torch.tensor(
        np.hstack([z_cross, r_cross.reshape(-1, 1)]), dtype=torch.float32
    )   # (6582, 31)

    edge_indices = data["edge_indices"]
    edge_weights = data["edge_weights"]
    train_idx    = torch.tensor(data["train_idx"], dtype=torch.long)
    val_idx      = torch.tensor(data["val_idx"],   dtype=torch.long)
    train_y      = data["train_y"].float()
    val_y        = data["val_y"].float()

    y_val_bin = (val_y.numpy() > CS_RISK_THRESH).astype(int)

    model     = TrajRiskGNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CS_LR, weight_decay=CS_WD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_auroc   = 0.0
    best_state       = None
    patience_counter = 0

    log.info(f"  Training up to {CS_MAX_EPOCHS} epochs (patience={CS_PATIENCE}) ...")

    for epoch in range(CS_MAX_EPOCHS):
        # ── Train step ────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        pred_all = model(x_cross, edge_indices, edge_weights)
        loss     = criterion(pred_all[train_idx], train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_all_val = model(x_cross, edge_indices, edge_weights)
        pred_val   = pred_all_val[val_idx].numpy()
        val_loss   = float(criterion(pred_all_val[val_idx], val_y).item())

        try:
            val_auroc = float(roc_auc_score(y_val_bin, pred_val))
        except Exception:
            val_auroc = 0.0

        scheduler.step(val_loss)

        if val_auroc > best_val_auroc:
            best_val_auroc   = val_auroc
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            log.info(f"    Epoch {epoch+1:3d} | val_loss={val_loss:.6f} | "
                     f"val_auroc={val_auroc:.4f} | best={best_val_auroc:.4f}")

        if patience_counter >= CS_PATIENCE:
            log.info(f"  Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Sanity check: val AUROC should be close to the Week 7 result (~0.90)
    log.info(f"\n  Cross-species TrajRisk best val AUROC: {best_val_auroc:.4f}")
    if best_val_auroc < 0.80:
        log.warning("  Val AUROC < 0.80 — cross-species scVI alignment may be poor.")

    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_auroc":   best_val_auroc,
        "epoch_stopped":    epoch + 1,
    }, str(ckpt_path))
    log.info("  Saved: week9b_trajrisk_cross_species.pt\n")
    return model, best_val_auroc


# =============================================================================
# 8.  GENE SIGNATURE  (from ALL Walesky genes, then ortholog-mapped)
# =============================================================================

def build_gene_signatures(data: Dict, gi: Dict) -> Dict:
    log.info("Building gene signatures from ALL Walesky genes ...")

    adata_wal_full = data["mouse_adata"][
        data["mouse_adata"].obs["dataset"] == "Walesky"
    ]

    # Barcode alignment check (Bug 5 fix)
    assert set(adata_wal_full.obs_names).issubset(set(data["preds_w7"].index)), \
        "Walesky barcodes not all present in week7_predictions.csv"

    pseudotime_norm = data["preds_w7"].loc[
        adata_wal_full.obs_names, "pseudotime_norm"
    ].values

    log.info(f"  Computing Spearman ρ across {adata_wal_full.n_vars:,} Walesky genes ...")

    X = adata_wal_full.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    pt_rank = rankdata(pseudotime_norm)
    X_rank  = np.apply_along_axis(rankdata, 0, X)
    pt_c    = pt_rank - pt_rank.mean()
    X_c     = X_rank - X_rank.mean(axis=0)
    numer   = pt_c @ X_c
    denom   = np.sqrt((pt_c**2).sum()) * np.sqrt((X_c**2).sum(axis=0)) + 1e-12
    rho     = numer / denom
    gene_rho = pd.Series(rho, index=adata_wal_full.var_names)

    log.info(f"  ρ range: [{gene_rho.min():.4f}, {gene_rho.max():.4f}]")

    # Map to human orthologs; stop once N_SIG_GENES mapped genes found
    m2h       = gi["mouse_to_human"]
    hum_genes = set(data["human"].var_names)

    def map_top_genes(ranked_series: pd.Series) -> Tuple[List[str], List[str]]:
        """Return (mouse_genes_used, human_genes_used) — actual mapped set."""
        mouse_mapped = []
        human_mapped = []
        for g in ranked_series.index:
            hg = m2h.get(g)
            if hg and hg in hum_genes:
                mouse_mapped.append(g)
                human_mapped.append(hg)
            if len(human_mapped) == N_SIG_GENES:
                break
        return mouse_mapped, human_mapped

    risk_mouse, risk_human     = map_top_genes(gene_rho.sort_values(ascending=False))
    recovery_mouse, recovery_human = map_top_genes(gene_rho.sort_values(ascending=True))

    log.info(f"  Risk signature:     {len(risk_human)} human genes")
    log.info(f"  Recovery signature: {len(recovery_human)} human genes")

    # Score human cells using NORMALIZED expression
    # Bug 3 fix: normalize if needed before scoring
    Xh = data["human"].X
    if hasattr(Xh, "toarray"):
        Xh = Xh.toarray()
    Xh = np.asarray(Xh, dtype=np.float32)
    # Check if normalised
    if float(Xh[:100].max()) > 50.0:
        log.info("  Human .X appears raw — normalising for signature scoring")
        import scipy.sparse as sp_sparse
        # Normalise on a copy without modifying the original adata
        row_sums = np.asarray(Xh.sum(axis=1)).flatten()
        scale    = 1e4 / np.where(row_sums == 0, 1, row_sums)
        Xh_norm  = Xh * scale[:, np.newaxis]
        Xh_norm  = np.log1p(Xh_norm)
    else:
        Xh_norm = Xh

    hum_var_list = list(data["human"].var_names)

    def score(gene_list):
        idxs = [hum_var_list.index(g) for g in gene_list if g in hum_var_list]
        return Xh_norm[:, idxs].mean(axis=1).astype(np.float32) if idxs \
               else np.zeros(Xh_norm.shape[0], dtype=np.float32)

    risk_score     = score(risk_human)
    recovery_score = score(recovery_human)
    net_score      = risk_score - recovery_score

    # Bug 4 fix: save the ACTUAL mapped genes used for scoring
    sig_df = pd.concat([
        pd.DataFrame({
            "mouse_gene": risk_mouse,
            "human_gene": risk_human,
            "spearman_rho": gene_rho[risk_mouse].values,
            "signature": "risk",
        }),
        pd.DataFrame({
            "mouse_gene": recovery_mouse,
            "human_gene": recovery_human,
            "spearman_rho": gene_rho[recovery_mouse].values,
            "signature": "recovery",
        }),
    ], ignore_index=True)
    sig_df.to_csv(RES / "week9b_gene_signature.csv", index=False)
    log.info("  Saved: week9b_gene_signature.csv  (actual mapped genes used)\n")

    return {
        "risk_score":     risk_score,
        "recovery_score": recovery_score,
        "net_score":      net_score,
        "risk_genes":     risk_human,
        "recovery_genes": recovery_human,
        "gene_rho":       gene_rho,
    }


# =============================================================================
# 9.  COMPUTE HUMAN RISK SCORES
# =============================================================================

def _gaussian_knn_edges(
    z_query: np.ndarray,    # (n_query, 30) human cells
    z_ref:   np.ndarray,    # (n_ref,   30) Walesky cells
    n_ref:   int,           # n_wal = 6582 (offset for tgt indices)
    k:       int = KNN_K,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Find k nearest Walesky cells for each query (human) cell.
    Returns local_indices, edge_index tensor, edge_weight tensor.
    """
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(z_ref)
    dists, local_idx = knn.kneighbors(z_query)   # (n_query, k)

    nz    = dists[dists > 0]
    sigma = float(np.median(nz)) if len(nz) else 1.0
    wgts  = np.exp(-dists**2 / (2*sigma**2)).astype(np.float32)
    wgts  = (wgts / wgts.sum(axis=1, keepdims=True).clip(1e-12)).astype(np.float32)

    n_query = z_query.shape[0]
    src = local_idx.flatten().astype(np.int64)                          # Walesky positions
    tgt = (np.repeat(np.arange(n_query), k) + n_ref).astype(np.int64)  # human positions

    ei = torch.tensor(np.stack([src, tgt]), dtype=torch.long)
    ew = torch.tensor(wgts.flatten(), dtype=torch.float32)
    return local_idx, ei, ew


def _assign_scale(neighbor_hours: np.ndarray) -> int:
    """
    Assign a human cell to the most frequent phase among its Walesky
    reference neighbours: 0h/6h -> Scale 0, 24h -> Scale 1,
    48h -> Scale 2, and later times -> Scale 3 (zero-based indices).
    Counter.most_common resolves equal phase counts by the first phase
    encountered in the returned neighbour order.
    """
    def tp_to_phase(h: float) -> int:
        if h <= 0.0:  return 0
        if h <= 6.0:  return 0   # 6h → boundary → Scale 0
        if h <= 24.0: return 1
        if h <= 48.0: return 2
        return 3

    phases = [tp_to_phase(h) for h in neighbor_hours]
    from collections import Counter
    return Counter(phases).most_common(1)[0][0]


def compute_human_risk_scores(
    combined:     sc.AnnData,
    data:         Dict,
    cs_trajrisk:  TrajRiskGNN,    # cross-species TrajRisk (this version's model)
    cross_seeder: XGBClassifier,
    n_wal:        int,
) -> Dict:
    """
    Compute risk scores for 49,984 human cells.

    Layer 2A (W_self-only) — PRIMARY
    ---------------------------------
    Both Walesky and human cells use cross-species scVI coordinates.
    Node tensor: x_all = [x_walesky_cross ; x_human_cross]  (56566 × 31)
    Human cells at positions n_wal+ receive no reference edges.
    h_i = W_self(x_i)  (same pathway for all cells with no incoming edges)

    Layer 2B (scale-specific KNN) — SENSITIVITY
    --------------------------------------------
    Each human cell connects to k=15 nearest Walesky cells in the
    cross-species scVI space.  Reference edges are added to the
    SINGLE scale that best matches the majority neighbour timepoint.
    This avoids injecting the same edges into all 4 scale branches.
    """
    log.info("Computing human risk scores ...")

    edge_indices  = data["edge_indices"]
    edge_weights  = data["edge_weights"]
    walesky_hours = data["walesky_hours"]

    z_all   = combined.obsm["X_scvi_cross"]
    z_wal   = z_all[:n_wal].astype(np.float32)    # (6582, 30)
    z_hum   = z_all[n_wal:].astype(np.float32)    # (n_human, 30)
    n_human = z_hum.shape[0]

    # Cross-species seeder risk for all cells
    r_wal = cross_seeder.predict_proba(z_wal)[:, 1].astype(np.float32)
    r_hum = cross_seeder.predict_proba(z_hum)[:, 1].astype(np.float32)

    # Build node feature tensors — SAME coordinate system for both species
    x_wal = torch.tensor(np.hstack([z_wal, r_wal.reshape(-1,1)]), dtype=torch.float32)
    x_hum = torch.tensor(np.hstack([z_hum, r_hum.reshape(-1,1)]), dtype=torch.float32)
    x_all = torch.cat([x_wal, x_hum], dim=0)   # (n_wal + n_human, 31)

    # ── Layer 2A: W_self-only (PRIMARY) ───────────────────────────────────────
    log.info("  Layer 2A: W_self-only (PRIMARY) ...")
    with torch.no_grad():
        cs_trajrisk.eval()
        risk_2a_all = cs_trajrisk(x_all, edge_indices, edge_weights).numpy()
    risk_2a = risk_2a_all[n_wal:].astype(np.float32)
    log.info(f"    Range: [{risk_2a.min():.4f}, {risk_2a.max():.4f}]")

    # ── KNN edges ─────────────────────────────────────────────────────────────
    log.info(f"  Building KNN edges (k={KNN_K}, cross-species scVI space) ...")
    local_idx, ref_ei, ref_ew = _gaussian_knn_edges(z_hum, z_wal, n_wal, k=KNN_K)

    # Neighbour timepoint composition (biological interpretation)
    nbr_hours = walesky_hours[local_idx]   # (n_human, k)
    hour_vals  = [0.0, 6.0, 24.0, 48.0, 96.0]
    hour_names = ["0h", "6h", "24h", "48h", "96h"]
    composition = {f"pct_{hn}": (nbr_hours==hv).mean(axis=1) for hv,hn in zip(hour_vals,hour_names)}

    is_alf = (data["human"].obs["condition"] == "APAP_ALF").values
    log.info("  Neighbour composition (ALF | Healthy):")
    for key, vals in composition.items():
        log.info(f"    {key:10s}: ALF={vals[is_alf].mean():.3f}  "
                 f"Healthy={vals[~is_alf].mean():.3f}")

    # ── Layer 2B: Scale-specific KNN (SENSITIVITY) ────────────────────────────
    log.info("  Layer 2B: scale-specific KNN (SENSITIVITY) ...")
    scale_src  = [[] for _ in range(N_SCALES)]
    scale_tgt  = [[] for _ in range(N_SCALES)]
    scale_wgt  = [[] for _ in range(N_SCALES)]

    ref_ew_np    = ref_ew.numpy().reshape(n_human, KNN_K)
    local_idx_np = local_idx  # (n_human, k)

    for i in range(n_human):
        s = _assign_scale(nbr_hours[i])
        src_i = local_idx_np[i].astype(np.int64)
        tgt_i = np.full(KNN_K, n_wal + i, dtype=np.int64)
        wgt_i = ref_ew_np[i]
        scale_src[s].extend(src_i.tolist())
        scale_tgt[s].extend(tgt_i.tolist())
        scale_wgt[s].extend(wgt_i.tolist())

    aug_indices = []
    aug_weights = []
    for s in range(N_SCALES):
        if len(scale_src[s]) == 0:
            aug_indices.append(edge_indices[s])
            aug_weights.append(edge_weights[s])
        else:
            es = torch.tensor(np.stack([scale_src[s], scale_tgt[s]]), dtype=torch.long)
            ew = torch.tensor(np.array(scale_wgt[s], dtype=np.float32), dtype=torch.float32)
            aug_indices.append(torch.cat([edge_indices[s], es], dim=1))
            aug_weights.append(torch.cat([edge_weights[s], ew]))
        log.info(f"    Scale {s+1}: {len(scale_src[s]):,} reference edges added")

    with torch.no_grad():
        cs_trajrisk.eval()
        risk_2b_all = cs_trajrisk(x_all, aug_indices, aug_weights).numpy()
    risk_2b = risk_2b_all[n_wal:].astype(np.float32)
    log.info(f"  Layer 2B range: [{risk_2b.min():.4f}, {risk_2b.max():.4f}]\n")

    return {
        "risk_2a":       risk_2a,
        "risk_2b":       risk_2b,
        "seeder_risk":   r_hum,
        "composition":   composition,
    }


# =============================================================================
# 10.  DONOR-LEVEL ANALYSIS (PRIMARY + SENSITIVITY)
# =============================================================================

def donor_analysis(
    human:    sc.AnnData,
    risks:    Dict,
    sigs:     Dict,
    label:    str,
    exclude:  List[str] = None,
) -> Tuple[Dict, pd.DataFrame]:

    exclude = exclude or []
    obs = human.obs.copy()
    obs["risk_2a"]   = risks["risk_2a"]
    obs["risk_2b"]   = risks["risk_2b"]
    obs["sig_net"]   = sigs["net_score"]
    obs["sig_risk"]  = sigs["risk_score"]
    obs["sig_recov"] = sigs["recovery_score"]

    if exclude:
        obs = obs[~obs["patient_id"].isin(exclude)]

    donor = obs.groupby("patient_id").agg(
        condition=("condition", "first"),
        n_cells  =("condition", "count"),
        risk_2a  =("risk_2a",   "mean"),
        risk_2b  =("risk_2b",   "mean"),
        sig_net  =("sig_net",   "mean"),
        sig_risk =("sig_risk",  "mean"),
    ).reset_index()

    alf = donor[donor["condition"] == "APAP_ALF"]
    hlt = donor[donor["condition"] == "Healthy"]

    log.info(f"\n  [{label}] n_ALF={len(alf)} | n_Healthy={len(hlt)}")
    res = {"label": label, "n_alf": len(alf), "n_healthy": len(hlt)}

    for col, name in [
        ("risk_2a",  "Layer 2A W_self (PRIMARY)"),
        ("risk_2b",  "Layer 2B KNN (SENSITIVITY)"),
        ("sig_net",  "Layer 3 net signature"),
        ("sig_risk", "Layer 3 risk signature"),
    ]:
        a, h = alf[col].values.astype(float), hlt[col].values.astype(float)
        if len(a) < 2 or len(h) < 2:
            continue
        _, pval  = mannwhitneyu(a, h, alternative="greater")
        y_bin    = np.array([1]*len(a) + [0]*len(h))
        try:
            auroc = float(roc_auc_score(y_bin, np.concatenate([a,h])))
        except Exception:
            auroc = float("nan")
        log.info(f"    {name:<38} "
                 f"ALF={a.mean():.4f}  Healthy={h.mean():.4f}  "
                 f"MW_p={pval:.3e}  AUROC={auroc:.3f}")
        res[f"{col}_alf_mean"]     = float(a.mean())
        res[f"{col}_healthy_mean"] = float(h.mean())
        res[f"{col}_mw_pval"]      = float(pval)
        res[f"{col}_auroc"]        = auroc

    rho, p = spearmanr(donor["risk_2b"], donor["sig_net"])
    log.info(f"    Convergence 2B↔sig (donor-level): ρ={rho:+.4f}  p={p:.3e}")
    res["convergence_rho"] = float(rho)
    res["convergence_p"]   = float(p)
    return res, donor


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t0 = time.time()
    log.info("=" * 68)
    log.info("TRAJRISK — WEEK 9B: HUMAN CROSS-SPECIES VALIDATION")
    log.info("=" * 68)

    data     = load_all_data()
    gi       = prepare_ortholog_gene_space(data)
    combined, n_wal = build_combined_adata(data, gi)
    combined = train_cross_species_scvi(combined, n_wal)
    seeder   = train_cross_species_seeder(combined, data, n_wal)
    cs_model, cs_val_auroc = retrain_trajrisk_cross_species(combined, data, seeder, n_wal)
    sigs     = build_gene_signatures(data, gi)
    risks    = compute_human_risk_scores(combined, data, cs_model, seeder, n_wal)

    log.info("\n" + "=" * 68)
    log.info("PRIMARY ANALYSIS — DONOR LEVEL (10 ALF vs 9 Healthy)")
    log.info("=" * 68)
    primary_res, donor_primary = donor_analysis(
        data["human"], risks, sigs, label="primary"
    )

    log.info("\n" + "=" * 68)
    log.info(f"SENSITIVITY — Excluding {data['low_cell']} (< {LOW_CELL_THRESH} cells)")
    log.info("=" * 68)
    sens_res, donor_sens = donor_analysis(
        data["human"], risks, sigs, label="sensitivity",
        exclude=data["low_cell"],
    )

    # Cell-level supplement
    log.info("\n[SUPPLEMENT] Cell-level statistics (NOT independent observations)")
    y_bin = (data["human"].obs["condition"] == "APAP_ALF").values.astype(int)
    for name, scores in [
        ("Layer 2A W_self", risks["risk_2a"]),
        ("Layer 2B KNN",    risks["risk_2b"]),
        ("Layer 3 net sig", sigs["net_score"]),
    ]:
        auroc = float(roc_auc_score(y_bin, scores))
        _, pv = mannwhitneyu(scores[y_bin==1], scores[y_bin==0], alternative="greater")
        log.info(f"  {name:<22} cell-AUROC={auroc:.4f}  MW_p={pv:.2e}")

    # Save
    log.info("\nSaving outputs ...")
    cell_df = data["human"].obs[["sample","condition","patient_id"]].copy()
    cell_df["risk_2a"]  = risks["risk_2a"]
    cell_df["risk_2b"]  = risks["risk_2b"]
    cell_df["sig_net"]  = sigs["net_score"]
    cell_df.to_csv(RES / "week9b_cell_risk_scores.csv")
    donor_primary.to_csv(RES / "week9b_donor_primary.csv", index=False)
    donor_sens.to_csv(RES / "week9b_donor_sensitivity.csv", index=False)

    summary = {
        "script": "validate_human_transfer.py",
        "n_orthologs": len(gi["shared_mouse"]),
        "n_wal": n_wal, "n_human": combined.n_obs - n_wal,
        "cs_trajrisk_val_auroc": cs_val_auroc,
        "integration_quality": combined.uns.get("integration_quality", {}),
        "primary": primary_res, "sensitivity": sens_res,
        "runtime_minutes": round((time.time()-t0)/60, 2),
    }
    with open(RES / "week9b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    for fn in ["week9b_cell_risk_scores.csv","week9b_donor_primary.csv",
               "week9b_donor_sensitivity.csv","week9b_summary.json",
               "week9b_gene_signature.csv"]:
        log.info(f"  {fn}")

    log.info(f"\nWeek 9B complete in {(time.time()-t0)/60:.1f} minutes.")


if __name__ == "__main__":
    main()
