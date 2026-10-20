#!/usr/bin/env python3
"""
TrajRisk — Missing Figure Asset Extraction
==========================================
Creates two files required for manuscript figures but absent from saved outputs.
NO models are retrained. ALL values computed from saved checkpoints only.

Output files:
  results/manuscript_mouse_cell_scores.csv
    cell_id, dataset, treatment, timepoint_hours,
    trajrisk_knn, trajrisk_wself, seeder, cdkn1a_expression

  results/manuscript_human_composition.csv
    patient_id, condition, n_cells,
    pct_0h, pct_6h, pct_24h, pct_48h, pct_96h

Usage:
    conda activate trajrisk
    cd /path/to/TrajRisk
    python scripts/extract_figure_data.py
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import MessagePassing

import scanpy as sc

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Extract")

BASE = Path(__file__).resolve().parents[2]
DATA    = BASE / "Data" / "processed"
MODELS  = BASE / "models"
RES     = BASE / "results"

SCALE_NAMES = [
    "Scale1_AcuteStress", "Scale2_PeakInjury",
    "Scale3_FateDecision", "Scale4_Resolution"
]
KNN_K = 15

# =============================================================================
# 0. FILE EXISTENCE CHECKS — stop immediately if anything is missing
# =============================================================================
log.info("=" * 68)
log.info("STEP 0: File existence checks")
log.info("=" * 68)

REQUIRED = {
    "Integrated mouse adata":
        DATA / "adata_integrated_with_trajectory.h5ad",
    "Walesky cell index":
        DATA / "walesky_cell_index.csv",
    "Walesky node features":
        DATA / "week6_node_features.pt",
    "Multi-scale transition graphs":
        DATA / "multiscale_graphs.pt",
    "TrajRisk model (FINAL)":
        MODELS / "week7_trajrisk_FINAL.pt",
    "Cross-species combined adata":
        DATA / "adata_cross_species_combined.h5ad",
    "Human cell risk scores":
        RES / "week9b_cell_risk_scores.csv",
}

# Seeder: try multiple possible filenames
SEEDER_CANDIDATES = [
    MODELS / "week6_xgboost_model.pkl",
    MODELS / "week6_seeder.pkl",
    MODELS / "week6_xgboost_seeder.pkl",
]
seeder_path = next((p for p in SEEDER_CANDIDATES if p.exists()), None)
if seeder_path:
    REQUIRED["Week 6 seeder"] = seeder_path
else:
    REQUIRED["Week 6 seeder (MISSING — checked multiple names)"] = None

# B1 XGBoost: try multiple possible filenames
B1_CANDIDATES = [
    MODELS / "week8_b1_xgboost.pkl",
    MODELS / "week8_b1_regressor.pkl",
    MODELS / "b1_xgboost_regressor.pkl",
    MODELS / "week8_xgboost_b1.pkl",
    MODELS / "b1_regressor.pkl",
]
b1_path = next((p for p in B1_CANDIDATES if p.exists()), None)
# B1 is optional for the panels we need — note status but don't abort
if b1_path:
    log.info(f"  B1 XGBoost found at: {b1_path.name}")
else:
    log.warning("  B1 XGBoost model not found (tried 5 filenames). "
                "b1_xgb column will be omitted from mouse cell scores CSV.")

all_ok = True
for label, path in REQUIRED.items():
    if path is None or not path.exists():
        log.error(f"  MISSING: {label}")
        all_ok = False
    else:
        log.info(f"  OK:  {path.name}")

if not all_ok:
    log.error("One or more required files are missing. Stopping.")
    sys.exit(1)

log.info("  All required files present.\n")


# =============================================================================
# 1. MODEL CLASSES — exact copies from week9a / week7
# =============================================================================
class TrajRiskConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr="add", flow="source_to_target")
        self.W_nbr  = nn.Linear(in_ch, out_ch, bias=False)
        self.W_self = nn.Linear(in_ch, out_ch, bias=True)
        nn.init.xavier_uniform_(self.W_nbr.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
    def forward(self, x, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.W_nbr(agg) + self.W_self(x)
    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j

class TrajRiskGNN(nn.Module):
    def __init__(self, node_dim=31, hidden=64, n_scales=4, dropout=0.3):
        super().__init__()
        self.n_scales    = n_scales
        self.scale_convs = nn.ModuleList()
        self.scale_bns   = nn.ModuleList()
        for _ in range(n_scales):
            self.scale_convs.append(nn.ModuleList([
                TrajRiskConv(node_dim, hidden),
                TrajRiskConv(hidden,   hidden),
            ]))
            self.scale_bns.append(nn.ModuleList([
                nn.BatchNorm1d(hidden),
                nn.BatchNorm1d(hidden),
            ]))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(n_scales * hidden, 128), nn.ReLU(),
            nn.Dropout(p=dropout),
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
# 2. LOAD COMMON RESOURCES
# =============================================================================
log.info("=" * 68)
log.info("STEP 1: Loading common resources")
log.info("=" * 68)

mouse_adata = sc.read_h5ad(str(DATA / "adata_integrated_with_trajectory.h5ad"))
log.info(f"  Integrated adata: {mouse_adata.n_obs:,} cells × {mouse_adata.n_vars:,} genes")
log.info(f"  Datasets: {mouse_adata.obs['dataset'].value_counts().to_dict()}")

cell_index = pd.read_csv(DATA / "walesky_cell_index.csv", index_col=0)
walesky_hours = cell_index["hours"].values.astype(float)

x_walesky = torch.load(DATA / "week6_node_features.pt", map_location="cpu")
assert x_walesky.shape == (6582, 31), f"Unexpected shape: {x_walesky.shape}"
log.info(f"  x_walesky: {tuple(x_walesky.shape)}")

graphs       = torch.load(DATA / "multiscale_graphs.pt", map_location="cpu")
edge_indices = [graphs[sn]["edge_index"] for sn in SCALE_NAMES]
edge_weights = [graphs[sn]["edge_weight"] for sn in SCALE_NAMES]

trajrisk_model = TrajRiskGNN()
ckpt_path = MODELS / "week7_trajrisk_FINAL.pt"
ckpt = torch.load(str(ckpt_path), map_location="cpu")
trajrisk_model.load_state_dict(ckpt["model_state_dict"])
trajrisk_model.eval()
log.info(f"  TrajRisk model loaded: {ckpt_path.name}")

seeder = joblib.load(str(seeder_path))
log.info(f"  Seeder loaded: {seeder_path.name}")

b1_model = joblib.load(str(b1_path)) if b1_path else None
if b1_model:
    log.info(f"  B1 XGBoost loaded: {b1_path.name}")

log.info("")


# =============================================================================
# 3. HELPER: score external cells
# =============================================================================

def score_external_cells(z_ext, n_wal, edge_indices, edge_weights,
                          walesky_hours_arr, cell_timepoints_arr,
                          cell_treatments_arr, scale_map):
    """
    Score external cells using W_self-only and reference-KNN.

    Parameters
    ----------
    z_ext             : (n_ext, 30) scVI features of external cells
    n_wal             : 6582
    edge_indices, edge_weights : Walesky transition graphs
    walesky_hours_arr : (6582,) hours for each Walesky cell
    cell_timepoints_arr : (n_ext,) timepoint_hours per external cell
    cell_treatments_arr : (n_ext,) str treatment per external cell (APAP/Control)
    scale_map         : dict mapping (treatment, timepoint) → (scale_idx, anchor_hours)

    Returns
    -------
    wself (n_ext,), knn (n_ext,), seeder_risk (n_ext,)
    """
    n_ext = z_ext.shape[0]

    # Seeder risk
    r_ext = seeder.predict_proba(z_ext.astype(np.float32))[:, 1].astype(np.float32)

    # Node features
    x_ext = np.hstack([z_ext, r_ext.reshape(-1, 1)]).astype(np.float32)
    x_all = torch.tensor(
        np.vstack([x_walesky.numpy(), x_ext]), dtype=torch.float32
    )   # (6582 + n_ext, 31)

    # ── W_self (no reference edges) ──────────────────────────────────────────
    with torch.no_grad():
        risk_wself_all = trajrisk_model(x_all, edge_indices, edge_weights).numpy()
    risk_wself = risk_wself_all[n_wal:].astype(np.float32)

    # ── Reference-KNN edges (cell-specific, scale-specific) ──────────────────
    # Build per-scale source/target/weight lists
    scale_src = [[] for _ in range(4)]
    scale_tgt = [[] for _ in range(4)]
    scale_wgt = [[] for _ in range(4)]

    for s_idx in range(4):
        # Identify which external cells belong to this scale
        cell_mask = np.zeros(n_ext, dtype=bool)
        anchor_mask = np.zeros(6582, dtype=bool)

        for i, (tp, trt) in enumerate(zip(cell_timepoints_arr, cell_treatments_arr)):
            key = (str(trt), float(tp))
            if key in scale_map and scale_map[key][0] == s_idx:
                cell_mask[i] = True
                anch_hours = scale_map[key][1]
                for ah in anch_hours:
                    anchor_mask |= (walesky_hours_arr == ah)

        if not cell_mask.any():
            continue

        z_query  = z_ext[cell_mask]
        z_anchor = mouse_adata[
            mouse_adata.obs["dataset"] == "Walesky"
        ].obsm["X_scvi"][anchor_mask]
        anchor_positions = np.where(anchor_mask)[0]

        knn = NearestNeighbors(n_neighbors=min(KNN_K, len(anchor_positions)),
                               metric="euclidean", n_jobs=-1)
        knn.fit(z_anchor)
        dists, local_idx = knn.kneighbors(z_query)

        nz    = dists[dists > 0]
        sigma = float(np.median(nz)) if len(nz) else 1.0
        wgts  = np.exp(-dists**2 / (2 * sigma**2))
        wgts  = (wgts / wgts.sum(axis=1, keepdims=True).clip(1e-12)).astype(np.float32)

        cell_positions = np.where(cell_mask)[0]
        for ci, pos in enumerate(cell_positions):
            for ki in range(local_idx.shape[1]):
                src = int(anchor_positions[local_idx[ci, ki]])
                tgt = int(n_wal + pos)
                wgt = float(wgts[ci, ki])
                scale_src[s_idx].append(src)
                scale_tgt[s_idx].append(tgt)
                scale_wgt[s_idx].append(wgt)

    # Augment edge lists
    aug_indices = []
    aug_weights = []
    for s in range(4):
        if scale_src[s]:
            es = torch.tensor(np.stack([scale_src[s], scale_tgt[s]]), dtype=torch.long)
            ew = torch.tensor(np.array(scale_wgt[s], dtype=np.float32))
            aug_indices.append(torch.cat([edge_indices[s], es], dim=1))
            aug_weights.append(torch.cat([edge_weights[s], ew]))
        else:
            aug_indices.append(edge_indices[s])
            aug_weights.append(edge_weights[s])

    with torch.no_grad():
        risk_knn_all = trajrisk_model(x_all, aug_indices, aug_weights).numpy()
    risk_knn = risk_knn_all[n_wal:].astype(np.float32)

    return risk_wself, risk_knn, r_ext


# =============================================================================
# 4. EXTRACT UMBAUGH 2H CELL-LEVEL SCORES
# =============================================================================
log.info("=" * 68)
log.info("STEP 2: Umbaugh 2h cell-level scores")
log.info("=" * 68)

u2h_mask = mouse_adata.obs["dataset"] == "Umbaugh_2h"
u2h_adata = mouse_adata[u2h_mask]
log.info(f"  Umbaugh_2h: {u2h_adata.n_obs:,} cells")
log.info(f"  Treatment values: {u2h_adata.obs['treatment'].value_counts().to_dict()}")
log.info(f"  Timepoints: {sorted(u2h_adata.obs['timepoint_hours'].unique().tolist())}")

z_u2h = u2h_adata.obsm["X_scvi"].astype(np.float32)
trt_u2h = u2h_adata.obs["treatment"].values.astype(str)
tp_u2h  = u2h_adata.obs["timepoint_hours"].values.astype(float)

# Scale map for Umbaugh 2h:
# Control (0h) → Scale 1, anchors: Walesky 0h controls
# APAP (2h)    → Scale 1, anchors: Walesky 0h + 6h
scale_map_u2h = {
    ("Control", 0.0): (0, [0.0]),
    ("APAP",    2.0): (0, [0.0, 6.0]),
}

wself_u2h, knn_u2h, seed_u2h = score_external_cells(
    z_u2h, 6582, edge_indices, edge_weights,
    walesky_hours, tp_u2h, trt_u2h, scale_map_u2h
)
log.info(f"  W_self range: [{wself_u2h.min():.4f}, {wself_u2h.max():.4f}]")
log.info(f"  KNN range:    [{knn_u2h.min():.4f},   {knn_u2h.max():.4f}]")

df_u2h = pd.DataFrame({
    "cell_id":          u2h_adata.obs_names.tolist(),
    "dataset":          "Umbaugh_2h",
    "treatment":        trt_u2h,
    "timepoint_hours":  tp_u2h,
    "trajrisk_wself":   wself_u2h,
    "trajrisk_knn":     knn_u2h,
    "seeder":           seed_u2h,
    "cdkn1a_expression": np.nan,   # Cdkn1a not relevant for 2h; filled later for 48h/72h
})

if b1_model:
    df_u2h["b1_xgb"] = b1_model.predict(z_u2h).astype(np.float32)
    log.info("  B1 XGBoost scores computed.")
else:
    df_u2h["b1_xgb"] = np.nan

log.info(f"  Umbaugh 2h records: {len(df_u2h):,}")


# =============================================================================
# 5. EXTRACT UMBAUGH 48H/72H CELL-LEVEL SCORES + CDKN1A
# =============================================================================
log.info("\n" + "=" * 68)
log.info("STEP 3: Umbaugh 48h/72h cell-level scores + Cdkn1a expression")
log.info("=" * 68)

u72_mask  = mouse_adata.obs["dataset"] == "Umbaugh_48h72h"
u72_adata = mouse_adata[u72_mask]
log.info(f"  Umbaugh_48h72h: {u72_adata.n_obs:,} cells")
log.info(f"  Treatment values: {u72_adata.obs['treatment'].value_counts().to_dict()}")
log.info(f"  Timepoints: {sorted(u72_adata.obs['timepoint_hours'].unique().tolist())}")

z_u72  = u72_adata.obsm["X_scvi"].astype(np.float32)
trt_u72 = u72_adata.obs["treatment"].values.astype(str)
tp_u72  = u72_adata.obs["timepoint_hours"].values.astype(float)

# Scale map for Umbaugh 48h/72h:
# Control (0h)  → Scale 1, anchors: Walesky 0h
# APAP (48h)    → Scale 3, anchors: Walesky 24h + 48h
# APAP (72h)    → Scale 4, anchors: Walesky 48h + 96h
scale_map_u72 = {
    ("Control",  0.0): (0, [0.0]),
    ("APAP",    48.0): (2, [24.0, 48.0]),
    ("APAP",    72.0): (3, [48.0, 96.0]),
}

wself_u72, knn_u72, seed_u72 = score_external_cells(
    z_u72, 6582, edge_indices, edge_weights,
    walesky_hours, tp_u72, trt_u72, scale_map_u72
)
log.info(f"  W_self range: [{wself_u72.min():.4f}, {wself_u72.max():.4f}]")
log.info(f"  KNN range:    [{knn_u72.min():.4f},   {knn_u72.max():.4f}]")

# Extract Cdkn1a expression
cdkn1a_gene = "Cdkn1a"
if cdkn1a_gene not in u72_adata.var_names:
    # Try alternate capitalisations
    alts = [g for g in u72_adata.var_names if g.lower() == "cdkn1a"]
    if alts:
        cdkn1a_gene = alts[0]
        log.info(f"  Found Cdkn1a as: {cdkn1a_gene}")
    else:
        log.error(f"  Gene '{cdkn1a_gene}' not found in Umbaugh_48h72h var_names.")
        sys.exit(1)

cdkn1a_idx = list(u72_adata.var_names).index(cdkn1a_gene)
X72 = u72_adata.X
if hasattr(X72, "toarray"):
    cdkn1a_expr = X72[:, cdkn1a_idx].toarray().flatten().astype(np.float32)
else:
    cdkn1a_expr = np.asarray(X72[:, cdkn1a_idx]).flatten().astype(np.float32)

pct_expressed = (cdkn1a_expr > 0).mean() * 100
log.info(f"  Cdkn1a: mean={cdkn1a_expr.mean():.4f}  "
         f"% expressed={pct_expressed:.1f}%  "
         f"(expected ~34.1%)")

df_u72 = pd.DataFrame({
    "cell_id":           u72_adata.obs_names.tolist(),
    "dataset":           "Umbaugh_48h72h",
    "treatment":         trt_u72,
    "timepoint_hours":   tp_u72,
    "trajrisk_wself":    wself_u72,
    "trajrisk_knn":      knn_u72,
    "seeder":            seed_u72,
    "cdkn1a_expression": cdkn1a_expr,
})

if b1_model:
    df_u72["b1_xgb"] = b1_model.predict(z_u72).astype(np.float32)
else:
    df_u72["b1_xgb"] = np.nan

log.info(f"  Umbaugh 48h/72h records: {len(df_u72):,}")


# =============================================================================
# 6. SAVE MOUSE CELL SCORES CSV
# =============================================================================
log.info("\n" + "=" * 68)
log.info("STEP 4: Saving mouse cell scores")
log.info("=" * 68)

df_mouse = pd.concat([df_u2h, df_u72], ignore_index=True)

# Reorder columns to match spec
col_order = ["cell_id", "dataset", "treatment", "timepoint_hours",
             "trajrisk_knn", "trajrisk_wself", "seeder",
             "cdkn1a_expression", "b1_xgb"]
df_mouse = df_mouse[col_order]

out_mouse = RES / "manuscript_mouse_cell_scores.csv"
df_mouse.to_csv(str(out_mouse), index=False)
log.info(f"  Saved: {out_mouse.name}")
log.info(f"  Rows: {len(df_mouse):,}  |  Columns: {df_mouse.columns.tolist()}")

# Quick sanity check against known AUROC
from sklearn.metrics import roc_auc_score
u2h_rows = df_mouse[df_mouse["dataset"] == "Umbaugh_2h"].copy()
y_u2h = (u2h_rows["treatment"] == "APAP").astype(int).values
auroc_check_knn   = roc_auc_score(y_u2h, u2h_rows["trajrisk_knn"].values)
auroc_check_wself = roc_auc_score(y_u2h, u2h_rows["trajrisk_wself"].values)
log.info(f"\n  Sanity check Umbaugh 2h AUROC:")
log.info(f"    KNN:   {auroc_check_knn:.4f}  (expected 0.6524 from week9a run)")
log.info(f"    W_self:{auroc_check_wself:.4f}  (expected 0.6087 from week9a run)")


# =============================================================================
# 7. EXTRACT HUMAN NEIGHBOUR COMPOSITION
# =============================================================================
log.info("\n" + "=" * 68)
log.info("STEP 5: Human neighbour composition")
log.info("=" * 68)

combined = sc.read_h5ad(str(DATA / "adata_cross_species_combined.h5ad"))
assert "X_scvi_cross" in combined.obsm, \
    "X_scvi_cross not found — did week9b save the combined adata correctly?"

n_wal_cs = 6582   # Walesky cells are first in the combined adata
z_combined = combined.obsm["X_scvi_cross"]
z_wal_cs   = z_combined[:n_wal_cs].astype(np.float32)
z_hum_cs   = z_combined[n_wal_cs:].astype(np.float32)
n_human    = z_hum_cs.shape[0]
log.info(f"  Combined adata: {combined.n_obs:,} cells")
log.info(f"  Walesky positions: 0–{n_wal_cs-1}  |  Human: {n_wal_cs}–{combined.n_obs-1}")

# Load Walesky hours from walesky_cell_index (already loaded above)
# walesky_hours is already aligned to positions 0-6581 in the combined adata
assert len(walesky_hours) == n_wal_cs

log.info(f"  Finding {KNN_K} nearest Walesky cells for each of {n_human:,} human cells ...")
knn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean", n_jobs=-1)
knn.fit(z_wal_cs)
_, local_idx = knn.kneighbors(z_hum_cs)   # (n_human, KNN_K)

# Neighbour timepoint composition per cell
nbr_hours = walesky_hours[local_idx]       # (n_human, KNN_K)
hour_vals  = [0.0, 6.0, 24.0, 48.0, 96.0]
hour_names = ["pct_0h", "pct_6h", "pct_24h", "pct_48h", "pct_96h"]
comp_arrays = {
    name: (nbr_hours == hv).mean(axis=1).astype(np.float32)
    for hv, name in zip(hour_vals, hour_names)
}

# Map each human cell to its patient_id and condition
# Human cells are at positions n_wal_cs onward in combined adata
# The week9b_cell_risk_scores.csv has patient_id and condition in the same order
cell_scores = pd.read_csv(RES / "week9b_cell_risk_scores.csv", index_col=0)
assert len(cell_scores) == n_human, \
    f"Human cell count mismatch: adata={n_human}, CSV={len(cell_scores)}"

patient_ids = cell_scores["patient_id"].values
conditions  = cell_scores["condition"].values
log.info(f"  Patient IDs mapped: {pd.unique(patient_ids).tolist()}")

# Build per-cell composition DataFrame
df_comp_cells = pd.DataFrame(comp_arrays)
df_comp_cells["patient_id"] = patient_ids
df_comp_cells["condition"]  = conditions

# Sanity: log group-level means vs logged values
log.info("\n  Composition (mean per condition) — verify against log output:")
is_alf = conditions == "APAP_ALF"
for col in hour_names:
    v = df_comp_cells[col].values
    log.info(f"    {col}: ALF={v[is_alf].mean():.3f}  Healthy={v[~is_alf].mean():.3f}")
log.info("  (Expected from run log: ALF: 0h=0.268, 6h=0.078, 24h=0.251, 48h=0.244, 96h=0.159)")
log.info("               Healthy: 0h=0.387, 6h=0.057, 24h=0.078, 48h=0.123, 96h=0.355)")

# Aggregate per donor
donor_comp = (
    df_comp_cells.groupby("patient_id")
    .agg(
        condition  = ("condition", "first"),
        n_cells    = ("condition", "count"),
        pct_0h     = ("pct_0h",  "mean"),
        pct_6h     = ("pct_6h",  "mean"),
        pct_24h    = ("pct_24h", "mean"),
        pct_48h    = ("pct_48h", "mean"),
        pct_96h    = ("pct_96h", "mean"),
    )
    .reset_index()
)
log.info(f"\n  Donor-level composition: {len(donor_comp)} rows")


# =============================================================================
# 8. SAVE HUMAN COMPOSITION CSV
# =============================================================================
log.info("\n" + "=" * 68)
log.info("STEP 6: Saving human composition CSV")
log.info("=" * 68)

out_comp = RES / "manuscript_human_composition.csv"
donor_comp.to_csv(str(out_comp), index=False)
log.info(f"  Saved: {out_comp.name}")
log.info(f"  Rows: {len(donor_comp)}  |  Columns: {donor_comp.columns.tolist()}")


# =============================================================================
# 9. FINAL SUMMARY
# =============================================================================
log.info("\n" + "=" * 68)
log.info("EXTRACTION COMPLETE")
log.info("=" * 68)
log.info(f"  manuscript_mouse_cell_scores.csv  : {len(df_mouse):,} rows")
log.info(f"  manuscript_human_composition.csv  : {len(donor_comp)} rows")
log.info("\n  Panel readiness after this run:")
log.info("    Mouse Panel A (AUROC bar chart):      Ready (week9a_mouse_validation_table.csv)")
log.info("    Mouse Panel B (Umbaugh 2h violin):    Ready (manuscript_mouse_cell_scores.csv)")
log.info("    Mouse Panel C (Cdkn1a scatter):       Ready (manuscript_mouse_cell_scores.csv)")
log.info("    Mouse Panel D (Neighbour composition):Ready (manuscript_human_composition.csv)")
log.info("    Human Panel B (Donor strip plot):     Ready (week9b_donor_primary.csv)")
log.info("    Human Panel C (Sensitivity strip):    Ready (week9b_donor_sensitivity.csv)")
log.info("    Human Panel D (Convergence scatter):  Ready (week9b_donor_primary.csv)")
log.info("\n  Now proceed to figure generation script.")
