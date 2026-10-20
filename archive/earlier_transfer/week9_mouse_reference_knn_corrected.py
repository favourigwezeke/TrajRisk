#!/usr/bin/env python3
"""
TrajRisk — Week 9: Corrected Reference-KNN Mouse Cross-Dataset Validation
========================================================================

Corrected design:
- External cells receive reference KNN messages only in biologically relevant
  TrajRisk scale branches.
- Anchor cells are chosen per external cell, not per dataset globally.
- Control cells anchor to Walesky 0h controls.
- APAP cells anchor to phase-appropriate Walesky reference states.
- W_self-only, Seeder, and B1 XGBoost are retained for comparison.

This is still a validation experiment, not retraining.
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

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("W9-KNN-CORRECTED")


# =============================================================================
# 0. CONFIG
# =============================================================================

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "Data" / "processed"
MODELS = BASE / "models"
RES = BASE / "results"

RES.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
RANDOM_SEED = 42
N_BOOTSTRAP = 1000
MIN_POS = 5
HIGH_RISK_THR = 0.10
CDKN1A_GENE = "Cdkn1a"
KNN_K = 15

SCALE_NAMES = [
    "Scale1_AcuteStress",   # 0h -> 6h
    "Scale2_PeakInjury",    # 6h -> 24h
    "Scale3_FateDecision",  # 24h -> 48h
    "Scale4_Resolution",    # 48h -> 96h
]

NODE_DIM = 31
HIDDEN_DIM = 64
N_SCALES = 4
DROPOUT = 0.3

XGB_B1_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    verbosity=0,
    n_jobs=1,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# 1. MODEL CLASSES
# =============================================================================

class TrajRiskConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.W_nbr = nn.Linear(in_channels, out_channels, bias=False)
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
        node_dim=NODE_DIM,
        hidden_dim=HIDDEN_DIM,
        n_scales=N_SCALES,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.scale_convs = nn.ModuleList()
        self.scale_bns = nn.ModuleList()

        for _ in range(n_scales):
            self.scale_convs.append(nn.ModuleList([
                TrajRiskConv(node_dim, hidden_dim),
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
# 2. LOAD DATA
# =============================================================================

def load_and_verify() -> Dict:
    log.info("=" * 72)
    log.info("LOADING AND VERIFYING DATA")
    log.info("=" * 72)

    adata = sc.read_h5ad(str(DATA / "adata_integrated_with_trajectory.h5ad"))
    assert adata.n_obs == 36067, f"Unexpected n_obs: {adata.n_obs}"
    assert "X_scvi" in adata.obsm, "X_scvi missing"
    assert CDKN1A_GENE in adata.var_names, f"{CDKN1A_GENE} missing"

    required_obs = {"dataset", "treatment", "timepoint_hours"}
    missing_obs = required_obs - set(adata.obs.columns)
    assert not missing_obs, f"Missing obs columns: {missing_obs}"

    log.info(f"adata: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    log.info(f"datasets:\n{adata.obs['dataset'].value_counts().to_string()}")

    cell_index = pd.read_csv(DATA / "walesky_cell_index.csv", index_col=0)
    assert len(cell_index) == 6582, f"cell_index length wrong: {len(cell_index)}"

    required_ci = {"hours", "timepoint_str", "treatment_str"}
    missing_ci = required_ci - set(cell_index.columns)
    assert not missing_ci, f"cell_index missing columns: {missing_ci}"

    if "global_idx" in cell_index.columns:
        assert np.all(cell_index["global_idx"].values == np.arange(len(cell_index))), (
            "cell_index global_idx does not match row order"
        )

    x_walesky = torch.load(DATA / "week6_node_features.pt", map_location="cpu")
    assert x_walesky.shape == (6582, 31), f"x_walesky shape wrong: {x_walesky.shape}"

    # Strong alignment check: compare x_walesky scVI to adata Walesky X_scvi order if possible
    adata_wal = adata[adata.obs["dataset"] == "Walesky"]
    overlap = list(set(cell_index.index[:100]) & set(adata_wal.obs_names))
    assert len(overlap) >= 50, "Poor barcode overlap between cell_index and adata Walesky cells"

    graphs = torch.load(DATA / "multiscale_graphs.pt", map_location="cpu")
    edge_indices = [graphs[sn]["edge_index"] for sn in SCALE_NAMES]
    edge_weights = [graphs[sn]["edge_weight"] for sn in SCALE_NAMES]

    for s, ei in enumerate(edge_indices):
        assert ei.max().item() <= 6581, f"Scale {s} edge index exceeds Walesky range"

    ckpt_path = MODELS / "week7_trajrisk_FINAL.pt"
    if not ckpt_path.exists():
        ckpt_path = MODELS / "week7_trajrisk_BEST_AUROC.pt"
    assert ckpt_path.exists(), "No Week 7 TrajRisk checkpoint found"

    model = TrajRiskGNN().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seeder = joblib.load(MODELS / "week6_xgboost_model.pkl")
    seeder_walesky = np.load(DATA / "week6_xgboost_risk.npy")

    preds_w7 = pd.read_csv(DATA / "week7_predictions.csv", index_col=0)
    trajrisk_walesky = preds_w7["risk_predicted"].values.astype(np.float32)

    td = torch.load(DATA / "week6_training_data.pt", map_location="cpu")

    log.info("All files loaded and verified.\n")

    return {
        "adata": adata,
        "cell_index": cell_index,
        "x_walesky": x_walesky,
        "edge_indices": edge_indices,
        "edge_weights": edge_weights,
        "model": model,
        "seeder": seeder,
        "seeder_walesky": seeder_walesky,
        "trajrisk_walesky": trajrisk_walesky,
        "train_idx": td["train_idx"].numpy(),
        "val_idx": td["val_idx"].numpy(),
        "train_y": td["train_y"].numpy(),
        "val_y": td["val_y"].numpy(),
    }


# =============================================================================
# 3. B1 XGBOOST
# =============================================================================

def fit_b1_xgboost(data: Dict) -> Tuple[XGBRegressor, np.ndarray]:
    log.info("Fitting B1 XGBoost...")
    x_scvi_wal = data["x_walesky"][:, :30].numpy()

    model = XGBRegressor(**XGB_B1_PARAMS)
    model.fit(x_scvi_wal[data["train_idx"]], data["train_y"])

    pred_wal = np.clip(model.predict(x_scvi_wal).astype(np.float32), 0, 1)
    log.info(f"B1 Walesky range: [{pred_wal.min():.4f}, {pred_wal.max():.4f}]\n")
    return model, pred_wal


# =============================================================================
# 4. CELL-SPECIFIC ANCHOR ASSIGNMENT
# =============================================================================

def scale_and_anchor_hours_for_cell(dataset: str, treatment: str, hour: float) -> Tuple[List[int], List[float]]:
    """
    Returns:
        scales_to_receive_edges: list of scale indices 0-3
        anchor_hours: Walesky hours used as anchors

    Core correction:
    - Controls anchor to 0h only.
    - APAP cells anchor to phase-appropriate Walesky states.
    - Edges are injected only into biologically relevant scale branches.
    """
    treatment = str(treatment)
    hour = float(hour)

    # Controls should not receive messages from injured/recovery states
    if treatment == "Control":
        return [0], [0.0]

    # Umbaugh 2h APAP: early injury between 0h and 6h
    if dataset == "Umbaugh_2h":
        return [0], [0.0, 6.0]

    # Ben-Moshe and Umbaugh late APAP
    if dataset in {"Ben_Moshe", "Umbaugh_48h72h"}:
        if hour <= 48:
            # target resembles 48h phase; use Scale3 branch, source 24h/48h neighborhood
            return [2], [24.0, 48.0]
        else:
            # 72h lies between 48h and 96h; use Scale4 branch
            return [3], [48.0, 96.0]

    # Fallback: W_self only, no added reference edges
    return [], []


def build_cell_specific_reference_edges(
    data: Dict,
    z_ext: np.ndarray,
    obs_ext: pd.DataFrame,
    dataset_name: str,
    k: int = KNN_K,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Build separate reference edge tensors for each TrajRisk scale.

    Unlike the previous version, this does NOT add identical edges to all scales.
    Each external cell receives edges only in the biologically relevant scale.
    """
    n_wal = data["x_walesky"].shape[0]
    n_ext = z_ext.shape[0]

    ci = data["cell_index"]
    z_wal = data["x_walesky"][:, :30].numpy()

    scale_srcs = [[] for _ in range(N_SCALES)]
    scale_tgts = [[] for _ in range(N_SCALES)]
    scale_wgts = [[] for _ in range(N_SCALES)]

    # Pre-cache anchor NN models by anchor hour tuple
    anchor_cache = {}

    treatments = obs_ext["treatment"].astype(str).values
    hours = obs_ext["timepoint_hours"].astype(float).values

    for i in range(n_ext):
        scales, anchor_hours = scale_and_anchor_hours_for_cell(
            dataset=dataset_name,
            treatment=treatments[i],
            hour=hours[i],
        )

        if not scales or not anchor_hours:
            continue

        key = tuple(anchor_hours)

        if key not in anchor_cache:
            mask = ci["hours"].isin(anchor_hours).values

            # If anchoring controls, restrict to Walesky Control where possible
            if anchor_hours == [0.0] or key == (0.0,):
                if "treatment_str" in ci.columns:
                    mask = mask & (ci["treatment_str"].astype(str).values == "Control")

            anchor_idx = np.where(mask)[0].astype(np.int64)
            z_anchor = z_wal[anchor_idx]

            if len(anchor_idx) == 0:
                raise ValueError(f"No anchors found for hours={anchor_hours}")

            kk = min(k, len(anchor_idx))
            nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", n_jobs=-1)
            nn.fit(z_anchor)

            anchor_cache[key] = {
                "nn": nn,
                "anchor_idx": anchor_idx,
                "k": kk,
            }

        cached = anchor_cache[key]
        nn = cached["nn"]
        anchor_idx = cached["anchor_idx"]
        kk = cached["k"]

        dist, loc = nn.kneighbors(z_ext[i:i+1])
        dist = dist.flatten()
        loc = loc.flatten()

        nonzero = dist[dist > 0]
        sigma = float(np.median(nonzero)) if len(nonzero) else 1.0
        weights = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
        weights = weights / (weights.sum() + 1e-8)

        src_global = anchor_idx[loc]
        tgt_global = np.full(kk, n_wal + i, dtype=np.int64)

        for s in scales:
            scale_srcs[s].extend(src_global.tolist())
            scale_tgts[s].extend(tgt_global.tolist())
            scale_wgts[s].extend(weights.tolist())

    ref_edge_indices = []
    ref_edge_weights = []

    for s in range(N_SCALES):
        if len(scale_srcs[s]) == 0:
            ref_edge_indices.append(torch.empty((2, 0), dtype=torch.long))
            ref_edge_weights.append(torch.empty((0,), dtype=torch.float32))
        else:
            ei = torch.tensor(
                np.vstack([np.array(scale_srcs[s]), np.array(scale_tgts[s])]),
                dtype=torch.long,
            )
            ew = torch.tensor(np.array(scale_wgts[s], dtype=np.float32), dtype=torch.float32)
            ref_edge_indices.append(ei)
            ref_edge_weights.append(ew)

        log.info(
            f"    Scale {s+1} ref edges: {ref_edge_indices[s].shape[1]:,}"
        )

    return ref_edge_indices, ref_edge_weights


# =============================================================================
# 5. SCORE COMPUTATION
# =============================================================================

def compute_wself_scores(data: Dict, b1: XGBRegressor, z_ext: np.ndarray) -> Dict[str, np.ndarray]:
    n_wal = data["x_walesky"].shape[0]

    seeder = data["seeder"].predict_proba(z_ext)[:, 1].astype(np.float32)
    b1_pred = np.clip(b1.predict(z_ext).astype(np.float32), 0, 1)

    x_ext = np.hstack([z_ext, seeder.reshape(-1, 1)]).astype(np.float32)
    x_all = torch.tensor(
        np.vstack([data["x_walesky"].numpy(), x_ext]),
        dtype=torch.float32,
    )

    with torch.no_grad():
        data["model"].eval()
        risk_all = data["model"](x_all, data["edge_indices"], data["edge_weights"]).numpy()

    return {
        "trajrisk_wself": risk_all[n_wal:].astype(np.float32),
        "seeder": seeder,
        "b1_xgb": b1_pred,
    }


def compute_knn_scores(
    data: Dict,
    b1: XGBRegressor,
    z_ext: np.ndarray,
    obs_ext: pd.DataFrame,
    dataset_name: str,
) -> Dict[str, np.ndarray]:
    """
    Corrected reference-KNN scoring.

    Adds scale-specific, cell-specific edges and compares with W_self/seeder/B1.
    """
    n_wal = data["x_walesky"].shape[0]

    base = compute_wself_scores(data, b1, z_ext)

    log.info(f"  Building corrected cell-specific KNN edges for {dataset_name}...")
    ref_eis, ref_ews = build_cell_specific_reference_edges(
        data=data,
        z_ext=z_ext,
        obs_ext=obs_ext,
        dataset_name=dataset_name,
        k=KNN_K,
    )

    aug_indices = []
    aug_weights = []

    for s in range(N_SCALES):
        aug_indices.append(torch.cat([data["edge_indices"][s], ref_eis[s]], dim=1))
        aug_weights.append(torch.cat([data["edge_weights"][s], ref_ews[s]], dim=0))

    seeder = base["seeder"]
    x_ext = np.hstack([z_ext, seeder.reshape(-1, 1)]).astype(np.float32)
    x_all = torch.tensor(
        np.vstack([data["x_walesky"].numpy(), x_ext]),
        dtype=torch.float32,
    )

    with torch.no_grad():
        data["model"].eval()
        risk_all = data["model"](x_all, aug_indices, aug_weights).numpy()

    knn = risk_all[n_wal:].astype(np.float32)

    log.info(f"  KNN range: [{knn.min():.4f}, {knn.max():.4f}]")
    log.info(f"  W_self range: [{base['trajrisk_wself'].min():.4f}, {base['trajrisk_wself'].max():.4f}]")

    return {
        "trajrisk_knn": knn,
        "trajrisk_wself": base["trajrisk_wself"],
        "seeder": base["seeder"],
        "b1_xgb": base["b1_xgb"],
    }


# =============================================================================
# 6. METRICS
# =============================================================================

def safe_auroc(y, s):
    if y.sum() < MIN_POS or (1 - y).sum() < MIN_POS:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_auprc(y, s):
    if y.sum() < MIN_POS:
        return np.nan
    return float(average_precision_score(y, s))


def bootstrap_ci(y, s, n=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_SEED)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(roc_auc_score(yy, s[idx]))
    if not vals:
        return np.nan, np.nan
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def report_metrics(test_name: str, y: np.ndarray, scores: Dict[str, np.ndarray]) -> List[Dict]:
    log.info(f"\n{test_name}")
    log.info(f"n={len(y):,} | positives={int(y.sum()):,} | prevalence={y.mean():.3f}")
    log.info(f"{'Method':<18} {'AUROC':>8} {'95% CI':>18} {'AUPRC':>8}")
    log.info("-" * 60)

    rows = []

    for method, s in scores.items():
        auroc = safe_auroc(y, s)
        auprc = safe_auprc(y, s)
        lo, hi = bootstrap_ci(y, s) if not np.isnan(auroc) else (np.nan, np.nan)

        ci = f"({lo:.3f}-{hi:.3f})" if not np.isnan(lo) else "(—)"
        log.info(f"{method:<18} {auroc:>8.4f} {ci:>18} {auprc:>8.4f}")

        rows.append({
            "test": test_name,
            "method": method,
            "n_total": int(len(y)),
            "n_pos": int(y.sum()),
            "prevalence": round(float(y.mean()), 4),
            "auroc": round(float(auroc), 4) if not np.isnan(auroc) else np.nan,
            "auroc_ci_lo": round(float(lo), 4) if not np.isnan(lo) else np.nan,
            "auroc_ci_hi": round(float(hi), 4) if not np.isnan(hi) else np.nan,
            "auprc": round(float(auprc), 4) if not np.isnan(auprc) else np.nan,
            "auprc_rand": round(float(y.mean()), 4),
        })

    return rows


# =============================================================================
# 7. TESTS
# =============================================================================

def run_t0(data: Dict, b1_wal: np.ndarray) -> List[Dict]:
    vi = data["val_idx"]
    y = (data["val_y"] > HIGH_RISK_THR).astype(int)

    scores = {
        "trajrisk_knn": data["trajrisk_walesky"][vi],
        "trajrisk_wself": data["trajrisk_walesky"][vi],
        "seeder": data["seeder_walesky"][vi],
        "b1_xgb": b1_wal[vi],
    }

    return report_metrics("T0: Walesky val reference", y, scores)


def run_external_test(data: Dict, b1: XGBRegressor, dataset_name: str, label_col: str = "treatment") -> List[Dict]:
    adata = data["adata"]
    sub = adata[adata.obs["dataset"] == dataset_name]

    z = sub.obsm["X_scvi"].astype(np.float32)
    obs = sub.obs.copy()

    y = (obs[label_col].astype(str).values == "APAP").astype(int)

    log.info("=" * 72)
    log.info(f"{dataset_name}: APAP vs Control")
    log.info("=" * 72)
    log.info(pd.crosstab(obs["treatment"], obs["timepoint_hours"]).to_string())

    scores = compute_knn_scores(data, b1, z, obs, dataset_name)
    return report_metrics(f"{dataset_name}: APAP vs Control", y, scores)


def run_p21_test(data: Dict, b1: XGBRegressor) -> Tuple[List[Dict], pd.DataFrame]:
    adata = data["adata"]
    sub = adata[adata.obs["dataset"] == "Umbaugh_48h72h"]

    z = sub.obsm["X_scvi"].astype(np.float32)
    obs = sub.obs.copy()

    gene_idx = list(sub.var_names).index(CDKN1A_GENE)
    X = sub.X
    cdkn = np.asarray(X[:, gene_idx].toarray()).flatten() if hasattr(X, "toarray") else np.asarray(X[:, gene_idx]).flatten()

    log.info("=" * 72)
    log.info("Umbaugh_48h72h: Cdkn1a / p21 validation")
    log.info("=" * 72)
    log.info(f"Cdkn1a mean={cdkn.mean():.4f} median={np.median(cdkn):.4f} expressed={(cdkn > 0).mean()*100:.1f}%")

    scores = compute_knn_scores(data, b1, z, obs, "Umbaugh_48h72h")

    # Because median is zero, use expressed vs non-expressed and top tertile check
    y_expr = (cdkn > 0).astype(int)
    rows = report_metrics("Umbaugh_48h72h: Cdkn1a expressed vs zero", y_expr, scores)

    p21_rows = []

    for method, s in scores.items():
        rho, p = spearmanr(cdkn, s)
        high = s[y_expr == 1]
        low = s[y_expr == 0]
        stat, p_mw = mannwhitneyu(high, low, alternative="greater")

        p21_rows.append({
            "method": method,
            "spearman_rho_cdkn1a": round(float(rho), 4),
            "spearman_p": float(p),
            "mean_cdkn1a_expressed": round(float(high.mean()), 4),
            "mean_cdkn1a_zero": round(float(low.mean()), 4),
            "mw_p": float(p_mw),
        })

    y_apap = (obs["treatment"].astype(str).values == "APAP").astype(int)
    rows += report_metrics("Umbaugh_48h72h: APAP vs Control", y_apap, scores)

    return rows, pd.DataFrame(p21_rows)


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    t0 = time.time()
    log.info("=" * 72)
    log.info("TRAJRISK — WEEK 9 CORRECTED REFERENCE-KNN VALIDATION")
    log.info("=" * 72)

    data = load_and_verify()
    b1, b1_wal = fit_b1_xgboost(data)

    all_rows = []

    all_rows += run_t0(data, b1_wal)
    all_rows += run_external_test(data, b1, "Ben_Moshe")
    all_rows += run_external_test(data, b1, "Umbaugh_2h")
    p21_rows, p21_df = run_p21_test(data, b1)
    all_rows += p21_rows

    df = pd.DataFrame(all_rows)

    log.info("=" * 72)
    log.info("SUMMARY")
    log.info("=" * 72)

    summary = []

    for test in df["test"].unique():
        sub = df[df["test"] == test]

        def get(method):
            vals = sub.loc[sub["method"] == method, "auroc"].values
            return float(vals[0]) if len(vals) else np.nan

        knn = get("trajrisk_knn")
        wself = get("trajrisk_wself")
        seed = get("seeder")
        b1x = get("b1_xgb")

        summary.append({
            "test": test,
            "trajrisk_knn": knn,
            "trajrisk_wself": wself,
            "seeder": seed,
            "b1_xgb": b1x,
            "delta_knn_vs_wself_pp": round((knn - wself) * 100, 2),
            "delta_knn_vs_seeder_pp": round((knn - seed) * 100, 2),
            "delta_knn_vs_b1_pp": round((knn - b1x) * 100, 2),
        })

    summary_df = pd.DataFrame(summary)
    log.info("\n" + summary_df.to_string(index=False))

    df.to_csv(RES / "week9_corrected_knn_validation_table.csv", index=False)
    p21_df.to_csv(RES / "week9_corrected_knn_p21_analysis.csv", index=False)
    summary_df.to_csv(RES / "week9_corrected_knn_summary.csv", index=False)

    with open(RES / "week9_corrected_knn_summary.json", "w") as f:
        json.dump({
            "variant": "corrected_cell_specific_scale_specific_reference_knn",
            "knn_k": KNN_K,
            "runtime_minutes": round((time.time() - t0) / 60, 2),
            "summary": summary,
        }, f, indent=2)

    log.info(f"\nDone in {(time.time() - t0)/60:.2f} minutes.")
    log.info("Outputs saved to results/.")


if __name__ == "__main__":
    main()
