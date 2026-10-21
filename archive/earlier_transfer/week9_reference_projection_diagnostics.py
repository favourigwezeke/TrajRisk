#!/usr/bin/env python3
"""Earlier TrajRisk analysis retained for source comparison.

See docs/workflow.md for the study workflow and docs/reproduction.md for
the commands used to reproduce the reported analyses.
"""

import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

import scanpy as sc
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("W9C-DIAG")

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "Data" / "processed"
MODELS = BASE / "models"
RES = BASE / "results"
RES.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
HIGH_RISK_THR = 0.10
CDKN1A_GENE = "Cdkn1a"
MIN_POS = 5

SCALE_NAMES = ["Scale1_AcuteStress", "Scale2_PeakInjury", "Scale3_FateDecision", "Scale4_Resolution"]
NODE_DIM, HIDDEN_DIM, N_SCALES, DROPOUT = 31, 64, 4, 0.3

K_VALUES = [5, 15, 30]
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.00]
PROJECTION_MODES = ["wself", "apap_only", "all_cells"]

XGB_B1_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbosity=0, n_jobs=1,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class TrajRiskConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add", flow="source_to_target")
        self.W_nbr = nn.Linear(in_channels, out_channels, bias=False)
        self.W_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, edge_weight):
        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.W_nbr(agg) + self.W_self(x)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j


class TrajRiskGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_scales = N_SCALES
        self.scale_convs = nn.ModuleList()
        self.scale_bns = nn.ModuleList()
        for _ in range(N_SCALES):
            self.scale_convs.append(nn.ModuleList([
                TrajRiskConv(NODE_DIM, HIDDEN_DIM),
                TrajRiskConv(HIDDEN_DIM, HIDDEN_DIM),
            ]))
            self.scale_bns.append(nn.ModuleList([
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
            ]))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(N_SCALES * HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_indices, edge_weights):
        parts = []
        for s in range(N_SCALES):
            h = x
            for layer in range(2):
                h = self.scale_convs[s][layer](h, edge_indices[s], edge_weights[s])
                h = self.scale_bns[s][layer](h)
                h = torch.relu(h)
            parts.append(h)
        return self.fusion_mlp(torch.cat(parts, dim=1)).squeeze(-1)


def load_data() -> Dict:
    log.info("Loading data...")
    adata = sc.read_h5ad(str(DATA / "adata_integrated_with_trajectory.h5ad"))
    cell_index = pd.read_csv(DATA / "walesky_cell_index.csv", index_col=0)
    x_walesky = torch.load(DATA / "week6_node_features.pt", map_location="cpu")

    graphs = torch.load(DATA / "multiscale_graphs.pt", map_location="cpu")
    edge_indices = [graphs[s]["edge_index"] for s in SCALE_NAMES]
    edge_weights = [graphs[s]["edge_weight"] for s in SCALE_NAMES]

    ckpt_path = MODELS / "week7_trajrisk_FINAL.pt"
    if not ckpt_path.exists():
        ckpt_path = MODELS / "week7_trajrisk_BEST_AUROC.pt"

    model = TrajRiskGNN()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seeder = joblib.load(MODELS / "week6_xgboost_model.pkl")
    td = torch.load(DATA / "week6_training_data.pt", map_location="cpu")
    seeder_walesky = np.load(DATA / "week6_xgboost_risk.npy")
    preds_w7 = pd.read_csv(DATA / "week7_predictions.csv", index_col=0)

    assert adata.n_obs == 36067
    assert x_walesky.shape == (6582, 31)
    assert CDKN1A_GENE in adata.var_names
    if "global_idx" in cell_index.columns:
        assert np.all(cell_index["global_idx"].values == np.arange(len(cell_index)))

    return dict(
        adata=adata,
        cell_index=cell_index,
        x_walesky=x_walesky,
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        model=model,
        seeder=seeder,
        train_idx=td["train_idx"].numpy(),
        val_idx=td["val_idx"].numpy(),
        train_y=td["train_y"].numpy(),
        val_y=td["val_y"].numpy(),
        seeder_walesky=seeder_walesky,
        trajrisk_walesky=preds_w7["risk_predicted"].values.astype(np.float32),
    )


def fit_b1(data):
    x = data["x_walesky"][:, :30].numpy()
    b1 = XGBRegressor(**XGB_B1_PARAMS)
    b1.fit(x[data["train_idx"]], data["train_y"])
    return b1, np.clip(b1.predict(x).astype(np.float32), 0, 1)


def safe_metrics(y, score):
    y = np.asarray(y).astype(int)
    if y.sum() < MIN_POS or (1 - y).sum() < MIN_POS:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def phase_for_cell(dataset, treatment, hour):
    treatment = str(treatment)
    hour = float(hour)

    if treatment == "Control":
        return [0], [0.0]

    if dataset == "Umbaugh_2h":
        return [0], [0.0, 6.0]

    if dataset in {"Ben_Moshe", "Umbaugh_48h72h"}:
        if hour <= 48:
            return [2], [24.0, 48.0]
        return [3], [48.0, 96.0]

    return [], []


def build_ref_edges(data, z_ext, obs, dataset_name, k, alpha, mode):
    n_wal = data["x_walesky"].shape[0]
    ci = data["cell_index"]
    z_wal = data["x_walesky"][:, :30].numpy()

    src_by_scale = [[] for _ in range(N_SCALES)]
    tgt_by_scale = [[] for _ in range(N_SCALES)]
    w_by_scale = [[] for _ in range(N_SCALES)]

    cache = {}

    treatments = obs["treatment"].astype(str).values
    hours = obs["timepoint_hours"].astype(float).values

    for i in range(z_ext.shape[0]):
        treatment = treatments[i]

        if mode == "wself":
            continue
        if mode == "apap_only" and treatment == "Control":
            continue

        scales, anchor_hours = phase_for_cell(dataset_name, treatment, hours[i])
        if not scales:
            continue

        key = tuple(anchor_hours)
        if key not in cache:
            mask = ci["hours"].isin(anchor_hours).values
            if key == (0.0,) and "treatment_str" in ci.columns:
                mask = mask & (ci["treatment_str"].astype(str).values == "Control")

            anchor_idx = np.where(mask)[0].astype(np.int64)
            z_anchor = z_wal[anchor_idx]
            kk = min(k, len(anchor_idx))
            nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", n_jobs=-1)
            nn.fit(z_anchor)
            cache[key] = (nn, anchor_idx, kk)

        nn, anchor_idx, kk = cache[key]
        dist, loc = nn.kneighbors(z_ext[i:i+1])
        dist, loc = dist.flatten(), loc.flatten()

        nz = dist[dist > 0]
        sigma = float(np.median(nz)) if len(nz) else 1.0
        weights = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
        weights = weights / (weights.sum() + 1e-8)
        weights = weights * alpha

        src = anchor_idx[loc]
        tgt = np.full(kk, n_wal + i, dtype=np.int64)

        for s in scales:
            src_by_scale[s].extend(src.tolist())
            tgt_by_scale[s].extend(tgt.tolist())
            w_by_scale[s].extend(weights.tolist())

    eis, ews = [], []
    for s in range(N_SCALES):
        if len(src_by_scale[s]) == 0:
            eis.append(torch.empty((2, 0), dtype=torch.long))
            ews.append(torch.empty((0,), dtype=torch.float32))
        else:
            eis.append(torch.tensor(np.vstack([src_by_scale[s], tgt_by_scale[s]]), dtype=torch.long))
            ews.append(torch.tensor(np.array(w_by_scale[s], dtype=np.float32), dtype=torch.float32))

    return eis, ews


def score_variant(data, b1, z_ext, obs, dataset_name, k, alpha, mode):
    n_wal = data["x_walesky"].shape[0]

    seeder = data["seeder"].predict_proba(z_ext)[:, 1].astype(np.float32)
    b1_pred = np.clip(b1.predict(z_ext).astype(np.float32), 0, 1)

    x_ext = np.hstack([z_ext, seeder.reshape(-1, 1)]).astype(np.float32)
    x_all = torch.tensor(np.vstack([data["x_walesky"].numpy(), x_ext]), dtype=torch.float32)

    ref_eis, ref_ews = build_ref_edges(data, z_ext, obs, dataset_name, k, alpha, mode)

    aug_eis = [torch.cat([data["edge_indices"][s], ref_eis[s]], dim=1) for s in range(N_SCALES)]
    aug_ews = [torch.cat([data["edge_weights"][s], ref_ews[s]], dim=0) for s in range(N_SCALES)]

    with torch.no_grad():
        data["model"].eval()
        pred_all = data["model"](x_all, aug_eis, aug_ews).numpy()

    return pred_all[n_wal:].astype(np.float32), seeder, b1_pred


def run_dataset(data, b1, dataset_name):
    adata = data["adata"]
    sub = adata[adata.obs["dataset"] == dataset_name].copy()
    z = sub.obsm["X_scvi"].astype(np.float32)
    obs = sub.obs.copy()

    y_apap = (obs["treatment"].astype(str).values == "APAP").astype(int)

    rows = []

    for mode in PROJECTION_MODES:
        for k in K_VALUES:
            for alpha in ALPHAS:
                if mode == "wself" and (k != K_VALUES[0] or alpha != 0.0):
                    continue
                if mode != "wself" and alpha == 0.0:
                    continue

                pred, seed, b1p = score_variant(data, b1, z, obs, dataset_name, k, alpha, mode)

                for method, score in [
                    ("trajrisk", pred),
                    ("seeder", seed),
                    ("b1_xgb", b1p),
                ]:
                    auroc, auprc = safe_metrics(y_apap, score)
                    rows.append(dict(
                        dataset=dataset_name,
                        test=f"{dataset_name}: APAP vs Control",
                        label="APAP_vs_Control",
                        mode=mode,
                        k=k,
                        alpha=alpha,
                        method=method,
                        n_total=len(y_apap),
                        n_pos=int(y_apap.sum()),
                        prevalence=round(float(y_apap.mean()), 4),
                        auroc=round(auroc, 4) if not np.isnan(auroc) else np.nan,
                        auprc=round(auprc, 4) if not np.isnan(auprc) else np.nan,
                    ))

                if dataset_name == "Umbaugh_48h72h":
                    gene_idx = list(sub.var_names).index(CDKN1A_GENE)
                    X = sub.X
                    cdkn = np.asarray(X[:, gene_idx].toarray()).flatten() if hasattr(X, "toarray") else np.asarray(X[:, gene_idx]).flatten()

                    labels = {
                        "Cdkn1a_expressed": (cdkn > 0).astype(int),
                        "Cdkn1a_top_quartile": (cdkn >= np.percentile(cdkn, 75)).astype(int),
                    }

                    for label_name, y in labels.items():
                        for method, score in [("trajrisk", pred), ("seeder", seed), ("b1_xgb", b1p)]:
                            auroc, auprc = safe_metrics(y, score)
                            rho, p = spearmanr(cdkn, score)
                            rows.append(dict(
                                dataset=dataset_name,
                                test=f"{dataset_name}: {label_name}",
                                label=label_name,
                                mode=mode,
                                k=k,
                                alpha=alpha,
                                method=method,
                                n_total=len(y),
                                n_pos=int(y.sum()),
                                prevalence=round(float(y.mean()), 4),
                                auroc=round(auroc, 4) if not np.isnan(auroc) else np.nan,
                                auprc=round(auprc, 4) if not np.isnan(auprc) else np.nan,
                                spearman_cdkn1a=round(float(rho), 4),
                                spearman_p=float(p),
                            ))

    return pd.DataFrame(rows)


def run_internal_reference(data, b1_wal):
    y = (data["val_y"] > HIGH_RISK_THR).astype(int)
    vi = data["val_idx"]

    rows = []
    for method, score in [
        ("trajrisk", data["trajrisk_walesky"][vi]),
        ("seeder", data["seeder_walesky"][vi]),
        ("b1_xgb", b1_wal[vi]),
    ]:
        auroc, auprc = safe_metrics(y, score)
        rows.append(dict(
            dataset="Walesky",
            test="Walesky: held-out internal",
            label="pseudotime_norm_gt_0.10",
            mode="internal_full_graph",
            k=np.nan,
            alpha=np.nan,
            method=method,
            n_total=len(y),
            n_pos=int(y.sum()),
            prevalence=round(float(y.mean()), 4),
            auroc=round(auroc, 4),
            auprc=round(auprc, 4),
        ))
    return pd.DataFrame(rows)


def summarize(df):
    traj = df[df["method"] == "trajrisk"].copy()
    baseline = df[df["method"].isin(["seeder", "b1_xgb"])].copy()

    summaries = []

    for (dataset, test, label, mode, k, alpha), group in traj.groupby(
        ["dataset", "test", "label", "mode", "k", "alpha"], dropna=False
    ):
        tr_auc = group["auroc"].iloc[0]
        tr_pr = group["auprc"].iloc[0]

        base_sub = baseline[
            (baseline["dataset"] == dataset) &
            (baseline["test"] == test) &
            (baseline["label"] == label) &
            (baseline["mode"] == mode) &
            (baseline["k"].fillna(-999) == (-999 if pd.isna(k) else k)) &
            (baseline["alpha"].fillna(-999) == (-999 if pd.isna(alpha) else alpha))
        ]

        seed = base_sub[base_sub["method"] == "seeder"]
        b1 = base_sub[base_sub["method"] == "b1_xgb"]

        seed_auc = seed["auroc"].iloc[0] if len(seed) else np.nan
        b1_auc = b1["auroc"].iloc[0] if len(b1) else np.nan

        summaries.append(dict(
            dataset=dataset,
            test=test,
            label=label,
            mode=mode,
            k=k,
            alpha=alpha,
            trajrisk_auroc=tr_auc,
            trajrisk_auprc=tr_pr,
            seeder_auroc=seed_auc,
            b1_auroc=b1_auc,
            delta_tr_vs_seed_pp=round((tr_auc - seed_auc) * 100, 2) if not pd.isna(seed_auc) else np.nan,
            delta_tr_vs_b1_pp=round((tr_auc - b1_auc) * 100, 2) if not pd.isna(b1_auc) else np.nan,
        ))

    out = pd.DataFrame(summaries)
    return out.sort_values(["dataset", "label", "delta_tr_vs_seed_pp"], ascending=[True, True, False])


def main():
    t0 = time.time()
    log.info("=" * 72)
    log.info("TrajRisk reference projection diagnostics")
    log.info("=" * 72)

    data = load_data()
    b1, b1_wal = fit_b1(data)

    all_results = [run_internal_reference(data, b1_wal)]

    for dataset in ["Ben_Moshe", "Umbaugh_2h", "Umbaugh_48h72h"]:
        log.info(f"Running diagnostics for {dataset}...")
        all_results.append(run_dataset(data, b1, dataset))

    df = pd.concat(all_results, ignore_index=True)
    summary = summarize(df)

    df.to_csv(RES / "week9_projection_diagnostics_all_results.csv", index=False)
    summary.to_csv(RES / "week9_projection_diagnostics_summary.csv", index=False)

    with open(RES / "week9_projection_diagnostics_config.json", "w") as f:
        json.dump(dict(
            k_values=K_VALUES,
            alphas=ALPHAS,
            projection_modes=PROJECTION_MODES,
            note="Diagnostics only. Do not cherry-pick a single best configuration without reporting full grid.",
            runtime_minutes=round((time.time() - t0) / 60, 2),
        ), f, indent=2)

    log.info("Top diagnostic rows by TrajRisk vs Seeder:")
    cols = ["dataset", "label", "mode", "k", "alpha", "trajrisk_auroc", "seeder_auroc", "delta_tr_vs_seed_pp", "trajrisk_auprc"]
    log.info("\n" + summary[cols].head(20).to_string(index=False))

    log.info(f"Done in {(time.time() - t0)/60:.2f} min.")
    log.info("Saved:")
    log.info("  results/week9_projection_diagnostics_all_results.csv")
    log.info("  results/week9_projection_diagnostics_summary.csv")


if __name__ == "__main__":
    main()
