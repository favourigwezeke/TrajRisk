#!/usr/bin/env python3
"""
TrajRisk: Week 5 — Learnable Transition Matrices
=================================================
Author: Favour Igwezeke
Project: TrajRisk — Trajectory-Informed Risk Propagation in APAP Hepatotoxicity
Target Journal: Cell Reports Methods

Overview
--------
This script trains one Random Forest (RF) classifier per temporal scale to
learn directed cell-cell transition probabilities from scVI latent space
representations and biological metadata.

The core idea: instead of using a symmetric Gaussian kernel (which treats
all directions equally and ignores biology), we learn WHICH cell-to-cell
transitions are biologically plausible at each phase of the APAP injury
response. The RF learns from the DATA that APAP→APAP transitions are valid,
that periportal-to-pericentral-state transitions happen during functional
compensation, and that some 0h cells have transcriptional profiles that
make them more likely to "become" 6h compensation cells.

Critical Design Decision — Why actual time, not pseudotime, defines direction
------------------------------------------------------------------------------
Our Monocle3 trajectory (Week 4) places 6h APAP cells at the HIGHEST
pseudotime (8.23), while 24h/48h/96h recovering cells fall at LOW pseudotime
(0.25–0.49). This is biologically correct — the trajectory captures
"transcriptional departure from baseline," and 6h cells have departed most.
However, it means "forward in pseudotime" would map to:
    0h → 6h (forward)
    then 6h → 24h (BACKWARD in pseudotime, but biologically forward)
    then 24h → 96h (continuing backward in pseudotime)
This inversion makes pseudotime useless as a directionality signal for
transition matrices. We therefore use ACTUAL BIOLOGICAL TIME (hours) as
the sole ordering principle. Pseudotime is preserved as a node FEATURE —
encoding the degree of functional compensation — but never as a direction.

Four Temporal Scales (calibrated to APAP biology per Walesky et al. 2020)
--------------------------------------------------------------------------
Scale 1: 0h  → 6h   (Δt =  6h) Acute hepatotoxicity — GSH depletion,
                                 pericentral necrosis. The RF learns which
                                 0h hepatocytes are transcriptionally
                                 "predisposed" to the 6h stress state.

Scale 2: 6h  → 24h  (Δt = 18h) Peak injury and compensation onset —
                                 surviving periportal cells begin adopting
                                 pericentral gene expression (Cyp2e1, Glul).
                                 The RF learns which stressed 6h cells
                                 transition into compensation mode.

Scale 3: 24h → 48h  (Δt = 24h) Fate decision — do cells continue
                                 compensating (high pseudotime, high risk)
                                 or begin recovering toward baseline?

Scale 4: 48h → 96h  (Δt = 48h) Resolution — the RF learns which 48h cells
                                 maintain the compensation program to 96h
                                 vs. which successfully return to baseline.

Feature Vector for Each Cell Pair (i → j), 66 dimensions total
---------------------------------------------------------------
Dim  0-29  : scVI latent representation of SOURCE cell i       (30D)
Dim 30-59  : scVI latent representation of TARGET cell j       (30D)
Dim 60     : Δt in hours (actual biological time gap)          (1D)
Dim 61     : Euclidean distance between i and j in scVI space  (1D)
Dim 62     : Treatment of source cell i (0=Control, 1=APAP)   (1D)
Dim 63     : Treatment of target cell j (0=Control, 1=APAP)   (1D)
Dim 64     : Normalized pseudotime of source cell i (0→1)     (1D)
Dim 65     : Normalized pseudotime of target cell j (0→1)     (1D)

The treatment features (dims 62-63) are the key differentiator from a
Gaussian kernel. The RF learns that APAP→APAP is the dominant valid
transition direction, while APAP→Control is biologically anomalous.

Outputs (all paths relative to BASE directory)
----------------------------------------------
models/rf_{ScaleName}.pkl                   — Trained RF per scale
Data/processed/T_{ScaleName}.npz           — Sparse transition matrix
Data/processed/multiscale_graphs.pt        — PyTorch Geometric format
Data/processed/walesky_cell_index.csv      — Global index mapping
Data/processed/adata_integrated...h5ad     — Updated (+pseudotime_norm)
results/week5_transition_summary.json      — Full metrics and statistics
results/figures/week5_*.png               — Validation figures

Usage
-----
    conda activate trajrisk
    python scripts/week5_transition_matrices.py

Expected runtime: 20–60 minutes depending on CPU cores (n_jobs=-1).
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ─── Scientific computing ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy, pearsonr

# ─── Machine learning ─────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
import joblib

# ─── Single-cell analysis ─────────────────────────────────────────────────────
import scanpy as sc

# ─── Deep learning graph format ───────────────────────────────────────────────
import torch

# ─── Plotting ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TrajRisk-W5")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0: CONFIGURATION
# All parameters are centralised here. If any biological assumption changes,
# only this section needs to be updated.
# ═════════════════════════════════════════════════════════════════════════════

# Project root — adjust if directory structure differs
BASE = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE / "Data"  / "processed"
MODEL_DIR = BASE / "models"
FIG_DIR   = BASE / "results" / "figures"
RES_DIR   = BASE / "results"

for _d in [DATA_DIR, MODEL_DIR, FIG_DIR, RES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Temporal scales ──────────────────────────────────────────────────────────
# Each tuple: (source_timepoint, target_timepoint, delta_hours, scale_name)
# Calibrated to known APAP biology (Walesky et al. 2020, Umbaugh et al. 2022)
SCALES: List[Tuple[str, str, int, str]] = [
    ("0h",  "6h",  6,  "Scale1_AcuteStress"),
    ("6h",  "24h", 18, "Scale2_PeakInjury"),
    ("24h", "48h", 24, "Scale3_FateDecision"),
    ("48h", "96h", 48, "Scale4_Resolution"),
]

# Lookup: timepoint label → hours (for ordering and delta verification)
TIMEPOINT_HOURS: Dict[str, int] = {
    "0h": 0, "6h": 6, "24h": 24, "48h": 48, "96h": 96,
}

# ── k-NN for positive transition labeling ────────────────────────────────────
# k=15 matches the n_neighbors=15 used in Monocle3 (Week 4) and the
# standard UMAP/Leiden graph construction in Scanpy (Week 2). Keeping this
# consistent ensures that our transition labels are coherent with the
# trajectory structure already inferred.
K_NEIGHBORS: int = 15

# ── Random Forest hyperparameters ─────────────────────────────────────────────
# n_estimators=200: more trees give better probability calibration and
#   more stable feature importances. 200 is sufficient for 66 features.
# max_depth=15: deep enough to model complex interactions between scVI
#   latent dimensions and biological metadata.
# min_samples_leaf=5: prevents memorizing individual cell pairs;
#   ensures each leaf represents a generalizable pattern.
# class_weight='balanced': automatically up-weights the minority class
#   if negative sampling produces imbalance.
# oob_score=True: gives a free out-of-bag accuracy estimate without
#   any extra computation (used for model quality monitoring).
RF_PARAMS: Dict[str, Any] = {
    "n_estimators":     200,
    "max_depth":        15,
    "max_features":     "sqrt",
    "min_samples_leaf": 5,
    "n_jobs":           -1,
    "random_state":     42,
    "class_weight":     "balanced",
    "oob_score":        True,
}

# ── Transition matrix construction ───────────────────────────────────────────
# RAW_PROB_THRESHOLD is applied to the RF's raw predicted probabilities
# BEFORE any row normalization. This is the critical order of operations.
#
# Why threshold BEFORE normalization?
# With n_target=419 target cells per source cell, row normalization divides
# all probabilities by the total sum (~25 for a typical RF with AUROC=0.99).
# Even a true positive pair scoring 0.85 becomes 0.85/25 = 0.034 after
# normalization — below any reasonable cutoff. We must threshold first,
# keep only high-confidence transitions, then normalize the survivors.
#
# With CV AUROC = 0.9936:
#   RF probability for positive (k-NN) pairs ≈ 0.80–0.95
#   RF probability for negative pairs        ≈ 0.01–0.10
#   RAW_PROB_THRESHOLD = 0.30 cleanly separates them.
RAW_PROB_THRESHOLD: float  = 0.30

# EDGE_PROB_THRESHOLD is applied AFTER row normalization as a final cleanup
# to drop any near-zero residual floating-point entries.
EDGE_PROB_THRESHOLD: float = 0.001

# Negative-to-positive ratio for RF training data.
# 2:1 (neg:pos) keeps training balanced while exposing the RF to enough
# hard negatives (biologically invalid transitions that are close in scVI space).
NEG_POS_RATIO: int = 2

# Source cells are processed in batches during matrix construction to
# avoid loading all (n_source × n_target × 66) features at once.
# 100 rows at a time is safe for any laptop; increase if RAM allows.
BATCH_SIZE_BUILD: int = 100

RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD AND EXTRACT WALESKY DATA
# ═════════════════════════════════════════════════════════════════════════════

def detect_column(obs: pd.DataFrame, candidates: List[str], role: str) -> str:
    """
    Return the first column name from `candidates` that exists in `obs`.
    Raises a clear error if none are found, listing available columns.
    This is defensive: column names can vary depending on how AnnData
    objects were assembled in Weeks 1-3.
    """
    for c in candidates:
        if c in obs.columns:
            return c
    raise ValueError(
        f"Cannot find {role} column. "
        f"Tried: {candidates}. "
        f"Available: {sorted(obs.columns.tolist())}"
    )


def load_walesky_data() -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Load the integrated AnnData and extract the 6,582 Walesky hepatocytes.

    Returns
    -------
    walesky : AnnData
        Walesky cells with scVI latent space (obsm['X_scvi']) and raw
        pseudotime (obs['pseudotime']). Expression matrix is log-normalised.
    cell_index : DataFrame
        One row per cell, indexed by cell barcode. Columns:
          global_idx       — integer 0..n_walesky-1 (position in walesky)
          timepoint_str    — '0h', '6h', '24h', '48h', '96h'
          hours            — integer hours (for ordering)
          treatment_str    — 'Control' or 'APAP'
          treatment_numeric — 0 or 1
          pseudotime       — raw Monocle3 pseudotime (0..8.742)
        This table is the single source of truth for all index lookups
        throughout this script.
    """
    h5ad_path = DATA_DIR / "adata_integrated_with_trajectory.h5ad"
    log.info(f"Loading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    log.info(f"  Total cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

    # Extract Walesky cells (they are the only ones with pseudotime)
    dataset_col = detect_column(adata.obs, ["dataset", "batch", "source"], "dataset")
    walesky_mask = adata.obs[dataset_col] == "Walesky"
    walesky = adata[walesky_mask].copy()
    log.info(f"  Walesky cells: {walesky.n_obs:,}")

    # Verify scVI latent space is present
    if "X_scvi" not in walesky.obsm:
        raise ValueError("X_scvi not found in walesky.obsm. Run Week 3 first.")
    log.info(f"  scVI latent: {walesky.obsm['X_scvi'].shape}")

    # Verify pseudotime is present and fully populated
    pt_col = detect_column(walesky.obs, ["pseudotime", "dpt_pseudotime"], "pseudotime")
    n_missing_pt = walesky.obs[pt_col].isna().sum()
    if n_missing_pt > 0:
        raise ValueError(f"{n_missing_pt} Walesky cells have missing pseudotime.")
    log.info(f"  Pseudotime range: {walesky.obs[pt_col].min():.3f} → {walesky.obs[pt_col].max():.3f}")

    # Detect timepoint and treatment column names defensively
    tp_col  = detect_column(walesky.obs, ["timepoint_str", "timepoint", "time"], "timepoint")
    trt_col = detect_column(walesky.obs,
                            ["treatment_str", "treatment", "condition", "group"], "treatment")

    # Build the global cell index table
    cell_index = pd.DataFrame(index=walesky.obs.index)
    cell_index["global_idx"]       = np.arange(walesky.n_obs, dtype=np.int32)
    cell_index["timepoint_str"]    = walesky.obs[tp_col].values
    cell_index["hours"]            = cell_index["timepoint_str"].map(TIMEPOINT_HOURS)
    cell_index["treatment_str"]    = walesky.obs[trt_col].values
    cell_index["treatment_numeric"] = (cell_index["treatment_str"] == "APAP").astype(np.int32)
    cell_index["pseudotime"]       = walesky.obs[pt_col].values.astype(np.float32)

    # Verify all timepoints are known
    unknown_tps = set(cell_index["timepoint_str"]) - set(TIMEPOINT_HOURS.keys())
    if unknown_tps:
        raise ValueError(f"Unknown timepoint labels: {unknown_tps}. Expected: {set(TIMEPOINT_HOURS)}")

    # Log distribution
    log.info("  Cells per timepoint:")
    for tp in sorted(TIMEPOINT_HOURS, key=TIMEPOINT_HOURS.get):
        n = (cell_index["timepoint_str"] == tp).sum()
        trt = cell_index.loc[cell_index["timepoint_str"] == tp, "treatment_str"].unique()
        log.info(f"    {tp:>4s}: {n:>5,}  treatments={sorted(trt)}")

    return walesky, cell_index


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: PSEUDOTIME NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════

def normalize_pseudotime(
    cell_index: pd.DataFrame,
) -> np.ndarray:
    """
    Normalise raw Monocle3 pseudotime to the [0, 1] interval.

    After normalisation:
      0.0 = 0h control cells (healthy baseline, trajectory root)
      1.0 = 6h APAP cells (maximum transcriptional departure from baseline,
            which Walesky et al. 2020 attributes to functional compensation)
      ~0.02–0.05 = recovering 48h/96h APAP cells (returned to baseline)
      ~0.10–0.78 = 96h cells still in compensation mode (high trajectory risk)

    The normalised values are used as:
      (a) Node features 64-65 in the RF feature vector — encoding where
          each cell sits in the injury-compensation trajectory
      (b) The GNN continuous risk label in Week 7 — pseudotime_norm × APAP
          treatment indicator gives a smooth, biologically grounded risk
          target that the GNN is trained to predict and propagate

    Parameters
    ----------
    cell_index : DataFrame
        Must have a 'pseudotime' column (raw Monocle3 values).

    Returns
    -------
    pt_norm : ndarray of shape (n_walesky,)
        Values in [0, 1], order matching cell_index rows.
    """
    pt_raw = cell_index["pseudotime"].values.astype(np.float64)
    pt_min, pt_max = pt_raw.min(), pt_raw.max()

    if pt_max - pt_min < 1e-8:
        raise ValueError("Pseudotime has no variance — check trajectory inference output.")

    pt_norm = ((pt_raw - pt_min) / (pt_max - pt_min)).astype(np.float32)

    log.info("Pseudotime normalisation:")
    log.info(f"  Raw   range: [{pt_min:.4f}, {pt_max:.4f}]")
    log.info(f"  Normed range: [{pt_norm.min():.4f}, {pt_norm.max():.4f}]")

    cell_index["pseudotime_norm"] = pt_norm

    # Log normalised values per timepoint (biological sanity check)
    log.info("  Normalised pseudotime by timepoint (median | std):")
    for tp in sorted(TIMEPOINT_HOURS, key=TIMEPOINT_HOURS.get):
        mask = cell_index["timepoint_str"] == tp
        vals = pt_norm[mask]
        log.info(f"    {tp:>4s}: median={vals.mean():.4f} | std={vals.std():.4f} "
                 f"| max={vals.max():.4f}")

    return pt_norm


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAINING DATA GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def find_knn_positive_mask(
    scvi_source: np.ndarray,
    scvi_target: np.ndarray,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    Identify positive training pairs via k-nearest-neighbour lookup.

    A pair (source_i → target_j) is POSITIVE if cell j is among the k
    cells in the target timepoint whose scVI representation is closest
    to cell i. The biological rationale: cells at the next timepoint that
    are most transcriptionally similar to a source cell are its most
    plausible biological successors.

    We use k=15 to match the neighbour count used in Monocle3 graph
    construction (Week 4), maintaining consistency across the pipeline.

    Parameters
    ----------
    scvi_source : ndarray (n_source, 30)
    scvi_target : ndarray (n_target, 30)
    k : int

    Returns
    -------
    positive_mask : bool ndarray of shape (n_source * n_target,)
        Flattened in row-major order: index i*n_target + j is True when
        j is a k-NN of source cell i.
    """
    n_source = scvi_source.shape[0]
    n_target = scvi_target.shape[0]
    k_actual = min(k, n_target)  # cap at available targets

    # Fit k-NN on the target pool, query with source cells
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", n_jobs=-1)
    nn.fit(scvi_target)
    _, knn_indices = nn.kneighbors(scvi_source)  # (n_source, k_actual)

    # Mark positive pairs in the flat (n_source * n_target) layout
    positive_mask = np.zeros(n_source * n_target, dtype=bool)
    for i in range(n_source):
        row_start = i * n_target
        positive_mask[row_start + knn_indices[i]] = True

    n_pos = positive_mask.sum()
    log.info(
        f"    k-NN positives: {n_pos:,} / {n_source*n_target:,} pairs "
        f"({n_pos/(n_source*n_target)*100:.2f}%)"
    )
    return positive_mask


def build_feature_matrix_selected(
    scvi_source: np.ndarray,
    scvi_target: np.ndarray,
    delta_hours: int,
    trt_source: np.ndarray,
    trt_target: np.ndarray,
    pt_norm_source: np.ndarray,
    pt_norm_target: np.ndarray,
    src_local_idx: np.ndarray,
    tgt_local_idx: np.ndarray,
) -> np.ndarray:
    """
    Build a 66D feature matrix for a SELECTED subset of cell pairs.

    Unlike building features for ALL pairs (which may not fit in RAM for
    Scale 1 with 3,401 × 419 = 1.4 M pairs), this function builds features
    only for the pairs whose flat indices are given by (src_local_idx,
    tgt_local_idx). This is used both for training data and for the batch
    construction of the full transition matrix.

    Feature layout (66 dimensions):
      [0:30]  scVI of source cell
      [30:60] scVI of target cell
      [60]    Δt in hours
      [61]    Euclidean distance (scVI space)
      [62]    Treatment source (0/1)
      [63]    Treatment target (0/1)
      [64]    Normalised pseudotime of source
      [65]    Normalised pseudotime of target

    Parameters
    ----------
    scvi_source  : (n_source, 30)
    scvi_target  : (n_target, 30)
    delta_hours  : int — same for all pairs in a scale
    trt_source   : (n_source,) int
    trt_target   : (n_target,) int
    pt_norm_source : (n_source,) float32
    pt_norm_target : (n_target,) float32
    src_local_idx  : (n_selected,) int — local (0-based) indices into source
    tgt_local_idx  : (n_selected,) int — local (0-based) indices into target

    Returns
    -------
    X : ndarray (n_selected, 66) float32
    """
    n = len(src_local_idx)
    X = np.zeros((n, 66), dtype=np.float32)

    X[:, 0:30]  = scvi_source[src_local_idx]
    X[:, 30:60] = scvi_target[tgt_local_idx]
    X[:, 60]    = float(delta_hours)

    diff        = X[:, 0:30] - X[:, 30:60]
    X[:, 61]    = np.sqrt((diff ** 2).sum(axis=1))

    X[:, 62]    = trt_source[src_local_idx].astype(np.float32)
    X[:, 63]    = trt_target[tgt_local_idx].astype(np.float32)
    X[:, 64]    = pt_norm_source[src_local_idx]
    X[:, 65]    = pt_norm_target[tgt_local_idx]

    return X


def generate_scale_training_data(
    source_tp: str,
    target_tp: str,
    delta_hours: int,
    walesky_scvi: np.ndarray,
    cell_index: pd.DataFrame,
    pt_norm: np.ndarray,
    neg_pos_ratio: int = NEG_POS_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced training data (features + labels) for one scale.

    Positive examples: k-NN pairs (biologically plausible transitions)
    Negative examples: randomly sampled non-k-NN pairs (implausible),
      sampled at neg_pos_ratio × n_positives.

    The combination of positives and negatives teaches the RF that:
      — Transitions to similar cells (close in scVI) are likely
      — Transitions to very different cells (far in scVI) are unlikely
      — APAP→APAP transitions are more frequent than APAP→Control
      — The treatment and pseudotime features carry additional signal
        beyond raw scVI distances

    Parameters
    ----------
    source_tp, target_tp : str  — e.g., '0h', '6h'
    delta_hours : int
    walesky_scvi : (n_walesky, 30)
    cell_index : DataFrame — with global_idx, timepoint_str, etc.
    pt_norm : (n_walesky,) float32
    neg_pos_ratio : int
    seed : int

    Returns
    -------
    X_train : (n_train, 66) float32
    y_train : (n_train,) int32  — 1=positive, 0=negative
    """
    rng = np.random.default_rng(seed)

    src_mask = cell_index["timepoint_str"] == source_tp
    tgt_mask = cell_index["timepoint_str"] == target_tp

    src_global = cell_index.loc[src_mask, "global_idx"].values  # (n_src,)
    tgt_global = cell_index.loc[tgt_mask, "global_idx"].values  # (n_tgt,)

    n_src = len(src_global)
    n_tgt = len(tgt_global)
    log.info(f"  {source_tp}→{target_tp}: {n_src:,} source × {n_tgt:,} target = {n_src*n_tgt:,} pairs")

    # Pull out the relevant arrays, ordered by local index (0..n_src-1, etc.)
    scvi_src      = walesky_scvi[src_global]          # (n_src, 30)
    scvi_tgt      = walesky_scvi[tgt_global]          # (n_tgt, 30)
    trt_src       = cell_index.iloc[src_global]["treatment_numeric"].values
    trt_tgt       = cell_index.iloc[tgt_global]["treatment_numeric"].values
    pt_src        = pt_norm[src_global]
    pt_tgt        = pt_norm[tgt_global]

    # Identify positive transitions via k-NN
    log.info(f"    Finding k={K_NEIGHBORS} nearest neighbors in scVI space...")
    positive_mask = find_knn_positive_mask(scvi_src, scvi_tgt, k=K_NEIGHBORS)

    # Separate positive and negative flat indices
    pos_flat = np.where( positive_mask)[0]   # (n_pos,)
    neg_flat = np.where(~positive_mask)[0]   # (n_neg_available,)

    n_pos             = len(pos_flat)
    n_neg_to_sample   = min(n_pos * neg_pos_ratio, len(neg_flat))

    log.info(f"    Positives: {n_pos:,} | Sampling negatives: {n_neg_to_sample:,}")

    sampled_neg_flat  = rng.choice(neg_flat, size=n_neg_to_sample, replace=False)

    # Combine and shuffle
    combined_flat     = np.concatenate([pos_flat, sampled_neg_flat])
    y_combined        = np.concatenate([
        np.ones(n_pos, dtype=np.int32),
        np.zeros(n_neg_to_sample, dtype=np.int32)
    ])

    shuffle_idx       = rng.permutation(len(combined_flat))
    combined_flat     = combined_flat[shuffle_idx]
    y_train           = y_combined[shuffle_idx]

    # Recover local source/target indices from the flat pair index
    src_local_idx     = combined_flat // n_tgt
    tgt_local_idx     = combined_flat  % n_tgt

    # Build feature matrix only for the selected pairs (memory-efficient)
    X_train = build_feature_matrix_selected(
        scvi_source=scvi_src,
        scvi_target=scvi_tgt,
        delta_hours=delta_hours,
        trt_source=trt_src,
        trt_target=trt_tgt,
        pt_norm_source=pt_src,
        pt_norm_target=pt_tgt,
        src_local_idx=src_local_idx,
        tgt_local_idx=tgt_local_idx,
    )

    log.info(f"    Training pairs: {len(y_train):,} | "
             f"Positive rate: {y_train.mean()*100:.1f}%")

    return X_train, y_train


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: RANDOM FOREST TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_name: str,
    n_cv_folds: int = 5,
) -> Tuple[RandomForestClassifier, np.ndarray]:
    """
    Train a Random Forest to predict cell-cell transition probabilities.

    We first run 5-fold stratified cross-validation to get an unbiased AUROC
    estimate. This number is reported in the paper's supplementary table and
    is the primary evidence that the RF learned meaningful transition patterns
    rather than memorising training pairs.

    After cross-validation, we retrain on all data to get the final model
    used for transition matrix construction.

    The OOB (out-of-bag) score gives an additional free estimate of accuracy:
    each tree only uses ~63% of data for training, so the remaining 37% serve
    as a validation set. OOB accuracy should be similar to CV accuracy.

    Parameters
    ----------
    X_train : (n_samples, 66)
    y_train : (n_samples,)
    scale_name : str — for logging
    n_cv_folds : int — 5 is standard

    Returns
    -------
    rf       : RandomForestClassifier trained on all X_train, y_train
    cv_auroc : ndarray (n_cv_folds,) — per-fold AUROC scores
    """
    log.info(f"  Training RF: {X_train.shape[0]:,} pairs × {X_train.shape[1]} features")

    # ── Cross-validation (for reporting) ─────────────────────────────────
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=RANDOM_SEED)
    rf_cv = RandomForestClassifier(**RF_PARAMS)

    log.info(f"  Running {n_cv_folds}-fold CV...")
    cv_auroc = cross_val_score(rf_cv, X_train, y_train,
                               cv=cv, scoring="roc_auc", n_jobs=-1)

    log.info(f"  CV AUROC: {cv_auroc.mean():.4f} ± {cv_auroc.std():.4f}")
    log.info(f"  Per-fold: {[f'{v:.4f}' for v in cv_auroc]}")

    if cv_auroc.mean() < 0.70:
        log.warning(
            f"  ⚠ CV AUROC < 0.70 for {scale_name}. "
            "The RF may not have learned strong transition patterns. "
            "Check that scVI latent space separates timepoints."
        )
    elif cv_auroc.mean() >= 0.85:
        log.info(f"  ✓ Strong RF transition learning for {scale_name}.")

    # ── Final model trained on all data ──────────────────────────────────
    log.info("  Training final RF on all data...")
    t0 = time.time()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    log.info(f"  Done in {time.time()-t0:.1f}s | OOB accuracy: {rf.oob_score_:.4f}")

    # Feature importance summary (helps interpret the model in Week 8)
    feature_names = (
        [f"scvi_src_{i}" for i in range(30)] +
        [f"scvi_tgt_{i}" for i in range(30)] +
        ["delta_hours", "euclidean_dist",
         "treatment_src", "treatment_tgt",
         "pseudotime_src", "pseudotime_tgt"]
    )
    importances = rf.feature_importances_
    top5 = np.argsort(importances)[-5:][::-1]

    log.info("  Top-5 feature importances:")
    for idx in top5:
        log.info(f"    [{idx:2d}] {feature_names[idx]:<25} {importances[idx]:.4f}")

    return rf, cv_auroc


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: TRANSITION MATRIX CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def build_transition_matrix(
    source_tp: str,
    target_tp: str,
    delta_hours: int,
    scale_name: str,
    rf_model: RandomForestClassifier,
    walesky_scvi: np.ndarray,
    cell_index: pd.DataFrame,
    pt_norm: np.ndarray,
    batch_size: int = BATCH_SIZE_BUILD,
) -> sparse.csr_matrix:
    """
    Build a sparse, row-stochastic transition matrix T for one temporal scale.

    T[i, j] = probability that source cell i (at source_tp) transitions
              to the transcriptional state of target cell j (at target_tp),
              as learned by the Random Forest.

    Row-stochastic means each source cell's outgoing probabilities sum to 1:
      ∑_j  T[i, j] = 1   for all source cells i

    The matrix uses GLOBAL Walesky cell indices (0..6581), directly
    compatible with the PyTorch Geometric GNN in Week 6.

    Memory management:
    We process source cells in batches of `batch_size` (default 100).
    Peak RAM per batch = batch_size × n_target × 66 × 4 bytes.
    For Scale 1 (n_target=419): ~11 MB per batch — safe on any machine.

    After computing RF probabilities, we:
    1. Normalise each source cell's row to sum to 1.
    2. Drop edges below EDGE_PROB_THRESHOLD (0.05) for sparsity.
    3. Re-normalise after thresholding.

    Entries are zero for:
    — All cell pairs that are NOT (source_tp → target_tp)
    — Terminal cells (96h) which have no outgoing transitions in any scale

    Parameters
    ----------
    source_tp, target_tp : str
    delta_hours : int
    scale_name : str
    rf_model : trained RandomForestClassifier
    walesky_scvi : (n_walesky, 30)
    cell_index : DataFrame
    pt_norm : (n_walesky,) float32
    batch_size : int

    Returns
    -------
    T : scipy.sparse.csr_matrix (n_walesky, n_walesky) float32
    """
    n_walesky = len(cell_index)

    src_mask   = cell_index["timepoint_str"] == source_tp
    tgt_mask   = cell_index["timepoint_str"] == target_tp

    src_global = cell_index.loc[src_mask, "global_idx"].values  # (n_src,)
    tgt_global = cell_index.loc[tgt_mask, "global_idx"].values  # (n_tgt,)

    n_src = len(src_global)
    n_tgt = len(tgt_global)

    log.info(f"  Building T_{scale_name}: {n_src} source × {n_tgt} target")

    # Pre-extract target arrays (fixed across all batches)
    scvi_tgt  = walesky_scvi[tgt_global]                              # (n_tgt, 30)
    trt_tgt   = cell_index.iloc[tgt_global]["treatment_numeric"].values  # (n_tgt,)
    pt_tgt    = pt_norm[tgt_global]                                    # (n_tgt,)

    # Accumulators for sparse matrix COO format
    rows_coo: List[int] = []
    cols_coo: List[int] = []
    vals_coo: List[float] = []

    n_batches = (n_src + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_end   = min(b_start + batch_size, n_src)
        b_local = np.arange(b_start, b_end)         # local indices in src_global
        b_global = src_global[b_local]               # global cell indices

        n_b = len(b_local)

        # Arrays for this batch of source cells
        scvi_src_b = walesky_scvi[b_global]                            # (n_b, 30)
        trt_src_b  = cell_index.iloc[b_global]["treatment_numeric"].values  # (n_b,)
        pt_src_b   = pt_norm[b_global]                                  # (n_b,)

        # Build feature matrix for all (n_b × n_tgt) pairs in one shot
        # Repeat source rows n_tgt times; tile target rows n_b times
        n_pairs_b = n_b * n_tgt
        X_b = np.zeros((n_pairs_b, 66), dtype=np.float32)

        X_b[:, 0:30]  = np.repeat(scvi_src_b, n_tgt, axis=0)
        X_b[:, 30:60] = np.tile(scvi_tgt,     (n_b, 1))
        X_b[:, 60]    = float(delta_hours)

        diff_b       = X_b[:, 0:30] - X_b[:, 30:60]
        X_b[:, 61]   = np.sqrt((diff_b ** 2).sum(axis=1))

        X_b[:, 62]   = np.repeat(trt_src_b, n_tgt).astype(np.float32)
        X_b[:, 63]   = np.tile(trt_tgt,     n_b).astype(np.float32)
        X_b[:, 64]   = np.repeat(pt_src_b,  n_tgt).astype(np.float32)
        X_b[:, 65]   = np.tile(pt_tgt,      n_b).astype(np.float32)

        # RF predicts P(positive transition) for every pair.
        # predict_proba returns (n_pairs, 2); column 1 = P(positive transition).
        probs_b  = rf_model.predict_proba(X_b)[:, 1].astype(np.float64)

        # Reshape to (n_b, n_tgt): one row per source cell in this batch.
        probs_raw = probs_b.reshape(n_b, n_tgt)   # raw RF probabilities, kept for fallback

        # ── CRITICAL: threshold on RAW probabilities BEFORE normalization ──
        # Row-normalizing first is the wrong order. With n_tgt=419, even a
        # strong positive pair (RF prob ≈ 0.85) gets divided by ~25 after
        # normalization → becomes ~0.034, below any useful cutoff.
        # Instead: keep only entries where the RF is already confident
        # (RAW_PROB_THRESHOLD = 0.30), then normalize the survivors.
        probs_2d = probs_raw.copy()
        probs_2d[probs_2d < RAW_PROB_THRESHOLD] = 0.0

        # ── Fallback: if no target exceeded the raw threshold for a source
        # cell, keep its top-K targets by raw probability. This guarantees
        # every source cell has at least one outgoing edge in the graph,
        # which is required for the GNN's message passing to function.
        top_k_fallback = min(K_NEIGHBORS, n_tgt)
        for local_i in range(n_b):
            if probs_2d[local_i].sum() < 1e-10:
                # No target passed the threshold — use top-K raw probabilities
                top_idx = np.argpartition(probs_raw[local_i], -top_k_fallback)[-top_k_fallback:]
                probs_2d[local_i, top_idx] = probs_raw[local_i, top_idx]

        # ── Row-normalise AFTER thresholding ──────────────────────────────
        # Now the surviving entries are the biologically plausible transitions.
        # Normalizing makes each source cell's outgoing probabilities sum to 1,
        # which is required for a proper stochastic transition matrix.
        row_sums = probs_2d.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
        probs_2d = probs_2d / row_sums

        # Final cleanup: drop any residual near-zero floating-point entries
        probs_2d[probs_2d < EDGE_PROB_THRESHOLD] = 0.0

        # ── Accumulate non-zero entries in COO format ─────────────────────
        for local_i in range(n_b):
            global_src_i  = b_global[local_i]
            row_probs     = probs_2d[local_i]      # (n_tgt,)
            nz_mask       = row_probs > 0
            nz_tgt_global = tgt_global[nz_mask]
            nz_probs      = row_probs[nz_mask]

            rows_coo.extend([global_src_i] * len(nz_tgt_global))
            cols_coo.extend(nz_tgt_global.tolist())
            vals_coo.extend(nz_probs.tolist())

        # Progress every 20% of batches
        if (b_idx + 1) % max(1, n_batches // 5) == 0 or b_idx == n_batches - 1:
            log.info(f"    Batch {b_idx+1}/{n_batches} ({b_end}/{n_src} source cells)")

    # Assemble sparse matrix in CSR format
    T = sparse.csr_matrix(
        (np.array(vals_coo, dtype=np.float32),
         (np.array(rows_coo, dtype=np.int32),
          np.array(cols_coo, dtype=np.int32))),
        shape=(n_walesky, n_walesky),
        dtype=np.float32,
    )

    nnz      = T.nnz
    sparsity = 1.0 - nnz / (n_walesky ** 2)
    log.info(f"  T_{scale_name}: {nnz:,} non-zeros | Sparsity: {sparsity*100:.2f}%")

    return T


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_stochasticity(
    T: sparse.csr_matrix,
    src_global: np.ndarray,
    scale_name: str,
) -> Dict:
    """
    Verify that source-cell rows of T are (approximately) stochastic.

    A perfectly stochastic row has sum = 1.0. Due to floating-point
    arithmetic and thresholding, small deviations are expected but
    should be < 1%. This check is important: a non-stochastic matrix
    would cause risk scores to not be properly normalised in the GNN.

    Returns a dict of metrics for the summary report.
    """
    T_src    = T[src_global, :]
    row_sums = np.asarray(T_src.sum(axis=1)).flatten()

    pct_valid  = float((np.abs(row_sums - 1.0) < 0.01).mean() * 100)
    mean_sum   = float(row_sums.mean())
    std_sum    = float(row_sums.std())
    min_sum    = float(row_sums.min())

    log.info(f"  Stochasticity check — {scale_name}:")
    log.info(f"    Row-sum: mean={mean_sum:.5f} | std={std_sum:.6f} | min={min_sum:.5f}")
    log.info(f"    % rows within 1% of 1.0: {pct_valid:.1f}%")

    if pct_valid < 95.0:
        log.warning(
            f"  ⚠ Only {pct_valid:.1f}% of rows are near-stochastic. "
            "Consider lowering EDGE_PROB_THRESHOLD."
        )

    return {
        "row_sum_mean":   mean_sum,
        "row_sum_std":    std_sum,
        "row_sum_min":    min_sum,
        "pct_valid_rows": pct_valid,
    }


def validate_no_backward_transitions(
    T: sparse.csr_matrix,
    src_global: np.ndarray,
    tgt_global: np.ndarray,
    scale_name: str,
) -> int:
    """
    Verify that no transitions cross into non-target timepoints.

    By construction, our RF only predicts probabilities for source → target
    pairs. This check confirms that no non-zero entries appear in wrong
    columns (which would indicate an indexing error).

    Returns the count of backward/off-target transitions (should be 0).
    """
    tgt_set = set(tgt_global.tolist())
    T_coo   = T[src_global, :].tocoo()

    # Adjust column indices: T_coo is relative to T[src_global, :]
    # but we stored global indices directly in the main matrix.
    # So we check the global column (col offset by src_global[0]).
    off_target = 0
    for col in T_coo.col:
        # col here is the column in the sliced matrix; need global col
        # Actually since T was built with global indices, we need global col
        pass
    # Simpler: check non-zero columns of the full T matrix for source rows
    T_src_full = T[src_global, :]
    T_src_coo  = T_src_full.tocoo()
    # Map local col indices back to global indices
    # T_src_coo.col refers to columns of the full n_walesky matrix
    # because T has shape (n_walesky, n_walesky)
    for col_global in T_src_coo.col:
        if col_global not in tgt_set:
            off_target += 1

    log.info(f"  Off-target transitions: {off_target} (should be 0)")
    return off_target


def validate_transition_entropy(
    T: sparse.csr_matrix,
    src_global: np.ndarray,
    scale_name: str,
) -> Dict:
    """
    Compute transition entropy for each source cell.

    High entropy = many equally probable downstream states (diverse fate)
    Low entropy  = one or few dominant downstream states (committed fate)

    Expected biological pattern (Walesky et al. 2020):
    - 0h cells: high entropy (heterogeneous fates — pericentral cells die,
                periportal cells survive with high stress)
    - 6h cells: lower entropy (stress response committed, compensation begins)
    - 24h/48h: mixed (some recovering toward baseline, some compensating)

    Entropy is computed in nats (natural logarithm) per the scipy convention.
    """
    entropies = []
    for gidx in src_global:
        row = np.asarray(T[gidx, :].todense()).flatten()
        row_nz = row[row > 0]
        entropies.append(float(scipy_entropy(row_nz)) if len(row_nz) > 0 else 0.0)

    ent_arr  = np.array(entropies)
    ent_mean = float(ent_arr.mean())
    ent_std  = float(ent_arr.std())

    log.info(f"  Transition entropy ({scale_name}): "
             f"mean={ent_mean:.3f} nats | std={ent_std:.3f}")

    return {"entropy_mean": ent_mean, "entropy_std": ent_std}


def compare_with_gaussian_baseline(
    T_learned: sparse.csr_matrix,
    source_tp: str,
    target_tp: str,
    walesky_scvi: np.ndarray,
    cell_index: pd.DataFrame,
    sigma: float = 1.0,
) -> Dict:
    """
    Compare learned transitions to a simple Gaussian kernel baseline.

    The Gaussian kernel computes transition probability purely from
    Euclidean distance: P_gauss(i→j) ∝ exp(−d²/2σ²). It is symmetric
    and ignores treatment, pseudotime, and temporal direction.

    We measure:
    1. Sparsity — learned transitions should be sparser (more selective)
    2. Row-wise Pearson correlation — low correlation means the RF
       learned DIFFERENT structure from a simple distance-based kernel,
       which is the primary motivation for Week 8 Ablation 1.

    This foreshadows the Week 8 ablation study comparing TrajRisk (learned
    transitions) versus a Gaussian propagation baseline.

    Parameters
    ----------
    T_learned : sparse matrix
    source_tp, target_tp : str
    walesky_scvi : (n_walesky, 30)
    cell_index : DataFrame
    sigma : float — Gaussian bandwidth (in scVI units)

    Returns
    -------
    comparison : dict
    """
    src_mask   = cell_index["timepoint_str"] == source_tp
    tgt_mask   = cell_index["timepoint_str"] == target_tp
    src_global = cell_index.loc[src_mask, "global_idx"].values
    tgt_global = cell_index.loc[tgt_mask, "global_idx"].values

    scvi_src = walesky_scvi[src_global]
    scvi_tgt = walesky_scvi[tgt_global]

    # Gaussian kernel (symmetric, no treatment awareness)
    D_gauss     = cdist(scvi_src, scvi_tgt, metric="euclidean")
    T_gauss     = np.exp(-D_gauss ** 2 / (2.0 * sigma ** 2))
    T_gauss    /= T_gauss.sum(axis=1, keepdims=True)  # row-normalise

    # Extract the same block from the learned matrix
    T_learned_block = np.asarray(
        T_learned[np.ix_(src_global, tgt_global)].todense(), dtype=np.float64
    )

    # Row-wise Pearson correlation between learned and Gaussian
    row_corrs = []
    for i in range(len(src_global)):
        learned_row = T_learned_block[i]
        gauss_row   = T_gauss[i]
        if learned_row.sum() > 0:
            r, _ = pearsonr(learned_row, gauss_row)
            if not np.isnan(r):
                row_corrs.append(r)

    mean_corr = float(np.mean(row_corrs)) if row_corrs else 0.0

    # Sparsity
    learned_sparsity = float((T_learned_block == 0).mean() * 100)
    gauss_sparsity   = float((T_gauss < EDGE_PROB_THRESHOLD).mean() * 100)

    comparison = {
        "learned_sparsity_pct":              learned_sparsity,
        "gaussian_sparsity_pct":             gauss_sparsity,
        "mean_row_pearson_vs_gaussian":      mean_corr,
    }

    log.info(f"  Gaussian comparison ({source_tp}→{target_tp}):")
    log.info(f"    Learned sparsity:   {learned_sparsity:.1f}%  |  "
             f"Gaussian sparsity: {gauss_sparsity:.1f}%")
    log.info(f"    Mean row correlation (learned vs Gaussian): {mean_corr:.3f}")
    log.info(f"    (A correlation < 0.8 indicates the RF learned "
             f"structure beyond distance alone)")

    return comparison


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: PYTORCH GEOMETRIC CONVERSION
# ═════════════════════════════════════════════════════════════════════════════

def matrix_to_pyg_edges(
    T: sparse.csr_matrix,
    scale_name: str,
    n_walesky: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a scipy sparse transition matrix to PyTorch Geometric format.

    PyTorch Geometric represents a directed graph as:
    - edge_index : LongTensor (2, E)
        Row 0 = source node indices
        Row 1 = target node indices
    - edge_weight : FloatTensor (E,)
        Transition probabilities (already row-normalised)

    Only entries above EDGE_PROB_THRESHOLD are included (guaranteed by
    the matrix construction step, but we verify here for safety).

    These tensors will be passed directly to the GNN in Week 6 as the
    multi-scale graph structure.

    Parameters
    ----------
    T : sparse matrix (n_walesky, n_walesky)
    scale_name : str
    n_walesky : int — total Walesky cells (for shape verification)

    Returns
    -------
    edge_index  : LongTensor  (2, E)
    edge_weight : FloatTensor (E,)
    """
    assert T.shape == (n_walesky, n_walesky), (
        f"Unexpected T shape {T.shape}; expected ({n_walesky}, {n_walesky})"
    )

    T_coo   = T.tocoo()
    valid   = T_coo.data > EDGE_PROB_THRESHOLD

    src_arr = T_coo.row[valid].astype(np.int64)
    tgt_arr = T_coo.col[valid].astype(np.int64)
    wgt_arr = T_coo.data[valid].astype(np.float32)

    edge_index  = torch.tensor(np.stack([src_arr, tgt_arr]), dtype=torch.long)
    edge_weight = torch.tensor(wgt_arr, dtype=torch.float32)

    n_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0

    if n_edges == 0:
        log.warning(
            f"  ⚠ PyG — {scale_name}: 0 edges. "
            "Transition matrix is empty. This should not happen after the "
            "raw-probability threshold fix. Check build_transition_matrix output."
        )
    else:
        log.info(
            f"  PyG — {scale_name}: {n_edges:,} edges | "
            f"weight ∈ [{edge_weight.min():.4f}, {edge_weight.max():.4f}]"
        )

    return edge_index, edge_weight


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: VALIDATION FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def make_validation_figures(
    all_results: Dict,
    cell_index: pd.DataFrame,
    pt_norm: np.ndarray,
) -> None:
    """
    Generate a four-panel validation figure saved to FIG_DIR.

    Panel A: RF cross-validation AUROC per scale (± std)
             Primary evidence that the RF learned meaningful patterns.
    Panel B: Sparsity comparison — Learned RF vs Gaussian kernel
             Shows that learned transitions are more selective.
    Panel C: Transition entropy by source timepoint
             Tests biological coherence: early cells should be more diverse.
    Panel D: Normalised pseudotime distribution by timepoint (violin)
             Confirms the TrajRisk signal (within-timepoint heterogeneity).
    """
    scale_names  = [s[3] for s in SCALES]
    scale_labels = [s.replace("_", "\n") for s in scale_names]
    x_pos        = np.arange(len(scale_names))
    colors       = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    tp_order     = ["0h", "6h", "24h", "48h", "96h"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "TrajRisk Week 5 — Transition Matrix Validation",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── A: CV AUROC per scale ─────────────────────────────────────────────
    ax = axes[0, 0]
    auroc_means = [all_results[sn]["cv_auroc_mean"] for sn in scale_names]
    auroc_stds  = [all_results[sn]["cv_auroc_std"]  for sn in scale_names]
    ax.bar(x_pos, auroc_means, yerr=auroc_stds, capsize=5,
           color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(0.80, color="red",  linestyle="--", alpha=0.6, label="0.80")
    ax.axhline(0.70, color="grey", linestyle=":",  alpha=0.6, label="0.70")
    ax.set_xticks(x_pos); ax.set_xticklabels(scale_labels, fontsize=9)
    ax.set_ylabel("5-fold CV AUROC"); ax.set_ylim(0.5, 1.02)
    ax.set_title("A  RF Transition Learning Quality")
    ax.legend(fontsize=8)
    for i, (m, s) in enumerate(zip(auroc_means, auroc_stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=8)

    # ── B: Sparsity comparison ─────────────────────────────────────────────
    ax = axes[0, 1]
    learned_sp = [all_results[sn]["gaussian_comparison"]["learned_sparsity_pct"]
                  for sn in scale_names]
    gauss_sp   = [all_results[sn]["gaussian_comparison"]["gaussian_sparsity_pct"]
                  for sn in scale_names]
    w = 0.35
    ax.bar(x_pos - w/2, learned_sp, w, label="Learned (RF)",     color="#3498db", alpha=0.85)
    ax.bar(x_pos + w/2, gauss_sp,   w, label="Gaussian kernel",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(scale_labels, fontsize=9)
    ax.set_ylabel("Sparsity (% zero entries)")
    ax.set_title("B  Learned vs Gaussian Sparsity")
    ax.legend(fontsize=9)

    # ── C: Transition entropy by source timepoint ──────────────────────────
    ax = axes[1, 0]
    src_tps   = [s[0] for s in SCALES]           # source timepoints
    ent_means = [all_results[sn]["entropy"]["entropy_mean"] for sn in scale_names]
    ent_stds  = [all_results[sn]["entropy"]["entropy_std"]  for sn in scale_names]
    ax.bar(range(len(src_tps)), ent_means, yerr=ent_stds, capsize=5,
           color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(src_tps)))
    ax.set_xticklabels(src_tps, fontsize=10)
    ax.set_xlabel("Source Timepoint")
    ax.set_ylabel("Transition Entropy (nats)")
    ax.set_title("C  Transition Diversity by Source Timepoint")

    # ── D: Pseudotime distribution by timepoint (violin) ──────────────────
    ax = axes[1, 1]
    plot_data = cell_index.copy()
    plot_data = plot_data[plot_data["timepoint_str"].isin(tp_order)]
    plot_data["timepoint_str"] = pd.Categorical(
        plot_data["timepoint_str"], categories=tp_order, ordered=True
    )
    sns.violinplot(
        data=plot_data, x="timepoint_str", y="pseudotime_norm",
        order=tp_order, palette=["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#3498db"],
        ax=ax, inner="box", linewidth=0.8,
    )
    ax.axhline(0.12, color="red", linestyle="--", alpha=0.5,
               label="pseudotime_norm = 0.12")
    ax.set_xlabel("Timepoint"); ax.set_ylabel("Normalised Pseudotime")
    ax.set_title("D  Within-Timepoint Pseudotime Heterogeneity\n"
                 "(TrajRisk risk signal)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "week5_transition_validation.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Figure saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: SAVE ALL OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def save_all_outputs(
    walesky: sc.AnnData,
    pt_norm: np.ndarray,
    cell_index: pd.DataFrame,
    all_rf_models: Dict[str, RandomForestClassifier],
    all_matrices: Dict[str, sparse.csr_matrix],
    all_pyg_graphs: Dict[str, Dict],
    all_results: Dict,
) -> None:
    """
    Persist all Week 5 outputs to disk.

    Output inventory:
    ├── models/
    │   └── rf_{ScaleName}.pkl           — RF model per scale
    ├── Data/processed/
    │   ├── T_{ScaleName}.npz            — Sparse transition matrix per scale
    │   ├── multiscale_graphs.pt         — All 4 scales in PyG format
    │   ├── walesky_cell_index.csv       — Global index↔barcode mapping
    │   └── adata_integrated_with_trajectory.h5ad  — Updated (adds pseudotime_norm)
    └── results/
        └── week5_transition_summary.json — Full metrics and params
    """
    log.info("Persisting outputs to disk...")

    # 1. RF models — one per scale
    for sn, rf in all_rf_models.items():
        path = MODEL_DIR / f"rf_{sn}.pkl"
        joblib.dump(rf, path)
        log.info(f"  Saved RF: {path.name}")

    # 2. Sparse transition matrices
    for sn, T in all_matrices.items():
        path = DATA_DIR / f"T_{sn}.npz"
        sparse.save_npz(str(path), T)
        log.info(f"  Saved T:  {path.name}  ({T.nnz:,} non-zeros)")

    # 3. PyTorch Geometric multi-scale graph dictionary
    pyg_path = DATA_DIR / "multiscale_graphs.pt"
    torch.save(all_pyg_graphs, pyg_path)
    log.info(f"  Saved PyG graphs: {pyg_path.name}")

    # 4. Cell index table (critical for Week 6 node feature assembly)
    idx_path = DATA_DIR / "walesky_cell_index.csv"
    cell_index.to_csv(idx_path, index=True)
    log.info(f"  Saved cell index: {idx_path.name}")

    # 5. Update AnnData with pseudotime_norm (needed as GNN node feature)
    h5ad_path = DATA_DIR / "adata_integrated_with_trajectory.h5ad"
    adata = sc.read_h5ad(h5ad_path)
    dataset_col = detect_column(adata.obs, ["dataset", "batch", "source"], "dataset")
    walesky_mask = adata.obs[dataset_col] == "Walesky"
    adata.obs.loc[walesky_mask, "pseudotime_norm"] = pt_norm
    # Non-Walesky cells get NaN (they don't have pseudotime)
    adata.write_h5ad(h5ad_path)
    log.info(f"  Updated adata: pseudotime_norm added for {walesky_mask.sum():,} Walesky cells")

    # 6. Summary JSON — all metrics for the paper's supplementary table
    # Tensors in all_pyg_graphs are not JSON-serialisable; extract scalars
    json_safe_results = {}
    for sn, r in all_results.items():
        json_safe_results[sn] = {
            k: v for k, v in r.items()
            if not isinstance(v, (np.ndarray, torch.Tensor))
        }

    summary = {
        "week":                  5,
        "objective":             "Learnable Transition Matrices",
        "n_walesky_cells":       int(len(cell_index)),
        "n_temporal_scales":     len(SCALES),
        "k_neighbors":           K_NEIGHBORS,
        "neg_pos_ratio":         NEG_POS_RATIO,
        "edge_prob_threshold":   EDGE_PROB_THRESHOLD,
        "rf_params":             RF_PARAMS,
        "scales":                json_safe_results,
        "pyg_graph_n_edges": {
            sn: int(all_pyg_graphs[sn]["n_edges"])
            for sn in all_pyg_graphs
        },
    }

    summary_path = RES_DIR / "week5_transition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Saved summary: {summary_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 70)
    log.info("TRAJRISK — WEEK 5: LEARNABLE TRANSITION MATRICES")
    log.info("=" * 70)
    log.info(f"Base directory : {BASE}")
    log.info(f"Temporal scales: {len(SCALES)}")
    log.info(f"k-NN threshold : {K_NEIGHBORS}")
    log.info(f"Edge threshold : {EDGE_PROB_THRESHOLD}")
    t_global_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("\n[1/9] Loading Walesky data...")
    walesky, cell_index = load_walesky_data()
    walesky_scvi = walesky.obsm["X_scvi"].astype(np.float32)

    # ── Normalise pseudotime ──────────────────────────────────────────────
    log.info("\n[2/9] Normalising pseudotime to [0, 1]...")
    pt_norm = normalize_pseudotime(cell_index)

    # Storage
    all_rf_models:  Dict[str, RandomForestClassifier] = {}
    all_matrices:   Dict[str, sparse.csr_matrix]      = {}
    all_pyg_graphs: Dict[str, Dict]                    = {}
    all_results:    Dict[str, Dict]                    = {}

    # ── Process each temporal scale ───────────────────────────────────────
    for scale_idx, (source_tp, target_tp, delta_hours, scale_name) in enumerate(SCALES):
        log.info(f"\n{'─'*70}")
        log.info(f"[Scale {scale_idx+1}/4] {scale_name}: {source_tp} → {target_tp}  (Δt={delta_hours}h)")
        log.info(f"{'─'*70}")
        t_scale_start = time.time()

        scale_results: Dict[str, Any] = {
            "source_tp":   source_tp,
            "target_tp":   target_tp,
            "delta_hours": delta_hours,
        }

        # ── Step A: Generate training data ────────────────────────────────
        log.info("[A] Generating training data...")
        X_train, y_train = generate_scale_training_data(
            source_tp=source_tp, target_tp=target_tp,
            delta_hours=delta_hours, walesky_scvi=walesky_scvi,
            cell_index=cell_index, pt_norm=pt_norm,
        )
        scale_results["n_training_pairs"] = int(len(y_train))
        scale_results["positive_rate_pct"] = float(y_train.mean() * 100)

        # ── Step B: Train Random Forest ───────────────────────────────────
        log.info("[B] Training Random Forest...")
        rf, cv_auroc = train_rf_model(X_train, y_train, scale_name)
        scale_results["cv_auroc_mean"] = float(cv_auroc.mean())
        scale_results["cv_auroc_std"]  = float(cv_auroc.std())
        scale_results["cv_auroc_folds"] = cv_auroc.tolist()
        scale_results["oob_score"]      = float(rf.oob_score_)
        all_rf_models[scale_name] = rf

        # Free training data — not needed after RF is trained
        del X_train, y_train

        # ── Step C: Build transition matrix ──────────────────────────────
        log.info("[C] Building transition matrix...")
        T = build_transition_matrix(
            source_tp=source_tp, target_tp=target_tp,
            delta_hours=delta_hours, scale_name=scale_name,
            rf_model=rf, walesky_scvi=walesky_scvi,
            cell_index=cell_index, pt_norm=pt_norm,
        )
        all_matrices[scale_name] = T
        scale_results["nnz"] = int(T.nnz)

        # ── Step D: Validate ──────────────────────────────────────────────
        log.info("[D] Validating...")
        src_mask   = cell_index["timepoint_str"] == source_tp
        tgt_mask   = cell_index["timepoint_str"] == target_tp
        src_global = cell_index.loc[src_mask, "global_idx"].values
        tgt_global = cell_index.loc[tgt_mask, "global_idx"].values

        stoch_metrics   = validate_stochasticity(T, src_global, scale_name)
        n_backward      = validate_no_backward_transitions(T, src_global, tgt_global, scale_name)
        entropy_metrics = validate_transition_entropy(T, src_global, scale_name)
        gauss_compare   = compare_with_gaussian_baseline(
            T, source_tp, target_tp, walesky_scvi, cell_index
        )

        scale_results["stochasticity"]      = stoch_metrics
        scale_results["backward_count"]     = n_backward
        scale_results["entropy"]            = entropy_metrics
        scale_results["gaussian_comparison"] = gauss_compare

        # ── Step E: Convert to PyG format ─────────────────────────────────
        log.info("[E] Converting to PyTorch Geometric format...")
        edge_index, edge_weight = matrix_to_pyg_edges(T, scale_name, walesky.n_obs)

        all_pyg_graphs[scale_name] = {
            "edge_index":  edge_index,
            "edge_weight": edge_weight,
            "source_tp":   source_tp,
            "target_tp":   target_tp,
            "delta_hours": delta_hours,
            "n_nodes":     walesky.n_obs,
            "n_edges":     int(edge_index.shape[1]),
        }

        all_results[scale_name] = scale_results

        t_scale = time.time() - t_scale_start
        log.info(
            f"  ✓ {scale_name} done in {t_scale/60:.1f}min | "
            f"CV AUROC={cv_auroc.mean():.4f} | "
            f"NNZ={T.nnz:,} | "
            f"PyG edges={edge_index.shape[1]:,}"
        )

    # ── Figures ───────────────────────────────────────────────────────────
    log.info("\n[3/9] Generating validation figures...")
    make_validation_figures(all_results, cell_index, pt_norm)

    # ── Save everything ───────────────────────────────────────────────────
    log.info("\n[4/9] Saving all outputs...")
    save_all_outputs(
        walesky=walesky, pt_norm=pt_norm,
        cell_index=cell_index, all_rf_models=all_rf_models,
        all_matrices=all_matrices, all_pyg_graphs=all_pyg_graphs,
        all_results=all_results,
    )

    # ── Final summary ─────────────────────────────────────────────────────
    t_total = time.time() - t_global_start
    log.info("\n" + "=" * 70)
    log.info("WEEK 5 COMPLETE")
    log.info("=" * 70)
    log.info(f"Total runtime: {t_total/60:.1f} minutes")
    log.info("")
    log.info("Per-scale results:")
    log.info(f"  {'Scale':<30} {'CV AUROC':>10} {'OOB':>8} {'NNZ':>10} {'PyG edges':>12}")
    log.info("  " + "─" * 74)
    for sn in all_results:
        r = all_results[sn]
        log.info(
            f"  {sn:<30} "
            f"{r['cv_auroc_mean']:.4f}±{r['cv_auroc_std']:.4f}  "
            f"{r['oob_score']:.4f}  "
            f"{r['nnz']:>10,}  "
            f"{all_pyg_graphs[sn]['n_edges']:>12,}"
        )
    log.info("")
    log.info("What these numbers mean:")
    log.info("  CV AUROC > 0.80 → RF learned meaningful transition patterns")
    log.info("  OOB ≈ CV AUROC  → No overfitting to training pairs")
    log.info("  NNZ << n²       → Transition matrices are appropriately sparse")
    log.info("")
    log.info("Files written:")
    log.info(f"  {MODEL_DIR}/rf_Scale*.pkl")
    log.info(f"  {DATA_DIR}/T_Scale*.npz")
    log.info(f"  {DATA_DIR}/multiscale_graphs.pt")
    log.info(f"  {DATA_DIR}/walesky_cell_index.csv")
    log.info(f"  {DATA_DIR}/adata_integrated_with_trajectory.h5ad  (+ pseudotime_norm)")
    log.info(f"  {RES_DIR}/week5_transition_summary.json")
    log.info(f"  {FIG_DIR}/week5_transition_validation.png")
    log.info("")
    log.info("→ Week 5 outputs are the inputs to Week 6 (GNN Architecture).")
    log.info("  Load multiscale_graphs.pt and walesky_cell_index.csv in Week 6.")


if __name__ == "__main__":
    main()
