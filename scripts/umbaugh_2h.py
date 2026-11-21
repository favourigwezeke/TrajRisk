"""Apply the GEO GSE200771 annotations to the Umbaugh 2 h cohort."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def sample_key():
    path = Path(__file__).resolve().parents[1] / "metadata/umbaugh_2h_samples.csv"
    return pd.read_csv(path).set_index("accession", verify_integrity=True)


def annotate(obs):
    """Return observations with treatment and sampling time assigned by GSM."""
    key = sample_key()
    ids = obs["sample"] if "sample" in obs else pd.Series(obs.index, index=obs.index)
    accessions = ids.astype(str).str.extract(r"(GSM\d+)", expand=False)
    if accessions.isna().any() or not accessions.isin(key.index).all():
        raise ValueError("Umbaugh 2 h observations contain an unknown GSM accession")
    result = obs.copy()
    result["sample_accession"] = accessions
    result["treatment"] = accessions.map(key.treatment)
    result["timepoint_hours"] = accessions.map(key.sampling_hour)
    result["timepoint_str"] = result.timepoint_hours.astype(str) + "h"
    if "timepoint" in result:
        result["timepoint"] = result.timepoint_str
    return result


def reference_edges(reference, latent, reference_hours, k=15):
    """Use one 0/6 h reference pool for every query, regardless of treatment."""
    anchors = np.flatnonzero(np.isin(np.asarray(reference_hours), [0, 6]))
    if len(anchors) < k:
        raise ValueError("Umbaugh 2 h projection requires at least k reference anchors")
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1)
    distances, indices = nn.fit(reference[anchors]).kneighbors(latent)
    sigma = np.nanmedian(np.where(distances > 0, distances, np.nan), axis=1)
    sigma = np.where(np.isnan(sigma), 1.0, sigma)
    weights = np.exp(-distances**2 / (2.0 * sigma[:, None]**2)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    edges = np.vstack([anchors[indices].ravel(), np.repeat(len(reference) + np.arange(len(latent)), k)])
    return edges, weights.ravel()
