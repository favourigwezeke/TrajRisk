"""Read plot metadata/coordinates from either an original H5AD or a compact packet.

Compact objects deliberately expose no expression matrix. They are sufficient
for the S1 QC plots, S2 representation plots, and S3 latent-space diagnostics.
"""
from pathlib import Path
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd


def view_path(original, compact_key):
    bundle = os.environ.get("TRAJRISK_BUNDLE")
    if bundle:
        return Path(bundle).expanduser().resolve() / "compact" / compact_key / "metadata.json"
    return original


class CompactView:
    def __init__(self, observations, arrays, n_vars):
        self.obs, self.obsm, self.n_vars = observations, arrays, int(n_vars)
        self.file = SimpleNamespace(close=lambda: None)

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def n_obs(self):
        return len(self.obs)

    def __getitem__(self, rows):
        if isinstance(rows, pd.Series):
            rows = rows.to_numpy()
        positions = np.arange(self.n_obs)[rows]
        return CompactView(self.obs.iloc[positions].copy(),
                           {key: values[positions] for key, values in self.obsm.items()}, self.n_vars)

    def copy(self):
        return CompactView(self.obs.copy(), {key: values.copy() for key, values in self.obsm.items()}, self.n_vars)


def read_view(path, backed=None):
    path = Path(path)
    if path.name != "metadata.json":
        import anndata
        return anndata.read_h5ad(path, backed=backed)
    metadata = json.loads(path.read_text())
    if metadata.get("expression_matrix_exported") is not False:
        raise ValueError("Expected an explicit compact metadata/coordinate object")
    obs = pd.read_csv(path.parent / "observations.csv", index_col=0, float_precision="round_trip")
    arrays = {}
    if (path.parent / "coordinates.npz").exists():
        with np.load(path.parent / "coordinates.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    if len(obs) != metadata["observations"]:
        raise ValueError("Compact observation count mismatch")
    return CompactView(obs, arrays, metadata["genes"])
