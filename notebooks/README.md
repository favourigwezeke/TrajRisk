# Preprocessing and trajectory inference

Run notebooks from the repository root or set `TRAJRISK_ROOT` to the project directory. Raw files are read from `Data/raw/`; processed files are written to `Data/processed/`.

| Notebook or script | Analysis |
| --- | --- |
| `01_walesky_loading.ipynb` | Walesky reference loading and hepatocyte selection. |
| `02_umbaugh_2h_loading.ipynb` | Umbaugh early-cohort loading, quality control and filtering. |
| `03_benmoshe_loading.ipynb` | Ben-Moshe cohort preparation. |
| `04_umbaugh_48h72h_loading.ipynb` | Umbaugh later-injury cohort preparation. |
| `05_matchett_human_loading.ipynb` | Matchett human hepatocyte nuclei. |
| `06_cohort_summary.ipynb` | Cohort composition summaries. |
| `07_normalization_clustering.ipynb` | Normalization and exploratory clustering. |
| `08_scvi_integration.ipynb` | Mouse scVI integration and diagnostics. |
| `09_monocle_export.ipynb` | Expression, metadata and coordinate export for Monocle 3. |
| `monocle3_trajectory.R` | Walesky trajectory inference. |
| `10_pseudotime_import.ipynb` | Pseudotime import and downstream summaries. |

Run the R script from the repository root or provide the project directory as its first argument. The trajectory environment is R 4.5.1 with Monocle 3 1.4.27.

These analyses require the expression and annotation files described in [datasets.md](../docs/datasets.md). For prediction and evaluation with the supplied checkpoints and coordinates, use the [reproduction workflow](../docs/reproduction.md).
