# Preprocessing and trajectory inference

Run notebooks from the repository root or set `TRAJRISK_ROOT` to the working project directory. Inputs use `Data/raw/` and `Data/processed/`. Distributed sources have notebook outputs and execution counts cleared.

| Source | Analysis |
| --- | --- |
| `01_walesky_loading.ipynb` | Walesky reference loading and selection. |
| `02_umbaugh_2h_loading.ipynb` | Umbaugh early cohort loading, QC annotations and recorded filters. |
| `03_benmoshe_loading.ipynb` | Ben-Moshe cohort preparation. |
| `04_umbaugh_48h72h_loading.ipynb` | Umbaugh later-injury cohort preparation. |
| `05_matchett_human_loading.ipynb` | Matchett human hepatocyte nuclei. |
| `06_cohort_summary.ipynb` | Cohort composition summaries. |
| `07_normalization_clustering.ipynb` | Normalization and exploratory clustering. |
| `08_scvi_integration.ipynb` | Mouse scVI integration and diagnostics. |
| `09_monocle_export.ipynb` | Export for Monocle 3. |
| `monocle3_trajectory.R` | Walesky trajectory inference. |
| `10_pseudotime_import.ipynb` | Pseudotime import and exploratory follow-up. |

The trajectory environment was R 4.5.1 / Monocle 3 1.4.27 in RStudio. Run the R source from the project root or pass its separate work directory as its first argument.

The notebooks preserve exploratory and repeated cells. Raw-data execution requires the full source files described in [datasets.md](../docs/datasets.md). The supported saved-model workflow is documented in [reproduction.md](../docs/reproduction.md).
