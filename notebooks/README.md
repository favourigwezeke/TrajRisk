# Preprocessing and trajectory sources

The ten numbered notebooks preserve all 97 collected code cells, with outputs and execution counts removed. Project paths use `TRAJRISK_ROOT` when set or locate this repository from the current working directory. The biological operations and thresholds are unchanged.

| Source | Role |
| --- | --- |
| 01–05 | Dataset loading, source metadata, hepatocyte selection and QC |
| 06 | Cohort summary |
| 07 | Normalization and exploratory clustering |
| 08 | Mouse scVI integration and diagnostics |
| 09 | Export to Monocle 3 |
| monocle3_trajectory.R | Walesky trajectory inference |
| 10 | Pseudotime import and exploratory follow-up |

The notebooks contain exploratory and repeated cells. They are preserved sources, not yet a validated “run all” pipeline. Static review identified missing QC setup in the saved Umbaugh 2 h notebook and a missing `csr_matrix` import on the dense-input branch of the export notebook. A clean raw-data replay must resolve these explicitly rather than silently changing a filter or claiming the source was executed as supplied.

For the R source, run from the repository root using the RStudio/R installation used for trajectory analysis, or pass a separate work directory as its first argument. The R session was not rerun during source preparation.
