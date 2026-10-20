# Reproduction status

| Component | Completed | Remaining |
| --- | --- | --- |
| Publication assets | Active manuscript, 12 figure PDFs, 20 table components and Data S1–S5 retained; S6F CSV endpoint agrees with the PDF | Final author/submission metadata |
| Core model code | All six sources matched the equation audit; frozen reference predictions/embeddings and external mouse/human scores replayed | Clean raw-data training replay |
| Source collection | 47 exported hashes verified; 10 notebooks/97 cells and R trajectory source recovered | Clean raw-data execution of the documented workflow |
| Environment | New isolated Linux/Python 3.12 frozen runtime tested; exact original Python/Conda snapshots retained | Full training/scVI installation and other-platform validation |
| Figures | S1–S5 regenerated from compact metadata/coordinates and saved CSVs; layouts inspected | Final-main-figure generator alignment; exact font rendering depends on installed fonts |
| Inventory | All 135 packet hashes verified; six compact objects present; Data/data alias confirmed on the source laptop | Public versioned archive; full expression, RF and scVI model deposition |
| Mouse transfer | Three complete external cohorts; Reference-KNN, W_self-only and seed AUROC/AUPRC and original bootstrap CIs match Table S3A | Expression-derived Cdkn1a labels and refitted B1 are outside the compact replay |
| Human transfer | Both frozen modes and primary/sensitivity donor score metrics replayed; strict 13,122 symbol pairs and gene-axis order checked | Expression-derived signature discovery is outside the compact replay |
| Derived results | All-edge flux/amplification, bottleneck counts, S6F membership and corrected Figure 6E distinct-human counts replayed | Remaining complete table-generating pipeline |
| Final data files | Complete machine-readable data and definitions supplied | Versioned deposition link |
| Release | Local working draft | License, authors/citation, repository URL, clean run, exact tag and archive |

Run `bash scripts/reproduce_saved.sh /path/to/TrajRisk_Repository_Check_2.zip` for the supported saved-input workflow. Generated files stay under `build/`; model fitting and canonical figure replacement are not part of the command.

The packet's historical `manuscript_mouse_cell_scores.csv`, `week12_task1_scale_summary.csv` and `week12_crossspecies_summary.json` are not corrected publication sources. The replay explicitly exports final mouse scores and distinct-human counts. S6F's historical display padding is normalized from 1.010 to 1.000 without changing membership or effects.

The two original user-requested verification exceptions remain closed: original DE/pathway-settings generator recovery and exact reproduction of six precision-sensitive historical Matchett statistics. S4 retains the locked annotations and existing consistency tolerances; successful plot replay does not certify exact reproduction of those six fields. RStudio R 4.5.1 / Monocle 3 1.4.27 remains user-confirmed; no R rerun or new assertion about the historical BioMart release is made.
