# Reproduction coverage

The repository supplies the reference implementation, preprocessing sources, machine-readable supplementary data, canonical publication exports and a tested saved-model reproduction workflow.

| Component | Supplied or executed | Additional execution or inputs |
| --- | --- | --- |
| Publication assets | LaTeX source, 12 figure PDFs, 20 table CSV components and Data S1–S5. | Final author/publication metadata when approved. |
| Reference model | Frozen predictions, embeddings and validation metrics replayed. | Clean raw-data training replay. |
| Preprocessing | Ten notebooks and R trajectory source; missing QC annotations and a dense-input sparse-matrix import repaired without changing thresholds. | Execution on complete raw cohort files. |
| Environment | Isolated Linux/Python 3.12 frozen runtime; recorded original Python/Conda and R versions. | Complete training/scVI installation on additional platforms. |
| Figures | S1–S5 regenerated; canonical main figure PDFs supplied. | Final main-figure composite assembly is not automated. |
| Mouse transfer | Three complete treatment endpoints, frozen scores, AUROC/AUPRC and bootstrap CIs. | Expression-derived Cdkn1a labels and refitted B1 require full expression inputs. |
| Human transfer | Both frozen modes, primary/sensitivity donor metrics and strict 13,122-pair gene-axis mapping. | Rebuilding coordinates and discovering signatures require expression data. |
| Graph-derived analyses | Flux/amplification, bottleneck counts, source-quartile membership and distinct-human-gene conservation counts checked. | Complete execution of every table-generating analysis. |
| Inputs | All 135 archived file checksums verified; versioned assets identified in [frozen_inputs.md](frozen_inputs.md). | Full expression matrices and RF/scVI model directories are separate assets. |

Canonical Table S6F displays the upper endpoint as 1.000. Historical input records can contain the earlier padding value 1.010; correcting the display does not change membership or perturbation effects. Historical mouse-score and conservation-summary files are not substituted for corrected replay outputs.

Spatial annotations retain locked statistics and established precision tolerances. Figure replay does not certify exact recomputation of every historical rank test. Recovery of the original DE/pathway generator is outside this pass. The recorded RStudio trajectory environment is R 4.5.1 / Monocle 3 1.4.27; no historical BioMart release is inferred.
