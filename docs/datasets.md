# Data and expected inputs

The following identifiers and retained cohort sizes are taken from the canonical manuscript. The original source accessions, file choices, and preprocessing scripts must be linked in the acquisition manifest before the repository is released.

| Cohort | Manuscript source | Retained analysis population |
| --- | --- | ---: |
| Walesky mouse APAP reference | GEO GSE136679 | 6,582 hepatocytes |
| Ben-Moshe mouse external cohort | Zenodo 6035873 | 2,770 hepatocytes |
| Umbaugh early mouse cohort | GEO GSE200771 | 21,409 hepatocytes |
| Umbaugh late mouse cohort | GEO GSE255834 | 5,306 hepatocytes |
| Matchett human cohort | GEO GSE223581, as stated in the manuscript | 49,984 hepatocyte nuclei from 19 donors |

Preprocessing includes the recorded dataset-specific filters, expression normalization, mouse scVI integration, Walesky-only trajectory inference, and the strict shared-symbol filtering used for cross-species integration. See `manuscript/sections/methods.tex` for the locked procedures. The ten preprocessing/integration/trajectory notebooks and R source have been collected. A complete raw-file acquisition manifest and clean raw-data replay remain pending.

Expected saved inputs include processed AnnData objects, the verified ortholog table, reference cell order and split indices, four directed graphs, the frozen XGBoost seed, the final GNN checkpoint, and scVI model directories. Obtain them from the research project without substituting a fresh retraining run for frozen-model replay.

Data S1–S5 and their field definitions are supplied in `manuscript/data/`. Complete expression matrices and model files will require a separate versioned archive with checksums.

The verified Check 2 packet supplies frozen graph/model inputs, result CSVs, and six compact observation/gene-axis objects with mouse and cross-species latent coordinates. It omits expression matrices and large RF/scVI models. `metadata/frozen_input_manifest.json` pins the supported packet; the frozen workflow preserves these inputs and writes separately under `build/`.
