# Study datasets

## Expression cohorts

| Cohort | Data source | Analysis population |
| --- | --- | ---: |
| Walesky mouse reference | [GEO GSE136679](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE136679) | 6,582 hepatocytes |
| Ben-Moshe mouse cohort | [Zenodo 6035873](https://zenodo.org/records/6035873) | 2,770 hepatocytes |
| Umbaugh early mouse cohort | [GEO GSE200771](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200771) | 21,409 hepatocytes |
| Umbaugh later-injury mouse cohort | [GEO GSE255834](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255834) | 5,306 hepatocytes |
| Matchett human cohort | [GEO GSE223581](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223581) | 49,984 hepatocyte nuclei from 19 donors |

Population sizes refer to the processed analysis cohorts. Dataset-specific filtering, normalization and hepatocyte selection are implemented in [notebooks/](../notebooks/README.md) and described in the manuscript Methods. Mouse integration uses scVI; trajectory inference uses Walesky cells. Cross-species analysis uses 13,122 shared mouse–human symbol pairs.

Place raw files under `Data/raw/` in the dataset directories used by the loading notebooks. Processed AnnData files are written to `Data/processed/`. Running these notebooks requires the corresponding expression and annotation files from the source repositories. The distributed inputs for saved-model analysis do not include complete expression matrices.

## Model and analysis inputs

The [input archive](frozen_inputs.md) contains model checkpoints, reference graphs, split indices, model features, latent coordinates, cohort annotations and result tables. These inputs support prediction and evaluation without refitting models. Full preprocessing and training additionally require expression matrices and the RF/scVI model files described in the [workflow](workflow.md).

Stored filenames retain the identifiers used by the model archive so that scripts and input files remain compatible.

## Supplementary data

Data S1–S5 are provided in `manuscript/data/`. The [data dictionary](data_dictionary.md) defines their fields, units, analysis populations and statistical comparisons. Spatial section and zone definitions are described there and in the manuscript Methods.
