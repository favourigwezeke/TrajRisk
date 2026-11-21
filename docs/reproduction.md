# Reproduce the study analyses

Install the [Linux CPU environment](../README.md#installation), download the [study inputs](frozen_inputs.md), then run:

```bash
bash scripts/reproduce_saved.sh build/downloads/TrajRisk_frozen_inputs.zip
```

The command evaluates the reference model and external cohorts, applies the corrected Umbaugh 2 h treatment labels, and checks 87 numerical results, including mouse AUROC intervals from 1,000 bootstrap samples. It rebuilds the supplementary tables and produces the figures described below. The paper assets remain in `manuscript/`.

| Output directory | Contents |
| --- | --- |
| `build/frozen_replay/` | Reference and external-cohort scores, reference scale embeddings, evaluation metrics and donor summaries. |
| `build/frozen_replay/supplementary_tables/` | 20 CSV components for Tables S1–S6. |
| `build/figures/supplementary/` | Figures S1–S5 in PDF and PNG. |
| `build/figures/corrected/` | Updated Umbaugh panels in the manuscript layouts. |
| `build/figures/redrawn/` | Main data panels redrawn from the underlying results for inspection. |

Read `reproduction_report.json` and `table_reproduction_report.json` under `build/frozen_replay/` for the checks and the input used for each table component. The corrected 2 h cell scores and cohort comparisons are in `umbaugh_2h_scores.csv` and `umbaugh_2h_reanalysis.json`.

## Figures

| Paper item | Regeneration from the study inputs |
| --- | --- |
| Figure 1 | Updates the Umbaugh sampling time in A and B and rebuilds the annotated cohort, time and treatment plots in C. The workflow schematic and the reference pseudotime plot in D are supplied in the manuscript export. |
| Figure 2 | Redraws A–D from reference pseudotime and archived RF diagnostics. |
| Figure 3 | Redraws E–F from reference scores and scale embeddings. A is a schematic; the original epoch-by-epoch curves in B–D are supplied as artwork. |
| Figure 4 | Redraws A–C from the rebuilt baseline, ablation and timepoint tables. |
| Figure 5 | Rebuilds the 2 h AUROC bars and intervals in A and the score distributions, medians and interquartile ranges in B. The Cdkn1a plot in C is supplied in the manuscript export. |
| Figure 6 | Redraws B–E from donor scores, archived gene signatures and distinct human-gene counts. A is a schematic. |
| Figure 7 | Redraws A–F from archived edge scores and pathway results, and the rebuilt enrichment and graph-perturbation statistics. |
| Figures S1–S5 | Generates all five from archived cohort, spatial and integration inputs and the corrected 2 h annotations. |

The figure scripts preserve the analysis values. Redrawn panels use a consistent plotting style; the manuscript PDF exports retain the paper layout.

## Tables

| Paper item | What is rebuilt | What is taken from the archive |
| --- | --- | --- |
| S1 | Graph counts and statistics; summaries of transition-classifier performance. | RF cross-validation and out-of-bag scores, RF–Gaussian correlations and fitting inputs already summarized in the archive. |
| S2A–C | Reference and seed performance, timepoint subsets and all comparison CSVs. | Original comparator and ablation fits and the single-scale training results. |
| S3A–B | External-mouse treatment performance and intervals, including the corrected 2 h expression-only comparator; Cdkn1a summary table. | Expression-only comparator results for the other cohorts and original Cdkn1a statistics. |
| S4A–C | Cross-species silhouette scores, donor means, tests, AUROC and signature concordance. | Cross-species training validation AUROC and donor signature scores. |
| S5A–D | Spatial summaries, including corrected recovery-signature medians and section-level summaries. | Published Matchett rank statistics and p values, which are sensitive to floating-point ties. The replay writes separate rank-test comparisons. |
| S6A–G | Edge summaries, bottleneck counts and enrichment, perturbations by scale and quartile, and edge-shift tests. | Saved graphs and model parameters used to rerun predictions and perturbations. |

The comparison report checks the generated CSVs against the manuscript tables. The [data dictionary](data_dictionary.md) describes Data S1–S5. To refit the study from the source expression datasets, follow the [analysis workflow](workflow.md).

## Version and manuscript

Record the code release or commit and the [model-input release](frozen_inputs.md) used. The model-input archive is shared across code versions; the corrected cohort mapping is applied by the analysis code.

Compile the combined manuscript with `python scripts/build_manuscript.py` in a TeX environment, or compile `manuscript/review_all.tex` in Overleaf. The PDF includes the manuscript figures and supporting information.
