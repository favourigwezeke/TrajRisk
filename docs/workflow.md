# Analysis workflow

TrajRisk uses a Walesky mouse injury–recovery reference, three external mouse cohorts and Matchett human hepatocyte nuclei. The node representation combines 30 scVI coordinates with an XGBoost seed score. Four graph branches produce 64-dimensional embeddings, concatenated into a 256-dimensional representation for score prediction.

## Preprocessing

Follow [notebooks/](../notebooks/README.md) to load the [study datasets](datasets.md), select hepatocytes, integrate mouse cohorts, infer the Walesky trajectory and import pseudotime. Use the analysis dependencies in [environments/](../environments/README.md).

## Training and evaluation

Run these commands from the repository root with expression and model inputs under `Data/processed/` and `models/`. Analyses write to `results/`; training can update processed AnnData files, so use a working copy of the inputs.

| Analysis | Command | Output |
| --- | --- | --- |
| Transition graphs | `python scripts/build_transition_graphs.py` | Four directed transition matrices, graph tensors and reference cell order. |
| Node features and split | `python scripts/prepare_graph_inputs.py` | XGBoost seed, 31-dimensional features, training/validation split and architecture diagnostics. |
| Reference model | `python scripts/train_reference_model.py` | Trained checkpoints, predictions, scale embeddings and evaluation metrics. |
| Baselines and ablations | `python scripts/evaluate_baselines.py` | Expression-only, scalar-propagation and GNN comparisons. |
| Within-timepoint evaluation | `python scripts/analyze_reference_predictions.py` | TrajRisk, seed and XGBoost metrics within sampling-time subsets. |
| External mouse evaluation | `python scripts/validate_mouse_transfer.py` | Reference-KNN, W_self-only and expression-only cohort comparisons. |
| Cross-species evaluation | `python scripts/validate_human_transfer.py` | Shared coordinates, mouse-supervised model and human donor summaries. |

Reference-KNN uses 15 reference neighbours. W_self-only is the primary human transfer configuration; Reference-KNN provides a sensitivity analysis. Human group comparisons use donors as biological units.

## Analyses with supplied checkpoints

The [reproduction guide](reproduction.md) describes prediction, evaluation and supplementary figure generation using the [input archive](frozen_inputs.md). This workflow uses saved coordinates and checkpoints. Full preprocessing and training require the source expression data and additional model inputs.
