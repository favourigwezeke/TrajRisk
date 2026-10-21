# Analysis workflow

The study uses a Walesky mouse injury–recovery reference, three external mouse cohorts and Matchett human hepatocyte nuclei. Dataset loading and trajectory inference are recorded in the [preprocessing notebooks](../notebooks/README.md). Reference-KNN uses 15 neighbours; the primary human transfer analysis uses W_self-only inference.

## Execution order

Run commands from the repository root with full-expression inputs under `Data/processed/`. Outputs use `models/` and `results/`. Training can update processed AnnData files, so use a working copy of the full inputs when rerunning it.

| Stage | Command | Principal inputs and outputs |
| --- | --- | --- |
| Transition graphs | `python scripts/build_transition_graphs.py` | Reads integrated expression/pseudotime; writes four sparse matrices, graph tensors and reference cell order. |
| Node inputs and reference split | `python scripts/prepare_graph_inputs.py` | Fits the seed, constructs 31-dimensional features, saves the split and runs the original short architecture diagnostic. |
| Reference training | `python scripts/train_reference_model.py` | Reads saved features, graphs and split; writes checkpoints, scores, scale embeddings and summaries. |
| Comparators and ablations | `python scripts/evaluate_baselines.py` | Uses the reference split for expression-only, propagation and GNN comparisons. |
| Sampling-time subsets | `python scripts/analyze_reference_predictions.py` | Evaluates TrajRisk, seed and the expression-only comparator within sampling-time subsets. |
| External mouse transfer | `python scripts/validate_mouse_transfer.py` | Evaluates frozen reference weights and comparators on external cohort expression/latent inputs. |
| Human cross-species analysis | `python scripts/validate_human_transfer.py` | Fits shared coordinates and a mouse-supervised compatible model; summarizes human primary/sensitivity scores over biological donors. |

Node input combines 30 latent coordinates with one initial seed score. Four graph branches produce 64-dimensional embeddings, concatenated into a 256-dimensional representation before the prediction head.

Saved artifact filenames retain the names in the scientific manifests. Source renames are recorded in `metadata/public_source_changes.json`, and original source fingerprints remain in `metadata/workflow_sources.json`.

## Inputs and versions

See [datasets.md](datasets.md) for cohort sources and [environments/README.md](../environments/README.md) for full-analysis dependencies. Full analyses additionally require expression matrices and RF/scVI model files. The compact input archive contains only assets needed for the supported saved-model replay.

Use [reproduction.md](reproduction.md) to replay saved scores, checked metrics, graph-derived results and Figures S1–S5. New training runs are separate experiments and may vary across software/hardware configurations.
