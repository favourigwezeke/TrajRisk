# TrajRisk

TrajRisk learns treatment-aware, directed transition graphs at four temporal scales and integrates expression-derived features with a graph neural network. Its dimensionless score estimates normalized Monocle 3 pseudotime in a mouse injury–recovery reference.

## Current contents

This repository draft contains the six audited core analysis scripts, preprocessing notebooks and R trajectory source, supplementary-figure generators, the active LaTeX manuscript, Figures 1–7 and S1–S5, 20 table CSV components, and Data S1–S5. Portable scripts use relative input paths and isolated build outputs. Source provenance is in `metadata/workflow_sources.json`.

## Reproduce saved-model results

Download the versioned frozen inputs using [the frozen-input reproduction instructions](docs/frozen_inputs.md). Input checksums are recorded in `metadata/frozen_input_manifest.json`.

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-frozen.txt
bash scripts/reproduce_saved.sh build/downloads/TrajRisk_frozen_inputs.zip
```

An existing environment with these dependencies can run the script directly. On Linux, CPU-only PyTorch can be installed first with `python3 -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu`.

The command verifies the supplied assets, replays frozen reference and external mouse/human models, checks published metrics and derived results, and regenerates Figures S1–S5. Predictions, metrics and a JSON report go to `build/frozen_replay/`; plots go to `build/figures/supplementary/`. Canonical manuscript assets are preserved. For inference alone, use `python3 scripts/reproduce_frozen.py --bundle build/downloads/TrajRisk_frozen_inputs.zip --bootstrap-ci`.

## Check the supplied files

From this directory, run:

```bash
python scripts/verify_release_assets.py
```

This checks file hashes, citation matching, equation count, and the supplied CSV structures. It does not rerun models or regenerate plots.

## Compile the combined review PDF

Upload `manuscript/` to Overleaf, select `review_all.tex`, and use pdfLaTeX. Alternatively, run `python scripts/build_manuscript.py` with pdfLaTeX installed. The combined review is the canonical reading version and includes the supplement. Its scientific sources retain the approved 178-word abstract, 126 references, and 26 numbered equations.

## Environment

The frozen workflow has been tested in a new Linux/Python 3.12 environment. Its resolved package snapshot is `environments/frozen-linux-py312.lock.txt`. Plot typography follows available fonts; the generators request Arial. The root `environment.yml` and `requirements.txt` record the broader project dependencies; full observed Python/Conda snapshots are in `environments/`. A clean installation of the full training environment remains to be tested.

## Analysis workflow

1. Acquire and preprocess the datasets described in `docs/datasets.md`.
2. Integrate expression data with scVI and derive Walesky Monocle 3 pseudotime.
3. Construct temporal transition matrices with `week5_transition_matrices.py`.
4. Build the graph inputs with `week6_gnn_architecture.py` and train with `week7_full_training.py`.
5. Evaluate baselines with `week8_baselines_ablations.py`.
6. Evaluate mouse and human transfer with `week9a_mouse_cross_dataset_final.py` and `week9b_human_cross_species.py`.
7. Use the supported frozen workflow for supplementary plots and checked endpoints. Complete final main-figure assembly and remaining table reproduction before claiming end-to-end reproduction.

Training scripts expect `Data/processed/`, `models/`, and `results/` beneath the project root. Full expression, RF and scVI model archives remain separate inputs. The user-confirmed trajectory environment is RStudio R 4.5.1 / Monocle 3 1.4.27. `docs/reproduction_status.md` records the tested scope and remaining work.

## Interpretation

The score describes a model-derived trajectory state. Reference evaluation is transductive, external performance varies across cohorts, and the candidate recovery gateway is a hypothesis for experimental testing. The manuscript reports expression-only comparators and null signature/spatial findings.

## Before public release

Complete final main-figure generator alignment, remaining table/raw-data reproduction, and versioned data/model deposition. Add the author-approved code license, citation metadata, repository URL and release identifier. This repository supports the TrajRisk preprint workflow. PyPI packaging can follow the preprint.
