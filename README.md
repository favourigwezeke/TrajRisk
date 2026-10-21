# TrajRisk

**Multi-scale graph neural networks for liver injury and recovery.**

TrajRisk combines treatment-aware, directed transition graphs with single-cell expression features to study hepatocyte injury and recovery. Four graph branches represent acute stress, peak injury, fate decision and resolution. The TrajRisk score estimates normalized Monocle 3 pseudotime in a mouse injury–recovery reference; it is a dimensionless trajectory-state score rather than an adverse-outcome probability.

This repository contains the implementation, preprocessing notebooks, model checkpoints, analysis scripts and supplementary data accompanying the TrajRisk study. Analyses include mouse reference training, external mouse and human projection, comparator and ablation experiments, and graph perturbations.

## Installation

```bash
git clone https://github.com/favourigwezeke/TrajRisk.git
cd TrajRisk
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/frozen-linux-py312.lock.txt
```

These instructions install the Linux CPU environment for analyses using the supplied checkpoints. See [environments](environments/README.md) for training and R dependencies.

## Reproduce the analyses

Download the model inputs with [GitHub CLI](https://cli.github.com/), then run:

```bash
gh release download v0.1.0-rc1 --repo favourigwezeke/TrajRisk \
  --pattern TrajRisk_frozen_inputs.zip --dir build/downloads
bash scripts/reproduce_saved.sh build/downloads/TrajRisk_frozen_inputs.zip
```

The ZIP is also available from the [model-input release](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc1). The command reproduces saved-model predictions, evaluation metrics and Figures S1–S5. Results are written to `build/frozen_replay/` and figures to `build/figures/supplementary/`.

See the [reproduction guide](docs/reproduction.md) for expected outputs and figure/table coverage. For preprocessing and training, follow the [analysis workflow](docs/workflow.md) using the [study datasets](docs/datasets.md).

## Repository contents

| Directory | Contents |
| --- | --- |
| [notebooks/](notebooks/README.md) | Dataset preprocessing, integration and trajectory inference. |
| [scripts/](docs/workflow.md) | Graph construction, training, evaluation and figure generation. |
| [manuscript/](manuscript/) | LaTeX source, figures, supplementary tables and Data S1–S5. |
| [environments/](environments/README.md) | Python and R environment information. |
| [docs/](docs/reproduction.md) | Data descriptions and analysis instructions. |

## Citation

Please cite the accompanying TrajRisk manuscript and identify the code version used. The [input archive](docs/frozen_inputs.md) has a separate version and checksum.

## License and support

See [LICENSE](LICENSE) for licensing terms. Questions and reproducible problems can be submitted through [GitHub Issues](https://github.com/favourigwezeke/TrajRisk/issues).
