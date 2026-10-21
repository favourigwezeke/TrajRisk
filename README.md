# TrajRisk: Multi-Scale Graph Neural Networks for Trajectory-Informed Risk Propagation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](environments/README.md)
[![Code v0.1.0-rc4](https://img.shields.io/badge/Code-v0.1.0--rc4-181717?logo=github)](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc4)
[![Repository checks](https://github.com/favourigwezeke/TrajRisk/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/favourigwezeke/TrajRisk/actions/workflows/validate.yml)
[![Study analyses](https://github.com/favourigwezeke/TrajRisk/actions/workflows/reproduce.yml/badge.svg?branch=main)](https://github.com/favourigwezeke/TrajRisk/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Mapping hepatocyte states and candidate transitions across liver injury and recovery.**

## Overview

TrajRisk learns directed, treatment-aware transition graphs and integrates expression-derived features through a multi-scale graph neural network. It was developed for single-cell studies of acetaminophen-induced liver injury and recovery.

The TrajRisk score estimates normalized Monocle 3 pseudotime in the mouse injury–recovery reference. It is a dimensionless trajectory-state score; it does not estimate the probability of future injury, failure to recover or a clinical outcome.

## Key features

- **Four temporal scales:** acute stress, peak injury, fate decision and resolution, with transition weights learned by Random Forest classifiers.
- **Multi-scale feature integration:** four graph branches combine latent expression features with an expression-derived seed score.
- **Reference projection:** `Reference-KNN` and `W_self-only` inference for external mouse cohorts and shared mouse–human coordinates.
- **Transition analysis:** graph flux, bottleneck identification and edge perturbations to nominate transitions for experimental follow-up.

## Installation

The following installs the Linux CPU environment used to evaluate the supplied model checkpoints and reproduce supplementary figures. It requires Python 3.12 and Git.

```bash
git clone https://github.com/favourigwezeke/TrajRisk.git
cd TrajRisk

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/frozen-linux-py312.lock.txt
```

See [software environments](environments/README.md) for preprocessing, training and R dependencies.

## Quick start

Download the versioned model inputs with [GitHub CLI](https://cli.github.com/), then run the analysis:

```bash
mkdir -p build/downloads
gh release download v0.1.0-rc1 --repo favourigwezeke/TrajRisk \
  --pattern TrajRisk_frozen_inputs.zip --dir build/downloads

bash scripts/reproduce_saved.sh build/downloads/TrajRisk_frozen_inputs.zip
```

Alternatively, download `TrajRisk_frozen_inputs.zip` from the [model-input release](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc1) and pass its path to the same command. Model inputs are versioned separately from the analysis code.

This command computes saved-model predictions and evaluation metrics, performs 79 numerical comparisons and generates Figures S1–S5. Results are written to `build/frozen_replay/` and figures to `build/figures/supplementary/`.

For raw-data preprocessing and model training, follow the [analysis workflow](docs/workflow.md). The [reproduction guide](docs/reproduction.md) specifies expected outputs and coverage for each figure and table.

## Documentation

| Resource | Contents |
| --- | --- |
| [Study datasets](docs/datasets.md) | External data sources and accession information. |
| [Preprocessing notebooks](notebooks/README.md) | Dataset preparation, integration and trajectory inference. |
| [Analysis workflow](docs/workflow.md) | Graph construction, training, evaluation and figure scripts. |
| [Reproduction guide](docs/reproduction.md) | Commands, expected outputs and figure/table coverage. |
| [Supplementary data dictionary](docs/data_dictionary.md) | Fields, units and descriptions for Data S1–S5. |
| [Manuscript](manuscript/) | LaTeX source, figure exports, supplementary tables and data. |

## Citation

When reporting analyses with TrajRisk, cite the accompanying manuscript and specify the code version used. The study implementation is available as [TrajRisk v0.1.0-rc4](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc4).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for the full terms.

## Contact

Email: [favour.igwezeke.249461@unn.edu.ng](mailto:favour.igwezeke.249461@unn.edu.ng).

For reproducible software problems, please open a [GitHub issue](https://github.com/favourigwezeke/TrajRisk/issues).
