# TrajRisk

**Treatment-aware, multi-scale graph neural networks for studying liver injury and recovery.**

TrajRisk learns directed transition graphs from sampling times, treatment labels and single-cell transcriptomic features. Four graph branches represent acute stress, peak injury, fate decision and resolution. Their embeddings are combined to estimate normalized Monocle 3 pseudotime in the mouse injury–recovery reference. The resulting dimensionless trajectory-state score is the operational meaning of “risk” in TrajRisk; it is not a probability of an adverse clinical outcome.

The analysis includes mouse reference training, external mouse and human reference projection, expression-only comparators, and graph perturbations that nominate candidate recovery-associated transitions.

## Get started

```bash
git clone https://github.com/favourigwezeke/TrajRisk.git
cd TrajRisk
```

For saved-model reproduction on Linux, use Python 3.12 and the tested environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/frozen-linux-py312.lock.txt
```

Download the versioned frozen inputs and reproduce the saved-model analyses and supplementary figures:

```bash
gh release download v0.1.0-rc1 --repo favourigwezeke/TrajRisk \
  --pattern TrajRisk_frozen_inputs.zip --pattern SHA256SUMS --dir build/downloads
bash scripts/reproduce_saved.sh build/downloads/TrajRisk_frozen_inputs.zip
```

The download command uses [GitHub CLI](https://cli.github.com/). Alternatively, download the ZIP from the [input release](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc1) and pass its local filename to the same reproduction command. Input checksums are verified before analysis.

Generated predictions, metrics and the report are written to `build/frozen_replay/`; Figures S1–S5 are written to `build/figures/supplementary/`. See the [reproduction guide](docs/reproduction.md) for outputs, expected checks and supported scope.

## Analysis workflow

To preprocess the study data, train the reference model or run the complete external-cohort analyses, follow the [analysis workflow](docs/workflow.md). It links each stage to its inputs, script and outputs. Full-expression analyses use the [study datasets](docs/datasets.md) and the [analysis environment](environments/README.md).

| Component | Location |
| --- | --- |
| Dataset preprocessing and trajectory inference | [notebooks/](notebooks/README.md) |
| Transition graphs, model training and external validation | [scripts/](docs/workflow.md) |
| Supplementary figure generation | [scripts/figures/](scripts/figures/README.md) |
| Manuscript, canonical figures and supplementary data | [manuscript/](manuscript/) |
| Environment specifications | [environments/](environments/README.md) |
| Data definitions and reproduction instructions | [docs/](docs/reproduction.md) |

## Interpretation and supported use

TrajRisk describes position in an injury–recovery reference. Reference-KNN projects query cells onto that reference; it does not infer a human temporal trajectory. External performance is cohort-dependent. Candidate bottlenecks and recovery gateways are model-derived hypotheses for experimental follow-up. The accompanying manuscript reports the comparator results and biological limitations.

## Citation, license and support

When using TrajRisk, cite the accompanying manuscript and identify the software commit or release used. Frozen input archives have their own version and checksum; the [input guide](docs/frozen_inputs.md) explains how they relate to the code version.

See [LICENSE](LICENSE) for the code's licensing terms. Please report questions and reproducible problems through [GitHub Issues](https://github.com/favourigwezeke/TrajRisk/issues).
