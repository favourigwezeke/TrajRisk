# TrajRisk: Multi-Scale Graph Neural Network for Trajectory-Informed Risk Propagation

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.07.XXXXX-b31b1b.svg)](https://biorxiv.org/YOURLINK)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Predicting drug-induced hepatotoxicity through trajectory-aware risk propagation in single-cell transcriptomics.**

## Overview

TrajRisk models cellular toxicity as a dynamic process unfolding through developmental trajectories. Using multi-scale graph neural networks, we propagate risk from early responses to predict late outcomes and identify critical transition bottlenecks.

**Key Features:**
- 🧬 Multi-scale graph construction from cellular trajectories
- 🤖 Learnable transition probabilities via Random Forest
- 🔥 Graph neural network for trajectory-aware risk propagation
- 🎯 Systematic bottleneck analysis via edge perturbation
- 📊 Cross-dataset and human validation

## Installation
```bash
git clone https://github.com/yourusername/TrajRisk.git
cd TrajRisk

conda create -n trajrisk python=3.9
conda activate trajrisk

pip install -r requirements.txt
pip install -e .
```

## Quick Start
```python
import trajrisk as tr
import scanpy as sc

# Load data
adata = sc.read_h5ad("path/to/data.h5ad")

# Build multi-scale graphs
graphs = tr.build_multiscale_graphs(
    adata, 
    pseudotime_key='dpt_pseudotime',
    scales=[0.05, 0.1, 0.2]
)

# Train GNN
model = tr.TrajRiskGNN(input_dim=30, hidden_dim=64)
trainer = tr.Trainer(model, graphs)
trainer.train(epochs=100)

# Predict risk
risk_scores = model.predict(adata)

# Identify bottlenecks
bottlenecks = tr.find_bottlenecks(model, graphs, top_k=10)
```

## Reproducing Paper Results
```bash
# Download data
python scripts/download_data.py

# Run full pipeline
python scripts/run_trajrisk.py --config configs/default.yaml

# Generate figures
python scripts/generate_figures.py
```

## Citation
```bibtex
@article{yourname2025trajrisk,
  title={TrajRisk: Multi-Scale Graph Neural Network Framework for Trajectory-Informed Risk Propagation},
  author={Your Name},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.07.XXXXX}
}
```

## Documentation

See [notebooks](notebooks/) for tutorials and examples.

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

For questions: your.email@university.edu
