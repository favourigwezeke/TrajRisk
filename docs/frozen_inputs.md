# Frozen-input reproduction

Candidate: [v0.1.0-rc2](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc2).

Run these commands from the repository root after installing the documented frozen environment:

```bash
gh release download v0.1.0-rc2 --repo favourigwezeke/TrajRisk --pattern TrajRisk_frozen_inputs.zip --pattern SHA256SUMS --dir build/downloads
(cd build/downloads && shasum -a 256 -c SHA256SUMS)
bash scripts/reproduce_saved.sh build/downloads/TrajRisk_frozen_inputs.zip
```

The input archive contains 135 checksum-verified files. The workflow replays frozen models and regenerates Figures S1–S5. The fresh GitHub Actions result determines whether that run passes.

Main-figure assembly, complete table generation, full raw-data preprocessing/training, and the final submission release remain separate reproducibility tasks.
