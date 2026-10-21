# Model-input archive

The saved-model analyses use the following versioned archive:

| Item | Value |
| --- | --- |
| Release | [v0.1.0-rc1](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc1) |
| File | `TrajRisk_frozen_inputs.zip` |
| SHA-256 | `e02575bdef206e3fdadcb8910fb4bce5f7cc30ba902191eb22db3bb3bec39457` |
| Manifest | `metadata/frozen_input_manifest.json` |
| Files | 135 |

```bash
gh release download v0.1.0-rc1 --repo favourigwezeke/TrajRisk \
  --pattern TrajRisk_frozen_inputs.zip --dir build/downloads
```

Alternatively, download the ZIP from the release page. Private repositories require an account with access.

The archive includes trained checkpoints, directed graphs, reference features and split indices, latent coordinates, cohort annotations, gene axes and result tables. Complete expression matrices and large RF/scVI model directories are separate inputs for preprocessing and training.

The analysis command verifies the archive checksum and all manifest entries before inference. Record both the code version and the input-archive version when reporting results.
