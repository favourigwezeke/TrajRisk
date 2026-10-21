# Frozen model inputs

The compact archive contains graph tensors, saved weights, result tables and compact observation/gene-axis/coordinate arrays. Expression matrices and complete RF/scVI model directories are separate full-workflow inputs.

| Field | Value |
| --- | --- |
| Input release | `v0.1.0-rc1` |
| Asset | `TrajRisk_frozen_inputs.zip` |
| SHA-256 | `e02575bdef206e3fdadcb8910fb4bce5f7cc30ba902191eb22db3bb3bec39457` |
| Manifest | `metadata/frozen_input_manifest.json` |
| Verified files | 135 |

```bash
gh release download v0.1.0-rc1 --repo favourigwezeke/TrajRisk \
  --pattern TrajRisk_frozen_inputs.zip --pattern SHA256SUMS --dir build/downloads
```

Alternatively use the [release page](https://github.com/favourigwezeke/TrajRisk/releases/tag/v0.1.0-rc1). A private repository requires existing access; the same links become publicly accessible when the repository is public.

The archive has identical verified bytes to the original packet named `TrajRisk_Repository_Check_2.zip`. Both filenames are accepted. The reproduction code validates the archive digest and every manifest entry before inference.

Record the **code commit or code release separately from the input release**. Reusing these fixed inputs across documentation and source-path changes avoids duplicating the archive or moving earlier release tags.
