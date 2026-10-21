# Repository Check 2 review — 20 October 2026

All 135 manifest-listed files verify by byte size and SHA-256. All six compact objects are present; no required files are missing. The source laptop's `Data/` and `data/` paths refer to the same directory. The portable workflow uses `Data/` consistently.

The five Data files and 20 final table components initially match their canonical copies byte for byte. One presentation discrepancy was corrected: Table S6F's CSV now ends its final interval at 1.000, matching the already-correct PDF/caption. Other CSV fields and all remaining publication assets are unchanged. Original uploaded files remain intact.

The supported replay uses the source architecture, frozen checkpoints and seeders on the saved coordinates. It checks reference predictions/embeddings, all three external mouse cohorts, both human modes, primary/sensitivity donor endpoints, graph-derived quantities and corrected distinct-human conservation counts. Mouse AUROC intervals use the original 1,000 resamples and seed 42. B1 is not refitted; unavailable expression labels are not synthesized.

The plotting scripts accept an explicit compact metadata/coordinate reader for S1–S3. S4/S5 use saved CSV inputs. All five plots are generated under `build/figures/supplementary/`; final manuscript figure PDFs are preserved. S5 translates only historical Q4 display padding in memory. The human scale-assignment docstring now describes the existing phase vote and first-encountered tie behavior; executable assignment logic is unchanged.

The frozen workflow is tested in a new isolated Linux/Python 3.12 environment. Its resolved package snapshot is retained. This does not constitute a clean rerun of raw-expression preprocessing, scVI/RF fitting, GNN training or all main-figure assembly. See `reproduction_status.md` for scope and remaining work.
