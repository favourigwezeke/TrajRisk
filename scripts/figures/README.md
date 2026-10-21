# Figure generation

The active supplementary generators are `make_figure_s1.py` through `make_figure_s5.py`. Run them through `bash scripts/reproduce_saved.sh INPUT_ZIP`, which verifies and resolves the compact metadata, latent coordinates and result CSVs first. Outputs are written beneath `build/figures/supplementary/`.

Canonical publication exports are in `manuscript/figures/`. Older main-figure component sources are retained under `archive/figure_components/` and excluded from the active wrapper because they predate final panel/composite corrections. See [the reproduction map](../../docs/reproduction.md) for exact coverage.
