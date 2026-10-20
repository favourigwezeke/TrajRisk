# Figure sources

`make_figure_s1.py` through `make_figure_s5.py` are the collected supplementary-figure generators. Their calculations and statistical assertions are unchanged. File paths now select saved inputs using `TRAJRISK_ROOT`, use the canonical manuscript table/data CSVs, and write generated outputs beneath this repository's `build/figures/supplementary/`.

The candidate main-figure component scripts in `candidates/` still require alignment with the canonical final figures. Earlier full-figure sources are archived. In particular, an earlier Figure 6 generator still reads the old overall percentage and calls it a mean; it is not the publication generator.

The final PDFs in `manuscript/figures/` are the canonical assets. They are not overwritten by these generators. Numerical replay and plot comparisons remain pending the saved-input packet.
