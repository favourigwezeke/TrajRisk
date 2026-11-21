# Study figures

Run the figure workflows with the [study inputs](../../docs/frozen_inputs.md):

```bash
bash scripts/reproduce_saved.sh INPUT_ZIP
```

- `make_figure_s1.py` through `make_figure_s5.py` generate supplementary Figures S1–S5.
- `regenerate_umbaugh_panels.py` rebuilds the corrected Umbaugh portions of Figures 1, 5 and S1 in copies of the manuscript layouts.
- `redraw_main_panels.py` visualizes the data behind Figures 2, 3, 4, 6 and 7.

All generated figures go to `build/figures/`. The manuscript artwork is in `manuscript/figures/`. See [figure and table coverage](../../docs/reproduction.md#figures) for input dependencies and details.
