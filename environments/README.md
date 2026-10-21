# Software environments

## Saved-model analysis

`frozen-linux-py312.lock.txt` specifies the Linux/Python 3.12 CPU environment used for prediction, numerical evaluation and supplementary figures. Install CPU PyTorch first, then the remaining dependencies as shown in the [README](../README.md). This environment has passed the 79 numerical comparisons in the saved-model workflow.

## Training and preprocessing

The root `environment.yml` and `requirements.txt` specify 20 direct analysis dependencies. `pip-observed.txt` lists the 214 Python distributions in the analysis environment; `conda-observed.json` records its 591 Conda package/build entries. These files document that environment rather than a cross-platform solver lock. Complete raw-data retraining has not been tested through the saved-model command.

## R trajectory inference

The trajectory analysis used R 4.5.1 and Monocle 3 1.4.27 in RStudio. The separate Conda snapshot contains r-base 4.5.3. Use the RStudio versions when matching the trajectory analysis.
