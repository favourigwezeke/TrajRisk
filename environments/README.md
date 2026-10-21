# Environments

## Tested frozen-model runtime

`frozen-linux-py312.lock.txt` pins the Linux/Python 3.12 CPU runtime used for saved-model reproduction. Install CPU PyTorch first, then the lock file, as shown in the [README](../README.md). A fresh installation reproduced the 79 numerical checks; supplementary figure generation uses the same environment. This test covers frozen inference and plotting.

## Recorded analysis environment

The root `environment.yml` and `requirements.txt` pin 20 direct analysis dependencies. `pip-observed.txt` preserves 214 installed Python distributions, and `conda-observed.json` records 591 Conda entries with build strings. These snapshots describe the original analysis environment; they are not a solver-generated cross-platform lock. A complete clean installation and raw-data training replay are separate from the tested frozen runtime.

## R trajectory inference

Trajectory inference used R 4.5.1 and Monocle 3 1.4.27 in RStudio. The separate Conda snapshot includes r-base 4.5.3. The RStudio versions identify the trajectory execution environment.
