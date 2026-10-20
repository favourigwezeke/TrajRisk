# Environments

The root `environment.yml` and `requirements.txt` pin the observed versions of 20 direct project dependencies. Exact joblib is now recorded as 1.5.3. `pip-observed.txt` preserves all 214 installed Python distributions, and `conda-observed.json` records 591 Conda entries with build strings.

These are collected specifications, not a solver-generated, cross-platform lock or a clean-install test. The original Conda channel URLs were not collected. Recreate and test the environment before claiming independent installation success.

The manuscript's trajectory inference used the previously user-confirmed RStudio session: R 4.5.1 and Monocle 3 1.4.27. The current Conda environment contains r-base 4.5.3. These describe separate execution contexts. Do not replace the manuscript's RStudio version with the Conda snapshot.
