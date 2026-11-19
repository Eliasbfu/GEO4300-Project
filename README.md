# GEO4300 — Discharge and flood analysis at Narsjø

Group members: Bryndís Jóna Birgisdóttir, Elias Bjørgum Micaelsen-Fuglesang and Jenny Heggheim Skrivarvik.

Open **Main.ipynb** and run the cells in order. It reproduces the reconstruction comparison, flood-frequency estimates, table values and figures used in the project.

Use Python 3.12 and install the dependencies with `pip install -r requirements.txt`. Start Jupyter with `jupyter lab Main.ipynb`.

- `reconstruction.py` contains data preparation, gap filling and validation functions.
- `eva.py` calculates event samples, exposure, GPD fits and bootstrap intervals.
- `plots.py` produces the figures. `checks.py` contains numerical checks.
- `data/raw` contains the original NVE and SMHI exports. Sources are listed in `data/sources.csv`.
- `data/processed` contains the validated reconstruction and test results. The final 28-year discharge series is **data/narsjo_hourly.csv**.
- `Figures` and `results` contain the figure and numerical exports.

The default run uses saved reconstruction results and recalculates EVA. Set `rerun_reconstruction = True` in the notebook to repeat gap validation and the matched comparison from raw data. This takes longer. The common threshold, 312-hour separation and two-thirds rule are used throughout. Bootstrap intervals condition on the selected events, threshold, rate and reconstructed record.
