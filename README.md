# Donor Integrative Analysis

Python workflow for combining donor metadata with T-cell subset measurements and analyzing demographic associations across tissues. The notebook contains the primary analysis sequence; reusable statistical and plotting functions live in `src/pipeline.py`.

## Repository layout

- `analysis_script.ipynb` — analysis notebook, with saved outputs cleared for review and version control.
- `src/data_merge.py` — merges the assay workbook with the donor metadata workbook.
- `src/pipeline.py` — data summaries, demographic models, tissue interaction analyses, and visualizations.
- `requirements.txt` — Python dependencies for the notebook and analysis modules.

## Setup

Use Python 3.10 or newer. Create an environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the source workbooks expected by `src/data_merge.py` in the repository root, then build the local analysis workbook and open the notebook:

```bash
python src/data_merge.py
jupyter lab analysis_script.ipynb
```

The merge script writes `merged_data.xlsx`; the notebook reads that file. Donor-level workbooks and derived data are local inputs and are excluded from Git. Obtain and handle them through the project’s approved data-sharing process; do not commit donor-level data or outputs containing donor information.

## Analysis workflow

Run notebook cells in order. The notebook standardizes demographic column names, runs tissue-stratified and mixed-effects demographic analyses, then creates summary tables and plots. Several models and plots can take time depending on dataset size. Re-run cells after changing inputs or analysis parameters, and clear large or sensitive outputs before committing.