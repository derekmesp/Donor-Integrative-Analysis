# Repository Guidelines

## Project Structure

`flow_data_scripts/analysis_script.ipynb` is the flow cytometry workflow. `scRNA-seq-scripts/CZI_analysis.ipynb` contains the single-cell workflow. Reusable flow modules are in `src/flow/`, single-cell analysis modules are in `src/scrna/`, and shared palettes/helpers are in `src/shared/`. Keep notebook cells focused on configuration and analysis orchestration; put reusable logic in the matching `src/` package.

## Setup and Development

Create a virtual environment, install dependencies, and start the notebook:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r flow_data_scripts/requirements.txt
python -m src.flow.data_merge
jupyter lab flow_data_scripts/analysis_script.ipynb
```

The merge script expects the two source workbooks named in its `__main__` block and writes `merged_data.xlsx` in the repository root. For the single-cell notebook, install `scRNA-seq-scripts/requirements.txt`, place the approved `52.h5ad` file in that folder, and launch Jupyter from the folder. Keep donor-level source and derived data out of Git; ignore rules cover workbook, CSV, H5AD, and generated plot files.

## Code Style

Use four spaces for indentation, descriptive `snake_case` function and variable names, and clear docstrings for public analysis functions. Prefer shared helpers in the workflow packages over duplicating model or plotting logic. Keep notebook cells focused and run them in order after changing analysis parameters.

## Validation

There is no automated test suite yet. For code changes, perform a syntax/import check and run the affected notebook analysis using approved local data. Do not commit notebook outputs that expose donor-level records.

## Commits and Pull Requests

Use short, imperative commit subjects, consistent with the existing history (for example, `Refine demographic analysis`). A pull request should summarize the analysis or behavior change, list the validation performed, link relevant issues when available, and include figures only when they contain no sensitive donor information.
