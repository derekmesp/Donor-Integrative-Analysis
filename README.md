# Donor Integrative Analysis

This repository contains flow cytometry and single-cell RNA-seq analyses of donor immune profiles across tissues and demographic groups.

## Repository layout

- `flow_data_scripts/analysis_script.ipynb` — flow cytometry cohort analysis.
- `scRNA-seq-scripts/CZI_analysis.ipynb` — CZI single-cell workflow, including donor-level cell-composition and pseudobulk differential-expression analyses.
- `scRNA-seq-scripts/requirements.txt` — dependencies for the single-cell workflow.
- `src/flow/` — flow data preparation, statistical analyses, and plots.
- `src/scrna/` — reusable single-cell abundance, CMV-sensitivity, and pseudobulk analyses.
- `src/shared/` — shared palettes and analysis utilities.

## Flow analysis setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r flow_data_scripts/requirements.txt
python -m src.flow.data_merge
jupyter lab flow_data_scripts/analysis_script.ipynb
```

The merge script expects the two source workbooks named in its `__main__` block in the repository root and writes `merged_data.xlsx` there.

## Single-cell analysis setup

Use Python 3.11 for the single-cell environment. scCODA depends on TensorFlow Probability, which needs the separate TF-Keras package; the requirements pin a compatible TensorFlow 2.18 / TensorFlow Probability 0.25 set. Create a fresh environment, install dependencies, place the approved `52.h5ad` input in `scRNA-seq-scripts/`, and launch Jupyter from that directory:

```bash
python3.11 -m venv .venv-scrna
source .venv-scrna/bin/activate
python -m pip install -r scRNA-seq-scripts/requirements.txt
cd scRNA-seq-scripts
jupyter lab CZI_analysis.ipynb
```

After changing dependencies, restart the notebook kernel so it uses the updated environment.

The analysis requires donor-level replication for demographic comparisons. Cohort metadata curation has confirmed that records labeled `unknown` in the current CZI input are White; the notebook applies that correction and reports donor counts before testing. Cell-composition analyses use scCODA within tissue, and groups without sufficient donor replication are excluded from contrasts.

## Data handling

Donor-level workbooks, `.h5ad` files, CSV data, and generated plot images are excluded from Git. Keep source and derived donor data within the project’s approved data-sharing process, and clear notebook outputs that contain donor-level information before committing.
