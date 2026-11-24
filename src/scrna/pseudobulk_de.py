"""Donor-level pseudobulk differential expression by ethnicity."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scanpy as sc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy import sparse
from statsmodels.stats.multitest import multipletests


def analyze_pseudobulk_ethnicity_de(
    adata,
    cell_type,
    min_donors_per_group=3,
    min_cells_per_pseudobulk=20,
    alpha=0.05,
):
    """Fit sex-adjusted pseudobulk DESeq2 models for each tissue.

    Counts are summed per donor, ethnicity, and sex. Donor pseudobulks below
    ``min_cells_per_pseudobulk`` are excluded, and groups are rechecked against
    the donor threshold after that filtering. Results include DESeq2's
    within-contrast adjusted p-value (``padj``) and a BH adjustment across all
    gene/tissue/ethnicity contrasts (``padj_across_tissues``).

    Returns a dictionary with all results, globally significant rows, and
    per-tissue/contrast statuses.
    """
    all_results = []
    status_rows = []

    for tissue in sorted(adata.obs["tissue"].dropna().unique().tolist()):
        subset_mask = (
            adata.obs["MMoCHi_annotation"].eq(cell_type)
            & adata.obs["tissue"].eq(tissue)
            & adata.obs["sex"].notna()
        )
        de_adata = adata[subset_mask].copy()
        if de_adata.n_obs == 0:
            continue

        donor_ethnicity_counts = de_adata.obs.groupby(
            "donor_id", observed=True
        )["self_reported_ethnicity"].nunique()
        if donor_ethnicity_counts.gt(1).any():
            raise ValueError(f"Inconsistent ethnicity labels among donors in {tissue}.")
        donor_sex_counts = de_adata.obs.groupby("donor_id", observed=True)[
            "sex"
        ].nunique()
        if donor_sex_counts.gt(1).any():
            raise ValueError(f"Inconsistent sex labels among donors in {tissue}.")

        donor_counts = (
            de_adata.obs[["donor_id", "self_reported_ethnicity"]]
            .drop_duplicates()
            .groupby("self_reported_ethnicity", observed=True)["donor_id"]
            .nunique()
        )
        eligible_groups = sorted(
            donor_counts[donor_counts >= min_donors_per_group].index.tolist()
        )
        if len(eligible_groups) < 2:
            status_rows.append({
                "tissue": tissue,
                "status": "Skipped: fewer than two groups meet donor threshold",
                "donors_by_ethnicity": donor_counts.to_dict(),
            })
            continue

        group_columns = ["donor_id", "self_reported_ethnicity", "sex"]
        pseudobulk_input = de_adata[
            de_adata.obs["self_reported_ethnicity"].isin(eligible_groups)
        ].copy()
        pseudobulk = sc.get.aggregate(
            pseudobulk_input,
            by=group_columns,
            func="sum",
            layer="counts",
        )
        cell_counts = pseudobulk_input.obs.groupby(
            group_columns, observed=True
        ).size()
        pseudobulk_groups = pd.MultiIndex.from_frame(
            pseudobulk.obs[group_columns]
        )
        pseudobulk.obs["n_obs_aggregated"] = cell_counts.reindex(
            pseudobulk_groups
        ).to_numpy()
        pseudobulk = pseudobulk[
            pseudobulk.obs["n_obs_aggregated"] >= min_cells_per_pseudobulk
        ].copy()
        donors_after_qc = (
            pseudobulk.obs.groupby("self_reported_ethnicity", observed=True)[
                "donor_id"
            ].nunique()
        )
        eligible_groups = sorted(
            donors_after_qc[
                donors_after_qc >= min_donors_per_group
            ].index.tolist()
        )
        if len(eligible_groups) < 2:
            status_rows.append({
                "tissue": tissue,
                "status": (
                    "Skipped: fewer than two groups meet donor threshold "
                    "after cell QC"
                ),
                "donors_by_ethnicity": donors_after_qc.to_dict(),
            })
            continue

        pseudobulk = pseudobulk[
            pseudobulk.obs["self_reported_ethnicity"].isin(eligible_groups)
        ].copy()
        sample_ids = pseudobulk.obs["donor_id"].astype(str).to_numpy()
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"Expected one pseudobulk sample per donor in {tissue}.")

        counts_matrix = pseudobulk.layers["sum"]
        if sparse.issparse(counts_matrix):
            counts_matrix = counts_matrix.toarray()
        counts = pd.DataFrame(
            counts_matrix, index=sample_ids, columns=pseudobulk.var_names
        ).astype("int64")
        metadata = pseudobulk.obs.set_index("donor_id")[
            ["self_reported_ethnicity", "sex"]
        ].copy()
        metadata.index = metadata.index.astype(str)
        metadata["self_reported_ethnicity"] = pd.Categorical(
            metadata["self_reported_ethnicity"], categories=eligible_groups
        )
        metadata["sex"] = metadata["sex"].astype("category")

        try:
            dds = DeseqDataSet(
                counts=counts,
                metadata=metadata,
                design="~ sex + self_reported_ethnicity",
                quiet=True,
            )
            dds.deseq2()
        except Exception as exc:
            status_rows.append({
                "tissue": tissue,
                "status": f"Skipped: model fit failed ({exc})",
                "donors_by_ethnicity": donors_after_qc.to_dict(),
            })
            continue

        tissue_test_count = 0
        for focal_group, reference_group in combinations(eligible_groups, 2):
            try:
                de_stats = DeseqStats(
                    dds,
                    contrast=[
                        "self_reported_ethnicity", focal_group, reference_group
                    ],
                    alpha=alpha,
                    quiet=True,
                )
                de_stats.summary()
            except Exception as exc:
                status_rows.append({
                    "tissue": tissue,
                    "status": (
                        f"Skipped contrast {focal_group} vs "
                        f"{reference_group}: {exc}"
                    ),
                    "donors_by_ethnicity": donors_after_qc.to_dict(),
                })
                continue

            result = de_stats.results_df.rename_axis("gene").reset_index()
            result["focal_ethnicity"] = focal_group
            result["reference_ethnicity"] = reference_group
            result["cell_type"] = cell_type
            result["tissue"] = tissue
            result["n_focal_donors"] = int(donors_after_qc[focal_group])
            result["n_reference_donors"] = int(donors_after_qc[reference_group])
            all_results.append(result)
            tissue_test_count += 1

        status_rows.append({
            "tissue": tissue,
            "status": f"Fit {tissue_test_count} ethnicity contrast(s)",
            "donors_by_ethnicity": donors_after_qc.to_dict(),
        })

    if all_results:
        results = pd.concat(all_results, ignore_index=True)
        results["padj_across_tissues"] = np.nan
        valid_p = results["pvalue"].notna()
        results.loc[valid_p, "padj_across_tissues"] = multipletests(
            results.loc[valid_p, "pvalue"], method="fdr_bh"
        )[1]
        significant = results.loc[
            results["padj_across_tissues"] < alpha
        ].sort_values(["tissue", "padj_across_tissues"])
    else:
        results = pd.DataFrame()
        significant = pd.DataFrame()

    return {
        "results": results,
        "significant": significant,
        "status": pd.DataFrame(status_rows),
    }
