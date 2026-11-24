"""Reusable donor-level CMV sensitivity analysis for the scRNA-seq workflow."""

from __future__ import annotations

import math

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sccoda.util.comp_ana import CompositionalAnalysis

from src.shared.palettes import antibody_status_palette, publication_palette


ETHNICITY_GROUPS = ("White", "Hispanic or Latin")
ETHNICITY_PALETTE = {
    "White": publication_palette["White"],
    "Hispanic or Latin": publication_palette["Hispanic/Latino"],
}
CMV_PALETTE = {
    "negative": antibody_status_palette["Negative"],
    "positive": antibody_status_palette["Positive"],
}
CMV_STRATA = (
    ("White", "negative"),
    ("White", "positive"),
    ("Hispanic or Latin", "negative"),
    ("Hispanic or Latin", "positive"),
)


def prepare_cmv_data(adata, sample_metadata, proportions_long):
    """Attach consistent donor-tissue CMV labels to metadata and proportions."""
    cmv_cells = adata.obs[["donor_id", "tissue", "cmv"]].dropna(
        subset=["donor_id", "tissue", "cmv"]
    ).copy()
    cmv_cells["cmv"] = (
        cmv_cells["cmv"].astype("string").str.strip().str.lower()
    )
    conflicts = cmv_cells.groupby(
        ["donor_id", "tissue"], observed=True
    )["cmv"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("A donor-tissue sample has inconsistent CMV labels.")

    cmv_by_sample = cmv_cells.drop_duplicates(["donor_id", "tissue"])
    sample_metadata_cmv = (
        sample_metadata.reset_index()
        .merge(
            cmv_by_sample,
            on=["donor_id", "tissue"],
            how="left",
            validate="one_to_one",
        )
        .set_index(["donor_id", "tissue"])
    )
    proportions_long_cmv = proportions_long.merge(
        cmv_by_sample,
        on=["donor_id", "tissue"],
        how="left",
        validate="many_to_one",
    )
    return sample_metadata_cmv, proportions_long_cmv


def _original_effect_rows(results_by_tissue):
    rows = []
    for tissue, result in results_by_tissue.items():
        effects = result.effect_df.reset_index().rename(
            columns={"Covariate": "covariate", "Cell Type": "cell_type"}
        )
        ethnicity_effects = effects[
            effects["covariate"].astype(str).str.startswith(
                "C(self_reported_ethnicity"
            )
            & effects["covariate"].astype(str).str.contains(
                "[T.Hispanic or Latin]", regex=False
            )
        ]
        for _, effect in ethnicity_effects.iterrows():
            rows.append({
                "tissue": tissue,
                "cell_type": effect["cell_type"],
                "original_log2_fold_change": effect["log2-fold change"],
            })
    return pd.DataFrame(
        rows,
        columns=["tissue", "cell_type", "original_log2_fold_change"],
    )


def fit_cmv_adjusted_models(
    sample_metadata_cmv,
    cell_type_counts,
    original_results_by_tissue,
    credible_findings,
    min_donors_per_group=3,
    alpha=0.05,
):
    """Refit tissue-specific scCODA models with ethnicity, CMV, and sex.

    Returns model results, a compact effect comparison for the original
    credible findings, and per-tissue fit/skip statuses.
    """
    adjusted_results = {}
    status_rows = []
    adjusted_effect_rows = []

    for tissue in sorted(
        sample_metadata_cmv.index.get_level_values("tissue").unique()
    ):
        tissue_meta = sample_metadata_cmv.xs(tissue, level="tissue").copy()
        tissue_counts = cell_type_counts.xs(tissue, level="tissue").copy()
        donor_counts = tissue_meta.groupby(
            "self_reported_ethnicity", observed=True
        ).size()
        eligible_groups = [
            group for group in ETHNICITY_GROUPS
            if donor_counts.get(group, 0) >= min_donors_per_group
        ]
        status = {
            "tissue": tissue,
            "donors_by_ethnicity": donor_counts.to_dict(),
        }
        if len(eligible_groups) < 2:
            status_rows.append({
                **status,
                "status": "Skipped: insufficient donors in both ethnicity groups",
            })
            continue

        tissue_meta = tissue_meta.loc[
            tissue_meta["self_reported_ethnicity"].isin(eligible_groups)
        ].copy()
        if tissue not in original_results_by_tissue:
            status_rows.append({
                **status,
                "status": "Skipped: original ethnicity model did not fit",
            })
            continue
        if tissue_meta["cmv"].isna().any():
            status_rows.append({
                **status,
                "status": "Skipped: CMV missing for an included sample",
            })
            continue
        if tissue_meta["cmv"].nunique() < 2:
            status_rows.append({
                **status,
                "status": "Skipped: CMV status does not vary",
            })
            continue

        tissue_counts = tissue_counts.loc[tissue_meta.index]
        tissue_counts = tissue_counts.loc[:, tissue_counts.sum(axis=0) > 0]
        reference_index = int(
            original_results_by_tissue[tissue].model_specs["reference"]
        )
        if reference_index >= tissue_counts.shape[1]:
            status_rows.append({
                **status,
                "status": "Skipped: original reference cell type is absent",
            })
            continue
        reference_cell_type = tissue_counts.columns[reference_index]

        tissue_meta["self_reported_ethnicity"] = pd.Categorical(
            tissue_meta["self_reported_ethnicity"].astype("string"),
            categories=eligible_groups,
        )
        sex_levels = tissue_meta["sex"].astype("string").dropna().unique().tolist()
        tissue_meta["sex"] = pd.Categorical(
            tissue_meta["sex"].astype("string"), categories=sex_levels
        )
        tissue_meta["cmv"] = pd.Categorical(
            tissue_meta["cmv"], categories=["negative", "positive"]
        )
        sample_names = tissue_meta.index.astype(str)
        tissue_meta.index = sample_names
        tissue_counts.index = sample_names
        coda_data = ad.AnnData(
            X=tissue_counts.to_numpy(dtype=np.int64),
            obs=tissue_meta[["self_reported_ethnicity", "sex", "cmv"]].copy(),
            var=pd.DataFrame(index=tissue_counts.columns.astype(str)),
        )

        formula = (
            "C(self_reported_ethnicity, Treatment('White'))"
            " + C(cmv, Treatment('negative'))"
        )
        if coda_data.obs["sex"].nunique() > 1:
            formula += " + C(sex, Treatment('male'))"

        try:
            model = CompositionalAnalysis(
                coda_data,
                formula=formula,
                reference_cell_type=reference_cell_type,
            )
            result = model.sample_hmc()
            result.set_fdr(est_fdr=alpha)
            adjusted_results[tissue] = result
            effects = result.effect_df.reset_index().rename(
                columns={"Covariate": "covariate", "Cell Type": "cell_type"}
            )
            ethnicity_effects = effects[
                effects["covariate"].astype(str).str.startswith(
                    "C(self_reported_ethnicity"
                )
                & effects["covariate"].astype(str).str.contains(
                    "[T.Hispanic or Latin]", regex=False
                )
            ]
            cmv_effects = effects[
                effects["covariate"].astype(str).str.startswith("C(cmv")
            ].set_index("cell_type")
            for _, effect in ethnicity_effects.iterrows():
                cell_type = effect["cell_type"]
                cmv_effect = cmv_effects.loc[cell_type]
                adjusted_effect_rows.append({
                    "tissue": tissue,
                    "cell_type": cell_type,
                    "cmv_adjusted_credible": effect["Final Parameter"] != 0,
                    "cmv_adjusted_log2_fold_change": effect[
                        "log2-fold change"
                    ],
                    "cmv_effect_credible": cmv_effect["Final Parameter"] != 0,
                    "cmv_log2_fold_change": cmv_effect["log2-fold change"],
                })
            status_rows.append({
                **status,
                "status": "Fit CMV-adjusted scCODA model",
                "donors_by_cmv": tissue_meta["cmv"].value_counts().to_dict(),
                "reference_cell_type": reference_cell_type,
            })
        except Exception as exc:  # retain a per-tissue failure and continue
            status_rows.append({**status, "status": f"Model failed: {exc}"})

    original_hits = credible_findings.drop_duplicates(
        ["tissue", "cell_type"]
    ).copy()
    adjusted_effects = pd.DataFrame(
        adjusted_effect_rows,
        columns=[
            "tissue", "cell_type", "cmv_adjusted_credible",
            "cmv_adjusted_log2_fold_change", "cmv_effect_credible",
            "cmv_log2_fold_change",
        ],
    )
    summary = (
        original_hits[
            ["tissue", "cell_type", "n_white_donors",
             "n_hispanic_or_latin_donors"]
        ]
        .merge(
            _original_effect_rows(original_results_by_tissue),
            on=["tissue", "cell_type"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            adjusted_effects,
            on=["tissue", "cell_type"],
            how="left",
            validate="one_to_one",
        )
    )
    return adjusted_results, summary, pd.DataFrame(status_rows)


def plot_cmv_stratified_proportions(
    proportions_long_cmv, findings, ncols=3, figsize_per_panel=(4.5, 3.7)
):
    """Plot donor proportions by ethnicity and CMV for selected findings."""
    findings = findings.drop_duplicates(["tissue", "cell_type"]).copy()
    if findings.empty:
        print("No original scCODA findings are available to plot.")
        return None

    nrows = math.ceil(len(findings) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols,
                 figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    for ax, row in zip(axes.flat, findings.itertuples(index=False)):
        panel = proportions_long_cmv.loc[
            proportions_long_cmv["tissue"].eq(row.tissue)
            & proportions_long_cmv["cell_type"].eq(row.cell_type)
        ]
        positions, values_by_group, tick_labels = [], [], []
        for position, (ethnicity, cmv) in enumerate(CMV_STRATA, start=1):
            values = panel.loc[
                panel["self_reported_ethnicity"].eq(ethnicity)
                & panel["cmv"].eq(cmv), "proportion"
            ].to_numpy()
            positions.append(position)
            values_by_group.append(values)
            ethnicity_label = (
                "Hispanic/Latin" if ethnicity == "Hispanic or Latin" else ethnicity
            )
            tick_labels.append(
                f"{ethnicity_label}\nCMV {cmv[0].upper()} (n={len(values)})"
            )

        present = [
            (pos, values, cmv)
            for pos, values, (_, cmv) in zip(
                positions, values_by_group, CMV_STRATA
            ) if len(values)
        ]
        if present:
            boxes = ax.boxplot(
                [item[1] for item in present],
                positions=[item[0] for item in present],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#263238", "linewidth": 1.4},
                whiskerprops={"color": "#53636e"},
                capprops={"color": "#53636e"},
            )
            for box, item in zip(boxes["boxes"], present):
                box.set_facecolor(CMV_PALETTE[item[2]])
                box.set_edgecolor("#53636e")
                box.set_alpha(0.8)

        rng = np.random.default_rng(42)
        for position, values, (ethnicity, _) in zip(
            positions, values_by_group, CMV_STRATA
        ):
            if len(values):
                jitter = rng.uniform(-0.08, 0.08, size=len(values))
                ax.scatter(
                    position + jitter,
                    values,
                    s=22,
                    color=ETHNICITY_PALETTE[ethnicity],
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )
        ax.axvline(2.5, color="#8b949e", linewidth=1, alpha=0.7)
        ax.set_xticks(positions, tick_labels, fontsize=7)
        ax.set_title(f"{row.tissue}\n{row.cell_type}", fontsize=10)
        ax.set_ylabel("Donor proportion")
        observed_max = max(
            (float(np.max(values)) for values in values_by_group if len(values)),
            default=0.05,
        )
        ax.set_ylim(0, max(0.05, observed_max) * 1.12)
        ax.grid(axis="y", alpha=0.2)

    for ax in axes.flat[len(findings):]:
        ax.set_visible(False)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=color,
               markeredgecolor="none", markersize=8, label=f"CMV {status}")
        for status, color in CMV_PALETTE.items()
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
               markeredgecolor="none", markersize=7, label=ethnicity)
        for ethnicity, color in ETHNICITY_PALETTE.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("CMV-stratified donor proportions in scCODA findings", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes


def plot_cmv_sensitivity(summary, proportions_long_cmv, ncols=3):
    """Compare original and CMV-adjusted scCODA findings with donor plots."""
    if summary.empty:
        print("No scCODA findings are available for the CMV comparison plot.")
        return None

    unadjusted_color = "#c23b4a"
    adjusted_color = "#7353a6"
    not_credible_color = "#7b8790"
    nrows = math.ceil(len(summary) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False
    )
    for ax, row in zip(axes.flat, summary.itertuples(index=False)):
        panel = proportions_long_cmv.loc[
            proportions_long_cmv["tissue"].eq(row.tissue)
            & proportions_long_cmv["cell_type"].eq(row.cell_type)
        ]
        values_by_group = []
        tick_labels = []
        for ethnicity, cmv in CMV_STRATA:
            values = panel.loc[
                panel["self_reported_ethnicity"].eq(ethnicity)
                & panel["cmv"].eq(cmv), "proportion"
            ].to_numpy()
            values_by_group.append(values)
            ethnicity_label = (
                "Hispanic/Latin" if ethnicity == "Hispanic or Latin" else ethnicity
            )
            tick_labels.append(
                f"{ethnicity_label}\nCMV {cmv[0].upper()} (n={len(values)})"
            )
        present = [
            (position, values, cmv)
            for position, values, (_, cmv) in zip(
                range(1, 5), values_by_group, CMV_STRATA
            ) if len(values)
        ]
        if present:
            boxes = ax.boxplot(
                [item[1] for item in present],
                positions=[item[0] for item in present], widths=0.55,
                patch_artist=True, showfliers=False,
                medianprops={"color": "#263238", "linewidth": 1.4},
                whiskerprops={"color": "#53636e"},
                capprops={"color": "#53636e"},
            )
            for box, item in zip(boxes["boxes"], present):
                box.set_facecolor(CMV_PALETTE[item[2]])
                box.set_edgecolor("#53636e")
                box.set_alpha(0.8)
        rng = np.random.default_rng(42)
        for position, values, (ethnicity, _) in zip(
            range(1, 5), values_by_group, CMV_STRATA
        ):
            if len(values):
                jitter = rng.uniform(-0.08, 0.08, size=len(values))
                ax.scatter(
                    position + jitter, values, s=22,
                    color=ETHNICITY_PALETTE[ethnicity], edgecolor="white",
                    linewidth=0.5, alpha=0.9, zorder=3,
                )
        ax.axvline(2.5, color="#8b949e", linewidth=1, alpha=0.7)
        ax.set_xticks(range(1, 5), tick_labels, fontsize=7)
        ax.set_title(f"{row.tissue}\n{row.cell_type}", fontsize=10)
        ax.set_ylabel("Donor proportion")
        ax.grid(axis="y", alpha=0.2)

        observed_max = max(
            (float(np.max(values)) for values in values_by_group if len(values)),
            default=0.05,
        )
        scale = max(observed_max, 0.05)
        bar_height = 0.035 * scale
        adjusted_credible = (
            pd.notna(row.cmv_adjusted_credible)
            and bool(row.cmv_adjusted_credible)
        )
        for y, color, linestyle in [
            (observed_max + 0.10 * scale, unadjusted_color, "-"),
            (
                observed_max + 0.25 * scale,
                adjusted_color if adjusted_credible
                else not_credible_color,
                "-" if adjusted_credible else "--",
            ),
        ]:
            ax.plot(
                [0.7, 0.7, 4.3, 4.3],
                [y - bar_height, y, y, y - bar_height],
                color=color, linewidth=2, linestyle=linestyle,
            )
        ax.set_ylim(0, observed_max + 0.43 * scale)

    for ax in axes.flat[len(summary):]:
        ax.set_visible(False)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=color,
               markeredgecolor="none", markersize=8, label=f"CMV {status}")
        for status, color in CMV_PALETTE.items()
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
               markeredgecolor="none", markersize=7, label=ethnicity)
        for ethnicity, color in ETHNICITY_PALETTE.items()
    ] + [
        Line2D([0], [0], color=unadjusted_color, lw=2,
               label="Original scCODA credible (5% FDR)"),
        Line2D([0], [0], color=adjusted_color, lw=2,
               label="CMV-adjusted credible"),
        Line2D([0], [0], color=not_credible_color, lw=2, linestyle="--",
               label="Not credible after adjustment"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Ethnicity findings before and after CMV adjustment", y=1.11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes
