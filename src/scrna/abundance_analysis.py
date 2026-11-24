"""Donor-level cell-type abundance analysis for the scRNA-seq workflow."""

from __future__ import annotations

import math

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sccoda.util.comp_ana import CompositionalAnalysis

from src.shared.palettes import publication_palette


COMPARISON_GROUPS = ("White", "Hispanic or Latin")
ETHNICITY_PALETTE = {
    "White": publication_palette["White"],
    "Hispanic or Latin": publication_palette["Hispanic/Latino"],
}
SC_CODA_COLOR = "#c23b4a"


def analyze_donor_cell_type_abundance(
    adata,
    min_cells_per_sample=100,
    min_donors_per_group=3,
    alpha=0.05,
):
    """Build donor-tissue proportions and fit ethnicity-adjusted scCODA models.

    The ethnicity contrast is Hispanic/Latin versus White, adjusted for sex
    when both sex categories occur in a tissue. The returned dictionary
    contains sample-level data, summaries, models, and per-tissue statuses.
    """
    cell_metadata = adata.obs[
        ["donor_id", "tissue", "self_reported_ethnicity", "sex",
         "MMoCHi_annotation"]
    ].dropna().copy()
    metadata_conflicts = (
        cell_metadata.groupby(["donor_id", "tissue"], observed=True)[
            ["self_reported_ethnicity", "sex"]
        ]
        .nunique()
        .gt(1)
        .any(axis=1)
    )
    if metadata_conflicts.any():
        raise ValueError("A donor-tissue sample has inconsistent demographic labels.")

    cell_type_counts = pd.crosstab(
        index=[cell_metadata["donor_id"], cell_metadata["tissue"]],
        columns=cell_metadata["MMoCHi_annotation"],
    )
    cell_type_counts.columns = cell_type_counts.columns.astype(str)
    sample_metadata = (
        cell_metadata[
            ["donor_id", "tissue", "self_reported_ethnicity", "sex"]
        ]
        .drop_duplicates(["donor_id", "tissue"])
        .set_index(["donor_id", "tissue"])
    )
    cell_type_counts = cell_type_counts.reindex(sample_metadata.index).fillna(0).astype(int)
    sample_metadata["n_cells"] = cell_type_counts.sum(axis=1)
    keep_samples = sample_metadata["n_cells"] >= min_cells_per_sample
    cell_type_counts = cell_type_counts.loc[keep_samples]
    sample_metadata = sample_metadata.loc[keep_samples]

    cell_type_proportions = cell_type_counts.div(
        cell_type_counts.sum(axis=1), axis=0
    )
    proportions_long = (
        cell_type_proportions
        .rename_axis(index=["donor_id", "tissue"], columns="cell_type")
        .stack()
        .rename("proportion")
        .reset_index()
        .merge(
            sample_metadata.reset_index()[
                ["donor_id", "tissue", "self_reported_ethnicity"]
            ],
            on=["donor_id", "tissue"],
            how="left",
        )
    )
    abundance_summary = (
        proportions_long.groupby(
            ["tissue", "self_reported_ethnicity", "cell_type"], observed=True
        )
        .agg(
            n_donors=("donor_id", "nunique"),
            median_proportion=("proportion", "median"),
            mean_proportion=("proportion", "mean"),
        )
        .reset_index()
        .sort_values(["tissue", "cell_type", "self_reported_ethnicity"])
    )

    results_by_tissue = {}
    credible_rows = []
    status_rows = []
    for tissue in sorted(sample_metadata.index.get_level_values("tissue").unique()):
        tissue_meta = sample_metadata.xs(tissue, level="tissue")
        tissue_counts = cell_type_counts.xs(tissue, level="tissue")
        donor_counts = tissue_meta.groupby(
            "self_reported_ethnicity", observed=True
        ).size()
        groups = [
            group for group in COMPARISON_GROUPS
            if donor_counts.get(group, 0) >= min_donors_per_group
        ]
        if len(groups) < 2:
            status_rows.append({
                "tissue": tissue,
                "status": "Skipped: insufficient donors in both comparison groups",
                "donors_by_ethnicity": donor_counts.to_dict(),
            })
            continue

        keep = tissue_meta["self_reported_ethnicity"].isin(groups)
        tissue_meta = tissue_meta.loc[keep].copy()
        tissue_counts = tissue_counts.loc[tissue_meta.index].copy()
        tissue_counts = tissue_counts.loc[:, tissue_counts.sum(axis=0) > 0]
        tissue_meta["self_reported_ethnicity"] = pd.Categorical(
            tissue_meta["self_reported_ethnicity"].astype("string"),
            categories=groups,
        )
        sex_levels = tissue_meta["sex"].astype("string").dropna().unique().tolist()
        tissue_meta["sex"] = pd.Categorical(
            tissue_meta["sex"].astype("string"), categories=sex_levels
        )
        sample_names = tissue_meta.index.astype(str)
        tissue_meta.index = sample_names
        tissue_counts.index = sample_names
        coda_data = ad.AnnData(
            X=tissue_counts.to_numpy(dtype=np.int64),
            obs=tissue_meta[["self_reported_ethnicity", "sex"]].copy(),
            var=pd.DataFrame(index=tissue_counts.columns.astype(str)),
        )
        formula = "C(self_reported_ethnicity, Treatment('White'))"
        if coda_data.obs["sex"].nunique() > 1:
            formula += " + C(sex, Treatment('male'))"

        try:
            model = CompositionalAnalysis(
                coda_data, formula=formula, reference_cell_type="automatic"
            )
            result = model.sample_hmc()
            result.set_fdr(est_fdr=alpha)
            results_by_tissue[tissue] = result
            effects = result.credible_effects().rename(
                "credible_effect"
            ).reset_index()
            covariate, cell_type = effects.columns[:2]
            ethnicity_effects = effects[
                effects[covariate].astype(str).str.startswith(
                    "C(self_reported_ethnicity"
                )
                & effects[covariate].astype(str).str.contains(
                    "[T.Hispanic or Latin]", regex=False
                )
                & effects["credible_effect"]
            ]
            for _, effect in ethnicity_effects.iterrows():
                credible_rows.append({
                    "tissue": tissue,
                    "covariate": effect[covariate],
                    "cell_type": effect[cell_type],
                    "credible_effect_at_5pct_fdr": True,
                    "n_white_donors": int(donor_counts.get("White", 0)),
                    "n_hispanic_or_latin_donors": int(
                        donor_counts.get("Hispanic or Latin", 0)
                    ),
                })
            status_rows.append({
                "tissue": tissue,
                "status": "Fit scCODA model",
                "donors_by_ethnicity": donor_counts.to_dict(),
            })
        except Exception as exc:
            status_rows.append({
                "tissue": tissue,
                "status": f"Model failed: {exc}",
                "donors_by_ethnicity": donor_counts.to_dict(),
            })

    credible_findings = pd.DataFrame(credible_rows)
    return {
        "cell_type_counts": cell_type_counts,
        "sample_metadata": sample_metadata,
        "proportions_long": proportions_long,
        "summary": abundance_summary,
        "results_by_tissue": results_by_tissue,
        "credible_findings": credible_findings,
        "status": pd.DataFrame(status_rows),
    }


def plot_credible_abundance_findings(proportions_long, findings, ncols=3):
    """Plot donor proportions for ethnicity effects credible under scCODA."""
    findings = findings.sort_values(["tissue", "cell_type"]).reset_index(drop=True)
    if findings.empty:
        print("No scCODA-credible findings are available to plot.")
        return None

    nrows = math.ceil(len(findings) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False
    )
    for ax, row in zip(axes.flat, findings.itertuples(index=False)):
        panel = proportions_long.loc[
            proportions_long["tissue"].eq(row.tissue)
            & proportions_long["cell_type"].eq(row.cell_type)
            & proportions_long["self_reported_ethnicity"].isin(COMPARISON_GROUPS)
        ]
        values_by_group = [
            panel.loc[
                panel["self_reported_ethnicity"].eq(group), "proportion"
            ].to_numpy()
            for group in COMPARISON_GROUPS
        ]
        counts_by_group = [len(values) for values in values_by_group]
        boxes = ax.boxplot(
            values_by_group,
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#263238", "linewidth": 1.5},
            whiskerprops={"color": "#53636e"},
            capprops={"color": "#53636e"},
        )
        for box, group in zip(boxes["boxes"], COMPARISON_GROUPS):
            box.set_facecolor(ETHNICITY_PALETTE[group])
            box.set_edgecolor("#53636e")
            box.set_alpha(0.75)
        rng = np.random.default_rng(42)
        for x, group, values in zip([1, 2], COMPARISON_GROUPS, values_by_group):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                x + jitter,
                values,
                s=22,
                color=ETHNICITY_PALETTE[group],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )
        observed_max = max(float(np.max(values)) for values in values_by_group)
        scale = max(observed_max, 0.05)
        bar_y = observed_max + 0.10 * scale
        bar_height = 0.035 * scale
        ax.plot(
            [1, 1, 2, 2],
            [bar_y - bar_height, bar_y, bar_y, bar_y - bar_height],
            color=SC_CODA_COLOR,
            linewidth=2,
            clip_on=False,
        )
        ax.set_ylim(0, bar_y + 0.13 * scale)
        labels = [
            "White" if group == "White" else "Hispanic/Latin"
            for group in COMPARISON_GROUPS
        ]
        ax.set_xticks(
            [1, 2],
            [f"{label}\n(n={count})"
             for label, count in zip(labels, counts_by_group)],
            fontsize=8,
        )
        ax.set_title(f"{row.tissue}\n{row.cell_type}", fontsize=10)
        ax.set_ylabel("Donor proportion")
        ax.grid(axis="y", alpha=0.2)

    for ax in axes.flat[len(findings):]:
        ax.set_visible(False)
    legend = Line2D(
        [0], [0], color=SC_CODA_COLOR, lw=2,
        label="scCODA credible effect (5% FDR)",
    )
    fig.legend(handles=[legend], loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cell-type proportions by ethnicity in scCODA findings", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes


def plot_credible_effect_sizes(results_by_tissue, findings):
    """Plot scCODA ethnicity log2-fold changes and return the plotted table."""
    if findings.empty:
        print("No credible ethnicity findings are available to plot.")
        return pd.DataFrame()
    effect_frames = []
    for tissue, result in results_by_tissue.items():
        effects = result.effect_df.reset_index().rename(
            columns={"Cell Type": "cell_type", "Covariate": "covariate"}
        )
        effects = effects[
            effects["covariate"].astype(str).str.startswith(
                "C(self_reported_ethnicity"
            )
            & effects["Final Parameter"].ne(0)
        ].copy()
        effects["tissue"] = tissue
        effect_frames.append(effects)
    if not effect_frames:
        print("No fitted scCODA effects are available to plot.")
        return pd.DataFrame()

    effect_table = pd.concat(effect_frames, ignore_index=True)
    donor_columns = [
        "tissue", "cell_type", "n_white_donors", "n_hispanic_or_latin_donors"
    ]
    donor_counts = findings[donor_columns].drop_duplicates(
        ["tissue", "cell_type"]
    )
    effect_table = effect_table.merge(
        donor_counts,
        on=["tissue", "cell_type"],
        how="inner",
        validate="many_to_one",
    ).sort_values("log2-fold change").reset_index(drop=True)
    if effect_table.empty:
        print("No credible ethnicity effects are available to plot.")
        return effect_table

    labels = [
        f"{row.tissue} — {row['cell_type']} "
        f"(White n={row.n_white_donors}, "
        f"Hispanic/Latin n={row.n_hispanic_or_latin_donors})"
        for _, row in effect_table.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.65 * len(effect_table))))
    y = range(len(effect_table))
    ax.scatter(effect_table["log2-fold change"], y, color=SC_CODA_COLOR, s=60)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("scCODA estimated log2 fold change (Hispanic/Latin vs White)")
    ax.set_title("Credible cell-type abundance differences by tissue")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return effect_table
