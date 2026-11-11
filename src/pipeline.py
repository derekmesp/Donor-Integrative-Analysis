
from patsy import build_design_matrices, dmatrices
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import logging
import re
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from statannotations.Annotator import Annotator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

publication_palette = {
    "Hispanic/Latino": "#1762a4",          # Muted Blue
    "White": "#bd6319",                    # Muted Orange
    "Black or African American": "#1cb251",  # Muted Green
    "Asian": "#97182f"                     # Muted Rose Pink
}
sex_palette = {
    "MALE": "#0c6e97",
    "FEMALE": "#84109e"  # Muted Orange
}


def _palette_for_categories(categories, base_palette):
    """Keep known category colors fixed and fill missing categories consistently."""
    extra_categories = sorted(set(categories).difference(base_palette))
    extra_colors = sns.color_palette("husl", n_colors=len(extra_categories))
    palette = dict(base_palette)
    palette.update(dict(zip(extra_categories, extra_colors)))
    return {category: palette[category] for category in categories}


def summarize_categorical_data(df, columns=None, include_missing=True):
    """
    Summarizes categorical data in a DataFrame, providing counts and percentages for each category.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing categorical data.
    columns : list, optional
        List of column names to summarize. If None, object, category, and boolean columns are summarized,
        except for the Donor identifier.
    include_missing : bool, optional
        Whether to include missing values in the summary. Default is True. For Sex and Ethnicity,
        percentages use unique donors; conflicting labels are counted in each observed category and logged.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the summary of categorical data with columns: 'Variable', 'Category',
        'Count', and 'Percent'.
    """
    if columns is None:
        columns = [
            col for col in df.select_dtypes(include=["object", "category", "bool"]).columns
            if col.lower() != "donor"
        ]

    rows = []
    for column in columns:
        demographic = column.strip().rstrip(":").strip()
        if demographic in {"Sex", "Ethnicity"}:
            if "Donor" not in df.columns:
                raise ValueError(
                    f"A Donor column is required to summarize {demographic} by unique donor")

            donor_values = df[["Donor", column]].copy()
            donor_values["Donor"] = (
                donor_values["Donor"].astype("string").str.strip())
            donor_values[column] = donor_values[column].map(
                lambda value: value.strip() if isinstance(value, str) else value)
            donor_values = donor_values.dropna(subset=["Donor"])
            donor_categories = donor_values.groupby(
                "Donor", sort=False)[column].agg(
                    lambda values: list(pd.unique(values.dropna())))
            conflicting_donors = donor_categories[
                donor_categories.map(len) > 1].index.tolist()
            if conflicting_donors:
                logging.warning(
                    "Donors with multiple %s labels are counted once in each label: %s",
                    demographic, conflicting_donors)

            categories = []
            for donor_labels in donor_categories:
                categories.extend(donor_labels or ["(Missing)"])
            counts = pd.Series(categories, dtype="object").value_counts()
            denominator = (
                len(donor_categories) if include_missing
                else donor_categories.map(bool).sum()
            )
            if not include_missing:
                counts = counts.drop("(Missing)", errors="ignore")
        else:
            counts = df[column].value_counts(dropna=include_missing)
            denominator = len(
                df) if include_missing else df[column].notna().sum()

        for category, count in counts.items():
            rows.append({
                "Variable": demographic if demographic in {"Sex", "Ethnicity"} else column,
                "Category": "(Missing)" if pd.isna(category) else category,
                "Count": int(count),
                "Percent": (100 * count / denominator) if denominator else 0.0,
            })

    return pd.DataFrame(rows, columns=["Variable", "Category", "Count", "Percent"])


def plot_categorical_summary(categorical_summary, variables=None):
    """
    Plot category counts as one pie chart per variable.

    Parameters:
    -----------
    categorical_summary : pd.DataFrame
        A DataFrame containing the summary of categorical data with columns: 'Variable', 'Category',
        'Count', and 'Percent'.
    variables : list, optional
        List of variable names to plot. If None, all variables in the summary are plotted.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the generated plots.

    """
    required = {"Variable", "Category", "Count"}
    missing = required.difference(categorical_summary.columns)
    if missing:
        raise ValueError(
            f"Categorical summary is missing required columns: {sorted(missing)}")

    if variables is None:
        variables = categorical_summary["Variable"].drop_duplicates().tolist()
    plot_data = categorical_summary[
        categorical_summary["Variable"].isin(variables)]
    variables = [v for v in variables if v in plot_data["Variable"].values]
    if not variables:
        raise ValueError("No requested variables are present in the summary")

    fig, axes = plt.subplots(
        nrows=1, ncols=len(variables), figsize=(6 * len(variables), 6),
        squeeze=False)
    for ax, variable in zip(axes[0], variables):
        data = plot_data[plot_data["Variable"]
                         == variable].sort_values("Count")
        counts = data["Count"].to_numpy()
        categories = data["Category"].astype(str).to_numpy()
        wedges, _, _ = ax.pie(
            counts, startangle=90, autopct="%1.1f%%", pctdistance=0.72,
            textprops={"fontsize": 9})
        ax.legend(
            wedges,
            [f"{category} (n={count})" for category,
             count in zip(categories, counts)],
            title=variable, loc="center left", bbox_to_anchor=(1, 0.5),
            fontsize=8)
        ax.set_title(variable)
        ax.axis("equal")

    fig.tight_layout()
    plt.show()
    return fig


def column_clean(col_name):
    """
    Clean up column names for Patsy formula compatibility.

    Parameters:
    -----------
    col_name : str
        The original column name to be cleaned.

    Returns:
    --------
    str
        A cleaned version of the column name suitable for use in Patsy formulas.
    """

    safe_name = col_name.replace(
        '+', '_pos').replace('-', '_neg').replace(' ', '_').replace('/', '_')
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '', safe_name)
    if safe_name[0].isdigit():
        safe_name = "subset_" + safe_name
    return safe_name


def analyze_tissue_demographics(df, subset_columns, min_samples=20, alpha=0.05):
    """
    Iterates through cell subsets running tissue-stratified OLS models with robust errors.

    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset containing demographics and cell markers.
    subset_columns : list
        List of strings corresponding to the T-cell subset columns to evaluate.
    min_samples : int
        Minimum row count required per tissue to calculate valid statistics.
    alpha : float
        P-value threshold to flag significant demographic discoveries.

    Returns:
    --------
    pd.DataFrame
        A long-form summary table tracking every single demographic coefficient, 
        standard error, p-value, and significance flag across all subsets and tissues.
    """

    demographics = ["Age", "BMI", "Sex", "Ethnicity", "Tissue"]

    name_mapping = {col: column_clean(col) for col in subset_columns}
    safe_subsets = list(name_mapping.values())

    reverse_mapping = {v: k for k, v in name_mapping.items()}
    df_working = df.rename(columns=name_mapping).copy()
    for safe_subset in safe_subsets:
        df_working[safe_subset] = pd.to_numeric(
            df_working[safe_subset], errors="coerce")

    all_results = []

    for safe_subset in safe_subsets:
        original_name = reverse_mapping[safe_subset]
        logging.info(f"Processing Target Subset: {original_name}")

        model_features = demographics + [safe_subset]
        df_subset_clean = df_working[model_features].dropna()

        tissues = df_subset_clean["Tissue"].unique()

        for t in tissues:
            tissue_df = df_subset_clean[df_subset_clean["Tissue"] == t]

            ethnicity_counts = tissue_df["Ethnicity"].value_counts()
            singleton_ethnicity_mask = (
                tissue_df["Ethnicity"].map(ethnicity_counts).eq(1)
                & tissue_df["Ethnicity"].ne("White")
            )
            if singleton_ethnicity_mask.any():
                removed_rows = tissue_df.index[singleton_ethnicity_mask].tolist(
                )
                removed_ethnicities = tissue_df.loc[
                    singleton_ethnicity_mask, "Ethnicity"].tolist()
                logging.warning(
                    "Excluding singleton non-reference ethnicity observation(s) "
                    "for %s in tissue %s: %s",
                    original_name, t,
                    list(zip(removed_rows, removed_ethnicities)))
                tissue_df = tissue_df.loc[~singleton_ethnicity_mask]

            if len(tissue_df) >= min_samples:
                try:
                    formula = f"{safe_subset} ~ C(Sex, Treatment(reference='MALE')) + C(Ethnicity, Treatment(reference='White')) + Age + BMI"

                    preliminary_model = smf.ols(
                        formula, data=tissue_df).fit()
                    leverage = preliminary_model.get_influence().hat_matrix_diag
                    high_leverage_rows = tissue_df.index[leverage >= 1 - 1e-10]
                    if len(high_leverage_rows):
                        logging.warning(
                            "Skipping model for %s in tissue %s: "
                            "observation(s) %s have leverage 1, so HC3 covariance is undefined",
                            original_name, t, high_leverage_rows.tolist())
                        continue

                    model = smf.ols(formula, data=tissue_df).fit(
                        cov_type="HC3")

                    params = model.params
                    pvalues = model.pvalues
                    bse = model.bse
                    conf_int = model.conf_int()

                    for coef_name in params.index:
                        if coef_name == 'Intercept':
                            continue

                        clean_coef_label = (coef_name
                                            .replace("C(Sex, Treatment(reference='MALE'))[T.", "Sex: ")
                                            .replace("C(Ethnicity, Treatment(reference='White'))[T.", "Ethnicity: ")
                                            .replace("]", ""))

                        all_results.append({
                            "Subset": original_name,
                            "Tissue": t,
                            "N_Samples": len(tissue_df),
                            "Demographic_Variable": clean_coef_label,
                            "Coefficient": params[coef_name],
                            "Std_Err": bse[coef_name],
                            "p_value": pvalues[coef_name],
                            "CI_Lower": conf_int.loc[coef_name, 0],
                            "CI_Upper": conf_int.loc[coef_name, 1],
                            "Is_Significant": pvalues[coef_name] < alpha
                        })
                except Exception as e:
                    logging.warning(
                        f"Could not compute model for {original_name} in tissue {t}: {str(e)}")

    results = pd.DataFrame(all_results)
    if not results.empty:
        valid_pvalues = results["p_value"].notna()
        results["p_value_FDR"] = float("nan")
        if valid_pvalues.any():
            results.loc[valid_pvalues, "p_value_FDR"] = multipletests(
                results.loc[valid_pvalues, "p_value"],
                alpha=alpha,
                method="fdr_bh",
            )[1]
        results["Is_Significant_FDR"] = results["p_value_FDR"] < alpha

    return results


def analyze_tissue_demographic_group_combinations(
        df, subset_columns, min_samples=20, alpha=0.05):
    """Compare every observed sex-by-ethnicity group pair within each tissue.

    The model treats each Sex × Ethnicity combination as a group and adjusts
    for Age and BMI. Groups with fewer than two complete observations are
    excluded. Each reported coefficient is Comparison_Group minus
    Reference_Group. Both directions are returned for every pair, while BH
    correction is applied once per unique pair and shared by its two rows.
    """
    demographics = ["Age", "BMI", "Sex", "Ethnicity", "Tissue"]
    name_mapping = {col: column_clean(col) for col in subset_columns}
    reverse_mapping = {safe: original for original,
                       safe in name_mapping.items()}
    df_working = df.rename(columns=name_mapping).copy()
    safe_subsets = list(name_mapping.values())

    for safe_subset in safe_subsets:
        df_working[safe_subset] = pd.to_numeric(
            df_working[safe_subset], errors="coerce")

    all_results = []
    for safe_subset in safe_subsets:
        original_name = reverse_mapping[safe_subset]
        features = demographics + [safe_subset]
        clean_df = df_working[features].dropna().copy()
        clean_df["Sex"] = clean_df["Sex"].astype("string").str.strip()
        clean_df["Ethnicity"] = clean_df["Ethnicity"].astype(
            "string").str.strip()

        for tissue, tissue_df in clean_df.groupby("Tissue", sort=False):
            tissue_df = tissue_df.copy()
            tissue_df["Demographic_Group"] = (
                tissue_df["Sex"] + " / " + tissue_df["Ethnicity"])
            group_counts = tissue_df["Demographic_Group"].value_counts()
            rare_groups = group_counts[group_counts < 2].index.tolist()
            if rare_groups:
                logging.warning(
                    "Excluding sex/ethnicity groups with fewer than 2 observations "
                    "for %s in tissue %s: %s",
                    original_name, tissue, rare_groups)
                tissue_df = tissue_df[
                    ~tissue_df["Demographic_Group"].isin(rare_groups)].copy()
                group_counts = tissue_df["Demographic_Group"].value_counts()

            groups = sorted(group_counts.index.tolist())
            if len(tissue_df) < min_samples or len(groups) < 2:
                continue

            reference = groups[0]
            formula = (
                f'{safe_subset} ~ C(Demographic_Group, '
                f'Treatment(reference={reference!r})) + Age + BMI'
            )
            try:
                preliminary_model = smf.ols(formula, data=tissue_df).fit()
                leverage = preliminary_model.get_influence().hat_matrix_diag
                high_leverage = tissue_df.index[leverage >= 1 - 1e-10]
                if len(high_leverage):
                    logging.warning(
                        "Skipping group-combination model for %s in tissue %s: "
                        "observation(s) %s have leverage 1",
                        original_name, tissue, high_leverage.tolist())
                    continue

                model = smf.ols(formula, data=tissue_df).fit(cov_type="HC3")
                group_design = build_design_matrices(
                    [model.model.data.design_info],
                    pd.DataFrame({
                        "Demographic_Group": groups,
                        "Age": [0.0] * len(groups),
                        "BMI": [0.0] * len(groups),
                    }, index=groups),
                    return_type="dataframe",
                )[0]

                for reference_group in groups:
                    for comparison_group in groups:
                        if comparison_group == reference_group:
                            continue
                        contrast = (
                            group_design.loc[comparison_group]
                            - group_design.loc[reference_group]
                        ).to_numpy()
                        test = model.t_test(contrast)
                        confidence_interval = np.asarray(
                            test.conf_int()).reshape(-1, 2)[0]
                        all_results.append({
                            "Subset": original_name,
                            "Tissue": tissue,
                            "N_Samples": len(tissue_df),
                            "Reference_Group": reference_group,
                            "N_Reference": int(group_counts[reference_group]),
                            "Comparison_Group": comparison_group,
                            "N_Comparison": int(group_counts[comparison_group]),
                            "Coefficient": float(np.asarray(test.effect).item()),
                            "Std_Err": float(np.asarray(test.sd).item()),
                            "p_value": float(np.asarray(test.pvalue).item()),
                            "CI_Lower": float(confidence_interval[0]),
                            "CI_Upper": float(confidence_interval[1]),
                        })
            except Exception as e:
                logging.warning(
                    "Could not compute group-combination model for %s in tissue %s: %s",
                    original_name, tissue, e)

    result_columns = [
        "Subset", "Tissue", "N_Samples", "Reference_Group", "N_Reference",
        "Comparison_Group", "N_Comparison", "Coefficient", "Std_Err",
        "p_value", "CI_Lower", "CI_Upper",
    ]
    results = pd.DataFrame(all_results, columns=result_columns)
    results["p_value_FDR"] = float("nan")
    results["Is_Significant_FDR"] = False
    if not results.empty:
        results["_pair_key"] = results.apply(
            lambda row: (
                row["Subset"], row["Tissue"],
                *sorted([row["Reference_Group"], row["Comparison_Group"]])
            ),
            axis=1,
        )
        unique_pairs = results.drop_duplicates("_pair_key")
        valid_pairs = unique_pairs[unique_pairs["p_value"].notna()]
        if not valid_pairs.empty:
            corrected_pvalues = multipletests(
                valid_pairs["p_value"], alpha=alpha, method="fdr_bh")[1]
            fdr_lookup = dict(zip(valid_pairs["_pair_key"], corrected_pvalues))
            results["p_value_FDR"] = results["_pair_key"].map(fdr_lookup)
        results["Is_Significant_FDR"] = results["p_value_FDR"] < alpha
        results = results.drop(columns="_pair_key")
    return results


def stat_boxplot(df, population_name, tissue):
    """
    Generates a boxplot for a specified population across different ethnicities within a given tissue.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing the population data.
    population_name : str
        The name of the population column to plot.
    tissue : str
        The tissue type to filter the data by.

    Returns:
    --------
    None
    """

    df = df.dropna(subset=["Ethnicity", "Tissue", population_name, "Donor"])
    df = df[df["Tissue"] == tissue].reset_index(drop=True)

    if df.empty:
        logging.warning(f"No data available for tissue {tissue}")
        return

    ethnicity_counts = df["Ethnicity"].value_counts()
    excluded_ethnicities = ethnicity_counts[ethnicity_counts < 2].index.tolist(
    )
    if excluded_ethnicities:
        logging.info(
            "Excluding ethnicity groups with fewer than 2 observations "
            "for %s in tissue %s: %s",
            population_name, tissue, excluded_ethnicities)
        df = df[~df["Ethnicity"].isin(excluded_ethnicities)].copy()

    if df.empty:
        logging.warning(
            "No ethnicity groups have at least 2 observations for %s in tissue %s",
            population_name, tissue)
        return

    unique_ethnicities = list(df["Ethnicity"].unique())

    grouped_values = df.groupby("Ethnicity")[population_name]
    q1 = grouped_values.transform(lambda values: values.quantile(0.25))
    q3 = grouped_values.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    is_outlier = ((df[population_name] < q1 - 1.5 * iqr)
                  | (df[population_name] > q3 + 1.5 * iqr))

    outlier_donors = df[is_outlier]["Donor"].unique()
    outlier_str = ", ".join(map(str, outlier_donors))
    logging.info(
        f"Outlier donors for {population_name} in tissue {tissue}: {outlier_str}"
    )

    df_filtered = df

    box_palette = _palette_for_categories(
        unique_ethnicities, publication_palette)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.boxplot(
        x="Ethnicity",
        y=population_name,
        data=df_filtered,
        hue="Ethnicity",
        order=unique_ethnicities,
        palette=box_palette,
        dodge=False,
        boxprops=dict(alpha=0.4),
        showfliers=True,
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        ax=ax,
        zorder=1,
    )

    sns.lineplot(
        x="Ethnicity",
        y=population_name,
        data=df_filtered,
        units="Donor",
        estimator=None,
        color="black",
        alpha=0.3,
        linewidth=1,
        ax=ax,
        zorder=2,
    )

    sns.stripplot(
        x="Ethnicity",
        y=population_name,
        data=df_filtered.loc[~is_outlier],
        hue="Ethnicity",
        order=unique_ethnicities,
        palette=box_palette,
        legend=False,
        jitter=True,
        size=6,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        zorder=3,
    )

    if len(unique_ethnicities) >= 2:
        pairs = list(combinations(unique_ethnicities, 2))
        annotator = Annotator(
            ax,
            pairs,
            data=df,
            x="Ethnicity",
            y=population_name,
            order=unique_ethnicities,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="outside",
            comparisons_correction="bonferroni",
            line_width=1.2,
            hide_non_significant=True,
            verbose=False,
            line_offset=0.05,
            line_height=0.02,
            text_offset=2.0,
        )
        try:
            annotator.apply_and_annotate()
        except Exception as e:
            logging.warning(
                f"Could not apply statistical annotations for {population_name} in tissue {tissue}: {str(e)}"
            )
    else:
        logging.warning(
            f"Not enough valid ethnicities for statistical comparison in {population_name} for tissue {tissue}."
        )

    ethnicity_positions = {
        ethnicity: position
        for position, ethnicity in enumerate(unique_ethnicities)
    }
    for ethnicity, outliers in df.loc[is_outlier].groupby("Ethnicity", sort=False):
        outliers = outliers.sort_values(population_name)
        for index, (_, outlier) in enumerate(outliers.iterrows()):
            to_right = index % 2 == 0
            ax.annotate(
                str(outlier["Donor"]),
                xy=(ethnicity_positions[ethnicity], outlier[population_name]),
                xytext=(10 if to_right else -10, 5 * (index // 2)),
                textcoords="offset points",
                ha="left" if to_right else "right",
                va="center",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.7},
                annotation_clip=False,
                zorder=5,
            )

    ethnicity_donor_counts = df_filtered.groupby("Ethnicity")[
        "Donor"].nunique()
    ax.set_xticks(range(len(unique_ethnicities)))
    ax.set_xticklabels(
        [f"{ethnicity}\n(n={ethnicity_donor_counts.get(ethnicity, 0)})"
         for ethnicity in unique_ethnicities],
        rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=10)
    ax.set_title(f"{population_name} in {tissue}", fontsize=14, pad=38)

    sns.despine()
    fig.tight_layout(rect=(0, 0, 0.82, 0.9))

    plt.savefig(
        f"{population_name}_{tissue}_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)


def sex_boxplot(df, population_name, tissue):
    """
    Generates a boxplot for a specified population across different sexes within a given tissue.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing the population data.
    population_name : str
        The name of the population column to plot.
    tissue : str
        The tissue type to filter the data by.

    Returns:
    --------
    None
    """

    df = df.dropna(subset=["Sex", "Tissue", population_name, "Donor"])
    df = df[df["Tissue"] == tissue].reset_index(drop=True)

    if df.empty:
        logging.warning(f"No data available for tissue {tissue}")
        return

    unique_sexes = list(df["Sex"].unique())

    grouped_values = df.groupby("Sex")[population_name]
    q1 = grouped_values.transform(lambda values: values.quantile(0.25))
    q3 = grouped_values.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    is_outlier = ((df[population_name] < q1 - 1.5 * iqr)
                  | (df[population_name] > q3 + 1.5 * iqr))

    outlier_donors = df[is_outlier]["Donor"].unique()
    outlier_str = ", ".join(map(str, outlier_donors))
    logging.info(
        f"Outlier donors for {population_name} in tissue {tissue}: {outlier_str}"
    )

    df_filtered = df

    box_palette = _palette_for_categories(unique_sexes, sex_palette)

    fig, ax = plt.subplots(figsize=(9, 8))

    sns.boxplot(
        x="Sex",
        y=population_name,
        data=df_filtered,
        hue="Sex",
        order=unique_sexes,
        palette=box_palette,
        dodge=False,
        boxprops=dict(alpha=0.4),
        showfliers=True,
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        ax=ax,
        zorder=1,
    )

    sns.lineplot(
        x="Sex",
        y=population_name,
        data=df_filtered,
        units="Donor",
        estimator=None,
        color="black",
        alpha=0.3,
        linewidth=1,
        ax=ax,
        zorder=2,
    )

    sns.stripplot(
        x="Sex",
        y=population_name,
        data=df_filtered.loc[~is_outlier],
        hue="Sex",
        order=unique_sexes,
        palette=box_palette,
        legend=False,
        jitter=True,
        size=6,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        zorder=3,
    )

    sex_counts = df_filtered["Sex"].value_counts()
    valid_sexes = [
        sex for sex in unique_sexes if sex_counts[sex] >= 2
    ]

    if len(valid_sexes) >= 2:
        pairs = list(combinations(valid_sexes, 2))
        annotator = Annotator(
            ax,
            pairs,
            data=df,
            x="Sex",
            y=population_name,
            order=unique_sexes,
        )
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
            loc="outside",
            comparisons_correction="bonferroni",
            line_width=1.2,
            hide_non_significant=True,
            verbose=False,
            line_offset=0.05,
            line_height=0.02,
            text_offset=2.0,
        )
        try:
            annotator.apply_and_annotate()
        except Exception as e:
            logging.warning(
                f"Could not apply statistical annotations for {population_name} in tissue {tissue}: {str(e)}"
            )
    else:
        logging.warning(
            f"Not enough valid sexes for statistical comparison in {population_name} for tissue {tissue}."
        )

    sex_positions = {sex: position for position,
                     sex in enumerate(unique_sexes)}
    for sex, outliers in df.loc[is_outlier].groupby("Sex", sort=False):
        outliers = outliers.sort_values(population_name)
        for index, (_, outlier) in enumerate(outliers.iterrows()):
            to_right = index % 2 == 0
            ax.annotate(
                str(outlier["Donor"]),
                xy=(sex_positions[sex], outlier[population_name]),
                xytext=(10 if to_right else -10, 5 * (index // 2)),
                textcoords="offset points",
                ha="left" if to_right else "right",
                va="center",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.7},
                annotation_clip=False,
                zorder=5,
            )

    sex_donor_counts = df_filtered.groupby("Sex")["Donor"].nunique()
    ax.set_xticks(range(len(unique_sexes)))
    ax.set_xticklabels(
        [f"{sex}\n(n={sex_donor_counts.get(sex, 0)})" for sex in unique_sexes],
        rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=10)
    ax.set_title(f"{population_name} in {tissue}", fontsize=14, pad=38)

    sns.despine()
    fig.tight_layout(rect=(0, 0, 0.9, 0.9))

    plt.savefig(
        f"{population_name}_{tissue}_boxplot.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)


def regression_bmi(df, subset, tissue):
    """
    Performs linear regression of a specified subset against BMI for a given tissue and plots the results.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing the subset and BMI data.
    subset : str
        The name of the subset column to regress against BMI.
    tissue : str
        The tissue type to filter the data by.

    Returns:
    --------
    None
    """

    df_plot = df[df['Tissue'] == tissue].copy()
    clean_df = df_plot.dropna(subset=['BMI', subset])

    x = clean_df['BMI']
    y = clean_df[subset]

    plt.plot(x, y, 'o', label='Data points')

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    stats_label = f'Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.3f}\np = {p_value:.3e}'

    plt.plot(x, slope * x + intercept, color='red', label=stats_label)

    plt.xlabel('BMI')
    plt.ylabel(subset)
    plt.title(f'Regression of {subset} vs BMI in {tissue}')
    plt.legend(loc='best')
    plt.savefig(f"{subset}_{tissue}_regression.png",
                dpi=300, bbox_inches="tight")

    plt.show()

    print(f"--- Regression Statistics ---")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"p-value: {p_value:.4e}")


def ols_summary(df):
    """
    Generates a scatter plot of OLS regression coefficients for demographic variables across T-cell subsets and tissues.

    Parameters:
    -----------
    df : pd.DataFrame
        The summary DataFrame containing regression coefficients and related statistics.

    Returns:
    --------
    None
    """

    df = df[df["Demographic_Variable"] != "BMI"].copy()
    if df.empty:
        logging.warning("No FDR-significant OLS results to plot.")
        return

    df["Label"] = df["Subset"] + \
        " (" + df["Tissue"] + ")"

    df = df.sort_values(
        by=["Demographic_Variable", "Coefficient"], ascending=[True, True])

    figure_height = max(
        8,
        0.55 * df["Subset"].nunique(),
        0.3 * df["Label"].nunique(),
    )
    plt.figure(figsize=(11, figure_height))
    sns.set_theme(style="whitegrid")

    ax = sns.scatterplot(
        data=df,
        x="Coefficient",
        y="Label",
        hue="Demographic_Variable",
        palette="Set1",
        s=120,
        edgecolor="black",
        linewidth=1,
        zorder=3
    )

    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5,
                alpha=0.7, label="Baseline (White/Male)")
    plt.title("Tissue-Stratified OLS Demographic Coefficients on T-Cell Subsets\n(All points pass Benjamini-Hochberg correction, q < 0.05)",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Regression Coefficient (Effect Size)",
               fontsize=12, labelpad=10)
    plt.ylabel("T-Cell Subset",
               fontsize=12, labelpad=10)

    plt.xlim(-25, 30)
    plt.tight_layout()

    plt.legend(title="Demographic Group", title_fontsize="11",
               loc="best", frameon=True, shadow=False)

    plt.show()


def plot_group_comparison_summary(group_comparison_stats_df, significant_only=True):
    """Plot unique sex-by-ethnicity contrasts with confidence intervals."""
    required = {
        "Subset", "Tissue", "Reference_Group", "Comparison_Group",
        "N_Reference", "N_Comparison", "Coefficient", "CI_Lower",
        "CI_Upper", "p_value_FDR",
        "Is_Significant_FDR",
    }
    missing = required.difference(group_comparison_stats_df.columns)
    if missing:
        raise ValueError(
            f"Group comparison results are missing columns: {sorted(missing)}")

    data = group_comparison_stats_df[
        group_comparison_stats_df["Reference_Group"]
        < group_comparison_stats_df["Comparison_Group"]
    ].copy()
    if significant_only:
        data = data[data["Is_Significant_FDR"]]
    if data.empty:
        logging.warning("No group-combination contrasts to plot.")
        return

    data = data.sort_values("Coefficient").reset_index(drop=True)
    data["Plot_Label"] = data.apply(
        lambda row: (
            f"{row['Subset']} ({row['Tissue']}): "
            f"{row['Reference_Group']} (n={row['N_Reference']}) → "
            f"{row['Comparison_Group']} (n={row['N_Comparison']}) "
            f"(q={row['p_value_FDR']:.3g})"
        ),
        axis=1,
    )
    tissues = data["Tissue"].drop_duplicates().tolist()
    palette = dict(zip(
        tissues, sns.color_palette("tab20", n_colors=len(tissues))))
    figure_height = max(
        7, 0.55 * data["Subset"].nunique(), 0.38 * len(data))
    fig, ax = plt.subplots(figsize=(12, figure_height))

    for tissue in tissues:
        tissue_data = data[data["Tissue"] == tissue]
        positions = tissue_data.index.to_numpy()
        coefficients = tissue_data["Coefficient"].to_numpy()
        lower_errors = coefficients - tissue_data["CI_Lower"].to_numpy()
        upper_errors = tissue_data["CI_Upper"].to_numpy() - coefficients
        ax.errorbar(
            coefficients, positions,
            xerr=[lower_errors, upper_errors],
            fmt="o", color=palette[tissue], ecolor=palette[tissue],
            capsize=3, label=tissue,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(data.index)
    ax.set_yticklabels(data["Plot_Label"], fontsize=8)
    ax.set_xlabel("Adjusted outcome difference (comparison − reference)")
    ax.set_title("FDR-significant Sex × Ethnicity group comparisons")
    ax.legend(title="Tissue", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    plt.show()
    return fig


def visualize_ols_vs_mann_whitney(
        df, population_name, tissue, target_variable="BMI",
        adjusted_p_value=None):
    """
    Generates a 2-panel figure comparing univariable data visualization vs. an 
    Added Variable Plot (Partial Regression Plot) by bypassing Patsy parsing errors.

    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset containing demographics and cell markers.
    population_name : str
        The raw cell marker name (e.g., 'CD4 CD45RA-CCR7-').
    tissue : str
        The tissue string to filter by (e.g., 'MLN').
    target_variable : str
        The exact string parameter from your OLS summary. If categorical, 
        provide the Patsy formatted string (e.g., "C(Sex, Treatment(reference='MALE'))[T.FEMALE]" or 
        "C(Ethnicity, Treatment(reference='White'))[T.Asian]").
    adjusted_p_value : float, optional
        Multiple-testing-adjusted p-value from the full analysis, if available.
    """
    safe_pop = column_clean(population_name)
    df_working = df.rename(columns={population_name: safe_pop}).copy()

    df_tissue = df_working[df_working["Tissue"] == tissue].dropna(
        subset=[safe_pop, "Sex", "Ethnicity", "Age", "BMI"]
    )

    if df_tissue.empty:
        print(f"No complete data available for tissue {tissue}")
        return

    formula = f"{safe_pop} ~ C(Sex, Treatment(reference='MALE')) + C(Ethnicity, Treatment(reference='White')) + Age + BMI"
    y, X = dmatrices(formula, data=df_tissue, return_type='dataframe')

    if target_variable not in X.columns:
        raise ValueError(f"Variable '{target_variable}' not found in model columns. "
                         f"Available columns are: {list(X.columns)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    if "Sex" in target_variable:
        sns.boxplot(x="Sex", y=safe_pop, data=df_tissue,
                    ax=axes[0],
                    palette=_palette_for_categories(
                        df_tissue["Sex"].unique(), sex_palette),
                    hue="Sex", legend=False)
        sns.stripplot(x="Sex", y=safe_pop, data=df_tissue,
                      ax=axes[0], color="black", alpha=0.6, jitter=True)
        axes[0].set_title(
            f"Univariable View: {population_name} by Sex\n(Confounded / Mann-Whitney Null) in {tissue}")
        sex_counts = df_tissue.groupby("Sex")["Donor"].nunique()
        axes[0].set_xticklabels([
            f"{label.get_text()}\n(n={sex_counts.get(label.get_text(), 0)})"
            for label in axes[0].get_xticklabels()
        ])
    elif "Ethnicity" in target_variable:
        sns.boxplot(x="Ethnicity", y=safe_pop, data=df_tissue,
                    ax=axes[0], hue="Ethnicity",
                    palette=_palette_for_categories(
                        df_tissue["Ethnicity"].unique(), publication_palette),
                    legend=False)
        sns.stripplot(x="Ethnicity", y=safe_pop, data=df_tissue,
                      ax=axes[0], color="black", alpha=0.6, jitter=True)
        ethnicity_counts = df_tissue.groupby("Ethnicity")["Donor"].nunique()
        axes[0].set_xticklabels([
            f"{label.get_text()}\n(n={ethnicity_counts.get(label.get_text(), 0)})"
            for label in axes[0].get_xticklabels()
        ], rotation=45, ha='right')
        axes[0].set_title(
            f"Univariable View: {population_name} by Ethnicity in {tissue}")
    else:
        sns.regplot(x=target_variable, y=safe_pop, data=df_tissue, ax=axes[0],
                    scatter_kws={'alpha': 0.6, 'color': 'gray'}, line_kws={'color': 'red', 'linestyle': '--'})
        axes[0].set_title(
            f"Univariable View: {population_name} vs {target_variable}\n(Unadjusted Structural Noise)")
        axes[0].set_xlabel(f"Raw {target_variable}")

    axes[0].set_ylabel(f"{population_name} Frequency")
    full_model = sm.OLS(y, X).fit(cov_type="HC3")
    beta_coefficient = full_model.params[target_variable]
    confidence_interval = full_model.conf_int().loc[target_variable]
    ethnicity_columns = [
        col for col in X.columns
        if col.startswith("C(Ethnicity, Treatment(reference='White'))[T.")
    ]
    ethnicity_omnibus_p = None
    if ethnicity_columns:
        restrictions = pd.DataFrame(0.0, index=range(len(ethnicity_columns)),
                                    columns=X.columns)
        for row, column in enumerate(ethnicity_columns):
            restrictions.loc[row, column] = 1.0
        ethnicity_omnibus_p = float(full_model.f_test(restrictions).pvalue)

    other_cols = [col for col in X.columns if col != target_variable]

    sm.graphics.plot_partregress(
        endog=y,
        exog_i=X[target_variable],
        exog_others=X[other_cols],
        obs_labels=False,
        ax=axes[1]
    )

    clean_label = (
        target_variable.rsplit("[T.", 1)[1].rstrip("]")
        if "[T." in target_variable else target_variable
    )
    contrast = " vs White" if "Ethnicity" in target_variable else (
        " vs Male" if "Sex" in target_variable else "")
    p_label = f"raw p = {full_model.pvalues[target_variable]:.4f}"
    if adjusted_p_value is not None:
        p_label += f", BH q = {adjusted_p_value:.4f}"
    omnibus_label = (
        f"; ethnicity omnibus raw p = {ethnicity_omnibus_p:.4f}"
        if ethnicity_omnibus_p is not None else ""
    )
    axes[1].set_title(
        f"Adjusted OLS: {clean_label}{contrast} in {tissue}\n"
        f"β = {beta_coefficient:.2f}, 95% CI [{confidence_interval.iloc[0]:.2f}, "
        f"{confidence_interval.iloc[1]:.2f}], {p_label}{omnibus_label}")
    axes[1].set_xlabel(f"Residuals of {clean_label}")
    axes[1].set_ylabel(f"Residuals of {population_name}")

    sns.despine()
    plt.tight_layout()

    clean_filename = (target_variable
                      .replace("C(", "")
                      .replace(")[T.", "_")
                      .replace("]", "")
                      .replace(" ", "_")
                      .replace("/", "_"))
    plt.savefig(f"{safe_pop}_{tissue}_{clean_filename}_comparison_panel.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def visualize_ols_vs_mann_whitney_colored(df, population_name, tissue, target_variable="C(Sex, Treatment(reference='MALE'))[T.FEMALE]", color_by="Ethnicity"):
    """
    Generates a 2-panel figure comparing raw univariable data vs. an Added Variable Plot,
    with color-coded data points to expose cohort imbalances and demographic confounding.

    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset containing demographics and cell markers.
    population_name : str
        The raw cell marker name (e.g., 'Treg' or 'CD4 CD45RA-CCR7-').
    tissue : str
        The tissue string to filter by (e.g., 'MLN' or 'SPL').
    target_variable : str
        The exact string parameter from your OLS summary (e.g., "C(Sex, Treatment(reference='MALE'))[T.FEMALE]" or "BMI").
    color_by : str
        The metadata column used to color-code the points (e.g., 'Ethnicity' or 'Tissue').
    """
    safe_pop = column_clean(population_name)
    df_working = df.rename(columns={population_name: safe_pop}).copy()

    df_tissue = df_working[df_working["Tissue"] == tissue].dropna(
        subset=[safe_pop, "Sex", "Ethnicity", "Age", "BMI", color_by]
    ).reset_index(drop=True)

    if df_tissue.empty:
        print(f"No complete data available for tissue {tissue}")
        return

    formula = f"{safe_pop} ~ C(Sex, Treatment(reference='MALE')) + C(Ethnicity, Treatment(reference='White')) + Age + BMI"
    y, X = dmatrices(formula, data=df_tissue, return_type='dataframe')

    if target_variable not in X.columns:
        raise ValueError(
            f"Variable '{target_variable}' not found. Choose from: {list(X.columns)}")

    other_cols = [col for col in X.columns if col != target_variable]
    model_y = sm.OLS(y, X[other_cols]).fit()
    y_residuals = model_y.resid

    model_x = sm.OLS(X[target_variable], X[other_cols]).fit()
    x_residuals = model_x.resid

    df_res = pd.DataFrame({
        "X_res": x_residuals,
        "Y_res": y_residuals,
        "Color_Group": df_tissue[color_by],
        "Sex": df_tissue["Sex"]
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    unique_groups = sorted(df_tissue[color_by].unique())
    custom_palette = dict(
        zip(unique_groups, sns.color_palette("Set2", len(unique_groups))))

    if "Sex" in target_variable:
        sns.boxplot(x="Sex", y=safe_pop, data=df_tissue,
                    ax=axes[0], palette=sex_palette, hue="Sex", legend=False, showfliers=False)
        sns.stripplot(x="Sex", y=safe_pop, hue=color_by, data=df_tissue,
                      ax=axes[0], palette=custom_palette, alpha=0.8, jitter=True, size=7, edgecolor="black", linewidth=0.5)
        axes[0].set_title(
            f"Univariable View: {population_name} by Sex\n(Confounded / Overlapping Variance) in {tissue}")
    else:
        sns.scatterplot(x=target_variable, y=safe_pop, hue=color_by, data=df_tissue,
                        ax=axes[0], palette=custom_palette, s=70, edgecolor="black", alpha=0.8)
        axes[0].set_title(
            f"Univariable View: {population_name} vs {target_variable}")

    axes[0].set_ylabel(f"{population_name} Frequency")
    axes[0].legend(title=color_by, bbox_to_anchor=(1.02, 1), loc='upper left')

    full_model = sm.OLS(y, X).fit(cov_type="HC3")
    beta_coefficient = full_model.params[target_variable]
    intercept = 0

    line_x = [x_residuals.min(), x_residuals.max()]
    line_y = [intercept + beta_coefficient * x for x in line_x]
    axes[1].plot(line_x, line_y, color="black",
                 linestyle="-", linewidth=1.5, zorder=1)

    sns.scatterplot(
        x="X_res", y="Y_res", hue="Color_Group", data=df_res, ax=axes[1],
        palette=custom_palette, s=80, edgecolor="black", alpha=0.9, zorder=2
    )

    clean_label = target_variable.split(
        '.')[-1].replace(']', '') if '[' in target_variable else target_variable
    axes[1].set_title(
        f"Adjusted OLS View: Isolated Effect of {clean_label} in {tissue}\n(OLS β = {beta_coefficient:.3f}, p = {full_model.pvalues[target_variable]:.4f})")
    axes[1].set_xlabel(f"Residuals of {clean_label}")
    axes[1].set_ylabel(f"Residuals of {population_name}")
    axes[1].get_legend().remove()

    sns.despine()
    plt.tight_layout()

    plt.savefig(f"{safe_pop}_{tissue}_{color_by}_colored_panel.png",
                dpi=300, bbox_inches="tight")
    plt.show()
