
from patsy import build_design_matrices, dmatrices
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations
from scipy import stats
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
    """
    Create a color palette for a set of categories, using a base palette and adding extra colors for any missing categories.

    Parameters:
    -----------
    categories : list
        A list of category names for which to create a color palette.
    base_palette : dict
        A dictionary mapping category names to colors. This serves as the base palette.

    Returns:
    --------
    dict
        A dictionary mapping each category in `categories` to a color. Categories not in the base
        palette will be assigned colors from a seaborn color palette.
    """
    extra_categories = sorted(set(categories).difference(base_palette))
    extra_colors = sns.color_palette("husl", n_colors=len(extra_categories))
    palette = dict(base_palette)
    palette.update(dict(zip(extra_categories, extra_colors)))
    return {category: palette[category] for category in categories}


def _prepare_ethnicity_boxplot_data(
        df, population_name, tissue, sex=None, min_donors=3):
    """Prepare the shared donor-filtered data used by ethnicity boxplots."""
    required = ["Ethnicity", "Tissue", population_name, "Donor"]
    if sex is not None:
        required.append("Sex")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = df.dropna(subset=required).copy()
    data["Ethnicity"] = data["Ethnicity"].astype("string").str.strip()
    data["Tissue"] = data["Tissue"].astype("string").str.strip()
    data["Donor"] = data["Donor"].astype("string").str.strip()
    data[population_name] = pd.to_numeric(
        data[population_name], errors="coerce")
    data = data.dropna(subset=[population_name])
    data = data[data["Tissue"] == str(tissue).strip()]
    if sex is not None:
        data["Sex"] = data["Sex"].astype("string").str.strip()
        data = data[data["Sex"] == str(sex).strip()]
    data = data.reset_index(drop=True)

    donor_counts = data.groupby("Ethnicity")["Donor"].nunique()
    categories = [
        ethnicity for ethnicity in data["Ethnicity"].drop_duplicates()
        if donor_counts.get(ethnicity, 0) >= min_donors
    ]
    excluded = [
        ethnicity for ethnicity in data["Ethnicity"].drop_duplicates()
        if ethnicity not in categories
    ]
    if excluded:
        logging.info(
            "Excluding ethnicity groups with fewer than %d donors for %s in %s: %s",
            min_donors, population_name, tissue, excluded)
    data = data[data["Ethnicity"].isin(categories)].copy()
    return data, categories, donor_counts


def _ethnicity_iqr_outliers(data, population_name):
    """Return an outlier mask calculated within each ethnicity group."""
    grouped = data.groupby("Ethnicity")[population_name]
    q1 = grouped.transform(lambda values: values.quantile(0.25))
    q3 = grouped.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    return ((data[population_name] < q1 - 1.5 * iqr)
            | (data[population_name] > q3 + 1.5 * iqr))


def _annotate_ethnicity_comparisons(ax, data, population_name, categories):
    """Apply the shared Bonferroni-corrected Mann–Whitney annotations."""
    if len(categories) < 2:
        return
    annotator = Annotator(
        ax, list(combinations(categories, 2)), data=data,
        x="Ethnicity", y=population_name, order=categories,
    )
    annotator.configure(
        test="Mann-Whitney", text_format="star", loc="outside",
        comparisons_correction="bonferroni", line_width=1.2,
        hide_non_significant=True, verbose=False, line_offset=0.05,
        line_height=0.02, text_offset=2.0,
    )
    try:
        annotator.apply_and_annotate()
    except Exception as exc:
        logging.warning(
            "Could not apply ethnicity significance annotations for %s: %s",
            population_name, exc)


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
    """
    Iterates through cell subsets running tissue-stratified OLS models with robust errors
    for every combination of sex and ethnicity groups.

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
            ethnicity_counts = tissue_df["Ethnicity"].value_counts()
            rare_ethnicities = ethnicity_counts[ethnicity_counts < 2].index.tolist(
            )
            if rare_ethnicities:
                tissue_df = tissue_df[
                    ~tissue_df["Ethnicity"].isin(rare_ethnicities)].copy()

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


def analyze_paired_tissue_demographics(
        df, subset_columns, tissue_class_map, class_a="Mucosal",
        class_b="Lymphoid", sex_reference="MALE",
        ethnicity_reference="White", min_donors=20,
        min_group_donors=3, alpha=0.05):
    """Test whether demographics predict donor-level class differences.

    Each donor's values are averaged within each tissue class. The response is
    ``class_a - class_b`` for donors represented in both classes, then modeled
    against sex, ethnicity, age, and BMI with HC3 robust standard errors.
    Ethnicity is also tested jointly with an omnibus F-test. FDR correction is
    applied across all reported tests and subsets.
    """
    subset_columns = list(subset_columns)
    if not subset_columns:
        raise ValueError("Provide at least one subset column")
    if min_donors < 3 or min_group_donors < 3:
        raise ValueError("Donor minimums must be at least 3")
    required = {"Donor", "Tissue", "Sex", "Ethnicity", "Age", "BMI",
                *subset_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not tissue_class_map:
        raise ValueError("Provide an explicit tissue_class_map")

    tissue_class_map = {
        str(tissue).strip(): str(group).strip()
        for tissue, group in tissue_class_map.items()
    }
    class_a, class_b = str(class_a).strip(), str(class_b).strip()
    if class_a == class_b:
        raise ValueError("class_a and class_b must be different")
    if not {class_a, class_b}.issubset(set(tissue_class_map.values())):
        raise ValueError(
            f"Both requested classes must be mapped; available classes are "
            f"{sorted(set(tissue_class_map.values()))}")

    metadata = df[["Donor", "Sex", "Ethnicity", "Age", "BMI"]].copy()
    metadata["Donor"] = metadata["Donor"].astype("string").str.strip()
    for column in ["Sex", "Ethnicity"]:
        metadata[column] = metadata[column].astype("string").str.strip()
    for column in ["Age", "BMI"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    metadata = metadata.dropna(
        subset=["Donor", "Sex", "Ethnicity", "Age", "BMI"])
    conflicts = metadata.groupby("Donor")[
        ["Sex", "Ethnicity", "Age", "BMI"]].nunique().gt(1).any(axis=1)
    if conflicts.any():
        logging.warning(
            "Using the first complete demographic record for donors with "
            "conflicting metadata: %s", conflicts[conflicts].index.tolist())
    metadata = metadata.groupby("Donor", as_index=False).first()

    formula = (
        "Tissue_Class_Difference ~ "
        f"C(Sex, Treatment(reference={sex_reference!r})) + "
        f"C(Ethnicity, Treatment(reference={ethnicity_reference!r})) + Age + BMI"
    )
    all_results = []
    for subset in subset_columns:
        values = df[["Donor", "Tissue", subset]].copy()
        values["Donor"] = values["Donor"].astype("string").str.strip()
        values["Tissue"] = values["Tissue"].astype("string").str.strip()
        values[subset] = pd.to_numeric(values[subset], errors="coerce")
        values["Tissue_Class"] = values["Tissue"].map(tissue_class_map)
        values = values.dropna(subset=["Donor", subset, "Tissue_Class"])
        values = values[values["Tissue_Class"].isin([class_a, class_b])]
        class_means = values.groupby(
            ["Donor", "Tissue_Class"], observed=True)[subset].mean()
        paired = class_means.unstack("Tissue_Class").reindex(
            columns=[class_a, class_b]).dropna(subset=[class_a, class_b])
        paired["Tissue_Class_Difference"] = paired[class_a] - paired[class_b]
        donor_data = paired[["Tissue_Class_Difference"]].merge(
            metadata, left_index=True, right_on="Donor", how="inner")

        for demographic in ["Sex", "Ethnicity"]:
            counts = donor_data[demographic].value_counts()
            rare_groups = counts[counts < min_group_donors].index
            donor_data = donor_data[
                ~donor_data[demographic].isin(rare_groups)].copy()
        if len(donor_data) < min_donors:
            logging.info(
                "Skipping demographic model for %s: %d complete paired donors",
                subset, len(donor_data))
            continue
        if sex_reference not in set(donor_data["Sex"]):
            logging.info("Skipping %s: sex reference %s is absent",
                         subset, sex_reference)
            continue
        if ethnicity_reference not in set(donor_data["Ethnicity"]):
            logging.info("Skipping %s: ethnicity reference %s is absent",
                         subset, ethnicity_reference)
            continue

        try:
            ordinary_model = smf.ols(formula, data=donor_data).fit()
            leverage = ordinary_model.get_influence().hat_matrix_diag
            if np.any(leverage >= 1 - 1e-10):
                logging.warning(
                    "Skipping demographic model for %s: a donor has leverage 1",
                    subset)
                continue
            model = smf.ols(formula, data=donor_data).fit(cov_type="HC3")
        except Exception as exc:
            logging.warning(
                "Could not fit paired demographic model for %s: %s", subset, exc)
            continue

        terms = [term for term in model.params.index if term != "Intercept"]
        for term in terms:
            if term.startswith("C(Sex,"):
                label = "Sex: " + term.rsplit("[T.", 1)[1].rstrip("]")
            elif term.startswith("C(Ethnicity,"):
                label = "Ethnicity: " + term.rsplit("[T.", 1)[1].rstrip("]")
            else:
                label = term
            interval = model.conf_int().loc[term]
            all_results.append({
                "Subset": subset, "Class_A": class_a, "Class_B": class_b,
                "N_Donors": len(donor_data), "Demographic_Variable": label,
                "Test_Type": "coefficient", "Coefficient": float(model.params[term]),
                "Std_Err": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "CI_Lower": float(interval.iloc[0]),
                "CI_Upper": float(interval.iloc[1]),
            })

        ethnicity_terms = [term for term in terms
                           if term.startswith("C(Ethnicity,")]
        if ethnicity_terms:
            restrictions = pd.DataFrame(
                0.0, index=range(len(ethnicity_terms)),
                columns=model.params.index)
            for row, term in enumerate(ethnicity_terms):
                restrictions.loc[row, term] = 1.0
            all_results.append({
                "Subset": subset, "Class_A": class_a, "Class_B": class_b,
                "N_Donors": len(donor_data),
                "Demographic_Variable": "Ethnicity: omnibus",
                "Test_Type": "omnibus", "Coefficient": np.nan,
                "Std_Err": np.nan,
                "p_value": float(model.f_test(restrictions).pvalue),
                "CI_Lower": np.nan, "CI_Upper": np.nan,
            })

    output_columns = [
        "Subset", "Class_A", "Class_B", "N_Donors", "Demographic_Variable",
        "Test_Type", "Coefficient", "Std_Err", "p_value", "CI_Lower",
        "CI_Upper", "p_value_FDR", "Is_Significant_FDR",
    ]
    results = pd.DataFrame(all_results)
    if results.empty:
        return pd.DataFrame(columns=output_columns)
    valid = results["p_value"].notna()
    results["p_value_FDR"] = np.nan
    if valid.any():
        results.loc[valid, "p_value_FDR"] = multipletests(
            results.loc[valid, "p_value"], alpha=alpha, method="fdr_bh")[1]
    results["Is_Significant_FDR"] = results["p_value_FDR"] < alpha
    return results[output_columns]


def plot_paired_tissue_demographic_summary(
        results, demographic_variable=None, significant_only=True):
    """Plot adjusted demographic effects on paired tissue-class differences."""
    required = {
        "Subset", "Class_A", "Class_B", "N_Donors", "Demographic_Variable",
        "Test_Type", "Coefficient", "CI_Lower", "CI_Upper", "p_value_FDR",
        "Is_Significant_FDR",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(
            f"Paired demographic results are missing columns: {sorted(missing)}")
    data = results[results["Test_Type"] == "coefficient"].dropna(
        subset=["Coefficient", "CI_Lower", "CI_Upper"]).copy()
    if demographic_variable is not None:
        data = data[data["Demographic_Variable"] == demographic_variable]
    if significant_only:
        data = data[data["Is_Significant_FDR"]]
    if data.empty:
        logging.warning("No paired demographic coefficient results to plot")
        return

    data = data.sort_values("Coefficient").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(data))))
    for position, row in data.iterrows():
        color = "#b23a48" if row["Is_Significant_FDR"] else "#526777"
        lower = max(0, row["Coefficient"] - row["CI_Lower"])
        upper = max(0, row["CI_Upper"] - row["Coefficient"])
        ax.errorbar(
            row["Coefficient"], position, xerr=[[lower], [upper]],
            fmt="o", color=color, ecolor=color, capsize=3,
        )
    first = data.iloc[0]
    labels = [
        f"{row['Subset']}: {row['Demographic_Variable']} "
        f"(n={row['N_Donors']}, q={row['p_value_FDR']:.3g})"
        for _, row in data.iterrows()
    ]
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(
        f"Change in {first['Class_A']} − {first['Class_B']} difference")
    ax.set_title("Demographic associations with paired tissue-class differences")
    ax.grid(axis="x", alpha=0.25)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#b23a48",
               markeredgecolor="#b23a48", label="FDR significant (q < 0.05)")
    ]
    if not significant_only and (~data["Is_Significant_FDR"]).any():
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#526777", markeredgecolor="#526777",
                   label="Not FDR significant"))
    ax.legend(handles=legend_handles, title="Result", loc="upper left",
              bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()


def analyze_paired_tissue_classes(
        df, subset_columns, tissue_class_map, class_a="Mucosal",
        class_b=None, min_paired_donors=3, bootstrap_samples=2000,
        random_state=42, alpha=0.05):
    """Summarize each subset's paired mean difference between tissue classes."""
    tissue_class_map = {
        str(tissue).strip(): str(group).strip()
        for tissue, group in tissue_class_map.items()
    }
    class_a = str(class_a).strip()
    available_classes = set(tissue_class_map.values())
    if class_b is None:
        other_classes = available_classes - {class_a}
        if len(other_classes) != 1:
            raise ValueError(
                "Specify class_b when the map does not contain exactly two classes")
        class_b = other_classes.pop()
    class_b = str(class_b).strip()
    if class_a == class_b or not {class_a, class_b}.issubset(available_classes):
        raise ValueError("class_a and class_b must be distinct mapped classes")
    if min_paired_donors < 3 or bootstrap_samples < 1:
        raise ValueError("Use at least 3 paired donors and 1 bootstrap sample")

    results = []
    rng = np.random.default_rng(random_state)
    for subset in subset_columns:
        values = df[["Donor", "Tissue", subset]].copy()
        values["Donor"] = values["Donor"].astype("string").str.strip()
        values["Tissue"] = values["Tissue"].astype("string").str.strip()
        values[subset] = pd.to_numeric(values[subset], errors="coerce")
        values["Tissue_Class"] = values["Tissue"].map(tissue_class_map)
        values = values.dropna(subset=["Donor", subset, "Tissue_Class"])
        values = values[values["Tissue_Class"].isin([class_a, class_b])]
        means = values.groupby(
            ["Donor", "Tissue_Class"], observed=True)[subset].mean()
        paired = means.unstack("Tissue_Class").reindex(
            columns=[class_a, class_b]).dropna(subset=[class_a, class_b])
        difference = (paired[class_a] - paired[class_b]).to_numpy(dtype=float)
        n_donors = len(difference)
        mean = float(np.mean(difference)) if n_donors else np.nan
        median = float(np.median(difference)) if n_donors else np.nan
        se = (float(np.std(difference, ddof=1) / np.sqrt(n_donors))
              if n_donors > 1 else np.nan)
        ci_low = ci_high = p_ttest = p_wilcoxon = np.nan
        if n_donors >= min_paired_donors:
            boot_means = rng.choice(
                difference, size=(bootstrap_samples, n_donors), replace=True
            ).mean(axis=1)
            ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
            p_ttest = float(stats.ttest_1samp(difference, popmean=0).pvalue)
            try:
                p_wilcoxon = float(stats.wilcoxon(difference).pvalue)
            except ValueError:
                pass
        results.append({
            "Subset": subset, "Class_A": class_a, "Class_B": class_b,
            "N_Paired_Donors": n_donors, "Mean_Difference": mean,
            "Median_Difference": median, "SE_Difference": se,
            "CI_Lower": float(ci_low), "CI_Upper": float(ci_high),
            "p_value_ttest": p_ttest, "p_value_wilcoxon": p_wilcoxon,
        })

    output = pd.DataFrame(results)
    for p_col, q_col in [("p_value_ttest", "p_value_FDR"),
                         ("p_value_wilcoxon", "p_value_wilcoxon_FDR")]:
        output[q_col] = np.nan
        valid = output[p_col].notna()
        if valid.any():
            output.loc[valid, q_col] = multipletests(
                output.loc[valid, p_col], alpha=alpha, method="fdr_bh")[1]
    output["Is_Significant_FDR"] = output["p_value_FDR"] < alpha
    return output


def plot_paired_tissue_class_summary(results, significant_only=False):
    """Plot paired mean differences with bootstrap confidence intervals."""
    required = {
        "Subset", "Class_A", "Class_B", "N_Paired_Donors", "Mean_Difference",
        "CI_Lower", "CI_Upper", "p_value_FDR", "Is_Significant_FDR",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Paired tissue results are missing columns: {sorted(missing)}")
    data = results.dropna(
        subset=["Mean_Difference", "CI_Lower", "CI_Upper"]).copy()
    if significant_only:
        data = data[data["Is_Significant_FDR"]]
    if data.empty:
        logging.warning("No paired tissue-class results are available to plot")
        return
    data = data.sort_values("Mean_Difference").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(data))))
    for position, row in data.iterrows():
        color = "#b23a48" if row["Is_Significant_FDR"] else "#526777"
        ax.errorbar(
            row["Mean_Difference"], position,
            xerr=[[max(0, row["Mean_Difference"] - row["CI_Lower"])],
                  [max(0, row["CI_Upper"] - row["Mean_Difference"])]],
            fmt="o", color=color, ecolor=color, capsize=3,
        )
    first = data.iloc[0]
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([
        f"{row['Subset']} (n={row['N_Paired_Donors']}, "
        f"q={row['p_value_FDR']:.3g})" for _, row in data.iterrows()
    ], fontsize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Mean difference ({first['Class_A']} − {first['Class_B']})")
    ax.set_title("Paired tissue-class differences by T-cell subset")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#b23a48",
               markeredgecolor="#b23a48", label="FDR significant (q < 0.05)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#526777",
               markeredgecolor="#526777", label="Not FDR significant"),
    ], title="Result", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()


def stat_boxplot(df, population_name, tissue, sex=None):
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
    sex : str, optional
        If provided, include only donors in this sex group.

    Returns:
    --------
    None
    """

    df, unique_ethnicities, donor_counts = _prepare_ethnicity_boxplot_data(
        df, population_name, tissue, sex=sex, min_donors=3)
    if df.empty:
        group_description = (
            f"tissue {tissue} and sex {sex}"
            if sex is not None else f"tissue {tissue}"
        )
        logging.warning(f"No data available for {group_description}")
        return
    if not unique_ethnicities:
        logging.warning(
            "No ethnicity groups have at least 3 donors for %s in tissue %s",
            population_name, tissue)
        return
    is_outlier = _ethnicity_iqr_outliers(df, population_name)

    outlier_donors = df[is_outlier]["Donor"].unique()
    outlier_str = ", ".join(map(str, outlier_donors))
    logging.info(
        f"Outlier donors for {population_name} in tissue {tissue}: {outlier_str}"
    )

    def draw_boxplot(plot_df, categories):
        if plot_df.empty:
            logging.warning(
                "No observations remain to plot for %s in tissue %s",
                population_name, tissue)
            return

        box_palette = _palette_for_categories(categories, publication_palette)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(
            x="Ethnicity", y=population_name, data=plot_df, hue="Ethnicity",
            order=categories, palette=box_palette, dodge=False,
            boxprops=dict(alpha=0.4), showfliers=True,
            flierprops={
                "marker": "o", "markerfacecolor": "white",
                "markeredgecolor": "black", "markersize": 6,
            },
            ax=ax, zorder=1,
        )
        sns.lineplot(
            x="Ethnicity", y=population_name, data=plot_df, units="Donor",
            estimator=None, color="black", alpha=0.3, linewidth=1,
            ax=ax, zorder=2,
        )
        point_data = plot_df.loc[~is_outlier.loc[plot_df.index]]
        sns.stripplot(
            x="Ethnicity", y=population_name, data=point_data, hue="Ethnicity",
            order=categories, palette=box_palette, legend=False, jitter=True,
            size=6, edgecolor="black", linewidth=0.5, ax=ax, zorder=3,
        )

        _annotate_ethnicity_comparisons(
            ax, plot_df, population_name, categories)

        positions = {ethnicity: i for i,
                     ethnicity in enumerate(categories)}
        for ethnicity, outliers in df.loc[is_outlier].groupby(
                "Ethnicity", sort=False):
            if ethnicity not in positions:
                continue
            for index, (_, outlier) in enumerate(
                    outliers.sort_values(population_name).iterrows()):
                to_right = index % 2 == 0
                ax.annotate(
                    str(outlier["Donor"]),
                    xy=(positions[ethnicity], outlier[population_name]),
                    xytext=(10 if to_right else -10, 5 * (index // 2)),
                    textcoords="offset points",
                    ha="left" if to_right else "right", va="center",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-",
                                "color": "gray", "lw": 0.7},
                    annotation_clip=False, zorder=5,
                )

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(
            [f"{ethnicity}\n(n={donor_counts.get(ethnicity, 0)})"
             for ethnicity in categories],
            rotation=45, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        title_pad = 55
        layout_top = 0.82
        title = f"{population_name} in {tissue}"
        if sex is not None:
            title += f" - {sex}"
        ax.set_title(
            title, fontsize=14, pad=title_pad)
        sns.despine()
        fig.tight_layout(rect=(0, 0, 0.94, layout_top))
        filename = f"{population_name}_{tissue}_boxplot_with_outliers"
        if sex is not None:
            safe_sex = re.sub(r"[^A-Za-z0-9_-]+", "_", str(sex)).strip("_")
            filename += f"_{safe_sex}"
        plt.savefig(
            f"{filename}.png",
            dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    draw_boxplot(df, unique_ethnicities)


def sex_boxplot(df, population_name, tissue, ethnicity=None):
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
    ethnicity : str, optional
        If provided, include only donors with this ethnicity.

    Returns:
    --------
    None
    """

    df = df.dropna(subset=["Sex", "Tissue", population_name, "Donor"])
    df = df[df["Tissue"] == tissue].reset_index(drop=True)
    if ethnicity is not None:
        if "Ethnicity" not in df.columns:
            raise ValueError("The dataset must contain an 'Ethnicity' column")
        df = df[df["Ethnicity"] == ethnicity].reset_index(drop=True)

    if df.empty:
        group_description = (
            f"tissue {tissue} and ethnicity {ethnicity}"
            if ethnicity is not None else f"tissue {tissue}"
        )
        logging.warning(f"No data available for {group_description}")
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
    title = f"{population_name} in {tissue}"
    if ethnicity is not None:
        title += f" ({ethnicity})"
    ax.set_title(title, fontsize=14, pad=55)

    sns.despine()
    fig.tight_layout(rect=(0, 0, 0.9, 0.82))

    filename = f"{population_name}_{tissue}_boxplot"
    if ethnicity is not None:
        safe_ethnicity = re.sub(r"[^A-Za-z0-9_-]+", "_", ethnicity).strip("_")
        filename += f"_{safe_ethnicity}"
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def sex_ethnicity_boxplot(df, population_name, tissue, groups):
    """Compare selected (sex, ethnicity) groups within one tissue.

    ``groups`` is an ordered sequence of pairs, for example
    ``[("FEMALE", "White"), ("MALE", "White")]``.
    Mann–Whitney annotations are only computed for groups with at least three
    donors; all selected groups with data remain visible in the plot.
    """
    if not groups:
        raise ValueError("Provide at least one (sex, ethnicity) group")

    selected_groups = [(str(sex).strip(), str(ethnicity).strip())
                       for sex, ethnicity in groups]
    group_labels = [f"{sex} / {ethnicity}"
                    for sex, ethnicity in selected_groups]
    if len(set(selected_groups)) != len(selected_groups):
        raise ValueError("Selected sex/ethnicity groups must be unique")

    required = ["Sex", "Ethnicity", "Tissue", "Donor", population_name]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    plot_df = df.dropna(subset=required).copy()
    plot_df = plot_df[plot_df["Tissue"] == tissue]
    plot_df["_Sex_Ethnicity_Group"] = (
        plot_df["Sex"].astype("string").str.strip() + " / "
        + plot_df["Ethnicity"].astype("string").str.strip()
    )
    plot_df = plot_df[
        plot_df["_Sex_Ethnicity_Group"].isin(group_labels)
    ].copy()
    present_groups = [label for label in group_labels
                      if label in plot_df["_Sex_Ethnicity_Group"].values]
    if plot_df.empty:
        logging.warning(
            "No data for selected sex/ethnicity groups in tissue %s", tissue)
        return

    group_counts = plot_df.groupby("_Sex_Ethnicity_Group")["Donor"].nunique()
    q1 = plot_df.groupby("_Sex_Ethnicity_Group")[population_name].transform(
        lambda values: values.quantile(0.25))
    q3 = plot_df.groupby("_Sex_Ethnicity_Group")[population_name].transform(
        lambda values: values.quantile(0.75))
    iqr = q3 - q1
    is_outlier = ((plot_df[population_name] < q1 - 1.5 * iqr)
                  | (plot_df[population_name] > q3 + 1.5 * iqr))

    palette = _palette_for_categories(present_groups, {})
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(present_groups)), 8))
    sns.boxplot(
        data=plot_df, x="_Sex_Ethnicity_Group", y=population_name,
        hue="_Sex_Ethnicity_Group", order=present_groups,
        palette=palette, dodge=False, legend=False, showfliers=True,
        boxprops={"alpha": 0.4},
        flierprops={"marker": "o", "markerfacecolor": "white",
                    "markeredgecolor": "black", "markersize": 6},
        ax=ax, zorder=1,
    )
    sns.stripplot(
        data=plot_df.loc[~is_outlier], x="_Sex_Ethnicity_Group",
        y=population_name, hue="_Sex_Ethnicity_Group", order=present_groups,
        palette=palette, legend=False, jitter=True, size=6,
        edgecolor="black", linewidth=0.5, ax=ax, zorder=3,
    )

    testable_groups = [group for group in present_groups
                       if group_counts.get(group, 0) >= 3]
    if len(testable_groups) >= 2:
        annotator = Annotator(
            ax, list(combinations(testable_groups, 2)), data=plot_df,
            x="_Sex_Ethnicity_Group", y=population_name,
            order=present_groups,
        )
        annotator.configure(
            test="Mann-Whitney", text_format="star", loc="outside",
            comparisons_correction="bonferroni", hide_non_significant=True,
            verbose=False, line_offset=0.05, line_height=0.02,
            text_offset=2.0,
        )
        try:
            annotator.apply_and_annotate()
        except Exception as exc:
            logging.warning(
                "Could not annotate selected group comparisons for %s in %s: %s",
                population_name, tissue, exc)

    positions = {group: index for index, group in enumerate(present_groups)}
    for group, outliers in plot_df.loc[is_outlier].groupby(
            "_Sex_Ethnicity_Group", sort=False):
        for index, (_, outlier) in enumerate(
                outliers.sort_values(population_name).iterrows()):
            side = 1 if index % 2 == 0 else -1
            ax.annotate(
                str(outlier["Donor"]),
                xy=(positions[group], outlier[population_name]),
                xytext=(10 * side, 5 * (index // 2)),
                textcoords="offset points", ha="left" if side > 0 else "right",
                va="center", fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.7},
                annotation_clip=False, zorder=5,
            )

    ax.set_xticks(range(len(present_groups)))
    ax.set_xticklabels(
        [f"{group}\n(n={group_counts.get(group, 0)})"
         for group in present_groups], rotation=35, ha="right")
    ax.set_title(f"{population_name} in {tissue}: selected sex/ethnicity groups",
                 fontsize=14, pad=55)
    ax.set_xlabel("Sex / ethnicity")
    ax.set_ylabel(population_name)
    sns.despine()
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    filename_parts = [population_name, tissue, *present_groups]
    filename = "_vs_".join(
        re.sub(r"[^A-Za-z0-9_-]+", "_", str(part)).strip("_")
        for part in filename_parts
    )
    plt.savefig(f"{filename}_boxplot.png", dpi=300, bbox_inches="tight")
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
    """
    Generates a summary plot of group comparison results, showing the adjusted outcome differences
    between reference and comparison groups for each subset and tissue.

    Parameters:
    -----------
    group_comparison_stats_df : pd.DataFrame
        The DataFrame containing group comparison statistics.
    significant_only : bool, optional
        If True, only significant comparisons (based on FDR) will be plotted. Default is
        True.

    Returns:
    --------
    None

    """
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
    data = data[
        (data["N_Reference"] > 2) & (data["N_Comparison"] > 2)
    ]
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
    return None


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
        unique_sexes = list(df_tissue["Sex"].drop_duplicates())
        sns.boxplot(x="Sex", y=safe_pop, data=df_tissue,
                    ax=axes[0],
                    palette=_palette_for_categories(
                        unique_sexes, sex_palette),
                    hue="Sex", order=unique_sexes, legend=False)
        sns.stripplot(x="Sex", y=safe_pop, data=df_tissue,
                      ax=axes[0], color="black", alpha=0.6, jitter=True,
                      order=unique_sexes)
        sex_counts = df_tissue["Sex"].value_counts()
        valid_sexes = [
            sex for sex in unique_sexes
            if sex_counts.get(sex, 0) >= 2
        ]
        if len(valid_sexes) >= 2:
            annotator = Annotator(
                axes[0], list(combinations(valid_sexes, 2)),
                data=df_tissue, x="Sex", y=safe_pop, order=unique_sexes,
            )
            annotator.configure(
                test="Mann-Whitney", text_format="star", loc="outside",
                comparisons_correction="bonferroni", line_width=1.2,
                hide_non_significant=True, verbose=False,
                line_offset=0.05, line_height=0.02, text_offset=2.0,
            )
            try:
                annotator.apply_and_annotate()
            except Exception as exc:
                logging.warning(
                    "Could not annotate sex comparisons for %s in %s: %s",
                    population_name, tissue, exc)
        axes[0].set_title(
            f"Univariable View: {population_name} by Sex\n"
            f"(Confounded / Mann-Whitney Null) in {tissue}", pad=45)
        sex_donor_counts = df_tissue.groupby("Sex")["Donor"].nunique()
        axes[0].set_xticklabels([
            f"{label.get_text()}\n(n={sex_donor_counts.get(label.get_text(), 0)})"
            for label in axes[0].get_xticklabels()
        ])
    elif "Ethnicity" in target_variable:
        ethnicity_data, valid_ethnicities, ethnicity_counts = (
            _prepare_ethnicity_boxplot_data(
                df_working, safe_pop, tissue, min_donors=3)
        )
        if ethnicity_data.empty:
            axes[0].text(
                0.5, 0.5,
                "No ethnicity groups with at least 3 donors",
                ha="center", va="center", transform=axes[0].transAxes,
            )
            axes[0].set_xticks([])
        else:
            ethnicity_palette = _palette_for_categories(
                valid_ethnicities, publication_palette)
            sns.boxplot(
                x="Ethnicity", y=safe_pop, data=ethnicity_data,
                ax=axes[0], hue="Ethnicity", order=valid_ethnicities,
                palette=ethnicity_palette, legend=False, showfliers=True,
                flierprops={
                    "marker": "o", "markerfacecolor": "white",
                    "markeredgecolor": "black", "markersize": 6,
                },
            )
            is_outlier = _ethnicity_iqr_outliers(ethnicity_data, safe_pop)
            sns.stripplot(
                x="Ethnicity", y=safe_pop,
                data=ethnicity_data.loc[~is_outlier],
                ax=axes[0], hue="Ethnicity", order=valid_ethnicities,
                palette=ethnicity_palette, legend=False, jitter=True,
                size=6, edgecolor="black", linewidth=0.5,
            )
            _annotate_ethnicity_comparisons(
                axes[0], ethnicity_data, safe_pop, valid_ethnicities)

            axes[0].set_xticks(range(len(valid_ethnicities)))
            axes[0].set_xticklabels(
                [f"{ethnicity}\n(n={ethnicity_counts[ethnicity]})"
                 for ethnicity in valid_ethnicities],
                rotation=45, ha="right",
            )
        axes[0].set_title(
            f"Univariable View: {population_name} by Ethnicity in {tissue}",
            pad=45,
        )
    else:
        sns.regplot(x=target_variable, y=safe_pop, data=df_tissue, ax=axes[0],
                    scatter_kws={'alpha': 0.6, 'color': 'gray'}, line_kws={'color': 'red', 'linestyle': '--'})
        axes[0].set_title(
            f"Univariable View: {population_name} vs {target_variable}\n(Unadjusted Structural Noise)")
        axes[0].set_xlabel(f"Raw {target_variable}")

    axes[0].set_ylabel(f"{population_name} Frequency")
    full_model = sm.OLS(y, X).fit(cov_type="HC3")
    beta_coefficient = full_model.params[target_variable]

    other_cols = [col for col in X.columns if col != target_variable]
    outcome_residuals = sm.OLS(y.iloc[:, 0], X[other_cols]).fit().resid
    predictor_residuals = sm.OLS(
        X[target_variable], X[other_cols]).fit().resid
    partial_data = pd.DataFrame({
        "Predictor_Residual": predictor_residuals,
        "Outcome_Residual": outcome_residuals,
        "Ethnicity": df_tissue.loc[y.index, "Ethnicity"],
    })
    ethnicity_order = list(partial_data["Ethnicity"].drop_duplicates())
    ethnicity_palette = _palette_for_categories(
        ethnicity_order, publication_palette)
    line_x = [partial_data["Predictor_Residual"].min(),
              partial_data["Predictor_Residual"].max()]
    axes[1].plot(
        line_x, [beta_coefficient * value for value in line_x],
        color="black", linewidth=1.5, zorder=1,
    )
    sns.scatterplot(
        data=partial_data, x="Predictor_Residual", y="Outcome_Residual",
        hue="Ethnicity", hue_order=ethnicity_order,
        palette=ethnicity_palette, s=65, edgecolor="black", alpha=0.85,
        ax=axes[1], zorder=2,
    )

    clean_label = (
        target_variable.rsplit("[T.", 1)[1].rstrip("]")
        if "[T." in target_variable else target_variable
    )
    contrast = " vs White" if "Ethnicity" in target_variable else (
        " vs Male" if "Sex" in target_variable else "")
    q_label = (
        f", BH q = {adjusted_p_value:.4f}"
        if adjusted_p_value is not None else ""
    )
    axes[1].set_title(
        f"Adjusted OLS: {clean_label}{contrast} in {tissue}\n"
        f"β = {beta_coefficient:.2f}{q_label}", pad=20)
    axes[1].set_xlabel(f"Residuals of {clean_label}")
    axes[1].set_ylabel(f"Residuals of {population_name}")
    axes[1].legend(title="Ethnicity", bbox_to_anchor=(
        1.02, 1), loc="upper left")

    sns.despine()
    plt.tight_layout(rect=(0, 0, 0.9, 0.84))

    clean_filename = (target_variable
                      .replace("C(", "")
                      .replace(")[T.", "_")
                      .replace("]", "")
                      .replace(" ", "_")
                      .replace("/", "_"))
    plt.savefig(f"{safe_pop}_{tissue}_{clean_filename}_comparison_panel.png",
                dpi=300, bbox_inches="tight")
    plt.show()
