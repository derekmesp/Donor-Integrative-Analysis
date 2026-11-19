
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
antibody_status_palette = {
    "Negative": "#009E73",  # Teal
    "Positive": "#CC79A7",  # Orchid
    "Not tested": "#8C8C8C",  # Gray
}
ANTIBODY_STATUS_COLUMNS = {
    "Ab CMV:": "CMV_Status",
    "Ab EBV (IgG):": "EBV_IgG_Status",
}


def _add_antibody_covariates(data):
    """
    Add formula-safe antibody statuses to the data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing the data.

    Returns:
    --------
    tuple
        A tuple containing the modified DataFrame, a list of formula-safe column names,
        and a list of the new antibody status column names.

    """
    data = data.copy()
    status_columns = []
    for source, target in ANTIBODY_STATUS_COLUMNS.items():
        if source not in data.columns:
            continue

        values = data[source].astype("string").str.strip()
        normalized = values.str.casefold()
        status = values.copy()
        status = status.mask(normalized.isin({"positive", "pos"}), "Positive")
        status = status.mask(normalized.isin({"negative", "neg"}), "Negative")
        status = status.mask(
            normalized.isin({"not done", "not tested", "not performed"}),
            "Not tested")
        status = status.mask(normalized.isin({"", "unknown", "n/a", "na"}),
                             pd.NA)
        data[target] = status
        status_columns.append(target)
    return data, [f"C({column})" for column in status_columns], status_columns


def _format_model_term(term):
    """
    Convert a model term to a more readable format.

    Parameters:
    -----------
    term : str
        The model term to be formatted.

    Returns:
    --------
    str
        A more readable version of the model term, if applicable. If the term does not match
        any known patterns, it is returned unchanged.
    """
    for column, label in [("CMV_Status", "CMV status"),
                          ("EBV_IgG_Status", "EBV IgG status")]:
        marker = f"C({column})[T."
        if marker in term:
            return f"{label}: {term.split(marker, 1)[1].rstrip(']')}"
    return term


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
    """
    Prepare data for ethnicity boxplots by filtering and cleaning the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    population_name : str
        The name of the column containing the population data.
    tissue : str
        The name of the tissue to filter by.
    sex : str, optional
        The sex to filter by, if applicable.
    min_donors : int, default=3
        The minimum number of donors required for each ethnicity group.

    Returns:
    --------
    tuple
        A tuple containing the filtered DataFrame, the list of categories, and the donor counts.
    """
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
    """
    Identify outliers in the population data based on the interquartile range (IQR) for each ethnicity group.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the data to be analyzed.
    population_name : str
        The name of the column containing the population data.

    Returns:
    --------
    pandas.Series
        A boolean Series indicating whether each row in the DataFrame is an outlier (True) or not (False).
    """

    grouped = data.groupby("Ethnicity")[population_name]
    q1 = grouped.transform(lambda values: values.quantile(0.25))
    q3 = grouped.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    return ((data[population_name] < q1 - 1.5 * iqr)
            | (data[population_name] > q3 + 1.5 * iqr))


def _annotate_ethnicity_comparisons(ax, data, population_name, categories):
    """
    Annotate pairwise comparisons between ethnicity groups on a boxplot.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object on which to annotate.
    data : pandas.DataFrame
        The DataFrame containing the data to be annotated.
    population_name : str
        The name of the column containing the population data.
    categories : list
        The list of ethnicity categories to compare.

    Returns:
    --------
    None
        The function modifies the axes object in place and does not return a value.

    """
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


def _permutation_ethnicity_status_test(
        ethnicities, statuses, n_permutations, rng):
    """
    Perform a permutation test for association between ethnicity and antibody status.

    Parameters:
    -----------
    ethnicities : array-like
        An array of ethnicity labels for each donor.
    statuses : array-like
        An array of antibody status labels for each donor.
    n_permutations : int
        The number of permutations to perform for the test.
    rng : numpy.random.Generator
        A random number generator for reproducibility.

    Returns:
    --------
    float
        The p-value from the permutation test, or NaN if the test is not applicable due to insufficient data.
    """
    group_levels = sorted(pd.Series(ethnicities).dropna().unique())
    group_codes = pd.Categorical(ethnicities, categories=group_levels).codes
    status_codes = pd.Categorical(
        statuses, categories=["Negative", "Positive"]).codes
    valid = (group_codes >= 0) & (status_codes >= 0)
    group_codes, status_codes = group_codes[valid], status_codes[valid]
    if (len(group_levels) < 2 or len(np.unique(status_codes)) < 2
            or len(group_codes) < 3):
        return np.nan

    observed = np.zeros((len(group_levels), 2), dtype=int)
    np.add.at(observed, (group_codes, status_codes), 1)

    def pearson_statistic(table):
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
        valid_cells = expected > 0
        return float(np.sum(
            (table[valid_cells] - expected[valid_cells]) ** 2
            / expected[valid_cells]))

    observed_statistic = pearson_statistic(observed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted_groups = rng.permutation(group_codes)
        permuted = np.zeros_like(observed)
        np.add.at(permuted, (permuted_groups, status_codes), 1)
        exceedances += pearson_statistic(permuted) >= observed_statistic
    return (exceedances + 1) / (n_permutations + 1)


def plot_antibody_status_by_ethnicity(
        df, n_permutations=10000, random_state=42,
        min_donors_per_ethnicity=2):
    """
    Plot CMV and EBV IgG antibody status by donor ethnicity, with permutation tests.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing donor data, including ethnicity and antibody status.
    n_permutations : int, optional
        The number of permutations to perform for the permutation test. Default is 10,000.
    random_state : int, optional
        The random seed for reproducibility of permutation tests. Default is 42.
    min_donors_per_ethnicity : int, optional
        The minimum number of donors required for each ethnicity group to be included in the analysis. Default is 2.

    Returns:
    --------
    tuple
        A tuple containing the figure and axes objects of the generated plots.
    """
    if "Ethnicity" not in df.columns and "Ethnicity:" in df.columns:
        df = df.rename(columns={"Ethnicity:": "Ethnicity"})
    required = {"Donor", "Ethnicity", *ANTIBODY_STATUS_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data, _, status_columns = _add_antibody_covariates(df)
    plot_columns = ["Donor", "Ethnicity", *status_columns]
    data = data[plot_columns].copy()
    data["Donor"] = data["Donor"].astype("string").str.strip()
    data["Ethnicity"] = data["Ethnicity"].astype("string").str.strip()
    data = data.dropna(subset=["Donor"])

    donor_data = data.groupby("Donor", sort=False).agg(
        lambda values: values.dropna().iloc[0]
        if values.notna().any() else pd.NA
    ).reset_index()
    ethnicity_conflicts = data.groupby("Donor")["Ethnicity"].nunique(
        dropna=True).gt(1)
    if ethnicity_conflicts.any():
        logging.warning(
            "Donors with conflicting ethnicity labels; using first observed: %s",
            ethnicity_conflicts[ethnicity_conflicts].index.tolist())

    donor_data = donor_data.dropna(subset=["Ethnicity"])
    ethnicity_counts = donor_data.groupby("Ethnicity")["Donor"].nunique()
    excluded_ethnicities = ethnicity_counts[
        ethnicity_counts < min_donors_per_ethnicity].index.tolist()
    if excluded_ethnicities:
        logging.info(
            "Omitting ethnicity groups with fewer than %d donors: %s",
            min_donors_per_ethnicity, excluded_ethnicities)
        donor_data = donor_data[
            ~donor_data["Ethnicity"].isin(excluded_ethnicities)]
    ethnicities = sorted(donor_data["Ethnicity"].dropna().unique())
    if not ethnicities:
        logging.warning("No donor ethnicity data available for antibody plot")
        return
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    status_order = ["Negative", "Positive", "Not tested"]
    observed_statuses = set()
    count_tables = {}
    p_values = {}
    rng = np.random.default_rng(random_state)
    for status_column in status_columns:
        known = donor_data.dropna(subset=[status_column])
        table = pd.crosstab(known["Ethnicity"], known[status_column])
        extra_statuses = sorted(set(table.columns).difference(status_order))
        order = [status for status in status_order if status in table.columns]
        order.extend(extra_statuses)
        table = table.reindex(index=ethnicities, columns=order, fill_value=0)
        count_tables[status_column] = table
        observed_statuses.update(order)

        tested = known[known[status_column].isin(["Negative", "Positive"])]
        p_values[status_column] = _permutation_ethnicity_status_test(
            tested["Ethnicity"].to_numpy(),
            tested[status_column].to_numpy(), n_permutations, rng)

    q_values = {column: np.nan for column in status_columns}
    valid_tests = [column for column in status_columns
                   if pd.notna(p_values[column])]
    if valid_tests:
        adjusted = multipletests(
            [p_values[column] for column in valid_tests], method="fdr_bh")[1]
        q_values.update(dict(zip(valid_tests, adjusted)))

    status_colors = dict(antibody_status_palette)
    extras = sorted(observed_statuses.difference(status_colors))
    extra_colors = sns.color_palette("husl", n_colors=len(extras))
    status_colors.update(dict(zip(extras, extra_colors)))

    fig, axes = plt.subplots(1, len(status_columns),
                             figsize=(13, 5), squeeze=False)
    axes = axes[0]
    titles = {"CMV_Status": "CMV", "EBV_IgG_Status": "EBV IgG"}
    for ax, status_column in zip(axes, status_columns):
        counts = count_tables[status_column]
        totals = counts.sum(axis=1)
        percentages = counts.div(totals.replace(
            0, np.nan), axis=0).fillna(0) * 100
        bottoms = np.zeros(len(ethnicities))
        x = np.arange(len(ethnicities))
        for status in counts.columns:
            heights = percentages[status].to_numpy(dtype=float)
            ax.bar(
                x, heights, bottom=bottoms, color=status_colors[status],
                label=status, edgecolor="white", linewidth=0.6)
            bottoms += heights
        for position, total in enumerate(totals):
            ax.text(position, 102, f"n={int(total)}", ha="center", va="bottom",
                    fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ethnicities, rotation=35, ha="right")
        ax.set_ylim(0, 114)
        ax.set_ylabel("Donors (%)")
        p_value = p_values[status_column]
        q_value = q_values[status_column]
        test_text = (
            f"Positive vs negative: permutation p={p_value:.3g}, "
            f"BH q={q_value:.3g}"
            if pd.notna(p_value) else "Positive vs negative: not testable")
        ax.set_title(
            f"{titles.get(status_column, status_column)}\n{test_text}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    legend_statuses = [status for status in status_order
                       if status in observed_statuses]
    legend_statuses.extend(extras)
    if legend_statuses:
        handles = [Line2D(
            [0], [0], marker="s", linestyle="none", markerfacecolor=status_colors[status],
            markeredgecolor=status_colors[status], label=status)
            for status in legend_statuses]
        fig.legend(handles=handles, title="Antibody status", loc="center left",
                   bbox_to_anchor=(0.88, 0.5), frameon=False)
    fig.suptitle("CMV and EBV IgG status by ethnicity", y=1.02)
    fig.text(
        0.5, 0.01,
        "Each donor counted once; n is the number with a recorded result. "
        "Bars include Not tested; the permutation test compares positive vs "
        "negative among tested donors (BH correction across CMV and EBV IgG). "
        "Missing clinical results are excluded.",
        ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 0.86, 0.96))
    plt.show()
    return fig, axes


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


def summarize_donor_cohort(
        df, subset_columns=None, categorical_columns=None,
        continuous_columns=None):
    """
    Summarize donor cohort characteristics, including categorical and continuous variables.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing donor data.
    subset_columns : list, optional
        List of subset column names to summarize. If None, no subset columns are summarized.
    categorical_columns : list, optional
        List of categorical column names to summarize. If None, defaults to ["Sex", "Ethnicity"].
    continuous_columns : list, optional
        List of continuous column names to summarize. If None, defaults to ["Age", "BMI"].

    Returns:
    --------
    dict
        A dictionary containing summary DataFrames for donor characteristics, including:
        - "donor_summary": Summary of donor metadata and tissue counts.
        - "group_counts": Counts and percentages of donors by categorical variables.
        - "continuous_summary": Summary statistics for continuous variables.
        - "missingness": Counts and percentages of missing data for each variable.
    """
    categorical_columns = list(categorical_columns or ["Sex", "Ethnicity"])
    continuous_columns = list(continuous_columns or ["Age", "BMI"])
    subset_columns = list(subset_columns or [])
    required = ["Donor", "Tissue", *categorical_columns,
                *continuous_columns, *subset_columns]
    missing_columns = [
        column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}")

    data = df[required].copy()
    data["Donor"] = data["Donor"].astype("string").str.strip()
    data["Tissue"] = data["Tissue"].astype("string").str.strip()
    data = data.dropna(subset=["Donor"])
    for column in categorical_columns:
        data[column] = data[column].astype("string").str.strip()
    for column in continuous_columns + subset_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    metadata_columns = categorical_columns + continuous_columns
    donor_metadata = data.groupby("Donor", sort=False)[metadata_columns].agg(
        lambda values: values.dropna().iloc[0]
        if values.notna().any() else np.nan)
    conflicting = data.groupby("Donor", sort=False)[metadata_columns].nunique(
        dropna=True).gt(1)
    if conflicting.to_numpy().any():
        conflict_counts = conflicting.sum(axis=0)
        logging.warning(
            "Donors with inconsistent metadata values (using first non-missing): %s",
            conflict_counts[conflict_counts > 0].to_dict())

    donor_tissue_count = data.dropna(subset=["Tissue"]).groupby(
        "Donor")["Tissue"].nunique()
    donor_summary = donor_metadata.copy()
    donor_summary["N_Tissues"] = donor_tissue_count.reindex(
        donor_summary.index, fill_value=0)
    if subset_columns:
        numeric_values = data[subset_columns]
        donor_summary["N_Subsets_Measured"] = numeric_values.notna().groupby(
            data["Donor"]).max().sum(axis=1).reindex(donor_summary.index)
        donor_summary["N_Subsets_Total"] = len(subset_columns)
        donor_summary["Subset_Coverage"] = (
            donor_summary["N_Subsets_Measured"] / len(subset_columns))
    donor_summary = donor_summary.reset_index()

    group_rows = []
    for column in categorical_columns:
        values = donor_metadata[column].fillna("Missing").astype(str)
        counts = values.value_counts(dropna=False)
        for category, count in counts.items():
            group_rows.append({
                "Variable": column, "Category": category,
                "N_Donors": int(count),
                "Percent_Donors": float(count / max(len(donor_metadata), 1) * 100),
            })
    group_counts = pd.DataFrame(group_rows, columns=[
        "Variable", "Category", "N_Donors", "Percent_Donors"])

    continuous_rows = []
    for column in continuous_columns:
        values = pd.to_numeric(
            donor_metadata[column], errors="coerce").dropna()
        continuous_rows.append({
            "Variable": column, "N_Donors": int(values.size),
            "N_Missing": int(len(donor_metadata) - values.size),
            "Mean": float(values.mean()) if not values.empty else np.nan,
            "SD": float(values.std()) if values.size > 1 else np.nan,
            "Median": float(values.median()) if not values.empty else np.nan,
            "Min": float(values.min()) if not values.empty else np.nan,
            "Max": float(values.max()) if not values.empty else np.nan,
        })
    continuous_summary = pd.DataFrame(continuous_rows)

    missing_rows = []
    for column in required:
        row_missing = data[column].isna()
        donor_missing = data.assign(_Missing=row_missing).groupby(
            "Donor")["_Missing"].all()
        missing_rows.append({
            "Variable": column,
            "N_Missing_Rows": int(row_missing.sum()),
            "Percent_Missing_Rows": float(row_missing.mean() * 100),
            "N_Donors_Missing_All_Records": int(donor_missing.sum()),
        })
    missingness = pd.DataFrame(missing_rows)
    donor_count = int(data["Donor"].nunique())
    return {
        "donor_summary": donor_summary,
        "group_counts": group_counts,
        "continuous_summary": continuous_summary,
        "missingness": missingness,
        "n_donors": donor_count,
        "n_donor_tissue_rows": int(data.dropna(subset=["Tissue"])[
            ["Donor", "Tissue"]].drop_duplicates().shape[0]),
    }


def analyze_demographic_mixed_models(
        df, subset_columns, min_donors=20, min_group_donors=5,
        alpha=0.05, ethnicity_reference="White", sex_reference="MALE",
        return_marginal_means=False):
    """
    Analyze demographic effects on continuous outcomes using mixed-effects models.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing donor data, including demographic and outcome variables.
    subset_columns : list
        List of continuous outcome column names to analyze.
    min_donors : int, optional
        Minimum number of donors required for analysis. Default is 20.
    min_group_donors : int, optional
        Minimum number of donors required per demographic group. Default is 5.
    alpha : float, optional
        Significance level for statistical tests. Default is 0.05.
    ethnicity_reference : str, optional
        Reference category for ethnicity in the mixed-effects model. Default is "White".
    sex_reference : str, optional
        Reference category for sex in the mixed-effects model. Default is "MALE".
    return_marginal_means : bool, optional
        Whether to return estimated marginal means for each subset. Default is False.

    Returns:
    --------
    tuple
        A tuple containing:
        - all_results : pd.DataFrame
            Summary of mixed-effects model results for each subset.
        - statuses : pd.DataFrame
            Status of each subset analysis, including reasons for skipping or failure.
    """
    subset_columns = list(subset_columns)
    if not subset_columns:
        raise ValueError("Provide at least one subset column")
    required = {"Donor", "Tissue", "Sex", "Ethnicity", "Age", "BMI",
                *subset_columns}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if min_donors < 10 or min_group_donors < 3:
        raise ValueError("Use at least 10 total and 3 donors per group")

    df, antibody_terms, antibody_columns = _add_antibody_covariates(df)
    metadata_columns = ["Sex", "Ethnicity", "Age", "BMI", *antibody_columns]
    base = df[["Donor", "Tissue", *metadata_columns]].copy()
    base["Donor"] = base["Donor"].astype("string").str.strip()
    base["Tissue"] = base["Tissue"].astype("string").str.strip()
    for column in ["Sex", "Ethnicity"]:
        base[column] = base[column].astype("string").str.strip()
    for column in ["Age", "BMI"]:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    metadata = base.dropna(subset=["Donor"]).groupby(
        "Donor", sort=False)[metadata_columns].agg(
            lambda values: values.dropna().iloc[0]
            if values.notna().any() else np.nan).reset_index()

    all_results = []
    statuses = []
    marginal_mean_rows = []
    for subset in subset_columns:
        values = df[["Donor", "Tissue", subset]].copy()
        values["Donor"] = values["Donor"].astype("string").str.strip()
        values["Tissue"] = values["Tissue"].astype("string").str.strip()
        values[subset] = pd.to_numeric(values[subset], errors="coerce")
        values = values.dropna(subset=["Donor", "Tissue", subset])
        values = values.groupby(["Donor", "Tissue"], as_index=False,
                                sort=False)[subset].mean()
        model_data = values.merge(metadata, on="Donor", how="inner").dropna(
            subset=["Sex", "Ethnicity", "Age", "BMI",
                    *antibody_columns])
        model_data = model_data.rename(columns={subset: "Outcome"})
        excluded_groups = []
        for variable in ["Sex", "Ethnicity"]:
            donor_counts = model_data[["Donor", variable]].drop_duplicates()[
                variable].value_counts()
            rare = donor_counts[donor_counts < min_group_donors].index.tolist()
            if rare:
                excluded_groups.extend(
                    [f"{variable}={group}" for group in rare])
                model_data = model_data[~model_data[variable].isin(rare)]
        n_donors = model_data["Donor"].nunique()
        if n_donors < min_donors:
            statuses.append({"Subset": subset, "Status": "skipped",
                             "Reason": "too few complete donors",
                             "N_Donors": n_donors,
                             "N_Observations": len(model_data),
                             "Excluded_Groups": "; ".join(excluded_groups)})
            continue
        if ethnicity_reference not in set(model_data["Ethnicity"]):
            statuses.append({"Subset": subset, "Status": "skipped",
                             "Reason": f"reference ethnicity {ethnicity_reference!r} absent",
                             "N_Donors": n_donors,
                             "N_Observations": len(model_data),
                             "Excluded_Groups": "; ".join(excluded_groups)})
            continue
        if sex_reference not in set(model_data["Sex"]):
            statuses.append({"Subset": subset, "Status": "skipped",
                             "Reason": f"reference sex {sex_reference!r} absent",
                             "N_Donors": n_donors,
                             "N_Observations": len(model_data),
                             "Excluded_Groups": "; ".join(excluded_groups)})
            continue

        formula = (
            "Outcome ~ C(Sex, Treatment(reference='" + sex_reference + "')) + "
            "C(Ethnicity, Treatment(reference='" + ethnicity_reference + "')) + "
            "Age + BMI + C(Tissue)"
        )
        if antibody_terms:
            formula += " + " + " + ".join(antibody_terms)
        try:
            model = smf.mixedlm(
                formula, data=model_data, groups=model_data["Donor"]
            ).fit(reml=False, method="lbfgs", disp=False)
            if not model.converged:
                statuses.append({"Subset": subset, "Status": "failed",
                                 "Reason": "model did not converge",
                                 "N_Donors": n_donors,
                                 "N_Observations": len(model_data),
                                 "Excluded_Groups": "; ".join(excluded_groups)})
                continue
        except Exception as exc:
            statuses.append({"Subset": subset, "Status": "failed",
                             "Reason": str(exc), "N_Donors": n_donors,
                             "N_Observations": len(model_data),
                             "Excluded_Groups": "; ".join(excluded_groups)})
            continue

        fixed_terms = list(model.fe_params.index)
        covariance = model.cov_params().loc[
            fixed_terms, fixed_terms].to_numpy(dtype=float)
        standardization_data = model_data.copy()
        tissue_counts = standardization_data.groupby("Donor")["Tissue"].transform(
            "nunique")
        donor_weights = 1.0 / tissue_counts.to_numpy(dtype=float)
        design_info = model.model.data.design_info
        for demographic, reference in [
                ("Ethnicity", ethnicity_reference), ("Sex", sex_reference)]:
            if not any(term.startswith(f"C({demographic},")
                       for term in fixed_terms):
                continue
            for level in sorted(standardization_data[demographic].dropna().unique()):
                prediction_data = standardization_data.copy()
                prediction_data[demographic] = level
                design = build_design_matrices(
                    [design_info], prediction_data, return_type="dataframe")[0]
                design = design[fixed_terms].to_numpy(dtype=float)
                average_design = np.average(
                    design, axis=0, weights=donor_weights)
                estimate = float(average_design @ model.fe_params.to_numpy())
                variance = float(average_design @ covariance @ average_design)
                standard_error = np.sqrt(max(variance, 0.0))
                critical_value = stats.norm.ppf(1 - alpha / 2)
                marginal_mean_rows.append({
                    "Subset": subset,
                    "Demographic": demographic,
                    "Group": str(level),
                    "Reference_Group": reference,
                    "Adjusted_Mean": estimate,
                    "Std_Err": standard_error,
                    "CI_Lower": estimate - critical_value * standard_error,
                    "CI_Upper": estimate + critical_value * standard_error,
                    "N_Donors_Group": int(standardization_data.loc[
                        standardization_data[demographic] == level,
                        "Donor"].nunique()),
                    "N_Donors_Model": n_donors,
                })

        for term in fixed_terms:
            if term == "Intercept" or term.startswith("C(Tissue)"):
                continue
            estimate = float(model.fe_params[term])
            interval = model.conf_int().loc[term]
            if term.startswith("C(Sex,"):
                label = "Sex: " + term.rsplit("[T.", 1)[1].rstrip("]")
            elif term.startswith("C(Ethnicity,"):
                label = "Ethnicity: " + term.rsplit("[T.", 1)[1].rstrip("]")
            else:
                label = _format_model_term(term)
            all_results.append({
                "Subset": subset, "Demographic_Variable": label,
                "Test_Type": "coefficient", "Coefficient": estimate,
                "Std_Err": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "CI_Lower": float(interval.iloc[0]),
                "CI_Upper": float(interval.iloc[1]),
                "N_Donors": n_donors, "N_Observations": len(model_data),
                "Excluded_Groups": "; ".join(excluded_groups),
            })

        for variable in ["Sex", "Ethnicity"]:
            terms = [term for term in fixed_terms
                     if term.startswith(f"C({variable},")]
            if not terms:
                continue
            parameter_names = list(model.params.index)
            restriction = np.zeros((len(terms), len(parameter_names)))
            for row, term in enumerate(terms):
                restriction[row, parameter_names.index(term)] = 1.0
            try:
                p_value = float(model.wald_test(
                    restriction, use_f=False, scalar=True).pvalue)
            except Exception as exc:
                logging.warning("Could not run %s omnibus test for %s: %s",
                                variable, subset, exc)
                continue
            all_results.append({
                "Subset": subset,
                "Demographic_Variable": f"{variable}: omnibus",
                "Test_Type": "omnibus", "Coefficient": np.nan,
                "Std_Err": np.nan, "p_value": p_value,
                "CI_Lower": np.nan, "CI_Upper": np.nan,
                "N_Donors": n_donors, "N_Observations": len(model_data),
                "Excluded_Groups": "; ".join(excluded_groups),
            })
        statuses.append({"Subset": subset, "Status": "fit",
                         "Reason": "", "N_Donors": n_donors,
                         "N_Observations": len(model_data),
                         "Excluded_Groups": "; ".join(excluded_groups)})

    result_columns = [
        "Subset", "Demographic_Variable", "Test_Type", "Coefficient",
        "Std_Err", "p_value", "CI_Lower", "CI_Upper", "N_Donors",
        "N_Observations", "Excluded_Groups", "p_value_FDR",
        "Is_Significant_FDR",
    ]
    results = pd.DataFrame(all_results)
    if results.empty:
        results = pd.DataFrame(columns=result_columns)
    else:
        valid = results["p_value"].notna()
        results["p_value_FDR"] = np.nan
        if valid.any():
            results.loc[valid, "p_value_FDR"] = multipletests(
                results.loc[valid, "p_value"], alpha=alpha,
                method="fdr_bh")[1]
        results["Is_Significant_FDR"] = results["p_value_FDR"] < alpha
        results = results[result_columns]
    status_table = pd.DataFrame(statuses, columns=[
        "Subset", "Status", "Reason", "N_Donors", "N_Observations",
        "Excluded_Groups"])
    if return_marginal_means:
        marginal_means = pd.DataFrame(marginal_mean_rows, columns=[
            "Subset", "Demographic", "Group", "Reference_Group",
            "Adjusted_Mean", "Std_Err", "CI_Lower", "CI_Upper",
            "N_Donors_Group", "N_Donors_Model"])
        return results, status_table, marginal_means
    return results, status_table


def plot_adjusted_demographic_means(
        marginal_means, results, demographics=("Ethnicity", "Sex"),
        significant_only=True, alpha=0.05):
    """
    Plot adjusted means for demographic groups by subset, with confidence intervals.

    Parameters:
    -----------
    marginal_means : pd.DataFrame
        DataFrame containing estimated marginal means for each subset and demographic group.
    results : pd.DataFrame
        DataFrame containing mixed-model results, including p-values and significance indicators.
    demographics : tuple, optional
        Tuple of demographic variables to plot. Default is ("Ethnicity", "Sex").
    significant_only : bool, optional
        If True, only plot subsets with at least one FDR-significant test. Default is True.
    alpha : float, optional
        Significance level for determining FDR significance. Default is 0.05.

    Returns:
    --------
    dict
        A dictionary mapping each demographic variable to a tuple of (figure, axes) for the generated
        plots. Each figure contains subplots for each subset, showing adjusted means with confidence intervals
        and annotations for significant tests.
    """
    required_means = {
        "Subset", "Demographic", "Group", "Reference_Group",
        "Adjusted_Mean", "CI_Lower", "CI_Upper", "N_Donors_Group",
    }
    required_results = {
        "Subset", "Demographic_Variable", "Test_Type", "p_value_FDR",
        "Is_Significant_FDR",
    }
    missing = sorted(required_means.difference(marginal_means.columns))
    if missing:
        raise ValueError(f"Marginal means are missing columns: {missing}")
    missing = sorted(required_results.difference(results.columns))
    if missing:
        raise ValueError(f"Mixed-model results are missing columns: {missing}")

    figures = {}
    for demographic in demographics:
        omnibus = results[
            results["Test_Type"].eq("omnibus")
            & results["Demographic_Variable"].eq(f"{demographic}: omnibus")
        ]
        coefficients = results[
            results["Test_Type"].eq("coefficient")
            & results["Demographic_Variable"].astype(str).str.startswith(
                f"{demographic}:")
        ]
        relevant_subsets = set(coefficients.loc[
            coefficients["Is_Significant_FDR"], "Subset"])
        relevant_subsets.update(omnibus.loc[
            omnibus["Is_Significant_FDR"], "Subset"])
        if not significant_only:
            relevant_subsets.update(coefficients["Subset"])
        data = marginal_means[
            marginal_means["Demographic"].eq(demographic)
            & marginal_means["Subset"].isin(relevant_subsets)
        ].copy()
        if data.empty:
            continue
        subsets = data["Subset"].drop_duplicates().tolist()
        fig, axes = plt.subplots(
            len(subsets), 1,
            figsize=(9, max(3.2, 2.8 * len(subsets))),
            squeeze=False)
        axes = axes.ravel()
        palette_base = (publication_palette if demographic == "Ethnicity"
                        else sex_palette)
        for ax, subset in zip(axes, subsets):
            panel = data[data["Subset"] == subset].copy()
            reference = str(panel.iloc[0]["Reference_Group"])
            levels = panel["Group"].tolist()
            levels = [reference] + sorted(
                level for level in levels if level != reference)
            panel = panel.set_index("Group").reindex(levels).dropna(
                subset=["Adjusted_Mean"])
            x = np.arange(len(panel))
            for position, (_, row) in enumerate(panel.iterrows()):
                color = _palette_for_categories(
                    [row.name], palette_base)[row.name]
                lower = max(0, row["Adjusted_Mean"] - row["CI_Lower"])
                upper = max(0, row["CI_Upper"] - row["Adjusted_Mean"])
                ax.errorbar(
                    position, row["Adjusted_Mean"],
                    yerr=[[lower], [upper]], fmt="o", color=color,
                    ecolor=color, capsize=4, markersize=7)
            ax.set_xticks(x)
            ax.set_xticklabels([
                f"{row.name}\n(n={int(row['N_Donors_Group'])})"
                for _, row in panel.iterrows()
            ])
            ax.set_ylabel("Adjusted mean frequency")
            ax.set_title(subset, loc="left", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlim(-0.5, len(panel) - 0.5)

            subset_tests = results[results["Subset"] == subset]
            omnibus_row = subset_tests[
                subset_tests["Test_Type"].eq("omnibus")
                & subset_tests["Demographic_Variable"].eq(
                    f"{demographic}: omnibus")]
            contrast_rows = subset_tests[
                subset_tests["Test_Type"].eq("coefficient")
                & subset_tests["Demographic_Variable"].astype(str).str.startswith(
                    f"{demographic}:")]
            annotations = []
            if not omnibus_row.empty:
                annotations.append(
                    f"omnibus q={omnibus_row.iloc[0]['p_value_FDR']:.3g}")
            significant_contrasts = contrast_rows[
                contrast_rows["Is_Significant_FDR"]]
            for _, contrast in significant_contrasts.iterrows():
                group = contrast["Demographic_Variable"].split(": ", 1)[-1]
                annotations.append(
                    f"{group} vs {reference} q={contrast['p_value_FDR']:.3g}")
            if annotations:
                ax.text(0.99, 0.98, "\n".join(annotations),
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8)
        fig.suptitle(f"Adjusted group means by {demographic}", y=1.01)
        fig.tight_layout()
        plt.show()
        figures[demographic] = (fig, axes)
    return figures


def plot_demographic_effect_heatmap(
        results, demographics=("Ethnicity", "Sex"),
        significant_only=False):
    """Heatmap of adjusted categorical demographic coefficients by subset.

    Cell text shows coefficient size and an asterisk for FDR-significant tests.
    Set ``significant_only=True`` to mask non-significant coefficient cells.
    """
    required = {
        "Subset", "Demographic_Variable", "Test_Type", "Coefficient",
        "p_value_FDR", "Is_Significant_FDR",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Mixed-model results are missing columns: {missing}")
    data = results[
        results["Test_Type"].eq("coefficient")
        & results["Demographic_Variable"].astype(str).str.startswith(
            tuple(f"{name}:" for name in demographics))
    ].dropna(subset=["Coefficient"]).copy()
    if data.empty:
        logging.warning("No sex/ethnicity coefficients available for heatmap")
        return
    data["Contrast"] = data["Demographic_Variable"].astype(str)
    data = data.sort_values(["Subset", "Contrast"])
    matrix = data.pivot(index="Subset", columns="Contrast",
                        values="Coefficient")
    significance = data.pivot(index="Subset", columns="Contrast",
                              values="Is_Significant_FDR").reindex_like(matrix)
    if significant_only:
        matrix = matrix.where(significance)
    annotations = np.empty(matrix.shape, dtype=object)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if pd.isna(value):
                annotations[row, column] = ""
            else:
                star = "*" if bool(significance.iloc[row, column]) else ""
                annotations[row, column] = f"{value:.2f}{star}"
    finite_effects = np.abs(matrix.to_numpy(dtype=float))
    finite_effects = finite_effects[np.isfinite(finite_effects)]
    limit = finite_effects.max() if finite_effects.size else 0.0
    if limit == 0:
        limit = 1.0
    width = max(8, 0.9 * len(matrix.columns) + 5)
    height = max(5, 0.34 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix, mask=matrix.isna(), cmap="RdBu_r", center=0,
        vmin=-limit, vmax=limit, annot=annotations, fmt="",
        linewidths=0.35, linecolor="white", cbar_kws={
            "label": "Adjusted coefficient (frequency units)"}, ax=ax)
    ax.set_title("Adjusted sex and ethnicity effect sizes by T-cell subset")
    ax.set_xlabel(
        "Adjusted comparison (reference: White or MALE); * FDR q < 0.05")
    ax.set_ylabel("T-cell subset")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.show()
    return fig, ax


def _analyze_group_tissue_interaction(
        df, population_name, group_column, focal_group, reference_group,
        min_donors_per_group=5,
        min_total_donors=20, alpha=0.05, reference_tissue=None):
    """
    Analyze interaction between a demographic group and tissue for a given population.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing donor data, including group and tissue information.
    population_name : str
        The name of the population column to analyze (e.g., "CD4_Tcell_Frequency").
    group_column : str
        The name of the column containing the group information (e.g., "Ethnicity" or
        "Sex").
    focal_group : str
        The name of the focal group to compare against the reference group.
    reference_group : str
        The name of the reference group to compare against the focal group.
    min_donors_per_group : int, optional
        Minimum number of donors required per group for inclusion in the analysis. Default is 5.
    min_total_donors : int, optional
        Minimum total number of donors required for inclusion in the analysis. Default is 20.
    alpha : float, optional
        Significance level for statistical tests. Default is 0.05.
    reference_tissue : str, optional
        The name of the reference tissue to use in the interaction analysis. If None, the tissue
        with the most donors will be used as the reference.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the results of the interaction analysis, including p-values and
        adjusted means for each group and tissue.
    """
    required = {"Donor", "Tissue", "Sex", "Ethnicity", "Age", "BMI",
                population_name}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if min_donors_per_group < 3 or min_total_donors < 10:
        raise ValueError("Use at least 3 donors per group and 10 total donors")
    if group_column not in {"Ethnicity", "Sex"}:
        raise ValueError("group_column must be 'Ethnicity' or 'Sex'")
    if focal_group == reference_group:
        raise ValueError("focal_group and reference_group must differ")
    adjustment_group = "Sex" if group_column == "Ethnicity" else "Ethnicity"
    adjustment_reference = "MALE" if adjustment_group == "Sex" else "White"

    df, antibody_terms, antibody_columns = _add_antibody_covariates(df)
    metadata_columns = ["Sex", "Ethnicity", "Age", "BMI", *antibody_columns]
    metadata = df[["Donor", *metadata_columns]].copy()
    metadata["Donor"] = metadata["Donor"].astype("string").str.strip()
    for column in ["Sex", "Ethnicity"]:
        metadata[column] = metadata[column].astype("string").str.strip()
    for column in ["Age", "BMI"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce")
    metadata = metadata.dropna(subset=["Donor"]).groupby(
        "Donor", sort=False)[metadata_columns].agg(
            lambda values: values.dropna().iloc[0]
            if values.notna().any() else np.nan).reset_index()

    values = df[["Donor", "Tissue", population_name]].copy()
    values["Donor"] = values["Donor"].astype("string").str.strip()
    values["Tissue"] = values["Tissue"].astype("string").str.strip()
    values[population_name] = pd.to_numeric(
        values[population_name], errors="coerce")
    values = values.dropna(subset=["Donor", "Tissue", population_name])
    values = values.groupby(["Donor", "Tissue"], as_index=False,
                            sort=False)[population_name].mean()
    model_data = values.merge(metadata, on="Donor", how="inner").dropna(
        subset=["Sex", "Ethnicity", "Age", "BMI", *antibody_columns])
    model_data = model_data[model_data[group_column].isin(
        [reference_group, focal_group])].copy()

    group_counts = model_data.groupby(["Tissue", group_column])[
        "Donor"].nunique().unstack(fill_value=0)
    eligible_tissues = [
        tissue for tissue in group_counts.index
        if group_counts.get(reference_group, pd.Series(dtype=float)).get(
            tissue, 0) >= min_donors_per_group
        and group_counts.get(focal_group, pd.Series(dtype=float)).get(
            tissue, 0) >= min_donors_per_group
    ]
    model_data = model_data[model_data["Tissue"].isin(eligible_tissues)].copy()
    counts_by_tissue = model_data.groupby(["Tissue", group_column])[
        "Donor"].nunique().unstack(fill_value=0)
    status = {
        "Subset": population_name,
        "Status": "skipped",
        "Reason": "",
        "N_Donors": int(model_data["Donor"].nunique()),
        "N_Observations": int(len(model_data)),
        "N_Tissues": int(model_data["Tissue"].nunique()),
        "Interaction_P": np.nan,
        "Reference_Tissue": None,
        "Eligible_Tissue_Counts": counts_by_tissue.to_dict(orient="index"),
    }
    if len(eligible_tissues) < 2:
        status["Reason"] = (
            "fewer than two tissues have both groups adequately represented")
        return pd.DataFrame(), status
    if status["N_Donors"] < min_total_donors:
        status["Reason"] = "too few complete donors"
        return pd.DataFrame(), status
    if adjustment_reference not in set(model_data[adjustment_group]):
        status["Reason"] = (
            f"{adjustment_reference} adjustment reference is absent")
        return pd.DataFrame(), status

    tissues = sorted(model_data["Tissue"].unique().tolist())
    if reference_tissue is None:
        reference_tissue = max(
            tissues, key=lambda tissue: model_data.loc[
                model_data["Tissue"] == tissue, "Donor"].nunique())
    else:
        reference_tissue = str(reference_tissue).strip()
        if reference_tissue not in tissues:
            status["Reason"] = f"reference tissue {reference_tissue!r} is absent"
            return pd.DataFrame(), status
    status["Reference_Tissue"] = reference_tissue

    formula = (
        f"Outcome ~ C({group_column}, Treatment(reference={reference_group!r})) * "
        f"C(Tissue, Treatment(reference={reference_tissue!r})) + "
        f"C({adjustment_group}, Treatment(reference={adjustment_reference!r})) "
        "+ Age + BMI"
    )
    if antibody_terms:
        formula += " + " + " + ".join(antibody_terms)
    model_data = model_data.rename(columns={population_name: "Outcome"})
    try:
        model = smf.mixedlm(
            formula, data=model_data, groups=model_data["Donor"]
        ).fit(reml=False, method="lbfgs", disp=False)
    except Exception as exc:
        status["Reason"] = f"model fit failed: {exc}"
        return pd.DataFrame(), status
    if not model.converged:
        status["Reason"] = "model did not converge"
        return pd.DataFrame(), status

    fixed_terms = list(model.fe_params.index)
    interaction_terms = [
        term for term in fixed_terms
        if ":" in term and f"C({group_column}," in term
        and "C(Tissue," in term
    ]
    if not interaction_terms:
        status["Reason"] = (
            f"no estimable {group_column.lower()}-by-tissue interaction terms")
        return pd.DataFrame(), status
    parameter_names = list(model.params.index)
    restriction = np.zeros((len(interaction_terms), len(parameter_names)))
    for row, term in enumerate(interaction_terms):
        restriction[row, parameter_names.index(term)] = 1.0
    try:
        interaction_p = float(model.wald_test(
            restriction, use_f=False, scalar=True).pvalue)
    except Exception as exc:
        status["Reason"] = f"interaction omnibus test failed: {exc}"
        return pd.DataFrame(), status

    fixed_covariance = model.cov_params().loc[
        fixed_terms, fixed_terms].to_numpy(dtype=float)
    design_info = model.model.data.design_info
    contrast_rows = []
    covariate_values = {
        adjustment_group: adjustment_reference,
        "Age": float(model_data["Age"].mean()),
        "BMI": float(model_data["BMI"].mean()),
    }
    covariate_values.update({
        column: model_data[column].mode().iloc[0]
        for column in antibody_columns
    })
    for tissue in tissues:
        prediction_data = pd.DataFrame([
            {**covariate_values, group_column: focal_group, "Tissue": tissue},
            {**covariate_values, group_column: reference_group, "Tissue": tissue},
        ])
        design = build_design_matrices(
            [design_info], prediction_data, return_type="dataframe")[0]
        contrast = (design.iloc[0][fixed_terms]
                    - design.iloc[1][fixed_terms]).to_numpy(dtype=float)
        estimate = float(contrast @ model.fe_params.to_numpy())
        variance = float(contrast @ fixed_covariance @ contrast)
        standard_error = np.sqrt(max(variance, 0.0))
        z_score = estimate / standard_error if standard_error > 0 else np.nan
        p_value = float(2 * stats.norm.sf(abs(z_score)))
        critical = stats.norm.ppf(1 - alpha / 2)
        tissue_counts = counts_by_tissue.loc[tissue]
        adjusted_group_means = {}
        tissue_standardization_data = model_data[
            model_data["Tissue"] == tissue].copy()
        for level in [reference_group, focal_group]:
            prediction_data = tissue_standardization_data.copy()
            prediction_data[group_column] = level
            group_design = build_design_matrices(
                [design_info], prediction_data, return_type="dataframe")[0]
            average_design = group_design[fixed_terms].mean(axis=0).to_numpy(
                dtype=float)
            group_mean = float(average_design @ model.fe_params.to_numpy())
            group_variance = float(
                average_design @ fixed_covariance @ average_design)
            group_se = np.sqrt(max(group_variance, 0.0))
            adjusted_group_means[level] = {
                "mean": group_mean,
                "lower": group_mean - critical * group_se,
                "upper": group_mean + critical * group_se,
            }
        contrast_rows.append({
            "Subset": population_name,
            "Group_Column": group_column,
            "Tissue": tissue,
            "Contrast": f"{focal_group} - {reference_group}",
            "Coefficient": estimate,
            "Std_Err": standard_error,
            "CI_Lower": estimate - critical * standard_error,
            "CI_Upper": estimate + critical * standard_error,
            "p_value": p_value,
            "N_Reference": int(tissue_counts.get(reference_group, 0)),
            "N_Focal": int(tissue_counts.get(focal_group, 0)),
            "Adjusted_Reference_Mean": adjusted_group_means[
                reference_group]["mean"],
            "Reference_Mean_CI_Lower": adjusted_group_means[
                reference_group]["lower"],
            "Reference_Mean_CI_Upper": adjusted_group_means[
                reference_group]["upper"],
            "Adjusted_Focal_Mean": adjusted_group_means[
                focal_group]["mean"],
            "Focal_Mean_CI_Lower": adjusted_group_means[
                focal_group]["lower"],
            "Focal_Mean_CI_Upper": adjusted_group_means[
                focal_group]["upper"],
        })
    contrasts = pd.DataFrame(contrast_rows)
    valid = contrasts["p_value"].notna()
    contrasts["p_value_FDR"] = np.nan
    if valid.any():
        contrasts.loc[valid, "p_value_FDR"] = multipletests(
            contrasts.loc[valid, "p_value"], alpha=alpha,
            method="fdr_bh")[1]
    contrasts["Is_Significant_FDR"] = contrasts["p_value_FDR"] < alpha
    contrasts["Interaction_P"] = interaction_p
    status.update({
        "Status": "fit", "Interaction_P": interaction_p,
        "N_Tissues": len(tissues),
        "Group_Column": group_column,
    })
    return contrasts, status


def analyze_ethnicity_tissue_interaction(
        df, population_name, focal_ethnicity,
        reference_ethnicity="White", min_donors_per_group=5,
        min_total_donors=20, alpha=0.05, reference_tissue=None):
    """
    Analyze focal-versus-reference ethnicity subset differences by tissue.
    """
    return _analyze_group_tissue_interaction(
        df, population_name, "Ethnicity", focal_ethnicity,
        reference_ethnicity, min_donors_per_group=min_donors_per_group,
        min_total_donors=min_total_donors, alpha=alpha,
        reference_tissue=reference_tissue)


def analyze_sex_tissue_interaction(
        df, population_name, focal_sex="FEMALE", reference_sex="MALE",
        min_donors_per_group=5, min_total_donors=20, alpha=0.05,
        reference_tissue=None):
    """
    Analyze female-versus-male subset differences by tissue.
    """
    return _analyze_group_tissue_interaction(
        df, population_name, "Sex", focal_sex, reference_sex,
        min_donors_per_group=min_donors_per_group,
        min_total_donors=min_total_donors, alpha=alpha,
        reference_tissue=reference_tissue)


def _plot_group_tissue_adjusted_means(contrasts, status=None):
    """
    Plot adjusted means for a demographic group across tissues.

    Parameters:
    -----------
    contrasts : pd.DataFrame
        DataFrame containing adjusted means and confidence intervals for each tissue.
    status : dict, optional
        Dictionary containing additional information about the analysis, such as interaction p-values.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    required = {
        "Subset", "Group_Column", "Tissue", "Contrast", "Adjusted_Reference_Mean",
        "Reference_Mean_CI_Lower", "Reference_Mean_CI_Upper",
        "Adjusted_Focal_Mean", "Focal_Mean_CI_Lower",
        "Focal_Mean_CI_Upper", "p_value_FDR",
    }
    missing = sorted(required.difference(contrasts.columns))
    if missing:
        raise ValueError(
            f"Interaction contrasts are missing columns: {missing}")
    if contrasts.empty:
        logging.warning(
            "No adjusted ethnicity-by-tissue means are available to plot")
        return

    data = contrasts.sort_values("Tissue").reset_index(drop=True)
    positions = np.arange(len(data), dtype=float)
    offset = 0.08
    focal_group, reference_group = data.iloc[0]["Contrast"].split(" - ", 1)
    group_column = data.iloc[0]["Group_Column"]
    palette = sex_palette if group_column == "Sex" else publication_palette
    group_specs = [
        ("Adjusted_Reference_Mean", "Reference_Mean_CI_Lower",
         "Reference_Mean_CI_Upper", reference_group,
         palette.get(reference_group, "#bd6319")),
        ("Adjusted_Focal_Mean", "Focal_Mean_CI_Lower",
         "Focal_Mean_CI_Upper", focal_group,
         palette.get(focal_group, "#1762a4")),
    ]
    fig, ax = plt.subplots(figsize=(max(11, 0.72 * len(data)), 6.5))
    for direction, (mean_col, lower_col, upper_col, group_name, color) in enumerate(group_specs):
        x = positions + (direction - 0.5) * 2 * offset
        means = data[mean_col].to_numpy(dtype=float)
        lower = means - data[lower_col].to_numpy(dtype=float)
        upper = data[upper_col].to_numpy(dtype=float) - means
        ax.errorbar(
            x, means, yerr=[np.maximum(lower, 0), np.maximum(upper, 0)],
            fmt="o-", color=color, ecolor=color, capsize=3,
            linewidth=1.5, markersize=5, label=group_name)

    significant = data["p_value_FDR"] < 0.05
    labels = [
        f"{row['Tissue']}*" if significant.iloc[i] else row["Tissue"]
        for i, (_, row) in enumerate(data.iterrows())
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    for tick_label, is_significant in zip(ax.get_xticklabels(), significant):
        if is_significant:
            tick_label.set_color("#b23a48")
            tick_label.set_fontweight("bold")
    ax.set_xlabel("Tissue")
    ax.set_ylabel(f"Adjusted {data.iloc[0]['Subset']} frequency")
    interaction_text = ""
    if status and pd.notna(status.get("Interaction_P")):
        interaction_text = (
            f"\n{group_column} × tissue omnibus p="
            f"{status['Interaction_P']:.3g}")
    ax.set_title(
        f"{data.iloc[0]['Subset']}: adjusted means by tissue and {group_column.lower()}"
        f"{interaction_text}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title=group_column, frameon=False, loc="upper right")
    fig.text(
        0.01, 0.015,
        f"* q < 0.05: within-tissue two-sided Wald z-tests, "
        f"Benjamini–Hochberg FDR across tissues. Omnibus interaction: "
        f"Wald chi-square test ({focal_group} × tissue).",
        ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    plt.show()
    return fig, ax


def plot_ethnicity_tissue_adjusted_means(contrasts, status=None):
    """
    Plot adjusted ethnicity means across tissues.

    Parameters:
    -----------
    contrasts : pd.DataFrame
        DataFrame containing adjusted means and confidence intervals for each tissue.
    status : dict, optional
        Dictionary containing additional information about the analysis, such as interaction p-values.
        Default is None.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    return _plot_group_tissue_adjusted_means(contrasts, status)


def plot_sex_tissue_adjusted_means(contrasts, status=None):
    """
    Plot adjusted sex means across tissues.

    Parameters:
    -----------
    contrasts : pd.DataFrame
        DataFrame containing adjusted means and confidence intervals for each tissue.
    status : dict, optional
        Dictionary containing additional information about the analysis, such as interaction p-values.
        Default is None.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """
    return _plot_group_tissue_adjusted_means(contrasts, status)


def plot_demographic_mixed_model_findings(
        results, demographics=("Ethnicity", "Sex"),
        significant_only=True, alpha=0.05):
    """
    Plot adjusted differences in subset frequency for sex and ethnicity.

    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame containing mixed-model results with coefficients, confidence intervals,
        and significance flags for each demographic variable.
    demographics : tuple, optional
        Tuple of demographic variables to plot. Default is ("Ethnicity", "Sex").
    significant_only : bool, optional
        If True, only plot results that are significant after FDR correction. Default is True.
    alpha : float, optional
        Significance level for FDR correction. Default is 0.05.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    axes : numpy.ndarray
        Array of axes objects containing the subplots for each demographic variable.
    """
    required = {
        "Subset", "Demographic_Variable", "Test_Type", "Coefficient",
        "CI_Lower", "CI_Upper", "N_Donors", "p_value_FDR",
        "Is_Significant_FDR",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Mixed-model results are missing columns: {missing}")
    demographics = list(demographics)
    unsupported = sorted(set(demographics).difference({"Sex", "Ethnicity"}))
    if unsupported:
        raise ValueError(
            f"Only Sex and Ethnicity can be plotted: {unsupported}")

    panels = []
    for demographic in demographics:
        coefficient_mask = (
            results["Test_Type"].eq("coefficient")
            & results["Demographic_Variable"].astype(str).str.startswith(
                f"{demographic}:")
        )
        data = results.loc[coefficient_mask].dropna(
            subset=["Coefficient", "CI_Lower", "CI_Upper"]).copy()
        if significant_only:
            data = data[data["Is_Significant_FDR"]]
        if data.empty:
            continue
        data = data.sort_values("Coefficient").reset_index(drop=True)

        omnibus = results[
            results["Test_Type"].eq("omnibus")
            & results["Demographic_Variable"].eq(f"{demographic}: omnibus")
            & results["Is_Significant_FDR"]
        ].sort_values("p_value_FDR")
        omnibus_text = ""
        if not omnibus.empty:
            labels = [
                f"{row['Subset']} (q={row['p_value_FDR']:.2g})"
                for _, row in omnibus.iterrows()
            ]
            omnibus_text = "\nOverall association: " + "; ".join(labels)
        panels.append((demographic, data, omnibus_text))

    if not panels:
        logging.warning("No demographic mixed-model findings to plot")
        return

    fig, axes = plt.subplots(
        1, len(panels), figsize=(7 * len(panels),
                                 max(4.5, 0.55 * max(len(data) for _, data, _ in panels))),
        sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax, (demographic, data, omnibus_text) in zip(axes, panels):
        data = data.sort_values("Coefficient").reset_index(drop=True)
        y_positions = np.arange(len(data))
        for position, row in data.iterrows():
            color = "#b23a48" if row["Is_Significant_FDR"] else "#526777"
            lower = max(0, row["Coefficient"] - row["CI_Lower"])
            upper = max(0, row["CI_Upper"] - row["Coefficient"])
            ax.errorbar(
                row["Coefficient"], position,
                xerr=[[lower], [upper]], fmt="o", color=color,
                ecolor=color, capsize=3,
            )
        comparisons = []
        reference = "White" if demographic == "Ethnicity" else "MALE"
        for _, row in data.iterrows():
            group = row["Demographic_Variable"].split(": ", 1)[-1]
            comparisons.append(
                f"{row['Subset']}: {group} vs {reference} "
                f"(n={row['N_Donors']}, q={row['p_value_FDR']:.2g})")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(comparisons, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{demographic} contrasts{omnibus_text}")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Adjusted difference in subset frequency")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#b23a48",
               markeredgecolor="#b23a48", label=f"FDR significant (q < {alpha:g})")
    ]
    if not significant_only:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#526777", markeredgecolor="#526777",
                   label="Not FDR significant"))
    fig.legend(handles=legend_handles, title="Result", loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=len(legend_handles),
               frameon=False)
    fig.suptitle("Adjusted sex and ethnicity associations with T-cell subsets",
                 y=1.08)
    fig.tight_layout()
    plt.show()
    return fig, axes


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

    name_mapping = {col: column_clean(col) for col in subset_columns}
    safe_subsets = list(name_mapping.values())

    reverse_mapping = {v: k for k, v in name_mapping.items()}
    df_working, antibody_terms, antibody_columns = _add_antibody_covariates(
        df.rename(columns=name_mapping))
    for safe_subset in safe_subsets:
        df_working[safe_subset] = pd.to_numeric(
            df_working[safe_subset], errors="coerce")

    all_results = []

    for safe_subset in safe_subsets:
        original_name = reverse_mapping[safe_subset]
        logging.info(f"Processing Target Subset: {original_name}")

        model_features = [
            "Age", "BMI", "Sex", "Ethnicity", "Tissue",
            *antibody_columns, safe_subset,
        ]
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
                    if antibody_terms:
                        formula += " + " + " + ".join(antibody_terms)

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
                        clean_coef_label = _format_model_term(
                            clean_coef_label)

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
    name_mapping = {col: column_clean(col) for col in subset_columns}
    reverse_mapping = {safe: original for original,
                       safe in name_mapping.items()}
    df_working, antibody_terms, antibody_columns = _add_antibody_covariates(
        df.rename(columns=name_mapping))
    safe_subsets = list(name_mapping.values())

    for safe_subset in safe_subsets:
        df_working[safe_subset] = pd.to_numeric(
            df_working[safe_subset], errors="coerce")

    all_results = []
    for safe_subset in safe_subsets:
        original_name = reverse_mapping[safe_subset]
        features = [
            "Age", "BMI", "Sex", "Ethnicity", "Tissue",
            *antibody_columns, safe_subset,
        ]
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
            if antibody_terms:
                formula += " + " + " + ".join(antibody_terms)
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
                reference_statuses = {
                    column: tissue_df[column].mode().iloc[0]
                    for column in antibody_columns
                }
                group_design = build_design_matrices(
                    [model.model.data.design_info],
                    pd.DataFrame({
                        "Demographic_Group": groups,
                        "Age": [0.0] * len(groups),
                        "BMI": [0.0] * len(groups),
                        **{column: [value] * len(groups)
                           for column, value in reference_statuses.items()},
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
            correction_result = multipletests(
                valid_pairs["p_value"], alpha=alpha, method="fdr_bh")
            corrected_pvalues = (
                correction_result[1] if correction_result is not None else None
            )
            if corrected_pvalues is not None:
                fdr_lookup = {}
                pair_keys = valid_pairs["_pair_key"].tolist()
                for index, pair_key in enumerate(pair_keys):
                    fdr_lookup[pair_key] = corrected_pvalues[index]
                results["p_value_FDR"] = results["_pair_key"].map(fdr_lookup)
        results["Is_Significant_FDR"] = results["p_value_FDR"] < alpha
        results = results.drop(columns="_pair_key")
    return results


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

    # Keep each demographic group's color consistent with the other plots.
    categories = df["Demographic_Variable"].dropna().unique().tolist()
    ethnicity_groups = [
        category.split(": ", 1)[1]
        for category in categories
        if category.startswith("Ethnicity: ")
    ]
    sex_groups = [
        category.split(": ", 1)[1]
        for category in categories
        if category.startswith("Sex: ")
    ]
    ethnicity_colors = _palette_for_categories(
        ethnicity_groups, publication_palette)
    sex_colors = _palette_for_categories(sex_groups, sex_palette)
    demographic_palette = {}
    for category in categories:
        if category.startswith("Ethnicity: "):
            demographic_palette[category] = ethnicity_colors[
                category.split(": ", 1)[1]]
        elif category.startswith("Sex: "):
            demographic_palette[category] = sex_colors[
                category.split(": ", 1)[1]]
        elif category.startswith(("CMV status: ", "EBV IgG status: ")):
            status = category.split(": ", 1)[1]
            demographic_palette[category] = antibody_status_palette.get(
                status, "#777777")
        elif category == "Age":
            demographic_palette[category] = "#555555"
        else:
            demographic_palette[category] = "#777777"

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
        palette=demographic_palette,
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
        adjusted_p_value=None, sex=None, ethnicity=None,
        group_comparison_stats_df=None):
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
    sex : str, optional
        If provided, restrict both panels and the adjusted model to this sex.
    ethnicity : str, optional
        If provided, restrict both panels and the adjusted model to this ethnicity.
    group_comparison_stats_df : pd.DataFrame, optional
        Pairwise group-comparison results used to look up the FDR q-value for
        the selected sex-by-ethnicity contrast in ``target_variable``.

    Returns:
    --------
    None
    """
    safe_pop = column_clean(population_name)
    df_working, antibody_terms, antibody_columns = _add_antibody_covariates(
        df.rename(columns={population_name: safe_pop}))

    df_tissue = df_working[df_working["Tissue"] == tissue].dropna(
        subset=[safe_pop, "Sex", "Ethnicity", "Age", "BMI",
                *antibody_columns]
    )
    if sex is not None:
        selected_sex = str(sex).strip().casefold()
        df_tissue = df_tissue[
            df_tissue["Sex"].astype("string").str.strip().str.casefold()
            == selected_sex
        ].copy()
    if ethnicity is not None:
        selected_ethnicity = str(ethnicity).strip().casefold()
        df_tissue = df_tissue[
            df_tissue["Ethnicity"].astype("string").str.strip().str.casefold()
            == selected_ethnicity
        ].copy()
    if df_tissue.empty:
        filters = []
        if sex is not None:
            filters.append(f"sex {sex}")
        if ethnicity is not None:
            filters.append(f"ethnicity {ethnicity}")
        filter_text = f" and {' and '.join(filters)}" if filters else ""
        print(f"No complete data available for tissue {tissue}{filter_text}")
        return

    if "Sex" in target_variable and sex is not None:
        raise ValueError(
            "A sex coefficient cannot be estimated after filtering to one sex")
    if "Ethnicity" in target_variable and ethnicity is not None:
        raise ValueError(
            "An ethnicity coefficient cannot be estimated after filtering to "
            "one ethnicity")
    if "Ethnicity" in target_variable and df_tissue["Ethnicity"].nunique() < 2:
        raise ValueError(
            "An ethnicity coefficient cannot be estimated with only one "
            "ethnicity in the filtered data")

    ethnicity_reference = "White"
    ethnicity_reference_match = re.search(
        r"C\(Ethnicity,\s*Treatment\(reference=['\"]([^'\"]+)['\"]\)",
        target_variable)
    if ethnicity_reference_match:
        ethnicity_reference = ethnicity_reference_match.group(1)
    sex_reference = "MALE"
    sex_reference_match = re.search(
        r"C\(Sex,\s*Treatment\(reference=['\"]([^'\"]+)['\"]\)",
        target_variable)
    if sex_reference_match:
        sex_reference = sex_reference_match.group(1)

    if adjusted_p_value is None and group_comparison_stats_df is not None:
        required_comparison_columns = {
            "Subset", "Tissue", "Reference_Group", "Comparison_Group",
            "p_value_FDR",
        }
        missing_comparison_columns = required_comparison_columns.difference(
            group_comparison_stats_df.columns)
        if missing_comparison_columns:
            raise ValueError(
                "Group comparison results are missing columns: "
                f"{sorted(missing_comparison_columns)}")
        if ("[T." in target_variable
                and ((sex is not None and "Ethnicity" in target_variable)
                     or (ethnicity is not None and "Sex" in target_variable))):
            focal_group = target_variable.rsplit("[T.", 1)[1].rstrip("]")
            if "Ethnicity" in target_variable:
                sex_label = str(df_tissue["Sex"].dropna().iloc[0])
                reference_group = f"{sex_label} / {ethnicity_reference}"
                comparison_group = f"{sex_label} / {focal_group}"
            else:
                ethnicity_label = str(df_tissue["Ethnicity"].dropna().iloc[0])
                reference_group = f"{sex_reference} / {ethnicity_label}"
                comparison_group = f"{focal_group} / {ethnicity_label}"
            same_pair = (
                (group_comparison_stats_df["Reference_Group"].eq(
                    reference_group)
                 & group_comparison_stats_df["Comparison_Group"].eq(
                    comparison_group))
                | (group_comparison_stats_df["Reference_Group"].eq(
                    comparison_group)
                   & group_comparison_stats_df["Comparison_Group"].eq(
                    reference_group))
            )
            matching_q = group_comparison_stats_df.loc[
                group_comparison_stats_df["Subset"].eq(population_name)
                & group_comparison_stats_df["Tissue"].eq(tissue)
                & same_pair,
                "p_value_FDR",
            ].dropna()
            if not matching_q.empty:
                adjusted_p_value = float(matching_q.iloc[0])
            else:
                logging.warning(
                    "No group-comparison FDR q-value found for %s vs %s in %s/%s",
                    comparison_group, reference_group, population_name, tissue)

    model_terms = []
    sex_levels = df_tissue["Sex"].dropna().unique().tolist()
    if len(sex_levels) > 1:
        if sex_reference not in sex_levels:
            raise ValueError(
                f"Sex reference {sex_reference!r} is absent from the filtered data")
        model_terms.append(
            f"C(Sex, Treatment(reference={sex_reference!r}))")
    ethnicity_levels = df_tissue["Ethnicity"].dropna().unique().tolist()
    if len(ethnicity_levels) > 1:
        if ethnicity_reference not in ethnicity_levels:
            raise ValueError(
                f"Ethnicity reference {ethnicity_reference!r} is absent "
                "from the filtered data")
        model_terms.append(
            f"C(Ethnicity, Treatment(reference={ethnicity_reference!r}))")
    model_terms.extend(["Age", "BMI", *antibody_terms])
    formula = f"{safe_pop} ~ " + " + ".join(model_terms)
    y, X = dmatrices(formula, data=df_tissue, return_type='dataframe')

    if target_variable not in X.columns:
        raise ValueError(f"Variable '{target_variable}' not found in model columns. "
                         f"Available columns are: {list(X.columns)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    if "Sex" in target_variable:
        sex_plot_data = df_working.copy()
        sex_plot_data[safe_pop] = pd.to_numeric(
            sex_plot_data[safe_pop], errors="coerce")
        sex_plot_data["Sex"] = sex_plot_data["Sex"].astype(
            "string").str.strip()
        sex_plot_data = sex_plot_data.dropna(
            subset=["Sex", "Tissue", safe_pop, "Donor"])
        sex_plot_data = sex_plot_data[
            sex_plot_data["Tissue"].astype("string").str.strip().eq(
                str(tissue).strip())]
        if ethnicity is not None:
            sex_plot_data = sex_plot_data[
                sex_plot_data["Ethnicity"].astype("string").str.strip()
                .str.casefold().eq(str(ethnicity).strip().casefold())]
        unique_sexes = list(sex_plot_data["Sex"].drop_duplicates())
        sex_counts = sex_plot_data.groupby("Sex")["Donor"].nunique()
        sex_palette_map = _palette_for_categories(unique_sexes, sex_palette)
        sns.boxplot(
            x="Sex", y=safe_pop, data=sex_plot_data, hue="Sex",
            order=unique_sexes, palette=sex_palette_map, legend=False,
            dodge=False, boxprops=dict(alpha=0.4), showfliers=True,
            flierprops={
                "marker": "o", "markerfacecolor": "white",
                "markeredgecolor": "black", "markersize": 6,
            }, ax=axes[0], zorder=1,
        )
        sns.lineplot(
            x="Sex", y=safe_pop, data=sex_plot_data, units="Donor",
            estimator=None, color="black", alpha=0.3, linewidth=1,
            ax=axes[0], zorder=2,
        )
        grouped_values = sex_plot_data.groupby("Sex")[safe_pop]
        q1 = grouped_values.transform(lambda values: values.quantile(0.25))
        q3 = grouped_values.transform(lambda values: values.quantile(0.75))
        iqr = q3 - q1
        is_outlier = ((sex_plot_data[safe_pop] < q1 - 1.5 * iqr)
                      | (sex_plot_data[safe_pop] > q3 + 1.5 * iqr))
        sns.stripplot(
            x="Sex", y=safe_pop, data=sex_plot_data.loc[~is_outlier],
            hue="Sex", order=unique_sexes, palette=sex_palette_map,
            legend=False, jitter=True, size=6, edgecolor="black",
            linewidth=0.5, ax=axes[0], zorder=3,
        )
        valid_sexes = [
            sex for sex in unique_sexes
            if sex_counts.get(sex, 0) >= 2
        ]
        if len(valid_sexes) >= 2:
            annotator = Annotator(
                axes[0], list(combinations(valid_sexes, 2)),
                data=sex_plot_data, x="Sex", y=safe_pop, order=unique_sexes,
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
        sex_positions = {group: position for position, group in
                         enumerate(unique_sexes)}
        for group, outliers in sex_plot_data.loc[is_outlier].groupby(
                "Sex", sort=False):
            for index, (_, outlier) in enumerate(
                    outliers.sort_values(safe_pop).iterrows()):
                to_right = index % 2 == 0
                axes[0].annotate(
                    str(outlier["Donor"]),
                    xy=(sex_positions[group], outlier[safe_pop]),
                    xytext=(10 if to_right else -10, 5 * (index // 2)),
                    textcoords="offset points",
                    ha="left" if to_right else "right", va="center",
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": "gray",
                                "lw": 0.7},
                    annotation_clip=False, zorder=5,
                )
        axes[0].set_xticks(range(len(unique_sexes)))
        axes[0].set_xticklabels(
            [f"{group}\n(n={sex_counts.get(group, 0)})"
             for group in unique_sexes], rotation=45, ha="right")
        axes[0].set_title(
            f"{population_name} in {tissue}"
            + (f" ({ethnicity})" if ethnicity is not None else "")
            + "\nPairwise significance: Mann–Whitney U "
            "(Bonferroni-adjusted)", fontsize=13, pad=55)
    elif "Ethnicity" in target_variable:
        ethnicity_data, valid_ethnicities, ethnicity_counts = (
            _prepare_ethnicity_boxplot_data(
                df_working, safe_pop, tissue, sex=sex, min_donors=3)
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
                boxprops=dict(alpha=0.4),
                flierprops={
                    "marker": "o", "markerfacecolor": "white",
                    "markeredgecolor": "black", "markersize": 6,
                },
            )
            sns.lineplot(
                x="Ethnicity", y=safe_pop, data=ethnicity_data,
                units="Donor", estimator=None, color="black", alpha=0.3,
                linewidth=1, ax=axes[0], zorder=2,
            )
            is_outlier = _ethnicity_iqr_outliers(ethnicity_data, safe_pop)
            point_data = ethnicity_data.loc[~is_outlier]
            sns.stripplot(
                x="Ethnicity", y=safe_pop,
                data=point_data,
                ax=axes[0], hue="Ethnicity", order=valid_ethnicities,
                palette=ethnicity_palette, legend=False, jitter=True,
                size=6, edgecolor="black", linewidth=0.5,
            )
            _annotate_ethnicity_comparisons(
                axes[0], ethnicity_data, safe_pop, valid_ethnicities)

            positions = {group: index for index, group in
                         enumerate(valid_ethnicities)}
            for group, outliers in ethnicity_data.loc[is_outlier].groupby(
                    "Ethnicity", sort=False):
                for index, (_, outlier) in enumerate(
                        outliers.sort_values(safe_pop).iterrows()):
                    to_right = index % 2 == 0
                    axes[0].annotate(
                        str(outlier["Donor"]),
                        xy=(positions[group], outlier[safe_pop]),
                        xytext=(10 if to_right else -10, 5 * (index // 2)),
                        textcoords="offset points",
                        ha="left" if to_right else "right", va="center",
                        fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "gray",
                                    "lw": 0.7},
                        annotation_clip=False, zorder=5,
                    )

            axes[0].set_xticks(range(len(valid_ethnicities)))
            axes[0].set_xticklabels(
                [f"{ethnicity}\n(n={ethnicity_counts[ethnicity]})"
                 for ethnicity in valid_ethnicities],
                rotation=45, ha="right",
            )
        axes[0].set_title(
            f"{population_name} in {tissue}"
            + (f" - {sex}" if sex is not None else "")
            + "\nPairwise significance: Mann–Whitney U "
            "(Bonferroni-adjusted)", fontsize=13, pad=55,
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
        "Sex": df_tissue.loc[y.index, "Sex"],
    })
    color_by = "Sex" if "Sex" in target_variable else "Ethnicity"
    color_order = list(partial_data[color_by].drop_duplicates())
    color_base_palette = (
        sex_palette if color_by == "Sex" else publication_palette)
    color_palette = _palette_for_categories(
        color_order, color_base_palette)
    line_x = [partial_data["Predictor_Residual"].min(),
              partial_data["Predictor_Residual"].max()]
    axes[1].plot(
        line_x, [beta_coefficient * value for value in line_x],
        color="black", linewidth=1.5, zorder=1,
    )
    sns.scatterplot(
        data=partial_data, x="Predictor_Residual", y="Outcome_Residual",
        hue=color_by, hue_order=color_order,
        palette=color_palette, s=65, edgecolor="black", alpha=0.85,
        ax=axes[1], zorder=2,
    )

    clean_label = (
        target_variable.rsplit("[T.", 1)[1].rstrip("]")
        if "[T." in target_variable else target_variable
    )
    contrast = (f" vs {ethnicity_reference}"
                if "Ethnicity" in target_variable else
                f" vs {sex_reference}" if "Sex" in target_variable else "")
    subset_label = ", ".join(
        f"{value} only" for value in [sex, ethnicity] if value is not None)
    subset_label = f" ({subset_label})" if subset_label else ""
    q_label = (
        f", BH q = {adjusted_p_value:.4f}"
        if adjusted_p_value is not None else ""
    )
    axes[1].set_title(
        f"Adjusted OLS: {clean_label}{contrast} in {tissue}\n"
        f"β = {beta_coefficient:.2f}{q_label}"
        + subset_label, pad=20)
    axes[1].set_xlabel(f"Residuals of {clean_label}")
    axes[1].set_ylabel(f"Residuals of {population_name}")
    axes[1].legend(title=color_by, bbox_to_anchor=(
        1.02, 1), loc="upper left")

    sns.despine()
    plt.tight_layout(rect=(0, 0, 0.9, 0.84))

    clean_filename = (target_variable
                      .replace("C(", "")
                      .replace(")[T.", "_")
                      .replace("]", "")
                      .replace(" ", "_")
                      .replace("/", "_"))
    for filter_value in [sex, ethnicity]:
        if filter_value is not None:
            safe_value = re.sub(
                r"[^A-Za-z0-9_-]+", "_", str(filter_value)).strip("_")
            clean_filename += f"_{safe_value}"
    plt.savefig(f"{safe_pop}_{tissue}_{clean_filename}_comparison_panel.png",
                dpi=300, bbox_inches="tight")
    plt.show()
