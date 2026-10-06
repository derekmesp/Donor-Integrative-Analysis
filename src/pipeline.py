
from patsy import dmatrices
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import logging
import re
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
from statannotations.Annotator import Annotator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

publication_palette = {
    "Hispanic/Latino": "#9ec1df",          # Muted Blue
    "White": "#fbc599",                    # Muted Orange
    "Black or African American": "#a8d8b9",  # Muted Green
    "Asian": "#e6afb9"                     # Muted Rose Pink
}
sex_palette = {
    "MALE": "#9ec1df",
    "FEMALE": "#fbc599"  # Muted Orange
}


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

    all_results = []

    for safe_subset in safe_subsets:
        original_name = reverse_mapping[safe_subset]
        logging.info(f"Processing Target Subset: {original_name}")

        model_features = demographics + [safe_subset]
        df_subset_clean = df_working[model_features].dropna()

        tissues = df_subset_clean["Tissue"].unique()

        for t in tissues:
            tissue_df = df_subset_clean[df_subset_clean["Tissue"] == t]

            if len(tissue_df) >= min_samples:
                try:
                    formula = f"{safe_subset} ~ C(Sex) + C(Ethnicity, Treatment(reference='White')) + Age + BMI"

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
                                            .replace("C(Sex)[T.", "Sex: ")
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

    return pd.DataFrame(all_results)


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

    unique_ethnicities = list(df["Ethnicity"].unique())

    is_outlier = (
        df.groupby("Ethnicity")[population_name]
        .transform(lambda x: (x - x.mean()).abs() > 2 * x.std())
        .fillna(False)
        .astype(bool)
    )

    outlier_donors = df[is_outlier]["Donor"].unique()
    outlier_str = ", ".join(map(str, outlier_donors))
    logging.info(
        f"Outlier donors for {population_name} in tissue {tissue}: {outlier_str}"
    )

    df_filtered = df[~is_outlier].reset_index(drop=True)

    default_colors = sns.color_palette(n_colors=len(unique_ethnicities))
    box_palette = dict(zip(unique_ethnicities, default_colors))

    fig, ax = plt.subplots(figsize=(8, 8))

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
        data=df_filtered,
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

    ethnicity_counts = df_filtered["Ethnicity"].value_counts()
    valid_ethnicities = [
        eth for eth in unique_ethnicities if ethnicity_counts[eth] >= 2
    ]

    if len(valid_ethnicities) >= 2:
        pairs = list(combinations(valid_ethnicities, 2))
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
            loc="inside",
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

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=10)
    plt.title(f"{population_name} in {tissue}", fontsize=14)

    sns.despine()
    plt.tight_layout()

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

    is_outlier = (
        df.groupby("Sex")[population_name]
        .transform(lambda x: (x - x.mean()).abs() > 2 * x.std())
        .fillna(False)
        .astype(bool)
    )

    outlier_donors = df[is_outlier]["Donor"].unique()
    outlier_str = ", ".join(map(str, outlier_donors))
    logging.info(
        f"Outlier donors for {population_name} in tissue {tissue}: {outlier_str}"
    )

    # df_filtered = df[~is_outlier].reset_index(drop=True)
    df_filtered = df

    default_colors = sns.color_palette(n_colors=len(unique_sexes))
    box_palette = dict(zip(unique_sexes, default_colors))

    fig, ax = plt.subplots(figsize=(8, 8))

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
        data=df_filtered,
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
            loc="inside",
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

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=10)
    plt.title(f"{population_name} in {tissue}", fontsize=14)

    sns.despine()
    plt.tight_layout()

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

    df["Label"] = df["Subset"] + \
        " (" + df["Tissue"] + ")"

    df = df.sort_values(
        by=["Demographic_Variable", "Coefficient"], ascending=[True, True])

    plt.figure(figsize=(11, 8))
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
                alpha=0.7, label="Baseline (White/Female)")
    plt.title("Tissue-Stratified OLS Demographic Coefficients on T-Cell Subsets\n(All points represent statistically significant changes, p < 0.05)",
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


def visualize_ols_vs_mann_whitney(df, population_name, tissue, target_variable="BMI"):
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
        provide the Patsy formatted string (e.g., "C(Sex)[T.MALE]" or 
        "C(Ethnicity, Treatment(reference='White'))[T.Asian]").
    """
    safe_pop = column_clean(population_name)
    df_working = df.rename(columns={population_name: safe_pop}).copy()

    df_tissue = df_working[df_working["Tissue"] == tissue].dropna(
        subset=[safe_pop, "Sex", "Ethnicity", "Age", "BMI"]
    )

    if df_tissue.empty:
        print(f"No complete data available for tissue {tissue}")
        return

    formula = f"{safe_pop} ~ C(Sex) + C(Ethnicity, Treatment(reference='White')) + Age + BMI"
    y, X = dmatrices(formula, data=df_tissue, return_type='dataframe')

    if target_variable not in X.columns:
        raise ValueError(f"Variable '{target_variable}' not found in model columns. "
                         f"Available columns are: {list(X.columns)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    if "Sex" in target_variable:
        sns.boxplot(x="Sex", y=safe_pop, data=df_tissue,
                    ax=axes[0], palette=sex_palette, hue="Sex", legend=False)
        sns.stripplot(x="Sex", y=safe_pop, data=df_tissue,
                      ax=axes[0], color="black", alpha=0.6, jitter=True)
        axes[0].set_title(
            f"Univariable View: {population_name} by Sex\n(Confounded / Mann-Whitney Null) in {tissue}")
    elif "Ethnicity" in target_variable:
        sns.boxplot(x="Ethnicity", y=safe_pop, data=df_tissue,
                    ax=axes[0], hue="Ethnicity", palette=publication_palette, legend=False)
        sns.stripplot(x="Ethnicity", y=safe_pop, data=df_tissue,
                      ax=axes[0], color="black", alpha=0.6, jitter=True)
        axes[0].set_xticklabels(
            axes[0].get_xticklabels(), rotation=45, ha='right')
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
    other_cols = [col for col in X.columns if col != target_variable]

    sm.graphics.plot_partregress(
        endog=y,
        exog_i=X[target_variable],
        exog_others=X[other_cols],
        obs_labels=False,
        ax=axes[1]
    )

    clean_label = target_variable.split(
        '.')[-1].replace(']', '') if '[' in target_variable else target_variable
    axes[1].set_title(
        f"Adjusted OLS View: Isolated Effect of {clean_label} in {tissue}\n(OLS β = {beta_coefficient:.3f}, p = {full_model.pvalues[target_variable]:.4f})")
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


def visualize_ols_vs_mann_whitney_colored(df, population_name, tissue, target_variable="C(Sex)[T.MALE]", color_by="Ethnicity"):
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
        The exact string parameter from your OLS summary (e.g., "C(Sex)[T.MALE]" or "BMI").
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

    formula = f"{safe_pop} ~ C(Sex) + C(Ethnicity, Treatment(reference='White')) + Age + BMI"
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
