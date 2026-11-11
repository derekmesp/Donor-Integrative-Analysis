import pandas as pd


def merge_dataframes(data_path, metadata_path):
    """
    Merges two dataframes based on the 'Donor' column.

    Parameters:
    data_path (str): Path to the first dataframe (data).
    metadata_path (str): Path to the second dataframe (metadata).

    Returns:
    pd.DataFrame: Merged dataframe with selected columns from the metadata.
    """

    df1 = pd.read_excel(data_path)
    df2 = pd.read_excel(metadata_path)

    df1['Donor'] = df1['Donor'].astype('string').str.strip()
    df2['Donor'] = df2['Donor'].astype('string').str.strip()

    merged_df = pd.merge(
        df1, df2[['Donor', 'Ethnicity:', 'Age (yr):', 'BMI (kg/m^2):', 'Sex:']], on='Donor', how='left')

    merged_df['Age (yr):'] = pd.to_numeric(
        merged_df['Age (yr):'], errors='coerce')
    merged_df['BMI (kg/m^2):'] = pd.to_numeric(merged_df['BMI (kg/m^2):'],
                                               errors='coerce')

    return merged_df


if __name__ == "__main__":
    merged_df = merge_dataframes(
        "Donor Master Sheet.xlsx",
        "Donors Information sheet(D0001-D0708).xlsx",
    )
    merged_df.to_excel("merged_data.xlsx", index=False)
