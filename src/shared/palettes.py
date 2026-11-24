"""Shared category colors used across flow and single-cell analyses."""

publication_palette = {
    "Hispanic/Latino": "#1762a4",
    "White": "#bd6319",
    "Black or African American": "#1cb251",
    "Asian": "#97182f",
}

sex_palette = {
    "MALE": "#0c6e97",
    "FEMALE": "#84109e",
}

antibody_status_palette = {
    "Negative": "#009E73",
    "Positive": "#CC79A7",
    "Not tested": "#8C8C8C",
}

ANTIBODY_STATUS_COLUMNS = {
    "Ab CMV:": "CMV_Status",
    "Ab EBV (IgG):": "EBV_IgG_Status",
}
