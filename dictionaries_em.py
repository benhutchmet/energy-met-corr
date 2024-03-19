"""
Dictionaries for looking at observed correlations between reanalysis products and the observed energy data.
"""

clearheads_dir = "/home/users/benhutch/CLEARHEADS_EU_Power_Data"

era5_azores = {
    "lat1": 40.0,  # index from 90 to -90 (N to S) for lat
    "lat2": 36.0,
    "lon1": 332.0,
    "lon2": 340.0,
}

era5_iceland = {
    "lat1": 70.0,  # index from 90 to -90 (N to S) for lat
    "lat2": 63.0,
    "lon1": 335.0,
    "lon2": 344.0,
}

country_codes = [
    "ES",
    "NO",
    "UK",
    "IE",
    "FI",
    "SE",
    "BE",
    "NL",
    "DE",
    "DK",
    "PO",
    "FR",
    "IT",
    "PT",
    "EE",
    "LI",
    "LV",
    "HR",
    "RO",
    "SI",
    "GR",
    "TR",
    "MT",
    "AL",
    "BG",
]
countries_list = [
    "Spain",
    "Norway",
    "United Kingdom",
    "Ireland",
    "Finland",
    "Sweden",
    "Belgium",
    "Netherlands",
    "Germany",
    "Denmark",
    "Poland",
    "France",
    "Italy",
    "Portugal",
    "Estonia",
    "Lithuania",
    "Latvia",
    "Croatia",
    "Romania",
    "Slovenia",
    "Greece",
    "Turkey",
    "Malta",
    "Albania",
    "Bulgaria",
]

# Create a dictionary that maps country codes to country names
country_dict = dict(zip(country_codes, countries_list))

# Define the dimensions for the gridbox for the azores
azores_grid_corrected = {"lon1": -28, "lon2": -20, "lat1": 36, "lat2": 40}

# Define the dimensions for the gridbox for the azores
iceland_grid_corrected = {"lon1": -25, "lon2": -16, "lat1": 63, "lat2": 70}

# Define a dictionary to map the season strings to their corresponding months
season_month_map = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "MAY": [3, 4, 5],
    "JJA": [6, 7, 8],
    "ULG": [6, 7, 8],
    "JJAS": [6, 7, 8, 9],
    "SON": [9, 10, 11],
    "SOND": [9, 10, 11, 12],
    "NDJF": [11, 12, 1, 2],
    "DJFM": [12, 1, 2, 3],
    "djfm": [12, 1, 2, 3],
    "ONDJFM": [10, 11, 12, 1, 2, 3],
    "ondjfm": [10, 11, 12, 1, 2, 3],
    "JFMA": [1, 2, 3, 4],
    "AYULGS": [4, 5, 6, 7, 8, 9],
    "AMJJAS": [4, 5, 6, 7, 8, 9],
}

# Define the dimensions for the gridbox for the azores
azores_grid = {"lon1": 152, "lon2": 160, "lat1": 36, "lat2": 40}


# Define the dimensions for the gridbox for iceland
iceland_grid = {"lon1": 155, "lon2": 164, "lat1": 63, "lat2": 70}

# regrid file for pr
regrid_file_pr = "/home/users/benhutch/ERA5/global_regrid_sel_region_var228.nc"

# regrid file for for other variables
regrid_file = "/home/users/benhutch/ERA5/global_regrid_sel_region.nc"

era5_msl_path = "~/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

# define the uk grid from Clarke et al. 2017
uk_grid = {"lon1": -10, "lon2": 3, "lat1": 50, "lat2": 60}

north_sea_kay = {
    "lon1": 1,  # degrees east
    "lon2": 7,
    "lat1": 53,  # degrees north
    "lat2": 59,
}

# Define the dimensions for the gridbox for the azores
azores_grid_corrected = {"lon1": -28, "lon2": -20, "lat1": 36, "lat2": 40}

# Define the dimensions for the gridbox for the azores
iceland_grid_corrected = {"lon1": -25, "lon2": -16, "lat1": 63, "lat2": 70}

# Define this but corrected
uk_n_box_corrected = {"lon1": -27, "lon2": 21, "lat1": 57, "lat2": 70}

# Define this but corrected
uk_s_box_corrected = {"lon1": -27, "lon2": 21, "lat1": 38, "lat2": 51}

iso_mapping = {
    "ES": "ESP",
    "NO": "NOR",
    "UK": "GBR",
    "IE": "IRL",
    "FI": "FIN",
    "SE": "SWE",
    "BE": "BEL",
    "NL": "NLD",
    "DE": "DEU",
    "DK": "DNK",
    "PO": "POL",
    "FR": "FRA",
    "IT": "ITA",
    "PT": "PRT",
    "EE": "EST",
    "LI": "LIE",
    "LV": "LVA",
    "HR": "HRV",
    "RO": "ROU",
    "GR": "GRC",
    "TR": "TUR",
    "MT": "MLT",
    "AL": "ALB",
    "BG": "BGR",
    "SI": "SVN",
    # Add the rest of your mapping here
}

# EEZ countries to aggregate
eez_agg_countries = [
    "GBR",
    "NOR",
    "SWE",
    "FIN",
    "DNK",
    "DEU",
    "NLD",
    "BEL",
    "EST",
    "LVA",
    "POL",
]

# Full names of these countries
eez_agg_countries_full_names = [
    "United Kingdom",
    "Norway",
    "Sweden",
    "Finland",
    "Denmark",
    "Germany",
    "Netherlands",
    "Belgium",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Poland",
]

countries = [
    "ES",
    "NO",
    "UK",
    "IE",
    "FI",
    "SE",
    "BE",
    "NL",
    "DE",
    "DK",
    "PO",
    "FR",
    "IT",
    "PT",
    "EE",
    "LI",
    "LV",
    "HR",
    "RO",
    "GR",
    "TR",
    "MT",
    "AL",
    "BG",
]

iso_sov1 = [iso_mapping[country] for country in countries]

# Set up a northern europe grid box
northern_europe_grid = {"lon1": -10, "lon2": 25, "lat1": 55, "lat2": 70}
