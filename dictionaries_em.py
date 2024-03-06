"""
Dictionaries for looking at observed correlations between reanalysis products and the observed energy data.
"""

clearheads_dir = "/home/users/benhutch/CLEARHEADS_EU_Power_Data"

era5_azores = {
    'lat1': 40.0, # index from 90 to -90 (N to S) for lat
    'lat2': 36.0,
    'lon1': 332.0, 
    'lon2': 340.0
}

era5_iceland = {
    'lat1': 70.0, # index from 90 to -90 (N to S) for lat
    'lat2': 63.0,
    'lon1': 335.0,
    'lon2': 344.0
}

country_codes = ['ES', 'NO', 'UK', 'IE', 'FI', 'SE', 'BE', 'NL', 'DE', 'DK', 'PO', 'FR', 'IT', 'PT', 'EE', 'LI', 'LV', 'HR', 'RO', 'SI', 'GR', 'TR', 'MT', 'AL', 'BG']
countries_list = ["Spain", "Norway", "United Kingdom", "Ireland", "Finland", "Sweden", "Belgium", "Netherlands", "Germany", "Denmark", "Poland", "France", "Italy", "Portugal", "Estonia", "Lithuania", "Latvia", "Croatia", "Romania", "Slovenia", "Greece", "Turkey", "Malta", "Albania", "Bulgaria"]

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