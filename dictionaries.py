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