"""
Functions for looking at observed correlations between reanalysis products and the observed energy data.

Author: Ben Hutchins
Date: February 2023
"""

# Import local modules
import sys
import os
import glob

# Import third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iris
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from scipy.stats import pearsonr, linregress, t

# Import local modules
import dictionaries_em as dicts

# Import external modules
sys.path.append("/home/users/benhutch/skill-maps/python/")
import paper1_plots_functions as p1p_funcs
import nao_alt_lag_functions as nal_funcs

# Define a function to form the dataframe for the offshore wind farm data
def extract_offshore_eez_to_df(
    filepath: str,
    countries_list: list = ["France", "Italy", "Portugal", "Estonia", "Latvia", "Lithuania", "Croatia", "Romania", "Slovenia", "Greece", "Montenegro", "Albania", "Bulgaria", "Spain", "Norway", "United Kingdom", "Ireland", "Finland", "Sweden", "Belgium", "Netherlands", "Germany", "Denmark", "Poland"],
    rolling_window: int = 8,
    centre: bool = True,
    annual_offset: int = 3,
    months: list = [10, 11, 12, 1, 2, 3],
    start_date: str = '1950-01-01',
    time_unit: str = 'h',
) -> pd.DataFrame:
    """
    Extracts the offshore wind farm data from the given file and returns it as a dataframe.
    
    Args:
        filepath: str
            The path to the file containing the offshore wind farm data.
        rolling_window: int
            The number of hours to use for the rolling window average.
        centre: bool
            Whether to centre the rolling window average.
        annual_offset: int
            The number of months to offset the annual average by.
        months: list
            The months to include in the annual average.
        start_date: str
            The start date for the data.
        time_unit: str
            The time unit for the data.
    Returns:
        df: pd.DataFrame
            The dataframe containing the offshore wind farm data.
    """
    # Find files
    files = glob.glob(filepath)

    # Assert that the file exists
    assert len(files) > 0, f"No files found at {filepath}"

    # Assert that there is only one file
    assert len(files) == 1, f"Multiple files found at {filepath}"

    # Load the data
    ds = xr.open_dataset(files[0])

    # Extract the values
    nuts_keys = ds.NUTS_keys.values

    # Turn the data into a dataframe
    df = ds.to_dataframe()

    # Create columns for each of the indexed NUTS regions
    # Pivot the DataFrame
    df_pivot = df.reset_index().pivot(index='time_in_hours_from_first_jan_1950',
                                       columns='NUTS',
                                         values='timeseries_data')
    
    # Assuming country_dict is a dictionary that maps NUTS keys to country names
    df_pivot.columns = [f"{dicts.country_dict[nuts_keys[i]]}_{col}" for i, col in enumerate(df_pivot.columns)]
    
    # Convert 'time_in_hours_from_first_jan_1950' column to datetime
    df_pivot.index = pd.to_datetime(df_pivot.index, unit=time_unit, origin=start_date)

    # Collapse the dataframes into monthly averages
    df_pivot = df_pivot.resample('M').mean()

    # Select only the months of interest
    df_pivot = df_pivot[df_pivot.index.month.isin(months)]

    # Shift the data by the annual offset
    df_pivot.index = df_pivot.index - pd.DateOffset(months=annual_offset)

    # TODO: Fix hard coded here
    # Throw away the first 3 months of data and last 3 months of data
    df_pivot = df_pivot.iloc[3:-3]

    # Calculate the annual average
    df_pivot = df_pivot.resample('A').mean()

    # Take the rolling average
    df_pivot = df_pivot.rolling(window=rolling_window, center=centre).mean()

    # Throw away the NaN values
    df_pivot = df_pivot.dropna()

    # Return the dataframe
    return df_pivot


# Write a function to calculate the stats
def calc_nao_spatial_corr(season: str,
                          forecast_range: str,
                          start_year: int,
                          end_year: int,
                          corr_var: str = "tos",
                          corr_var_obs_file: str = dicts.regrid_file_pr,
                          nao_obs_var: str = "msl",
                          nao_obs_file: str = dicts.regrid_file,
                          nao_n_grid: dict = dicts.iceland_grid_corrected,
                          nao_s_grid: dict = dicts.azores_grid_corrected,
                          sig_threshold: float = 0.05,
):
    """
    Calculates the spatial correlations between the NAO index (winter default) 
    and the variable to correlate for the observations.

    Args:
    -----

    season: str
        The season to calculate the correlation for.

    forecast_range: str
        The forecast range to calculate the correlation for.

    start_year: int
        The start year to calculate the correlation for.

    end_year: int
        The end year to calculate the correlation for.

    corr_var: str
        The variable to correlate with the NAO index.

    corr_var_obs_file: str
        The file containing the observations of the variable to correlate.

    nao_obs_var: str
        The variable to use for the NAO index.

    nao_obs_file: str
        The file containing the observations of the NAO index.

    nao_n_grid: dict
        The dictionary containing the grid information for the northern node
        of the winter NAO index.

    nao_s_grid: dict
        The dictionary containing the grid information for the southern node
        of the winter NAO index.

    sig_threshold: float
        The significance threshold for the correlation.

    Returns:
    --------

    stats_dict: dict
        The dictionary containing the correlation statistics.
    """

    # Set up the mdi
    mdi = -9999.0

    # Form the dictionary
    stats_dict = {
        "nao": [],
        "corr_var_ts": [],
        "corr_var": corr_var,
        "corr_nao_var": [],
        "corr_nao_var_pval": [],
        "init_years": [],
        "valid_years": [],
        "lats": [],
        "lons": [],
        "season": season,
        "forecast_range": forecast_range,
        "start_year": start_year,
        "end_year": end_year,
        "sig_threshold": sig_threshold
    }

    # Set up the init years
    stats_dict["init_years"] = np.arange(start_year, end_year + 1)

    # Assert that the season is a winter season
    assert season in ["DJF", "ONDJFM", "DJFM"], "The season must be a winter season."

    # Assert that the forecast range is a valid forecast range
    assert "-" in forecast_range, "The forecast range must be a valid forecast range."

    # Set up the lons and lats for the south grid
    s_lon1, s_lon2 = nao_s_grid["lon1"], nao_s_grid["lon2"]
    s_lat1, s_lat2 = nao_s_grid["lat1"], nao_s_grid["lat2"]

    # and for the north grid
    n_lon1, n_lon2 = nao_n_grid["lon1"], nao_n_grid["lon2"]
    n_lat1, n_lat2 = nao_n_grid["lat1"], nao_n_grid["lat2"]

    # First check that the file exists for psl
    assert os.path.exists(corr_var_obs_file), "The file for the variable to correlate does not exist."

    # Check that the file exists for the NAO index
    assert os.path.exists(nao_obs_file), "The file for the NAO index does not exist."

    # Load the observations for psl
    psl = fnc.load_obs(variable=nao_obs_var,
                   regrid_obs_path=nao_obs_file)
    
    # Load the observations for the matching var
    corr_var_field = fnc.load_obs(variable=corr_var,
                        regrid_obs_path=corr_var_obs_file)
    
    # extract the months
    months = dicts.season_month_map[season]

    # Set up an iris constraint for the start and end years
    start_date = datetime(int(start_year), months[0], 1)
    end_date = datetime(int(end_year), months[-1], 31)

    # Form the constraint
    time_constraint = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)

    # Apply the constraint
    psl = psl.extract(time_constraint)

    # Apply the constraint
    corr_var_field = corr_var_field.extract(time_constraint)

    # Set up the constrain for months
    month_constraint = iris.Constraint(time=lambda cell: cell.point.month in months)

    # Apply the constraint
    psl = psl.extract(month_constraint)
    
    # Apply the constraint
    corr_var_field = corr_var_field.extract(month_constraint)
    
    # Calculate the climatology by collapsing the time dimension
    psl_clim = psl.collapsed("time", iris.analysis.MEAN)

    # Calculate the climatology by collapsing the time dimension
    corr_var_clim = corr_var_field.collapsed("time", iris.analysis.MEAN)

    # Calculate the anomalies
    psl_anom = psl - psl_clim

    # Calculate the anomalies
    corr_var_anom = corr_var_field - corr_var_clim

    # Calculate the annual mean anoms
    psl_anom = fnc.calculate_annual_mean_anomalies(obs_anomalies=psl_anom,
                                               season=season)
    
    # Calculate the annual mean anoms
    corr_var_anom = fnc.calculate_annual_mean_anomalies(obs_anomalies=corr_var_anom,
                                               season=season)
    
    # # Print psl anom at the first time step
    # print("psl anom at the first time step: ", psl_anom.isel(time=0).values)
    
    # # print corr_var anom at the first time step
    # print("corr_var anom at the first time step: ", corr_var_anom.isel(time=0).values)

    # Select the forecast range
    psl_anom = fnc.select_forecast_range(obs_anomalies_annual=psl_anom,
                                        forecast_range=forecast_range)
    
    # Select the forecast range
    corr_var_anom = fnc.select_forecast_range(obs_anomalies_annual=corr_var_anom,
                                        forecast_range=forecast_range)
    
    # Years 2-9, gives an 8 year running mean
    # Which means that the first 4 years (1960, 1961, 1962, 1963) are not valid
    # And the last 4 years (2011, 2012, 2013, 2014) are not valid
    # extract the digits from the forecast range
    digits = [int(x) for x in forecast_range.split("-")]
    # Find the absolute difference between the digits
    diff = abs(digits[0] - digits[1])

    # Find the number of invalid years after centred running mean on each end
    n_invalid_years = diff + 1 / 2

    # Subset corr_var_anom to remove the invalid years
    corr_var_anom = corr_var_anom.isel(time=slice(int(n_invalid_years), -int(n_invalid_years)))
    
    # # Loop over the years in psl_anom
    # for year in psl_anom.time.dt.year.values:
    #     # Extract the data for the year
    #     psl_anom_year = psl_anom.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(psl_anom_year).any():
    #         print("There are NaNs in the psl_anom_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(psl_anom_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the psl_anom
    #             psl_anom = psl_anom.sel(time=psl_anom.time.dt.year != year)

    # # Loop over the first 10 years and last 10 years in psl_anom
    # for year in corr_var_anom.time.dt.year.values[:10]:
    #     # Extract the data for the year
    #     corr_var_anom_year = corr_var_anom.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(corr_var_anom_year).any():
    #         print("There are NaNs in the corr_var_anom_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(corr_var_anom_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the psl_anom
    #             corr_var_anom = corr_var_anom.sel(time=corr_var_anom.time.dt.year != year)

    # # Loop over the last 10 years in psl_anom
    # for year in corr_var_anom.time.dt.year.values[-10:]:
    #     # Extract the data for the year
    #     corr_var_anom_year = corr_var_anom.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(corr_var_anom_year).any():
    #         print("There are NaNs in the corr_var_anom_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(corr_var_anom_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the psl_anom
    #             corr_var_anom = corr_var_anom.sel(time=corr_var_anom.time.dt.year != year)
    
    # print the type of psl_anom
    print("type of psl_anom: ", type(psl_anom))

    # print the type of corr_var_anom
    print("type of corr_var_anom: ", type(corr_var_anom))

    # Extract the years for psl anom
    # years_psl = psl_anom.time.dt.year.values
    years_corr_var = corr_var_anom.time.dt.year.values

    # # Set the time axis for psl_anom to the years
    # psl_anom = psl_anom.assign_coords(time=years_psl)

    # Set the time axis for corr_var_anom to the years
    corr_var_anom = corr_var_anom.assign_coords(time=years_corr_var)

    # Lat goes from 90 to -90
    # Lon goes from 0 to 360

    # # If s_lat1 is smaller than s_lat2, then we need to switch them
    # if s_lat1 < s_lat2:
    #     s_lat1, s_lat2 = s_lat2, s_lat1

    # # If n_lat1 is smaller than n_lat2, then we need to switch them
    # if n_lat1 < n_lat2:
    #     n_lat1, n_lat2 = n_lat2, n_lat1

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= s_lon1 <= 360, "The southern lon is not within the range of 0 to 360."

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= s_lon2 <= 360, "The southern lon is not within the range of 0 to 360."

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= n_lon1 <= 360, "The northern lon is not within the range of 0 to 360."

    # # Asert that the lons are within the range of 0 to 360
    # assert 0 <= n_lon2 <= 360, "The northern lon is not within the range of 0 to 360."

    # Constraint the psl_anom to the south grid
    psl_anom_s = psl_anom.sel(lon=slice(s_lon1, s_lon2),
                               lat=slice(s_lat1, s_lat2)
                               ).mean(dim=["lat", "lon"])

    # Constraint the psl_anom to the north grid
    psl_anom_n = psl_anom.sel(lon=slice(n_lon1, n_lon2),
                               lat=slice(n_lat1, n_lat2)
                               ).mean(dim=["lat", "lon"])
    
    # Calculate the nao index azores - iceland
    nao_index = psl_anom_s - psl_anom_n

    # Loop over the first 10 years and last 10 years in nao_index
    # for year in nao_index.time.dt.year.values:
    #     # Extract the data for the year
    #     nao_index_year = nao_index.sel(time=f"{year}")

    #     # If there are any NaNs, log it
    #     if np.isnan(nao_index_year).any():
    #         print("There are NaNs in the nao_index_year for year: ", year)
    #         # if all values are NaN, then continue
    #         if np.all(np.isnan(nao_index_year)):
    #             print("All values are NaN for year: ", year)
    #             print("Removing the year: ", year)
    #             # Remove the year from the nao_index
    #             nao_index = nao_index.sel(time=nao_index.time.dt.year != year)

    # Subset the nao_index to remove the invalid years
    nao_index = nao_index.isel(time=slice(int(n_invalid_years), -int(n_invalid_years)))

    # Extract the years for nao_index
    years_nao = nao_index.time.dt.year.values

    # Extract the years for corr_var_anom
    years_corr_var = corr_var_anom.time.values

    # Assert that the years are the same
    assert np.array_equal(years_nao, years_corr_var), "The years for the NAO index and the variable to correlate are not the same."

    # Set the valid years
    stats_dict["valid_years"] = years_nao

    # extract tyhe lats and lons
    lats = corr_var_anom.lat.values

    # extract the lons
    lons = corr_var_anom.lon.values

    # Store the lats and lons in the dictionary
    stats_dict["lats"] = lats
    stats_dict["lons"] = lons

    # Extract the values for the NAO index
    nao_index_values = nao_index.values

    # Extract the values for the variable to correlate
    corr_var_anom_values = corr_var_anom.values

    # Store the nao index values in the dictionary
    stats_dict["nao"] = nao_index_values

    # Store the variable to correlate values in the dictionary
    stats_dict["corr_var_ts"] = corr_var_anom_values

    # # Create an empty array with the correct shape for the correlation
    # corr_nao_var = np.empty((len(lats), len(lons)))

    # # Create an empty array with the correct shape for the p-value
    # corr_nao_var_pval = np.empty((len(lats), len(lons)))

    # # Loop over the lats
    # for i in tqdm(range(len(lats)), desc="Calculating spatial correlation"):
    #     # Loop over the lons
    #     for j in range(len(lons)):
    #         # Extract the values for the variable to correlate
    #         corr_var_anom_values = corr_var_anom.values[:, i, j]

    #         # Calculate the correlation
    #         corr, pval = pearsonr(nao_index_values, corr_var_anom_values)

    #         # Store the correlation in the array
    #         corr_nao_var[i, j] = corr

    #         # Store the p-value in the array
    #         corr_nao_var_pval[i, j] = pval

    # # Store the correlation in the dictionary
    # stats_dict["corr_nao_var"] = corr_nao_var

    # # Store the p-value in the dictionary
    # stats_dict["corr_nao_var_pval"] = corr_nao_var_pval

    # return none
    return stats_dict

# define a simple function for plotting the correlation
def plot_corr(corr_array: np.ndarray,
                pval_array: np.ndarray,
                lats: np.ndarray,
                lons: np.ndarray,
                sig_threshold: float = 0.05):
    """
    Plots the correlation and p-values for the spatial correlation.

    Args:
    -----

    corr_array: np.ndarray
        The array containing the correlation values.

    pval_array: np.ndarray
        The array containing the p-values.

    lats: np.ndarray
        The array containing the latitudes.

    lons: np.ndarray
        The array containing the longitudes.

    sig_threshold: float
        The significance threshold for the correlation.

    Returns:
    --------

    None
    """

    # Plot these values
    # Set up a single subplot
    fig = plt.figure(figsize=(10, 5))

    # Set up the projection
    proj = ccrs.PlateCarree(central_longitude=0)

    # Focus on the euro-atlantic region
    lat1_grid, lat2_grid = 20, 80
    lon1_grid, lon2_grid = -100, 40


    lat1_idx_grid = np.argmin(np.abs(lats - lat1_grid))
    lat2_idx_grid = np.argmin(np.abs(lats - lat2_grid))

    lon1_idx_grid = np.argmin(np.abs(lons - lon1_grid))
    lon2_idx_grid = np.argmin(np.abs(lons - lon2_grid))

    # Print the indices
    print("lon1_idx_grid: ", lon1_idx_grid)
    print("lon2_idx_grid: ", lon2_idx_grid)
    print("lat1_idx_grid: ", lat1_idx_grid)
    print("lat2_idx_grid: ", lat2_idx_grid)

    # # If lat1_idx_grid is greater than lat2_idx_grid, then switch them
    # if lat1_idx_grid > lat2_idx_grid:
    #     lat1_idx_grid, lat2_idx_grid = lat2_idx_grid, lat1_idx_grid

    # Print the indices
    print("lon1_idx_grid: ", lon1_idx_grid)
    print("lon2_idx_grid: ", lon2_idx_grid)
    print("lat1_idx_grid: ", lat1_idx_grid)
    print("lat2_idx_grid: ", lat2_idx_grid)

    # Constrain the lats and lons to the grid
    lats = lats[lat1_idx_grid:lat2_idx_grid]
    lons = lons[lon1_idx_grid:lon2_idx_grid]

    # Constrain the corr_array to the grid
    corr_array = corr_array[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # Constrain the pval_array to the grid
    pval_array = pval_array[lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid]

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Set up the axis
    ax = plt.axes(projection=proj)

    # Include coastlines
    ax.coastlines()

    # # Shift lon back to -180 to 180
    # lons = lons - 180

    # Set up the contour plot
    cf = ax.contourf(lons, lats, corr_array, clevs, transform=proj, cmap="RdBu_r")

    # if any of the p values are greater or less than the significance threshold
    sig_threshold = 0.05
    pval_array[(pval_array > sig_threshold) & (pval_array < 1 - sig_threshold)] = np.nan

    # Plot the p-values
    ax.contourf(lons, lats, pval_array, hatches=["...."], alpha=0.0, transform=proj)

    # Set up the colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    # Set up the colorbar label
    cbar.set_label("correlation coefficient")

    # Add a title
    ax.set_title("Corr(obs NAO, obs precip)")

    # Render the plot
    plt.show()

    # Return none
    return None

# Define a function to process the data for plotting scatter plots
def process_data_for_scatter(
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    predictor_var: str,
    predictor_var_dict: dict,
    predictand_var: str,
    predictand_var_file: str,
    region: dict,
    region_name: str,
    quantiles: list = [0.75, 0.95],
    nao_n_grid: dict = dicts.iceland_grid_corrected,
    nao_s_grid: dict = dicts.azores_grid_corrected,
):
    """
    Function which processes the data for the scatter plots.
    
    Args:
    -----
    
    season: str
        The season to calculate the correlation for.
        E.g. ONDJFM, DJFM, DJF
        
    forecast_range: str
        The forecast range to calculate the correlation for.
        E.g. 2-5, 2-9
        
    start_year: int
        The start year to calculate the correlation for.
        E.g. 1960
        
    end_year: int
        The end year to calculate the correlation for.
        E.g. 2014
        
    predictor_var: str
        The variable to use as the predictor.
        E.g. "si10"

    predictor_var_dict: dict
        The dictionary containing the grid information for the predictor variable.
        E.g. {
            "lag": 0, 
            "alt_lag": False,
            "region": "global",
        }

    predictand_var: str
        The variable to use as the predictand.
        E.g. "pr"

    predictand_var_file: str
        The file containing the predictand variable.
        Could be the observed or model data.

    region: dict
        The dictionary containing the region information.
        E.g. {"lon1": 332, "lon2": 340, "lat1": 40, "lat2": 36}
        Could also be a shapefile.

    region_name: str
        The name of the region to use for the scatter plot.
        E.g. "europe"

    quantiles: list
        The quantiles to calculate for the scatter plot.
        E.g. [0.75, 0.95]

    nao_n_grid: dict
        The dictionary containing the grid information for the northern node of the winter NAO index.
        Default is the iceland grid for the regridded data.

    nao_s_grid: dict
        The dictionary containing the grid information for the southern node of the winter NAO index.
        Default is the azores grid for the regridded data.

    Returns:
    --------

    scatter_dict: dict
        The dictionary containing the scatter plot data.
    """

    # Set up the mdi
    mdi = -9999.0

    # Set up the scatter dictionary
    scatter_dict = {
        "predictor_var": predictor_var,
        "predictand_var": predictand_var,
        "season": season,
        "forecast_range": forecast_range,
        "start_year": start_year,
        "end_year": end_year,
        "quantiles": quantiles,
        "region": region,
        "predictor_var_ts": [],
        "predictand_var_ts": [],
        "rval": mdi,
        "pval": mdi,
        "slope": mdi,
        "intercept": mdi,
        "std_err": mdi,
        f"first_quantile_{quantiles[0]}": mdi,
        f"second_quantile_{quantiles[1]}": mdi,
        "init_years": [],
        "valid_years": [],
        "nens": mdi,
        "ts_corr": mdi,
        "ts_pval": mdi,
        "ts_rpc": mdi,
        "ts_rps": mdi,
        "lag": mdi,
        "gridbox": region,
        "gridbox_name": region_name,
        "method": mdi,
    }

    # Set up the init years
    scatter_dict["init_years"] = np.arange(start_year, end_year + 1)

    # Assert that the season is a winter season
    assert season in ["DJF", "ONDJFM", "DJFM"], "The season must be a winter season."

    # # Assert that the file exists for the predictor variable
    # assert os.path.exists(predictor_var_file), "The file for the predictor variable does not exist."

    # # Assert that the file exists for the predictand variable
    # assert os.path.exists(predictand_var_file), "The file for the predictand variable does not exist."

    # Assert that predictor_var_dict is a dictionary
    assert isinstance(predictor_var_dict, dict), "The predictor_var_dict must be a dictionary."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert "lag" in predictor_var_dict.keys(), "The predictor_var_dict must contain a key for lag."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert "alt_lag" in predictor_var_dict.keys(), "The predictor_var_dict must contain a key for alt_lag."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert "region" in predictor_var_dict.keys(), "The predictor_var_dict must contain a key for region."

    # If the region is a dictionary
    if isinstance(region, dict):
        print("The region is a dictionary.")
        print("Extracting the lats and lons from the region dictionary.")
        # Extract the lats and lons from the region dictionary
        lon1, lon2 = region["lon1"], region["lon2"]
        lat1, lat2 = region["lat1"], region["lat2"]
    else:
        print("The region is not a dictionary.")
        AssertionError("The region must be a dictionary. Not a shapefile.")

    # If the predictor var is nao
    if predictor_var == "nao":
        print("The predictor variable is the NAO index.")
        print("Extracting the NAO index from the predictor variable file.")

        # Extract the lats and lons from the region dictionary
        n_lon1, n_lon2 = nao_n_grid["lon1"], nao_n_grid["lon2"]
        n_lat1, n_lat2 = nao_n_grid["lat1"], nao_n_grid["lat2"]

        # Extract the lats and lons from the region dictionary
        s_lon1, s_lon2 = nao_s_grid["lon1"], nao_s_grid["lon2"]
        s_lat1, s_lat2 = nao_s_grid["lat1"], nao_s_grid["lat2"]

        # Echo an error that we have not implemented this yet
        raise NotImplementedError("We have not implemented this yet.")
    else:
        print("The predictor variable is not the NAO index.")
        print("Extracting the predictor variable from the predictor variable file.")

        # Extract the data for the predictor variable
        predictor_var_data = nal_funcs.load_data(
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            method=predictor_var_dict["method"],
            region=predictor_var_dict["region"],
            variable=predictor_var,
        )

        # Load the data for the predictor variable
        rm_dict = p1p_funcs.load_ts_data(
            data=predictor_var_data,
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            gridbox=region,
            gridbox_name=region_name,
            variable=predictor_var,
            alt_lag=predictor_var_dict["alt_lag"],
            region=predictor_var_dict["region"],
        )

        # Append to the scatter dictionary
        scatter_dict["predictor_var_ts"] = rm_dict["fcst_ts_mean"]
        scatter_dict["predictand_var_ts"] = rm_dict["obs_ts"]

        # append the init years
        scatter_dict["init_years"] = rm_dict["init_years"]

        # append the valid years
        scatter_dict["valid_years"] = rm_dict["valid_years"]

        # append the nens
        scatter_dict["nens"] = rm_dict["nens"]

        # append the ts_corr
        scatter_dict["ts_corr"] = rm_dict["corr"]

        # append the ts_pval
        scatter_dict["ts_pval"] = rm_dict["p"]

        # append the ts_rpc
        scatter_dict["ts_rpc"] = rm_dict["rpc"]

        # append the ts_rps
        scatter_dict["ts_rps"] = rm_dict["rps"]

        # append the lag
        scatter_dict["lag"] = rm_dict["lag"]

        # append the gridbox
        scatter_dict["gridbox"] = rm_dict["gridbox"]

        # Append the method
        scatter_dict["method"] = rm_dict["alt_lag"]

        # Ussing the time series
        # perform a linear regression
        # and calculate the quantiles
        # for the scatter plot
        # Calculate the linear regression
        slope, intercept, r_value, p_value, std_err = linregress(scatter_dict["predictor_var_ts"], scatter_dict["predictand_var_ts"])

        # Store the linear regression values in the dictionary
        scatter_dict["rval"] = r_value
        scatter_dict["pval"] = p_value
        scatter_dict["slope"] = slope
        scatter_dict["intercept"] = intercept
        scatter_dict["std_err"] = std_err

        # Define a lamda function for the quantiles
        tinv = lambda p, df: abs(t.ppf(p/2, df))

        # Calculate the degrees of freedom
        df = len(scatter_dict["predictor_var_ts"]) - 2

        # Calculate the first quantile
        q1 = tinv(1 - quantiles[0], df) * scatter_dict["std_err"]

        # Calculate the second quantile
        q2 = tinv(1 - quantiles[1], df) * scatter_dict["std_err"]

        # Store the quantiles in the dictionary
        scatter_dict[f"first_quantile_{quantiles[0]}"] = q1

        # Store the quantiles in the dictionary
        scatter_dict[f"second_quantile_{quantiles[1]}"] = q2

    # Return the dictionary
    return scatter_dict

# Define a function to plot the scatter plot
def plot_scatter(
    scatter_dict: dict,
):
    """
    Function which plots the scatter plot.

    Args:
    -----

    scatter_dict: dict
        The dictionary containing the scatter plot data.

    Returns:
    --------

    None
    """

    # Set up the mdi
    mdi = -9999.0

    # Set up the figure
    fig = plt.figure(figsize=(10, 5))

    # Set up the axis
    ax = fig.add_subplot(111)

    # Set the reg
    reg_line = scatter_dict["slope"] * scatter_dict["predictor_var_ts"] + scatter_dict["intercept"]

    # Set up the 95% confidence interval
    reg_line_95 = scatter_dict["slope"] * scatter_dict["predictor_var_ts"] + scatter_dict["intercept"] + 1.96 * scatter_dict["std_err"]

    # Set up x axis
    x = scatter_dict["predictor_var_ts"]

    # Plot the line
    ax.plot(x, reg_line, color="black", label="regression line")

    # # Plot the first quantile either side of the linear regression line
    ax.fill_between(x, reg_line - reg_line_95, reg_line + reg_line_95, color="lightgray", label="95% confidence interval")

    # Plot the scatter plot
    ax.plot(x, scatter_dict["predictand_var_ts"], "o", label="scatter plot")

    # Set up the x-axis label
    ax.set_xlabel(f"Model {scatter_dict['predictor_var']} anomalies")
                  
    # Set up the y-axis label
    ax.set_ylabel(f"Obs {scatter_dict['predictand_var']} anomalies")

    # Set up the title
    ax.set_title(f"Scatter plot for {scatter_dict['season']} {scatter_dict['forecast_range']} {scatter_dict['start_year']} - {scatter_dict['end_year']} {scatter_dict['gridbox_name']} gridbox")

    # Set up the legend
    ax.legend()

    # # Set up the grid
    # ax.grid()

    # Show the plot
    plt.show()

    # Return none
    return None