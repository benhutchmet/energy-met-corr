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
from datetime import datetime
import geopandas as gpd
import regionmask

# Import local modules
import dictionaries_em as dicts

# Import external modules
sys.path.append("/home/users/benhutch/skill-maps/python/")
import paper1_plots_functions as p1p_funcs
import nao_alt_lag_functions as nal_funcs
import functions as fnc


# Define a function to form the dataframe for the offshore wind farm data
def extract_offshore_eez_to_df(
    filepath: str,
    countries_list: list = [
        "France",
        "Italy",
        "Portugal",
        "Estonia",
        "Latvia",
        "Lithuania",
        "Croatia",
        "Romania",
        "Slovenia",
        "Greece",
        "Montenegro",
        "Albania",
        "Bulgaria",
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
    ],
    rolling_window: int = 8,
    centre: bool = True,
    annual_offset: int = 3,
    months: list = [10, 11, 12, 1, 2, 3],
    start_date: str = "1950-01-01",
    time_unit: str = "h",
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
    df = df.reset_index().pivot(
        index="time_in_hours_from_first_jan_1950",
        columns="NUTS",
        values="timeseries_data",
    )

    # Assuming country_dict is a dictionary that maps NUTS keys to country names
    df.columns = [
        f"{dicts.country_dict[nuts_keys[i]]}_{col}" for i, col in enumerate(df.columns)
    ]

    # Convert 'time_in_hours_from_first_jan_1950' column to datetime
    df.index = pd.to_datetime(df.index, unit=time_unit, origin=start_date)

    # Collapse the dataframes into monthly averages
    df = df.resample("M").mean()

    # Select only the months of interest
    df = df[df.index.month.isin(months)]

    # Shift the data by the annual offset
    df.index = df.index - pd.DateOffset(months=annual_offset)

    # TODO: Fix hard coded here
    # Throw away the first 3 months of data and last 3 months of data
    df = df.iloc[3:-3]

    # Calculate the annual average
    df = df.resample("A").mean()

    # Take the rolling average
    df = df.rolling(window=rolling_window, center=centre).mean()

    # Throw away the NaN values
    df = df.dropna()

    # Return the dataframe
    return df


# Write a function to calculate the stats
def calc_nao_spatial_corr(
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    corr_var: str = "tos",
    corr_var_obs_file: str = dicts.regrid_file,
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
        "sig_threshold": sig_threshold,
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
    assert os.path.exists(
        corr_var_obs_file
    ), "The file for the variable to correlate does not exist."

    # Check that the file exists for the NAO index
    assert os.path.exists(nao_obs_file), "The file for the NAO index does not exist."

    # Load the observations for psl
    psl = fnc.load_obs(variable=nao_obs_var, regrid_obs_path=nao_obs_file)

    # Load the observations for the matching var
    corr_var_field = fnc.load_obs(variable=corr_var, regrid_obs_path=corr_var_obs_file)

    # extract the months
    months = dicts.season_month_map[season]

    # Set up an iris constraint for the start and end years
    start_date = datetime(int(start_year), months[0], 1)
    end_date = datetime(int(end_year), months[-1], 31)

    # Form the constraint
    time_constraint = iris.Constraint(
        time=lambda cell: start_date <= cell.point <= end_date
    )

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
    psl_anom = fnc.calculate_annual_mean_anomalies(
        obs_anomalies=psl_anom, season=season
    )

    # Calculate the annual mean anoms
    corr_var_anom = fnc.calculate_annual_mean_anomalies(
        obs_anomalies=corr_var_anom, season=season
    )

    # # Print psl anom at the first time step
    # print("psl anom at the first time step: ", psl_anom.isel(time=0).values)

    # # print corr_var anom at the first time step
    # print("corr_var anom at the first time step: ", corr_var_anom.isel(time=0).values)

    # Select the forecast range
    psl_anom = fnc.select_forecast_range(
        obs_anomalies_annual=psl_anom, forecast_range=forecast_range
    )

    # Select the forecast range
    corr_var_anom = fnc.select_forecast_range(
        obs_anomalies_annual=corr_var_anom, forecast_range=forecast_range
    )

    # Print the length of the time axis for psl_anom
    print("len(psl_anom.time): ", len(psl_anom.time))

    # Print the length of the time axis for corr_var_anom
    print("len(corr_var_anom.time): ", len(corr_var_anom.time))

    # Years 2-9, gives an 8 year running mean
    # Which means that the first 4 years (1960, 1961, 1962, 1963) are not valid
    # And the last 4 years (2011, 2012, 2013, 2014) are not valid
    # extract the digits from the forecast range
    digits = [int(x) for x in forecast_range.split("-")]
    # Find the absolute difference between the digits
    diff = abs(digits[0] - digits[1])

    # Find the number of invalid years after centred running mean on each end
    n_invalid_years = (diff + 1) / 2

    # Print the number of invalid years
    print("n_invalid_years: ", n_invalid_years)

    # Subset corr_var_anom to remove the invalid years
    corr_var_anom = corr_var_anom.isel(
        time=slice(int(n_invalid_years), -int(n_invalid_years))
    )

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
    psl_anom_s = psl_anom.sel(
        lon=slice(s_lon1, s_lon2), lat=slice(s_lat1, s_lat2)
    ).mean(dim=["lat", "lon"])

    # Constraint the psl_anom to the north grid
    psl_anom_n = psl_anom.sel(
        lon=slice(n_lon1, n_lon2), lat=slice(n_lat1, n_lat2)
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
    assert np.array_equal(
        years_nao, years_corr_var
    ), "The years for the NAO index and the variable to correlate are not the same."

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
def plot_corr(
    corr_array: np.ndarray,
    pval_array: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable: str,
    sig_threshold: float = 0.05,
    plot_gridbox: list = None,
    nao: np.ndarray = None,
    corr_var_ts: np.ndarray = None,
):
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

    variable: str
        The variable to use for the plot title.

    sig_threshold: float
        The significance threshold for the correlation.

    plot_gridbox: list
        List of gridboxes to plot on the plot.

    nao: np.ndarray
        The array containing the NAO index values.

    corr_var_ts: np.ndarray
        The array containing the variable to correlate values.

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

    # If nao and corr_var_ts are not None
    if nao is not None and corr_var_ts is not None:
        # Constraint the corr_var_ts array to the grid
        corr_var_ts = corr_var_ts[
            :, lat1_idx_grid:lat2_idx_grid, lon1_idx_grid:lon2_idx_grid
        ]

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

    # If the plot_gridbox is not None
    if plot_gridbox is not None:
        # Assert that it is a list
        assert isinstance(
            plot_gridbox, list
        ), "The plot_gridbox must be a list of gridboxes."

        # Assert that it is not empty
        assert len(plot_gridbox) > 0, "The plot_gridbox list is empty."

        # Loop over the gridboxes
        for gridbox in plot_gridbox:
            # Extract the lons and lats
            lon1, lon2 = gridbox["lon1"], gridbox["lon2"]
            lat1, lat2 = gridbox["lat1"], gridbox["lat2"]

            # Find the indices for the lons and lats
            lon1_idx = np.argmin(np.abs(lons - lon1))
            lon2_idx = np.argmin(np.abs(lons - lon2))

            lat1_idx = np.argmin(np.abs(lats - lat1))
            lat2_idx = np.argmin(np.abs(lats - lat2))

            # Add the gridbox to the plot
            ax.plot(
                [lon1, lon2, lon2, lon1, lon1],
                [lat1, lat1, lat2, lat2, lat1],
                color="green",
                linewidth=2,
                transform=proj,
            )

            # Constrain the corr_var_ts array to the gridbox
            corr_var_ts_gridbox = corr_var_ts[
                :, lat1_idx:lat2_idx, lon1_idx:lon2_idx
            ].mean(axis=(1, 2))

            # Print the len of the time series
            print("len(corr_var_ts_gridbox): ", len(corr_var_ts_gridbox))
            print("len(nao): ", len(nao))

            # Calculate the correlation
            corr, pval = pearsonr(nao, corr_var_ts_gridbox)

            # Include the correlation on the plot
            ax.text(
                lon2,
                lat2,
                f"r = {corr:.2f}",
                fontsize=8,
                color="white",
                transform=proj,
                bbox=dict(facecolor="green", alpha=0.5, edgecolor="black"),
            )
    else:
        print("No gridboxes to plot.")

        # Add a title
        ax.set_title(f"Correlation (obs NAO, obs {variable})")

    # Set up the colorbar label
    cbar.set_label("correlation coefficient")

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
        "predictor_var_mean": mdi,
        "predictand_var_mean": mdi,
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
    assert isinstance(
        predictor_var_dict, dict
    ), "The predictor_var_dict must be a dictionary."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert (
        "lag" in predictor_var_dict.keys()
    ), "The predictor_var_dict must contain a key for lag."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert (
        "alt_lag" in predictor_var_dict.keys()
    ), "The predictor_var_dict must contain a key for alt_lag."

    # Assert that predictor_var_dict contains keys for lag, alt_lag, and region
    assert (
        "region" in predictor_var_dict.keys()
    ), "The predictor_var_dict must contain a key for region."

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

        # Load the psl data for processing the NAO stats
        psl_data = nal_funcs.load_data(
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            method="alt_lag",
            region=predictor_var_dict["region"],
            variable="psl",
        )

        # Use the function to calculate the NAO stats
        nao_stats = nal_funcs.calc_nao_stats(
            data=psl_data,
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            alt_lag=True,
        )

        # Extract the data for the predictor variable
        predictor_var_data = nal_funcs.load_data(
            season=season,
            forecast_range=forecast_range,
            start_year=start_year,
            end_year=end_year,
            lag=predictor_var_dict["lag"],
            method=predictor_var_dict["method"],
            region=predictor_var_dict["region"],
            variable=predictand_var,
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
            variable=predictand_var,
            alt_lag=predictor_var_dict["alt_lag"],
            region=predictor_var_dict["region"],
        )

        # Append to the scatter dictionary
        scatter_dict["predictor_var_ts"] = nao_stats["model_nao_mean"]
        scatter_dict["predictand_var_ts"] = rm_dict["obs_ts"]

        # append the init years
        scatter_dict["init_years"] = rm_dict["init_years"]

        # append the valid years
        scatter_dict["valid_years"] = rm_dict["valid_years"]

        # append the nens
        scatter_dict["nens"] = nao_stats["nens"]

        # append the ts_corr
        scatter_dict["ts_corr"] = nao_stats["corr1"]

        # append the ts_pval
        scatter_dict["ts_pval"] = nao_stats["p1"]

        # append the ts_rpc
        scatter_dict["ts_rpc"] = nao_stats["rpc1"]

        # append the ts_rps
        scatter_dict["ts_rps"] = nao_stats["rps1"]

        # append the lag
        scatter_dict["lag"] = rm_dict["lag"]

        # append the gridbox
        scatter_dict["gridbox"] = rm_dict["gridbox"]

        # Append the method
        scatter_dict["method"] = "alt_lag"

        # If the predictand variable is 'pr'
        if predictand_var == "pr":
            # Convert obs to mm day-1
            scatter_dict["predictand_var_ts"] = scatter_dict["predictand_var_ts"] * 1000

        # Divide the predictor variable by 100
        scatter_dict["predictor_var_ts"] = scatter_dict["predictor_var_ts"] / 100

        # # Standardize predictor_var_ts
        # scatter_dict["predictor_var_ts"] = (
        #     scatter_dict["predictor_var_ts"] - np.mean(scatter_dict["predictor_var_ts"])
        # ) / np.std(scatter_dict["predictor_var_ts"])

        # # Standardize predictand_var_ts
        # scatter_dict["predictand_var_ts"] = (
        #     scatter_dict["predictand_var_ts"]
        #     - np.mean(scatter_dict["predictand_var_ts"])
        # ) / np.std(scatter_dict["predictand_var_ts"])

        # Perform a linear regression
        # and calculate the quantiles
        slope, intercept, r_value, p_value, std_err = linregress(
            scatter_dict["predictor_var_ts"], scatter_dict["predictand_var_ts"]
        )

        # Store the linear regression values in the dictionary
        scatter_dict["rval"] = r_value
        scatter_dict["pval"] = p_value
        scatter_dict["slope"] = slope
        scatter_dict["intercept"] = intercept
        scatter_dict["std_err"] = std_err

        # Define a lamda function for the quantiles
        tinv = lambda p, df: abs(t.ppf(p / 2, df))

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

        # Calculate the mean of the predictor variable
        scatter_dict["predictor_var_mean"] = np.mean(scatter_dict["predictor_var_ts"])

        # Calculate the mean of the predictand variable
        scatter_dict["predictand_var_mean"] = np.mean(scatter_dict["predictand_var_ts"])
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

        # if the predictor variable is 'pr'
        if predictor_var == "pr":
            # ERA5 is in units of m day-1
            # Model is in units of kg m-2 s-1
            # Convert the model units to m day-1
            scatter_dict["predictor_var_ts"] = scatter_dict["predictor_var_ts"] * 86400

            # ERA5 is in units of m day-1
            # Convert to mm day-1
            scatter_dict["predictand_var_ts"] = scatter_dict["predictand_var_ts"] * 1000

        # Ussing the time series
        # perform a linear regression
        # and calculate the quantiles
        # for the scatter plot
        # Calculate the linear regression
        slope, intercept, r_value, p_value, std_err = linregress(
            scatter_dict["predictor_var_ts"], scatter_dict["predictand_var_ts"]
        )

        # Store the linear regression values in the dictionary
        scatter_dict["rval"] = r_value
        scatter_dict["pval"] = p_value
        scatter_dict["slope"] = slope
        scatter_dict["intercept"] = intercept
        scatter_dict["std_err"] = std_err

        # Define a lamda function for the quantiles
        tinv = lambda p, df: abs(t.ppf(p / 2, df))

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

        # Calculate the mean of the predictor variable
        scatter_dict["predictor_var_mean"] = np.mean(scatter_dict["predictor_var_ts"])

        # Calculate the mean of the predictand variable
        scatter_dict["predictand_var_mean"] = np.mean(scatter_dict["predictand_var_ts"])

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
    fig = plt.figure(figsize=(8, 8))

    # Set up the axis
    ax = fig.add_subplot(111)

    # Find the minimum and maximum of the current x values
    x_min = np.min(scatter_dict["predictor_var_ts"])
    x_max = np.max(scatter_dict["predictor_var_ts"])

    # Extend the range by a certain amount (e.g., 10% of the range)
    x_range = x_max - x_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range

    # Create a new array of x values using this extended range
    x_values_extended = np.linspace(x_min, x_max, 100)

    # print the stanard error from the scatter dictionary
    print("Standard error: ", scatter_dict["std_err"])

    # Plot the regression line using the new array of x values
    ax.plot(
        x_values_extended,
        scatter_dict["slope"] * x_values_extended + scatter_dict["intercept"],
        "k",
    )

    # Print the first quantile
    print(
        "First quantile: ",
        scatter_dict[f"first_quantile_{scatter_dict['quantiles'][0]}"],
    )

    # Print the second quantile
    print(
        "Second quantile: ",
        scatter_dict[f"second_quantile_{scatter_dict['quantiles'][1]}"],
    )

    # Fill between the 95th quantile
    ax.fill_between(
        x_values_extended,
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        - scatter_dict[f"second_quantile_{scatter_dict['quantiles'][1]}"],
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        + scatter_dict[f"second_quantile_{scatter_dict['quantiles'][1]}"],
        color="0.8",
        alpha=0.5,
    )

    # Fill between the 75th quantile
    ax.fill_between(
        x_values_extended,
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        - scatter_dict[f"first_quantile_{scatter_dict['quantiles'][0]}"],
        scatter_dict["slope"] * x_values_extended
        + scatter_dict["intercept"]
        + scatter_dict[f"first_quantile_{scatter_dict['quantiles'][0]}"],
        color="0.6",
        alpha=0.5,
    )

    # Plot the scatter plot
    ax.scatter(scatter_dict["predictor_var_ts"], scatter_dict["predictand_var_ts"])

    # Plot the mean of the predictor variable as a vertical line
    ax.axvline(
        scatter_dict["predictor_var_mean"],
        color="k",
        linestyle="--",
        alpha=0.5,
        linewidth=0.5,
    )

    # Plot the mean of the predictand variable as a horizontal line
    ax.axhline(
        scatter_dict["predictand_var_mean"],
        color="k",
        linestyle="--",
        alpha=0.5,
        linewidth=0.5,
    )

    # Plot the r value in a text box in the top left corner
    ax.text(
        0.05,
        0.95,
        f"r = {scatter_dict['rval']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=12,
    )

    # Set up the x-axis label
    ax.set_xlabel(f"Hindcast {scatter_dict['predictor_var']} anomalies (hPa)")

    # Set up the y-axis label
    ax.set_ylabel(f"Observed {scatter_dict['predictand_var']} anomalies (m/s)")

    # Set up the title
    ax.set_title(
        f"Scatter plot for {scatter_dict['season']} {scatter_dict['forecast_range']} {scatter_dict['start_year']} - {scatter_dict['end_year']} {scatter_dict['gridbox_name']} gridbox"
    )

    # limit the x-axis
    ax.set_xlim(x_min, x_max)

    # Include axis ticks on the inside and outside
    ax.tick_params(axis="both", direction="in", top=True, right=True)

    # Set up the legend
    ax.legend()

    # # Set up the grid
    # ax.grid()

    # Show the plot
    plt.show()

    # Return none
    return None


# Write a function to correlate the NAO with wind power CFs/demand/irradiance
# Within a specific region as defined by the UREAD file being extracted
def correlate_nao_uread(
    filename: str,
    shp_file: str = None,
    shp_file_dir: str = None,
    forecast_range: str = "2-9",
    months: list = [10, 11, 12, 1, 2, 3],
    annual_offset: int = 3,
    start_date: str = "1950-01-01",
    time_unit: str = "h",
    centre: bool = True,
    directory: str = dicts.clearheads_dir,
    obs_var: str = "msl",
    obs_var_data_path: str = dicts.regrid_file,
    start_year: str = "1960",
    end_year: str = "2019",
    nao_n_grid: dict = dicts.iceland_grid_corrected,
    nao_s_grid: dict = dicts.azores_grid_corrected,
    avg_grid: dict = None,
    use_model_data: bool = False,
    model_config: dict = None,
    df_dir: str = "/gws/nopw/j04/canari/users/benhutch/nao_stats_df/",
) -> pd.DataFrame:
    """
    Function which correlates the observed NAO (from ERA5) with demand,
    wind power CFs, irradiance, or other variables, from the UREAD datasets and
    returns the correlation values.

    Args:

    filename: str
        The filename to use for extracting the UREAD data.

    shp_file: str
        The shapefile to use for extracting the UREAD data.

    shp_file_dir: str
        The directory to use for extracting the UREAD data.

    forecast_range: str
        The forecast range to use for extracting the UREAD data.
        Default is "2-9", 8-year running mean.

    months: list
        The months to use for extracting the UREAD data.
        Default is the winter months, October to March.

    annual_offset: int
        The annual offset to use for extracting the UREAD data.
        Default is 3, for the winter months.

    start_date: str
        The start date to use for extracting the UREAD data.
        Default is "1950-01-01".

    time_unit: str
        The time unit to use for extracting the UREAD data.
        Default is "h".

    centre: bool
        Whether to use the centre of the window for the rolling average.

    directory: str
        The directory to use for extracting the UREAD data.

    obs_var: str
        The observed variable to use for calculating the NAO index.
        Default is "msl".

    obs_var_data_path: str
        The path to the observed variable data.

    start_year: int
        The start year to use for extracting the UREAD data.

    end_year: int
        The end year to use for extracting the UREAD data.

    nao_n_grid: dict
        The dictionary containing the grid information for the northern NAO grid.

    nao_s_grid: dict
        The dictionary containing the grid information for the southern NAO grid.

    avg_grid: dict
        The dictionary containing the grid information for the average grid.

    use_model_data
        Whether to use model data or not.

    model_config
        The set up of the model used

    df_dir
        The directory in which the dataframes are stored for the model data

    Returns:

    df: pd.DataFrame
        The dataframe containing the correlation values.
    """

    # Find the files
    files = glob.glob(f"{directory}/{filename}")

    # If there are no files, raise an error
    if len(files) == 0:
        raise FileNotFoundError("No files found.")

    # If there are multiple files, raise an error
    if len(files) > 1:
        raise ValueError("Multiple files found.")

    # Load the data
    data = xr.open_dataset(files[0])

    # Assert that NUTS_keys can be extracted from the data
    assert "NUTS_keys" in data.variables, "NUTS_keys not found in the data."

    # Extract the NUTS keys
    NUTS_keys = data["NUTS_keys"].values

    # Print the nuts keys
    print("NUTS_keys for UREAD data: ", NUTS_keys)

    # Turn this data into a dataframe
    df = data.to_dataframe()

    # # Print the head of the dataframe
    # print("Head of the dataframe: ", df.head())

    # Pivot the dataframe
    df = df.reset_index().pivot(
        index="time_in_hours_from_first_jan_1950",
        columns="NUTS",
        values="timeseries_data",
    )

    # # Print the head of the dataframe again
    # print("Head of the dataframe: ", df.head())

    # Add the nuts keys to the columns
    df.columns = NUTS_keys

    # Convert 'time_in_hours_from_first_jan_1950' column to datetime
    df.index = pd.to_datetime(df.index, unit=time_unit, origin=start_date)

    # Collapse the dataframes into monthly averages
    df = df.resample("M").mean()

    # Select only the months of interest
    df = df[df.index.month.isin(months)]

    # Shift the data by the annual offset
    df.index = df.index - pd.DateOffset(months=annual_offset)

    # TODO: Fix hard coded here
    # Throw away the first 3 months of data and last 3 months of data
    df = df.iloc[3:-3]

    # Calculate the annual average
    df = df.resample("A").mean()

    # Calculate the rolling window
    ff_year = int(forecast_range.split("-")[1])
    lf_year = int(forecast_range.split("-")[0])

    # Calculate the rolling window
    rolling_window = (ff_year - lf_year) + 1  # e.g. (9-2) + 1 = 8

    # # Print the first 10 rows of the dataframe
    # print("First 10 rows of the dataframe: ", df.head(10))

    # Take the rolling average
    df = df.rolling(window=rolling_window, center=centre).mean()

    # Throw away the NaN values
    df = df.dropna()

    # load in the ERA5 data
    clim_var = xr.open_mfdataset(
        obs_var_data_path,
        combine="by_coords",
        parallel=True,
        chunks={"time": "auto", "latitude": "auto", "longitude": "auto"},
    )[
        obs_var
    ]  # for mean sea level pressure

    # If expver is a variable in the dataset
    if "expver" in clim_var.coords:
        # Combine the first two expver variables
        clim_var = clim_var.sel(expver=1).combine_first(clim_var.sel(expver=5))

    # Constrain obs to ONDJFM
    clim_var = clim_var.sel(time=clim_var.time.dt.month.isin(months))

    # Shift the time index back by 3 months
    clim_var_shifted = clim_var.shift(time=-annual_offset)

    # Take annual means
    clim_var_annual = clim_var_shifted.resample(time="Y").mean()

    # Throw away years 1959, 2021, 2022 and 2023
    clim_var_annual = clim_var_annual.sel(time=slice(start_year, end_year))

    # Remove the climatology
    clim_var_anomaly = clim_var_annual - clim_var_annual.mean(dim="time")

    # If the obs var is "msl"
    if obs_var == "msl" and use_model_data is False:
        # Print that we are using msl and calculating the NAO index
        print("Using mean sea level pressure to calculate the NAO index.")

        # Extract the lat and lons of iceland
        lat1_n, lat2_n = nao_n_grid["lat1"], nao_n_grid["lat2"]
        lon1_n, lon2_n = nao_n_grid["lon1"], nao_n_grid["lon2"]

        # Extract the lat and lons of the azores
        lat1_s, lat2_s = nao_s_grid["lat1"], nao_s_grid["lat2"]
        lon1_s, lon2_s = nao_s_grid["lon1"], nao_s_grid["lon2"]

        # Calculate the msl mean for the icealndic region
        msl_mean_n = clim_var_anomaly.sel(
            lat=slice(lat1_n, lat2_n), lon=slice(lon1_n, lon2_n)
        ).mean(dim=["lat", "lon"])

        # Calculate the msl mean for the azores region
        msl_mean_s = clim_var_anomaly.sel(
            lat=slice(lat1_s, lat2_s), lon=slice(lon1_s, lon2_s)
        ).mean(dim=["lat", "lon"])

        # Calculate the NAO index (azores - iceland)
        nao_index = msl_mean_s - msl_mean_n

        # Extract the time values
        time_values = nao_index.time.values

        # Extract the values
        nao_values = nao_index.values

        # Create a dataframe for the NAO data
        nao_df = pd.DataFrame({"time": time_values, "NAO anomaly (Pa)": nao_values})

        # Take a central rolling average
        nao_df = (
            nao_df.set_index("time")
            .rolling(window=rolling_window, center=centre)
            .mean()
        )

        # Drop the NaN values
        nao_df = nao_df.dropna()

        # Print the head of nao_df
        print("NAO df head: ", nao_df.head())

        # Print the head of df
        print("df head: ", df.head())

        # Merge the dataframes, using the index of the first
        merged_df = df.join(nao_df, how="inner")

        # Drop the NaN values
        merged_df = merged_df.dropna()

        # Create a new dataframe for the correlations
        corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

        # Loop over the columns
        for col in merged_df.columns[:-1]:
            # Calculate the correlation
            corr, pval = pearsonr(merged_df[col], merged_df["NAO anomaly (Pa)"])

            # Append to the dataframe
            corr_df_to_append = pd.DataFrame(
                {"region": [col], "correlation": [corr], "p-value": [pval]}
            )

            # Append to the dataframe
            corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)
    elif obs_var == "msl" and use_model_data is True:
        print("Extracting the stored NAO data from the model data.")

        # set up the file name using model config
        model_filename = f"""{model_config["variable"]}_{model_config["season"]}_{model_config["region"]}_{model_config["start_year"]}_{model_config["end_year"]}_{model_config["forecast_range"]}_{model_config['lag']}_{model_config['nao']}.csv"""

        # Set up the path to the file
        filepath = f"{df_dir}{model_filename}"

        # assert that the file exists
        assert os.path.exists(filepath), "The file does not exist."

        # print the filepath
        print("Filepath: ", filepath)

        # Load the dataframe
        df_model_nao = pd.read_csv(filepath)

        # process the observations
        # Extract the lat and lons of iceland
        lat1_n, lat2_n = nao_n_grid["lat1"], nao_n_grid["lat2"]
        lon1_n, lon2_n = nao_n_grid["lon1"], nao_n_grid["lon2"]

        # Extract the lat and lons of the azores
        lat1_s, lat2_s = nao_s_grid["lat1"], nao_s_grid["lat2"]
        lon1_s, lon2_s = nao_s_grid["lon1"], nao_s_grid["lon2"]

        # Calculate the msl mean for the icealndic region
        msl_mean_n = clim_var_anomaly.sel(
            lat=slice(lat1_n, lat2_n), lon=slice(lon1_n, lon2_n)
        ).mean(dim=["lat", "lon"])

        # Calculate the msl mean for the azores region
        msl_mean_s = clim_var_anomaly.sel(
            lat=slice(lat1_s, lat2_s), lon=slice(lon1_s, lon2_s)
        ).mean(dim=["lat", "lon"])

        # Calculate the NAO index (azores - iceland)
        nao_index = msl_mean_s - msl_mean_n

        # Extract the time values
        time_values = nao_index.time.values

        # Extract the values
        nao_values = nao_index.values

        # Create a dataframe for the NAO data
        nao_df = pd.DataFrame({"time": time_values, "NAO anomaly (Pa)": nao_values})

        # Take a central rolling average
        nao_df = (
            nao_df.set_index("time")
            .rolling(window=rolling_window, center=centre)
            .mean()
        )

        # Drop the NaN values
        nao_df = nao_df.dropna()

        # Set the year as the index
        nao_df.index = nao_df.index.year

        # Set the index for the loaded data as valid_time
        df_model_nao = df_model_nao.set_index("valid_time")

        # join the two dataframes
        merged_df = df_model_nao.join(nao_df)

        # Set the volumn with the name value to obs_nao_pd
        merged_df = merged_df.rename(columns={"value": "obs_nao_pd"})

        # Print the head of this df
        print("Head of merged_df: ", merged_df.head())

        # For rthe uread dataset, set the index to years
        df.index = df.index.year

        # print the head of the UREAD df
        print("Head of UREAD df: ", df.head())

        # merge with the CF data
        merged_df_full = df.join(merged_df, how="inner")

        # print the head of the merged
        print(merged_df_full.head())

        # Create a new dataframe for the correlations
        corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

        # Loop over the columns
        for col in merged_df_full.columns[:-6]:
            # Calculate the correlation
            corr, pval = pearsonr(merged_df_full[col], merged_df_full["model_nao_mean"])

            # Append to the dataframe
            corr_df_to_append = pd.DataFrame(
                {"region": [col], "correlation": [corr], "p-value": [pval]}
            )

            # Append to the dataframe
            corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

        return df, merged_df, merged_df_full, corr_df

    else:
        print("The observed variable is not mean sea level pressure.")
        print("calculating correlation skill for gridpoint variable")

        # If the filename contains the string "eez"
        if shp_file is not None and "eez" in shp_file:
            print("Averaging data for EEZ domains")

            # Assert that shp_file is not None
            assert shp_file is not None, "The shapefile is None."

            # Assert that shp_file_dir is not None
            assert shp_file_dir is not None, "The shapefile directory is None."

            # Assert that the shp_file_dir exists
            assert os.path.exists(
                shp_file_dir
            ), "The shapefile directory does not exist."

            # Load the shapefile
            shapefile = gpd.read_file(os.path.join(shp_file_dir, shp_file))

            # Assert that the shp_file exists
            assert os.path.exists(
                os.path.join(shp_file_dir, shp_file)
            ), "The shapefile does not exist."

            # Throw away all columns
            # Apart from "GEONAME", "ISO_SOV1", and "geometry"
            shapefile = shapefile[["GEONAME", "ISO_SOV1", "geometry"]]

            # Pass the NUTS keys through the filter
            iso_sov_values = [dicts.iso_mapping[key] for key in NUTS_keys]

            # Constrain the geo dataframe to only include these values
            shapefile = shapefile[shapefile["ISO_SOV1"].isin(iso_sov_values)]

            # Filter df to only include the rows where GEONAME includes: "Exclusive Economic Zone"
            shapefile = shapefile[
                shapefile["GEONAME"].str.contains("Exclusive Economic Zone")
            ]

            # Remove any rows from EEZ shapefile which contain "(*)" in the GEONAME column
            # To limit to only Exlusive economic zones
            shapefile = shapefile[~shapefile["GEONAME"].str.contains(r"\(.*\)")]

            # Print the shape of clim_var_anomaly
            print("Shape of clim_var_anomaly: ", clim_var_anomaly.shape)

            # Calculate the mask
            # CALCULATE MASK
            shapefile["numbers"] = range(len(shapefile))

            # test the function
            eez_mask_poly = regionmask.from_geopandas(
                shapefile,
                names="GEONAME",
                abbrevs="ISO_SOV1",
                numbers="numbers",
            )

            # Create a mask to apply to the gridded dataset
            clim_var_anomaly_subset = clim_var_anomaly.isel(time=0)

            # Create the eez mask
            eez_mask = eez_mask_poly.mask(
                clim_var_anomaly_subset["lon"],
                clim_var_anomaly_subset["lat"],
            )

            # Create a dataframe
            df_ts = pd.DataFrame({"time": clim_var_anomaly.time.values})

            # Extract the lat and lons for the mask
            lat = eez_mask.lat.values
            lon = eez_mask.lon.values

            # Set up the n_flags
            n_flags = len(eez_mask.attrs["flag_values"])

            # Loop over the regions
            for i in tqdm((range(n_flags))):
                # Add a new column to the dataframe
                df_ts[eez_mask.attrs["flag_meanings"].split(" ")[i]] = np.nan

                # # Print the region
                print(
                    f"Calculating correlation for region: {eez_mask.attrs['flag_meanings'].split(' ')[i]}"
                )

                # Extract the mask for the region
                sel_mask = eez_mask.where(eez_mask == i).values

                # Set up the lon indices
                id_lon = lon[np.where(~np.all(np.isnan(sel_mask), axis=0))]

                # Set up the lat indices
                id_lat = lat[np.where(~np.all(np.isnan(sel_mask), axis=1))]

                # If the length of id_lon is 0 and the length of id_lat is 0
                if len(id_lon) == 0 and len(id_lat) == 0:
                    print(
                        f"Region {eez_mask.attrs['flag_meanings'].split(' ')[i]} is empty."
                    )
                    print("Continuing to the next region.")
                    continue

                # Print the id_lat and id_lon
                print("id_lat[0], id_lat[-1]: ", id_lat[0], id_lat[-1])

                # Print the id_lat and id_lon
                print("id_lon[0], id_lon[-1]: ", id_lon[0], id_lon[-1])

                # Select the region from the anoms
                out_sel = (
                    clim_var_anomaly.sel(
                        lat=slice(id_lat[0], id_lat[-1]),
                        lon=slice(id_lon[0], id_lon[-1]),
                    )
                    .compute()
                    .where(eez_mask == i)
                )

                # Group this into a mean
                out_sel = out_sel.mean(dim=["lat", "lon"])

                # Add this to the dataframe
                df_ts[eez_mask.attrs["flag_meanings"].split(" ")[i]] = out_sel.values

            # Take the central rolling average
            df_ts = (
                df_ts.set_index("time")
                .rolling(window=rolling_window, center=centre)
                .mean()
            )

            # modify each of the column names to include '_si10'
            # at the end of the string
            df_ts.columns = [
                f"{col}_{obs_var}" for col in df_ts.columns if col != "time"
            ]

            # pRINT THE column names
            print("Column names df_ts: ", df_ts.columns)

            # print the shape of df_ts
            print("Shape of df_ts: ", df_ts.shape)

            # Print df_ts head
            print("df_ts head: ", df_ts.head())

            # Drop the first rolling_window/2 rows
            df_ts = df_ts.iloc[int(rolling_window / 2) :]

            # join the dataframes
            merged_df = df.join(df_ts, how="inner")

            # Print merged df
            print("Merged df before NaN removed: ", merged_df.head())

            # # Drop the NaN values
            # merged_df = merged_df.dropna()

            # Print merged df
            print("Merged df: after Nan removed ", merged_df.head())

            # Print the column names in merged df
            print("Column names in merged df: ", merged_df.columns)

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Find the length of the merged_df.columns which don't contain "Si10"
            n_cols = len(
                [
                    col
                    for col in merged_df.columns
                    if obs_var not in col and "time" not in col
                ]
            )

            # print ncols
            print("Number of columns: ", n_cols)

            # Loop over the columns
            for i in tqdm(range(n_cols)):
                # Extract the column
                col = merged_df.columns[i]

                # Convert col to iso bname
                col_iso = dicts.iso_mapping[col]

                # If merged_df[f"{col_iso}_{obs_var}"] doesn't exist
                # Then create this
                # and fill with NaN values
                if f"{col_iso}_{obs_var}" not in merged_df.columns:
                    merged_df[f"{col_iso}_{obs_var}"] = np.nan

                # Check whether the length of the column is 4
                assert (
                    len(merged_df[col]) >= 2
                ), f"The length of the column is less than 2 for {col}"

                # Same check for the other one
                assert (
                    len(merged_df[f"{col_iso}_{obs_var}"]) >= 2
                ), f"The length of the column is less than 2 for {col_iso}_{obs_var}"

                # If merged_df[f"{col_iso}_{obs_var}"] contains NaN values
                # THEN fill the corr and pval with NaN
                if merged_df[f"{col_iso}_{obs_var}"].isnull().values.any():
                    corr = np.nan
                    pval = np.nan

                    # Append to the dataframe
                    corr_df_to_append = pd.DataFrame(
                        {"region": [col], "correlation": [corr], "p-value": [pval]}
                    )

                    # Append to the dataframe
                    corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

                    # continue to the next iteration
                    continue

                # Calculate corr between wind power (GW) and wind speed
                corr, pval = pearsonr(merged_df[col], merged_df[f"{col_iso}_{obs_var}"])

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

            # Return the dataframe
            return merged_df, corr_df, shapefile
        elif shp_file is not None and "NUTS" in shp_file:
            print("Averaging data for NUTS regions")

            NotImplementedError(
                "This function is not yet implemented for NUTS regions."
            )
        elif use_model_data is False:
            print("Averaging over specified gridbox")

            # assert that avg_grid is not none
            assert avg_grid is not None, "The average grid is None."

            # Extract the lat and lons from the avg_grid
            lon1, lon2 = avg_grid["lon1"], avg_grid["lon2"]
            lat1, lat2 = avg_grid["lat1"], avg_grid["lat2"]

            # Calculate the mean of the clim var anomalies for this region
            clim_var_mean = clim_var_anomaly.sel(
                lat=slice(lat1, lat2), lon=slice(lon1, lon2)
            ).mean(dim=["lat", "lon"])

            # Extract the time values
            time_values = clim_var_mean.time.values

            # Extract the values
            clim_var_values = clim_var_mean.values

            # Create a dataframe for this data
            clim_var_df = pd.DataFrame(
                {"time": time_values, f"{obs_var} anomaly mean": clim_var_values}
            )

            # Take the central rolling average
            clim_var_df = (
                clim_var_df.set_index("time")
                .rolling(window=rolling_window, center=centre)
                .mean()
            )

            # Drop the NaN values
            clim_var_df = clim_var_df.dropna()

            # Merge the dataframes
            merged_df = df.join(clim_var_df, how="inner")

            # Drop the NaN values
            merged_df = merged_df.dropna()

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Loop over the columns
            for col in merged_df.columns[:-1]:
                # Calculate the correlation
                corr, pval = pearsonr(
                    merged_df[col], merged_df[f"{obs_var} anomaly mean"]
                )

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)
        elif use_model_data is True:
            print(
                "Extracting the stored gridbox averaged variable data for the specified box"
            )

            # TODO: finish off this function here
            # Set up the filename for the data
            # in the format:
            model_filename = f"""{model_config["variable"]}_{model_config["region"]}_{model_config["season"]}_{model_config["forecast_range"]}_{model_config["start_year"]}_{model_config["end_year"]}_{model_config["lag"]}_{model_config["gridbox"]}_{model_config["method"]}.csv"""

            # Print the filename
            print("Model filename: ", model_filename)

            # Set up the filepath
            filepath = f"{df_dir}{model_filename}"

            # Asser that the filepath exists
            assert os.path.exists(filepath), f"The filepath: {filepath} does not exist."

            # Load the dataframe
            df_model = pd.read_csv(filepath)

            # Set the index for the loaded data as valid time
            df_model = df_model.set_index("valid_years")

            # Set the df index as the year
            df.index = df.index.year

            # # Print the head of the df_model dataframe
            # print("Head of df_model: ", df_model.head())

            # # Print the head of the UREAD data
            # print("Head of UREAD data: ", df.head())

            # Try to join the two datadrames
            try:
                merged_df = df.join(df_model, how="inner")
            except Exception as e:
                print("Error: ", e)

            # Create a new dataframe for the correlations
            corr_df = pd.DataFrame(columns=["region", "correlation", "p-value"])

            # Loop over the columns
            for col in merged_df.columns[:-6]:
                # Calculate the correlation
                corr, pval = pearsonr(merged_df[col], merged_df["fcst_ts_mean"])

                # Append to the dataframe
                corr_df_to_append = pd.DataFrame(
                    {"region": [col], "correlation": [corr], "p-value": [pval]}
                )

                # Append to the dataframe
                corr_df = pd.concat([corr_df, corr_df_to_append], ignore_index=True)

            # Return these dfs
            return df, df_model, merged_df, corr_df

        else:
            raise ValueError("The shapefile is not recognised.")

    # Return the dataframe
    return merged_df, corr_df
