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
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import local modules
import dictionaries as dicts

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
