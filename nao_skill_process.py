# batch script for running in background

# Import local modules
import sys
import os
import glob

# Import third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import iris
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# import cdsapi
# import xesmf as xe
from datetime import datetime
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.append("/home/users/benhutch/energy-met-corr-functions")

# Import the semi-local functions
import functions_em as funcs_em

sys.path.append("/home/users/benhutch/energy-met-corr")
import dictionaries_em as dicts

sys.path.append("/home/users/benhutch/skill-maps/python")
import functions as fnc

sys.path.append("/home/users/benhutch/skill-maps")
import dictionaries as dicts_sm

# Test this function
stats_dict_tas = funcs_em.calc_nao_spatial_corr(
    season="ONDJFM",
    forecast_range="2-9",
    start_year=1960,
    end_year=2023,
    corr_var="t2m",
    corr_var_obs_file=dicts.regrid_file,
    nao_n_grid=dicts.uk_n_box_corrected,
    nao_s_grid=dicts.uk_s_box_corrected,
)

# Calculate the stats for the precipitation
stats_dict_sfcWind = funcs_em.calc_nao_spatial_corr(
    season="ONDJFM",
    forecast_range="2-9",
    start_year=1960,
    end_year=2023,
    corr_var="si10",
    nao_n_grid=dicts.uk_n_box_corrected,
    nao_s_grid=dicts.uk_s_box_corrected,
)

# Calculate the stats for the precipitation
stats_dict_rsds = funcs_em.calc_nao_spatial_corr(
    season="ONDJFM",
    forecast_range="2-9",
    start_year=1960,
    end_year=2023,
    corr_var="ssrd",
    nao_n_grid=dicts.uk_n_box_corrected,
    nao_s_grid=dicts.uk_s_box_corrected,
)

# Calculate the stats for the precipitation
stats_dict_pr = funcs_em.calc_nao_spatial_corr(
    season="ONDJFM",
    forecast_range="2-9",
    start_year=1960,
    end_year=2023,
    corr_var="var228",
    corr_var_obs_file=dicts.regrid_file_pr,
    nao_n_grid=dicts.uk_n_box_corrected,
    nao_s_grid=dicts.uk_s_box_corrected,
)

corr_tas, pval_tas = funcs_em.calculate_correlation_and_pvalue(
    stats_dict = stats_dict_tas,
    nboot=1000,
)

# Same for sfcwind
corr_sfcWind, pval_sfcWind = funcs_em.calculate_correlation_and_pvalue(
    stats_dict = stats_dict_sfcWind,
    nboot=1000,
)

# Same for rsds
corr_rsds, pval_rsds = funcs_em.calculate_correlation_and_pvalue(
    stats_dict = stats_dict_rsds,
    nboot=1000,
)

# same for pr
corr_pr, pval_pr = funcs_em.calculate_correlation_and_pvalue(
    stats_dict = stats_dict_pr,
    nboot=1000,
)

# Form the lists
corr_arrays = [
    corr_tas,
    corr_sfcWind,
    corr_rsds,
    corr_pr,
]
pval_arrays = [
    pval_tas,
    pval_sfcWind,
    pval_rsds,
    pval_pr,
]

# corr_var_ts
corr_var_ts = [
    stats_dict_tas["corr_var_ts"],
    stats_dict_sfcWind["corr_var_ts"],
    stats_dict_rsds["corr_var_ts"],
    stats_dict_pr["corr_var_ts"],
]

# List of variables
variables = [
    "tas",
    "sfcWind",
    "rsds",
    "pr",
]

# List of fig labels
fig_labels = [
    "a",
    "b",
    "c",
    "d",
]

# Plot_gribdox
plot_gridboxes = [
    dicts_sm.uk_grid,
    dicts_sm.north_sea_kay,
    dicts_sm.med_box_focus,
    dicts_sm.scandi_box,
]

# Test the subplots functions
funcs_em.plot_corr_subplots(
    corr_arrays=corr_arrays,
    pval_arrays=pval_arrays,
    lats=stats_dict_tas["lats"],
    lons=stats_dict_tas["lons"],
    variables=variables,
    sig_threshold=0.05,
    plot_gridbox=plot_gridboxes,
    nao=stats_dict_tas["nao"],
    corr_var_ts=corr_var_ts,
    figsize_x=10,
    figsize_y=8,
    save_dpi=1000,
    plot_dir="/home/users/benhutch/energy-met-corr/plots",
    fig_labels=fig_labels,
    fontsize=10,
    w_space=0.05,
    h_space=0.05,
)