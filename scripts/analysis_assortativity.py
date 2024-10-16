### export PYTHONPATH="${PYTHONPATH}:" if want to run from chile_emergent
### python3 scripts/analysis_assortativity.py 

#################### Dependencies

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns

from scipy.stats import pearsonr, spearmanr

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

import datetime

import os
import time
from tqdm import tqdm

from utils import helper

#################### Functions

def assortativity_values_all(df_input):
    
    assortativity_values_ = []
    
    for date in df_input['date'].sort_values().unique():
        df = df_input[df_input['date'] == date].reset_index(drop=True)
        heatmap = df.groupby(['quantile_pop_bins_percent_bachelor_home', 'quantile_pop_bins_percent_bachelor_current'], observed=False).size()
        heatmap = heatmap.reset_index().pivot_table(index='quantile_pop_bins_percent_bachelor_current', columns='quantile_pop_bins_percent_bachelor_home', values=0)
        normalised_heatmap = heatmap / np.sum(np.matrix(heatmap))
        assortativity = helper.rho_assortativity(np.matrix(normalised_heatmap))
        assortativity_values_.append(assortativity)

    return assortativity_values_

def assortativity_values_moved(df_input):
    
    assortativity_values_ = []
    
    for date in df_input['date'].sort_values().unique():
        df = df_input[df_input['moved']==1]
        df = df[df['date'] == date].reset_index(drop=True)
        heatmap = df.groupby(['quantile_pop_bins_percent_bachelor_home', 'quantile_pop_bins_percent_bachelor_current'], observed=False).size()
        heatmap = heatmap.reset_index().pivot_table(index='quantile_pop_bins_percent_bachelor_current', columns='quantile_pop_bins_percent_bachelor_home', values=0)
        normalised_heatmap = heatmap / np.sum(np.matrix(heatmap))
        assortativity = helper.rho_assortativity(np.matrix(normalised_heatmap))
        assortativity_values_.append(assortativity)

    return assortativity_values_

def assortativity_values_moved_boosted(df_input, num_simulations = 1000):
    
    dates_ = []
    assortativity_values_ = []
    
    for date in df_input['date'].sort_values().unique():
        df = df_input[df_input['moved']==1]
        df = df[df['date'] == date].reset_index(drop=True)
        heatmap = df.groupby(['quantile_pop_bins_percent_bachelor_home', 'quantile_pop_bins_percent_bachelor_current'], observed=False).size()
        heatmap = heatmap.reset_index().pivot_table(index='quantile_pop_bins_percent_bachelor_current', columns='quantile_pop_bins_percent_bachelor_home', values=0)
        normalised_heatmap = heatmap / np.sum(np.matrix(heatmap))
        
        
        date_assortativity_values = []

        for n in tqdm(range(num_simulations)):
            # Resample the data with replacement
            resampled_df = df.sample(frac=1, replace=True)
            
            # Create the heatmap for the resampled data
            resampled_heatmap = resampled_df.groupby(['quantile_pop_bins_percent_bachelor_home', 'quantile_pop_bins_percent_bachelor_current'], observed=False).size()
            resampled_heatmap = resampled_heatmap.reset_index().pivot_table(index='quantile_pop_bins_percent_bachelor_current', columns='quantile_pop_bins_percent_bachelor_home', values=0)
            resampled_normalised_heatmap = resampled_heatmap / np.sum(np.matrix(resampled_heatmap))
    
            # Compute assortativity for the resampled heatmap
            assortativity = helper.rho_assortativity(np.matrix(resampled_normalised_heatmap))
            date_assortativity_values.append(assortativity)

        assortativity_values_.append(date_assortativity_values)
        dates_.append(date)

    results_df_notaffected = pd.DataFrame(assortativity_values_, index=dates_).T
    return results_df_notaffected

def assortativity_values_all_boosted(df_input, num_simulations = 1000):
    
    dates_ = []
    assortativity_values_ = []
    
    for date in df_input['date'].sort_values().unique():
        df = df_input[df_input['date'] == date].reset_index(drop=True)
        heatmap = df.groupby(['quantile_pop_bins_percent_bachelor_home', 'quantile_pop_bins_percent_bachelor_current'], observed=False).size()
        heatmap = heatmap.reset_index().pivot_table(index='quantile_pop_bins_percent_bachelor_current', columns='quantile_pop_bins_percent_bachelor_home', values=0)
        normalised_heatmap = heatmap / np.sum(np.matrix(heatmap))
        
        
        date_assortativity_values = []

        for n in tqdm(range(num_simulations)):
            # Resample the data with replacement
            resampled_df = df.sample(frac=0.1, replace=True)
            
            # Create the heatmap for the resampled data
            resampled_heatmap = resampled_df.groupby(['quantile_pop_bins_percent_bachelor_home', 'quantile_pop_bins_percent_bachelor_current'], observed=False).size()
            resampled_heatmap = resampled_heatmap.reset_index().pivot_table(index='quantile_pop_bins_percent_bachelor_current', columns='quantile_pop_bins_percent_bachelor_home', values=0)
            resampled_normalised_heatmap = resampled_heatmap / np.sum(np.matrix(resampled_heatmap))
    
            # Compute assortativity for the resampled heatmap
            assortativity = helper.rho_assortativity(np.matrix(resampled_normalised_heatmap))
            date_assortativity_values.append(assortativity)

        assortativity_values_.append(date_assortativity_values)
        dates_.append(date)

    results_df = pd.DataFrame(assortativity_values_, index=dates_).T
    return results_df

def compute_boosted_metrics(assortativity_df, boosted_notaffected, boosted_evacuated):
    
    assortativity_df['mean_simulated_notaff']=boosted_notaffected.mean().values
    assortativity_df['ci_lower_notaff']=boosted_notaffected.quantile(0.025).values
    assortativity_df['ci_upper_notaff']=boosted_notaffected.quantile(0.975).values
    assortativity_df['mean_simulated_evac']=boosted_evacuated.mean().values
    assortativity_df['ci_lower_evac']=boosted_evacuated.quantile(0.025).values
    assortativity_df['ci_upper_evac']=boosted_evacuated.quantile(0.975).values

    return assortativity_df

    
#################### Loading Data
start_time = time.time()

print('Loading data...')

homelocations_merged=pd.read_parquet('data_created/homelocations_merged.parquet')
evacuated_people = helper.upload_list_txt('data_created/evacuated.txt')

result = helper.upload_if_exists('data_created/bootstrapped_merged_noses.parquet')
result_evac = helper.upload_if_exists('data_created/bootstrapped_evac_noses.parquet')
result_ses = helper.upload_if_exists('data_created/bootstrapped_merged_ses.parquet')
result_evac_ses = helper.upload_if_exists('data_created/bootstrapped_evac_ses.parquet')

#################### Analysis
print('Started computing assortativities...')

homelocations_notaffected = homelocations_merged[homelocations_merged['affected'] == 0].dropna(subset=['home_bts_id','quantile_pop_bins_percent_bachelor_home','quantile_pop_bins_percent_bachelor_current'])
homelocations_evacuated = homelocations_merged[homelocations_merged['PHONE_ID'].isin(evacuated_people)].dropna(subset=['home_bts_id','quantile_pop_bins_percent_bachelor_home','quantile_pop_bins_percent_bachelor_current'])

assortativity_df_plot = pd.DataFrame()
assortativity_moved_df_plot = pd.DataFrame()

assortativity_moved_df_plot['dates']=homelocations_merged['date'].sort_values().unique()
assortativity_moved_df_plot['assortativity_control']=assortativity_values_moved(homelocations_notaffected)
assortativity_moved_df_plot['assortativity_evacuated']=assortativity_values_moved(homelocations_evacuated)

print(assortativity_moved_df_plot.head(10))

assortativity_df_plot['dates']=homelocations_merged['date'].sort_values().unique()
assortativity_df_plot['assortativity_control']=assortativity_values_all(homelocations_notaffected)
assortativity_df_plot['assortativity_evacuated']=assortativity_values_all(homelocations_evacuated)

print(assortativity_df_plot.head(10))

print('Started bootstrapping...')

results_df_moved_boosted_notaffected = assortativity_values_moved_boosted(homelocations_notaffected)
results_df_moved_boosted_evacuated = assortativity_values_moved_boosted(homelocations_evacuated)

results_df_boosted_notaffected = assortativity_values_all_boosted(homelocations_notaffected)
results_df_boosted_evacuated = assortativity_values_all_boosted(homelocations_evacuated)

assortativity_moved_df_plot = compute_boosted_metrics(assortativity_moved_df_plot, results_df_moved_boosted_notaffected, results_df_moved_boosted_evacuated)
assortativity_df_plot = compute_boosted_metrics(assortativity_df_plot, results_df_boosted_notaffected, results_df_boosted_evacuated)

#################### Save Datasets
print('Saving data...')

assortativity_df_plot.to_csv('data_created/assortativity_all.csv')
assortativity_moved_df_plot.to_csv('data_created/assortativity_moved.csv')

print("--- %s seconds ---" % (time.time() - start_time))