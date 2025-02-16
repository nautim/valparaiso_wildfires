### export PYTHONPATH="${PYTHONPATH}:" if want to run from chile_emergent
### python3 scripts/analysis_regressions.py 

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

from utils import helper


#################### Functions

def process_data(df, ses=False):

    if ses==False:
        total_residents_per_date = df.groupby(['date', 'affected'], observed=False)['PHONE_ID'].count()
        grouped_data_fraction = df.groupby(['date', 'affected', 'moved'],observed=False)['PHONE_ID'].count() / total_residents_per_date
    
    else:
        total_residents_per_date = df.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'], observed=False)['PHONE_ID'].count()
        grouped_data_fraction = df.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home', 'moved'], observed=False)['PHONE_ID'].count() / total_residents_per_date
        
    grouped_data_fraction = grouped_data_fraction.reset_index()
    grouped_data_fraction = grouped_data_fraction[grouped_data_fraction['moved']==1]
    grouped_data_fraction = grouped_data_fraction.drop('moved', axis=1)

    return grouped_data_fraction

def create_reg_variables(df, variable='PHONE_ID', ses=False, dd=False):
    df = df.assign(threshold=(df['date'] > pd.Timestamp('2024-02-02')).astype(int))
    df = df.assign(first_day=(df['date'] == pd.Timestamp('2024-02-03')).astype(int))
    df['time_delta'] = (df['date'] - pd.Timestamp('2024-02-03')).dt.days
    df['weekday'] = df['date'].dt.weekday
    
    if dd==True:
        if ses:
            df = pd.melt(df, id_vars=['quantile_pop_bins_percent_bachelor_home', 'threshold', 'time_delta', 'weekday', 'first_day'], 
                                value_vars=[f'{variable}_notaff',	f'{variable}_evac'], 
                                var_name='group', value_name='value')
        else:
            df = pd.melt(df, id_vars=['threshold', 'time_delta', 'weekday', 'first_day'], 
                                value_vars=[f'{variable}_notaff',	f'{variable}_evac'], 
                                var_name='group', value_name='value')
        df['treatment'] = df['group'].apply(lambda x: 1 if x == f'{variable}_evac' else 0)
    
    return df

def prepare_reg_dataset(df_notaff, df_evac, variable='PHONE_ID', ses=False):

    rdd_df = df_evac

    if ses:
        dd_df = df_notaff[df_notaff['affected']==0].drop('affected', axis=1).reset_index(drop=True)
        dd_df = dd_df.merge(df_evac, on=['date', 'quantile_pop_bins_percent_bachelor_home'], suffixes=('_notaff', '_evac'))
    else:
        dd_df = df_notaff[df_notaff['affected']==0].drop('affected', axis=1).reset_index(drop=True)
        dd_df = dd_df.merge(df_evac, on='date', suffixes=('_notaff', '_evac'))

    rdd_df = create_reg_variables(rdd_df, variable='PHONE_ID')
    dd_df = create_reg_variables(dd_df, variable=variable, ses=ses, dd=True)

    return rdd_df, dd_df


#################### Loading Data
start_time = time.time()

print('Loading data...')

homelocations_merged=pd.read_parquet('data_created/homelocations_merged_cleaned.parquet')
evacuated_people = helper.upload_list_txt('data_created/evacuated.txt')

result = helper.upload_if_exists('data_created/bootstrapped_merged_noses.parquet')
result_evac = helper.upload_if_exists('data_created/bootstrapped_evac_noses.parquet')
result_ses = helper.upload_if_exists('data_created/bootstrapped_merged_ses.parquet')
result_evac_ses = helper.upload_if_exists('data_created/bootstrapped_evac_ses.parquet')

## Set a colour palette: 
# Original colors
orange = '#E69F00'
blue = '#56B4E9'
green = '#009E73'
# Darker shades
darker_orange = '#A87000'
darker_blue = '#3A81A3'
darker_green = '#00674D'
# Lighter shades
lighter_orange = '#FFD080'
lighter_blue = '#7FCFFF'
lighter_green = '#33E6B2'

#################### Aggregated Analysis
print('Starting Analysis...')

homelocations_warned_frac = homelocations_merged.dropna(subset=['current_bts_id', 'home_bts_id']).reset_index(drop=True)

homelocations_warned_frac2 = homelocations_warned_frac[homelocations_warned_frac['PHONE_ID'].isin(evacuated_people)].dropna(subset=['current_bts_id', 'home_bts_id']).reset_index(drop=True)

grouped_data_fraction1 = process_data(homelocations_warned_frac) #notaffected
grouped_data_fraction2 = process_data(homelocations_warned_frac2) #evacuated
grouped_data_fraction1_ses = process_data(homelocations_warned_frac, ses=True) #notaffected
grouped_data_fraction2_ses = process_data(homelocations_warned_frac2, ses=True) #evacuated

folder_path = 'data_created/fraction'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
grouped_data_fraction1.to_csv('data_created/fraction/grouped_data_fraction1.csv') #notaffected
grouped_data_fraction2.to_csv('data_created/fraction/grouped_data_fraction2.csv') #evacuated
grouped_data_fraction1_ses.to_csv('data_created/fraction/grouped_data_fraction1_ses.csv') #notaffected
grouped_data_fraction2_ses.to_csv('data_created/fraction/grouped_data_fraction2_ses.csv') #evacuated

if all(dataset is None for dataset in [result, result_evac]):
    print('Bootstrapping...')
    result, result_evac = helper.bootstrap_fractions(homelocations_warned_frac, fraction=0.1, evacuated_people=evacuated_people, ses=False)
    
    result.to_parquet('data_created/bootstrapped_merged_noses.parquet')
    result_evac.to_parquet('data_created/bootstrapped_evac_noses.parquet')
else:
    print("Files already exist, no need for bootstrapping.")

if all(dataset is None for dataset in [result_ses, result_evac_ses]):
    result_ses, result_evac_ses = helper.bootstrap_fractions(homelocations_warned_frac, evacuated_people=evacuated_people, fraction=0.1, ses=True)
    
    result_ses.to_parquet('data_created/bootstrapped_merged_ses.parquet')
    result_evac_ses.to_parquet('data_created/bootstrapped_evac_ses.parquet')
else:
    print("Files already exist, no need for bootstrapping.")


#################### Regressions - Fraction Away
print('Starting Regressions - Fraction Away...')

rdd_df, dd_df = prepare_reg_dataset(grouped_data_fraction1, grouped_data_fraction2, variable='PHONE_ID')
rdd_df_ses, dd_df_ses = prepare_reg_dataset(grouped_data_fraction1_ses, grouped_data_fraction2_ses, variable='PHONE_ID', ses=True)

# Define RDiT models
rdd_model1 = smf.ols('PHONE_ID ~ weekday + time_delta * threshold', data=rdd_df).fit(cov_type='HAC',cov_kwds={'maxlags':1})
rdd_model2 = smf.ols('PHONE_ID ~ weekday + time_delta * threshold + I(time_delta**2)', data=rdd_df).fit(cov_type='HAC',cov_kwds={'maxlags':1})
rdd_model3 = smf.ols('PHONE_ID ~ weekday + time_delta * threshold + I(time_delta**2) + I(time_delta**3)', data=rdd_df).fit(cov_type='HAC',cov_kwds={'maxlags':1})

rdd_model4 = smf.ols('PHONE_ID ~ weekday + time_delta * threshold + quantile_pop_bins_percent_bachelor_home * threshold', data=rdd_df_ses).fit(cov_type='HAC',cov_kwds={'maxlags':1})
rdd_model5 = smf.ols('PHONE_ID ~ weekday + time_delta * threshold + quantile_pop_bins_percent_bachelor_home * threshold + I(time_delta**2)', data=rdd_df_ses).fit(cov_type='HAC',cov_kwds={'maxlags':1})
rdd_model6 = smf.ols('PHONE_ID ~ weekday + time_delta * threshold + quantile_pop_bins_percent_bachelor_home * threshold + I(time_delta**2) + I(time_delta**3)', data=rdd_df_ses).fit(cov_type='HAC',cov_kwds={'maxlags':1})

print(dd_df_ses.head())

# Define DiD models
did_model1 = smf.ols('value ~ weekday + time_delta + threshold * treatment', data=dd_df).fit(cov_type='HAC',cov_kwds={'maxlags':1})
did_model2 = smf.ols('value ~ weekday + time_delta + threshold * treatment + I(time_delta**2)', data=dd_df).fit(cov_type='HAC',cov_kwds={'maxlags':1})
did_model3 = smf.ols('value ~ weekday + time_delta + threshold * treatment + I(time_delta**2) + I(time_delta**3)', data=dd_df).fit(cov_type='HAC',cov_kwds={'maxlags':1})

did_model4 = smf.ols('value ~ weekday + time_delta + threshold * treatment + quantile_pop_bins_percent_bachelor_home * treatment + quantile_pop_bins_percent_bachelor_home * treatment * threshold', data=dd_df_ses).fit(cov_type='HAC',cov_kwds={'maxlags':1})
did_model5 = smf.ols('value ~ weekday + time_delta + threshold * treatment + quantile_pop_bins_percent_bachelor_home * treatment + quantile_pop_bins_percent_bachelor_home * threshold + quantile_pop_bins_percent_bachelor_home * treatment * threshold', data=dd_df_ses).fit(cov_type='HAC',cov_kwds={'maxlags':1})
did_model6 = smf.ols('value ~ weekday + time_delta + threshold * treatment + quantile_pop_bins_percent_bachelor_home * treatment + quantile_pop_bins_percent_bachelor_home * threshold + quantile_pop_bins_percent_bachelor_home * treatment * threshold + I(time_delta**2) + I(time_delta**3)', data=dd_df_ses).fit(cov_type='HAC',cov_kwds={'maxlags':1})


models_rdd = [rdd_model1, rdd_model2, rdd_model3, rdd_model4, rdd_model5, rdd_model6]
models_did = [did_model1, did_model2, did_model3, did_model4, did_model5, did_model6]
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']

summary_table_rdd = summary_col(models_rdd, stars=True, float_format='%0.3f',
                               model_names=model_names,
                               info_dict={'R-squared': lambda x: f"{x.rsquared:.3f}",
                                          'Adjusted R-squared': lambda x: f"{x.rsquared_adj:.3f}",
                                          'No. observations': lambda x: f"{int(x.nobs)}"})

summary_table_did = summary_col(models_did, stars=True, float_format='%0.3f',
                               model_names=model_names,
                               info_dict={'R-squared': lambda x: f"{x.rsquared:.3f}",
                                          'Adjusted R-squared': lambda x: f"{x.rsquared_adj:.3f}",
                                          'No. observations': lambda x: f"{int(x.nobs)}"})

print(summary_table_rdd)
print(summary_table_did)

folder_path = 'data_created/regression_results'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

summary_table_rdd.tables[0].to_csv('data_created/regression_results/rdd_table1.csv')
summary_table_did.tables[0].to_csv('data_created/regression_results/did_table1.csv')


rdd_df = rdd_df.assign(predictions=rdd_model1.fittedvalues)
dd_df = dd_df.assign(predictions=did_model1.fittedvalues)

rdd_df_ses = rdd_df_ses.assign(predictions=rdd_model4.fittedvalues)
dd_df_ses = dd_df_ses.assign(predictions=did_model4.fittedvalues)

rdd_df.to_csv('data_created/fraction/rdd_df.csv')
dd_df.to_csv('data_created/fraction/dd_df.csv')
rdd_df_ses.to_csv('data_created/fraction/rdd_df_ses.csv')
dd_df_ses.to_csv('data_created/fraction/dd_df_ses.csv')

#################### Regressions - Distances (Mean and Median)
print('Starting Regressions - Mean and Median...')

subset = homelocations_merged[~(homelocations_merged['home_bts_id'].isna())&(homelocations_merged['distance_km'] > 0)]
subset_evac = homelocations_merged[homelocations_merged['PHONE_ID'].isin(evacuated_people)&(homelocations_merged['distance_km'] > 0)]

subset_mean = subset.groupby(['date', 'affected'])['distance_km'].mean().reset_index()
subset_evac_mean = subset_evac.groupby(['date'])['distance_km'].mean().reset_index()

subset_ses_mean = subset.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'])['distance_km'].mean().reset_index()
subset_evac_ses_mean = subset_evac.groupby(['date', 'quantile_pop_bins_percent_bachelor_home'])['distance_km'].mean().reset_index()

subset_median = subset.groupby(['date', 'affected'])['distance_km'].median().reset_index()
subset_evac_median = subset_evac.groupby(['date'])['distance_km'].median().reset_index()

subset_ses_median = subset.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'])['distance_km'].median().reset_index()
subset_evac_ses_median = subset_evac.groupby(['date', 'quantile_pop_bins_percent_bachelor_home'])['distance_km'].median().reset_index()

rdd_df_mean, dd_df_mean = prepare_reg_dataset(subset_mean, subset_evac_mean, variable='distance_km')
rdd_df_median, dd_df_median = prepare_reg_dataset(subset_median, subset_evac_median, variable='distance_km')

rdd_df_ses_mean, dd_df_ses_mean = prepare_reg_dataset(subset_ses_mean, subset_evac_ses_mean, variable='distance_km', ses=True)
rdd_df_ses_median, dd_df_ses_median = prepare_reg_dataset(subset_ses_median, subset_evac_ses_median, variable='distance_km', ses=True)

for type in ['mean','median']:
    
    rdd_df_ = globals()[f'rdd_df_{type}']
    dd_df_ = globals()[f'dd_df_{type}']
    rdd_df_ses_ = globals()[f'rdd_df_ses_{type}']
    dd_df_ses_ = globals()[f'dd_df_ses_{type}']

    # Define RDiT models
    rdd_model1 = smf.ols('distance_km ~ C(weekday) + time_delta * threshold', data=rdd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    rdd_model2 = smf.ols('distance_km ~ C(weekday) + time_delta * threshold + I(time_delta**2)', data=rdd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    rdd_model3 = smf.ols('distance_km ~ C(weekday) + time_delta * threshold + I(time_delta**2) + I(time_delta**3)', data=rdd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    
    rdd_model4 = smf.ols('distance_km ~ C(weekday) + time_delta * threshold + quantile_pop_bins_percent_bachelor_home * threshold', data=rdd_df_ses_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    rdd_model5 = smf.ols('distance_km ~ C(weekday) + time_delta * threshold + quantile_pop_bins_percent_bachelor_home * threshold + I(time_delta**2)', data=rdd_df_ses_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    rdd_model6 = smf.ols('distance_km ~ C(weekday) + time_delta * threshold + quantile_pop_bins_percent_bachelor_home * threshold + I(time_delta**2) + I(time_delta**3)', data=rdd_df_ses_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    
    print(dd_df_ses.head())
    
    # Define DiD models
    did_model1 = smf.ols('value ~ C(weekday) + time_delta + threshold * treatment', data=dd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    did_model2 = smf.ols('value ~ C(weekday) + time_delta * treatment + threshold * treatment', data=dd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    did_model3 = smf.ols('value ~ C(weekday) + time_delta * treatment + threshold * treatment + I(time_delta**2) + I(time_delta**3)', data=dd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    
    did_model4 = smf.ols('value ~ C(weekday) + time_delta + threshold * treatment + quantile_pop_bins_percent_bachelor_home * treatment + quantile_pop_bins_percent_bachelor_home * treatment * threshold', data=dd_df_ses_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    did_model5 = smf.ols('value ~ C(weekday) + time_delta * treatment + threshold * treatment + quantile_pop_bins_percent_bachelor_home * treatment + quantile_pop_bins_percent_bachelor_home * threshold + quantile_pop_bins_percent_bachelor_home * treatment * threshold', data=dd_df_ses_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    did_model6 = smf.ols('value ~ C(weekday) + time_delta * treatment + threshold * treatment + quantile_pop_bins_percent_bachelor_home * treatment + quantile_pop_bins_percent_bachelor_home * threshold + quantile_pop_bins_percent_bachelor_home * treatment * threshold + I(time_delta**2) + I(time_delta**3)', data=dd_df_ses_).fit(cov_type='HAC',cov_kwds={'maxlags':1})


    models_rdd = [rdd_model1, rdd_model2, rdd_model3, rdd_model4, rdd_model5, rdd_model6]
    models_did = [did_model1, did_model2, did_model3, did_model4, did_model5, did_model6]
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']
    
    summary_table_rdd = summary_col(models_rdd, stars=True, float_format='%0.3f',
                                   model_names=model_names,
                                   info_dict={'R-squared': lambda x: f"{x.rsquared:.3f}",
                                              'Adjusted R-squared': lambda x: f"{x.rsquared_adj:.3f}",
                                              'No. observations': lambda x: f"{int(x.nobs)}"})
    
    summary_table_did = summary_col(models_did, stars=True, float_format='%0.3f',
                                   model_names=model_names,
                                   info_dict={'R-squared': lambda x: f"{x.rsquared:.3f}",
                                              'Adjusted R-squared': lambda x: f"{x.rsquared_adj:.3f}",
                                              'No. observations': lambda x: f"{int(x.nobs)}"})
    print(f'====={type}=====')
    print(summary_table_rdd)
    print(summary_table_did)

    summary_table_rdd.tables[0].to_csv(f'data_created/regression_results/rdd_table1_{type}.csv')
    summary_table_did.tables[0].to_csv(f'data_created/regression_results/did_table1_{type}.csv')

    globals()[f'rdd_df_{type}'] = rdd_df_.assign(predictions=rdd_model1.fittedvalues)
    globals()[f'dd_df_{type}'] = dd_df_.assign(predictions=did_model1.fittedvalues)
    globals()[f'rdd_df_ses_{type}'] = rdd_df_ses_.assign(predictions=rdd_model4.fittedvalues)
    globals()[f'dd_df_ses_{type}'] = dd_df_ses_.assign(predictions=did_model4.fittedvalues)

rdd_df_mean.to_csv('data_created/fraction/rdd_df_mean.csv')
dd_df_mean.to_csv('data_created/fraction/dd_df_mean.csv')
rdd_df_ses_mean.to_csv('data_created/fraction/rdd_df_ses_mean.csv')
dd_df_ses_mean.to_csv('data_created/fraction/dd_df_ses_mean.csv')
rdd_df_median.to_csv('data_created/fraction/rdd_df_median.csv')
dd_df_median.to_csv('data_created/fraction/dd_df_median.csv')
rdd_df_ses_median.to_csv('data_created/fraction/rdd_df_ses_median.csv')
dd_df_ses_median.to_csv('data_created/fraction/dd_df_ses_median.csv')


print("--- %s seconds ---" % (time.time() - start_time))