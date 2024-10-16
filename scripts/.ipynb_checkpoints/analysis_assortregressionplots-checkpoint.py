### export PYTHONPATH="${PYTHONPATH}:" if want to run from chile_emergent
### python3 scripts/analysis_assortregressionplots.py 

#################### Dependencies

import pandas as pd

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

import datetime

import os
import time
from tqdm import tqdm

from utils import helper

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns

#################### Functions

def create_reg_variables(df, variable='PHONE_ID', ses=False, dd=False):
    df = df.assign(threshold=(df['dates'] > pd.Timestamp('2024-02-02')).astype(int))
    df = df.assign(first_day=(df['dates'] == pd.Timestamp('2024-02-03')).astype(int))
    df['time_delta'] = (df['dates'] - pd.Timestamp('2024-02-03')).dt.days
    df['weekday'] = df['dates'].dt.weekday
    
    if dd==True:
        df = pd.melt(df, id_vars=['threshold', 'time_delta', 'weekday', 'first_day'], 
                                value_vars=['assortativity_evacuated', 'assortativity_control'], 
                                var_name='group', value_name='value')
        df['treatment'] = df['group'].apply(lambda x: 1 if x == 'assortativity_evacuated' else 0)
    
    return df


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

#################### Loading Data
start_time = time.time()

print('Loading data...')

assortativity_df_plot=pd.read_csv('data_created/assortativity_all.csv')
assortativity_moved_df_plot=pd.read_csv('data_created/assortativity_moved.csv')

assortativity_moved_df_plot['dates']=pd.to_datetime(assortativity_moved_df_plot['dates'])
assortativity_df_plot['dates']=pd.to_datetime(assortativity_df_plot['dates'])

#################### Regressions - Assortativity (All and Moved)
print('Starting Regressions for Assortativity...')

rdd_df_all = create_reg_variables(assortativity_df_plot, variable='assortativity_evacuated', dd=False)
dd_df_all = create_reg_variables(assortativity_df_plot, variable='assortativity_evacuated', dd=True)

rdd_df_moved = create_reg_variables(assortativity_moved_df_plot, variable='assortativity_evacuated', dd=False)
dd_df_moved = create_reg_variables(assortativity_moved_df_plot, variable='assortativity_evacuated', dd=True)


for type in ['all','moved']:
    
    rdd_df_ = globals()[f'rdd_df_{type}']
    dd_df_ = globals()[f'dd_df_{type}']

    # Define RDiT models
    rdd_model1 = smf.ols('assortativity_evacuated ~ weekday + first_day + time_delta * threshold', data=rdd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    rdd_model2 = smf.ols('assortativity_evacuated ~ weekday + first_day + time_delta * threshold + I(time_delta**2)', data=rdd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    rdd_model3 = smf.ols('assortativity_evacuated ~ weekday + first_day + time_delta * threshold + I(time_delta**2) + I(time_delta**3)', data=rdd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    # Define DiD models
    did_model1 = smf.ols('value ~ weekday + first_day * treatment + time_delta + threshold * treatment', data=dd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    did_model2 = smf.ols('value ~ weekday + first_day * treatment + time_delta * treatment + threshold * treatment', data=dd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    did_model3 = smf.ols('value ~ weekday + first_day * treatment + time_delta * treatment + threshold * treatment + I(time_delta**2) + I(time_delta**3)', data=dd_df_).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    

    models_rdd = [rdd_model1, rdd_model2, rdd_model3]
    models_did = [did_model1, did_model2, did_model3]
    model_names = ['Model 1', 'Model 2', 'Model 3']
    
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

    summary_table_rdd.tables[0].to_csv(f'data_created/regression_results/rdd_table_assortativity_{type}.csv')
    summary_table_did.tables[0].to_csv(f'data_created/regression_results/did_table_assortativity_{type}.csv')

    globals()[f'rdd_df_{type}'] = rdd_df_.assign(predictions=rdd_model1.fittedvalues)
    globals()[f'dd_df_{type}'] = dd_df_.assign(predictions=did_model1.fittedvalues)

#################### Plots

print('Plotting...')

plt.rcParams.update({'font.size': 13, 'font.style': 'normal', 'figure.facecolor':'white'})



######## Figure 1 ########

fig, axs = plt.subplot_mosaic([
    ['a', 'b',],
    ['a', 'c']], 
    #layout="constrained"
    figsize =(12, 5), height_ratios=[1,1],width_ratios=[2,1],
                            gridspec_kw={'wspace':0.3, 'hspace':0.5})

# Plotting the assortativity values over time
axs['a'].plot(assortativity_moved_df_plot['dates'], assortativity_moved_df_plot['assortativity_control'], marker='o', linestyle='-', color=orange, label='Not Affected')
axs['a'].plot(assortativity_moved_df_plot['dates'], assortativity_moved_df_plot['assortativity_evacuated'], marker='o', linestyle='-', color=green, label='Evacuated')
axs['a'].set_title('')
axs['a'].set_ylabel('Assortativity')

# Set x-axis to display only the dates
axs['a'].xaxis.set_major_locator(mdates.DayLocator())
axs['a'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

axs['a'].axvline(x=datetime.datetime(2024, 2, 2), color='red', linestyle=':', lw =1.5, alpha = 0.6, zorder=100)
axs['a'].text(x=datetime.datetime(2024, 2, 1), y=0.05, s='1-2 February',
        fontsize=13, rotation='vertical', verticalalignment='bottom', color='red', bbox=dict(facecolor='w', edgecolor='none', pad=0.0, alpha=0.6), zorder=100)


# Setting x-ticks and labels based on homelocations_warned dataframe
unique_dates = assortativity_df_plot.drop_duplicates(subset=['dates'])['dates'].unique()
axs['a'].set_xticks(unique_dates)
xticklabels = [date.strftime('%d \n%m \n%a ') if i % 2 == 0 else '' for i, date in enumerate(unique_dates)]
axs['a'].set_xticklabels(xticklabels, fontsize=8)

axs['a'].set_xlim(datetime.datetime(2024, 1, 19), datetime.datetime(2024, 2, 19))
axs['a'].set_ylim(0, 0.55)

axs['a'].fill_between(assortativity_moved_df_plot['dates'], assortativity_moved_df_plot['ci_lower_notaff'], assortativity_moved_df_plot['ci_upper_notaff'], alpha=0.3, color=orange)
axs['a'].fill_between(assortativity_moved_df_plot['dates'], assortativity_moved_df_plot['ci_lower_evac'], assortativity_moved_df_plot['ci_upper_evac'], alpha=0.3, color=green)

axs['a'].legend(frameon=False)

##### Top Right

rdd_df_moved.plot.line(x="time_delta", y="assortativity_evacuated", color=green, ax=axs['b'], legend=False)
rdd_df_moved.plot(x="time_delta", y="predictions", ax=axs['b'], color=green, linestyle='--', legend=False)
plt.axvline(0, color='red', linestyle='--', label='2-3 February')

axs['b'].set_title("")
axs['b'].set_ylabel('Assortativity')
axs['b'].set_xlabel('')

axs['b'].axvline(x=0, color='red', linestyle=':', lw =1.5, alpha = 0.6, zorder=100)
axs['b'].text(x=-3, y=0.05, s='2-3 February',
        fontsize=13, rotation='vertical', verticalalignment='bottom', color='red', bbox=dict(facecolor='w', edgecolor='none', pad=0.0, alpha=0.6), zorder=100)


##### Bottom Right

# Plot original data
sns.lineplot(ax=axs['c'], x='time_delta', y='value', hue='group', data=dd_df_moved, 
             hue_order=['assortativity_control', 'assortativity_evacuated'], palette=[orange, green], legend=False)

# Plot fitted predictions
sns.lineplot(ax=axs['c'], x='time_delta', y='predictions', hue='group', data=dd_df_moved,
             hue_order=['assortativity_control', 'assortativity_evacuated'], palette=[orange, green], linestyle='--', legend=False)

axs['c'].axvline(0, color='red', linestyle='--', label='2-3 February')


axs['c'].set_title("")
axs['c'].set_ylabel('Assortativity')
axs['c'].set_xlabel('Time Delta')

axs['c'].axvline(x=0, color='red', linestyle=':', lw =1.5, alpha = 0.6, zorder=100)
axs['c'].text(x=-3, y=0.05, s='2-3 February',
        fontsize=13, rotation='vertical', verticalalignment='bottom', color='red', bbox=dict(facecolor='w', edgecolor='none', pad=0.0, alpha=0.6), zorder=100)

##### Legend

c_differences   =[(Line2D([0],[0], marker='s', color=c, markerfacecolor=c,ls='',  markersize=8, label=mod)) for c,mod in zip([green, orange],
                                                                                                                           ['Likely Evacuated','Not Affected'])]
lines =[(Line2D([0],[0], marker='', color='grey', markerfacecolor='grey',ls=l,  lw=2.5, label=mod)) for l,mod in zip(['-', '--'],['Data','Fit'])]

fig.legend(handles=lines,bbox_to_anchor=(1.08,0.6), fontsize = 13, ncols= 2, frameon=False,columnspacing=0.8, handletextpad=0.1,labelspacing=0.1)
fig.legend(handles=c_differences,bbox_to_anchor=(1.09,0.55), fontsize = 13, title= '', frameon=False,columnspacing=0.5, handletextpad=0.1,labelspacing=0.1)

plots_l = ['a)', 'b)', 'c)']
for ia, ax in enumerate(fig.axes): 
    x,y = -0.05,1.06
    ax.text(x,y,  plots_l[ia],
        color = 'k',#'lightgreen',
        horizontalalignment='left',
        verticalalignment='bottom',
        weight = 'bold',
        transform=ax.transAxes,fontsize = 13)

plt.savefig('visuals_created/assortativity_moved.pdf', bbox_inches = 'tight')

##########################


######## Figure 2 ########

fig, axs = plt.subplot_mosaic([
    ['a', 'b',],
    ['a', 'c']], 
    #layout="constrained"
    figsize =(12, 5), height_ratios=[1,1],width_ratios=[2,1],
                            gridspec_kw={'wspace':0.3, 'hspace':0.5})

# Plotting the assortativity values over time
axs['a'].plot(assortativity_df_plot['dates'], assortativity_df_plot['assortativity_control'], marker='o', linestyle='-', color=orange, label='Not Affected')
axs['a'].plot(assortativity_df_plot['dates'], assortativity_df_plot['assortativity_evacuated'], marker='o', linestyle='-', color=green, label='Evacuated')
axs['a'].set_title('')
axs['a'].set_ylabel('Assortativity')

# Set x-axis to display only the dates
axs['a'].xaxis.set_major_locator(mdates.DayLocator())
axs['a'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

axs['a'].axvline(x=datetime.datetime(2024, 2, 2), color='red', linestyle=':', lw =1.5, alpha = 0.6, zorder=100)
axs['a'].text(x=datetime.datetime(2024, 2, 1), y=0.05, s='1-2 February',
        fontsize=13, rotation='vertical', verticalalignment='bottom', color='red', bbox=dict(facecolor='w', edgecolor='none', pad=0.0, alpha=0.6), zorder=100)


# Setting x-ticks and labels based on homelocations_warned dataframe
unique_dates = assortativity_df_plot.drop_duplicates(subset=['dates'])['dates'].unique()
axs['a'].set_xticks(unique_dates)
xticklabels = [date.strftime('%d \n%m \n%a ') if i % 2 == 0 else '' for i, date in enumerate(unique_dates)]
axs['a'].set_xticklabels(xticklabels, fontsize=8)

axs['a'].set_xlim(datetime.datetime(2024, 1, 19), datetime.datetime(2024, 2, 19))
axs['a'].set_ylim(0, 1)

axs['a'].fill_between(assortativity_df_plot['dates'], assortativity_df_plot['ci_lower_notaff'], assortativity_df_plot['ci_upper_notaff'], alpha=0.3, color=orange)
axs['a'].fill_between(assortativity_df_plot['dates'], assortativity_df_plot['ci_lower_evac'], assortativity_df_plot['ci_upper_evac'], alpha=0.3, color=green)

axs['a'].legend(frameon=False)

##### Top Right

rdd_df_all.plot.line(x="time_delta", y="assortativity_evacuated", color=green, ax=axs['b'], legend=False)
rdd_df_all.plot(x="time_delta", y="predictions", ax=axs['b'], color=green, linestyle='--', legend=False)
plt.axvline(0, color='red', linestyle='--', label='2-3 February')

axs['b'].set_title("")
axs['b'].set_ylabel('Assortativity')
axs['b'].set_xlabel('')

axs['b'].axvline(x=0, color='red', linestyle=':', lw =1.5, alpha = 0.6, zorder=100)
axs['b'].text(x=-3, y=0.5, s='2-3 February',
        fontsize=13, rotation='vertical', verticalalignment='bottom', color='red', bbox=dict(facecolor='w', edgecolor='none', pad=0.0, alpha=0.6), zorder=100)


##### Bottom Right

# Plot original data
sns.lineplot(ax=axs['c'], x='time_delta', y='value', hue='group', data=dd_df_all, 
             hue_order=['assortativity_control', 'assortativity_evacuated'], palette=[orange, green], legend=False)

# Plot fitted predictions
sns.lineplot(ax=axs['c'], x='time_delta', y='predictions', hue='group', data=dd_df_all,
             hue_order=['assortativity_control', 'assortativity_evacuated'], palette=[orange, green], linestyle='--', legend=False)

axs['c'].axvline(0, color='red', linestyle='--', label='2-3 February')


axs['c'].set_title("")
axs['c'].set_ylabel('Assortativity')
axs['c'].set_xlabel('Time Delta')

axs['c'].axvline(x=0, color='red', linestyle=':', lw =1.5, alpha = 0.6, zorder=100)
axs['c'].text(x=-3, y=0.5, s='2-3 February',
        fontsize=13, rotation='vertical', verticalalignment='bottom', color='red', bbox=dict(facecolor='w', edgecolor='none', pad=0.0, alpha=0.6), zorder=100)

##### Legend

c_differences   =[(Line2D([0],[0], marker='s', color=c, markerfacecolor=c,ls='',  markersize=8, label=mod)) for c,mod in zip([green, orange],
                                                                                                                           ['Likely Evacuated','Not Affected'])]
lines =[(Line2D([0],[0], marker='', color='grey', markerfacecolor='grey',ls=l,  lw=2.5, label=mod)) for l,mod in zip(['-', '--'],['Data','Fit'])]

fig.legend(handles=lines,bbox_to_anchor=(1.08,0.6), fontsize = 13, ncols= 2, frameon=False,columnspacing=0.8, handletextpad=0.1,labelspacing=0.1)
fig.legend(handles=c_differences,bbox_to_anchor=(1.09,0.55), fontsize = 13, title= '', frameon=False,columnspacing=0.5, handletextpad=0.1,labelspacing=0.1)

plots_l = ['a)', 'b)', 'c)']
for ia, ax in enumerate(fig.axes): 
    x,y = -0.05,1.06
    ax.text(x,y,  plots_l[ia],
        color = 'k',#'lightgreen',
        horizontalalignment='left',
        verticalalignment='bottom',
        weight = 'bold',
        transform=ax.transAxes,fontsize = 13)

plt.savefig('visuals_created/assortativity_all.pdf', bbox_inches = 'tight')

##########################




print("--- %s seconds ---" % (time.time() - start_time))








