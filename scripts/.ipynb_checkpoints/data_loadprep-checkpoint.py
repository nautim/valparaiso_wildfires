### export PYTHONPATH="${PYTHONPATH}:" if want to run from chile_emergent
### python3 scripts/data_loadprep.py

#################### Dependencies

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr

import datetime

import os
import time

from utils import helper

#################### Loading Data
start_time = time.time()

print('Loading data...')

homelocations = helper.upload_homelocations_data('data/homelocs_extended/')
homelocations = homelocations[homelocations['date']>'2024-01-18'].reset_index(drop=True)

distances5 = pd.read_csv('data/distances5ta.csv.tar.gz')

geo_antennas = gpd.read_file('data/affected_btsid/affected_btsid.shp')
geo_antennas.crs = 'EPSG:4674'

zonas_geo = gpd.read_file('data/zonas/ZONA_C17.shp')
zonas_geo.crs = 'EPSG:4674'

zonas_se = pd.read_csv('data/auxvars_zone_census.csv.tar.gz')

zonas_geo.GEOCODIGO = zonas_geo.GEOCODIGO.astype(str)
zonas_se.ZONA = zonas_se.ZONA.astype(str)

zonas_geo = zonas_geo.merge(zonas_se, left_on='GEOCODIGO', right_on='ZONA')

geo_antennas = geo_antennas.sjoin(zonas_geo, how='left', predicate='intersects')

warned_towers = pd.read_csv('data/warned_towers.csv')
warned_towers['date'] = pd.to_datetime(warned_towers['date'])
warned_towers = warned_towers.merge(geo_antennas[['bts_id', 'geometry']], on='bts_id', how='left')
warned_towers = gpd.GeoDataFrame(warned_towers, geometry='geometry', crs='EPSG:4674')

chile_border_adm0 = gpd.read_file('../data_all/chile_borders/chl_admbnda_adm0_bcn_20211008.shp')
chile_border_adm3 = gpd.read_file('../data_all/chile_borders/chl_admbnda_adm3_bcn_20211008.shp')
incendio_gpd = gpd.read_file('data/ED_AreasAfectadasIncendio_Valparaiso/ED_AreasAfectadasIncendio_Valparaiso.shp')

#################### Identify Home Locations

print('Identifying Home Locations...')

most_common_bts_id_1 = helper.get_homelocated_unique_phoneIDs(homelocations, '2024-01-26', '2024-02-01', 5)
most_common_bts_id_2 = helper.get_homelocated_unique_phoneIDs(homelocations, '2024-01-19', '2024-02-01', 6)

homelocations_past = helper.upload_homelocations_data('data/homelocs_past/')
most_common_bts_id_old = helper.get_homelocated_unique_phoneIDs(homelocations_past, '2023-11-11', '2023-11-17', 5)

homelocations_1 = helper.merge_home_locations(homelocations, most_common_bts_id_1)
print("==================")
homelocations_2 = helper.merge_home_locations(homelocations, most_common_bts_id_2)
print("==================")
homelocations_past_1 = helper.merge_home_locations(homelocations_past, most_common_bts_id_old)

#################### Merge with Socioeconomic Groups

print('Merging with SES...')

geo_antennas_se, linear_range, quantile_range, quantile_pop_range = helper.create_se_bins(geo_antennas, 'percent_bachelor', 'total_people')

folder_path = 'data_created'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
geo_antennas_se[['bts_id', 'linear_bins_percent_bachelor', 'quantile_pop_bins_percent_bachelor']].to_csv('data_created/tower_ses.csv')

columns_to_merge = {
    column_name: geo_antennas_se[column_name] 
    for column_name in geo_antennas_se.columns 
    if column_name.startswith('linear_bins') 
    # or column_name.startswith('quantile_bins') 
    or column_name.startswith('quantile_pop_bins')
    or column_name in ['bts_id', 'ZONA']
}
columns_to_merge = pd.DataFrame(columns_to_merge)

homelocations_1 = helper.merge_ses_towers(homelocations_1, columns_to_merge)
print("==================")
homelocations_2 = helper.merge_ses_towers(homelocations_2, columns_to_merge)
print("==================")
homelocations_past_1 = helper.merge_ses_towers(homelocations_past_1, columns_to_merge)

#################### Plot Preliminary Correlations

print('Preliminary Correlation Plots...')

helper.plot_correlation_population(homelocations_1, zonas_geo, save_folder='visuals_created/', title_add='shorter')
helper.plot_correlation_population(homelocations_2, zonas_geo, save_folder='visuals_created/', title_add='longer')
helper.plot_correlation_population(homelocations_past_1, zonas_geo, save_folder='visuals_created/', title_add='past')

#################### Match the biggest set from January with the one in November, keep only those who appear in both

check_homelocs = homelocations_2[['PHONE_ID', 'home_bts_id']].drop_duplicates().dropna().reset_index(drop=True)
check_homelocs = check_homelocs.merge(homelocations_past_1, on='PHONE_ID', how='left').dropna()
check_homelocs=check_homelocs[check_homelocs['home_bts_id_x'] == check_homelocs['home_bts_id_y']].reset_index(drop=True)

helper.plot_correlation_population(check_homelocs, zonas_geo, save_folder='visuals_created/', title_add='longerpast')


#################### Define Affected and Not-Affected Populations

print('Selecting Affected, Evacuated, and Not-Affected Populations...')

warned_towers_list = warned_towers['bts_id'].unique()

affected_people = (homelocations.loc[((homelocations['date'].isin([datetime.datetime(2024, 2, 1), datetime.datetime(2024, 2, 2)])) & 
                                     (homelocations['bts_id'].isin(warned_towers_list)))]['PHONE_ID'].unique())


feb_1st = set(homelocations.loc[(homelocations['date'] == datetime.datetime(2024, 2, 1)) & ~(homelocations['bts_id'].isin(warned_towers_list)), 'PHONE_ID'].unique())
feb_2nd = set(homelocations.loc[(homelocations['date'] == datetime.datetime(2024, 2, 2)) & ~(homelocations['bts_id'].isin(warned_towers_list)), 'PHONE_ID'].unique())
non_affected_people = list(feb_1st.intersection(feb_2nd))

print(len(affected_people))
print(len(non_affected_people))
print(len(np.intersect1d(affected_people, non_affected_people)))

homelocations_warned = helper.select_group_people(homelocations, distances5, affected_people)
homelocations_counterfactual = helper.select_group_people(homelocations, distances5, non_affected_people)

check_homelocs=check_homelocs.rename(columns={'home_bts_id_x':'home_bts_id'})

homelocations_warned = helper.merge_home_locations(homelocations_warned, check_homelocs)
homelocations_counterfactual = helper.merge_home_locations(homelocations_counterfactual, check_homelocs)

evacuated_people = (homelocations_warned.loc[
                     (homelocations_warned['date'].isin(['2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04']))&
                     (homelocations_warned['current_bts_id'] != homelocations_warned['home_bts_id'])&
                     (homelocations_warned['home_bts_id'].isin(warned_towers_list))])['PHONE_ID'].unique()

homelocations_warned['affected'] = 1
homelocations_counterfactual['affected'] = 0
homelocations_merged = pd.concat([homelocations_warned, homelocations_counterfactual]).reset_index(drop=True)

np.random.seed(1234)
homelocations_merged['affected_shuffled'] = homelocations_merged['affected'].sample(frac=1).reset_index(drop=True)

homelocations_merged = homelocations_merged.merge(
    columns_to_merge, 
    left_on='home_bts_id', right_on='bts_id', how='left').drop('bts_id', axis=1)

homelocations_merged = homelocations_merged.merge(
    columns_to_merge, 
    left_on='current_bts_id', right_on='bts_id',
    suffixes=('_home', '_current'), how='left').drop('bts_id', axis=1)

print("Distribution of SES:")
print((homelocations_merged.dropna(subset=['home_bts_id'])[['PHONE_ID', 'affected', 'quantile_pop_bins_percent_bachelor_home']]
                     .astype({'quantile_pop_bins_percent_bachelor_home': str})
                     .fillna('NA')
                     .drop_duplicates()
                     .groupby('affected')['quantile_pop_bins_percent_bachelor_home'].value_counts()))

print("Distribution of SES among evacuated people only:")
print((homelocations_merged.dropna(subset=['home_bts_id'])[['PHONE_ID', 'affected', 'quantile_pop_bins_percent_bachelor_home']]
                     .astype({'quantile_pop_bins_percent_bachelor_home': str})
                     .fillna('NA')
                     .loc[homelocations_merged['PHONE_ID'].isin(evacuated_people)]
                     .drop_duplicates()
                     .quantile_pop_bins_percent_bachelor_home.value_counts()))

homelocations_merged['moved'] = (homelocations_merged['current_bts_id'] != homelocations_merged['home_bts_id']).apply(int)
    
homelocations_merged.to_parquet('data_created/homelocations_merged.parquet')

helper.plot_correlation_population(homelocations_merged, zonas_geo, save_folder='visuals_created/', title_add='merged')

affected_people = homelocations_warned['PHONE_ID'].unique()
notaffected_people = homelocations_counterfactual['PHONE_ID'].unique()

np.savetxt('data_created/evacuated.txt', evacuated_people, fmt='%s')
np.savetxt('data_created/affected.txt', affected_people, fmt='%s')
np.savetxt('data_created/notaffected.txt', notaffected_people, fmt='%s')

#################### Plot Correlation in Different SES

folder_path = 'visuals_created'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

merged_dataset = (zonas_geo[['ZONA', 'total_people']].merge((check_homelocs[['PHONE_ID', 'ZONA_home', 'quantile_pop_bins_percent_bachelor_home']]
                 .drop_duplicates()
                 .groupby('ZONA_home').agg({'PHONE_ID':'size',
                                            'quantile_pop_bins_percent_bachelor_home':'first'})
                 .reset_index()), how='right', left_on='ZONA', right_on='ZONA_home'))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

sessions = ['Low', 'Medium', 'High']

for i, ses in enumerate(sessions):

    sub_merged_dataset=merged_dataset[merged_dataset['quantile_pop_bins_percent_bachelor_home']==ses]
    
    sns.regplot(ax=ax[i], data=sub_merged_dataset, x='total_people', y='PHONE_ID',
                line_kws={'color':'green', 'linewidth': 2})

    ax[i].set_title(f'SE status: {ses}', fontsize=12)
    ax[i].set_xlabel('Census: Total Number of People', fontsize=12)
    ax[i].set_ylabel('Telefonica: Total Number of \n Unique Phone IDs', fontsize=12)

    pearson_corr, pearson_p = pearsonr(sub_merged_dataset['total_people'], sub_merged_dataset['PHONE_ID'])
    spearman_corr, spearman_p = spearmanr(sub_merged_dataset['total_people'], sub_merged_dataset['PHONE_ID'])

    # Annotate correlation coefficients and p-values
    ax[i].annotate(f'Pearson r: {pearson_corr:.3f} (p={pearson_p:.2g})', xy=(0.05, 0.95), 
                   xycoords='axes fraction', fontsize=12, ha='left', va='top')
    ax[i].annotate(f'Spearman ρ: {spearman_corr:.3f} (p={spearman_p:.2g})', xy=(0.05, 0.88), 
                   xycoords='axes fraction', fontsize=12, ha='left', va='top')

plt.tight_layout()
plt.savefig('visuals_created/correlations_ses.pdf')
plt.show();

print("--- %s seconds ---" % (time.time() - start_time))

