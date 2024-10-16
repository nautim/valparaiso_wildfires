#########################
# Dependencies
#########################

import pandas as pd
import logging
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from tqdm import tqdm


#########################
# Functions
#########################

def upload_homelocations_data(folder_path):

    # uploading homelocations data
    df = pd.DataFrame()
    
    for file in os.listdir(folder_path):
        if file.endswith('.parquet'):
            df_current = pd.read_parquet(folder_path+file)
            df = pd.concat([df_current, df], ignore_index=True)
    
    df = df.sort_values(['PHONE_ID', 'date']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    return df

def upload_list_txt(file_path):
    list = []
    with open(file_path, 'r') as file:
        for line in file:
            list.append(line.strip())
    return list

def upload_if_exists(file_path):
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        print("File does not exist.")
        return None

def get_homelocated_unique_phoneIDs(data, start_date, end_date, count, save_folder=None):

    """
    Filters data to find PHONE_IDs with the most common BTS ID within a date range and saves the results if specified.

    Parameters:
    data (pd.DataFrame): Input data containing 'date', 'PHONE_ID', and 'bts_id' columns.
    start_date (pd.datetime): Start date for filtering data (inclusive).
    end_date (pd.datetime): End date for filtering data (inclusive).
    count (int): Minimum count threshold for the most common BTS ID.
    save_folder (str): Directory to save the result. If empty, the result is not saved.

    Returns:
    pd.DataFrame: DataFrame with PHONE_IDs and their most common BTS ID where the count is above the specified threshold.
    """
    
    logging.info("Processing data, this might take some time...")

    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data should be a pandas DataFrame.")
    required_columns = {'date', 'PHONE_ID', 'bts_id'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"data should contain the following columns: {required_columns}")

    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise ValueError("start_date and end_date should be strings in 'YYYY-MM-DD' format.")
    
    if not isinstance(count, int) or count < 0:
        raise ValueError("count should be a non-negative integer.")
    
    if save_folder is not None and not isinstance(save_folder, str):
        raise ValueError("save_folder should be a string representing the directory path if provided.")
    
    # Filter data within the date range
    filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Calculate the most common current BTS ID for each PHONE_ID
    most_common_bts_id = filtered_data.groupby('PHONE_ID')['bts_id'].agg(lambda x: (x.mode()[0], x.value_counts().max()))
    most_common_bts_id = most_common_bts_id.reset_index()
    
    # Separate the tuple into two columns
    most_common_bts_id[['home_bts_id', 'count']] = most_common_bts_id['bts_id'].apply(pd.Series)
    most_common_bts_id = most_common_bts_id.drop(columns=['bts_id'])

    # Filter according to the predefined count number
    most_common_bts_id = most_common_bts_id[most_common_bts_id['count'] >= count].reset_index(drop=True)

    # Save the result if save_folder is provided
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_path = os.path.join(save_folder, f'homelocated_phone_ids_{start_date}_to_{end_date}.csv')
        most_common_bts_id.to_csv(file_path, index=False)
        logging.info(f"Results saved to {file_path}")

    return most_common_bts_id


def create_se_bins(df, column_bin, column_population):
    
    # labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    labels = ['Low', 'Medium', 'High']

    df = df.dropna(subset=[column_bin]).reset_index(drop=True) # IMPORTANT - if you want to save NAs, need to change them beforehand
    
    df[f'linear_bins_{column_bin}'], linear_range = pd.cut(df[column_bin], bins=3, labels=labels, retbins=True) 
    df[f'quantile_bins_{column_bin}'], quantile_range = pd.qcut(df[column_bin], q=3, labels=labels, retbins=True)

    synthetic_population = (df[[column_bin, column_population]].drop_duplicates()
                                                               .sort_values(by=column_population)
                                                               .reset_index(drop=True))
    synthetic_population = np.repeat(synthetic_population[column_bin], synthetic_population[column_population]).reset_index(drop=True)
    _ , quantile_pop_range = pd.qcut(synthetic_population, q=3, labels=labels, retbins=True)

    df[f'quantile_pop_bins_{column_bin}'] = pd.cut(df[column_bin], bins=quantile_pop_range, labels=labels, include_lowest=True)

    return df, linear_range, quantile_range, quantile_pop_range
    

def merge_home_locations(data, home_tower):

    """
    Merges the home tower information with the main data based on PHONE_ID.

    Parameters:
    data (pd.DataFrame): Main data containing 'PHONE_ID' column.
    home_tower (pd.DataFrame): Data containing 'PHONE_ID' and 'home_bts_id' columns.

    Returns:
    pd.DataFrame: Merged DataFrame with home tower information.
    """

    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data should be a pandas DataFrame.")
    if not isinstance(home_tower, pd.DataFrame):
        raise ValueError("home_tower should be a pandas DataFrame.")
    
    if 'PHONE_ID' not in data.columns:
        raise ValueError("data DataFrame must contain 'PHONE_ID' column.")
    if 'PHONE_ID' not in home_tower.columns or 'home_bts_id' not in home_tower.columns:
        raise ValueError("home_tower DataFrame must contain 'PHONE_ID' and 'home_bts_id' columns.")


    home_tower = home_tower[['PHONE_ID', 'home_bts_id']].drop_duplicates().reset_index(drop=True)
    
    df = data.merge(home_tower[['PHONE_ID', 'home_bts_id']], on='PHONE_ID', how='left')
    
    # Print the number of unique PHONE_IDs
    unique_phone_ids = len(df['PHONE_ID'].unique())
    unique_phone_ids_with_home = len(df[~df['home_bts_id'].isna()]['PHONE_ID'].unique())
    print(f"Number of unique PHONE_IDs: {unique_phone_ids}")
    print(f"Number of unique PHONE_IDs with home location: {unique_phone_ids_with_home}")

    return df


def merge_ses_towers(data, data_ses):

    data = data.rename(columns={'bts_id':'current_bts_id'})
    data = data.merge(data_ses, left_on='home_bts_id', right_on='bts_id', how='left')
    data = data.merge(data_ses, left_on='current_bts_id', right_on='bts_id', suffixes=('_home', '_current'), how='left')

    unique_phone_ids_with_home_ses = len(data[~(data['home_bts_id'].isna())&~(data['linear_bins_percent_bachelor_current'].isna())]['PHONE_ID'].unique())
    print(f"Number of unique PHONE_IDs with home location and home SES: {unique_phone_ids_with_home_ses}")
    
    return data


def select_group_people(data, distances, phoneid_list):

    # homelocations - warned
    df = data[data['PHONE_ID'].isin(phoneid_list)].reset_index(drop=True)
    
    # Make it complete
    df_complete = df[['PHONE_ID', 'date', 'bts_id']].pivot(index='date', columns=['PHONE_ID'], values='bts_id')
    df_complete=df_complete.stack(dropna=False).reset_index().rename(columns={0:'bts_id'})
    df = df_complete.merge(df, on=['PHONE_ID', 'date', 'bts_id'], how='left')
    
    # Sort the dataframe by 'PHONE_ID' and 'date' columns
    df = df.sort_values(by=['PHONE_ID', 'date'])
    
    # Shift the 'bts_id' column by one day to get the previous day's bts_id
    df['previous_bts_id'] = df.groupby('PHONE_ID')['bts_id'].shift(1)
    df = df.rename(columns={'bts_id': 'current_bts_id'}).reset_index(drop=True)
    
    df = df.merge(distances, left_on=['current_bts_id', 'previous_bts_id'], right_on=['bts_id1', 'bts_id2'], how='left')
    df.loc[df['current_bts_id'] == df['previous_bts_id'], 'distance_km'] = 0.0
    df = df.drop(columns=['bts_id1', 'bts_id2'], axis=1)
    
    # Extract weekday name from the 'date' column
    df['weekday'] = df['date'].dt.strftime('%A')

    return df


def plot_correlation_population(data, data_zonas, save_folder=None, title_add=None):
    
    merged_dataset = (data_zonas[['ZONA', 'total_people']].merge((data[['PHONE_ID', 'ZONA_home']]
                     .drop_duplicates()
                     .groupby('ZONA_home')['PHONE_ID']
                     .size()
                     .reset_index()), how='right', left_on='ZONA', right_on='ZONA_home'))

    pearson_corr, pearson_p = pearsonr(merged_dataset['total_people'], merged_dataset['PHONE_ID'])
    spearman_corr, spearman_p = spearmanr(merged_dataset['total_people'], merged_dataset['PHONE_ID'])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.regplot(ax=ax, x=merged_dataset['total_people'], y=merged_dataset['PHONE_ID'],
               line_kws={'color':'green'})
    
    ax.set_xlabel('Census: Number of People', fontsize=14)
    ax.set_ylabel('Telefonica: Unique Individuals \n (with Home Location)', fontsize=14)

    # Annotate correlation coefficients and p-values
    ax.annotate(f'Pearson r: {pearson_corr:.3f} (p={pearson_p:.2g})', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    ax.annotate(f'Spearman ρ: {spearman_corr:.3f} (p={spearman_p:.2g})', xy=(0.05, 0.88), xycoords='axes fraction', fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    plt.tight_layout()

    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_path = os.path.join(save_folder, f'correlation_{title_add}.pdf')
        plt.savefig(file_path)
    
    plt.show();
    

def calculate_mover_fraction(df, ses=False):
    if ses == False:
        total_residents_per_date_ = df.groupby(['date', 'affected'], observed=False)['PHONE_ID'].count()
        grouped_data_fraction_ = df.groupby(['date', 'affected', 'moved'], observed=False)['PHONE_ID'].count() / total_residents_per_date_
        
    else:
        total_residents_per_date_ = df.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'], observed=False)['PHONE_ID'].count()
        grouped_data_fraction_ = df.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home', 'moved'], observed=False)['PHONE_ID'].count() / total_residents_per_date_

    grouped_data_fraction_ = grouped_data_fraction_.reset_index()
    mover_fraction = grouped_data_fraction_[grouped_data_fraction_['moved'] == 1]

    return mover_fraction.drop('moved', axis=1)


def calculate_mean_ci(df, ses=False):
    if ses == False:
        mean_fraction = df.groupby(['date', 'affected']).mean().reset_index()
        ci_lower = df.groupby(['date', 'affected']).quantile(0.025).reset_index()
        ci_upper = df.groupby(['date', 'affected']).quantile(0.975).reset_index()
    
        result = mean_fraction.merge(ci_lower, on=['date', 'affected'], suffixes=('', '_lower'))
        result = result.merge(ci_upper, on=['date', 'affected'], suffixes=('', '_upper'))
        result.columns = ['date', 'affected', 'mean_fraction', 'ci_lower', 'ci_upper']

    else:
        mean_fraction = df.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'], observed=False).mean().reset_index()
        ci_lower = df.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'], observed=False).quantile(0.025).reset_index()
        ci_upper = df.groupby(['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'], observed=False).quantile(0.975).reset_index()
    
        result = mean_fraction.merge(ci_lower, on=['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'], suffixes=('', '_lower'))
        result = result.merge(ci_upper, on=['date', 'affected', 'quantile_pop_bins_percent_bachelor_home'], suffixes=('', '_upper'))
        result.columns = ['date', 'affected', 'quantile_pop_bins_percent_bachelor_home', 'mean_fraction', 'ci_lower', 'ci_upper']
        
    return result


def bootstrap_fractions(df, fraction, evacuated_people, n_bootstraps = 1000, random_state=1234, ses=False):
    np.random.seed(random_state)
    bootstrap_samples = []
    bootstrap_samples_evac = []

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        bootstrap_sample = df.sample(frac=fraction, replace=True)
        bootstrap_sample_evac = bootstrap_sample[bootstrap_sample['PHONE_ID'].isin(evacuated_people)].dropna(subset=['current_bts_id', 'home_bts_id']).reset_index(drop=True)
        bootstrap_fraction = calculate_mover_fraction(bootstrap_sample, ses)
        bootstrap_samples.append(bootstrap_fraction)
        bootstrap_fraction_evac = calculate_mover_fraction(bootstrap_sample_evac, ses)
        bootstrap_samples_evac.append(bootstrap_fraction_evac)
    
    bootstrap_df = pd.concat(bootstrap_samples)
    bootstrap_df_evac = pd.concat(bootstrap_samples_evac)

    # Mean and confidence intervals
    result = calculate_mean_ci(bootstrap_df, ses)
    result_evac = calculate_mean_ci(bootstrap_df_evac, ses)

    return result, result_evac


def rho_assortativity(normalized_matrix):
    """
    Calculates the correlation coefficient of a normalized matrix.
    """
    
    # Calculate numerator
    numerator = (
        np.nansum(
            np.multiply(
                np.outer(np.arange(1, normalized_matrix.shape[0] + 1), np.arange(1, normalized_matrix.shape[1] + 1)),
                normalized_matrix
            )
        )
        - np.nansum(np.arange(1, normalized_matrix.shape[0] + 1) @ normalized_matrix)
        * np.nansum(np.arange(1, normalized_matrix.shape[1] + 1) @ normalized_matrix.T)
    )

    # Calculate denominator
    denominator = (
        np.sqrt(
            np.nansum((np.arange(1, normalized_matrix.shape[0] + 1) ** 2) @ normalized_matrix)
            - (np.nansum(np.arange(1, normalized_matrix.shape[0] + 1) @ normalized_matrix)) ** 2
        )
        * np.sqrt(
            np.nansum((np.arange(1, normalized_matrix.shape[1] + 1) ** 2) @ normalized_matrix.T)
            - (np.nansum(np.arange(1, normalized_matrix.shape[1] + 1) @ normalized_matrix.T)) ** 2
        )
    )

    # Calculate the Pearson correlation coefficient
    rho_X = numerator / denominator
    
    return np.round(rho_X, decimals=3)
