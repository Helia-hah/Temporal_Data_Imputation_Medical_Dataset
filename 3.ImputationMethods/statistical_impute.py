import pandas as pd
import numpy as np
import time


def stat_impute(statistical_strategy, train_dataset, target_column, test_dataset, original_mask_test):
    # Average/Median approach

    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original data set
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]
    
    # Compute the specified statistical measure (mean or median) for the target column in the training dataset
    computed_statistic = statistical_strategy(train_dataset[target_column]).round(2)
    
    # Impute missing values in the target column using the calculated statistical value
    test_dataset[target_column] = test_dataset[target_column].fillna(computed_statistic)
        
    # Extract indices and values of newly created missing data, excluding original nulls
    imputed_values = test_dataset[target_column].loc[indices_not_equal]
    
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
 
    return imputed_values, execution_time

def stat_hier_impute(statistical_strategy, target_column, test_dataset, original_mask_test):
    # Average-H/Median-H approach
    
    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original dataset
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]

    # Get the name of the function
    strategy_string = statistical_strategy.__name__.replace('nan', '')
    
    # 1.Compute the specified statistical measure for each stay_id
    stat_by_stay = test_dataset.groupby('stay_id')[target_column].transform(strategy_string).round(2)
    # 2.Compute the specified statistical measure for each subject_id
    stat_by_subject = test_dataset.groupby('subject_id')[target_column].transform(strategy_string).round(2)
    # 3.Compute the specified statistical measure for the entire dataset
    overall_stat_value = statistical_strategy(test_dataset[target_column]).round(2)
    
    # Impute missing values hierarchically
    test_dataset[target_column] = test_dataset[target_column].fillna(stat_by_stay)
    test_dataset[target_column] = test_dataset[target_column].fillna(stat_by_subject)
    test_dataset[target_column] = test_dataset[target_column].fillna(overall_stat_value)
    
    # Extract indices and values of newly created missing data, excluding original nulls
    imputed_values = test_dataset[target_column].loc[indices_not_equal]
    
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    
    return imputed_values, execution_time

def stat_hier_hist_impute(statistical_strategy, train_dataset, target_column, test_dataset, original_mask_test):
    # MediTHIM-A/MediTHIM-M approach
    
    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original data set
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]
    
    # Define function to calculate previous values (mean or median) for a group
    def calculate_previous_stat_for_group(group, method):
        return group.expanding(min_periods=1).apply(method).shift()

    # 1.Impute missing values in the target column using historical statistical measures within the same stay_id group
    previous_stat = test_dataset.groupby('stay_id')[target_column].apply(calculate_previous_stat_for_group, method=statistical_strategy)
    previous_stat = previous_stat.round(2)
    test_dataset['previous_stat'] = previous_stat.reset_index(level=0, drop=True)

    test_dataset['target_modified'] = test_dataset[target_column].fillna(test_dataset['previous_stat'])
    test_dataset.drop(columns=['previous_stat'], inplace=True)

    # 2.Impute missing values in the target column using historical statistical measures within the same subject_id group
    previous_stat = test_dataset.groupby('subject_id')[target_column].apply(calculate_previous_stat_for_group, method=statistical_strategy)
    previous_stat = previous_stat.round(2)
    test_dataset['previous_stat'] = previous_stat.reset_index(level=0, drop=True)

    test_dataset['target_modified'] = test_dataset['target_modified'].fillna(test_dataset['previous_stat'])
    test_dataset.drop(columns=['previous_stat'], inplace=True)

    # 3.Impute missing values in the target column with statistical measure for the entire training dataset
    target_column_stat = statistical_strategy(train_dataset[target_column]).round(2)
    test_dataset[target_column] = test_dataset['target_modified'].fillna(target_column_stat)
    test_dataset.drop(columns=['target_modified'], inplace=True)
   
    # Extract indices and values of newly created missing data, excluding original nulls
    imputed_values = test_dataset[target_column].loc[indices_not_equal]
    
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    
    return imputed_values, execution_time




