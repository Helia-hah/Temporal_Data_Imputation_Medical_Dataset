import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from scipy.stats import kendalltau
import os

def correlation(dataset, column1, column2, type2):
    
    # Remove NaN rows so can calculate correlation
    no_nulls_dataset = dataset[[column1, column2]].dropna()
    
    # 0=numeric(continous) /1=binary / 2=categorical(nominal) 
    if type2 == 0:
        correlation = no_nulls_dataset[column1].corr(no_nulls_dataset[column2])
    
    elif type2 == 1:
        correlation, p_value = pointbiserialr(no_nulls_dataset[column2], no_nulls_dataset[column1])
        
    else:
        no_nulls_dataset['Nominal_encoded'] = no_nulls_dataset[column2].astype('category').cat.codes
        correlation, p_value = kendalltau(no_nulls_dataset[column1], no_nulls_dataset['Nominal_encoded'])
        
    abs_correlation = abs(correlation)
        
    return abs_correlation

def create_mar_missingness(dataset, target_column, ratio):
    
    correlation_with_target = []
    # 0=numeric(continous) /1=binary / 2=categorical(nominal) 
    all_types = [0,2,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    all_columns = dataset.columns
    selected_columns = [col for col in all_columns if col not in ['subject_id', 'stay_id', 'datetime']]

    # Calculate the correlation between the target column and other columns based on their data types.
    for index, column_name in enumerate(selected_columns):
        if column_name == target_column:
            correlation_with_target.append(0)  
        else:
            corr_value = correlation(dataset, target_column, column_name, all_types[index])
            correlation_with_target.append(corr_value)
            
    # Find the the feature with the highest correlation to target_column    
    max_correlation_index = np.argmax(correlation_with_target)
    max_correlation = selected_columns[max_correlation_index]
    
    # Estimate the desired number of missing values in the dataset
    count_missing_values = ratio * (len(dataset))
    half_missing_count = int(count_missing_values/2)
    
    not_null_dataset = dataset[dataset[target_column].notnull()]
    
    # Find smallest and largest indices and combine them
    smallest_indices = not_null_dataset[max_correlation].nsmallest(half_missing_count).index.tolist()
    largest_indices = not_null_dataset[max_correlation].nlargest(half_missing_count).index.tolist()
    all_indices = smallest_indices + largest_indices
    
    dataset_missing = dataset.copy()
    dataset_missing.loc[all_indices, target_column] = np.nan
    
    return dataset_missing


def main():
    """Main function to process dataset and introduce MAR missingness in various columns."""

    test_file = os.getenv('TEST_FILE', 'data/1h_test_set_m.csv')  
    if not os.path.exists(test_file):
        print(f"Error: File not found: {test_file}")
        return

    test_1h = pd.read_csv(test_file)

    # Define columns and their initials for naming
    all_columns = {
        'Heart Rate': 'HR',
        'Respiratory Rate': 'RR',
        'O2 saturation pulseoxymetry': 'O2SP',
        'Non Invasive Blood Pressure diastolic': 'NIBPD',
        'Non Invasive Blood Pressure mean': 'NIBPM',
        'Non Invasive Blood Pressure systolic': 'NIBPS'   
    }

    # Define different missing data ratios to be applied
    missing_data_ratios = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # Iterate through columns and missing data ratios, and apply MAR missingness
    for column, initials in all_columns.items():
        for ratio in missing_data_ratios:

            # Apply MAR missingness based on the ratio
            dataset_with_mcar = create_mar_missingness(test_1h.copy(), column, ratio)

            output_file = f'1h_test_mar_{initials}_{ratio}.csv'
            dataset_with_mcar.to_csv(output_file, index=False)
            print(f"Saved dataset with {ratio * 100}% missingness for {column} to {output_file}")

if __name__ == "__main__":
    main()

