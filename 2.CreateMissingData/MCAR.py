import wget
wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')

import numpy as np
import pandas as pd
from utils import *
import torch
import os

def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}


def create_mcar_missingness(dataset, target_column, ratio):


    dataset_filtered = dataset[target_column]

    # Find the not null values and their corresponding indices
    not_null_dataset = dataset_filtered[dataset_filtered.notnull()]
    not_null_dataset_index = not_null_dataset.index.tolist()
    not_null_dataset_values = not_null_dataset.to_numpy()
    not_null_dataset_values = not_null_dataset_values[:, np.newaxis]

    # Calculate the desired ratio of missing values in the non-null dataset
    count_missing_values = ratio *(len(dataset))
    desired_ratio = round(count_missing_values / len(not_null_dataset) , 2)

    # Create new missing values using the produce_NA function obtained from the GitHub repository
    X_miss_mcar = produce_NA(not_null_dataset_values, p_miss=desired_ratio, mecha="MCAR")
    X_mcar = X_miss_mcar['X_incomp']
    
    # Create a DataFrame with the target column and corresponding indices after generating the MCAR pattern
    target_column_after_mcar = pd.DataFrame({
        'index': not_null_dataset_index,
        target_column: pd.Series(X_mcar[:, 0], name=target_column)
        })
    target_column_after_mcar = target_column_after_mcar.set_index('index', drop=True)

    # Restore the originally null values to the dataset, filling the gaps between indices
    max_index = dataset.index.max()
    full_index = range(max_index+1)  
    target_column_after_mcar_reindexed = target_column_after_mcar.reindex(full_index)
    
    # Join the target_column with newly created missing values to the other columns in the dataset
    columns_order = dataset.columns
    dataset.drop(columns=target_column, inplace=True)
    dataset_missing = dataset.join(target_column_after_mcar_reindexed, how='inner')
    dataset_missing = dataset_missing[columns_order]

    return dataset_missing



def main():
    """Main function to process dataset and introduce MCAR missingness in various columns."""

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

    # Iterate through columns and missing data ratios, and apply MCAR missingness
    for column, initials in all_columns.items():
        for ratio in missing_data_ratios:

            # Apply MCAR missingness based on the ratio
            if ratio == 0.01:
                dataset_with_mcar = create_mcar_missingness(test_1h.copy(), column, ratio)
            elif ratio == 0.05:
                dataset_with_mcar = create_mcar_missingness(dataset_with_mcar.copy(), column, 0.04)
            else:
                dataset_with_mcar = create_mcar_missingness(dataset_with_mcar.copy(), column, 0.05)

            output_file = f'1h_test_mcar_{initials}_{ratio}.csv'
            dataset_with_mcar.to_csv(output_file, index=False)
            print(f"Saved dataset with {ratio * 100}% missingness for {column} to {output_file}")

if __name__ == "__main__":
    main()

