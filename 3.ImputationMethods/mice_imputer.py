import pandas as pd
import numpy as np
from statsmodels.imputation.mice import MICEData
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

from mean_imputer import mean_hierarchical_imputer

def remove_related_blood_pressure_columns(target_column, dataset):
    blood_pressure_columns = [
        'Non Invasive Blood Pressure diastolic',
        'Non Invasive Blood Pressure mean',
        'Non Invasive Blood Pressure systolic'
    ]
    
    columns_to_drop = []
    if target_column in blood_pressure_columns:
        columns_to_drop = [col for col in blood_pressure_columns if col != target_column]
        
    filtered_columns = [column for column in dataset.columns if column not in columns_to_drop]
    
    return filtered_columns


def mice(train_dataset, target_column, test_dataset, original_mask_test):

    # If the target_column is a type of Blood Pressure, the other related Blood Pressure columns 
    # will be removed due to high correlation
    filtered_columns = remove_related_blood_pressure_columns(target_column, train_dataset)
    train_dataset_filtered = train_dataset[filtered_columns]
    test_dataset_filtered = test_dataset[filtered_columns]
    
    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original data set
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]
    print(indices_not_equal)

    train_dataset_filtered.drop(columns=['subject_id','stay_id','datetime'], inplace=True)
    test_dataset_filtered.drop(columns=['subject_id','stay_id','datetime'], inplace=True)
    test_dataset_filtered = test_dataset_filtered.loc[indices_not_equal]
    
    # Normalize
    feature_scaler = MinMaxScaler()

    # Fit and transform the features (X)
    train_scaled = feature_scaler.fit_transform(train_dataset_filtered)
    test_scaled = feature_scaler.transform(test_dataset_filtered)
    
    lr =LinearRegression()
    imputer = IterativeImputer(estimator=lr, max_iter=10, random_state=42, n_jobs=-1)
    imputed_train_data = imputer.fit_transform(train_scaled)

    # Apply the imputer to the test data 
    imputed_test_data = imputer.transform(test_scaled)
    imputed_test_df = pd.DataFrame(imputed_test_data, columns=test_dataset_filtered.columns)

    # Save to a CSV file
    imputed_test_df.to_csv('imputed_test_data.csv', index=False)
    

    Y_predict_df = pd.DataFrame(
        feature_scaler.inverse_transform(imputed_test_data),
        columns=test_dataset_filtered.columns,  # Use column names from the original dataset
        index=indices_not_equal
        )

    # Extract the target column by name
    Y_predict = Y_predict_df[target_column]
    
    predict_df = pd.DataFrame(Y_predict.round(2), index=indices_not_equal, columns=[target_column])
    print('null:', predict_df[predict_df.isnull()])
    print('predict:', predict_df)

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return predict_df, execution_time


