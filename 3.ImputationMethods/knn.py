import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

from median_imputer import median_hierarchical_imputer

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


def knn_with_imputer(train_dataset, target_column, test_dataset, original_mask_test, validation_dataset):

    # If the target_column is a type of Blood Pressure, the other related Blood Pressure columns 
    # will be removed due to high correlation
    filtered_columns = remove_related_blood_pressure_columns(target_column, train_dataset)
    train_dataset_filtered = train_dataset[filtered_columns]
    test_dataset_filtered = test_dataset[filtered_columns]
    
    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original data set
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]

    # Drop rows with null values in the target_column of the training dataset
    train_dataset_filtered.dropna(subset=[target_column],inplace=True)
    
    # Separate features (X) and target variable (Y) from the training dataset
    Y_train = train_dataset_filtered[target_column]
    X_train = train_dataset_filtered.drop(columns=[target_column])

    
    X_test = test_dataset_filtered.drop(columns=[target_column])
    
    # Remove irrelevant columns that will not be used for the prediction model
    columns_with_null = ['Heart Rate',
       'Non Invasive Blood Pressure diastolic',
       'Non Invasive Blood Pressure mean',
       'Non Invasive Blood Pressure systolic', 'O2 saturation pulseoxymetry',
       'Respiratory Rate', 'age', 'gender_F', 'Height', 'BMI', 'Weight']
    
    columns_with_null = [col for col in columns_with_null if col != target_column]
    
    
    for column in columns_with_null:
        X_train[column] = median_hierarchical_imputer(column, X_train)
        
    for column in columns_with_null:
        X_test[column] = median_hierarchical_imputer(column, X_test)

    # Remove irrelevant columns that will not be used for the prediction model
    X_train_cleaned = X_train.drop(columns=['subject_id','stay_id','datetime'])
    X_test_cleaned = X_test.drop(columns=['subject_id','stay_id','datetime'])
    
    # Normalize
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit and transform the features (X)
    X_train_scaled = feature_scaler.fit_transform(X_train_cleaned)
    X_test_scaled = feature_scaler.transform(X_test_cleaned)

    #Fit and transform the target variable (Y)
    Y_train_scaled = target_scaler.fit_transform(Y_train.to_numpy().reshape(-1, 1))  # reshape for scaling
    

    model = KNeighborsRegressor(n_neighbors=5)

    model.fit(X_train_scaled, Y_train_scaled)
    
    X_test_scaled = X_test_scaled[indices_not_equal]
    Y_pred_scaled = model.predict(X_test_scaled)
    Y_predict = target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1))
    
    Y_predict_df = pd.DataFrame(Y_predict.round(2), index=indices_not_equal, columns=[target_column])

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return Y_predict_df, execution_time
