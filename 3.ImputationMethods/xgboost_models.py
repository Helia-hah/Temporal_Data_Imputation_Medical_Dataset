import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

from mean_imputer import mean_hierarchical_imputer
np.random.seed(42)

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

def objective(trial, X_train, y_train, X_valid, y_valid):

    params = {
        "objective": "reg:squarederror",  # Move this here
        "eval_metric": "rmse",  # Move this here
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),  # Learning rate
        "max_depth": trial.suggest_int("max_depth", 3, 10),  # Maximum tree depth
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),  # Number of boosting rounds
        "subsample": trial.suggest_float("subsample", 0.5, 1.0)  # Fraction of training data to use
    }

    model = XGBRegressor(random_state=42,early_stopping_rounds = 10, **params, n_jobs=-1)
    
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
              #early_stopping_rounds=10)
    
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    return rmse

def tune_xgboost_hyperparameters(X_train, y_train, X_valid, y_valid):
   
    # Create the Optuna study with NSGAII Sampler
    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=20)  
    
    best_params = study.best_trial.params

    # Train the best model on the same training and validation data
    best_model = XGBRegressor(random_state=42, early_stopping_rounds = 10, **best_params, n_jobs=-1)
    best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    
    return best_model



def xgboost(train_dataset, target_column, test_dataset, original_mask_test, validation_dataset):

    # If the target_column is a type of Blood Pressure, the other related Blood Pressure columns 
    # will be removed due to high correlation
    filtered_columns = remove_related_blood_pressure_columns(target_column, train_dataset)
    train_dataset_filtered = train_dataset[filtered_columns]
    validation_dataset_filtered = validation_dataset[filtered_columns]
    test_dataset_filtered = test_dataset[filtered_columns]
    
    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original data set
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]

    # Remove irrelevant columns that will not be used for the prediction model
    train_dataset_cleaned = train_dataset_filtered.drop(columns=['subject_id','stay_id','datetime'])
    validation_dataset_cleaned = validation_dataset_filtered.drop(columns=['subject_id','stay_id','datetime'])
    test_dataset_cleanded = test_dataset_filtered.drop(columns=['subject_id','stay_id','datetime'])

    # Drop rows with null values in the target_column of the training dataset
    train_dataset_cleaned.dropna(subset=[target_column],inplace=True)
    validation_dataset_cleaned.dropna(subset=[target_column],inplace=True)
    test_dataset_cleanded = test_dataset_cleanded.loc[indices_not_equal]

    # Separate features (X) and target variable (Y) from the training dataset
    Y_train = train_dataset_cleaned[target_column]
    X_train = train_dataset_cleaned.drop(columns=[target_column])
    
    # Separate features (X) and target variable (Y) from the validation dataset
    Y_val = validation_dataset_cleaned[target_column]
    X_val = validation_dataset_cleaned.drop(columns=[target_column])

    X_test = test_dataset_cleanded.drop(columns=[target_column])
    
    # Normalize
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit and transform the features (X)
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    #Fit and transform the target variable (Y)
    Y_train_scaled = target_scaler.fit_transform(Y_train.to_numpy().reshape(-1, 1))  
    Y_val_scaled = target_scaler.transform(Y_val.to_numpy().reshape(-1, 1))
    

    #best_model = tune_xgboost_hyperparameters(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled)
    best_model = XGBRegressor(learning_rate=0.056, max_depth=10, n_estimators=292, subsample=0.9, random_state=42, n_jobs=-1)
    best_model.fit(X_train_scaled, Y_train_scaled)
    
    Y_pred_scaled = best_model.predict(X_test_scaled)
    Y_predict = target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1))
    
    Y_test_indices = X_test.index.tolist()
    Y_predict_df = pd.DataFrame(Y_predict.round(2), index=Y_test_indices, columns=[target_column])

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return Y_predict_df, execution_time



def xgboost_with_imputer(train_dataset, target_column, test_dataset, original_mask_test, validation_dataset):

    # If the target_column is a type of Blood Pressure, the other related Blood Pressure columns 
    # will be removed due to high correlation
    filtered_columns = remove_related_blood_pressure_columns(target_column, train_dataset)
    train_dataset_filtered = train_dataset[filtered_columns]
    validation_dataset_filtered = validation_dataset[filtered_columns]
    test_dataset_filtered = test_dataset[filtered_columns]
    
    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original data set
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]

    # Drop rows with null values in the target_column of the training dataset
    train_dataset_filtered.dropna(subset=[target_column],inplace=True)
    validation_dataset_filtered.dropna(subset=[target_column],inplace=True)
    
    # Separate features (X) and target variable (Y) from the training dataset
    Y_train = train_dataset_filtered[target_column]
    X_train = train_dataset_filtered.drop(columns=[target_column])
    
    # Separate features (X) and target variable (Y) from the validation dataset
    Y_val = validation_dataset_filtered[target_column]
    X_val = validation_dataset_filtered.drop(columns=[target_column])

    X_test = test_dataset_filtered.drop(columns=[target_column])
    
    # Remove irrelevant columns that will not be used for the prediction model
    columns_with_null = ['Heart Rate',
       'Non Invasive Blood Pressure diastolic',
       'Non Invasive Blood Pressure mean',
       'Non Invasive Blood Pressure systolic', 'O2 saturation pulseoxymetry',
       'Respiratory Rate', 'Height', 'BMI', 'Weight']
    
    columns_with_null = [col for col in columns_with_null if col != target_column and col in X_train.columns]
    
    
    for column in columns_with_null:
        X_train[column] = mean_hierarchical_imputer(column, X_train)
        
    for column in columns_with_null:
        X_val[column] = mean_hierarchical_imputer(column, X_val)
        
    for column in columns_with_null:
        X_test[column] = mean_hierarchical_imputer(column, X_test)
    
    # Remove irrelevant columns that will not be used for the prediction model
    X_train_cleaned = X_train.drop(columns=['subject_id','stay_id','datetime'])
    X_val_cleaned = X_val.drop(columns=['subject_id','stay_id','datetime'])
    X_test_cleaned = X_test.drop(columns=['subject_id','stay_id','datetime'])
    
    # Normalize
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit and transform the features (X)
    X_train_scaled = feature_scaler.fit_transform(X_train_cleaned)
    X_val_scaled = feature_scaler.transform(X_val_cleaned)
    X_test_scaled = feature_scaler.transform(X_test_cleaned)

    #Fit and transform the target variable (Y)
    Y_train_scaled = target_scaler.fit_transform(Y_train.to_numpy().reshape(-1, 1))  # reshape for scaling
    Y_val_scaled = target_scaler.transform(Y_val.to_numpy().reshape(-1, 1))
    

    #best_model = tune_xgboost_hyperparameters(X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled)
    
    best_model = XGBRegressor(learning_rate=0.056, max_depth=10, n_estimators=292, subsample=0.9, random_state=42, n_jobs=-1)
    best_model.fit(X_train_scaled, Y_train_scaled)
    
    X_test_scaled = X_test_scaled[indices_not_equal]
    Y_pred_scaled = best_model.predict(X_test_scaled)
    Y_predict = target_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1))
    
    #Y_test_indices = X_test_scaled.index.tolist()
    Y_predict_df = pd.DataFrame(Y_predict.round(2), index=indices_not_equal, columns=[target_column])

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return Y_predict_df, execution_time


