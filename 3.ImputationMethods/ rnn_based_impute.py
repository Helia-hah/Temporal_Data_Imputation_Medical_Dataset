import pandas as pd
import numpy as np
import time
import torch
import os
from pypots.imputation import BRITS
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
torch.manual_seed(42)         
torch.cuda.manual_seed(42)    
torch.cuda.manual_seed_all(42)


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


def adjust_stay_length_to_3D(dataset, threshold=15, is_test=False):
    
    stay_matrices = []
    grouped = dataset.groupby('stay_id')

    for stay_id, group in grouped:
        
        group_size = len(group)

        if group_size < threshold:
            # Apply padding if the group size is smaller than the threshold
            
            # Drop columns that are not significant for imputation
            drop_columns = ['stay_id']
            if is_test:
                drop_columns.append('index_col')
            group_dropped = group.drop(columns=drop_columns)

            # Calculate the column-wise average to generate padding values
            group_mean = group_dropped.mean(axis=0).values
            group_mean = np.round(group_mean, 2)

            # Calculate how many rows to add
            rows_to_add = threshold - group_size
            rows_before = int(rows_to_add / 2)
            rows_after = rows_to_add - rows_before

            # Generate padding values
            padding_matrix_before = np.tile(group_mean, (rows_before, 1))
            padding_matrix_after = np.tile(group_mean, (rows_after, 1))

            if is_test:
                # Add 'index_col' padding for the prediction step in testing data
                padding_indices_before = np.tile(-1, (rows_before, 1))
                padding_indices_after = np.tile(-1, (rows_after, 1))

                # Include the extra 'index_col' column in padding matrices
                padding_matrix_before = np.hstack([padding_matrix_before, padding_indices_before])
                padding_matrix_after = np.hstack([padding_matrix_after, padding_indices_after])

                # Add 'index_col' back to group_filtered 
                group_dropped['index_col'] = group['index_col']

            # Combine the padding and the original data
            group_padded = np.vstack([padding_matrix_before, group_dropped.values, padding_matrix_after])
            group_padded_df = pd.DataFrame(group_padded, columns=group_dropped.columns)

            # Convert this 2D dataframe to a 2D numpy array
            stay_matrix = group_padded_df.values
            stay_matrices.append(stay_matrix)

        else:
            # Apply sliding window if group size is at least the threshold
            windowed_data = []
            group_reduced = group.drop(columns= 'stay_id')
            for start in range(group_size - threshold + 1):
                end = start + threshold
                window = group_reduced.iloc[start:end]
                windowed_data.append(window.values)

            stay_matrices.extend(windowed_data)

    stay_3D_array = np.stack(stay_matrices, axis=0)
    print(f"Final 3D array shape: {stay_3D_array.shape}")

    return stay_3D_array

def rnn_based_imputation(train_dataset, target_column, test_dataset, original_mask_test, rnn_model):

    # If the target_column is a type of Blood Pressure, the other related Blood Pressure columns 
    # will be removed due to high correlation
    filtered_columns = remove_related_blood_pressure_columns(target_column, train_dataset)
    train_dataset = train_dataset[filtered_columns]
    test_dataset= test_dataset[filtered_columns]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    
    # Get the indices of null values that are created in the desired column but do not exist in the original data set
    mask_after_missingness = np.where(test_dataset[target_column].isnull(), 0, 1)
    indices_not_equal = np.where(mask_after_missingness != original_mask_test[target_column].tolist())[0]
    
    train_stay = train_dataset['stay_id']
    test_stay = test_dataset['stay_id']

    train_dataset.drop(columns=['subject_id', 'stay_id', 'datetime'], inplace=True)
    test_dataset.drop(columns=['subject_id', 'stay_id', 'datetime'], inplace=True)


    # Apply MinMaxScaler and convert the output back to a pandas DataFrame
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(train_dataset)
    X_test_scaled = feature_scaler.transform(test_dataset)

    X_train_scaled_p = pd.DataFrame(X_train_scaled, columns=train_dataset.columns)
    X_test_scaled_p = pd.DataFrame(X_test_scaled, columns=test_dataset.columns)

    X_train_scaled_df = pd.concat([train_stay, X_train_scaled_p], axis=1)
    X_test_scaled_df = pd.concat([test_stay, X_test_scaled_p], axis=1)


    # Prepare the train dataset in a format suitable for the model
    train_adjusted = adjust_stay_length_to_3D(X_train_scaled_df, threshold=15)
    train_adjusted = train_adjusted.astype(float)

    # Add 'index_col' padding for the prediction step in testing data
    # Prepare the test dataset in a format suitable for the model
    X_test_scaled_df['index_col'] = X_test_scaled_df.index
    test_adjusted_with_index = adjust_stay_length_to_3D(X_test_scaled_df, threshold=15, is_test=True)
    test_adjusted_without_index = test_adjusted_with_index[:, :, :-1]
    test_adjusted_without_index = test_adjusted_without_index.astype(float)


    train_sequence = {"X": train_adjusted}
    test_sequence = {"X": test_adjusted_without_index}  


    # Initialize and fit the rnn_model
    n_features = train_adjusted.shape[2]
    print('n_features: ', n_features)
    model = rnn_model(n_steps=15, n_features=n_features, rnn_hidden_size=32, epochs=5, device=device)
    model.fit(train_sequence)

    # Perform imputation
    imputation = model.impute(test_sequence)

    # Reshape imputed data back to 2D for inverse transformation
    imputation_reshaped = imputation.reshape(-1, imputation.shape[2])
    # Inverse transform to get the original scale back
    imputation_scaled = feature_scaler.inverse_transform(imputation_reshaped)

    # Reshape the imputed data back to 3D
    imputation_scaled_3d = imputation_scaled.reshape(test_adjusted_without_index.shape[0], 
                                                test_adjusted_without_index.shape[1],
                                                test_adjusted_without_index.shape[2]) 

    # Append the 'index_col' back to the imputed test dataset
    index_column = test_adjusted_with_index[:, :, -1:]
    imputation_with_index = np.concatenate((imputation_scaled_3d, index_column), axis=2)

    # Extract imputed predictions for synthesized missing values
    last_column = imputation_with_index[:, :, -1]
    rows_to_extract = np.argwhere(np.isin(last_column, indices_not_equal))
    matching_rows = [imputation_with_index[row[0], row[1]] for row in rows_to_extract]
    matching_rows = np.array(matching_rows) 
    matching_rows_df = pd.DataFrame(matching_rows) 

    selected_columns = [col for col in test_dataset.columns if col != 'stay_id']
    matching_rows_df.columns = selected_columns

    # Calculate the average of predictions for stays longer than the threshold
    averaged_values = matching_rows_df.groupby('index_col')[target_column].mean().reset_index()
    averaged_values_sorted = averaged_values.sort_values(by='index_col', ascending=True).reset_index(drop=True) 
    print('shape of prediction', averaged_values_sorted.shape)

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return averaged_values_sorted[target_column], execution_time, indices_not_equal 



def evaulate(actual_values, imputed_values):
    
    mae = round(mean_absolute_error(actual_values, imputed_values),2)
    mse = round(mean_squared_error(actual_values, imputed_values),2)
    rmse = round(np.sqrt(mse),2)
    
    return mae, mse, rmse





