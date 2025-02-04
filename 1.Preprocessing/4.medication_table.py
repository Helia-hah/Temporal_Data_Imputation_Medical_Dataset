import pandas as pd
import numpy as np
import re
import os


def split_rows(df, hour):
  
    # Create new rows for each hour in the range
    rows = []
    for index, row in df.iterrows():

        start_time, end_time = row['min'], row['max']
        time_range = pd.date_range(start=start_time, end=end_time, freq=hour)
        
        for time in time_range:
            if row['starttime'] <= time <= row['endtime']:
                new_row = {
                    'subject_id': row['subject_id'],
                    'stay_id': row['stay_id'],
                    'ordercategoryname': row['ordercategoryname'],
                    'rounded_hours': time,
                    'amount': 1,
                    'patientweight': row['patientweight']
                }
                rows.append(new_row)
    

    return pd.DataFrame(rows)

def remove_number_prefix(category_name):
    return re.sub(r'^\d+-', '', category_name)


def preprocessing_inputevents_table(vital, inputevents):
    
    # Create a table containing subject_id,stay_id and min and max time for each stay
    min_max_stay = vital.groupby(['stay_id'])['rounded_hours'].agg(['min', 'max']).reset_index()
    min_max_stay['min'] = pd.to_datetime(min_max_stay['min'])
    min_max_stay['max'] = pd.to_datetime(min_max_stay['max'])
    
    
    # Select the positive amounts
    inputevents = inputevents[inputevents['amount'] > 0]
    columns = ['subject_id', 'stay_id', 'starttime', 'endtime','ordercategoryname', 'amount','patientweight']
    inputevents = inputevents[columns]
    inputevents['amount'] = 1
    # Convert kilograms to pounds
    inputevents['patientweight'] = round(inputevents['patientweight'] * 2.20462, 2)
    
    # Remove the outlier of weight. possible range (pounds) [40, 1435]
    inputevents['patientweight'] = np.where((inputevents['patientweight'] >= 40) & (inputevents['patientweight'] <= 1435), 
                        inputevents['patientweight'], 
                        np.nan)
    
    inputevents['starttime'] = pd.to_datetime(inputevents['starttime'])
    inputevents['endtime'] = pd.to_datetime(inputevents['endtime'])

    # Round 'starttime' and 'endtime' based on the minute threshold
    inputevents['starttime'] = np.where(inputevents['starttime'].dt.minute <= 30,
                                     inputevents['starttime'].dt.floor('h'),
                                     inputevents['starttime'].dt.ceil('h'))

    inputevents['endtime'] = np.where(inputevents['endtime'].dt.minute <= 30,
                                   inputevents['endtime'].dt.floor('h'),
                                   inputevents['endtime'].dt.ceil('h'))
    

    # Merge inputevents with min_max_time to keep only matching stay_ids 
    # and add the min and max time columns for each stay.
    selected_stay_id = min_max_stay['stay_id'].unique().tolist()
    inputevents_filtered = inputevents[inputevents['stay_id'].isin(selected_stay_id)]
    inputevents_merged_time = pd.merge(inputevents_filtered, min_max_stay, how='inner', on =['stay_id'])
    
    
    # Apply the split_rows method to convert each input into multiple rows based on the time range
    inputevents_split_rows = split_rows(inputevents_merged_time, 'h')
    
    inputevents_split_rows['rounded_hours'] = pd.to_datetime(inputevents_split_rows['rounded_hours'])
    inputevents_split_rows = inputevents_split_rows.sort_values(by=['subject_id', 'rounded_hours']).reset_index(drop=True)

    # Calculate the mean of weight for the same time of same subject_id
    inputevents_weight = inputevents_split_rows.groupby(['subject_id','rounded_hours'])['patientweight'].mean().reset_index()
    
    #Calculate the moving average for 'patientweight' over a 24-hour period
    all_columns = inputevents_weight.columns
    inputevents_weight_grouped = inputevents_weight.groupby('subject_id') [all_columns]
    inputevents_weight['weight_24h_mean'] = inputevents_weight_grouped.apply(
        lambda x: x.set_index('rounded_hours')['patientweight']
               .rolling('24h', min_periods=1)
               .mean()
        ).reset_index(drop=True)
    
    inputevents_split_rows.drop (columns='patientweight', inplace=True)
    inputevents_weight.drop (columns='patientweight', inplace=True)
    inputevents_merged_weight = pd.merge(inputevents_split_rows, inputevents_weight, on=['subject_id','rounded_hours'], how='left')
    inputevents_merged_weight['weight_24h_mean'] = round(inputevents_merged_weight['weight_24h_mean'], 2)
    
    # Remove rows that have the same stay_id, label, time, and weight
    inputevents_merged_weight = inputevents_merged_weight.drop_duplicates(keep='first')
    
    
    # Remove the numbers and prefixes at the beginning of the label categories
    inputevents_merged_weight['ordercategoryname'] = inputevents_merged_weight['ordercategoryname'].apply(remove_number_prefix)
    
    
    # Convert each category in 'ordercategoryname' into distinct columns
    inputevents_pivot = inputevents_merged_weight.pivot_table(
                        index=['subject_id', 'stay_id', 'rounded_hours','weight_24h_mean'],
                        columns='ordercategoryname',
                        values='amount',
                        aggfunc='first').reset_index()
    inputevents_pivot = inputevents_pivot.fillna(0)
    
    
    return inputevents_pivot


def find_closest_weight_24h(dataset):
    
    null_weight = dataset[dataset['weight_24h_mean'].isnull()]
    not_null_weight = dataset[dataset['weight_24h_mean'].notnull()]
    if null_weight.empty or not_null_weight.empty:
        return dataset
    
     # Find the index of the closest rounded_hour
    closest_time_index = null_weight['rounded_hours'].apply(
        lambda hour: (not_null_weight['rounded_hours'] - hour).abs().idxmin())

    # Calculate the time difference between closest rounded_hour and the null rows
    closest_time_diff = null_weight['rounded_hours'].apply(
        lambda hour: (not_null_weight['rounded_hours'] - hour).abs().min())
    
    # Define a time threshold (24hour) for selecting the closest rounded_hours 
    mask_within_threshhold = closest_time_diff <= pd.Timedelta(hours=24)
    closest_time = closest_time_index[mask_within_threshhold]
    dataset.loc[closest_time.index, 'weight_24h_mean'] = dataset.loc[closest_time.values, 'weight_24h_mean'].values

    
    return dataset

def merge_patient_records_with_medications(dataset, inputevents):
    
    dataset['rounded_hours'] = pd.to_datetime(dataset['rounded_hours'])
    inputevents['rounded_hours'] = pd.to_datetime(inputevents['rounded_hours'])
    inputevents.drop(columns=['subject_id'], inplace= True)
    
    # Merge main dataset with inputevents table based on 'stay_id' and 'rounded_hours' - add all medications
    merged_df_inputevents = pd.merge(dataset, inputevents, on=['stay_id', 'rounded_hours'], how='left')
    
    # Fill null values in the weight column with the closest valid point within a 24-hour window
    all_columns = merged_df_inputevents.columns
    merged_df_grouped = merged_df_inputevents.groupby('subject_id')[all_columns]
    merged_df_fillnan_weight = merged_df_grouped.apply(find_closest_weight_24h).reset_index(drop=True)
    
    # Fill the null values in medication columns with 0
    columns_to_fill = ['Antibiotics (IV)','Antibiotics (Non IV)', 'Blood Products', 'Drips', 'Enteral Nutrition',
       'Fluids (Colloids)', 'Fluids (Crystalloids)', 'IV Fluid Bolus', 'Insulin (Non IV)', 'Med Bolus', 'Oral/Gastric Intake',
       'Parenteral Nutrition', 'Pre Admission/Non-ICU', 'Prophylaxis (IV)','Prophylaxis (Non IV)', 'Supplements']
    merged_df_fillnan_weight[columns_to_fill] = merged_df_fillnan_weight[columns_to_fill].fillna(0)
    
    # Compute the average of 'Weight' and 'patientweight' for each row
    merged_df_fillnan_weight['Weight'] = merged_df_fillnan_weight[['Weight', 'weight_24h_mean']].mean(axis=1).round(2)
    merged_df_fillnan_weight.drop(columns=['weight_24h_mean'], inplace= True)
    
    return merged_df_fillnan_weight


def adjust_omr_data(dataset):
    
    # Recalculate the BMI using 'new_BMI' based on the 'Height' and updated 'Weight' values
    dataset['BMI_new'] = ((dataset['Weight']/ (dataset['Height'] ** 2)) * 703).round(2)
    # Remove the outlier of BMI. possible range[5, 190]
    dataset['BMI_new'] = np.where((dataset['BMI_new'] >= 5) & (dataset['BMI_new'] <= 190), 
                        dataset['BMI_new'], 
                        np.nan)
    
    # Use 'new_BMI' when both 'Height' and 'Weight' are not null; 
    # otherwise, keep the existing 'BMI' values
    dataset['BMI'] = np.where(dataset['Height'].notnull() & dataset['Weight'].notnull(), 
                              dataset['BMI_new'],  
                              dataset['BMI']
                              )
    dataset.drop(columns = ['BMI_new'],inplace =True)

    # Calculate the Height where it is null, but both BMI and Weight are available
    condition = (dataset['BMI'].notnull()) & (dataset['Weight'].notnull()) & (dataset['Height'].isnull())
    dataset.loc[condition, 'Height'] = np.sqrt((dataset['Weight']*703) / dataset['BMI'])
    dataset['Height'] = np.round(dataset['Height'], 2)
    
    return dataset

def shift_dates_group(group):
    
    # Shift the minimum recorded date for each patient to January 1, 2024,
    # and accordingly adjust the other dates to maintain the time intervals consistent with the original data.
    
    min_date = group['rounded_hours'].dt.date.min() 
    target_date = pd.Timestamp('2024-01-01').date() 
    shift_amount = pd.DateOffset(days=(target_date - min_date).days) 
    group['shifted_datetime'] = group['rounded_hours'] + shift_amount
    
    return group


def shift_dates(dataset):
    
    dataset = dataset.copy()
    
    dataset['rounded_hours'] = pd.to_datetime(dataset['rounded_hours'])
    columns = dataset.columns
    df_shifted = dataset.groupby('subject_id')[columns].apply(shift_dates_group)
    df_shifted.reset_index(drop=True, inplace=True)

    df_shifted.drop(columns=['rounded_hours'], axis=1, inplace=True)
    df_shifted.rename(columns={'shifted_datetime': 'datetime'}, inplace=True)
    
    # Rearrange the columns and sort the dataset by 'subject_id' and 'datetime'
    first_columns = ['subject_id', 'stay_id', 'datetime', 'age', 'gender']
    remaining_columns = [col for col in df_shifted.columns if col not in first_columns]
    df_shifted = df_shifted[first_columns + remaining_columns]
    
    df_shifted = df_shifted.sort_values(by=['subject_id','datetime']).reset_index(drop=True)
    
    return df_shifted



def main():
    """Main function to preprocess medication table and merge with vital signs."""
    
    vpo_file_path = os.getenv('VPO_FILE_PATH', 'data/1h_vpo.csv')  
    inputevents_file_path = os.getenv('INPUTEVENTS_FILE_PATH', 'data/inputevents.csv.gz')

    if not os.path.exists(vpo_file_path) or not os.path.exists(inputevents_file_path):
        print("Error: Data files not found. Please provide the correct paths.")
        return
    
    vpo_1h = pd.read_csv(vpo_file_path)
    inputevents_table = pd.read_csv(inputevents_file_path, compression='gzip')

    # Process the input events data
    preprocessed_inputevents = preprocessing_inputevents_table(vpo_1h, inputevents_table)

    # Merge vital signs with medication data
    vpo_medication = merge_patient_records_with_medications(vpo_1h, preprocessed_inputevents)

    # Adjust the medication data
    vpo_medication_adjusted = adjust_omr_data(vpo_medication)

    # Shift dates for medication data
    vpo_medication_shifted = shift_dates(vpo_medication_adjusted)


    output_file_path = '1h_vpoms.csv'
    os.makedirs('data', exist_ok=True) 
    vpo_medication_shifted.to_csv(output_file_path, index=False)
    print(f"Processed data saved to {output_file_path}")

if __name__ == "__main__":
    main()
