import pandas as pd
import numpy as np
import os


def remove_outliers(group, label, possible_range_dict):
    possible_min = possible_range_dict[label]['min']
    possible_max = possible_range_dict[label]['max']
    
    return group[(group['result_value'] >= possible_min) & (group['result_value'] <= possible_max)]

def preprocessing_omr_table(omr):
    
    # Define the mapping of old values to new values
    replacement_dict = {
        'Height (Inches)': 'Height',
        'Weight (Lbs)': 'Weight',
        'BMI (kg/m2)': 'BMI'
    }
    omr.loc[:, 'result_name'] = omr['result_name'].replace(replacement_dict)
    
    # Just select the BMI, Height and Weight
    selected_categories = ['BMI','Weight','Height']
    omr = omr[omr['result_name'].isin(selected_categories)]

    # Remove duplicate rows where 'subject_id', 'chartdate', 'result_name', and 'result_value' columns have identical values
    columns_to_check = ['subject_id', 'chartdate', 'result_name', 'result_value']
    omr = omr.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Define a valid range of values (min and max) for each category in the 'result_name' column
    possible_range = pd.DataFrame()
    selected_categories = omr['result_name'].unique().tolist()
    possible_range['label']= selected_categories     # selected_categories : weight, BMI, height
    possible_range['min']=[40,5,20]
    possible_range['max']=[1435,190,110]
    possible_range_dict = possible_range.set_index('label').T.to_dict()

    # Remove outliers in the 'result_value' column by filtering values outside the defined possible ranges
    #for each category in the 'result_name' column.
    omr['result_value'] = omr['result_value'].astype('float64')
    all_columns = omr.columns
    omr_with_removed_outliers = omr.groupby(['result_name'])[all_columns].apply(
                lambda group: remove_outliers(group, group.name, possible_range_dict)).reset_index(drop=True)
    
    # Calculate the mean of 'result_value' for rows with the same 'subject_id', 'chartdate', and 'result_name'
    grouped_omr = omr_with_removed_outliers.groupby(['subject_id', 'chartdate', 'result_name'])['result_value'].mean().reset_index()
    
    # Convert the categories in 'result_name' to distinct columns
    expanded_omr = grouped_omr.pivot_table(
                        index=['subject_id', 'chartdate'],
                        columns='result_name', 
                        values='result_value',
                        aggfunc='first').reset_index()
    
    return expanded_omr

def merge_patient_records_with_omr(dataset, omr):
    
    # Calculate the mean height for each patient and add it to the original dataset, disregarding the date.
    omr_height = omr.groupby(['subject_id'])['Height'].mean().reset_index()
    omr_height['Height'] = omr_height['Height'].round(1)
    merged_df_height = pd.merge(dataset, omr_height, on=['subject_id'], how='left')
    omr.drop(columns=['Height'],inplace=True)
    
    # Convert 'charttime' to datetime and extract date
    merged_df_height['rounded_hours'] = pd.to_datetime(merged_df_height['rounded_hours'])
    merged_df_height['date'] = merged_df_height['rounded_hours'].dt.date
    
    # Rename chartdate to 'date' and extract date
    omr = omr.rename(columns={'chartdate': 'date'})
    omr['date'] = pd.to_datetime(omr['date']).dt.date
    
    # Merge main dataset with OMR table based on 'subject_id' and 'date' - add weight & BMI
    merged_df_omr = pd.merge(merged_df_height, omr, on=['subject_id', 'date'], how='left')

    merged_df_omr = merged_df_omr.drop(columns=['date'])
    
    return merged_df_omr


def main():
    """Main function to preprocess OMR data and merge with vital signs."""
    
    vp_file_path = os.getenv('VP_FILE_PATH', 'data/1h_vp.csv')  
    omr_file_path = os.getenv('OMR_FILE_PATH', 'data/omr.csv.gz')

    if not os.path.exists(vp_file_path) or not os.path.exists(omr_file_path):
        print("Error: Data files not found")
        return
    
    vp_1h = pd.read_csv(vp_file_path)
    omr_table = pd.read_csv(omr_file_path, compression='gzip')

    # Process the OMR table
    preprocessed_omr = preprocessing_omr_table(omr_table)

    # Merge vital signs with OMR data
    vital_patient_omr = merge_patient_records_with_omr(vp_1h, preprocessed_omr)

    output_file_path = '1h_vpo.csv'
    os.makedirs('data', exist_ok=True)  
    vital_patient_omr.to_csv(output_file_path, index=False)
    print(f"Processed data saved to {output_file_path}")

if __name__ == "__main__":
    main()