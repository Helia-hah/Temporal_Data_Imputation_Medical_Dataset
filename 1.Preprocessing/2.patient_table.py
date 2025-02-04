import pandas as pd
import numpy as np
import os


def preprocessing_patient_table(patient):
    patient = patient[['subject_id','anchor_age','gender','anchor_year']]
    patient.rename(columns={'anchor_year': 'year', 'anchor_age': 'age'}, inplace=True)
    return patient

def merge_patient_records_with_demographic(dataset, patient):
    
    # Add gender and age to the dataset based on the information from the existing year.
    dataset['rounded_hours'] = pd.to_datetime(dataset['rounded_hours'])
    dataset['year'] = dataset['rounded_hours'].dt.year
    merged_dataset = pd.merge(dataset, patient, on =['subject_id','year'], how='left')
    
    null_rows = merged_dataset[merged_dataset['age'].isnull()]
    
    # Find the age of patients for whom we have their age recorded in another year.
    for index, row in null_rows.iterrows():
        
        matching_row = patient[patient['subject_id'] == row['subject_id']]

        # Calculate the age difference based on the difference in years
        year_difference = row['year'] - matching_row['year'].values[0]
        merged_dataset.loc[index, 'age'] = matching_row['age'].values[0] + year_difference
        merged_dataset.loc[index, 'gender'] = matching_row['gender'].values[0]
        
    merged_dataset.drop(columns=['year'],inplace=True)
    
    return merged_dataset

file_path = '/home/helia24/projects/def-pbranco/helia24/final_version/dataset-1/1h_vital.csv'
vital_1h = pd.read_csv(file_path)

file_path = '/home/helia24/projects/def-pbranco/helia24/original_dataset/patients.csv.gz'
patient_table = pd.read_csv(file_path, compression='gzip')


preprocessed_patient = preprocessing_patient_table(patient_table)
print('preprocessed_patient is done')
vital_patient = merge_patient_records_with_demographic(vital_1h, preprocessed_patient)
print('vital_patient is done')


vital_patient.to_csv('1h_vp.csv', index=False)


def main():
    """Main function to preprocess patient data and merge with vitals."""
    
    vital_file = os.getenv('VITAL_FILE', 'data/1h_vital.csv')  
    patient_file = os.getenv('PATIENT_FILE', 'data/patients.csv.gz')  

    if not os.path.exists(vital_file) or not os.path.exists(patient_file):
        print("Error: Data files not found")
        return
    
    vital_1h = pd.read_csv(vital_file)
    patient_table = pd.read_csv(patient_file, compression='gzip')

    # Process the patient table 
    preprocessed_patient = preprocessing_patient_table(patient_table)

    # Merge preprocessed patient data with vital signs
    vital_patient = merge_patient_records_with_demographic(vital_1h, preprocessed_patient)
 
    output_file = 'data/1h_vp.csv'
    os.makedirs('data', exist_ok=True)  
    vital_patient.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
