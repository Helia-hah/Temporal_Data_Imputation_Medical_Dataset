import pandas as pd
import numpy as np
import os


d_items_file_path = os.getenv('D_ITEMS_FILE', 'data/d_items.csv.gz')  
chartevents_file_path = os.getenv('CHARTEVENTS_FILE', 'data/chartevents.csv.gz')  

# Ensure that the file paths exist before reading
if not os.path.exists(d_items_file_path):
    raise FileNotFoundError(f"File not found: {d_items_file_path}")
if not os.path.exists(chartevents_file_path):
    raise FileNotFoundError(f"File not found: {chartevents_file_path}")

# Reading data files
d_items = pd.read_csv(d_items_file_path, compression='gzip')
chartevents = pd.read_csv(chartevents_file_path, compression='gzip')



# Select 'Routine Vital Signs' and 'Respiratory' 
selected_d_items = d_items[d_items['category'].isin(['Routine Vital Signs','Respiratory'])]
# Select just Numeric Features
selected_d_items = selected_d_items[selected_d_items['param_type']=='Numeric']
                           
selected_d_items = selected_d_items[['itemid', 'label']]


# Filter chartevents to include only those with item IDs selected from the d_items table
events = pd.merge(chartevents, selected_d_items, how='inner', on ='itemid')
columns = ['subject_id','stay_id','label', 'charttime', 'valuenum']
events = events[columns].sort_values(by=['subject_id','stay_id','label','charttime'])

# Identify the 6 most common vital signs (labels) based on the number of unique stay IDs,
# and filter the dataset to include only events corresponding to these vital signs.
frequent_labels = events.groupby(['label'])['stay_id'].nunique().sort_values(ascending=False)[:6].index.tolist()
events_with_frequent_labels = events[events['label'].isin(frequent_labels)].reset_index(drop=True)


# Define a valid range of values (min and max) for each category in the 'label' column
possible_range = pd.DataFrame()
possible_range['label']= frequent_labels
possible_range['min']=[-3,-3,-1,-5,-5,-5]
possible_range['max']=[303,303,101,505,505,505]
possible_range_dict = possible_range.set_index('label').T.to_dict()

def remove_outliers(group, label, possible_range_dict):
    possible_min = possible_range_dict[label]['min']
    possible_max = possible_range_dict[label]['max']
    
    return group[(group['valuenum'] >= possible_min) & (group['valuenum'] <= possible_max)]

# Remove outliers in the 'valuenum' column by filtering values outside the defined possible ranges
# for each category in the 'label' column.
all_columns= events_with_frequent_labels.columns
events_with_removed_outliers = events_with_frequent_labels.groupby(['label'])[all_columns].apply(
                lambda group: remove_outliers(group, group.name, possible_range_dict)).reset_index(drop=True)


# Filter dataset to include only stay IDs with a minimum label frequency of 5 or more
label_count_by_stay = events_with_removed_outliers.groupby(by=['stay_id','label']).size().reset_index(name='count')
min_label_count_by_stay = label_count_by_stay.groupby(['stay_id'])['count'].min().reset_index(name='min')
stay_ids_above_min_count = min_label_count_by_stay[min_label_count_by_stay['min'] >= 5]['stay_id'].tolist()
preprocessed_events = events_with_removed_outliers[events_with_removed_outliers['stay_id'].isin(stay_ids_above_min_count)].reset_index(drop=True)
preprocessed_events = preprocessed_events.sort_values(['subject_id','stay_id','label','charttime'])


# Calculate the min and max chart times for each stay, and round these times to the nearest hour 
# based on a 30-minute threshold
preprocessed_events['charttime'] = pd.to_datetime(preprocessed_events['charttime'])

stay_time_range = preprocessed_events.groupby(['stay_id'])['charttime'].agg(['min', 'max']).reset_index()
stay_time_range['min_charttime'] = stay_time_range['min'].dt.floor('h').where(stay_time_range['min'].dt.minute <= 30,
                                                                    stay_time_range['min'].dt.ceil('h'))
stay_time_range['max_charttime'] = stay_time_range['max'].dt.floor('h').where(stay_time_range['max'].dt.minute <= 30,
                                                                    stay_time_range['max'].dt.ceil('h'))
events_time_range = pd.merge(preprocessed_events, stay_time_range[['stay_id','min_charttime','max_charttime']], on=['stay_id'], how='left')


def find_closest_times_to_rounded_hours(group, hour):

    rounded_group = pd.DataFrame()

    first_row = group.iloc[0][['min_charttime','max_charttime']]
    start_time = first_row['min_charttime']
    end_time = first_row['max_charttime']

    # Create timestamps at specified intervals (defined by 'hour') from start_time to end_time
    rounded_group['rounded_hours']= pd.date_range(start=start_time, end=end_time, freq=hour)

    # Find the index of the closest charttime for each rounded hour in the group
    closest_time_index = rounded_group['rounded_hours'].apply(
    lambda hour: (group['charttime'] - hour).abs().idxmin())

    # Calculate the minimum time difference between each rounded hour and the corresponding charttime
    closest_time_diff = rounded_group['rounded_hours'].apply(
        lambda hour: (group['charttime'] - hour).abs().min())

    # Define a time threshold (30/60 minutes) for selecting the closest charttime 
    if hour == 'h':
        mask_within_threshhold = closest_time_diff <= pd.Timedelta(minutes=30)
    else:
        mask_within_threshhold = closest_time_diff <= pd.Timedelta(minutes=60)

    
    # If no charttime is within the threshold, fill the valuenum with null
    common_columns_Null = ['subject_id', 'stay_id', 'label']
    first_row_values = group.loc[group.index[0], common_columns_Null].values
    rounded_group.loc[~mask_within_threshhold, common_columns_Null] = first_row_values

    #If the time difference between the closest charttime and rounded hour is within the threshold,
    # fill rows with the closest charttime's data
    common_columns = ['subject_id', 'stay_id', 'label','charttime', 'valuenum']
    rounded_group.loc[mask_within_threshhold, common_columns] = group.loc[closest_time_index, common_columns].reset_index(drop=True)

    rounded_group[['subject_id', 'stay_id']] = rounded_group[['subject_id', 'stay_id']].astype('int64')


    return rounded_group


# Convert the dataset to regular time intervals in the charttime column. 
# If a record is within the threshold, fill with its value; otherwise, set as null.
all_columns = events_time_range.columns
grouped_dataset = events_time_range.groupby(['stay_id', 'label'])[all_columns]

modified_dataset = grouped_dataset.apply(lambda group: find_closest_times_to_rounded_hours(group, 'h'))
modified_dataset.index = range(len(modified_dataset))
modified_dataset = modified_dataset[['subject_id','stay_id', 'label', 'rounded_hours', 'valuenum']]
sorted_dataset = modified_dataset.sort_values(by=['subject_id','stay_id','label','rounded_hours'])

sorted_dataset['valuenum'] = sorted_dataset['valuenum'].astype(object)
sorted_dataset['valuenum'] = sorted_dataset['valuenum'].fillna('Absent')

# Convert the categories in the label column into separate distinct columns
sorted_dataset_pivot = sorted_dataset.pivot_table(
                        index=['subject_id', 'stay_id', 'rounded_hours'],
                        columns='label',
                        values='valuenum',
                        aggfunc='first').reset_index()


# Revert 'Absent' back to NaN
pd.set_option('future.no_silent_downcasting', True)
sorted_dataset_pivot.replace('Absent', np.nan, inplace=True)


# Convert the data type of the values to float64
columns_to_convert = ['Heart Rate', 'Respiratory Rate','O2 saturation pulseoxymetry', 'Non Invasive Blood Pressure diastolic',
                      'Non Invasive Blood Pressure mean', 'Non Invasive Blood Pressure systolic']
sorted_dataset_pivot[columns_to_convert] = sorted_dataset_pivot[columns_to_convert].astype('float64')
sorted_dataset_pivot = sorted_dataset_pivot.sort_values(['subject_id','rounded_hours'])



sorted_dataset_pivot.to_csv('1h_vital.csv', index=False)