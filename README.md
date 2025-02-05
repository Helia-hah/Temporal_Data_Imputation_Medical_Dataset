# Temporal Data Imputation for Medical Datasets

This repository presents a novel statistical hierarchical approach for imputing missing values in temporal medical datasets. For time series imputation in this research, the following information about ICU patients was extracted from the MIMIC-IV dataset:

- Six routine vital signs: heart rate (**HR**), respiratory rate (**RR**), oxygen saturation (pulse oximetry) (**O2SP**), and non-invasive blood pressure—diastolic (**NIBPD**), systolic (**NIBPS**), and mean (**NIBPM**).
  
- Two Types of Demographic Information: Gender and age.

- Three Physical Characteristics: Height, weight, and body mass index (BMI).

- Administered Medications.


### Data Preprocessing
---


- **Outlier Removal:** Outliers in demographic data, vital signs, and physical characteristics were removed based on the following value ranges:
    | Variable                   | Min Value | Max Value |
  |----------------------------|----------|----------|
  | Heart Rate                | -3       | 303      |
  | Respiratory Rate          | -3       | 303      |
  | Oxygen Saturation         | -1       | 101      |
  | Non-invasive BP diastolic | -5       | 505      |
  | Non-invasive BP systolic  | -5       | 505      |
  | Non-invasive BP mean      | -5       | 505      |
  | Weight                    | 40       | 1435     |
  | Height                    | 20       | 110      |
  | BMI                       | 5        | 190      |

- **Timestamp Standardization:**
  - Timestamps for the six vital signs were rounded to the nearest hour with a 30-minute threshold applied (If no record was available for a given time, it was replaced with null values). Data were aggregated at one-hour intervals for consistency across patients.
  - To standardize medication administration timing, start and end times were rounded to the nearest hour using a 30-minute threshold. New records were created for each medication at each rounded hour within the administration period.
 
- **Medication Categorization:** Medications were grouped into 16 main categories and encoded in binary format (1 for received, 0 for not received).
 
- **Physical Characteristics Preprocessing**:
  - Height values were averaged across all stays (All patients were adults, so height changes were negligible).
  - Weight values from the medication table were averaged hourly and smoothed using a 24-hour rolling mean to account for fluctuations. The resulting values were then averaged with weights from the OMR table.
  - Update BMI using the new height and weight values.








  





