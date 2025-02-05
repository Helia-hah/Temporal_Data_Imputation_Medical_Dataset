# Temporal Data Imputation for Medical Datasets

This repository presents a novel statistical hierarchical approach for imputing missing values in temporal medical datasets. For time series imputation in this research, the following information about ICU patients was extracted from the MIMIC-IV dataset:

- Six routine vital signs: heart rate (**HR**), respiratory rate (**RR**), oxygen saturation (pulse oximetry) (**O2SP**), and non-invasive blood pressure—diastolic (**NIBPD**), systolic (**NIBPS**), and mean (**NIBPM**).
  
- Two Types of Demographic Information: Gender and age.

- Three Physical Characteristics: Height, weight, and body mass index (BMI).

- Administered Medications.


### Data Preprocessing
___

- **Outlier Removal:** Outliers in demographic data, vital signs, and physical characteristics were removed based on the following value ranges:
- **Timestamp Standardization:**
  - Timestamps for the six vital signs were rounded to the nearest hour with a 30-minute threshold applied(If no record was available for a given time, it was replaced with null values).
  - Data were aggregated at one-hour intervals for consistency across patients.








  





