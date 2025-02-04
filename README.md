# Temporal Data Imputation for Medical Datasets

This repository presents a novel statistical hierarchical approach for imputing missing values in temporal medical datasets. For time series imputation in this research, the following information about ICU patients was extracted from the MIMIC-IV dataset:

- Six routine vital signs: heart rate (**HR**), respiratory rate (**RR**), oxygen saturation (pulse oximetry) (**O2SP**), and non-invasive blood pressure—diastolic (**NIBPD**), mean (**NIBPM**), and systolic (**NIBPS**).
  
- Two Types of Demographic Information: Gender and age.

- Three Physical Characteristics: Height, weight, and body mass index (BMI).

- Administered Medications.


### Data Preprocessing
---

- **Outlier Removal:** Outliers in demographic data, vital signs, and physical characteristics were removed based on the following value ranges:

  <table style="font-size: 12px; width: 60%; margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border: 1px solid black; padding: 5px;">Variable</th>
      <th style="border: 1px solid black; padding: 5px;">Min Value</th>
      <th style="border: 1px solid black; padding: 5px;">Max Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black; padding: 5px;"><strong>Heart Rate</strong></td>
      <td style="border: 1px solid black; padding: 5px;">-3</td>
      <td style="border: 1px solid black; padding: 5px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;"><strong>Respiratory Rate</strong></td>
      <td style="border: 1px solid black; padding: 5px;">-3</td>
      <td style="border: 1px solid black; padding: 5px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;"><strong>Oxygen Saturation</strong></td>
      <td style="border: 1px solid black; padding: 5px;">-1</td>
      <td style="border: 1px solid black; padding: 5px;">101</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;"><strong>Non-invasive BP diastolic</strong></td>
      <td style="border: 1px solid black; padding: 5px;">-5</td>
      <td style="border: 1px solid black; padding: 5px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px;"><strong>Non-invasive BP systolic</strong></td>
      <td style="border: 1px solid black; padding: 5px;">-5</td>
      <td style="border: 1px solid black; padding: 5px;">505</td>
    </tr>
    <tr>
      <td style


| Variable | Min Value | Max Value |
|------------------------------|-----------|-----------| 
| **Heart Rate** | -3 | 303 | 
| **Respiratory Rate** | -3 | 303 | 
| **Oxygen Saturation** | -1 | 101 | 
| **Non-invasive BP diastolic** | -5 | 505 | 
| **Non-invasive BP systolic** | -5 | 505 | 
| **Non-invasive BP mean** | -5 | 505 | 
| **Weight** | 40 | 1435 | 
| **Height** | 20 | 110 |
| **BMI** | 5 | 190 |


