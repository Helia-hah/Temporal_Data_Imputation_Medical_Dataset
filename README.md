# Temporal Data Imputation for Medical Datasets

This repository presents a novel statistical hierarchical approach for imputing missing values in temporal medical datasets. For time series imputation in this research, the following information about ICU patients was extracted from the MIMIC-IV dataset:

- Six routine vital signs: heart rate (**HR**), respiratory rate (**RR**), oxygen saturation (pulse oximetry) (**O2SP**), and non-invasive blood pressure—diastolic (**NIBPD**), systolic (**NIBPS**), and mean (**NIBPM**).
  
- Two Types of Demographic Information: Gender and age.

- Three Physical Characteristics: Height, weight, and body mass index (BMI).

- Administered Medications.


### Data Preprocessing
___

- **Outlier Removal:** Outliers in demographic data, vital signs, and physical characteristics were removed based on the following value ranges:

  <table style="font-size: 8px; line-height: 1; width: 50%; margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border: 1px solid black; padding: 1px; text-align: left;">Variable</th>
      <th style="border: 1px solid black; padding: 1px; text-align: left;">Min Value</th>
      <th style="border: 1px solid black; padding: 1px; text-align: left;">Max Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Heart Rate</td>
      <td style="border: 1px solid black; padding: 1px;">-3</td>
      <td style="border: 1px solid black; padding: 1px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Respiratory Rate</td>
      <td style="border: 1px solid black; padding: 1px;">-3</td>
      <td style="border: 1px solid black; padding: 1px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Oxygen Saturation</td>
      <td style="border: 1px solid black; padding: 1px;">-1</td>
      <td style="border: 1px solid black; padding: 1px;">101</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Non-invasive BP diastolic</td>
      <td style="border: 1px solid black; padding: 1px;">-5</td>
      <td style="border: 1px solid black; padding: 1px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Non-invasive BP systolic</td>
      <td style="border: 1px solid black; padding: 1px;">-5</td>
      <td style="border: 1px solid black; padding: 1px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Non-invasive BP mean</td>
      <td style="border: 1px solid black; padding: 1px;">-5</td>
      <td style="border: 1px solid black; padding: 1px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Weight</td>
      <td style="border: 1px solid black; padding: 1px;">40</td>
      <td style="border: 1px solid black; padding: 1px;">1435</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">Height</td>
      <td style="border: 1px solid black; padding: 1px;">20</td>
      <td style="border: 1px solid black; padding: 1px;">110</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 1px;">BMI</td>
      <td style="border: 1px solid black; padding: 1px;">5</td>
      <td style="border: 1px solid black; padding: 1px;">190</td>
    </tr>
  </tbody>
</table>

- Timestamp Standardization:
- Timestamps for the six vital signs were rounded to the nearest hour with a 30-minute threshold applied(If no record was available for a given time, it was replaced with null values).
- Data were aggregated at one-hour intervals for consistency across patients.








  





