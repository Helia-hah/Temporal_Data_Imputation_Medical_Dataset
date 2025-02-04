# Temporal Data Imputation for Medical Datasets

This repository presents a novel statistical hierarchical approach for imputing missing values in temporal medical datasets. For time series imputation in this research, the following information about ICU patients was extracted from the MIMIC-IV dataset:

- Six routine vital signs: heart rate (**HR**), respiratory rate (**RR**), oxygen saturation (pulse oximetry) (**O2SP**), and non-invasive blood pressure—diastolic (**NIBPD**), mean (**NIBPM**), and systolic (**NIBPS**).
  
- Two Types of Demographic Information: Gender and age.

- Three Physical Characteristics: Height, weight, and body mass index (BMI).

- Administered Medications.


### Data Preprocessing
___

- **Outlier Removal:** Outliers in demographic data, vital signs, and physical characteristics were removed based on the following value ranges:

  <table style="font-size: 10px; line-height: 1.2; width: 60%; margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border: 1px solid black; padding: 2px; text-align: left;">Variable</th>
      <th style="border: 1px solid black; padding: 2px; text-align: left;">Min Value</th>
      <th style="border: 1px solid black; padding: 2px; text-align: left;">Max Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Heart Rate</td>
      <td style="border: 1px solid black; padding: 2px;">-3</td>
      <td style="border: 1px solid black; padding: 2px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Respiratory Rate</td>
      <td style="border: 1px solid black; padding: 2px;">-3</td>
      <td style="border: 1px solid black; padding: 2px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Oxygen Saturation</td>
      <td style="border: 1px solid black; padding: 2px;">-1</td>
      <td style="border: 1px solid black; padding: 2px;">101</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Non-invasive BP diastolic</td>
      <td style="border: 1px solid black; padding: 2px;">-5</td>
      <td style="border: 1px solid black; padding: 2px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Non-invasive BP systolic</td>
      <td style="border: 1px solid black; padding: 2px;">-5</td>
      <td style="border: 1px solid black; padding: 2px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Non-invasive BP mean</td>
      <td style="border: 1px solid black; padding: 2px;">-5</td>
      <td style="border: 1px solid black; padding: 2px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Weight</td>
      <td style="border: 1px solid black; padding: 2px;">40</td>
      <td style="border: 1px solid black; padding: 2px;">1435</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">Height</td>
      <td style="border: 1px solid black; padding: 2px;">20</td>
      <td style="border: 1px solid black; padding: 2px;">110</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 2px;">BMI</td>
      <td style="border: 1px solid black; padding: 2px;">5</td>
      <td style="border: 1px solid black; padding: 2px;">190</td>
    </tr>
  </tbody>
</table>




  





