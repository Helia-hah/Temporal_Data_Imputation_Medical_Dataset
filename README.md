# Temporal Data Imputation for Medical Datasets

This repository presents a novel statistical hierarchical approach for imputing missing values in temporal medical datasets. For time series imputation in this research, the following information about ICU patients was extracted from the MIMIC-IV dataset:

- Six routine vital signs: heart rate (**HR**), respiratory rate (**RR**), oxygen saturation (pulse oximetry) (**O2SP**), and non-invasive blood pressure—diastolic (**NIBPD**), mean (**NIBPM**), and systolic (**NIBPS**).
  
- Two Types of Demographic Information: Gender and age.

- Three Physical Characteristics: Height, weight, and body mass index (BMI).

- Administered Medications.


### Data Preprocessing
___

- **Outlier Removal:** Outliers in demographic data, vital signs, and physical characteristics were removed based on the following value ranges:

  <table style="font-size: 8px; width: 60%; margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border: 1px solid black; padding: 5px; width: 25%;">Variable</th> <!-- Adjusted width of the first column -->
      <th style="border: 1px solid black; padding: 5px;">Min Value</th>
      <th style="border: 1px solid black; padding: 5px;">Max Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Heart Rate</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">-3</td>
      <td style="border: 1px solid black; padding: 5px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Respiratory Rate</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">-3</td>
      <td style="border: 1px solid black; padding: 5px;">303</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Oxygen Saturation</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">-1</td>
      <td style="border: 1px solid black; padding: 5px;">101</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Non-invasive BP diastolic</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">-5</td>
      <td style="border: 1px solid black; padding: 5px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Non-invasive BP systolic</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">-5</td>
      <td style="border: 1px solid black; padding: 5px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Non-invasive BP mean</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">-5</td>
      <td style="border: 1px solid black; padding: 5px;">505</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Weight</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">40</td>
      <td style="border: 1px solid black; padding: 5px;">1435</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>Height</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">20</td>
      <td style="border: 1px solid black; padding: 5px;">110</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 5px; width: 25%;"><strong>BMI</strong></td> <!-- Adjusted width of the first column -->
      <td style="border: 1px solid black; padding: 5px;">5</td>
      <td style="border: 1px solid black; padding: 5px;">190</td>
    </tr>
  </tbody>
</table>



  





