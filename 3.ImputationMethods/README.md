A variety of models were employed to evaluate the imputation process for medical time series, categorized based on their distinct methodological characteristics as follows:
🧮 Internal Imputation Models: These models rely solely on the target column for imputation, using its existing information.
- Average and Median: These imputation methods are the most basic strategies, where the respective statistic from the training set is used to fill missing values in the test set.

- Average_H and Median_H: To improve imputation accuracy, we implemented hierarchical approaches—Average_H and Median_H—that prioritize patient-specific data by imputing missing values in a stepwise manner: first using statistics from the same ICU stay, then from all stays of the same patient, and finally from the overall test set.

- MediTHIM-A and MediTHIM-M:  To preserve temporal integrity, the historical variants MediTHIM-A and MediTHIM-M impute missing values using only past data—first from the same ICU stay, then from the patient’s previous records, and finally from overall training set statistics—ensuring a strictly time-aware imputation process.
