A variety of models were employed to evaluate the imputation process for medical time series, categorized based on their distinct methodological characteristics as follows:

1) 🧮 **Internal Imputation Models**: These models rely solely on the target column for imputation.
- Average and Median: These imputation methods are the most basic strategies, where the respective statistic from the training set is used to fill missing values in the test set.

- Average_H and Median_H: To improve imputation accuracy, we implemented hierarchical approaches—Average_H and Median_H—that prioritize patient-specific data by imputing missing values in a stepwise manner: first using statistics from the same ICU stay, then from all stays of the same patient, and finally from the overall test set.

- MediTHIM-A and MediTHIM-M:  To preserve temporal integrity, the historical variants MediTHIM-A and MediTHIM-M impute missing values using only past data, first from the same ICU stay, then from the patient’s previous records, and finally from overall training set statistics, ensuring a strictly time-aware imputation process.

  All the above approaches are implemented in the statistical_impute.py file. Here, `stat_impute` corresponds to Average/Median, `stat_hier_impute` to Average-H/Median-H, and `stat_hier_hist_impute` to MediTHIM-A/MediTHIM-M.

2) 🔗 **Feature-Dependent Imputation Models**: These methods use other features within the dataset to predict missing values in the target column.
- Linear Regression
- K-Nearest Neighbour
- Random Forest
- XGBoost
- MICE [1]

3) 🔄 **Multivariate Imputation Models**: These models leverage both the target feature and other variables within the dataset to perform imputation. In the context of time series data, deep learning approaches—particularly Recurrent Neural Networks (RNNs)—have gained significant popularity.
- GRU-D [2]
- BRITS [3]
  
  All of the above approaches are implemented in rnn_based_impute.py. You can specify the model by setting the 'rnn_model' variable to either brits or grud.

<img src="../Figures/Models.png" alt="Data Imputation Approaches" width="60%">

## References:
[1]. Azur, Melissa J., et al. "Multiple imputation by chained equations: what is it and how does it work?." International journal of methods in psychiatric research 20.1 (2011): 40-49.

[2]. Che, Zhengping, et al. "Recurrent neural networks for multivariate time series with missing values." Scientific reports 8.1 (2018): 6085.

[3]. Cao, Wei, et al. "Brits: Bidirectional recurrent imputation for time series." Advances in neural information processing systems 31 (2018).
