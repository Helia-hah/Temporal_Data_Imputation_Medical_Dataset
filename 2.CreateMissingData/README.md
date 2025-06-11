### Introducing Missingness
---
To ensure a comprehensive evaluation of the proposed approach, **eleven missing data ratios** were applied to each vital sign—specifically 1%, and from 5% to 50% in 5% increments. Additionally, **three distinct missingness patterns**—Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR)—were introduced for each vital sign.
- MCAR: The probability of missing data is independent of both observed and unobserved values, implying no systematic pattern behind the missingness.

- MAR: The missingness depends only on observed data and can be explained by variables present in the dataset. To simulate this, we identified the feature most correlated with the target and introduced missing values evenly from its smallest and largest values.

- MNAR: The probability of missingness depends on unobserved values, meaning the missingness is systematically related to the values that are themselves missing.

To generate MCAR and MNAR, the `produce_NA` function was used, with references to [this guide](https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values) and the [GitHub repository](https://github.com/BorisMuzellec/MissingDataOT)




