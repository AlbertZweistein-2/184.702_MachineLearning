# Implementation Notes for 12.12.

Benchmark Regression Tree:
- Compare best hyperparameter version 
   - Do Cross validation for custom tree and sklearn tree
DONE-> Function to use: multiple_custom_cross_validate() and insert all models (as initiated class instances) to test

Benchmark Random Forest:
- 3 Configs für Random Forest (Custom vs. Sklearn)
    - 1 x best hyperparameters
    - 1 x sklearn standard configs
    - 1 x frei wählbar
- Holdout Benchmark!
DONE-> Function to use: multiple_custom_cross_validate() and insert all models (as initiated class instances) to test
DONE-> Function to use: custom_evaluate_holdout_performance() and insert all models (as initiated class instances) to test

Compare all benchmark results via Cross Validation for best hyperparameters.
- Run one 5-fold CV evaluation on all models (Custom Tree, Custom Forest, Sklearn Tree, Sklearn Forest, Sklearn Linear Regression)
- Measuring RMSE and MAE and compare all models to each other. 
- Errors in CSV-Format
DONE -> Function to use for SKLEARN: multiple_sklearn_cross_validate()
DONE -> Function to use for CUSTOM: multiple_custom_cross_validate()

Runtime Benchmarks for all models on best standard sklearn parameters (Custom Tree, Custom Forest, Sklearn Tree, Sklearn Forest, Sklearn Linear Regression):
- Benchmark fit time on training data
- Benchmark prediction time on test data
- Runtime returned as pd df and saved as csv
DONE -> Times are measured in scope of the cross validation functions. Use those!

Visualisation
    -Runtime plot über 3 forest configs custom vs sklearn. für fit und predict
    -RMSE and MAE of each Custom Model vs Sklearn Model  (from cross val)
    -Runtime comparison Custom vs. Sklearn

