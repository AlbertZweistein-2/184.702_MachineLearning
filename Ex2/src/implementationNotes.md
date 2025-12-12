# Implementation Notes for 12.12.

Benchmark Regression Tree:
- Compare best hyperparameter version 
   - Do Cross validation for custom tree and sklearn tree

Benchmark Random Forest:
- 3 Configs für Random Forest (Custom vs. Sklearn)
    - 1 x best hyperparameters
    - 1 x sklearn standard configs
    - 1 x frei wählbar
- Holdout Benchmark!

Compare all benchmark results via Cross Validation for best hyperparameters.
- Run one 5-fold CV evaluation on all models (Custom Tree, Custom Forest, Sklearn Tree, Sklearn Forest, Sklearn Linear Regression)
- Measuring RMSE and MAE and compare all models to each other. 
- Errors in CSV-Format

Runtime Benchmarks for all models on best standard sklearn parameters (Custom Tree, Custom Forest, Sklearn Tree, Sklearn Forest, Sklearn Linear Regression):
- Benchmark fit time on training data
- Benchmark prediction time on test data
- Runtime returned as pd df and saved as csv

Visualisation
    -RMSE and MAE of each Custom Model vs Sklearn Model 
    -Runtime comparison Custom vs. Sklearn

