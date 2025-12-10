import numpy as np
import pandas as pd

def custom_cross_validate(model_class, df, target_col, k=5, model_params=None):
    """
    Führt k-Fold CV durch und berechnet Train/Test Scores für mehrere Metriken.
    """
    if model_params is None:
        model_params = {}

    # Shuffle indices and create folds
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    # Results Dictionary 
    results = {
        'train_mse': [], 'test_mse': [],
        'train_rmse': [], 'test_rmse': [],
        'train_r2': [], 'test_r2': [],
        'train_mae': [], 'test_mae': []
    }
    
    print(f"Starting {k}-Fold cross validation...")
    
    for i in range(k):
        test_idx = folds[i]
        train_folds = [folds[j] for j in range(k) if j != i]
        train_idx = np.concatenate(train_folds)
        
        # Split train and test data
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # --- PREDICTIONS ---
        # Important: We also test on the training data
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        # --- CALCULATE METRICS ---
        
        # 1. MSE (Mean Squared Error)
        mse_train = np.mean((y_train - pred_train) ** 2)
        mse_test = np.mean((y_test - pred_test) ** 2)
        
        # 2. RMSE (Root Mean Squared Error) - better interpretable, as it has the same unit as the target
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        
        # 3. R^2 (Coefficient of Determination) - how well does the model explain the variance? (1.0 is perfect)
        # Formula: 1 - (Sum of squared residuals / Sum of squared deviations from the mean)
        ss_res_train = np.sum((y_train - pred_train) ** 2)
        ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
        r2_train = 1 - (ss_res_train / ss_tot_train) if ss_tot_train != 0 else 0
        
        ss_res_test = np.sum((y_test - pred_test) ** 2)
        ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_test = 1 - (ss_res_test / ss_tot_test) if ss_tot_test != 0 else 0
        
        # 4. MAE (Mean Absolute Error)
        mae_train = np.mean(np.abs(y_train - pred_train))
        mae_test = np.mean(np.abs(y_test - pred_test))
        
        # Save
        results['train_mse'].append(mse_train)
        results['test_mse'].append(mse_test)
        results['train_rmse'].append(rmse_train)
        results['test_rmse'].append(rmse_test)
        results['train_r2'].append(r2_train)
        results['test_r2'].append(r2_test)
        results['train_mae'].append(mae_train)
        results['test_mae'].append(mae_test)
        
        print(f"Fold {i+1}/{k} | Test RMSE: {rmse_test:.2f} | Train RMSE: {rmse_train:.2f}")
    return pd.DataFrame(results)

__all__ = ["custom_cross_validate"]