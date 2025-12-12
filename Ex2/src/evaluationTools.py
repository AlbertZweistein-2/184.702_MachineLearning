import numpy as np
import pandas as pd
from config import SCORING_METRICS, rmse, mae
from sklearn.model_selection import cross_validate
import time

def sklearn_evaluate_holdout_performance(model_pipelines:dict, X_train, y_train, X_test, y_test, metrics:list = ['RMSE', 'MAE']):
    """
    Evaluates multiple sklearn models on test data using specified metrics.
    """
    all_results = {}
    for model_name, model_pipeline in model_pipelines.items():
        print(f"Evaluating holdout performance for model: {model_name}...")
        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)
        results = {}
        
        if 'RMSE' in metrics:
            mse = np.mean((y_test - predictions) ** 2)
            rmse_value = np.sqrt(mse)
            results['RMSE'] = rmse_value
            
        if 'MAE' in metrics:
            mae_value = np.mean(np.abs(y_test - predictions))
            results['MAE'] = mae_value
        all_results[model_name] = results
    return pd.DataFrame.from_dict(all_results, orient='index')

def multiple_sklearn_cross_validate(model_pipelines:dict, X, y, cv_folds=5, scoring = SCORING_METRICS):
    """
    Wrapper around sklearn's cross_validate to return results for multiple models.
    """
    all_results = {}
    for model_name, model_pipeline in model_pipelines.items():
        print(f"Starting sklearn {cv_folds}-Fold cross validation for model: {model_name}...")
        df_results = sklearn_cross_validate(
            model_pipeline,
            X,
            y,
            cv_folds=cv_folds,
            scoring=scoring
        )
        all_results[model_name] = df_results
    #return dataframe with averaged results
    averaged_results = {}
    for model_name, df_results in all_results.items():
        averaged_results[model_name] = df_results.mean().to_dict()
        #ad avg_ prefix to keys
        averaged_results[model_name] = {f"avg_{k}": v for k, v in averaged_results[model_name].items()}
    return pd.DataFrame(averaged_results).T



def sklearn_cross_validate(model_pipeline, X, y, cv_folds=5, scoring = SCORING_METRICS):
    """
    Wrapper around sklearn's cross_validate to return results.
    """
    print(f"Starting sklearn {cv_folds}-Fold cross validation...")
    cv_results = cross_validate(
        model_pipeline,
        X,
        y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    #return pandas row dataframe
    return pd.DataFrame(cv_results)

def multiple_custom_cross_validate(model_pipelines:dict, X, y, target_col, k=5):
    """
    Wrapper around custom_cross_validate to return results for multiple models.
    """
    all_results = {}
    for model_name, model_pipeline in model_pipelines.items():
        print(f"Starting custom {k}-Fold cross validation for model: {model_name}...")
        df_results = custom_cross_validate(
            model_pipeline.__class__,
            X,
            y,
            target_col=target_col,
            k=k,
            model_params=model_pipeline.get_params()
        )
        all_results[model_name] = df_results
    #return dataframe with averaged results
    averaged_results = {}
    for model_name, df_results in all_results.items():
        averaged_results[model_name] = df_results.mean().to_dict()
        #ad avg_ prefix to keys
        averaged_results[model_name] = {f"avg_{k}": v for k, v in averaged_results[model_name].items()}
    return pd.DataFrame(averaged_results).T

def custom_cross_validate(model, X, y, k=5):
    """
    Führt k-Fold CV durch und berechnet Train/Test Scores für mehrere Metriken.
    """
        
    #combine X and y into a single DataFrame for easier indexing
    df = X.copy()
    target_col = y.name
    df[target_col] = y

    # Shuffle indices and create folds
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    # Results Dictionary 
    results = {
        'train_rmse': [], 'test_rmse': [],
        'train_mae': [], 'test_mae': []
    }
    
    print(f"Starting {k}-Fold cross validation...")
    
    for i in range(k):
        print(f"Processing fold {i+1}/{k}...")
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
        start_fit=time.time()
        
        model.fit(X_train, y_train)
        
        duration_fit=time.time()-start_fit
        # --- PREDICTIONS ---
        # Important: We also test on the training data
        pred_train = model.predict(X_train)
        start_predict=time.time()
        pred_test = model.predict(X_test)
        duration_score=time.time()-start_predict
        
        # --- CALCULATE METRICS ---
        
        # 1. MSE (Mean Squared Error)
        mse_train = np.mean((y_train - pred_train) ** 2)
        mse_test = np.mean((y_test - pred_test) ** 2)
        
        # 2. RMSE (Root Mean Squared Error) - better interpretable, as it has the same unit as the target
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        
        
        # 4. MAE (Mean Absolute Error)
        mae_train = np.mean(np.abs(y_train - pred_train))
        mae_test = np.mean(np.abs(y_test - pred_test))
        
        # Save
        results['train_rmse'].append(rmse_train)
        results['test_rmse'].append(rmse_test)
        results['train_mae'].append(mae_train)
        results['test_mae'].append(mae_test)
        results['fit_time'] = duration_fit
        results['score_time'] = duration_score
        
        print(f"Fold {i+1}/{k} | Test RMSE: {rmse_test:.2f} | Train RMSE: {rmse_train:.2f}")
    return pd.DataFrame(results)





def custom_evaluate_holdout_performance(models:dict, X_train, y_train, X_test, y_test):
    """
    Evaluates multiple custom models on test data using RMSE and MAE metrics.
    """
    all_results = {}
    for model_name, model in models.items():
        print(f"Evaluating holdout performance for model: {model_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # RMSE
        mse = np.mean((y_test - predictions) ** 2)
        rmse_value = np.sqrt(mse)
        
        # MAE
        mae_value = np.mean(np.abs(y_test - predictions))
        
        all_results[model_name] = {
            'RMSE': rmse_value,
            'MAE': mae_value
        }
    return pd.DataFrame.from_dict(all_results, orient='index')
    




__all__ = [
    "custom_cross_validate"
    ]