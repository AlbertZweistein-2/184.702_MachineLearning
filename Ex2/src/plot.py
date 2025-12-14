"""
Plotting utilities for comparing sklearn and custom model implementations.
All functions accept dataframes with flexible column and row names.
Consistent color scheme: sklearn = steelblue, custom = coral
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict


SKLEARN_COLOR = '#4682B4'
CUSTOM_COLOR = '#FF7F50'
TRAIN_COLOR = '#90EE90'
TEST_COLOR = '#FFB6C1'


def _find_matching_column(df: pd.DataFrame, patterns: List[str]):
    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern.lower() in col_lower:
                return col
    return None


def _prepare_comparison_data(
    sklearn_df: pd.DataFrame,
    custom_df: pd.DataFrame,
    metric_patterns: List[str],
    model_pairs: Dict[str, List[str]],
    sklearn_only_models: Optional[Dict[str, str]] = None
):
    sklearn_col = _find_matching_column(sklearn_df, metric_patterns)
    custom_col = _find_matching_column(custom_df, metric_patterns)
    
    if sklearn_col is None:
        raise ValueError(f"Could not find sklearn columns matching patterns: {metric_patterns}")
    
    data = []

    if sklearn_only_models:
        for display_name, sk_name in sklearn_only_models.items():
            if sk_name in sklearn_df.index:
                sklearn_val = abs(sklearn_df.loc[sk_name, sklearn_col])
                data.append({
                    'model_pair': display_name,
                    'sklearn': sklearn_val,
                    'custom': np.nan,  
                    'sklearn_name': sk_name,
                    'custom_name': None
                })

    for display_name, (sklearn_name, custom_name) in model_pairs.items():
        if sklearn_name in sklearn_df.index and custom_name in custom_df.index:
            sklearn_val = abs(sklearn_df.loc[sklearn_name, sklearn_col])
            custom_val = abs(custom_df.loc[custom_name, custom_col]) if custom_col else np.nan
            data.append({
                'model_pair': display_name,
                'sklearn': sklearn_val,
                'custom': custom_val,
                'sklearn_name': sklearn_name,
                'custom_name': custom_name
            })
    
    return pd.DataFrame(data)


def plot_cv_comparison(
    sklearn_cv_df: pd.DataFrame,
    custom_cv_df: pd.DataFrame,
    model_pairs: Dict[str, List[str]],
    sklearn_only_models: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    suptitle: Optional[str] = None
):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    plot_configs = [
        (['fit_time', 'avg_fit_time'], 'Fit Time (s)', True),
        (['score_time', 'avg_score_time'], 'Score Time (s)', True),
        (['test_rmse', 'avg_test_rmse'], 'Test RMSE', False),
        (['test_mae', 'avg_test_mae'], 'Test MAE', False),
    ]
    
    for idx, (patterns, label, log_scale) in enumerate(plot_configs):
        ax = axes[idx // 2, idx % 2]
        
        try:
            comp_data = _prepare_comparison_data(sklearn_cv_df, custom_cv_df, patterns, model_pairs, sklearn_only_models)
            
            if comp_data.empty:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            x = np.arange(len(comp_data))
            width = 0.35
            
            ax.bar(x - width/2, comp_data['sklearn'], width, label='sklearn', color=SKLEARN_COLOR)
            # Only plot custom where it exists
            custom_mask = ~comp_data['custom'].isna()
            ax.bar(x[custom_mask] + width/2, comp_data.loc[custom_mask, 'custom'], width, label='Custom', color=CUSTOM_COLOR)
            
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comp_data['model_pair'], rotation=30, ha='right', fontsize=10)
            ax.legend(fontsize=9)
            
            if log_scale and comp_data['sklearn'].min() > 0:
                ax.set_yscale('log')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)
    
    if suptitle is None:
        suptitle = 'Cross-Validation: sklearn vs Custom'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_holdout_comparison(
    sklearn_ho_df: pd.DataFrame,
    custom_ho_df: pd.DataFrame,
    model_pairs: Dict[str, List[str]],
    sklearn_only_models: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    suptitle: Optional[str] = None
):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    plot_configs = [
        (['fit_time'], 'Fit Time (s)', True),
        (['predict_time'], 'Predict Time (s)', True),
        (['rmse'], 'Test RMSE', False),
        (['mae'], 'Test MAE', False),
    ]
    
    for idx, (patterns, label, log_scale) in enumerate(plot_configs):
        ax = axes[idx // 2, idx % 2]
        
        try:
            comp_data = _prepare_comparison_data(sklearn_ho_df, custom_ho_df, patterns, model_pairs, sklearn_only_models)
            
            if comp_data.empty:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            x = np.arange(len(comp_data))
            width = 0.35
            
            ax.bar(x - width/2, comp_data['sklearn'], width, label='sklearn', color=SKLEARN_COLOR)

            custom_mask = ~comp_data['custom'].isna()
            ax.bar(x[custom_mask] + width/2, comp_data.loc[custom_mask, 'custom'], width, label='Custom', color=CUSTOM_COLOR)
            
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comp_data['model_pair'], rotation=30, ha='right', fontsize=10)
            ax.legend(fontsize=9)
            
            if log_scale and comp_data['sklearn'].min() > 0:
                ax.set_yscale('log')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error', ha='center', va='center', transform=ax.transAxes)
    
    if suptitle is None:
        suptitle = 'Holdout Evaluation: sklearn vs Custom (Test Set)'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_overfitting_comparison(
    sklearn_cv_df: pd.DataFrame,
    custom_cv_df: pd.DataFrame,
    model_pairs: Dict[str, List[str]],
    sklearn_only_models: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    suptitle: Optional[str] = None
):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    train_patterns = ['train_rmse', 'avg_train_rmse']
    test_patterns = ['test_rmse', 'avg_test_rmse']
    fit_patterns = ['fit_time', 'avg_fit_time']
    score_patterns = ['score_time', 'avg_score_time']
    
    train_col_sk = _find_matching_column(sklearn_cv_df, train_patterns)
    test_col_sk = _find_matching_column(sklearn_cv_df, test_patterns)
    train_col_cu = _find_matching_column(custom_cv_df, train_patterns)
    test_col_cu = _find_matching_column(custom_cv_df, test_patterns)
    fit_col_sk = _find_matching_column(sklearn_cv_df, fit_patterns)
    fit_col_cu = _find_matching_column(custom_cv_df, fit_patterns)
    score_col_sk = _find_matching_column(sklearn_cv_df, score_patterns)
    score_col_cu = _find_matching_column(custom_cv_df, score_patterns)

    sk_models = []
    sk_train = []
    sk_test = []
    
    if sklearn_only_models:
        for display_name, sk_name in sklearn_only_models.items():
            if sk_name in sklearn_cv_df.index:
                sk_models.append(display_name)
                sk_train.append(abs(sklearn_cv_df.loc[sk_name, train_col_sk]))
                sk_test.append(abs(sklearn_cv_df.loc[sk_name, test_col_sk]))
    
    for display_name, (sk_name, cu_name) in model_pairs.items():
        if sk_name in sklearn_cv_df.index:
            sk_models.append(display_name)
            sk_train.append(abs(sklearn_cv_df.loc[sk_name, train_col_sk]))
            sk_test.append(abs(sklearn_cv_df.loc[sk_name, test_col_sk]))

    cu_models = []
    cu_train = []
    cu_test = []
    
    for display_name, (sk_name, cu_name) in model_pairs.items():
        if cu_name in custom_cv_df.index:
            cu_models.append(display_name)
            cu_train.append(abs(custom_cv_df.loc[cu_name, train_col_cu]))
            cu_test.append(abs(custom_cv_df.loc[cu_name, test_col_cu]))

    speedup_models = []
    fit_sk = []
    fit_cu = []
    score_sk = []
    score_cu = []
    
    for display_name, (sk_name, cu_name) in model_pairs.items():
        if sk_name in sklearn_cv_df.index and cu_name in custom_cv_df.index:
            speedup_models.append(display_name)
            fit_sk.append(abs(sklearn_cv_df.loc[sk_name, fit_col_sk]))
            fit_cu.append(abs(custom_cv_df.loc[cu_name, fit_col_cu]))
            score_sk.append(abs(sklearn_cv_df.loc[sk_name, score_col_sk]))
            score_cu.append(abs(custom_cv_df.loc[cu_name, score_col_cu]))
    
    width = 0.35
    
    # Top Left: sklearn overfitting
    ax1 = axes[0, 0]
    x1 = np.arange(len(sk_models))
    ax1.bar(x1 - width/2, sk_train, width, label='Train RMSE', color=TRAIN_COLOR)
    ax1.bar(x1 + width/2, sk_test, width, label='Test RMSE', color=TEST_COLOR)
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_title('sklearn: Train vs Test RMSE', fontsize=12, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(sk_models, rotation=30, ha='right', fontsize=10)
    ax1.legend(fontsize=9)
    
    # Top Right: custom overfitting
    ax2 = axes[0, 1]
    x2 = np.arange(len(cu_models))
    ax2.bar(x2 - width/2, cu_train, width, label='Train RMSE', color=TRAIN_COLOR)
    ax2.bar(x2 + width/2, cu_test, width, label='Test RMSE', color=TEST_COLOR)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('Custom: Train vs Test RMSE', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cu_models, rotation=30, ha='right', fontsize=10)
    ax2.legend(fontsize=9)
    
    # Bottom Left: Fit Time Speedup (ratio custom/sklearn)
    ax3 = axes[1, 0]
    x3 = np.arange(len(speedup_models))
    fit_ratios = [cu / sk if sk > 0 else 0 for sk, cu in zip(fit_sk, fit_cu)]
    bars3 = ax3.bar(x3, fit_ratios, width=0.6, color=CUSTOM_COLOR)
    ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='Equal (1x)')
    ax3.set_ylabel('Slowdown Factor (Custom / sklearn)', fontsize=10)
    ax3.set_title('Fit Time: Custom Slowdown Factor', fontsize=12, fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(speedup_models, rotation=30, ha='right', fontsize=10)
    ax3.set_yscale('log')
    # Add value labels
    for bar, ratio in zip(bars3, fit_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{ratio:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Extend y-axis to make room for labels
    ax3.set_ylim(top=ax3.get_ylim()[1] * 2)
    
    # Bottom Right: Score Time Speedup (ratio custom/sklearn)
    ax4 = axes[1, 1]
    x4 = np.arange(len(speedup_models))
    score_ratios = [cu / sk if sk > 0 else 0 for sk, cu in zip(score_sk, score_cu)]
    bars4 = ax4.bar(x4, score_ratios, width=0.6, color=CUSTOM_COLOR)
    ax4.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='Equal (1x)')
    ax4.set_ylabel('Slowdown Factor (Custom / sklearn)', fontsize=10)
    ax4.set_title('Score Time: Custom Slowdown Factor', fontsize=12, fontweight='bold')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(speedup_models, rotation=30, ha='right', fontsize=10)
    ax4.set_yscale('log')
    # Add value labels
    for bar, ratio in zip(bars4, score_ratios):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{ratio:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Extend y-axis to make room for labels
    ax4.set_ylim(top=ax4.get_ylim()[1] * 2)
    
    if suptitle is None:
        suptitle = 'Overfitting Analysis & Time Comparison'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig
