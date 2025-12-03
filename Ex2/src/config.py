from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from scipy.stats import randint, uniform

# --- Reproducibility ---
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

# --- Metrics used for evaluation (must include >=2) ---
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

SCORING_METRICS = {
    "rmse": make_scorer(rmse, greater_is_better=False),
    "mae": make_scorer(mae, greater_is_better=False),
}

# --- Baseline Models (sklearn) ---
BASELINE_MODELS = {
    "linear": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
    "random_forest": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=12),
}

# --- Hyperparameter Search Spaces ---
GRID_DECISION_TREE = {
    "model__max_depth": [3, 5, 10, 20, 30, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4],
    "model__criterion": ["squared_error"],
}

GRID_RANDOM_FOREST = {
    "model__n_estimators": [100, 500, 1000, 1500],
    "model__max_depth": [5, 10, 50, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__criterion": ["squared_error"],
}

# Randomized-search spaces
RAND_DECISION_TREE = {
    "model__max_depth": [None] + list(range(3, 60)),
    "model__min_samples_split": randint(2, 25),
    "model__min_samples_leaf": randint(1, 10),
    "model__criterion": ["squared_error"],
}

RAND_RANDOM_FOREST = {
    "model__n_estimators": randint(200, 1600),
    "model__max_depth": [None] + list(range(3, 60)),
    "model__min_samples_split": randint(2, 25),
    "model__min_samples_leaf": randint(1, 10),
    "model__max_features": ["sqrt", "log2", None],
    "model__bootstrap": [True],
    "model__criterion": ["squared_error"],
}

PARAM_GRIDS = {
    "decision_tree": GRID_DECISION_TREE,
    "random_forest": GRID_RANDOM_FOREST,
}

PARAM_DISTS = {
    "decision_tree": RAND_DECISION_TREE,
    "random_forest": RAND_RANDOM_FOREST,
}
__all__ = [
    "RANDOM_SEED",
    "TEST_SIZE",
    "CV_FOLDS",
    "rmse",
    "mae",
    "BASELINE_MODELS",
    "GRID_DECISION_TREE",
    "GRID_RANDOM_FOREST",
    "PARAM_GRIDS",
    "SCORING_METRICS",
    "RAND_DECISION_TREE",
    "RAND_RANDOM_FOREST",
    "PARAM_DISTS"
]