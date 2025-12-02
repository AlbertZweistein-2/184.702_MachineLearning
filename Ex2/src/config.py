from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Reproducibility ---
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Metrics used for evaluation (must include >=2) ---
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# --- Baseline Models (sklearn) ---
BASELINE_MODELS = {
    "linear": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
    "random_forest": RandomForestRegressor(random_state=RANDOM_SEED),
    "svr": SVR(),
}

# --- Hyperparameter Search Spaces ---
GRID_DECISION_TREE = {
    "model__max_depth": [3, 5, 10, None],
    "model__min_samples_split": [2, 5, 10],
}

GRID_RANDOM_FOREST = {
    "model__n_estimators": [50, 100, 300],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5, 10],
}

GRID_SVR = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["rbf", "linear"]
}
