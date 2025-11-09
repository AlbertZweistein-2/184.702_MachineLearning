from itertools import product
from typing import Optional, Dict, Any, List

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score


# Scoring dictionaries
scoring_multiclass = {
    'accuracy': 'accuracy',
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
}

scoring_binary = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
}


def _xgb_estimator_for_task(task: str) -> XGBClassifier:
    task = task.lower()
    if task == 'binary':
        return XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
        )
    # default: multiclass
    return XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42,
        device='cuda'
    )


def get_classifier_configs(task: str = 'multiclass', classifier_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return classifier configs (model + param_grid) for a given task ('binary' or 'multiclass')."""
    task = task.lower()

    configs: Dict[str, Dict[str, Any]] = {
        # 'KNN': {
        #     'model': KNeighborsClassifier(),
        #     'param_grid': {
        #         'n_neighbors': [2 ,3, 6, 15],
        #         'weights': ['uniform', 'distance'],
        #         # 'p': [1, 2],
        #     },
        # },
        # 'Random Forest': {
        #     'model': RandomForestClassifier(
        #         max_features=None,
        #         bootstrap=True,
        #         criterion='entropy',
        #         oob_score=True,
        #         n_jobs=-1,
        #         n_estimators=1000,
        #         # random_state=42,
        #     ),
        #     'param_grid': {
        #         'criterion': ['entropy', 'gini', 'log_loss'],
        #         'max_features': [None, 'sqrt', 'log2'],
        #     },
        # },
        #  'Random Forest_adapt': {
        #     'model': RandomForestClassifier(
        #         max_features=None,
        #         bootstrap=True,
        #         criterion='entropy',
        #         oob_score=True,
        #         n_jobs=-1,
        #         n_estimators=200,
        #         # random_state=42,
        #     ),
        #     'param_grid': {
        #         'criterion': ['entropy', 'gini', 'log_loss'],
        #         'max_features': [None, 'sqrt', 'log2'],
        #     },
        #},
        'XGBoost': {
            'model': _xgb_estimator_for_task(task),
            'param_grid': {
                'n_estimators': [100, 500, 1000, 1500],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [5, 10, 15, 20],
            },
        },
    }

    if classifier_name:
        return {classifier_name: configs[classifier_name]}
    return configs


def get_all_classifier_names(task: str = 'multiclass') -> List[str]:
    return list(get_classifier_configs(task).keys())


def get_classifier_param_grid(classifier_name: str, task: str = 'multiclass') -> Dict[str, Any]:
    return get_classifier_configs(task)[classifier_name]['param_grid']


def _expand_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a param_grid into list of param dicts."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*(param_grid[k] for k in keys))]


def get_knn_classifiers(task: str = 'multiclass') -> List[KNeighborsClassifier]:
    cfg = get_classifier_configs(task)['KNN']
    combos = _expand_param_grid(cfg['param_grid'])
    return [KNeighborsClassifier(**p) for p in combos]


def get_rf_classifiers(task: str = 'multiclass') -> List[RandomForestClassifier]:
    cfg = get_classifier_configs(task)['Random Forest']
    base: RandomForestClassifier = cfg['model']
    combos = _expand_param_grid(cfg['param_grid'])
    return [RandomForestClassifier(**{**base.get_params(), **p}) for p in combos]


def get_xgb_classifiers(task: str = 'multiclass') -> List[XGBClassifier]:
    cfg = get_classifier_configs(task)['XGBoost']
    base: XGBClassifier = cfg['model']
    combos = _expand_param_grid(cfg['param_grid'])
    # no deprecated use_label_encoder
    return [XGBClassifier(**{**base.get_params(), **p}) for p in combos]


def get_all_classifiers(task: str = 'multiclass'):
    return get_knn_classifiers(task) + get_rf_classifiers(task) + get_xgb_classifiers(task)


def get_scorings(multiclass: bool = False):
    return scoring_multiclass if multiclass else scoring_binary


def model_requires_int_labels(model_name: str) -> bool:
    """KNN and XGBoost require integer-encoded labels for classification."""
    return model_name.strip().lower() in ['xgboost', 'knn']


__all__ = [
    'get_classifier_configs',
    'get_all_classifier_names',
    'get_classifier_param_grid',
    'get_knn_classifiers',
    'get_rf_classifiers',
    'get_xgb_classifiers',
    'get_all_classifiers',
    'get_scorings',
    'model_requires_int_labels',
]