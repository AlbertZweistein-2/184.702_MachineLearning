import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from joblib import Parallel, delayed

import pandas as pd
from regressionTree import RegressionTree
import numpy as np
import math


@dataclass
class TreeResult:
    bootstrapped_df: pd.DataFrame
    out_of_bag_df: pd.DataFrame
    trained_tree: Optional[RegressionTree] = None
    selected_features: Optional[list] = None




class RandomForest:
    def __init__(self, 
                 n_estimators, # number of trees in forest
                 max_depth=None, # maximum depth of tree
                 min_samples_split=2, # The minimum number of samples required to split an internal node
                 min_samples_leaf=1, # The minimum number of samples required to be at a leaf node
                 bootstrap=True, # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
                 max_features='sqrt', # The number of features to consider when looking for the best split
                 n_jobs=1,
                 verbose=1):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.verbose = verbose

        self.trained_trees = []
        self.X = None
        self.y = None
        self.df = None
        self.target = None
        

    def fit(self, X, y):
        """
        Fit the random forest to the provided dataset in parallel.
        Stores the trained trees in self.trained_trees.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array or Series.")
        if X.shape[0] < self.min_samples_split:
            raise ValueError("Number of samples is less than min_samples_split.")
        
        self.X = X
        self.y = y
        self.df = pd.concat([X, y], axis=1)
        self.target = y.name

        self.trained_trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._train_tree)(i) for i in range(self.n_estimators)
        )
        self.X = None
        self.y = None
        self.df = None
        
    
    def predict(self, X):
        """
        Predicts the target column for the provided dataframe.
        """
        tree_predictions = self._generate_tree_predictions(X)
        preds_df = pd.concat(tree_predictions, axis=1)
        averaged_predictions = preds_df.mean(axis=1)
        averaged_predictions = averaged_predictions.reindex(X.index)

        if averaged_predictions.isna().any():
            missing_indices = averaged_predictions[averaged_predictions.isna()].index
            raise ValueError(f"Index mismatch: missing predictions for indices {list(missing_indices)}.")

        return averaged_predictions


    def _train_tree(self, random_state):
        """
        Trains a (X-mas) tree :)
        """
        tree_result = self._get_bootstrap_samples(self.df, random_state)
        tree_result.selected_features = None
        
        tree_result.trained_tree = RegressionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=1,
            verbose=0
        )

        tree_result.trained_tree.max_features = self.max_features
        tree_result.trained_tree.random_state = random_state

        X = tree_result.bootstrapped_df.drop(columns=self.target).reset_index(drop=True)
        y = tree_result.bootstrapped_df[self.target].reset_index(drop=True)

        tree_result.trained_tree.fit(X=X, y=y)
        tree_result.bootstrapped_df = None  # free memory
        tree_result.out_of_bag_df = None  # free memory
        return tree_result

    
    def _get_bootstrap_samples(self, df, random_state):
        """
        Generate bootstrapped and out-of-bag samples.
        Returns a TreeResult object containing the bootstrapped and out-of-bag dataframes.
        """
        if not self.bootstrap:
            return TreeResult(
                bootstrapped_df=df,
                out_of_bag_df=pd.DataFrame(),
                trained_tree=None
            )

        bootstrap_df = df.sample(
            frac=1,                     
            replace=True,               
            random_state=random_state   
        )
        bootstrap_indices = bootstrap_df.index
        oob_indices = df.index.difference(bootstrap_indices) 
        
        bootstrap_df = bootstrap_df.reset_index(drop=True)
        oob_df = df.loc[oob_indices]
        oob_df = oob_df.reset_index(drop=True)
        
        return TreeResult(
            bootstrapped_df=bootstrap_df,
            out_of_bag_df=oob_df,
            trained_tree=None
        )

    
    def _get_selected_features(self, random_state):
        """
        Selects a subset of features based on the max_features parameter.
        Returns a list of selected feature names.
        """
        all_features = self.X.columns.tolist()
        n_features = len(all_features)

        if self.max_features == 'sqrt':
            k = max(1, int(n_features ** 0.5))
        elif self.max_features == 'log2':
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        else:
            k = n_features  # use all features

        rng = np.random.default_rng(random_state)
        selected_features = rng.choice(all_features, size=k, replace=False).tolist()
        return selected_features
        

    def _generate_tree_predictions(self, X):
        """
        Predicts target values for all trees of forest (in parallel).
        """
        if not self.trained_trees:
            raise ValueError("Random forest has not been trained")

        def _predict_single(tree_result):
            return tree_result.trained_tree.predict(X)

        all_predictions = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_predict_single)(tree_result)
            for tree_result in self.trained_trees
        )
        return all_predictions