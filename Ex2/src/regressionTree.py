import numpy as np
from joblib import Parallel, delayed
import pandas as pd


def calculate_rss(y):
    if len(y) == 0:
        return 0
    mean_value = np.mean(y)
    residuals = y - mean_value
    return np.sum(residuals ** 2)

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, depth=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Predicted value for leaf nodes
        self.depth = depth              # Tree depth at node
        
    def is_leaf(self):
        return (self.left is None) and (self.right is None)
    
    def __str__(self, level=0):
        ret = "\t" * level
        if self.is_leaf():
            ret += f"Leaf(value={self.value})\n"
        else:
            ret += f"[X{self.feature} <= {self.threshold}]\n"
            if self.left:
                ret += self.left.__str__(level + 1)
            if self.right:
                ret += self.right.__str__(level + 1)
        return ret


    
class Splitter:
    def __init__(self, df, target, n_jobs, verbose=1, max_features=None, random_state=None):
        self.df = df
        self.target = target
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_features = max_features
        self.random_state = random_state
        
        self.best_feature = None
        self.best_threshold = None
        self.minimal_rss = float('inf')
        self.best_left_indices = None
        self.best_right_indices = None
    
    def get_split(self):
        self._calculate_best_split()
        
        if self.best_feature is None:
            return None
        
        left = self.df.loc[self.best_left_indices].copy().reset_index(drop=True)
        right = self.df.loc[self.best_right_indices].copy().reset_index(drop=True)
        
        return self.best_feature, self.best_threshold, left, right
    
    def _skip_column(self, col):
        if col == self.target:
            return True
        if self.df[col].nunique() <= 1:
            return True
        return False
        
    def _calculate_best_split(self):
        candidate_features = [
            col for col in self.df.columns
            if (not self._skip_column(col))
        ]

        if not candidate_features:
            return

        selected_features = candidate_features
        if self.max_features is not None:
            n_features = len(candidate_features)
            if self.max_features == 'sqrt':
                k = max(1, int(n_features ** 0.5))
            elif self.max_features == 'log2':
                k = max(1, int(np.log2(n_features)))
            elif isinstance(self.max_features, int):
                k = min(self.max_features, n_features)
            elif isinstance(self.max_features, float):
                k = max(1, int(self.max_features * n_features))
            else:
                k = n_features

            rng = np.random.default_rng(self.random_state)
            selected_features = rng.choice(candidate_features, size=k, replace=False).tolist()

        def process_feature(feature):
            return self._calculate_best_split_for_feature(self.df, feature)

        if len(self.df) < 1000 or self.n_jobs == 1:
            results = [process_feature(col) for col in selected_features]
        else:
            results = Parallel(n_jobs=self.n_jobs, backend='threading', verbose=self.verbose)(
                delayed(process_feature)(col) for col in selected_features
            )
        
        valid_results = [res for res in results if res is not None]
        
        if not valid_results:
            return 
        
        best_result = min(valid_results, key=lambda x: x[0])
        
        found_rr, found_threshold, found_feature, left_idx, right_idx = best_result

        if found_rr < self.minimal_rss:
            self.minimal_rss = found_rr
            self.best_feature = found_feature
            self.best_threshold = found_threshold
            self.best_left_indices = left_idx
            self.best_right_indices = right_idx

    def _calculate_best_split_for_feature(self, df, feature):
        X_col = df[feature].values
        y_col = df[self.target].values

        # sort by feature
        order = np.argsort(X_col)
        x_sorted = X_col[order]
        y_sorted = y_col[order]

        N = len(y_sorted)
        if N < 2:
            return None

        valid_split = x_sorted[:-1] != x_sorted[1:]

        cum_y = y_sorted.cumsum()
        cum_y2 = (y_sorted ** 2).cumsum()

        n_left = np.arange(1, N)
        n_right = N - n_left

        sum_left = cum_y[:-1]
        sum_right = cum_y[-1] - cum_y[:-1]

        rss_left = cum_y2[:-1] - (sum_left ** 2) / n_left
        rss_right = (cum_y2[-1] - cum_y2[:-1]) - (sum_right ** 2) / n_right

        total_rss = rss_left + rss_right
        total_rss[~valid_split] = np.inf

        best_pos = np.argmin(total_rss)
        min_rr = total_rss[best_pos]

        best_thresh = (x_sorted[best_pos] + x_sorted[best_pos + 1]) / 2
        best_left_idx = df.index[order[:best_pos + 1]]
        best_right_idx = df.index[order[best_pos + 1:]]

        return (min_rr, best_thresh, feature, best_left_idx, best_right_idx)    



class RegressionTree:
    def __init__(self,
                 max_depth=None, # The maximum depth of the tree
                 min_samples_split=2, # The minimum number of samples required to split an internal node
                 min_samples_leaf=1, # The minimum number of samples required to be at a leaf node
                 max_features=None, # Number of features to consider per split (RF-style)
                 random_state=None,
                 n_jobs=1, # Number of parallel jobs to run
                 verbose=1,
                 ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.df = None
        self.target = None
        self.root = None
        self._rng = None
            
    
    def fit(self, X, y):
        """
        Creates the regression tree for the dataset.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array or Series.")
        if X.shape[0] < self.min_samples_split:
            raise ValueError("Number of samples is less than min_samples_split.")
        
        self.df = pd.concat([X, y], axis=1)
        self.target = y.name
        self._rng = np.random.default_rng(self.random_state)
        root = TreeNode()
        self.root = self._insert_node(root, self.df)
        #Free memory
        self.df = None
        self.target = None
        self._rng = None
    
    def predict(self, df):
        """
        Predicts target values for the provided dataframe.
        """
        if self.root is None:
            raise ValueError("The tree has not been trained yet.")
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        return df.apply(self._predict_row, axis=1)


    def _insert_node(self, node, df, depth = 0):
        """
        Inserts a node as leaf node or recursively calls the function for its left and right child.
        """
        if self._stop_splitting(df, depth):
            return self._leaf_node(df, node, depth)
        
        split_random_state = None
        if self._rng is not None:
            split_random_state = int(self._rng.integers(0, 2**32 - 1))
        splitter = Splitter(
            df=df,
            target=self.target,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            max_features=self.max_features,
            random_state=split_random_state,
        )
        split = splitter.get_split()
        
        if split is None:
            return self._leaf_node(df, node, depth)
        else:
            feature, threshold, left, right = split
        
        node.threshold = threshold
        node.feature = feature
        
        node.left = TreeNode()
        node.right = TreeNode()
        node.left = self._insert_node(node.left, left, depth + 1)
        node.right = self._insert_node(node.right, right, depth + 1)
        
        return node


    def _stop_splitting(self, df, depth):
        n = df.shape[0]
        
        if n < self.min_samples_split:
            return True
        
        if n < 2 * self.min_samples_leaf:
            return True
        
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        if df[self.target].nunique() <= 1:
            return True
        
        return False
    

    def _leaf_node(self, df, node, depth):
        node.threshold = None
        node.feature = None 
        node.depth = depth
        node.value = df[self.target].mean()
        return node


    def _predict_row(self, row):
        """
        Predicts target value for the given row
        """
        if self.root is None:
            raise ValueError("The tree has not been trained yet.")
        if not isinstance(row, pd.Series):
            raise ValueError("Input row must be a pandas Series.")
        
        node = self.root
        while not node.is_leaf():
            feature_value = row[node.feature]
            if feature_value <= node.threshold:
                node = node.left 
            else:
                node = node.right
        return node.value

__all__ = ['RegressionTree']