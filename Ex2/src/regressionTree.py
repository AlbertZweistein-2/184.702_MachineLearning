

# Imlemente Feature List to skip
# Implement max Features to consider at each split (randomly select features)
# Feature selection for random forest
# 



import numpy as np
from joblib import Parallel, delayed
import pandas as pd



# def rss(df, target):
#     if len(df) == 0:
#         return 0    
#     mean_value = df[target].mean()
#     residuals = df[target] - mean_value
#     return (residuals ** 2).sum() / len(df)

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
        # if self.value is None and (self.left is not None and self.right is not None):
        #     raise ValueError("Leaf node must have a value.")
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
    
    def __repr__(self):
        return(f"""
TODO
               """)










# class Splitter:
#     def __init__(self, df, target, n_jobs):
#         self.df = df
        
#         self.best_feature = None
#         self.best_threshold = None
#         self.left = None
#         self.right = None
#         self.minimal_rss = float('inf')
#         self.n_jobs = n_jobs
#         self.target = target
    
#     def get_split(self):
#         self._calculate_best_split()
        
#         if self.best_feature == None:
#             # No valid split found
#             return None
        
#         return self.best_feature, self.best_threshold, self.left, self.right
    
    
#     def _skip_column(self, col):
#         if col == self.target:
#             return True
        
#         # make it more efficient but think it's not necessary for our datasets
        
#         # if self.df[col].nunique() <= 1:
#           #  return True
        
#     def _calculate_best_split(self):
        
        
#         def process_feature(feature):
#             if self._skip_column(feature):
#                 return None
            
#             threshold, rr = self._calculate_best_split_for_feature(self.df, feature)
#             return (rr, threshold, feature)

#         # 2. Parallelisierung starten
#         results = Parallel(n_jobs=self.n_jobs)(
#             delayed(process_feature)(feature) for feature in self.df.columns
#         )
        
#         # 3. Ergebnisse filtern (None entfernen, falls skip_column zutraf i.e. Target Variable)
#         valid_results = [res for res in results if res is not None]
        
#         if not valid_results:
#             print("No valid split found.")
#             return

        
#         best_result = min(valid_results, key=lambda x: x[0])
        
#         found_rr, found_threshold, found_feature = best_result

#         # 5. State updaten, wenn besser als bisheriges Minimum
#         if found_rr < self.minimal_rss:
#             self.minimal_rss = found_rr
#             self.best_feature = found_feature
#             self.best_threshold = found_threshold
            
            
#             self.left, self.right = self._split_by_threshold(
#                 self.df, self.best_feature, self.best_threshold
#             )
             

#     def _split_by_index(self, df, index):
#         return df.iloc[:index], df.iloc[index:]

#     def _calculate_best_split_for_feature(self, df, feature, criteria='RSS'):
        
#         if criteria == 'RSS':
#             min_rr = float('inf')
#             min_threshold = None
        
#             for i in range(1, len(df[feature])):
#                 threshold = (df[feature].iloc[i-1] + df[feature].iloc[i]) / 2
            
#                 left, right = self._split_by_threshold(df, feature, threshold)
            
#                 rr = rss(left, self.target) + rss(right, self.target)
            
#                 if rr < min_rr:
#                     min_rr = rr
#                     min_threshold = threshold
#         else:
#             # TODO other criteria
#             raise ValueError(f"Criteria '{criteria}' not implemented.")
        
#         return min_threshold, min_rr
    
        
#     def _split_by_threshold(self, df, feature, threshold):
#         left = df[df[feature] <= threshold].copy()
#         right = df[df[feature] > threshold].copy()
        
#         left.reset_index(drop=True, inplace=True)
#         right.reset_index(drop=True, inplace=True)
#         return left, right
    
    
class Splitter:
    def __init__(self, df, target, n_jobs):
        self.df = df
        self.target = target
        self.n_jobs = n_jobs
        
        self.best_feature = None
        self.best_threshold = None
        self.minimal_rss = float('inf')
        # Wir speichern die Indizes für den Split, nicht die DataFrames (spart Speicher)
        self.best_left_indices = None
        self.best_right_indices = None
    
    def get_split(self):
        self._calculate_best_split()
        
        if self.best_feature is None:
            return None
        
        # Erst ganz am Ende splitten wir den DataFrame wirklich
        # Das spart massiv Zeit während der Suche
        left = self.df.loc[self.best_left_indices].copy().reset_index(drop=True)
        right = self.df.loc[self.best_right_indices].copy().reset_index(drop=True)
        
        return self.best_feature, self.best_threshold, left, right
    
    def _skip_column(self, col):
        if col == self.target:
            return True
        # Feature mit nur 1 Wert bringt keinen Split -> überspringen
        if self.df[col].nunique() <= 1:
            return True
        return False
        
    def _calculate_best_split(self):
        # Wrapper für Parallelisierung
        def process_feature(feature):
            if self._skip_column(feature):
                return None
            return self._calculate_best_split_for_feature(self.df, feature)

        # Parallelisierung
        # Hinweis: Bei sehr kleinen Knoten ist der Overhead von Parallelisierung oft 
        # höher als der Nutzen. Man könnte hier prüfen: if len(self.df) < 1000: n_jobs=1
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_feature)(col) for col in self.df.columns
        )
        
        valid_results = [res for res in results if res is not None]
        
        if not valid_results:
            return # Kein Split gefunden

        # Den besten aus allen Features finden (Tuple: rss, threshold, feature, masken)
        # Wir suchen das Minimum basierend auf dem ersten Element (rss)
        best_result = min(valid_results, key=lambda x: x[0])
        
        found_rr, found_threshold, found_feature, left_idx, right_idx = best_result

        if found_rr < self.minimal_rss:
            self.minimal_rss = found_rr
            self.best_feature = found_feature
            self.best_threshold = found_threshold
            self.best_left_indices = left_idx
            self.best_right_indices = right_idx

    def _calculate_best_split_for_feature(self, df, feature):
        # 1. Daten holen
        X_col = df[feature].values
        y_col = df[self.target].values
        
        # 2. Einzigartige Werte sortieren (WICHTIG!)
        unique_values = np.unique(X_col)
        
        if len(unique_values) < 2:
            return None
        
        # Thresholds sind die Mitten zwischen den sortierten Werten
        # (Viel weniger Iterationen als über jede Zeile)
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        
        min_rr = float('inf')
        best_thresh = None
        best_left_idx = None
        best_right_idx = None
        
        # 3. Iterieren
        # (Man könnte das hier auch noch vektorisieren, aber Loop ist verständlicher)
        for threshold in thresholds:
            # Maske erstellen (Boolean Array)
            left_mask = X_col <= threshold
            right_mask = ~left_mask
            
            y_left = y_col[left_mask]
            y_right = y_col[right_mask]
            
            # Wenn ein Split leer ist, überspringen
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            current_rr = calculate_rss(y_left) + calculate_rss(y_right)
            
            if current_rr < min_rr:
                min_rr = current_rr
                best_thresh = threshold
                # Wir merken uns die Indizes (oder die Maske), um später das DF zu teilen
                best_left_idx = df.index[left_mask]
                best_right_idx = df.index[right_mask]
                
        # Rückgabe an den Parallel-Worker
        return (min_rr, best_thresh, feature, best_left_idx, best_right_idx)    



class RegressionTree:
    def __init__(self, 
                 X,
                 y,
                 max_depth=None, # The maximum depth of the tree
                 min_samples_split=2, # The minimum number of samples required to split an internal node
                 min_samples_leaf=1, # The minimum number of samples required to be at a leaf node
                 n_jobs=1 # Number of parallel jobs to run
                 ):
        
        self.X = X
        self.y = y
        self.df = pd.concat([X, y], axis=1)
        self.target = y.name
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        
        self.root = None
            
    
    def fit(self):
        """
        Creates the regression tree for the dataset.
        """
        root = TreeNode()
        self.root = self._insert_node(root, self.df)
    
    
    def predict(self, df):
        """
        Predicts target values for the provided dataframe.
        """
        return df.apply(self._predict_row, axis=1)


    def _insert_node(self, node, df, depth = 0):
        """
        Inserts a node as leaf node or recursively calls the function for its left and right child.
        """
        if self._stop_splitting(df, depth):
            # base case, stop splitting
            return self._leaf_node(df, node, depth)
        
        # calculate possible split
        splitter = Splitter(df=df.copy(), target=self.target, n_jobs=self.n_jobs)
        split = splitter.get_split()
        
        if split is None:
            return self._leaf_node(df, node, depth)
        else:
            feature, threshold, left, right = split
        
        node.threshold = threshold
        node.feature = feature
        
        # create children
        node.left = TreeNode()
        node.right = TreeNode()
        node.left = self._insert_node(node.left, left, depth + 1)
        node.right = self._insert_node(node.right, right, depth + 1)
        # node.value = 
        
        return node


    def _stop_splitting(self, df, depth):
        n = df.shape[0]
        
        # Check minimum number of samples
        if n < self.min_samples_split:
            return True
        
        # Check if splitting into two valid child nodes is possible
        if n < 2 * self.min_samples_leaf:
            return True
        
        # Check maximum depth
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        # Check variance in target column (if 1, no further improvement possible)
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
        node = self.root
        while not node.is_leaf():
            feature_value = row[node.feature]
            if feature_value <= node.threshold:
                node = node.left 
            else:
                node = node.right
        return node.value

__all__ = ['RegressionTree']