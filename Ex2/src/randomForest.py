
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import joblib import Parallel, delayed

import pandas as pd
from regressionTree import RegressionTree


@dataclass
class TreeResult:
    bootstrapped_df: pd.DataFrame
    out_of_bag_df: pd.DataFrame
    trained_tree: Optional[RegressionTree] = None


#TODO: Umstellen von df auf X und Y


class RandomForest:
    def __init__(self, n_trees, X, y, min_samples_split=20, feature_selection=True, debug_trees=False, bootstrapping=True, n_jobs=1):
        self.n_trees = n_trees
        self.X = X
        self.y = y
        self.df = pd.concat([X, y], axis=1)
        self.target_column = y.name
        self.min_samples_split = min_samples_split  # the minimum number of samples required to split a node in the trees
                                                      # default value is 20, can be adjusted for testing

        self.feature_selection = feature_selection    # use feature selection in each node split (for random forest)
                                                      # set to False for debugging purposes

        self.debug_trees = debug_trees                # set to True to store the dataframes in each node for debugging purposes
        self.bootstrapping = bootstrapping
        self.n_jobs = n_jobs

        self.trained_trees = []                       # stores the trained trees as a list of RegressionTree objects

    
    def _train_tree(self, df, i, predefined_tree=None):
        """
        Hilfsmethode: Trainiert einen einzelnen Baum.
        Diese Methode wird parallel ausgeführt.
        """
        if predefined_tree:
            return predefined_tree
        
        # Bootstrapping durchführen
        tree_result = self.get_bootstrap_samples(df, i)

        # Baum initialisieren
        # WICHTIG: n_jobs=1 hier setzen, da die Parallelisierung 
        # jetzt auf Forest-Ebene stattfindet.
        tree = RegressionTree(
            df=tree_result.bootstrapped_df,
            target_column=self.target_column,
            min_samples_split=self.min_samples_split,
            feature_selection=self.feature_selection,
            debug=self.debug_trees,
            n_jobs=1  
        )
        tree.fit()
        return tree

    def fit(self, df, predefined_trees=None):
        """
        Fit the random forest to the provided dataset in parallel.
        Stores the trained trees in self.trained_trees.
        """
        start_time = time.time()
        
        # Parallelisierung mit joblib
        self.trained_trees = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(self._train_tree)(
                df, 
                i, 
                predefined_trees[i] if predefined_trees else None
            ) 
            for i in range(self.n_trees)
        )

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds using n_jobs={self.n_jobs}")
        
      
    def get_bootstrap_samples(self, df, tree_index):
        """
        Generate bootstrapped and out-of-bag samples for a given tree.
        Returns a TreeResult object containing the bootstrapped and out-of-bag dataframes.
        """
        if not self.bootstrapping:
            return TreeResult(
                bootstrapped_df=df,
                out_of_bag_df=pd.DataFrame(),  # empty dataframe
                trained_tree=None
            )

        n_samples = df.shape[0]
        bootstrapped_indices = pd.Series(
            pd.np.random.choice(n_samples, size=n_samples, replace=True)
        )
        bootstrapped_df = df.iloc[bootstrapped_indices].reset_index(drop=True)

        oob_indices = pd.Series(
            list(set(range(n_samples)) - set(bootstrapped_indices))
        )
        out_of_bag_df = df.iloc[oob_indices].reset_index(drop=True)

        return TreeResult(
            bootstrapped_df=bootstrapped_df,
            out_of_bag_df=out_of_bag_df,
            trained_tree=None
        )
        
    def predict(self, X):
       
    
    def _generate_tree_predictions(self, input_df):

        all_predictions = []
        for i, tree in enumerate(self.trained_trees):
          
            tree_pred_df = tree.predict_dataframe(input_df)
            all_predictions.append(tree_pred_df)
        return all_predictions