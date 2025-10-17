import numpy as np
import pickle

class Node:
    '''

    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_lead_node(self):
        return self.value is not None
        

class DecisionTree:
    '''
    A decision tree is a non-parametric supervised machine learning model that makes prediction by learning how to split data.
    '''
    def __init__(self, min_sample_leafs=2, max_depth=None, task_type='classification', min_samples_split=2, max_features=None, 
                 criterion='gini', random_state=42, min_impurity_decrease=0):
        """
        Parameters:
        max_depth: Maximum depth of the tree.
        task_type: 'classification' or 'regression'.
        min_samples_split: Minimum number of samples required to split an internal node.
        max_features: Number of features to consider when looking for the best split.
        criterion: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain.
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = None
        self.task_type = task_type # Classification or Regression
        self.min_samples_split = min_samples_split
        self.min_sample_leafs = min_sample_leafs
        self.max_features = max_features
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease

    def fit(X: np.ndarray, y: np.ndarray):
        '''
        
        '''
        pass

    def predict(X: np.ndarray):
        '''
        
        '''
        pass

    def predict_proba(X: np.ndarray):
        '''
        
        '''
        pass

    def score(X: np.ndarray, y: np.ndarray):
        '''
        
        '''
        pass