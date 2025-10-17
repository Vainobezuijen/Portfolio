import numpy as np
import pickle
from collections import Counter

class Node:
    '''
    The node represents a node in a decision tree. It makes a decision on how to split data based on a feature.
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
        

class DecisionTree:
    '''
    A decision tree is a non-parametric supervised machine learning model that makes predictions by learning how to split data.
    '''
    def __init__(self, max_depth=100, n_features=None, task_type='classification', min_samples_split=2,
                 criterion='gini', random_state=42):
        """
        Parameters:
        max_depth: Maximum depth of the tree.
        task_type: 'classification' or 'regression'.
        min_samples_split: Minimum number of samples required to split an internal node.
        max_features: Number of features to consider when looking for the best split.
        criterion: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain.
        """
        
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
        
        self.random_state = random_state
        self.tree = None
        self.task_type = task_type # Classification or Regression
        self.criterion = criterion

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        This function is used to train the model. X is the feature matrix, with each row being a sample and each column a feature. 
        y is the target variable containing the correct labels. 
        '''
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth=0):
        '''
        
        '''
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))
        
        # Find best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree((X[left_idxs, :], y[left_idxs]), depth+1)
        right = self._grow_tree((X[right_idxs, :], y[right_idxs]), depth+1)
        return Node(best_feature, best_threshold, left, right)

    def _most_common_label(self, y: np.ndarray):
      # __________________________________ # 
        counter = Counter(y) # CREATE
      # __________________________________ #
        return counter.most_common(1)[0][0]
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # calculate information gain
                gain = self._information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold
    
    def _information_gain(self, X_column, y, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate weighted avg of entropy children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column >= split_threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        p_s = hist / len(y)
        return np.sum([p * np.log(p) for p in p_s if p > 0])
        
    def predict(self, X:np.ndarray):
        '''
        
        '''
        return np.array([self.traverse_tree(x) for x in X])
    
    def _traverse_tree(self, x: np.ndarray , node: Node):
        if node._is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)