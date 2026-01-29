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
                 criterion='entropy', random_state=42):
        """
        Parameters:
        max_depth: Maximum depth of the tree.
        task_type: 'classification' or 'regression'.
        min_samples_split: Minimum number of samples required to split an internal node.
        max_features: Number of features to consider when looking for the best split.
        criterion: The function to measure the quality of a split. Supported criteria are 'MSE' 
                    for regression and 'entropy' or 'gini' for classification.
        """
        
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
        self.random_state = random_state
        self.task_type = task_type # Classification or Regression
        self.criterion = criterion

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        This function is used to train the model. X is the feature matrix, with each row being a sample and each column a feature. 
        y is the target variable containing the correct labels. 
        '''
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        if self.task_type == 'cls':
            self.root = self._grow_tree(X, y, depth=0, criterion=self.criterion)
        elif self.task_type == 'reg':
            ymean = np.mean(y)
            self.mse = self._get_mse(y, ymean)
            self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth=0, criterion='MSE'):
        '''
        This function grows the tree recursively. It starts at the root, decides on how to split the data and creates child nodes. 
        Then it does the same for those child nodes. The split is based on the highest information gain. 
        The features to split on are chosen randomly, introducing randomness in the algorithm
        Each feature is checked for the highest information gain.
        '''
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            if criterion == 'entropy':
                return Node(value=self._most_common_label(y))
            elif criterion == 'MSE':
                return Node(value=np.mean(y))
        
        # Find best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        if criterion == 'MSE':
            ymean = np.mean(y)
            self.mse = self._get_mse(y, ymean)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs, criterion)
        if best_feature is None or best_threshold is None:
            return Node(value=np.mean(y) if criterion == 'MSE' else self._most_common_label(y))


        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _most_common_label(self, y: np.array):
      # ___________________________________ # 
        counter = Counter(y) # CREATE
      # ___________________________________ #
        return counter.most_common(1)[0][0]
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idxs, criterion):
        if criterion == 'entropy':
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
        else:
            split_idx = None
            split_threshold = None
            mse_base = self.mse

            for feat_idx in feat_idxs:
                # sort values with the feat_idx
                X_column = X[:, feat_idx]
                order = np.argsort(X_column)
                X_column_sorted = X_column[order]
                y_sorted = y[order]

                x_mean = self._moving_average(X_column_sorted, 2)

                for value in x_mean:
                    # where x of a feature < value add corresponding y to left y
                    left_y = y_sorted[X_column_sorted <= value]
                    right_y = y_sorted[X_column_sorted > value]
                    if len(left_y) == 0 or len(right_y) == 0:
                        continue
                    left_mean = np.mean(left_y)
                    right_mean = np.mean(right_y)
                    residual_left = left_y - left_mean
                    residual_right = right_y - right_mean
                    r = np.concatenate((residual_left, residual_right), axis=None)
                    n = len(r)
                    if n == 0:
                        continue
                    split = np.sum(r**2)/n
                    if split < mse_base:
                        mse_base = split
                        split_idx = feat_idx
                        split_threshold = value
            return split_idx, split_threshold

    def _moving_average(self, x:np.array, window:int=2):
        return np.convolve(x, np.ones(window), 'valid') / window

    def _get_mse(self, y, ymean):
        return np.sum((y - ymean)**2) / len(y)
    
    def _information_gain(self, X_column, y, threshold):
        # parent entropy
        if self.criterion == 'gini':
            parent = self._gini(y)
        else:
            parent = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate weighted avg of entropy children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        if self.criterion == 'gini':
            child = (n_l/n) * self._gini(y[left_idx]) + (n_r/n) * self._gini(y[right_idx])
        else:
            child = (n_l/n) * self._entropy(y[left_idx]) + (n_r/n) * self._entropy(y[right_idx])
    
        return parent - child

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        y = np.asarray(y).flatten()
        if y.dtype != int:
            y = y.astype(int)  # convert labels like 0.0, 1.0 → 0, 1
        hist = np.bincount(y)
        p_s = hist / len(y)
        return -np.sum([p * np.log2(p) for p in p_s if p > 0]) 

    def _gini(self, y):
        y = np.asarray(y).flatten()
        if y.dtype != int:
            y = y.astype(int)
        hist = np.bincount(y)
        p = hist / len(y)
        return 1.0 - np.sum(p ** 2)


    def predict(self, X:np.ndarray):
        '''
        Predicting works by traversing the tree, it follows the decisions of the nodes recursively, 
        until it is at a leaf node and then it returns the leaf value.
        '''
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x: np.ndarray , node: Node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)