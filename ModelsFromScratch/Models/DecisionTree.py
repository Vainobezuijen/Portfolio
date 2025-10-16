import numpy as np
import pickle

class DecisionTree:
    def __init__(self, max_depth=None, task_type='classification', min_samples_split=2, max_features=None, criterion='gini'):
        """
        Initialize the Decision Tree model.
        Parameters:
        max_depth: Maximum depth of the tree.
        task_type: 'classification' or 'regression'.
        min_samples_split: Minimum number of samples required to split an internal node.
        max_features: Number of features to consider when looking for the best split.
        criterion: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain.
        """
        self.max_depth = max_depth
        self.tree = None
        self.task_type = None  # 'classification' or 'regression'
        self.task_type = task_type
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion

    def _entropy(self, class_probabilities: list) -> float:
        """Calculate the entropy of a list of class probabilities."""
        return -sum(p * np.log2(p) for p in class_probabilities if p > 0)
    
    def _class_probabilities(self, labels: list) -> list:
        """Calculate the class probabilities for a list of labels."""
        total_count = len(labels)
        return [count / total_count for count in np.bincount(labels)]
    
    def split(self, data: np.array, feature_idx: int, split_value: float) -> tuple:
        g1 = data[data[:, feature_idx] <= split_value]
        g2 = data[data[:, feature_idx] > split_value]
        return (g1, g2)
    
    def _part_entropy(self, g1: np.array, g2: np.array) -> float:
        total_count = len(g1) + len(g2)
        p1 = len(g1) / total_count
        p2 = len(g2) / total_count

        if self.criterion == 'gini':
            entropy_g1 = 1 - sum(p ** 2 for p in self._class_probabilities(g1[:, -1]))
            entropy_g2 = 1 - sum(p ** 2 for p in self._class_probabilities(g2[:, -1]))
        elif self.criterion == 'entropy':
            entropy_g1 = self._entropy(self._class_probabilities(g1[:, -1]))
            entropy_g2 = self._entropy(self._class_probabilities(g2[:, -1]))
        else:
            raise ValueError("Unsupported criterion. Use 'gini' or 'entropy'.")

        return p1 * entropy_g1 + p2 * entropy_g2

    def find_best_split(self, data: np.array) -> tuple:
        min_part_entropy = float('inf')
        min_entropy_feature_idx = None
        min_entropy_split_value = None

        for idx in range(data.shape[1] - 1):
            feature_values = np.median(data[:, idx])
            g1, g2 = self.split(data, idx, feature_values)
            part_entropy = self._part_entropy(g1, g2)
            if part_entropy < min_part_entropy:
                min_part_entropy = part_entropy
                min_entropy_feature_idx = idx
                min_entropy_split_value = feature_values
                g1_best, g2_best = g1, g2

        return min_entropy_feature_idx, min_entropy_split_value, g1_best, g2_best