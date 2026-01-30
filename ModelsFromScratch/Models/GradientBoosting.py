import numpy as np

from Models.DecisionTree import DecisionTree


class GradientBoosting:
    """
    Simple gradient boosting for regression using CART (squared loss).
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        n_features=None,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.estimators = []
        self.init_pred = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        self.estimators = []
        self.init_pred = np.mean(y)
        y_pred = np.full_like(y, self.init_pred, dtype=float)

        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = DecisionTree(
                max_depth=self.max_depth,
                n_features=self.n_features,
                task_type="reg",
                min_samples_split=self.min_samples_split,
                criterion="MSE",
                random_state=self.random_state,
            )
            tree.fit(X, residual)
            update = tree.predict(X).reshape(-1)
            y_pred += self.learning_rate * update
            self.estimators.append(tree)

    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.full((X.shape[0],), self.init_pred, dtype=float)
        for tree in self.estimators:
            y_pred += self.learning_rate * tree.predict(X).reshape(-1)
        return y_pred
