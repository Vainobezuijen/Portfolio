import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000, shuffle=True, seed=None):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.shuffle = shuffle
        self.seed = seed
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1).astype(float)

        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_iters):
            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = range(n_samples)

            for idx in indices:
                x_i = X[idx]
                y_i = y_[idx]
                margin = y_i * (np.dot(x_i, self.w) + self.b)

                if margin >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_i * x_i)
                    self.b += self.lr * y_i

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

class SVM_OVR:
    def __init__(self, **svm_kwargs):
        self.svm_kwargs = svm_kwargs
        self.models = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            y_bin = np.where(y == c, 1, -1)
            clf = SVM(**self.svm_kwargs)
            clf.fit(X, y_bin)
            self.models[c] = clf
        return self

    def decision_function(self, X):
        scores = []
        for c in self.classes_:
            clf = self.models[c]
            scores.append(X @ clf.w + clf.b)
        return np.column_stack(scores)

    def predict(self, X):
        scores = self.decision_function(X)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]
