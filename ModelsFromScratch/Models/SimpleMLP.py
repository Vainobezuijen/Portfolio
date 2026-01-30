import numpy as np


class NeuralNet:
    """
    Simple MLP with 2 hidden layers and ReLU activation.
    Output layer is linear (regression).
    """

    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs, learning_rate=0.01, epochs=1000, seed=None):
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)

        self.W1 = self._init_weights(n_inputs, n_hidden1)
        self.b1 = np.zeros((1, n_hidden1))
        self.W2 = self._init_weights(n_hidden1, n_hidden2)
        self.b2 = np.zeros((1, n_hidden2))
        self.W3 = self._init_weights(n_hidden2, n_outputs)
        self.b3 = np.zeros((1, n_outputs))

    def _init_weights(self, n_in, n_out):
        return self.rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(z.dtype)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3
        y_pred = z3
        return z1, a1, z2, a2, y_pred

    def fit(self, X, y, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            z1, a1, z2, a2, y_pred = self._forward(X)
            m = X.shape[0]

            loss = 0.5 * np.mean((y_pred - y) ** 2)
            dY = (y_pred - y) / m

            dW3 = a2.T @ dY
            db3 = np.sum(dY, axis=0, keepdims=True)

            dA2 = dY @ self.W3.T
            dZ2 = dA2 * self._relu_deriv(z2)
            dW2 = a1.T @ dZ2
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * self._relu_deriv(z1)
            dW1 = X.T @ dZ1
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            self.W3 -= self.learning_rate * dW3
            self.b3 -= self.learning_rate * db3
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}: loss={loss:.6f}")

    def predict(self, X):
        X = np.asarray(X)
        _, _, _, _, y_pred = self._forward(X)
        return y_pred
