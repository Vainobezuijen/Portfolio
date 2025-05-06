import numpy as np

class GradientDescent():

    '''
    This class updates weights using a standard gradient descent algorithm.
    '''

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, model, dW, db):
        model.W = model.W - self.learning_rate * dW
        model.b = model.b - self.learning_rate * db

class SGD:

    '''
    This class updates weights using a stochastic gradient descent algorithm.
    '''

    def __init__(self, learning_rate=0.01, batch_size=32, shuffle=True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle

    def update(self, model, dW, db):
        """
        Selects a mini-batch from X and y, computes the gradients using the model's backward pass,
        and updates the model's weights.
        """
        m = X.shape[0]
        indices = np.arange(m)
        if self.shuffle:
            np.random.shuffle(indices)
        # Select a mini-batch
        batch_idx = indices[:self.batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        # Forward pass on mini-batch
        predictions = model.forward(X_batch)
        # Compute gradients on mini-batch
        dW, db = model.backward(X_batch, y_batch, predictions)
        # Update model parameters
        model.W -= self.learning_rate * dW
        model.b -= self.learning_rate * db