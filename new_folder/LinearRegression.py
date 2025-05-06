import numpy as np
import pickle

from utils import initialize
from Loss.loss_mapping import *
from Optimizers.optimizers_mapping import *
from plotting import plot_costs

import os

class LinearRegression:

    '''
    Linear Regression models the relationship between a dependent variable and one or multiple independent variables by fitting a linear equation to the data

    Params:
        learning rate: float
        convergence_tol: float (optional)

    Attributes:
        weights w: np.ndarray
        bias b: float

    Methods:
        initialize_parameters(n_features): Initialize model parameters.
        forward(X): Compute the forward pass of the linear regression model.
        compute_cost(predictions): Compute the mean squared error cost.
        backward(predictions): Compute gradients for model parameters.
        fit(X, y, iterations, plot_cost=True): Fit the linear regression model to training data.
        predict(X): Predict target values for new input data.
        save_model(filename=None): Save the trained model to a file using pickle.
        load_model(filename): Load a trained model from a file using pickle.
    '''
    

    def __init__(self, learning_rate: float = 0.01, iterations: int = 10000, convergence_tol = 1e-6, init_mode: str = 'zero', loss_name: str = 'mse', optimizer: str = 'gradient_descent'):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None
        self.loss = loss_forward.get(loss_name)
        self.grad = loss_backward.get(loss_name)
        self.init_mode = init_mode
        self.iterations = iterations
        self.optimizer = optimizers_mapping.get(optimizer)(self.learning_rate)

    def initialize_parameters(self, shape, init_mode, seed=None):
        '''
        Initialize model parameters.

        Parameters:
            n_features (int): The number of features in the input data.
        '''
        self.W = initialize(shape, init_mode, seed=seed)
        self.b = initialize((1,), init_mode, seed=seed)


    def forward(self, X):
        '''
        Compute the forward pass of the linear regression model.

        Parameters:
            X (numpy.ndarray): Input data of shape (m, n_features).

        Returns:
            numpy.ndarray: Predictions of shape (m,).
        '''

        return np.squeeze(np.dot(X, self.W) + self.b)


    def compute_cost(self, y, y_pred):

        '''
        Compute the cost.

        Parameters:
            y (np.ndarray)      : Target values of shape (m,).
            y_pred (np.ndarray) : Predictions of shape (m,).

        Returns:
            float: cost.
        '''

        return self.loss(y, y_pred)
    


    
    def backward(self, X, y, y_pred):

        '''
        Computes the gradient of the loss

        Parameters:
            y       : target values
            y_pred  : predicted target values

        Returns:
            float: gradient.
        '''

        dldy = self.grad(y, y_pred)

        dW = X.T.dot(dldy)
        db = np.sum(dldy)

        return dW, db
    

    def fit(self, X, y, x_val=None, y_val=None, plot_cost=True):

        '''
        Fit the linear regression model to the training data.

        Parameters:
            X (numpy.ndarray): Training input data of shape (m, n_features).
            y (numpy.ndarray): Training labels of shape (m,).
            validation_data (tuple): Validation x and y
            iterations (int): The number of iterations for gradient descent.
            plot_cost (bool, optional): Whether to plot the cost during training. Defaults to True.

        Raises:
            AssertionError: If input data and labels are not NumPy arrays or have mismatched shapes.

        Plots:
            Plotly line chart showing cost vs. iteration (if plot_cost is True).
        '''

        assert isinstance(X, np.ndarray), "X must be a NumPy array"
        assert isinstance(y, np.ndarray), "y must be a NumPy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert self.iterations > 0, "Iterations must be greater than 0"

        self.X = X
        self.y = y

        w_shape = (X.shape[1],1)
        self.initialize_parameters(w_shape, self.init_mode)
        
        costs = []
        validation_costs = []

        for i in range(self.iterations):
            predictions = self.forward(X)
            cost = self.compute_cost(y, predictions)
            dW, db = self.backward(X, y, predictions)
            self.optimizer.update(self, dW, db)

            if x_val is None:
                continue
            else:
                val_predictions = self.forward(x_val)
                val_cost = self.compute_cost(y_val, val_predictions)
                validation_costs.append(val_cost)

            costs.append(cost)

            if i%100 == 0:
                print(f'Iteration: {i}, Cost: {cost}')
            
            if i > 0 and abs(costs[-1] - costs[-2]) < self.convergence_tol:
                print(f'Converged after {i} iterations.')
                break
        
        if plot_cost:
            if x_val is None:
                plot_costs([costs])
            else:
                plot_costs([costs, validation_costs])



    def predict(self, X):

        '''
        Predict target values for new input data.

        Parameters:
            X (numpy.ndarray): Input data of shape (m, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (m,).
        '''

        return self.forward(X)
    
    def save_model(self, filename=None):
        if filename is None:
            filename = 'model.pk1'
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        model_data = {
            'learning_rate': self.learning_rate,
            'convergence_tol': self.convergence_tol,
            'W': self.W,
            'b': self.b
        }
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)

    @classmethod
    def load_model(cls, filename):
        """
        Load a trained model from a file using pickle.

        Parameters:
            filename (str): The name of the file to load the model from.

        Returns:
            LinearRegression: An instance of the LinearRegression class with loaded parameters.
        """
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)

        # Create a new instance of the class and initialize it with the loaded parameters
        loaded_model = cls(model_data['learning_rate'], model_data['convergence_tol'])
        loaded_model.W = model_data['W']
        loaded_model.b = model_data['b']

        return loaded_model