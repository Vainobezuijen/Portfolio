import numpy as np
from utils import initialize
from loss_functions_mapping import loss_functions

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
    
    def __init__(self, learning_rate, convergence_tol = 1e-6, loss_name='mse'):
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.W = None
        self.b = None
        self.loss = loss_functions.get(loss_name)

    def initialize_parameters(self, shape, init_mode, seed=None):
        '''
        Initialize model parameters.

        Parameters:
            n_features (int): The number of features in the input data.
        '''
        self.W = initialize(shape, init_mode, seed=seed)
        self.b = initialize(shape, init_mode, seed=seed)

    def forward_pass(self, X):
        '''
        Compute the forward pass of the linear regression model.

        Parameters:
            X (numpy.ndarray): Input data of shape (m, n_features).

        Returns:
            numpy.ndarray: Predictions of shape (m,).
        '''

        return np.dot(X, self.W) + self.b

    def predict(self, X):

        '''
        Compute the predicted values based on the input by computing the dot product of the weights and the input and adding the bias.

        Parameters:
            X: input data of the model

        Returns:
            dot product of the weights and X + the bias
        '''

        return np.dot(X, self.W) + self.b

    def compute_cost(self, y_pred, y):

        '''
        
        '''

        return self.loss(y, y_pred)