import matplotlib.pyplot as plt
import numpy as np
from utils import sort_array

def plot_costs(costslist: list[list]) -> None:

    '''
    This function plots the training and optionally the validation loss as well.

    Parameters:
        costslist: a list containing lists of the training costs and optionally the validation loss
    '''

    names = ['Training', 'Validation']

    if costslist == None:
        return ValueError('No costs to plot')
    
    plt.figure(figsize=(12,8))
    for idx, cost in enumerate(costslist):
        plt.title("Cost vs Iteration")
        plt.plot(np.arange(len(cost)),cost,label=names[idx])
        plt.xticks(np.arange(len(cost),step=500),rotation=45)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.legend()
    plt.show()

def plot_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_pred: np.ndarray, title='Fitted model') -> None:
    
    '''
    This function plots the scattered train data and the predicted line from the model.

    Parameters:
        x_train: training data
        y_train: training data
        x_test: sorted test data
        y_test: sorted test data
    '''
    
    x_test = sort_array(x_test)
    y_pred = sort_array(y_pred)


    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.scatter(x_train, y_train, color='blue', label="Training data")
    plt.plot(x_test, y_pred, color='red', label="Regression line")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()