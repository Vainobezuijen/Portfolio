import numpy as np

# Regression Loss Functions

def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The MSE measures the average of the squared differences between the predicted values
    and the true values. It is always non-negative, sensitive to outliers, differentiable, 
    convex (not for deep NN's), scale-dependent (Use RMSE).

    Parameters:
        y       : True target values.
        y_pred  : Predicted values.
        
    Returns:
        Loss value.
    """
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    if y.shape != y_pred.shape:
        raise ValueError("Shapes of y and y_pred must match.")
    return 0.5 * np.mean(np.square(y_pred - y))

def mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It measures the average of the absolute differences between the predicted values and the true values.
    It is non-negative, robust to outliers, non-differentiable (use sub-gradient methods), 
    convex (not for deep NN's).

    Parameters:
        y       : True target values.
        y_pred  : Predicted values.
        
    Returns:
        Loss value.
    """
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    if y.shape != y_pred.shape:
        raise ValueError("Shapes of y and y_pred must match.")
    return np.mean(np.abs(y_pred - y))

def huber_loss(y: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """
    The Huber loss combines the properties of both Mean Squared Error (MSE) and Mean Absolute
    Error (MAE). Huber loss is designed to be more robust to outliers than MSE while maintaining smoothness and
    differentiability. When the error is small, the Huber loss function behaves like the MSE loss function, and when the error is large,
    the Huber loss function behaves like the MAE loss function.

    Parameters:
        y       : True target values.
        y_pred  : Predicted values.
        delta   : Threshold at which to change between quadratic and linear behavior.
        
    Returns:
        Loss value.
    """
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    if y.shape != y_pred.shape:
        raise ValueError("Shapes of y and y_pred must match.")
    error = y - y_pred
    abs_err = np.abs(error)
    quadratic = 0.5 * error**2
    linear = delta * abs_err - 0.5 * delta**2
    loss_per_sample = np.where(abs_err <= delta, quadratic, linear)

    return np.mean(loss_per_sample)

def quantile_loss(y: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    This function is often used for predicting an interval instead of a single value. 
    Overestimation occurs when a model’s prediction exceeds the actual value. 
    Underestimation is the opposite. It occurs when a model’s prediction is lower than the actual value.

    Parameters:
        y       : True target values.
        y_pred  : Predicted quantile values.
        alpha   : The quantile to estimate (between 0 and 1).
        
    Returns:
        Loss value.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    if y.shape != y_pred.shape:
        raise ValueError("Shapes of y and y_pred must match.")
    diff = y - y_pred
    loss_per_sample = np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)

    return np.mean(loss_per_sample)


def poisson_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Poisson loss is used in regression tasks when the target variable represents count data and is assumed to
    follow a Poisson distribution. The Poisson loss is derived from the negative log-likelihood of the Poisson distribution. 
    It maximizes the likelihood of observing the count data given the predicted values.
        
    Parameters:
        y       : True count data.
        y_pred  : Predicted log(counts); note that np.log(np.exp(y_pred)) returns y_pred.
        
    Returns:
        Loss value.
    """
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    if y.shape != y_pred.shape:
        raise ValueError("Shapes of y and y_pred must match.")
    return np.mean(np.exp(y_pred) - y * y_pred)


