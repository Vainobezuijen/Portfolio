import numpy as np

def mse_grad(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    d/d(y_pred) [0.5 * mean((y_pred - y)^2)]
        = (y_pred - y) / N
    """
    return (y_pred - y) / y.size

def mae_grad(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    d/d(y_pred) [mean(|y_pred - y|)]
    subgradient: sign(y_pred - y) / N
    """
    grad = np.sign(y_pred - y)
    return grad / y.size

def huber_grad(y: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    piecewise derivative:
        if |error| <= delta : (y_pred - y)
        else : delta * sign(y_pred - y)
    all scaled by 1/N.
    """
    error = y_pred - y
    abs_err = np.abs(error)
    is_small = abs_err <= delta
    grad = np.where(is_small, error, delta * np.sign(error))
    return grad / y.size

def quantile_grad(y: np.ndarray, y_pred: np.ndarray, alpha: float) -> np.ndarray:
    """
    d/d(y_pred) of alpha * max(y - y_pred, 0) + (1 - alpha)*max(y_pred - y, 0)
    piecewise:
        if y_pred < y: -alpha
        if y_pred > y: (1 - alpha)
        if y_pred = y: subgradient in [-alpha, 1-alpha]
    """
    diff = y_pred - y
    grad = np.zeros_like(diff)
    grad[diff > 0] = (1 - alpha)
    grad[diff < 0] = -alpha
    return grad / y.size

def poisson_grad(y: np.ndarray, log_y_pred: np.ndarray) -> np.ndarray:
    """
    d/d(log_y_pred) [mean(exp(log_y_pred) - y * log_y_pred)]
      = mean( exp(log_y_pred) - y )
    """
    return (np.exp(log_y_pred) - y) / y.size