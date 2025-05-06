from Loss.losses_forward import *
from Loss.losses_backward import *

# Mapping with the functions

loss_forward = {
    'mse': mse,
    'mae': mae,
    'huber': huber_loss,
    'quantile': quantile_loss,
    'poisson': poisson_loss
}

loss_backward = {
    'mse': mse_grad,
    'mae': mae_grad,
    'huber': huber_grad,
    'quantile': quantile_grad,
    'poisson': poisson_grad
}