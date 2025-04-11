from losses import *

# Mapping with the functions

loss_functions = {
    'mse': mse,
    'mae': mae,
    'huber': huber_loss,
    'quantile': quantile_loss,
    'poisson': poisson_loss
}