import numpy as np
import numpy.linalg as linalg


def j_mse(y, y_hat):
    return np.mean( (y-y_hat)**2 )


def grad_j_mse(y, y_hat):
    return (y_hat - y)


def l2_norm(w):
    return np.nan


def grad_l2_norm(w):
    return np.nan


def l1_norm(w):
    return np.nan


def grad_l1_norm(w):
    return np.nan
