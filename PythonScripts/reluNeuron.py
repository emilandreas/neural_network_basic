import numpy as np

def name():
    return "ReLu Neuron"

def g(z):
    return np.maximum(0, z) + 0.001*np.minimum(0,z)


def grad_g(z):
    if z < 0:
        return 0.001
    else:
        return 1
