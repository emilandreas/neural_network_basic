import numpy as np

def name():
    return "tanh Neuron"

def g(z):
    return np.tanh(z)


def grad_g(z):
    return 1-tanh(z)**2
