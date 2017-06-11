import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
  
def name():
    return "Sigmoid Neuron"

def g(z):
    return sigmoid(z)

def grad_g(z):
    return sigmoid(z)*(1-sigmoid(z))
