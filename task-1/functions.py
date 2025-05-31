import numpy as np

def sigmoid(z) -> np.float:
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z) -> np.float:
    return sigmoid(z)*(sigmoid(z) - 1)
