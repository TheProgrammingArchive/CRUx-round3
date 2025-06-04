import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def sigmoid_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

def softmax_loss(y, y_pred):
    loss = -np.sum(y * np.log(y_pred + 1e-15))
    return loss