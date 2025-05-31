import numpy as np
from functions import *

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.layer_count = len(layers)
        self.weights, self.biases = [None for _ in range(self.layer_count - 1)], [None for _ in range(self.layer_count - 1)]
        for k in range(1, self.layer_count):
            self.weights[k - 1] = np.random.randn(self.layers[k], self.layers[k - 1])
            self.biases[k - 1] = np.random.randn(self.layers[k], 1)

    def feed_forward(self, activation) -> list:
        '''
            Feed data through the layers, activation of layer l = w(l).a(l - 1) + b(l)
        '''
        z_vectors, activations = [], []
        for k in range(self.layer_count - 1):
            z_vectors.append(np.dot(self.weights[k], activation[k]) + self.bias[k])
            activation = sigmoid(np.dot(self.weights[k], activation[k]) + self.bias[k])
            activations.append(activation)

        return z_vectors, activations

    def back_propagate(self, x, y):
        cost_grad_wrt_bias = [np.zeros(bias.shape) for bias in self.biases]
        cost_grad_wrt_weights = [np.zeros(weights.shape) for weights in self.weights]
        activations = [x]   

net = MLP([2, 4, 1])
net.print_weights()