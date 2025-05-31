import numpy as np
from functions import *
import random

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
            Returns the zvectors and activations for each layer
        '''
        z_vectors, activations = [], [activation]
        for k in range(self.layer_count - 1):
            z_vectors.append(np.dot(self.weights[k], activation[k]) + self.bias[k])
            activation = sigmoid(np.dot(self.weights[k], activation[k]) + self.bias[k])
            activations.append(activation)

        return z_vectors, activations
    
    def SGD(self, train_data: list, n_epochs, batch_size, learning_rate, validation_data=None, early_stop_params: tuple = None):
        for iterations in range(n_epochs):
            random.shuffle(train_data)
            batches = []
            for k in range(0, len(train_data), batch_size):
                batches.append(train_data[k: k + batch_size])

            for batch in batches:
                bias_grads, weight_grads = self.back_propagate(batch[0], batch[1])
                for k in range(len(weight_grads)):
                    self.weights[k] = self.weights[k] - (learning_rate/batch_size)*weight_grads[k]
                    self.biases[k] = self.biases[k] - (learning_rate/batch_size)*bias_grads[k]

            best_acc = 0
            if early_stop_params is not None and validation_data is not None:
                patience = early_stop_params[1]
                accuracy = self.evaluate(validation_data)/len(validation_data)
                best_acc = max(accuracy, best_acc)
                cntr = 0
                if accuracy < best_acc:
                    cntr += 1
                    if cntr == patience:
                        return self.weights, self.biases
        
            print(f'Epoch {iterations}')

    def back_propagate(self, x, y):
        cost_grad_wrt_bias = [np.zeros(bias.shape) for bias in self.biases]
        cost_grad_wrt_weights = [np.zeros(weights.shape) for weights in self.weights]  

        z_vectors, activations = self.feed_forward(x)
        activations.insert(0, x)

        last_delta = np.multiply(self.cost_derivative_wrt_activations(activations[-1], y), sigmoid_prime(z_vectors[-1]))
        cost_grad_wrt_bias[-1] = last_delta
        cost_grad_wrt_weights[-1] = np.dot(last_delta, activations[-2].transpose())

        for l in range(2, self.layer_count):
            l_delta = np.multiply(np.dot(self.weights[-l + 1], last_delta), sigmoid_prime(z_vectors[-l]))
            last_delta = l_delta
            cost_grad_wrt_bias[-l] = l_delta
            cost_grad_wrt_weights[-l] = np.dot(l_delta, activations[-l - 1].transpose())

        return cost_grad_wrt_bias, cost_grad_wrt_weights
    
    def evaluate(self, data_):
        res = [(np.argmax(self.feed_forward(data[0])[1]), data[1]) for data in data_]
        corr = 0
        for z in res:
            if z[0] == z[1]:
                corr += 1

        return corr

    def cost_derivative_wrt_activations(self, a, y):
        return a - y

net = MLP([2, 4, 1])
net.print_weights()