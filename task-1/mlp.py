import numpy as np
import random
from functions import *
import copy

class EarlyStopping:
    def __init__(self, monitor='acc', patience=1, restore_best_weights=True, min_delta=0):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.iters_to_patience = 0
        self.min_delta = min_delta

        if monitor == 'acc':
            self.curr = 0

        elif monitor == 'loss':
            self.curr = float('inf')

    def continue_training(self, epoch_results: tuple):
        if self.monitor == 'acc':
            acc = epoch_results['acc']
            if acc > self.curr + self.min_delta:
                self.curr = acc
                self.iters_to_patience = 0
            else:
                self.iters_to_patience += 1

        else:
            loss = epoch_results['loss']
            if loss < self.curr - self.min_delta:
                self.curr = loss
                self.iters_to_patience = 0

            else:
                self.iters_to_patience += 1

        if self.iters_to_patience == self.patience:
            return False
        
        return True
    
class MLP:
    def __init__(self, layers: list):
        '''
            layers: Provide number of neurons per layer in form of a list, [input_layers, hidden1, hidden2, ..., hiddenk, output]
            output_activation: Change activation function depending on task (classification/regression), optimal loss function is chosen automatically
        '''
        self.layer_count = len(layers)
        self.hidden_layers = self.layer_count - 1
        self.layers = layers
        
        # Randomly initialize weights
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / (x + y))
                        for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1: ]]


    def feed_forward(self, activation):
        z_vectors = []
        activations = [activation]
        current_layer = 0

        for weight, bias in zip(self.weights, self.biases):
            z_vector = np.dot(weight, activation) + bias
            if current_layer == self.hidden_layers:
                activation = softmax(z_vector)

            else:
                activation = sigmoid(z_vector)

            current_layer += 1
            z_vectors.append(z_vector)
            activations.append(activation)

        return z_vectors, activations
    

    def update_weights(self, batch, learning_rate):
        weight_update = [np.zeros(w.shape) for w in self.weights]
        bias_update = [np.zeros(b.shape) for b in self.biases]
        for X, y in batch:
            grad_c_wrt_bias, grad_c_wrt_weights = self.back_propagate(X, y)
            weight_update = [wu + du for wu, du in zip(weight_update, grad_c_wrt_weights)]
            bias_update = [bu + du for bu, du in zip(bias_update, grad_c_wrt_bias)]

        self.weights = [weight - (learning_rate/len(batch))*update for weight, update in zip(self.weights, weight_update)]
        self.biases = [bias - (learning_rate/len(batch))*update for bias, update in zip(self.biases, bias_update)]


    def back_propagate(self, x, y):
        z_vectors, activations = self.feed_forward(x)
        delta = activations[-1] - y
        
        grad_c_wrt_weights = [np.zeros(w.shape) for w in self.weights]
        grad_c_wrt_bias = [np.zeros(b.shape) for b in self.biases]

        grad_c_wrt_bias[-1] = delta
        grad_c_wrt_weights[-1] = np.outer(delta, activations[-2].T)

        for layer in range(2, self.layer_count):
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime(z_vectors[-layer])

            grad_c_wrt_bias[-layer] = delta
            grad_c_wrt_weights[-layer] = np.outer(delta, activations[-layer - 1].T)

        return grad_c_wrt_bias, grad_c_wrt_weights
    

    def fit(self, train_data, n_epochs, learning_rate, batch_size=16, validation_data=None, early_stop=None, return_weights_biases=False):
        '''
        SGD optimizer on mini-batches to update weights and biases
        '''
        losses, accuracies = [], []

        best_weights, best_biases = None, None

        with open('dataw.txt', 'a+') as file:
            file.truncate(0)
            for epoch in range(n_epochs):
                random.shuffle(train_data)
                batches = []
                for k in range(0, len(train_data), batch_size):
                    batches.append(train_data[k: k + batch_size])

                for batch in batches:
                    self.update_weights(batch, learning_rate)

                if validation_data:
                    res = self.evaluate(validation_data)
                    print(f'Epoch: {epoch}| Accuracy: {res[0]/len(validation_data)}, Loss: {res[1]}')
                    losses.append(res[1])
                    accuracies.append(res[0]/len(validation_data))


                    file.write(f'{epoch},{accuracies[-1]},{losses[-1]}\n')
                    file.flush()

                else:
                    print(f'Epoch: {epoch} complete!')

                if early_stop:
                    cnt_trn = early_stop.continue_training({'acc': accuracies[-1], 'loss': losses[-1]})
                    if not cnt_trn:
                        if early_stop.restore_best_weights:
                            self.weights = copy.deepcopy(best_weights)
                            self.biases = copy.deepcopy(best_biases)

                        print(f'Early stopping called, training stopped on epoch: {epoch}. No improvement in metrics for {early_stop.patience} epochs')
                        break

                    else:
                        best_weights, best_biases = self.weights, self.biases

        if return_weights_biases:
            return losses, accuracies, self.weights, self.biases
        return losses, accuracies

    def predict(self, data):
        test_results = self.feed_forward(data)[1][-1]
        
        return np.argmax(test_results)


    def evaluate(self, validation_data):
        test_results = [(self.feed_forward(x)[1][-1], y) for (x, y) in validation_data]
        loss = 0
        correct = 0
        
        for predictions, y_true in test_results:
            loss += softmax_loss(y_true, predictions)
            correct += int(np.argmax(predictions) == np.argmax(y_true))
        
        return correct, loss / len(validation_data)
    
def grid_search(param_grid: dict):
    layers = param_grid.get('layers')
    