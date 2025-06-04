import numpy as np
import random
from functions import *
import copy

class MLP:
    def __init__(self, layers: list, output_activation: str):
        '''
            layers: Provide number of neurons per layer in form of a list, [input_layers, hidden1, hidden2, ..., hiddenk, output]
            output_activation: Change activation function depending on task (classification/regression), optimal loss function is chosen automatically
        '''
        self.layer_count = len(layers)
        self.hidden_layers = self.layer_count - 1
        self.layers = layers
        self.output_activation = output_activation
        
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
                if self.output_activation == 'sigmoid':
                    activation = sigmoid(z_vector)

                elif self.output_activation == 'softmax':
                    activation = softmax(z_vector)

                else:
                    activation = z_vector

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
        if self.output_activation == 'sigmoid':
            # MSE loss
            delta = (activations[-1] - y)*sigmoid_prime(z_vectors[-1])
        elif self.output_activation == 'softmax':
            # Assumes cross-entropy loss
            delta = (activations[-1] - y) 
        else:
            # MSE loss
            delta = (activations[-1] - y)
        
        grad_c_wrt_weights = [np.zeros(w.shape) for w in self.weights]
        grad_c_wrt_bias = [np.zeros(b.shape) for b in self.biases]

        grad_c_wrt_bias[-1] = delta
        grad_c_wrt_weights[-1] = np.outer(delta, activations[-2].T)

        for layer in range(2, self.layer_count):
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime(z_vectors[-layer])
            grad_c_wrt_bias[-layer] = delta
            grad_c_wrt_weights[-layer] = np.outer(delta, activations[-layer - 1].T)

        return grad_c_wrt_bias, grad_c_wrt_weights
    

    def SGD(self, train_data, n_epochs, learning_rate, batch_size=16, validation_data=None, early_stop_patience: int=None):
        '''
        SGD optimizer on mini-batches to update weights and biases
        '''
        losses, accuracies = [], []
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
                    if self.output_activation in ['sigmoid', 'softmax']:
                        print(f'Epoch: {epoch}| Accuracy: {res[0]/len(validation_data)}, Loss: {res[1]}')
                        losses.append(res[1])
                        accuracies.append(res[0]/len(validation_data))

                    else:
                        losses.append(res[1])
                        accuracies.append(0)
                        print(f'Epoch: {epoch}| Loss: {res[1]}')

                    file.write(f'{epoch},{accuracies[-1]},{losses[-1]}\n')
                    file.flush()

                else:
                    print(f'Epoch: {epoch} complete!')


        return losses, accuracies
        #To-implement: early stopping 

    def predict(self, data):
        test_results = self.feed_forward(data[0])[1][-1]
        if self.output_activation == 'softmax' or self.output_activation == 'sigmoid':
            return np.argmax(test_results)
        
        return test_results
    
        
    def fit_model(self, train_data, n_epochs, learning_rate, batch_size=16, validation_data=None, early_stop_patience: int=None):
        return self.SGD(train_data=train_data, n_epochs=n_epochs, learning_rate=learning_rate, batch_size=batch_size, validation_data=validation_data, early_stop_patience=early_stop_patience)


    def evaluate(self, validation_data):
        if self.output_activation == 'softmax' or self.output_activation == 'sigmoid':
            test_results = [(self.feed_forward(x)[1][-1], y) for (x, y) in validation_data]
            loss = 0
            correct = 0
            
            for predictions, y_true in test_results:
                if self.output_activation == 'sigmoid':
                    y_binary = np.argmax(y_true) if len(y_true.shape) > 0 and y_true.shape[0] > 1 else y_true
                    loss += sigmoid_loss(y_binary, predictions)
                    predicted_class = 1 if predictions > 0.5 else 0
                    correct += int(predicted_class == y_binary)
                    
                else:
                    loss += softmax_loss(y_true, predictions)
                    correct += int(np.argmax(predictions) == np.argmax(y_true))
            
            return correct, loss / len(validation_data)
        
        else:
            total_loss = 0
            for x, y in validation_data:
                prediction = self.feed_forward(x)[1][-1]
                total_loss += 0.5 * (prediction - y)**2
            
            return prediction, total_loss / len(validation_data)