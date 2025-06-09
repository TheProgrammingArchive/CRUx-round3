import numpy as np
import random
from functions import *
import copy
import itertools

class EarlyStopping:
    '''
    Used to implement early stopping when best model metrics are achieved

    Attributes
    -----------------------
    monitor: str
        'acc' or 'loss' to monitor accuracy or loss as early stopping metric

    patience: int (default = 1)
        Number of epochs to wait after no improvement in metric to end training

    restore_best_weights: bool (default=True)
        Restore model best weights after early stopping is called

    min_delta: float (default=0)
        Change in metric to be considered as an improvement in model parameters 

        
    Methods
    --------------------
    continue_training(epoch_results)
        Returns (bool) on whether to continue training or not based on current metric
    '''
    def __init__(self, monitor='acc', patience=1, restore_best_weights=True, min_delta=0):
        self.monitor = monitor

        assert(patience > 0)

        self.patience = patience
        self.restore_best_weights = restore_best_weights

        self.iters_to_patience = 0
        self.min_delta = min_delta

        if monitor == 'acc':
            self.curr = 0

        elif monitor == 'loss':
            self.curr = float('inf')

    def continue_training(self, epoch_results: tuple):
        if self.iters_to_patience == self.patience:
            return False, False
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
        
        return True, self.iters_to_patience == 0
    
class MLP:
    '''
    Classification MLP implemented using numpy, all hidden layers have sigmoid activation and the last layer has softmax activation. Implements SGD optimizer to 
    adjust weights and biases after each epoch.

    Attributes
    -----------------------
    layers: list
        Provide a list of integers which denotes the number of neurons in each layer. First element of the list is the input dimension (784 (28x28 img) for MNIST), 
        last element is the number of output classes (10 for MNIST). <b>example: [784, 128, 64, 10] -> initializes a model with 2 hidden layers with 128 and 64 units each </b>
    
    Methods
    ----------------------
    <b>fit(train_data, n_epochs, learning_rate, batch_size=16, validation_data=None, early_stop=None, return_weights_biases=False) \n
    predict(data) \n
    evaluate(validation_data)</b>
    '''
    def __init__(self, layers: list):
        self.layer_count = len(layers)
        self.hidden_layers = self.layer_count - 1
        self.layers = layers
        
        # Randomly initialize weights using Xavier initialization
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / (x + y))
                        for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1: ]]


    def __feed_forward(self, activation):
        '''
        Provided input activation vector, passes it through all layers in the neural network and returns a list of all activations and z vectors in each layer, 
        last element is the output activation/zvector

        Parameters
        -----------------------
        activation: np.ndarray
            input vector

        Returns
        -----------------------
        tuple(np.ndarray, np.ndarray):
            activations and z vectors
        '''
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
    

    def __update_weights(self, batch, learning_rate):
        '''
        Adjusts weights and biases after obtaining cost derivative wrt bias and weights by backpropagation.

        Parameters
        -----------------------------
        batch: list
            List of vectors, size of batch is provided in fit method

        Returns
        -----------------------------
        None
        '''
        weight_update = [np.zeros(w.shape) for w in self.weights]
        bias_update = [np.zeros(b.shape) for b in self.biases]
        for X, y in batch:
            grad_c_wrt_bias, grad_c_wrt_weights = self.__back_propagate(X, y)
            weight_update = [wu + du for wu, du in zip(weight_update, grad_c_wrt_weights)]
            bias_update = [bu + du for bu, du in zip(bias_update, grad_c_wrt_bias)]

        self.weights = [weight - (learning_rate/len(batch))*update for weight, update in zip(self.weights, weight_update)]
        self.biases = [bias - (learning_rate/len(batch))*update for bias, update in zip(self.biases, bias_update)]


    def __back_propagate(self, x, y):
        '''
        Backpropagate algorithm implementation. Provided input vector x and corresponding true y label for this vector,
        calculates gradient of cost wrt bias and gradient of cost wrt weights by chain rule

        Parameters
        -----------------------------
        x: np.ndarray
            Input vector
        y: np.ndarray
            y true labels

        Returns
        ----------------------------
        tuple(list, list):
            Gradient of cost function wrt bias and weights
        '''
        z_vectors, activations = self.__feed_forward(x)
        delta = activations[-1] - y
        
        grad_c_wrt_weights = [np.zeros(w.shape) for w in self.weights]
        grad_c_wrt_bias = [np.zeros(b.shape) for b in self.biases]

        grad_c_wrt_bias[-1] = delta
        grad_c_wrt_weights[-1] = np.dot(delta, activations[-2].T)

        for layer in range(2, self.layer_count):
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime(z_vectors[-layer])

            grad_c_wrt_bias[-layer] = delta
            grad_c_wrt_weights[-layer] = np.dot(delta, activations[-layer - 1].T)

        return grad_c_wrt_bias, grad_c_wrt_weights
    

    def fit(self, train_data, n_epochs, learning_rate, batch_size=16, validation_data=None, early_stop=None, return_weights_biases=False):
        '''
        Fits model on training data for given number of epochs and validates it on validation_data (if provided) and implements SGD algorithm 
        with provided learning rate.

        Parameters:
        train_data: list (Should be zipped along with y labels, that is each element of the list is a tuple with 0th index X and 1st index y true labels)
            Data on which the model trains on

        learning_rate: int
            Learning rate parameter for SGD algorithm (How much the weights and biases change)

        batch_size: int
            Batch size for SGD (Amount of samples fed into the MLP network)

        valiation_data: list (Should be zipped along with y labels, that is,  each element of the list is a tuple with 0th index X and 1st index y true labels)
            Data on which the model validates on, must not be None for EarlyStopping if used, no accuracy/loss metric is provided if None

        early_stop: EarlyStopping
            EarlyStopping object

        return_weights_biases: bool
            Returns weights and biases of the model if True, can be used to save model weights and biases in a binary file for future use.
        '''
        losses, accuracies = [], []

        best_weights, best_biases = None, None

        for epoch in range(n_epochs):
            random.shuffle(train_data)
            batches = []
            for k in range(0, len(train_data), batch_size):
                batches.append(train_data[k: k + batch_size])

            for batch in batches:
                self.__update_weights(batch, learning_rate)

            if validation_data:
                res = self.evaluate(validation_data)
                print(f'Epoch: {epoch}| Accuracy: {res[0]/len(validation_data)}, Loss: {res[1]}')
                losses.append(res[1])
                accuracies.append(res[0]/len(validation_data))

            else:
                print(f'Epoch: {epoch} complete!')

            if early_stop and validation_data:
                cnt_trn = early_stop.continue_training({'acc': accuracies[-1], 'loss': losses[-1]})

                if not cnt_trn[0]:
                    if early_stop.restore_best_weights:
                        self.weights = copy.deepcopy(best_weights)
                        self.biases = copy.deepcopy(best_biases)

                    print(f'Early stopping called, training stopped on epoch: {epoch}. No improvement in metrics for {early_stop.patience} epochs')
                    break

                else:
                    if cnt_trn[1]:
                        best_weights, best_biases = copy.deepcopy(self.weights), copy.deepcopy(self.biases)

        if return_weights_biases:
            return losses, accuracies, self.weights, self.biases
        return losses, accuracies

    def predict(self, data):
        '''
            Predict class of provided data, accepts only 1 data sample.

            Parameters
            ------------------------
            data: np.ndarray
                Data sample to predict class label for

            Returns
            -------------------------
            int: predicted label
        '''
        test_results = self.__feed_forward(data)[1][-1]
        
        return np.argmax(test_results)


    def evaluate(self, validation_data):
        '''
        Provided validation_data, returns number of correctly identifed samples and cross-entropy loss. 

        Parameters:
        -----------------------
        validation_data: list (Should be zipped along with y labels, that is, each element of the list is a tuple with 0th index X and 1st index y true labels)

        Returns:
        -----------------------
        tuple(int, float): Number of correctly identified samples and cross-entropy loss
        '''
        test_results = [(self.__feed_forward(x)[1][-1], y) for (x, y) in validation_data]
        loss = 0
        correct = 0
        
        for predictions, y_true in test_results:
            loss += softmax_loss(y_true, predictions)
            correct += int(np.argmax(predictions) == np.argmax(y_true))
        
        return correct, loss / len(validation_data)
    
def grid_search(train_data, test_data, param_grid: dict, metric='acc') -> MLP:
    '''
    Implements grid search to find and return best model weights and biases and the best parameters

    Parameters
    ---------------------
    train_data: list (Expects X and y values to be zipped together), (for MNIST) Dimensions of each X: (784, 1). Dimensions of each Y: (10, 1)
        Data to train model on
    test_data:  list (Expects X and y values to be zipped together), (for MNIST) Dimensions of each X: (784, 1). Dimensions of each Y: (10, 1)
        Data to evaluate model on
    param_grid: dict
        Parameters: layers, list of model layers. <b>example: [[784, 64, 10], [784, 128, 64, 10]]</b> \n
                    learning_rate: list of learning rates. <b>example: [0.1, 1]</b> \n
                    batch_size: list of batch sizes. <b>example: [32, 64]</b> \n
                    n_epochs: list of number of epochs <b>example: [10, 20]</b>

    Raises
    -----------------------
    Exception: if any of the grid search parameters are missing

    Returns
    -----------------------
    tuple(MlP, dict)
    '''
    layer_combs = param_grid.get('layers')
    learning_rates = param_grid.get('learning_rate')
    batch_sizes = param_grid.get('batch_size')
    epochs = param_grid.get('n_epochs')

    if layer_combs is None or learning_rates is None or batch_sizes is None or train_data is None or test_data is None:
        raise Exception("Grid search paramters cannot be none, validation data must be provided")
    
    combined = [layer_combs, learning_rates, batch_sizes, epochs]
    combinations = itertools.product(*combined)

    weights, biases = None, None
    layers, lr, bsz, ne = None, None, None, None
    best_metric = float('inf') if metric == 'loss' else 0
    for comb in combinations:
        print(f'Training with combination: {comb}')
        l, a, w, b = MLP(layers=comb[0]).fit(train_data=train_data, validation_data=test_data, n_epochs=comb[3], learning_rate=comb[1], batch_size=comb[2], 
                                        return_weights_biases=True, early_stop=EarlyStopping(metric, 5, True, 0))
        
        if metric == 'loss':
            if l[-5] < best_metric:
                best_metric = l[-5]
                weights, biases = w, b
                layers = comb[0]
                lr = comb[1]
                bsz = comb[2]
                ne = comb[3]

        else:
            if a[-5] > best_metric:
                best_metric = a[-5]
                weights, biases = w, b
                layers = comb[0]
                lr = comb[1]
                bsz = comb[2]
                ne = comb[3]

        print(f'Best metric so far: {best_metric} {metric}')

    best_model = MLP(layers)
    best_model.weights, best_model.biases = weights, biases

    return best_model, {'layers': layers, 'n_epochs': ne, 'batch_size': bsz, 'learning_rate': lr}