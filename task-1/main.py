import numpy as np
import keras
from mlp import MLP
from sklearn.linear_model import SGDClassifier

def prepare_data(data):
    X, y = data
    X = X.astype('float64')/255.
    X = [np.reshape(x, (784, 1)) for x in X]

    y_fin = [None for k in range(len(X))]
    for idx, y_ in enumerate(y):
        y_vec = np.zeros((10, 1))
        y_vec[y_] = 1
        y_fin[idx] = y_vec

    return list(zip(X, y_fin))

data = keras.datasets.fashion_mnist.load_data()
train_data, test_data = data

train_data = prepare_data(train_data)
test_data = prepare_data(test_data)

# Custom-model
model = MLP([784, 128, 64, 10], output_activation='softmax')
model.fit_model(train_data=train_data, n_epochs=50, learning_rate=0.1, batch_size=32, validation_data=test_data[:2500])

# Final accuracy = 0.8976