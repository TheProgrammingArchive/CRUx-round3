import numpy as np
import keras
from mlp import MLP, EarlyStopping, grid_search
from sklearn.linear_model import SGDClassifier
from liveplot import liveplot

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

# # Custom-model, Learning with no graph
# model = MLP([784, 128, 64, 10])
# estop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
# l, a = model.fit(train_data=train_data, n_epochs=10, learning_rate=0.1, batch_size=32, validation_data=test_data[:2500], early_stop=estop)

# # Custom-model, Learning with graph
# model = MLP([784, 128, 64, 10])
# liveplot(model, train_data=train_data, validation_data=test_data, learning_rate=0.1, batch_size=32, n_epochs=5)


# best_model, best_params = grid_search(train_data=train_data, test_data=test_data, param_grid={'layers': [[784, 32, 10], [784, 128, 64, 10]], 'n_epochs': [10], 'learning_rate': [1, 0.1], 'batch_size': [16, 32]})


# Best accuracy 0.8841 using grid search