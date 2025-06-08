from datetime import datetime
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from random import randrange


import numpy as np
import keras
from mlp import MLP, EarlyStopping, grid_search
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
modelr = MLP([784, 128, 64, 10])

def test(model, train_data, n_epochs, learning_rate, batch_size, validation_data):
    x_data, y_data, y1_data = [], [], []

    figure, ax = pyplot.subplots()
    line, = ax.plot(x_data, y_data, '-', label='loss')
    liner, = ax.plot(x_data, y1_data, '--', label='accuracy')

    assert(n_epochs >= 0)

    def update(frame):
        l, a = model.fit(train_data=train_data, n_epochs=1, learning_rate=learning_rate, batch_size=batch_size, validation_data=validation_data)
        x_data.append(len(x_data))
        y_data.append(l[-1])
        y1_data.append(a[-1])

        line.set_data(x_data, y_data)
        liner.set_data(x_data, y1_data)

        ax.relim()
        ax.autoscale_view()

        if len(x_data) == n_epochs:
            animation.event_source.stop()
            x = input('Training complete, press any key to terminate plot')
            pyplot.close(figure)

        return line, liner

    animation = FuncAnimation(figure, update, interval=200, cache_frame_data=False)

    pyplot.show()

test(modelr, train_data=train_data, n_epochs=3, learning_rate=0.1, batch_size=16, validation_data=test_data[:1000])