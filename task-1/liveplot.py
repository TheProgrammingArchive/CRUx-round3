from datetime import datetime
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from random import randrange

def liveplot(model, train_data, n_epochs, learning_rate, batch_size, validation_data):
    '''
    Plots live graphs showing accuracy and loss vs epoch 

    Parameters
    --------------------------
    model: MLP object \n
    train_data: Training data \n
    n_epochs: Epochs to train for \n
    learning_rate: SGD parameter \n
    batch_size: mini-batch size for SGD \n
    validation_data: test_data to validate model against

    Returns
    ----------------------------
    None
    '''
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