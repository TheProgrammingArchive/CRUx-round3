# MLP from scratch using numpy 
  Provides implementions of the following:
  1. A classification MLP using numpy. All hidden layers apply sigmoid activation and the output layer applies softmax activation.
  2. Early stopping
  3. Generating live plots of loss/accuracy during training.

  The mlp is pretty lightweight, and can be used for simple classfication tasks

## Dependencies 
  keras==3.10.0 <br>
  matplotlib==3.10.3 <br>
  numpy==2.3.0

## Usage (All code samples are based on MNIST/FashionMNIST)
  1. After cloning the repository, install all dependencies.
  2. <br>
  3. Import model from MLP module (EarlyStopping, grid_search if required)
     ```python
     from mlp import MLP, EarlyStopping, grid_search
     ```
     <br>
  4. For using the model with a custom dataset, follow all data-preprocessing steps as one normally would. <br>
       The model accepts training and validation data as a zipped list, that is, X and y labels must be zipped together. Y labels must be one-hot-encoded and each X should be flattened. <br>
       **Both X and y must have dimensions of (K, 1) not (K, ). For example: K is 784 for X in MNIST (28X28 img) and K is 10 for y (10 class labels)**
     <br>
  6. Initialize a new model as follows:
     ```python
     model = MLP([784, 128, 64, 10])
     ```
     First element of the list is the input size (28x28 -> 784) and last element is the number of class labels. The model has two hidden layers with 128 and 64 units each.
     <br>
  7. Fit the model on training data (without EarlyStopping):
     ```python
     loss, acc = model.fit(train_data=train_data, n_epochs=10, learning_rate=0.1, batch_size=32, validation_data=test_data)
     ```
     Prints current model metrics after each epoch, fit returns a list of loss and accuracies after every epoch, this can be used to plot loss/acc vs epoch. <br>
     Tweak model hyperparameters until best metrics are obtained or use grid search (8).
     <br>
  8. Fit the model on training data (with EarlyStopping):
     ```python
     model = MLP([784, 128, 64, 10])
     model = MLP([784, 128, 64, 10])
     estop = EarlyStopping(monitor='acc', patience=2, restore_best_weights=True)
     loss, acc = model.fit(train_data=train_data, n_epochs=10, learning_rate=0.1, batch_size=32, validation_data=test_data[:2500], early_stop=estop)
     ```
     Intialize an EarlyStopping object, choose whether to optimize model on accuracy (monitor='acc) or loss (monitor='loss'). Patience determines the number of epochs to wait after no improvement in metrics to terminate training. <br>
     restore_best_weights if True will return model to weights and biases that produced best metric, will skip this if False.
    <br>
9. Fit the model on training data (Live plot):
   ```python
   from liveplot import liveplot
   model = MLP([784, 128, 64, 10])
   liveplot(model, train_data=train_data, validation_data=test_data, learning_rate=0.1, batch_size=32, n_epochs=5)
   ```
   <br>
10. Grid search to find and return best model and model hyperparameters
   ```python
   from mlp import grid_search
   best_model, best_params = grid_search(train_data=train_data, test_data=test_data, param_grid={'layers': [[784, 32, 10], [784, 128, 64, 10]], 'n_epochs': [5, 10], 'learning_rate': [1, 0.1], 'batch_size': [16, 32]})
   ```
   Refer to grid_search docs for parameter descriptions.

11. Further, refer to main.py for all of the above examples.

## Features implemented  
* MLP using numpy
* Training MLP on FashionMnist
* Grid search
* EarlyStopping
* Live plots of accuracy and loss

## References 
http://neuralnetworksanddeeplearning.com/ for the math behind MLPs
