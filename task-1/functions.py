import numpy as np

def sigmoid(z):
    '''
    Applies sigmoid function to vector and returns it

    Parameters
    -------------------
    z: np.ndarray
        input vector

    Returns
    -------------------
    np.ndarray
    '''
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    '''
    Applies derivate of sigmoid function to vector and returns it

    Parameters
    ------------------
    z: np.ndarray
        input vector

    Returns
    ---------------
    np.ndarray
    '''
    return sigmoid(z)*(1 - sigmoid(z))

def softmax(z):
    '''
    Applies softmax function to vector and returns it

    Parameters
    ------------------
    z: np.ndarray
        input vector

    Returns
    ---------------
    np.ndarray
    '''
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def sigmoid_loss(y, y_pred):
    '''
    Calculates and returns log loss given yhat (y_pred) and ytrue

    Parameters
    -------------------
    y: np.ndarray
        y_true, the true labels \n
    y_pred: np.ndarray 
        predicted labels 

    Returns
    -------------
    loss: np.float64
    '''
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

def softmax_loss(y, y_pred):
    '''
    Calculates cross-entropy loss given yhat (y_pred) and ytrue

    Parameters
    -------------------
    y: np.ndarray
        y_true, the true labels \n
    y_pred: np.ndarray 
        predicted labels 

    Returns
    -----------------
    loss: np.float64
    '''
    loss = -np.sum(y * np.log(y_pred + 1e-15))
    return loss