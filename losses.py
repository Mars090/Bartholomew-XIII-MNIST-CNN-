import numpy as np

def mse(y_true, y_pred):
    """
    Calculate the Mean Squared Error between the true and predicted values.

    Arguments:
    y_true (numpy.ndarray): The true values
    y_pred (numpy.ndarray): The predicted values

    Returns:
    float: The Mean Squared Error
    """
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    """
    Calculate the derivative of the Mean Squared Error with respect to the predicted values.

    Arguments:
    y_true (numpy.ndarray): The true values
    y_pred (numpy.ndarray): The predicted values

    Returns:
    numpy.ndarray: The derivative of the Mean Squared Error
    """
    return 2 * (y_pred - y_true) / np.size(y_true)