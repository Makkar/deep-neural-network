"""
Utils for dnn.py
Author: Aditya Makkar
"""

import numpy as np

def sigmoid(z):
    """
    Returns the sigmoid z.
    """
    s = 1 / (1 + np.exp(-z))
    return s

def dsigmoid(z):
    """
    Derivative of sigmoid.
    """
    s = z * (1.0 - z)
    return s

def relu(z):
    """
    Returns the ReLU of z.
    """
    s = z * (z > 0)
    return s

def drelu(z):
    """
    Derivative of ReLU.
    """
    s = 1.0 * (z > 0)
    return s

def tanh(z):
    """
    Returns the tanh of z.
    """
    s = np.tanh(z)
    return s

def dtanh(z):
    """
    Derivative of tanh.
    """
    s = 1.0 - z * z
    return s