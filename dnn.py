"""
L layered feedforward neural network for classification.

Author: Aditya Makkar
"""

import numpy as np
from utils import *
import matplotlib.pyplot as plt

"""
This class can be used to implement an L layered neural network.
It takes as input a list of layer sizes and optionally a list of activation 
functions to be used.
"""
class DNN():
    
    def __init__(self, layer_dims, layer_activations=None, verbose=False):
        assert isinstance(layer_dims, list), 'Error: layer_dims must be a list'
        assert len(layer_dims) >= 2, 'Error: layer_dims must be of length at least 2'
        self.L = len(layer_dims) - 1 # Number of hidden layers + output layer
        self.layer_dims = layer_dims # Size of all layers
        if layer_activations is None:
            self.layer_activations = [''] + ['relu'] * (self.L-1) + ['sigmoid']
        else:
            assert len(layer_activations) == self.L
            self.layer_activations = layer_activations
        self.verbose = verbose
    
    def _initialize_params(self):
        """
        Initialize parameters W and b of each layer.
        W is initialized to values drawn from normal distribution to break symmetry.
        See here: https://stackoverflow.com/questions/20027598/why-should-weights-of-neural-networks-be-initialized-to-random-numbers
        The multiplication factor of 0.01 is to prevent gradient vanishing.
        """
        self.params = {}
        for l in range(1, (self.L + 1)):
            self.params['W' + str(l)] = 0.01 * np.random.randn(self.layer_dims[l], self.layer_dims[l-1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
    
    def _initialize_params_he(self):
        """
        He initialization.
        """
        self.params = {}
        for l in range(1, (self.L + 1)):
            self.params['W' + str(l)] = (2 / self.layer_dims[l-1])**0.5 * \
            np.random.randn(self.layer_dims[l], self.layer_dims[l-1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
    
    def _initialize_params_xavier(self):
        """
        Xavier initialization.
        """
        self.params = {}
        for l in range(1, (self.L + 1)):
            self.params['W' + str(l)] = (1 / self.layer_dims[l-1])**0.5 * \
            np.random.randn(self.layer_dims[l], self.layer_dims[l-1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
    
    def _initialize_params_mix(self):
        """
        He or Xavier initialization.
        """
        self.params = {}
        for l in range(1, (self.L + 1)):
            if self.layer_activations[l] == 'relu':
                x = 2
            else:
                x = 1
            self.params['W' + str(l)] = (x / self.layer_dims[l-1])**0.5 * \
            np.random.randn(self.layer_dims[l], self.layer_dims[l-1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
    
    def _forward(self):
        """
        One step of forward propagation.
        We cache intermediate values which will be used in backpropagation.
        """
        self.cache = []
        self.cache.append(())
        A = self.X
        L = self.L
        # For hidden layers
        for l in range(1, L):
            A_prev = A
            W, b = self.params['W' + str(l)], self.params['b' + str(l)]
            if self.layer_activations[l] == 'relu':
                g = relu
            elif self.layer_activations[l] == 'tanh':
                g = tanh
            else:
                raise Exception('Use either "relu" or "tanh" for hidden layers.')
            Z = np.dot(W, A_prev) + b
            A = g(Z)
            self.cache.append((A_prev, W, b, Z)) # Caching for backprop
        # For output layer
        W, b = self.params['W' + str(L)], self.params['b' + str(L)]
        Z = np.dot(W, A) + b
        self.AL = sigmoid(Z)
        self.cache.append((A, W, b, Z))
    
    def _get_cost(self):
        """
        Using the output produced and given labels, compute the cost function.
        Cost function is cross entropy error.
        """
        m = self.Y.shape[1]
        cost = (-1/m)*np.sum(np.multiply(self.Y, np.log(self.AL)) + np.multiply(1-self.Y, np.log(1-self.AL)))
        return np.squeeze(cost)
    
    def _backprop_one_layer(cache, dA_next, gd):
        """
        Helper function for _backprop.
        This function does backpropagation on one layer.
        """
        A_prev, W, b, Z = cache
        dZ = np.multiply(dA_next, gd(Z))
        m = A_prev.shape[1]
        dA_prev = np.dot(W.T, dZ)
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        return (dA_prev, dW, db)
    
    def _backprop(self):
        """
        One step of backward propagation.
        Parameters which were cached in forward propagation steps are used here.
        Gradients of parameters are cached here to be used in gradient descent.
        """
        grads = {} # Gradients of W's and b's
        L = self.L
        # This is computed by taking derivative of the cost function wrt AL
        dAL = - (np.divide(self.Y, self.AL) - np.divide(1 - self.Y, 1 - self.AL))
        # For output layer
        grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = \
        DNN._backprop_one_layer(self.cache[L], dAL, dsigmoid)
        # For hidden layers
        for l in reversed(range(1, L)):
            if self.layer_activations == 'relu':
                gd = drelu
            else:
                gd = dtanh
            grads['dA' + str(l)], grads['dW' + str(l)], grads['db' + str(l)] = \
            DNN._backprop_one_layer(self.cache[l], grads['dA' + str(l+1)], gd)
        self.grads = grads
    
    def _update(self, lr):
        """
        Update all parameters using gradient descent.
        """
        L = self.L
        for l in range(1, (L+1)):
            self.params['W' + str(l)] -= lr * self.grads['dW' + str(l)]
            self.params['b' + str(l)] -= lr * self.grads['db' + str(l)]
    
    def train(self, X, Y, num_iter=3000, lr=0.0075, verbose=False):
        """
        Train the neural network.
        
        X - A numpy array containing features for all training examples.
        X.shape[0] should be number of training examples. For example, for 100 
        training images of shape 200 x 200 pixels with 3 RGB components, 
        X.shape = (100, 200, 200, 3).
        
        Y - Training labels. Shape should be (m,), (m, 1), or (1, m).
        """
        m = X.shape[0] # Number of training examples
        X = X.reshape(m , -1).T
        Y = Y.reshape(1, m)
        self.X = X
        self.Y = Y
        self._initialize_params_mix()
        costs = []
        # Loop to perform forward propagation, backward propagataion and 
        # gradient descent num_iter times.
        for i in range(num_iter):
            self._forward()
            cost = self._get_cost()
            costs.append(cost)
            self._backprop()
            self._update(lr)
            if (verbose or self.verbose) and i % 100 == 0:
                print("Cost after iteration {0}: {1}".format(i, cost))
        if verbose or self.verbose:
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(lr))
            plt.show()
    
    def predict(self, X):
        """
        Using the trained model, predict on X.
        """
        m = X.shape[0]
        X = X.reshape(m, -1).T
        A = X
        L = self.L
        # For hidden layers
        for l in range(1, L):
            A_prev = A
            W, b = self.params['W' + str(l)], self.params['b' + str(l)]
            if self.layer_activations[l] == 'relu':
                g = relu
            elif self.layer_activations[l] == 'tanh':
                g = tanh
            else:
                raise Exception('Use either "relu" or "tanh" for hidden layers.')
            Z = np.dot(W, A_prev) + b
            A = g(Z)
        # For output layer
        W, b = self.params['W' + str(L)], self.params['b' + str(L)]
        Z = np.dot(W, A) + b
        AL = sigmoid(Z)
        Y = np.where(AL > 0.5, 1, 0)
        return Y
