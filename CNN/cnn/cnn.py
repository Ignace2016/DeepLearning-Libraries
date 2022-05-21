import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *

class CNN(object):
    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class
        """

        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr


        self.convolutional_layers = [Conv1D(24, 56, 5, 1),Conv1D(56, 28, 6, 2),Conv1D(28, 14, 2, 2)] # # in_channel, out_channel, kernel_size, stride
        self.flatten = Flatten()

        self.linear_layer = Linear(420, 10,weight_init_fn=linear_weight_init_fn, bias_init_fn=bias_init_fn)


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """

        ## Your code goes here -->
        # Iterate through each layer
        # <---------------------
        for i in range(len(self.convolutional_layers)):
            out = self.convolutional_layers[i](x)
            x = self.activations[i](out)
        # Save output (necessary for error and loss)
        x = self.flatten(x)
        x = self.linear_layer(x)
        self.output = x
        return self.output


    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()

        # Iterate through each layer in reverse order
        # <---------------------
        grad = self.linear_layer.backward(grad)
        grad = self.flatten.backward(grad)
        for i in range(len(self.convolutional_layers) - 1, -1, -1):
            grad = grad * self.activations[i].derivative()
            grad = self.convolutional_layers[i].backward(grad)
        return grad


    def zero_grads(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
