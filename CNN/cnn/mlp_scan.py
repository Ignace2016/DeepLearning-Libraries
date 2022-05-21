import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        # input_width = 128
        self.conv1 = Conv1D(24, 8, 8, 4) # in_channel, out_channel, kernel_size, stride
        self.conv2 = Conv1D(8, 16, 1, 1)
        self.conv3 = Conv1D(16, 4, 1, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()] # Linear(8 * 24, 8), ReLU(), Linear(8, 16), ReLU(), Linear(16, 4)

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        w1,w2,w3 = weights
        self.conv1.W = w1.reshape(self.conv1.W.T.shape).T # w: out_channel, in_channel, kernel_size
        self.conv2.W[:,:,0] = w2.T
        self.conv3.W[:,:,0] = w3.T

    def forward(self, x):
        """

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1D(24, 2, 2, 2) # in_channel, out_channel, kernel_size, stride24 8 2 2
        self.conv2 = Conv1D(2, 8, 2, 2) # 8 16 4 1
        self.conv3 = Conv1D(8, 4, 2, 1) # 16 4 2 1
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.W = w1[:48,:2].reshape(2,24,2).transpose()
        self.conv2.W = w2[:4, :8].reshape(2,2,8).transpose()
        self.conv3.W = w3.reshape(2,8,4).transpose()

    def forward(self, x):
        """

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
