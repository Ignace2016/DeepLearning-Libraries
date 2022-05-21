import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        all_layer_num = [input_size]
        for num in hiddens:
            all_layer_num.append(num)
        all_layer_num.append(output_size)
        layers_size = [list(i) for i in zip(all_layer_num, all_layer_num[1:])]
        self.linear_layers = [Linear(couple[0], couple[1], weight_init_fn, bias_init_fn) for couple in layers_size]
        self.z = None
        # self.linear_layers = Linear(input_size, output_size, weight_init_fn, bias_init_fn)
        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            bn_count = 0
            self.bn_layers = []
            for i in range(len(self.linear_layers)):
                if bn_count < self.num_bn_layers:
                    self.bn_layers.append(BatchNorm(layers_size[i][1]))
                    bn_count = bn_count + 1


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        out = x
        self.z = out
        for i in range(len(self.linear_layers)):
            out = self.linear_layers[i].forward(out)
            if i < self.num_bn_layers:
                if self.train_mode:
                    out = self.bn_layers[i].forward(out)
                else:
                    out = self.bn_layers[i].forward(out, True)
            out = self.activations[i].forward(out)
        return out

    def zero_grads(self):
        for i in (range(len(self.linear_layers))):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)
        if self.bn:
            for i in range(self.num_bn_layers):
                self.bn_layers[i].dgamma.fill(0.0)
                self.bn_layers[i].dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
            self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
            self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W
            self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_b
        # Do the same for batchnorm layers
        for i in range(self.num_bn_layers):
            self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
            self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta



    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        yn = self.activations[-1].state
        loss_val = self.criterion.forward(yn, labels)
        dy = self.criterion.derivative()
        dz = self.activations[-1].derivative()

        for i in range(len(self.linear_layers) - 1, -1, -1):
            dz = dy * self.activations[i].derivative()
            if self.num_bn_layers > i:
                dz = self.bn_layers[i].backward(dz)
            dy = self.linear_layers[i].backward(dz)
            dW = self.linear_layers[i].dW
            db = self.linear_layers[i].db

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False