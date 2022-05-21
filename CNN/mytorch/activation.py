import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        res = 1/(1+np.exp(-x))
        self.state = res

        return res

    def derivative(self):
        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # res = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        res = (np.exp(x - abs(x)) - np.exp(-x - abs(x))) / (np.exp(x - abs(x)) + np.exp(-x - abs(x)))
        self.state = res
        return res

    def derivative(self):
        return 1 - self.state * self.state


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        # x[x < 0] = 0 # np.where(x < 0 ,0,x)
        x = np.where(x > 0.0, x, 0.0)
        self.state = x
        return x

    def derivative(self):
        drvtv = self.state
        drvtv[drvtv<=0] = 0
        drvtv[drvtv>0] = 1
        return drvtv
