import numpy as np
import math
import loss



class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.momentum_W = np.zeros(self.W.shape)
        self.momentum_b = np.zeros(self.b.shape)
        self.state = None
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        z = np.dot(x, self.W) + self.b
        self.state = x
        return z

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        batch_size = delta.shape[0]
        dx = np.dot(delta, self.W.T) # (batch size, out feature) * (out feature, in feature), which should be (batch size, in feature)
        self.dW = np.dot(self.state.T, delta) / batch_size # (in feature, batch size) * (batch size, out feature)
        self.db = np.dot(np.ones((1, batch_size)), delta) / batch_size # (1,batch size) * (batch size, out feature)
        return dx
