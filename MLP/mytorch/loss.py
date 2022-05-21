import numpy as np
import os

class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.intermediate = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        res_softmax = self.softmax(x)
        res_ce = self.crossentropy(res_softmax, self.labels)
        # self.intermediate = res_softmax - self.labels # 28号1点29改了
        self.intermediate = res_softmax
        return res_ce

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        self.loss = self.intermediate - self.labels
        return self.loss

    def softmax(self, x):
        if x.ndim == 1:
            axis = 0
        else:
            axis = 1
        max_in_row = np.max(x, axis=axis, keepdims=True)
        exp_input = np.exp(x - max_in_row)
        sum_in_row = np.sum(exp_input, axis=axis, keepdims=True)
        res = exp_input / sum_in_row
        return res

    def crossentropy(self, y_data, gt_data):
        if y_data.ndim == 1:
            return -np.log(y_data[gt_data.argmax()])
        else:
            index_map = gt_data.argmax(axis=1)
            res = np.array([y_data[i, j] for i, j in enumerate(index_map)])
            return -np.log(res)


class L2Loss(Criterion):
    """
    L2 loss
    """

    def __init__(self):
        super(L2Loss, self).__init__()
        self.intermediate = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        res = 0.5*(x-y) * (x-y)
        self.intermediate = res
        return res

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return self.intermediate
