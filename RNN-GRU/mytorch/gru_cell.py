import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.bir = np.random.randn(h)
        self.biz = np.random.randn(h)
        self.bin = np.random.randn(h)

        self.bhr = np.random.randn(h)
        self.bhz = np.random.randn(h)
        self.bhn = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbir = np.zeros((h))
        self.dbiz = np.zeros((h))
        self.dbin = np.zeros((h))

        self.dbhr = np.zeros((h))
        self.dbhz = np.zeros((h))
        self.dbhn = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, bir, biz, bin, bhr, bhz, bhn):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.bir = bir
        self.biz = biz
        self.bin = bin
        self.bhr = bhr
        self.bhz = bhz
        self.bhn = bhn

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        self.r = self.r_act(np.dot(self.Wrx,x)+self.bir+np.dot(self.Wrh,h)+self.bhr)   # x: in_dim     self.Wrx:hidden_dim,in_dim
        self.z = self.z_act(np.dot(self.Wzx,x)+self.biz+np.dot(self.Wzh,h)+self.bhz)
        self.n = self.h_act(np.dot(self.Wnx,x)+self.bin+self.r*(np.dot(self.Wnh,h)+self.bhn))
        h_t = (1-self.z)*self.n+self.z*h
        
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.
        
        return h_t
        

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.h to (input_dim, 1) and (hidden_dim, 1) respectively self.x:inputdim self.h = hidden_dim
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the

        delta = np.reshape(delta, (-1, 1)) #  (hidden_dim,1)
        x = np.reshape(self.x, (1, -1))  # (1,in_dim)
        hid_pre = np.reshape(self.hidden, (-1, 1))
        r = np.reshape(self.r, (-1, 1))
        z = np.reshape(self.z, (-1, 1))
        n = np.reshape(self.n, (-1, 1))

        # compute dn
        dn = delta * (1 - z) * self.h_act.derivative(n)
        d_n_x = np.dot(dn.T, self.Wnx)
        
        self.dWnx = np.dot(dn, x)
        self.dbin = np.squeeze(dn)
        self.dWnh = np.dot(dn * r, hid_pre.T)
        self.dbhn = np.squeeze(dn * r)
        d_n_h = np.dot((dn * r).T, self.Wnh)

        # compute dz
        dz = delta * (hid_pre - n) * np.reshape(self.z_act.derivative(), (-1,1))
        self.dWzx = np.dot(dz, x)
        self.dbiz = np.squeeze(dz)
        self.dWzh = np.dot(dz, hid_pre.T)
        self.dbhz = np.squeeze(dz)
        d_z_x = np.dot(dz.T, self.Wzx)
        d_z_h = np.dot(dz.T, self.Wzh)

        # compute dr
        dr = dn * (np.dot(self.Wnh, hid_pre) + self.bhn.reshape(-1, 1))
        dr = dr * np.reshape(self.r_act.derivative(), (-1,1))
        self.dWrx = np.dot(dr, x)
        self.dbir = np.squeeze(dr)
        self.dWrh = np.dot(dr, hid_pre.T)
        self.dbhr = np.squeeze(dr)
        d_r_x = np.dot(dr.T, self.Wrx)
        d_r_h = np.dot(dr.T, self.Wrh)

        d_h_h = (delta * z).T

        dx = d_n_x + d_r_x + d_z_x
        dh = d_h_h + d_n_h + d_r_h + d_z_h

        
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh

