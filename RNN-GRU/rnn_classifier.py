import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = [
            RNNCell(input_size, hidden_size)
            if i == 0
            else RNNCell(hidden_size, hidden_size)
            for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [[W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                    [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1], ...]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(*linear_weights)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size)

        Output: logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros(
                (self.num_layers, batch_size, self.hidden_size), dtype=float
            )
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())

        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
        #   Iterate over the length of your self.rnn (through the layers)
        #       Run the rnn cell with the correct parameters and update
        #       the parameters as needed. Update hidden.
        #   Similar to above, append a copy of the current hidden array to the hiddens list

        # Get the outputs from the last time step using the linear layer and return it
        # logits =
        # <--------------------------
        for t in range(seq_len):
            # hidden: num_layers, batch_size, hidden_size
            # x: (batch_size, seq_len, input_size)
            # hidden[0,:,:] = self.rnn[0](x[:,t,:],self.hiddens[0][0])
            h_1 = x[:, t, :]
            for l in range(0, self.num_layers):
                hidden[l,:,:] = self.rnn[l](h_1,self.hiddens[t][l]) # 1. h(t,l-1) 2. h(t-1,l)
                h_1 = hidden[l,:,:]
            self.hiddens.append(hidden.copy())
        logits = self.output_layer(self.hiddens[-1][-1])
        return logits
    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        # formula: h_prime = self.activation(np.dot(x, self.W_ih.T) + self.b_ih + np.dot(h, self.W_hh.T) + self.b_hh)
        # shape of self.hiddens = [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        for t in range(seq_len,0,-1):
            for l in range(self.num_layers-1,0,-1):
                dx, dh[l] = self.rnn[l].backward(dh[l], self.hiddens[t][l], self.hiddens[t][l-1], self.hiddens[t-1][l])
                dh[l-1] += dx
                # h: (batch_size, hidden_size)
                #             Hidden state of the current time step and the current layer
                # h_prev_l: (batch_size, input_size)
                #            Hidden state at the current time step and previous layer
                # h_prev_t: (batch_size, hidden_size)
                #            Hidden state at previous time step and current layer
            dx1, dh[0] = self.rnn[0].backward(dh[0],self.hiddens[t][0], self.x[:,t-1,:], self.hiddens[t-1][0])
        dh_0 =  dh / batch_size

        return dh_0

