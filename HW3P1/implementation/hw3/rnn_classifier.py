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
             RNNCell(input_size, hidden_size) if i == 0
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
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

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

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())

        for seq in range(seq_len):
            x_init = x[:, seq, :]
            hidden_state = np.zeros(shape = (self.num_layers, batch_size, self.hidden_size))
        #   Iterate over the length of your self.rnn (through the layers)
            for l in range(self.num_layers):
        #       Run the rnn cell with the correct parameters and update
                h_t = self.rnn[l].forward(x_init, self.hiddens[seq][l])
                x_init = h_t.copy()
                hidden_state[l] = h_t.copy()
        #       the parameters as needed. Update hidden.
            self.hiddens.append(hidden_state.copy())

        logits = self.output_layer.forward(self.hiddens[-1][self.num_layers - 1])
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

        for seq in range(seq_len - 1, -1, -1):
            dx = np.zeros_like(dh[-1], dtype=float)
            for l in range(self.num_layers - 1, 0, -1):
                dh[l] += dx
                # Get h_prev_l either from hiddens or x depending on the layer
                dx, dh_prev = self.rnn[l].backward(dh[l], self.hiddens[seq + 1][l], self.hiddens[seq + 1][l-1], self.hiddens[seq][l])
                dh[l] = dh_prev   #dh[1] = dh_prev

            dh[0] += dx  # dh[0]
            dx, dh_prev = self.rnn[0].backward(dh[0], self.hiddens[seq + 1][0], self.x[:, seq, :], self.hiddens[seq][0])
            dh[0] = dh_prev   #dh[0]=

        return dh / batch_size