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

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r_x = self.Wrx @ self.x + self.brx
        self.r_h = self.Wrh @ h_prev_t + self.brh
        self.r = self.r_act(self.r_x + self.r_h)

        self.z_x = self.Wzx @ self.x + self.bzx
        self.z_h = self.Wzh @ h_prev_t + self.bzh
        self.z = self.z_act(self.z_x + self.z_h)

        self.n_x = self.Wnx @ self.x + self.bnx
        self.n_h = self.Wnh @ h_prev_t + self.bnh
        self.n_pre_act = self.n_x + self.r * (self.n_h)
        self.n = self.h_act(self.n_pre_act)

        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

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

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        self.x = self.x.reshape(1, -1)
        h = self.hidden.reshape(1, -1)
        z = self.z.reshape(1, -1)
        n = self.n.reshape(1, -1)
        r = self.r.reshape(1, -1)
        n_h = self.n_h.reshape(1, -1)

        tanh_backward = self.h_act.backward(n).reshape(1, -1)
        z_backward = self.z_act.backward().reshape(1, -1)
        r_backward = self.r_act.backward().reshape(1, -1)

        dn = delta * (1 - z)
        dz = delta * (h - n)
        dr = dn * tanh_backward * (n_h)

        # dx
        par_dn = dn @ (tanh_backward.T * self.Wnx)
        par_dz = dz @ (z_backward.T * self.Wzx)
        par_dr = dr @ (r_backward.T * self.Wrx)

        dx = par_dn + par_dz + par_dr
        print(dx.shape)

        # dh_prev_t
        par_dh = delta * z
        par_dnh = (dn * tanh_backward * r) @ self.Wnh
        par_dzh = (dz * z_backward) @ self.Wzh
        par_drh = (dr * r_backward) @ self.Wrh

        dh_prev_t = par_dh + par_dnh + par_dzh + par_drh

        # dWrx
        par_r = (dr * r_backward).T
        self.dWrx = par_r @ self.x

        # dWzx
        par_z = (dz * z_backward).T
        self.dWzx = par_z @ self.x

        # d_wnx
        self.dWnx = (dn * tanh_backward).T @ self.x

        # dWrh
        self.dWrh = par_r @ h

        # dWzh
        self.dWzh = par_z @ h

        # d_wnh
        self.dWnh = (dn * tanh_backward * r).T @ h

        # dbrx
        self.dbrx = par_r.T

        # dbzx
        self.dbzx = par_z.T

        # d_bnx
        self.dbnx = dn * tanh_backward

        # dbrh
        self.dbrh = par_r.T

        # dbzh
        self.dbzh = par_z.T

        # d_bnh
        self.dbnh = dn * tanh_backward * r

        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t