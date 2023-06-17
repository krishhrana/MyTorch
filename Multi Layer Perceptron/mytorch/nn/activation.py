import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self):
        return (np.multiply(self.A, (1 - self.A)))



class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """

    def forward(self, Z):
        expZ = np.exp(Z)
        neg_expZ = np.exp(-Z)
        self.A = (expZ - neg_expZ) / (expZ + neg_expZ)
        return self.A

    def backward(self):
        return 1 - np.square(self.A)


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):
        self.A = np.where(Z >= 0, Z, 0)
        return self.A

    def backward(self):
        return np.where(self.A > 0, 1, 0)
