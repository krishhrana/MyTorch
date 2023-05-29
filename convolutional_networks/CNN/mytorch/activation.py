import numpy as np


class Identity:
    """
    Indentity activation function
    f(Z) = Z
    """

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    Sigmoidal actiavtion
    f(Z) = (1 / (1 + e^-Z))
    forward() Returns value between (0, 1)
    backward returns the derivative in the backprop graph
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self):
        return (np.multiply(self.A, (1 - self.A)))



class Tanh:
    """
    Tanh activation
    f(Z) = (e^Z - e^-Z) / (e^Z + e^-Z)
    forward() calculated the tanh activated value. Returns value between (-1, 1)
    backward() calculates the derivative for the function
    """

    def forward(self, Z):
        '''expZ = np.exp(Z)
        neg_expZ = np.exp(-Z)
        self.A = (expZ - neg_expZ) / (expZ + neg_expZ)'''
        self.A = np.tanh(Z)
        return self.A

    def backward(self):
        return 1 - np.square(self.A)


class ReLU:
    """
    ReLU activation
    f(Z) = max(0, Z)
    forward() returns a ReLU activated value. The function clamps all the negative values at 0 and identity function for x > 0
    backward() returns the derivative.
    """

    def forward(self, Z):
        self.A = np.where(Z >= 0, Z, 0)
        return self.A

    def backward(self):
        return np.where(self.A > 0, 1, 0)
