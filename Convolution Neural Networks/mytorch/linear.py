import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros(shape=(out_features, in_features))  # TODO
        self.b = np.zeros(shape=(out_features, 1))  # TODO
        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = A.shape[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N, 1))
        Z = np.matmul(self.A, np.transpose(self.W)) + np.matmul(self.Ones, np.transpose(self.b))
        return Z

    def backward(self, dLdZ):

        dZdA = np.transpose(self.W)  # TODO
        dZdW = self.A  # TODO
        dZdb = np.ones(shape=(self.N, 1))  # TODO

        dLdA = np.matmul(dLdZ, np.transpose(dZdA)) # TODO
        dLdW = np.matmul(np.transpose(dLdZ), dZdW)  # TODO
        dLdb = np.matmul(np.transpose(dLdZ), dZdb)  # TODO
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
