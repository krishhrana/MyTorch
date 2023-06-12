import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = self.Y.shape[0]
        self.C = self.Y.shape[1]
        se = np.multiply((self.A - self.Y), (self.A - self.Y))
        column_N = np.ones(shape=(self.N, 1))
        column_C = np.ones(shape=(self.C, 1))
        sse = column_N.T @ se @ column_C
        mse = sse / (2 * self.N * self.C)
        return mse

    def backward(self):

        dLdA = (self.A - self.Y) / (self.N * self.C)

        return dLdA


class SoftmaxCrossEntropy:
    def __call__(self,A, Y):
        # No need to modify
        return self.forward(A, Y)

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y


        N = self.Y.shape[0]
        C = self.Y.shape[1]

        Ones_C = np.ones(shape=(C, 1))
        Ones_N = np.ones(shape=(N, 1))
        exp_A = np.exp(self.A)
        sum_exp = exp_A @ Ones_C @ Ones_C.T
        self.softmax = exp_A / sum_exp
        crossentropy = np.multiply(-self.Y, np.log(self.softmax)) @ Ones_C
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = self.softmax - self.Y

        return dLdA
