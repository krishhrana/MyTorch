import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = self.Z.shape[0]  # TODO
        self.M = np.mean(self.Z, axis=0).reshape(1, -1)  # TODO
        self.V = np.var(self.Z, axis=0).reshape(1, -1)  # TODO
        self.col_broadcast = np.ones(shape=(self.N, 1))

        if eval == False:
            # training mode
            self.broad_M = np.matmul(self.col_broadcast, self.M)
            self.broad_V = np.matmul(self.col_broadcast, self.V + self.eps)
            self.NZ = (self.Z - self.broad_M) / np.sqrt(self.broad_V)
            self.BZ = self.NZ * (np.matmul(self.col_broadcast, self.BW)) + np.matmul(self.col_broadcast, self.Bb)  # TODO

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M   # TODO
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V # TODO
        else:
            # inference mode
            self.NZ = (self.Z - np.matmul(self.col_broadcast, self.running_M)) / np.sqrt(np.matmul(self.col_broadcast, self.running_V + self.eps))  # TODO
            self.BZ = self.NZ * (np.matmul(self.col_broadcast, self.BW)) + np.matmul(self.col_broadcast, self.Bb)  # TODO

        return self.BZ

    def backward(self, dLdBZ):
        self.dLdBW = None  # TODO
        self.dLdBb = np.multiply(dLdBZ, self.NZ)  # TODO

        dLdNZ = np.multiply(dLdBZ, np.matmul(self.col_broadcast, self.BW))   # TODO
        dLdV = -0.5*(np.sum(dLdNZ * (self.Z - self.broad_M) * np.power(self.broad_V, -1.5), axis=0)).reshape((1, -1))
        term1 = np.matmul(self.col_broadcast, -1 * np.power((self.V + self.eps), -0.5))
        term2 = np.multiply(0.5 * (self.Z - self.broad_M), np.power(self.broad_V, -1.5))
        term3 = (-2 / self.N) * np.sum(self.Z - self.broad_M, axis=0)
        term3 = term3.reshape(1, -1)
        term3 = np.matmul(self.col_broadcast, term3)
        dNZdU = term1 - np.multiply(term3, term2)
        dLdM = np.sum(dLdNZ * dNZdU, axis=0).reshape(1,-1)  # TODO

        t1 = dLdNZ * np.power((self.broad_V), -0.5)
        t2 = np.matmul(self.col_broadcast, dLdV) * ((2/self.N) * (self.Z - self.broad_M))
        t3 = np.matmul(self.col_broadcast, dLdM/self.N)
        print(t1.shape)
        print(t2.shape)
        print(t3.shape)
        dLdZ = t1 + t2 + t3  # TODO

        return dLdZ
