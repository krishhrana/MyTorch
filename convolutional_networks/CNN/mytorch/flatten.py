import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A = A
        N, cin, win = A.shape
        Z = np.zeros(shape=(N, cin*win))
        for n in range(N):
            Z[n] = A[n].flatten()

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        dLdA = dLdZ.reshape(self.A.shape)

        return dLdA
