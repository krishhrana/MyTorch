from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):

        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, stride=1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3
        ]

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        w1 = w1.T.reshape(8, 8, 24)
        w2 = w2.T.reshape(16, 1, 8)
        w3 = w3.T.reshape(4, 1, 16)
        self.conv1.conv1d_stride1.W = w1.transpose(0, 2, 1)
        self.conv2.conv1d_stride1.W = w2.transpose(0, 2, 1)
        self.conv3.conv1d_stride1.W = w3.transpose(0, 2, 1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z.reshape(1, -1)

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1d(in_channels=24, out_channels=2, kernel_size=2, stride=2)
        self.conv2 = Conv1d(in_channels=2, out_channels=8, kernel_size=2, stride=2)
        self.conv3 = Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3
        ]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        w1 = w1[:48, :2].T.reshape(2, 2, 24)
        w2 = w2[:4, :8].T.reshape(8, 2, 2)
        w3 = w3[:16, :4].T.reshape(4, 2, 8)
        self.conv1.conv1d_stride1.W = w1.transpose(0, 2, 1)
        self.conv2.conv1d_stride1.W = w2.transpose(0, 2, 1)
        self.conv3.conv1d_stride1.W = w3.transpose(0, 2, 1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        print(Z.shape)
        return Z.reshape(1, -1)

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
