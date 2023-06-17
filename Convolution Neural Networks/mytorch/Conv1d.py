# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

import resampling
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        self.A = A
        b = self.b.reshape(-1, 1)
        output_shape = A.shape[-1] - self.kernel_size + 1
        Z = np.zeros(shape = (A.shape[0], self.W.shape[0], output_shape))
        for j in range(A.shape[0]):
            i = 0
            convolution = []
            while i <= A.shape[-1] - self.kernel_size:
                window = A[j, :, i: i + self.kernel_size]
                # print("window: \n", window)
                i = i + 1
                # print("W: \n", w)
                mul = np.multiply(window, self.W)
                # print("Mul: \n", mul)
                add = np.sum(mul, axis=(1, 2))
                # print("Add: \n", add)
                convolution.append(add)
            Z[j, :, :] = np.column_stack(convolution) + b
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        cout, cin, k = self.W.shape
        N, cout, wout = dLdZ.shape

        dLdA = np.zeros(self.A.shape)
        dLdW = np.zeros(self.W.shape)


        for n in range(N):
            for co in range(cout):
                for ci in range(cin):
                    for w in range(wout):
                        dLdA[n, ci, w:w + k] += dLdZ[n, co, w] * self.W[co, ci]
                        dLdW[co, ci] += self.A[n, ci, w:w + k] * dLdZ[n, co, w]

        self.dLdW = dLdW  # TODO
        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # TODO  # TODO
        self.dLdA = dLdA


        # TODO

        return self.dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = resampling.Downsample1d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        Z_1 = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward((Z_1))# TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dlda_1 = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dlda_1)  # TODO

        return dLdA
