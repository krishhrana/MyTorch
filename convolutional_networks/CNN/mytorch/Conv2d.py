import numpy as np

import resampling
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        output_width = A.shape[-1] - self.kernel_size + 1
        output_height = A.shape[-2] - self.kernel_size + 1
        Z = np.zeros(shape = (A.shape[0], self.out_channels, output_height, output_width))
        width = 0
        height = 0

        for n in range(A.shape[0]):
            a_batch = A[n]
            for i in range(self.W.shape[0]):
                w_batch = self.W[i]
                height = 0
                while height <= a_batch.shape[-2] - self.kernel_size:
                    width = 0
                    while width <= a_batch.shape[-1] - self.kernel_size:
                        patch = a_batch[:, height: height + self.kernel_size, width:width + self.kernel_size]
                        channel_add = 0
                        for k in range(w_batch.shape[0]):
                            w = w_batch[k]
                            a = patch[k]
                            mul = np.multiply(w, a)
                            add = np.sum(mul)
                            channel_add = channel_add + add
                        Z[n, i, height, width] = channel_add
                        width = width + 1
                    height = height + 1
                Z[n, i, :, :] += self.b[i]

        self.A = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        cout, cin, k, k_w = self.W.shape
        N, cout, hout, wout = dLdZ.shape

        dLdA = np.zeros(self.A.shape)
        dLdW = np.zeros(self.W.shape)

        for n in range(N):
            for co in range(cout):
                for ci in range(cin):
                    for h in range(hout):
                        for w in range(wout):
                            dLdA[n, ci, h:h + k, w:w + k_w] += dLdZ[n, co, h, w] * self.W[co, ci]
                            dLdW[co, ci] += self.A[n, ci, h:h + k, w:w + k_w] * dLdZ[n, co, h, w]

        self.dLdW = dLdW
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        self.dLdA = dLdA

        return self.dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = resampling.Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        Z_1 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward((Z_1))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdA_1 = self.downsample2d.backward((dLdZ))

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA_1)
        return dLdA
