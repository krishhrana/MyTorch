import numpy as np

import resampling
from mytorch.resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        N, cin, hin, win = A.shape
        hout = hin - self.kernel + 1
        wout = win - self.kernel + 1
        Z = np.zeros(shape=(N, cin, hout, wout))
        self.max_ind_A = []
        self.ind_Z = []
        for n in range(N):
            for c in range(cin):
                for h in range(hout):
                    for w in range(wout):
                        wh = np.argwhere(A[n, c, h: h + self.kernel, w: w + self.kernel] == np.max(A[n, c, h: h + self.kernel, w: w + self.kernel]))
                        Z[n, c, h, w] = A[n, c, wh[0][-2] + h, wh[0][-1] + w]
                        self.max_ind_A.append((n, c, wh[0][-2] + h, wh[0][-1] + w))
                        self.ind_Z.append((n, c, h, w))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)
        for i, j in zip(self.ind_Z, self.max_ind_A):
            an, ac, aw, ah = j
            zn, zc, zw, zh = i
            dLdA[j] += dLdZ[i]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.A = A
        N, cin, hin, win = A.shape
        hout = hin - self.kernel + 1
        wout = win - self.kernel + 1
        Z = np.zeros(shape=(N, cin, hout, wout))
        self.max_ind_A = []
        self.ind_Z = []
        for n in range(N):
            for c in range(cin):
                for h in range(hout):
                    for w in range(wout):
                        Z[n, c, h, w] = np.mean(A[n, c, h: h + self.kernel, w: w + self.kernel])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros_like(self.A)

        N, cin, hin, win = dLdZ.shape
        for n in range(N):
            for c in range(cin):
                for h in range(hin):
                    for w in range(win):
                        dLdA[n, c, h: h + self.kernel, w: w + self.kernel] += dLdZ[n, c, h, w] / (self.kernel * self.kernel)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = resampling.Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dlda_1 = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dlda_1)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = resampling.Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_1)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dlda_1 = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dlda_1)
        return dLdA
