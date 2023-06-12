import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.sampled_i = []

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        z_shape = self.upsampling_factor * (A.shape[-1] - 1) + 1
        Z = np.zeros(shape = (A.shape[0], A.shape[1], z_shape))
        for i in range(0, z_shape, self.upsampling_factor):
            Z[:, : , i] = A[:, :, i // self.upsampling_factor]
            self.sampled_i.append(i)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        #dLdA_shape = ((dLdZ.shape[-1] - 1) / self.upsampling_factor) + 1
        #dLdA = np.zeros(shape=(dLdZ.shape[0], dLdZ.shape[1], dLdA_shape))
        dLdA = dLdZ[:, :, self.sampled_i]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.sampling_i = []

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.a_shape = A.shape
        Z_shape = A.shape[-1] // (self.downsampling_factor) + 1 if A.shape[-1] % self.downsampling_factor != 0 else A.shape[-1] // (self.downsampling_factor)
        Z = np.zeros(shape=(A.shape[0], A.shape[1], Z_shape))  # TODO

        for i in range(0, A.shape[-1], self.downsampling_factor):
            self.sampling_i.append(i)
            Z[:, :, i // self.downsampling_factor] = A[:, :, i]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros(shape = self.a_shape)
        dLdA[:, :, self.sampling_i] = dLdZ[:, :, :]  # TODO

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.sampling_i = []

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        z_shape_w = self.upsampling_factor * (A.shape[-1] - 1) + 1
        z_shape_h = self.upsampling_factor * (A.shape[-2] - 1) + 1

        Z = np.zeros(shape=(A.shape[0], A.shape[1], z_shape_h, z_shape_w))
        for i in range(0, z_shape_w, self.upsampling_factor):
            for j in range(0, z_shape_h, self.upsampling_factor):
                self.sampling_i.append([i, j])
                Z[:, :, j, i] = A[:, :, j // self.upsampling_factor, i // self.upsampling_factor]


        '''Z = np.zeros(shape=(A.shape[0], A.shape[1],z_shape_h, Z_inter.shape[-1]))
        for i in range(0, z_shape_h, self.upsampling_factor):
            self.sampling_i[i].append(i)
            Z[:, :, i, :] = Z_inter[:, :, i // self.upsampling_factor, :]'''


        self.sampling_i = np.array(self.sampling_i)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA_shape_w = ((dLdZ.shape[-1] - 1) // self.upsampling_factor) + 1
        dLdA_shape_h = ((dLdZ.shape[-2] - 1) // self.upsampling_factor) + 1
        dLdA = np.zeros(shape=(dLdZ.shape[0], dLdZ.shape[1], dLdA_shape_h, dLdA_shape_w))
        width_mask = sorted(list(set(self.sampling_i[:, 0])))
        height_mask  = sorted(list(set(self.sampling_i[:, 1])))

        dLdA = dLdZ[:, :, :, height_mask]  # TODO
        dLdA = dLdA[:, :, width_mask, :]
        #print("Sampling: ", self.sampling_i)

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.sampling_i = []

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.a_shape = A.shape

        Z_shape_w = A.shape[-1] // (self.downsampling_factor) + 1 if A.shape[-1] % self.downsampling_factor != 0 else A.shape[-1] // (self.downsampling_factor)
        Z_shape_h = A.shape[-2] // (self.downsampling_factor) + 1 if A.shape[-2] % self.downsampling_factor != 0 else A.shape[-2] // (self.downsampling_factor)
        Z = np.zeros(shape=(A.shape[0], A.shape[1], Z_shape_h, Z_shape_w))  # TODO

        for i in range(0, A.shape[-1], self.downsampling_factor):
            for j in range(0, A.shape[-2], self.downsampling_factor):
                self.sampling_i.append([i, j])
                Z[:, :, j // self.downsampling_factor, i // self.downsampling_factor] = A[:, :, j, i]
            #Z[:, :, j // self.downsampling_factor, :] = A[:, :, j, :]

        self.sampling_i = np.array(self.sampling_i)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """


        dLdA_shape_w = (self.downsampling_factor - 1) * dLdZ.shape[-1]
        dLdA_shape_h = (self.downsampling_factor - 1) * dLdZ.shape[-2]
        dLdA_inter = np.zeros(shape = (dLdZ.shape[0], dLdZ.shape[1], self.a_shape[-2], dLdZ.shape[-1]))
        dLdA = np.zeros(shape = self.a_shape)
        width_mask = sorted(list(set(self.sampling_i[:, 0])))
        height_mask = sorted(list(set(self.sampling_i[:, 1])))

        dLdA_inter[:, :, height_mask, :] = dLdZ[:, :, :, :]  # TODO
        dLdA[:, :, :, width_mask] = dLdA_inter[:, :, :, :]

        return dLdA
