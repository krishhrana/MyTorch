# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​
import Conv1d
import flatten
import linear
from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Your code goes here -->
        # self.convolutional_layers (list Conv1d) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------

        self.convolutional_layers = [
            Conv1d(in_channels=num_input_channels, out_channels=num_channels[0], kernel_size=kernel_sizes[0], stride=strides[0], weight_init_fn=conv_weight_init_fn, bias_init_fn=bias_init_fn),
            Conv1d(num_channels[0], out_channels=num_channels[1], kernel_size=kernel_sizes[1], stride=strides[1], weight_init_fn=conv_weight_init_fn, bias_init_fn=bias_init_fn),
            Conv1d(num_channels[1], out_channels=num_channels[2], kernel_size=kernel_sizes[2], stride=strides[2], weight_init_fn= conv_weight_init_fn, bias_init_fn=bias_init_fn)
        ]
        self.flatten = Flatten()
        self.linear_layer = Linear(420, num_linear_neurons)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        print("Activations: ", self.activations)

        # Your code goes here -->
        # Iterate through each layer
        # <---------------------

        '''self.layer = []
        for l in range(len(self.convolutional_layers)):
            self.layer.append(self.convolutional_layers[l])
            self.layer.append(self.activations[l])

        self.layer.append(self.flatten)
        self.layer.append(self.linear_layer)


        for i in self.layer:
            A = i.forward(A)'''



        for l in range(len(self.convolutional_layers)):
            A = self.convolutional_layers[l].forward(A)
            #print(A)
            A = self.activations[l].forward(A)
            #print("Activation: ", A)

        A = self.flatten.forward(A)
        A = self.linear_layer.forward(A)

        # Save output (necessary for error and loss)
        self.Z = A

        return self.Z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        # Your code goes here -->
        # Iterate through each layer in reverse order
        # <---------------------
        # Start from here ------->
        grad = self.linear_layer.backward(grad)
        grad = self.flatten.backward(grad)
        grad *= self.activations[-1].backward()
        grad = self.convolutional_layers[-1].backward(grad)
        grad *= self.activations[-2].backward()
        grad = self.convolutional_layers[-2].backward(grad)
        grad *= self.activations[-3].backward()
        grad = self.convolutional_layers[-3].backward(grad)

        return grad

    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW.fill(0.0)
        self.linear_layer.dLdb.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
