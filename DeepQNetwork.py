import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor

#from Simulator import FRAME_STACK

class DeepQNetwork(nn.Module):

    def __init__(self, scale=1):
        super(DeepQNetwork, self).__init__()

        # convolution layers
        kernels = []

        kernel_size_conv1 = 3
        out_channels_conv1 = floor(8*scale)
        self.conv1 = nn.Conv2d(3, out_channels_conv1, kernel_size_conv1)
        kernels.append(kernel_size_conv1)

        kernel_size_conv2 = 3
        out_channels_conv2 = floor(16*scale)
        self.conv2 = nn.Conv2d(out_channels_conv1, out_channels_conv2, kernel_size_conv2)
        kernels.append(kernel_size_conv2)

        kernel_size_conv3 = 3
        out_channels_conv3 = floor(32*scale)
        self.conv3 = nn.Conv2d(out_channels_conv2, out_channels_conv3, kernel_size_conv2)
        kernels.append(kernel_size_conv3)

        # pooling operation
        pool = (2, 2)
        self.pool = nn.MaxPool2d(pool)
        # the pooling function is the same for both the conv layers so we add its dimension twice to the list of pooling operations
        poolings = [pool] * 3

        self.flat_dim_fc = self.flat_dim((96,96), out_channels_conv3, poolings, kernels)

        # For BAtch Normalization
        #self.bn2d_1 = nn.BatchNorm2d(out_channels_conv1)
        #self.bn2d_2 = nn.BatchNorm2d(out_channels_conv2)
        #self.bn2d_3 = nn.BatchNorm2d(out_channels_conv3)

        # fully connected layers
        self.fc1 = nn.Linear(self.flat_dim_fc, floor(64*scale))
        #self.bn1d_1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(floor(64*scale), floor(32*scale))
        #self.bn1d_2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(floor(32*scale), 5) # the output size is equal to the number of actions/q-values per state
        #self.bn1d_3 = nn.BatchNorm1d(1)

    def forward(self, x):

        # hinge together convolutions and pooling operations
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.bn2d_1(x)
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.bn2d_2(x)
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.bn2d_3(x)

        x = x.view(-1, self.flat_dim_fc)

        x = F.relu(self.fc1(x))
        # x = self.bn1d_1(x)
        x = F.relu(self.fc2(x))
        # x = self.bn1d_2(x)

        x = self.fc3(x)
        #x = torch.sigmoid(self.fc3(x))
        # x = self.bn1d_3(x)

        return x

    # This method computes the output dimension for a convolutional layer given the input dimension
    def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        return h, w

    # This method computes the flat dimension in output to a serie of convolutions given the input dimension and the
    # convolutional layer parameters
    def flat_dim(self, h_w, channels, poolings, kernels, strides=[], paddings=[], dilations=[]):

        if len(strides) == 0:
            strides = [1] * len(kernels)

        if len(paddings) == 0:
            paddings = [0] * len(kernels)

        if len(dilations) == 0:
            dilations = [1] * len(kernels)

        for i in range(len(kernels)):
            h, w = self.conv_output_shape(h_w, kernels[i], strides[i], paddings[i], dilations[i])
            h_w = (floor(h / poolings[i][0]), floor(w / poolings[i][1]))

        return h_w[0] * h_w[1] * channels