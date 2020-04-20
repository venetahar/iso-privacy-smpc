from torch import nn
import torch.nn.functional as F


class ConvPoolModel(nn.Module):
    """
    A custom CNN network.
    """

    def __init__(self, input_shape, num_classes, conv_kernel_sizes, channels, avg_pool_sizes, fc_units):
        """
        Creates a CNN.
        :param input_shape: the input shape (image_width, image_height, channels).
        :param num_classes: the number of classes.
        :param conv_kernel_sizes: an array containing the kernel sizes for each convolution.
        :param channels: an array containing the channels for each convolution.
        :param avg_pool_sizes: an array containing the avg pool kernel sizes for the avg pool layers.
        :param fc_units: an array containing the number of units in the fully connected layers.
        """
        super(ConvPoolModel, self).__init__()
        _, _, in_channels = input_shape
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=channels[0],
                                kernel_size=conv_kernel_sizes[0], stride=1)

        self.avg_pool_1 = nn.AvgPool2d(kernel_size=avg_pool_sizes[0])

        self.conv_2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                                kernel_size=conv_kernel_sizes[1], stride=1)

        self.avg_pool_2 = nn.AvgPool2d(kernel_size=avg_pool_sizes[1])

        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(channels[1]*5*5, fc_units[0])
        self.linear_2 = nn.Linear(fc_units[0], num_classes)

    def forward(self, x):
        """
        Computes the forward pass of the network without applying the final activation as this is done when computing
        the loss for efficiency.
        :param x: The input data.
        :return: The output after the forward pass.
        """
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.avg_pool_1(x)

        x = self.conv_2(x)
        x = F.relu(x)
        x = self.avg_pool_2(x)

        x = self.flatten(x)
        x = self.linear_1(x)
        x = F.relu(x)

        out = self.linear_2(x)

        return out
