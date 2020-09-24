import torch.nn as nn
import torch.nn.functional as F


class ConvModel(nn.Module):
    """
    Returns a convolution model.
    """

    def __init__(self, image_shape, out_channels, kernel_size, stride, padding, avg_pool_size, linear_units, num_classes):
        """
        Creates a ConvModel.
        :param image_shape: The image shape: (image_width, image_height, in_channels)
        :param out_channels: The output channels of the convolution.
        :param kernel_size: The kernel size.
        :param stride: The stride.
        :param padding: The padding.
        :param avg_pool_size: The average pool size.
        :param linear_units: The linear units.
        :param num_classes: The number of classes.
        """
        super(ConvModel, self).__init__()

        image_width, image_height, in_channels = image_shape

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)

        self.avg_pool = nn.AvgPool2d(kernel_size=avg_pool_size)

        self.flatten = nn.Flatten()

        in_features = self._calculate_linear_in_features(image_width, image_height, out_channels,
                                                         kernel_size, stride, padding, avg_pool_size)
        self.linear_1 = nn.Linear(in_features, linear_units)
        self.output_layer = nn.Linear(linear_units, num_classes)

    def forward(self, x):
        """
        Computes a forward pass through the network.
        :param x: The input data.
        :return: The output of the network.
        """
        x = self.conv_1(x)
        x = F.relu(x)

        x = self.avg_pool(x)

        x = self.flatten(x)
        x = self.linear_1(x)
        x = F.relu(x)

        out = self.output_layer(x)

        return out

    @staticmethod
    def _calculate_linear_in_features(image_width, image_height, output_channels, kernel_size,
                                      stride, padding, avg_pool_size):
        width_out = int((image_width - kernel_size + (2 * padding)) / stride + 1)
        height_out = int((image_height - kernel_size + (2 * padding)) / stride + 1)

        width_out = int(width_out / avg_pool_size)
        height_out = int(height_out / avg_pool_size)

        return width_out * height_out * output_channels

    def reset_parameters(self):
        """
        Initialize the model parameters.
        """
        for param in self.parameters():
            if param.requires_grad and param.dim() > 1:
                nn.init.xavier_uniform_(param)
