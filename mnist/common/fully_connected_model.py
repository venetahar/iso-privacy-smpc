import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedModel(nn.Module):
    """
    Fully connected model.
    """

    def __init__(self, input_shape, hidden_units, num_classes):
        """
        Returns a FullyConnectedModel.
        :param input_shape: The input shape: (image_width, image_height, channels)
        :param hidden_units: The hidden units.
        :param num_classes: The number of classes.
        """
        super(FullyConnectedModel, self).__init__()

        self.flatten = nn.Flatten()

        image_width, image_height, in_channels = input_shape
        in_features = image_width * image_height * in_channels

        self.linear_1 = nn.Linear(in_features, hidden_units[0])
        self.linear_2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.output_layer = nn.Linear(hidden_units[1], num_classes)

    def forward(self, x):
        """
        Computes a forward pass through the network.
        :param x: The input data.
        :return: The output of the network.
        """
        x = self.flatten(x)
        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)
        x = F.relu(x)

        out = self.output_layer(x)

        return out

    def reset_parameters(self):
        """
        Initialize the model parameters.
        """
        for param in self.parameters():
            if param.requires_grad and param.dim() > 1:
                nn.init.xavier_uniform_(param)
