import torch.nn as nn
import torch.nn.functional as F
from mnist.common.constants import MNIST_DIMENSIONS, HIDDEN_LAYER_ONE_CHANNELS, HIDDEN_LAYER_TWO_CHANNELS, NUM_CLASSES


class FullyConnectedModel(nn.Module):
    """
    Fully connected model.
    """

    def __init__(self):
        """
        Returns a three layer FullyConnectedModel.
        """
        super(FullyConnectedModel, self).__init__()

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(MNIST_DIMENSIONS, HIDDEN_LAYER_ONE_CHANNELS)
        self.linear_2 = nn.Linear(HIDDEN_LAYER_ONE_CHANNELS, HIDDEN_LAYER_TWO_CHANNELS)
        self.linear_3 = nn.Linear(HIDDEN_LAYER_TWO_CHANNELS, NUM_CLASSES)

    def forward(self, x):
        """
        Computes a forward pass through the network.
        :param x: The input data.
        :return: The output of the network.
        """
        x = self.flatten(x)
        x = self.linear_1(x)
        # Pysyft doesn't work with the nn.Relu
        x = F.relu(x)

        x = self.linear_2(x)
        x = F.relu(x)

        out = self.linear_3(x)

        return out
