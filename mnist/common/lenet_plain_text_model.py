import torch.nn as nn
import torch.nn.functional as F
from mnist.common.constants import MNIST_DIMENSIONS, HIDDEN_LAYER_ONE_CHANNELS, HIDDEN_LAYER_TWO_CHANNELS, NUM_CLASSES


class PlainTextNet(nn.Module):
    def __init__(self):
        super(PlainTextNet, self).__init__()

        self.linear_one = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MNIST_DIMENSIONS, HIDDEN_LAYER_ONE_CHANNELS),
        )

        self.relu = nn.ReLU()

        self.linear_two = nn.Sequential(
            nn.Linear(HIDDEN_LAYER_ONE_CHANNELS, HIDDEN_LAYER_TWO_CHANNELS),
        )

        self.linear_three = nn.Sequential(
            nn.Linear(HIDDEN_LAYER_TWO_CHANNELS, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.linear_one(x)
        # Pysyft doesn't work with the Relu being part of the sequential module
        x = self.relu(x)
        x = self.linear_two(x)
        x = F.relu(x)
        out = self.linear_three(x)

        return out
