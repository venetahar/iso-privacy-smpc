import torch.nn as nn

from cryp_ten.constants import MNIST_DIMENSIONS, HIDDEN_LAYER_ONE_CHANNELS, HIDDEN_LAYER_TWO_CHANNELS, NUM_CLASSES


class PlainTextNet(nn.Module):
    def __init__(self):
        super(PlainTextNet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MNIST_DIMENSIONS, HIDDEN_LAYER_ONE_CHANNELS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_ONE_CHANNELS, HIDDEN_LAYER_TWO_CHANNELS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_TWO_CHANNELS, NUM_CLASSES),
        )

    def forward(self, x):
        out = self.model(x)
        return out
