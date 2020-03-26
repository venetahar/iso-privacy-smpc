import torch.nn as nn
import torch.nn.functional as F

from common.constants import CONV_ONE_FILTERS, NUM_CLASSES, \
    KERNEL_SIZE, STRIDE, FC_LAYER_ONE_UNITS


class ConvPlainTextNet(nn.Module):
    def __init__(self):
        super(ConvPlainTextNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=CONV_ONE_FILTERS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
        )

        self.linear_one = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5*13*13, FC_LAYER_ONE_UNITS),
        )

        self.linear_two = nn.Sequential(
            nn.Linear(FC_LAYER_ONE_UNITS, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv_1(x)
        # PySyft doesn't work with the ReLU being part of the sequential module
        x = F.relu(x)
        x = self.linear_one(x)
        x = F.relu(x)
        out = self.linear_two(x)

        return out
