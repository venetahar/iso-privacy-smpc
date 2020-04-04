import torch.nn as nn
import torch.nn.functional as F

from mnist.common.constants import FC_LAYER_ONE_UNITS, CONV_ONE_FILTERS, KERNEL_SIZE, STRIDE, NUM_CLASSES


class ConvPlainTextNet(nn.Module):
    def __init__(self):
        super(ConvPlainTextNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=CONV_ONE_FILTERS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1),
        )

        self.max_pool = nn.AvgPool2d(kernel_size=2)

        self.linear_one = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV_ONE_FILTERS*6*6, FC_LAYER_ONE_UNITS),
        )

        self.linear_two = nn.Sequential(
            nn.Linear(FC_LAYER_ONE_UNITS, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv_1(x)
        # PySyft doesn't work with the ReLU being part of the sequential module
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2)
        x = self.linear_one(x)
        x = F.relu(x)
        out = self.linear_two(x)

        return out
