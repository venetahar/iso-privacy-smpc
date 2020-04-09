import torch.nn as nn
import torch.nn.functional as F

from mnist.common.constants import FC_LAYER_ONE_UNITS, CONV_ONE_FILTERS, KERNEL_SIZE, STRIDE, NUM_CLASSES


class ConvModel(nn.Module):

    def __init__(self):
        super(ConvModel, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=CONV_ONE_FILTERS,
                                kernel_size=KERNEL_SIZE, stride=STRIDE, padding=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(CONV_ONE_FILTERS*6*6, FC_LAYER_ONE_UNITS)
        self.linear_2 = nn.Linear(FC_LAYER_ONE_UNITS, NUM_CLASSES)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.avg_pool(x)

        x = self.flatten(x)
        x = self.linear_1(x)
        x = F.relu(x)

        out = self.linear_2(x)

        return out
