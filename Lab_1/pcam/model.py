import torch
from torch.nn import Module, Linear, Conv2d, ReLU, BatchNorm2d
from torch import nn


class PCamModel(Module):
    def __init__(self):
        super(PCamModel, self).__init__()

        self.conv1 = Conv2d(in_channels=3, out_channels=8,
                            stride=2, padding=1, kernel_size=(3, 3))
        self.conv2 = Conv2d(in_channels=8, out_channels=16,
                            kernel_size=(3, 3), padding=1, stride=2)
        self.conv3 = Conv2d(in_channels=16, out_channels=32,
                            kernel_size=(3, 3), padding=1, stride=2)
        self.conv4 = Conv2d(in_channels=32, out_channels=64,
                            kernel_size=(3, 3), padding=1, stride=2)

        # activations
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.relu4 = ReLU()

        # Batch norm
        self.bn1 = BatchNorm2d(8)
        self.bn2 = BatchNorm2d(16)
        self.bn3 = BatchNorm2d(32)
        self.bn4 = BatchNorm2d(64)

        # Linear
        self.linear = Linear(64*6*6, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (Conv2d, BatchNorm2d, Linear)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    module.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x = self.linear(x)

        return x, x.sigmoid()
