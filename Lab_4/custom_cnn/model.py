import torch
from torch.nn import Module, Linear, Conv2d, ReLU, BatchNorm2d, BatchNorm1d
from torch import nn


class DogsCatsModel_CNN(Module):
    def __init__(self):
        super(DogsCatsModel_CNN, self).__init__()

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
        self.linear = Linear(64*8*8, 1)

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


class DogsCatsModel_FCN(Module):
    def __init__(self):
        super(DogsCatsModel_FCN, self).__init__()

        # Fully connected
        self.lin1 = Linear(in_features=3*128*128, out_features=3*64*128)
        self.lin2 = Linear(in_features=3*64*128, out_features=3*32*128)
        self.lin3 = Linear(in_features=3*32*128, out_features=3*16*128)
        self.lin4 = Linear(in_features=3*16*128, out_features=3*8*128)

        # classifier
        self.clasifier = Linear(3*8*128, 1)

        # batch norm
        self.bn1 = BatchNorm1d(3*64*128)
        self.bn2 = BatchNorm1d(3*32*128)
        self.bn3 = BatchNorm1d(3*16*128)
        self.bn4 = BatchNorm1d(3*8*128)

        # Relu
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.relu4 = ReLU()

    def _init_weights(self, module):
        if isinstance(module, (Linear, BatchNorm1d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    module.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        # Flatten
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))

        # FCN forward
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.lin3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.lin4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.clasifier(x)

        return x, x.sigmoid()
