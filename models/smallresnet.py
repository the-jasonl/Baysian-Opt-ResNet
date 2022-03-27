from torch import nn


class ResBlock(nn.Module):
    """
    ResBlocks with skip connection

    Args:
        nn (Module): Torch Neural Network Base class
    """

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = None
        # the feature map size is halved, the number of filters is doubled
        if out_channels == 2*in_channels:
            self.downsample = self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # adjust identity dimensions map size is halved
        if self.downsample is not None:
            x_identity = self.downsample(x_identity)
            x_identity = self.bn1(x_identity)
        x += x_identity  # skip connection
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """
    Small ResNet with two ResBlocks and a fully connected layer

    Args:
        nn (Module): Torch Neural Network Base class
    """

    def __init__(self):
        super(ResNet, self).__init__()
        mnist_channels = 1
        mnist_classes = 10
        # mnist input dims 28x28x1
        self.conv1 = nn.Conv2d(
            mnist_channels, 8, kernel_size=7, stride=1, padding=3)
        # output dims = (n + 2p - f) / s + 1
        # output dims 28 + 6 - 7 + 1 = 28
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # output dims  28 - 3 / 2 + 1 = 13.5
        self.resblock1 = ResBlock(8, 8)
        # output dims same
        self.resblock2 = ResBlock(8, 16)
        # output dims halved
        self.fc = nn.Linear(6*6*16, mnist_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
