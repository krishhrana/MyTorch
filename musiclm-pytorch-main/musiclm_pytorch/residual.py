import torch
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                            stride = stride, padding = 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                     stride = 1, padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels


    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        output = output + residual
        output = self.relu(output)
        return output



class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes = 7000):
        super(ResNet, self).__init__()
        self.num_channels = 64
        self.num_classes = num_classes

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride = 2, padding=3),
            torch.nn.BatchNorm2d(self.num_channels),
            torch.nn.ReLU())

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self.layer_blocks(block, 64, layers[0], stride=1)
        self.layer1 = self.layer_blocks(block, 128, layers[1], stride=2)
        self.layer2 = self.layer_blocks(block, 256, layers[2], stride=2)
        self.layer3 = self.layer_blocks(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(512, self.num_classes)


    def layer_blocks(self, block, num_filters, num_blocks, stride = 1):
        downsample = None
        if (stride != 1) or (num_filters != self.num_channels):
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.num_channels, out_channels=num_filters, kernel_size=1, stride = stride),
                torch.nn.BatchNorm2d(num_filters)
            )
        layers = [block(self.num_channels, num_filters, stride, downsample)]
        self.num_channels = num_filters

        for i in range(1, num_blocks):
            layers.append(block(self.num_channels, num_filters))

        return torch.nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




