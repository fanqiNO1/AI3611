import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        The Squeeze-and-Excitation layer, which is an attention mechanism for channel-wise feature maps.

        @param channels: The number of input channels.
        @param reduction: The reduction factor for the number of channels and the dimension of hidden layer is channels // reduction.
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BottleneckX(nn.Module):
    def __init__(self, in_channels, channels, stride=1, groups=32, downsample=None, reduction=None):
        """
        The main block of SE-ResNeXt.

        @param in_channels: The number of input channels.
        @param channels: The number of hidden channels, and the output channels is 4 times of this.
        @param stride: The stride for the grouped convolutions.
        @param groups: The number of groups for grouped convolutions.
        @param downsample: The downsample layer.
        @param reduction: The reduction ratio for the SE layer. If None, no SE layer is used.
        """
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.1)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        if reduction is not None:
            self.se = SELayer(channels * 4, reduction)

        self.stride = stride
        self.downsample = downsample
        self.reduction = reduction

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.reduction is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    
    
class SEResNeXt(nn.Module):
    def __init__(self, channels=64, groups=32, reduction=16, layers=[2, 2, 2, 2]):
        """
        The SE-ResNeXt model for getting the feature maps.

        @param channels: The number of hidden channels.
        @param groups: The number of groups for the grouped convolution for the BottleneckX.
        @param reduction: The reduction factor for the number of channels for the SElayer.
        @param layers: The number of layers for each stage.
        """
        super(SEResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.channels = channels
        self.groups = groups
        self.reduction = reduction

        self.layer1 = self._make_layer(channels, layers[0], stride=1)
        self.layer2 = self._make_layer(channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(channels * 8, layers[3], stride=2)

    def _make_layer(self, channels, layer_number, stride=1):
        downsample = None
        if stride != 1 or self.channels != channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4, momentum=0.1),
            )

        layers = []
        if downsample is not None:
            layers.append(
                BottleneckX(
                    in_channels=self.channels,
                    channels=channels,
                    stride=stride,
                    groups=self.groups,
                    downsample=downsample,
                    reduction=self.reduction,
                )
            )
        else:
            layers.append(
                BottleneckX(
                    in_channels=self.channels,
                    channels=channels,
                    stride=stride,
                    groups=self.groups,
                    downsample=downsample,
                    reduction=None,
                )
            )
        self.channels = channels * 4
        for i in range(1, layer_number):
            layers.append(
                BottleneckX(
                    in_channels=self.channels,
                    channels=channels,
                    stride=1,
                    groups=self.groups,
                    downsample=None,
                    reduction=None,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    
def linear_softmax_pooling(x):
    return (x ** 2).sum(dim=1) / x.sum(dim=1)
    
    
class ResRNN(nn.Module):
    def __init__(self, num_freq, num_class):
        super(ResRNN, self).__init__()
        self.freq_bn = nn.BatchNorm1d(num_features=num_freq)
        self.resnet = SEResNeXt(channels=32, groups=16, reduction=8, layers=[3, 24, 36, 3])
        self.gru = nn.GRU(input_size=self.resnet.channels, hidden_size=self.resnet.channels // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.resnet.channels, out_features=self.resnet.channels // 2),
            nn.Linear(in_features=self.resnet.channels // 2, out_features=self.resnet.channels // 4),
            nn.Linear(in_features=self.resnet.channels // 4, out_features=self.resnet.channels // 8),
            nn.Linear(in_features=self.resnet.channels // 8, out_features=num_class)
        )
        self.upsample = nn.Upsample(size=501, mode="nearest")
        self.sigmoid = nn.Sigmoid()
        
    def detect(self, x):
        x = self.freq_bn(x.transpose(1, 2)).transpose(1, 2).unsqueeze(1)
        x = self.resnet(x)
        x = x.mean(-1).squeeze(-1).transpose(1, 2)
        out, _ = self.gru(x)
        out = self.linear(out)
        out = self.sigmoid(out)
        out = self.upsample(out.transpose(1, 2)).transpose(1, 2)
        return out
    
    def forward(self, x):
        frame_wise_prob = self.detect(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        return {
            "clip_prob": clip_prob,
            "time_prob": frame_wise_prob,
        }

    
if __name__ == "__main__":
    from torchinfo import summary
    model = ResRNN(num_freq=64, num_class=10)
    summary(model, input_size=(1, 501, 64))
    