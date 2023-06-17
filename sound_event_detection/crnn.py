from functools import partial

import torch
import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(dim=1) / x.sum(dim=1)


class CRNN(nn.Module):
    def __init__(self, num_freq, num_class, pooling_type="max"):
        super(CRNN, self).__init__()
        if pooling_type == "max":
            Pool = nn.MaxPool2d
        elif pooling_type == "avg":
            Pool = nn.AvgPool2d
        elif pooling_type == "lp":
            Pool = partial(nn.LPPool2d, norm_type=2)
        else:
            raise ValueError("pooling_type must be one of 'max', 'avg', 'adaptive_max', 'adaptive_avg'")
        self.freq_bn = nn.BatchNorm1d(num_features=num_freq)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            Pool(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            Pool(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            Pool(kernel_size=1, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            Pool(kernel_size=1, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            Pool(kernel_size=1, stride=2)
        )
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=256, out_features=num_class)
        self.upsample = nn.Upsample(size=501, mode="nearest")
        self.sigmoid = nn.Sigmoid()
        
    def detect(self, x):
        # x: [batch_size, time_step, num_freq]
        x = self.freq_bn(x.transpose(1, 2)).transpose(1, 2).unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x).mean(-1).squeeze(-1).transpose(1, 2)
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
    model = CRNN(num_freq=64, num_class=10)
    summary(model, input_size=(16, 501, 64))
    