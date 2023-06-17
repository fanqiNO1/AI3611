import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.2):
        super(CNN, self).__init__()
        self.embed_bn = nn.BatchNorm1d(num_features=embed_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128*3, out_features=num_classes)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        x = self.embed_bn(x.transpose(1, 2)).transpose(1, 2).unsqueeze(1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.dropout(x)
        x = x.mean(-1).squeeze(-1).transpose(1, 2)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
    
class CNNConcatLate(nn.Module):
    def __init__(self, audio_embed_dim, video_embed_dim, hidden_dim, num_classes, dropout=0.2):
        super(CNNConcatLate, self).__init__()
        self.audio_embed = CNN(embed_dim=audio_embed_dim, num_classes=num_classes, dropout=dropout)
        self.video_embed = CNN(embed_dim=video_embed_dim, num_classes=num_classes, dropout=dropout)
        
    def forward(self, audio, video):
        audio = self.audio_embed(audio)
        video = self.video_embed(video)
        x = (audio + video) / 2
        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    model = CNNConcatLate(audio_embed_dim=512, video_embed_dim=512, hidden_dim=512, num_classes=10)
    audio_feature = torch.randn(1, 96, 512)
    video_feature = torch.randn(1, 96, 512)
    summary(model, input_data=[audio_feature, video_feature])