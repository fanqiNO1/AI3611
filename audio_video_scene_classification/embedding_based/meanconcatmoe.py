import torch
import torch.nn as nn


class MeanConcatMoE(nn.Module):
    def __init__(self, audio_embed_dim, video_embed_dim, hidden_dim, num_classes, dropout=0.2):
        super(MeanConcatMoE, self).__init__()
        self.num_classes = num_classes
        self.audio = nn.Sequential(
            nn.Linear(audio_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        self.video = nn.Sequential(
            nn.Linear(audio_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        self.audio_video = nn.Sequential(
            nn.Linear(audio_embed_dim + video_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        self.weight = nn.Parameter(torch.randn(3))
        
    def forward(self, audio_feature, video_feature):
        # b, t, d
        audio_feature = audio_feature.mean(dim=1)
        video_feature = video_feature.mean(dim=1)
        # b, d
        audio_video_feature = torch.cat([audio_feature, video_feature], dim=1)
        # output
        audio_output = self.audio(audio_feature)
        video_output = self.video(video_feature)
        audio_video_output = self.audio_video(audio_video_feature)
        # fusion
        weight = torch.softmax(self.weight, dim=0)
        # output = weight[0] * audio_output + weight[1] * video_output + weight[2] * audio_video_output
        output = weight[0] * audio_output + weight[1] * video_output + weight[2] * audio_video_output
        return output
    

if __name__ == "__main__":
    from torchinfo import summary
    model = MeanConcatMoE(audio_embed_dim=512, video_embed_dim=512, hidden_dim=512, num_classes=10)
    audio_feature = torch.randn(1, 96, 512)
    video_feature = torch.randn(1, 96, 512)
    summary(model, input_data=[audio_feature, video_feature])
