import torch
import torch.nn as nn


class AdaptiveConcatBase(nn.Module):
    def __init__(self, audio_embed_dim, video_embed_dim, hidden_dim, num_classes, dropout=0.2):
        super(AdaptiveConcatBase, self).__init__()
        self.num_classes = num_classes
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.video_pool = nn.AdaptiveAvgPool1d(1)
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, audio_feature, video_feature):
        # b, t, d
        audio_feature = self.audio_pool(audio_feature.transpose(1, 2)).squeeze(-1)
        video_feature = self.video_pool(video_feature.transpose(1, 2)).squeeze(-1)
        # b, d
        audio_embed = self.audio_embed(audio_feature)
        video_embed = self.video_embed(video_feature)
        embed = torch.cat([audio_embed, video_embed], dim=1)
        output = self.output(embed)
        return output


if __name__ == "__main__":
    from torchinfo import summary
    model = AdaptiveConcatBase(audio_embed_dim=512, video_embed_dim=512, hidden_dim=512, num_classes=20)
    audio_feature = torch.randn(1, 96, 512)
    video_feature = torch.randn(1, 96, 512)
    summary(model, input_data=[audio_feature, video_feature])
