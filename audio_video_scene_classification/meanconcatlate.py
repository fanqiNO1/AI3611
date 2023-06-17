import torch
import torch.nn as nn


class MeanConcatLate(nn.Module):
    def __init__(self, audio_embed_dim, video_embed_dim, hidden_dim, num_classes, dropout=0.2):
        super(MeanConcatLate, self).__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, audio_feature, video_feature):
        # b, t, d
        audio_feature = audio_feature.mean(dim=1)
        video_feature = video_feature.mean(dim=1)
        # b, d
        audio_embed = self.audio_embed(audio_feature)
        video_embed = self.video_embed(video_feature)
        output = (audio_embed + video_embed) / 2
        return output


if __name__ == "__main__":
    from torchinfo import summary
    model = MeanConcatLate(audio_embed_dim=512, video_embed_dim=512, hidden_dim=512, num_classes=10)
    audio_feature = torch.randn(1, 96, 512)
    video_feature = torch.randn(1, 96, 512)
    summary(model, input_data=[audio_feature, video_feature])
