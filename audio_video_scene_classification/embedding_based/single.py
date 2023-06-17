import torch
import torch.nn as nn


class SingleModal(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes, dropout=0.2, modal='audio'):
        super(SingleModal, self).__init__()
        self.num_classes = num_classes
        self.modal = modal
        self.output = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, feature):
        # b, t, d
        feature = feature.mean(dim=1)
        # b, d
        output = self.output(feature)
        return output


if __name__ == "__main__":
    from torchinfo import summary
    model = SingleModal(embed_dim=512, hidden_dim=512, num_classes=20)
    feature = torch.randn(1, 96, 512)
    summary(model, input_data=[feature])
