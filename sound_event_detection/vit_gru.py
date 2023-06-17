import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


def linear_softmax_pooling(x):
    return (x ** 2).sum(dim=1) / x.sum(dim=1)


class ViT(nn.Module):
    def __init__(self, in_channel, patch_size=16, dim=256, depth=6, heads=8, mlp_dim=1024):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbed(
            img_size=(512, 64), patch_size=patch_size, in_chans=in_channel, embed_dim=dim)
        self.patch_size = patch_size
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        mlp_ratio = mlp_dim / dim
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: (B, C, H, W)
        b = x.shape[0]
        h_p = x.shape[2] // self.patch_size
        w_p = x.shape[3] // self.patch_size
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h_p, w_p).contiguous()
        return x
    
class ViTGRU(nn.Module):
    def __init__(self, num_freq, num_class, pooling_type="max"):
        super(ViTGRU, self).__init__()
        if pooling_type == "max":
            Pool = nn.MaxPool2d
        elif pooling_type == "avg":
            Pool = nn.AvgPool2d
        elif pooling_type == "adaptive_max":
            Pool = nn.AdaptiveMaxPool2d
        elif pooling_type == "adaptive_avg":
            Pool = nn.AdaptiveAvgPool2d
        else:
            raise ValueError("pooling_type must be one of 'max', 'avg', 'adaptive_max', 'adaptive_avg'")
        self.upsample1 = nn.Upsample(size=512, mode="nearest")
        self.freq_bn = nn.BatchNorm1d(num_features=num_freq)
        self.vit = ViT(
            in_channel=1,
            patch_size=16,
            dim=256,
            depth=5,
            heads=8,
            mlp_dim=1024
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            Pool(kernel_size=2, stride=2),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=256, out_features=num_class)
        self.upsample2 = nn.Upsample(size=501, mode="nearest")
        self.sigmoid = nn.Sigmoid()
        
    def detect(self, x):
        # x: [batch_size, time_step, num_freq]
        # upsample to 512
        x = self.upsample1(x.transpose(1, 2)).transpose(1, 2)
        x = self.freq_bn(x.transpose(1, 2)).transpose(1, 2).unsqueeze(1)
        x = self.vit(x)
        x = self.conv(x).mean(-1).squeeze(-1).transpose(1, 2)
        x, _ = self.gru(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        x = self.upsample2(x.transpose(1, 2)).transpose(1, 2)
        return x
    
    def forward(self, x):
        frame_wise_prob = self.detect(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        return {
            "clip_prob": clip_prob,
            "time_prob": frame_wise_prob,
        }
        
        
if __name__ == "__main__":
    from torchinfo import summary
    model = ViTGRU(num_freq=64, num_class=10)
    summary(model, input_size=(16, 501, 64))
    