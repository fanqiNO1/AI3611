from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(dim=1) / x.sum(dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout),
                nn.LayerNorm(dim),
            ]))
            
    def forward(self, x):
        for attn, attn_norm, ff, ff_norm in self.layers:
            x = x + attn_norm(attn(x))
            x = x + ff_norm(ff(x))
        return x


class ViT1d(nn.Module):
    def __init__(
        self, 
        channels, 
        seq_len, 
        patch_size, 
        num_classes, 
        dim=512, 
        depth=6, 
        heads=8, 
        mlp_dim=1024, 
        dim_head=64, 
        dropout=0.
    ):
        super(ViT1d, self).__init__()
        assert seq_len % patch_size == 0, "Sequence length must be divisible by the patch size."
        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.patch_embed = nn.Sequential(
            Rearrange("b c (n p) -> b n (p c)", p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        return x
    
    
class ViT1dGRU(nn.Module):
    def __init__(self, num_freq, num_class, pooling_type="max"):
        super(ViT1dGRU, self).__init__()
        self.num_freq = num_freq
        self.num_class = num_class
        self.upsample1 = nn.Upsample(size=512, mode="nearest")
        self.vit1d = ViT1d(
            channels=512, 
            seq_len=num_freq, 
            patch_size=4, 
            num_classes=num_class, 
            dim=128, 
            depth=12, 
            heads=8, 
            mlp_dim=1024, 
            dim_head=64, 
            dropout=0.
        )
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256, num_class)
        self.upsample2 = nn.Upsample(size=501, mode="nearest")
        self.sigmoid = nn.Sigmoid()
        
    def detect(self, x):
        x = self.upsample1(x.transpose(1, 2)).transpose(1, 2)
        x = self.vit1d(x)
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
    model = ViT1dGRU(num_freq=64, num_class=10)
    summary(model, input_size=(16, 501, 64))