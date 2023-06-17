from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


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
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        x = self.patch_embed(x)
        b, n, _ = x.shape
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.classifier(x[:, 0])
        return x
    
    
class ViTConcatLate(nn.Module):
    def __init__(self, audio_embed_dim, video_embed_dim, hidden_dim, num_classes, dropout=0.2):
        super(ViTConcatLate, self).__init__()
        self.audio_vit = ViT1d(channels=96, seq_len=audio_embed_dim, patch_size=16, num_classes=10, dropout=dropout)
        self.video_vit = ViT1d(channels=96, seq_len=video_embed_dim, patch_size=16, num_classes=10, dropout=dropout)
        
    def forward(self, audio, video):
        audio = self.audio_vit(audio)
        video = self.video_vit(video)
        x = (audio + video) / 2
        return x
    
    
if __name__ == "__main__":
    from torchinfo import summary
    model = ViTConcatLate(audio_embed_dim=512, video_embed_dim=512, hidden_dim=512, num_classes=10)
    audio_feature = torch.randn(1, 96, 512)
    video_feature = torch.randn(1, 96, 512)
    summary(model, input_data=[audio_feature, video_feature])