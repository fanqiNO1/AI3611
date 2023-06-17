import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


class Encoder(nn.Module):
    """
    Different from normal ViT, I remove the cls_token and add a fc layer to generate mu and log_var.
    """
    def __init__(self, in_channel=1, patch_size=7, dim=256, depth=16, heads=16, mlp_dim=1024, z_dim=1):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=28, patch_size=patch_size, in_chans=in_channel, embed_dim=dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        mlp_ratio = mlp_dim / dim
        # Main part
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
        )
        self.mu = nn.Linear(32, z_dim)
        self.var = nn.Linear(32, z_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # (b, num_patches, dim) -> (b, dim)
        x = x.mean(dim=1)
        x = self.fc(x)
        mu = self.mu(x)
        log_var = self.var(x)
        return mu, log_var


class Decoder(nn.Module):
    """
    Different from normal ViT, I remove the cls_token and add a fc layer to reconstruct the image.
    """
    def __init__(self, in_channel=1, patch_size=7, num_patches=4, dim=256, depth=16, heads=16, mlp_dim=1024, z_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, dim),
            nn.ReLU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        mlp_ratio = mlp_dim / dim
        # Main part
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=True) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch_size * patch_size)
        self.act = nn.Sigmoid()
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, z):
        x = self.fc(z)
        # Because in the forward process of Encoder, the mean of the patches is calculated
        # The dimensions should be restored to the original shape
        x = x.unsqueeze(1).repeat(1, self.num_patches, 1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.out(x)
        x = self.act(x)
        num_patches_per_side = int(self.num_patches ** 0.5)
        # (b, nh*nw, h*w*1) -> (b, nh, nw, h, w, 1)
        x = x.reshape(-1, num_patches_per_side, num_patches_per_side, self.patch_size, self.patch_size, self.in_channel)
        # (b, nh, nw, h, w, 1) -> (b, 1, nh, h, nw, w)
        x = x.permute(0, 5, 1, 3, 2, 4)
        # (b, 1, nh, h, nw, w) -> (b, 1, nh*h, nw*w) = (b, 1, H, W)
        x = x.reshape(-1, self.in_channel, num_patches_per_side * self.patch_size, num_patches_per_side * self.patch_size)
        return x


class VAEViT(nn.Module):
    def __init__(self, patch_size=7, dim=256, depth=16, heads=16, mlp_dim=1024, z_dim=1):
        super().__init__()
        self.encoder = Encoder(1, patch_size, dim, depth, heads, mlp_dim, z_dim)
        self.decoder = Decoder(1, patch_size, self.encoder.patch_embed.num_patches, dim, depth, heads, mlp_dim, z_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        # reparameterization trick
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(log_var * 0.5)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def inference(self, z):
        x_hat = self.decoder(z)
        return x_hat


if __name__ == '__main__':
    from torchinfo import summary
    vae = VAEViT(z_dim=1)
    summary(vae, input_size=(1, 1, 28, 28))
