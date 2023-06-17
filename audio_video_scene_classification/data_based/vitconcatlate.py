import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class ViT(VisionTransformer):
    def __init__(
        self, 
        channels, 
        image_size, 
        patch_size, 
        num_classes, 
        dim=512, 
        depth=6, 
        heads=8, 
        mlp_ratio=4,
        drop_path=0.2
    ):
        super(ViT, self).__init__(
            in_chans=channels,
            img_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
        )
        
    def forward(self, x):
        x = self.patch_embed(x)
        b, n, _ = x.shape
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.fc_norm(x[:, 0])
        x = self.head_drop(x)
        x = self.head(x)
        return x
    
    
class ViTConcatLate(nn.Module):
    def __init__(self, audio_shape, video_shape, num_classes, pretrained=None):
        super(ViTConcatLate, self).__init__()
        self.audio_vit = ViT(
            channels=3,
            image_size=audio_shape,
            patch_size=(16, 16),
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_ratio=4,
            drop_path=0.2
        )
        self.video_vit = ViT(
            channels=3,
            image_size=video_shape,
            patch_size=(16, 16),
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_ratio=4,
            drop_path=0.2
        )
        if not pretrained is None:
            self.audio_vit.load_state_dict(torch.load(pretrained, map_location="cpu")["model"], strict=False)
            self.video_vit.load_state_dict(torch.load(pretrained, map_location="cpu")["model"], strict=False)
        
        
    def forward(self, audio, video):
        audio = self.audio_vit(audio)
        video = self.video_vit(video)
        x = (audio + video) / 2
        return x
    
    
if __name__ == "__main__":
    from torchinfo import summary
    audio = torch.randn(1, 1, 512, 64)
    video = torch.randn(1, 3, 224, 224)
    model = ViTConcatLate(audio.shape[2:], video.shape[2:], 10)
    summary(model, input_data=[audio, video])
    