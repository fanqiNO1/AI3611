import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, z_dim=1):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Use conv2d to replace maxpool2d, because maxunpool2d needs the index of maxpool2d
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32),
            nn.ReLU(),
        )
        # For mu and log_var, two branches
        self.mu = nn.Linear(32, z_dim)
        self.var = nn.Linear(32, z_dim)

    def forward(self, x):
        x = self.conv(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu = self.mu(x)
        log_var = self.var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim=1):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        # Reshape to (b, c, h, w)
        x = x.view(x.size(0), 256, 7, 7)
        x = self.deconv(x)
        return x


class VAEConv(nn.Module):
    def __init__(self, z_dim=1):
        super(VAEConv, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

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


if __name__ == "__main__":
    from torchinfo import summary
    vae = VAEConv(z_dim=1)
    summary(vae, input_size=(1, 1, 28, 28))
    