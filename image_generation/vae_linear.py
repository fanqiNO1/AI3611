import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, z_dim=1):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.mu = nn.Linear(32, z_dim)
        self.var = nn.Linear(32, z_dim)

    def forward(self, x):
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
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        # Reshape to (b, c, h, w)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class VAELinear(nn.Module):
    def __init__(self, z_dim=1):
        super(VAELinear, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        eps = torch.randn_like(mu)
        # reparameterization trick
        z = mu + eps * torch.exp(log_var * 0.5)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def inference(self, z):
        x_hat = self.decoder(z)
        return x_hat


if __name__ == '__main__':
    from torchinfo import summary
    vae = VAELinear(z_dim=1)
    summary(vae, input_size=(1, 1, 28, 28))