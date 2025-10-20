import torch.nn as nn
from typing import Literal
import torch
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self, in_dim: int = 28*28, num_classes: int = 10, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x): return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # assumes ~224x224 inputs -> 64 * 56 * 56
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*56*56, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

class EnhancedCNN(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10, p_drop: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p_drop),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p_drop),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*56*56, 512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

def get_model(model_name: Literal["FCNN","CNN","EnhancedCNN"]="CNN",
              num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    name = model_name.upper()
    if name == "FCNN": return FCNN(28*28, num_classes)
    if name == "CNN": return SimpleCNN(in_channels, num_classes)
    if name == "ENHANCEDCNN": return EnhancedCNN(in_channels, num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")


class ConvVAE(nn.Module):
    """
    Convolutional VAE with dynamic flatten size calculation.
    Works for square images whose size is divisible by 4 (after 2 strides).
    Use normalize=None in the loader so inputs are ~[0,1] for BCE loss.
    """
    def __init__(self, in_ch: int = 1, image_size: int = 64, latent_dim: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Encoder: downsample by 2 twice => H/4, W/4
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1), nn.ReLU(),   # /2
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),      # /4
        )

        # compute flatten size dynamically
        with torch.no_grad():
            _dummy = torch.zeros(1, in_ch, image_size, image_size)
            _h = self.enc(_dummy)
            self.enc_shape = _h.shape  # (1, C, H, W)
            self.flat_dim = _h.numel()

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # Decoder: upsample by 2 twice => back to H, W
        C, H, W = self.enc_shape[1], self.enc_shape[2], self.enc_shape[3]
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(C, 32, 4, stride=2, padding=1), nn.ReLU(),  # x2
            nn.ConvTranspose2d(32, in_ch, 4, stride=2, padding=1),
            nn.Sigmoid()  # output in [0,1] for BCE
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.enc_shape[1], self.enc_shape[2], self.enc_shape[3])
        xhat = self.dec(h)
        return xhat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

def get_model(
    model_name: str = "CNN",
    num_classes: int = 10,
    in_channels: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory returning the requested model.
    Added: model_name="VAE" with kwargs: image_size (int), latent_dim (int).
    """
    name = model_name.upper()
    if name == "FCNN":
        return FCNN(in_dim=28*28, num_classes=num_classes)
    elif name == "CNN":
        return SimpleCNN(in_ch=in_channels, num_classes=num_classes)
    elif name == "ENHANCEDCNN":
        return EnhancedCNN(in_ch=in_channels, num_classes=num_classes)
    elif name == "VAE":
        image_size = kwargs.get("image_size", 64)
        latent_dim = kwargs.get("latent_dim", 32)
        return ConvVAE(in_ch=in_channels, image_size=image_size, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
