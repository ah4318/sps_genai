import torch
import torch.nn as nn

class SimpleDiffusion(nn.Module):
    def __init__(self, img_size=28, channels=1, timesteps=100):
        super().__init__()
        self.timesteps = timesteps
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1)
        )

    def forward(self, x, t):
        return self.model(x)

@torch.no_grad()
def sample_diffusion(model, num_samples=16, img_size=28, channels=1, steps=100):
    device = next(model.parameters()).device
    x = torch.randn(num_samples, channels, img_size, img_size, device=device)

    for t in reversed(range(steps)):
        noise_pred = model(x, t)
        x = x - 0.1 * noise_pred  # simple update rule

    return x
