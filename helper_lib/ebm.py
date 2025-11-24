import torch
import torch.nn as nn

class SimpleEBM(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28*channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.energy(x)

def sample_ebm(model, steps=60, lr=0.1, num_samples=16, img_size=28, channels=1):
    device = next(model.parameters()).device
    x = torch.randn(num_samples, channels, img_size, img_size, device=device)
    x.requires_grad = True

    optimizer = torch.optim.SGD([x], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        energy = model(x).sum()
        energy.backward()
        optimizer.step()

    return x.detach()
