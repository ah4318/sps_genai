from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import get_device

def train_model(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: str | torch.device = "auto",
    epochs: int = 10, log_every: int = 0
) -> nn.Module:
    device = get_device(device); model.to(device); model.train()
    for epoch in range(1, epochs+1):
        running_loss, running_correct, total = 0.0, 0, 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            if log_every and total % log_every == 0:
                pbar.set_postfix(loss=running_loss/total, acc=running_correct/total)
        print(f"[Train] Epoch {epoch}: loss={running_loss/total:.4f}, acc={running_correct/total:.4f}")
    return model

def _vae_loss_bce_kl(x, xhat, mu, logvar):
    # reconstruction loss (sum over pixels; average over batch)
    # Ensure x and xhat are in [0,1] (use normalize=None in loader)
    bce = nn.functional.binary_cross_entropy(
        xhat, x, reduction="sum"
    )
    # KL divergence (sum over latent dims; average over batch)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld)

def train_vae_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion=None,  # if None, uses BCE+KLD
    optimizer: torch.optim.Optimizer = None,
    device: str | torch.device = "auto",
    epochs: int = 10,
) -> nn.Module:
    device = get_device(device)
    model.to(device)
    model.train()

    if criterion is None:
        loss_fn = _vae_loss_bce_kl
    else:
        loss_fn = criterion

    for epoch in range(1, epochs + 1):
        total_loss, total = 0.0, 0
        pbar = tqdm(data_loader, desc=f"[VAE] Epoch {epoch}/{epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()

            xhat, mu, logvar = model(x)
            loss = loss_fn(x, xhat, mu, logvar)
            # average per batch
            loss = loss / x.size(0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pbar.set_postfix(loss=total_loss / total)

        print(f"[VAE][Train] Epoch {epoch}: loss={total_loss/total:.4f}")

    return model
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train_gan(model, batch_size=128, epochs=5, lr=2e-4, device="cpu"):
    device = torch.device(device)
    G, D = model.G.to(device), model.D.to(device)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()
    z_dim = model.z_dim

    for epoch in range(1, epochs+1):
        for real, _ in loader:
            real = real.to(device)
            z = torch.randn(real.size(0), z_dim, device=device)
            fake = G(z)

            logits_real = D(real)
            loss_real = bce(logits_real, torch.ones_like(logits_real))
            logits_fake = D(fake.detach())
            loss_fake = bce(logits_fake, torch.zeros_like(logits_fake))
            loss_D = loss_real + loss_fake
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            logits_fake = D(fake)
            loss_G = bce(logits_fake, torch.ones_like(logits_fake))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        print(f"Epoch {epoch}/{epochs} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

    return model

