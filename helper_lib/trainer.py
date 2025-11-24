from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import get_device
import torch.nn.functional as F
from torch import optim

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

def train_gan_model(model, batch_size=128, epochs=5, lr=2e-4, device="cpu"):
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

import torch
from tqdm import tqdm

def train_diffusion_model(model, dataloader, device="cpu", num_epochs=1):
    """
    Minimal diffusion training loop.

    Replace this with the real training logic later.
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for x, _ in dataloader:
            x = x.to(device)

            # Fake/no-op training step
            # (YOU WILL REPLACE THIS LATER)
            loss = model.training_step(x)

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

    return model

def make_beta_schedule(T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar

@torch.no_grad()
def q_sample(x0, t, alphas_bar):
    """x_t = sqrt(ā_t) x0 + sqrt(1-ā_t) * eps"""
    b = x0.size(0)
    a_bar_t = alphas_bar[t].view(b, 1, 1, 1)
    noise   = torch.randn_like(x0)
    return (a_bar_t.sqrt()) * x0 + (1 - a_bar_t).sqrt() * noise, noise

def train_diffusion(model, dataloader, device="cpu", num_steps=2000):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step, (x, _) in enumerate(dataloader):
        x = x.to(device)

        loss = model.training_step(x)  # your diffusion loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[Diffusion] step={step}, loss={loss.item():.4f}")

    return model

# ------------- Energy-Based Model (Langevin sampling) -------------
def langevin_step(x, energy_fn, step_size=0.1, n_steps=20):
    """
    Performs gradient descent on input x to lower energy (EBM sampling).
    """
    x.requires_grad_(True)
    for _ in range(n_steps):
        energy = energy_fn(x).sum()
        grad, = torch.autograd.grad(energy, x, create_graph=False)
        x = x - 0.5 * step_size * grad + torch.sqrt(torch.tensor(step_size, device=x.device)) * torch.randn_like(x)
        x = x.clamp(-1, 1).detach().requires_grad_(True)
    return x.detach()

def train_ebm_model(model, data_loader, device="cpu", epochs=3, lr=1e-4, n_neg=1, step_size=0.1, n_steps=20, l2=1e-4):
    """
    NCE-style: minimize energy on real, maximize on negatives drawn by Langevin.
    """
    model.to(device)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            # draw negative samples from noise then refine with Langevin
            x_neg = torch.randn_like(x)
            with torch.no_grad():
                x_neg = langevin_step(x_neg, model, step_size=step_size, n_steps=n_steps)

            e_real = model(x)       # low desired
            e_fake = model(x_neg)   # high desired
            loss = e_real.mean() - e_fake.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
