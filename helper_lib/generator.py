import math
import torch
import matplotlib.pyplot as plt
from .utils import get_device

@torch.no_grad()
def generate_samples(
    model,
    device: str | torch.device = "auto",
    num_samples: int = 16,
    nrow: int = 4,
    figsize=(6, 6),
    save_path: str | None = None,
):
    """
    Sample z ~ N(0, I), decode, and plot a grid.
    Works with ConvVAE (model.decode exists, output in [0,1]).
    """
    device = get_device(device)
    model.to(device)
    model.eval()

    z = torch.randn(num_samples, model.latent_dim, device=device)
    xhat = model.decode(z).cpu()  # (N, C, H, W) in [0,1]

    # build grid manually
    C, H, W = xhat.shape[1], xhat.shape[2], xhat.shape[3]
    nrow = nrow or int(math.sqrt(num_samples))
    ncol = math.ceil(num_samples / nrow)

    fig, axs = plt.subplots(ncol, nrow, figsize=figsize)
    axs = axs.ravel() if hasattr(axs, "ravel") else [axs]

    for i in range(num_samples):
        img = xhat[i]
        if C == 1:
            axs[i].imshow(img.squeeze(0), cmap="gray", vmin=0, vmax=1)
        else:
            axs[i].imshow(img.permute(1, 2, 0).clamp(0,1))
        axs[i].axis("off")

    for j in range(i+1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"Saved samples to {save_path}")
    plt.show()

from torchvision.utils import make_grid

@torch.no_grad()
def generate_samples(model, device="cpu", num_samples=16):
    device = torch.device(device)
    G = model.G.to(device).eval()
    z_dim = getattr(model, "z_dim", 100)
    z = torch.randn(num_samples, z_dim, device=device)
    imgs = G(z).cpu()
    imgs = (imgs + 1) / 2.0
    grid = make_grid(imgs, nrow=int(num_samples**0.5))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

@torch.no_grad()
def generate_samples(model, device='cpu', num_samples=16, diffusion_steps=100):
    """
    Reverse diffusion: start from noise and iteratively denoise.
    """
    model.to(device).eval()
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    for step in range(diffusion_steps):
        x = model(x)  # denoise slightly each step
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    grid = make_grid(x.cpu(), nrow=int(num_samples ** 0.5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

def generate_diffusion_samples(num_samples=16):
    model = get_model("DIFFUSION")
    return run_diffusion(model, num_samples)

def generate_ebm_samples(num_samples=16):
    model = get_model("EBM")
    return run_ebm(model, num_samples)
