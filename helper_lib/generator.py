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

