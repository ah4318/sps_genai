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

