from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from .utils import get_device

@torch.no_grad()
def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
    device: str | torch.device = "auto"
) -> Tuple[float,float]:
    device = get_device(device)
    model.to(device); model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x,y in data_loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    avg_loss, acc = total_loss/total, total_correct/total
    print(f"[Eval] loss={avg_loss:.4f}, acc={acc:.4f}")
    return avg_loss, acc

