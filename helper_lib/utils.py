import os, random, numpy as np, torch
from typing import Union
from torch import nn

def get_device(pref: Union[str, torch.device] = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

def load_model(model: nn.Module, path: str, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd); model.eval()
    print(f"Loaded model from {path}")
    return model

