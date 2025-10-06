from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(
    data_dir: str, batch_size: int = 32, train: bool = True,
    num_workers: int = 2, image_size: int = 224,
    normalize: Tuple[Tuple[float,float,float], Tuple[float,float,float]] =
        ((0.485,0.456,0.406),(0.229,0.224,0.225)),
    pin_memory: bool = True,
) -> DataLoader:
    mean, std = normalize
    if train:
        tfm = transforms.Compose([
            transforms.Resize(int(1.15*image_size)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        shuffle = True
    else:
        tfm = transforms.Compose([
            transforms.Resize(int(1.15*image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        shuffle = False

    ds = datasets.ImageFolder(root=data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)

