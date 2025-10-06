import torch.nn as nn
from typing import Literal

class FCNN(nn.Module):
    def __init__(self, in_dim: int = 28*28, num_classes: int = 10, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x): return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # assumes ~224x224 inputs -> 64 * 56 * 56
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*56*56, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

class EnhancedCNN(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10, p_drop: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p_drop),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p_drop),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*56*56, 512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

def get_model(model_name: Literal["FCNN","CNN","EnhancedCNN"]="CNN",
              num_classes: int = 10, in_channels: int = 3) -> nn.Module:
    name = model_name.upper()
    if name == "FCNN": return FCNN(28*28, num_classes)
    if name == "CNN": return SimpleCNN(in_channels, num_classes)
    if name == "ENHANCEDCNN": return EnhancedCNN(in_channels, num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")

