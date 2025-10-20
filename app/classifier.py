import io, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from app.cnn_model import SimpleCNN

CIFAR10_CLASSES = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tf = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
])
_model = SimpleCNN().to(_device)

def load_model_once(weights="models/cnn_cifar10.pt"):
    global _model
    if _model is None:
        _model = CNNClassifier().to(_device)
        ckpt = torch.load(weights, map_location=_device)
        _model.load_state_dict(ckpt["state_dict"])
        _model.eval()
    return _model

@torch.no_grad()
def predict_image_bytes(image_bytes: bytes):
    model = load_model_once()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(_device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    idx = int(torch.argmax(probs).item())
    return {
        "class_idx": idx,
        "class_name": CIFAR10_CLASSES[idx],
        "probability": float(probs[idx].item())
    }

