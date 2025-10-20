import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from app.cnn_model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Dataset ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# === 2. Model ===
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 3. Training ===
for epoch in range(2):  # small epochs for demo
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/2], Loss: {running_loss/len(trainloader):.4f}")

# === 4. Save Model ===
torch.save({"model_state": model.state_dict()}, "models/cnn_cifar10.pt")
print("✅ Model saved to models/cnn_cifar10.pt")

