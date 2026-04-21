import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader

from realiad_loader import RealIADDataset

# ================================
# CONFIG
# ================================
DATA_PATH = "Real-IAD/pcb"   # change per object
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# TRANSFORMS
# ================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# ================================
# DATASETS
# ================================
train_dataset = RealIADDataset(DATA_PATH, transform, split="train")
val_dataset   = RealIADDataset(DATA_PATH, transform, split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ================================
# MODEL
# ================================
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model.to(device)

# ================================
# TRAINING SETUP
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================================
# TRAIN LOOP
# ================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # ----------------------------
    # VALIDATION
    # ----------------------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {correct/total:.4f}")

# ================================
# SAVE MODEL
# ================================
torch.save(model.state_dict(), "resnet_realiad_pcb.pth")

print("Model saved!")