import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys

# ================================
# CONFIG
# ================================
CATEGORY = sys.argv[1]  # change per object
DATA_PATH = f"MVTEC-AD/{CATEGORY}/test"

BATCH_SIZE = 8
EPOCHS = 10
LR = 0.0005

# ================================
# TRANSFORMS (STRONG AUGMENTATION)
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

classes = dataset.classes
print("Classes:", classes)

# ================================
# MODEL (TRANSFER LEARNING)
# ================================
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# ================================
# TRAIN LOOP
# ================================
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

# ================================
# SAVE FINAL MODEL
# ================================
# Create model directory
model_dir = "mvtec_models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, f"resnet_mvtec_{CATEGORY}_final.pth")
torch.save(model.state_dict(), model_path)

print(f"Final model saved: {model_path}")