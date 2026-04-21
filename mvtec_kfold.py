import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# ================================
# CONFIG
# ================================
CATEGORY = "capsule"
DATA_PATH = f"MVTEC-AD/{CATEGORY}/test"

K = 5
EPOCHS = 12
BATCH_SIZE = 8

# ================================
# TRANSFORMS
# ================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# IMPORTANT: separate datasets
train_dataset_full = datasets.ImageFolder(DATA_PATH, transform=train_transform)
val_dataset_full = datasets.ImageFolder(DATA_PATH, transform=val_transform)

kf = KFold(n_splits=K, shuffle=True, random_state=42)

accuracies = []

# ================================
# K-FOLD LOOP
# ================================
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset_full)):

    print(f"\n===== Fold {fold+1} =====")

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(val_dataset_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    # Model
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last layer block
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset_full.classes))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0005)

    # --------------------
    # TRAIN
    # --------------------
    for epoch in range(EPOCHS):
        model.train()

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --------------------
    # VALIDATE
    # --------------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    accuracies.append(acc)

    print(f"Fold Accuracy: {acc}")

# ================================
# FINAL RESULTS
# ================================
print("\n===== FINAL K-FOLD RESULT =====")
print("Average Accuracy:", np.mean(accuracies))
print("Std Dev:", np.std(accuracies))