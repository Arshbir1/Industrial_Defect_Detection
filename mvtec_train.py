import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

print("Script started")

# ================================
# CONFIG
# ================================
DATASET_PATH = "MVTEC-AD/capsule"   # change category if needed
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001

# ================================
# CLASS MAPPING
# ================================
classes = set()

def collect_classes(root_path):
    for split in ["train", "test"]:
        split_path = os.path.join(root_path, split)
        for folder in os.listdir(split_path):
            if folder == "good":
                classes.add("no_defect")
            else:
                classes.add(folder)

collect_classes(DATASET_PATH)
classes = sorted(list(classes))
class_to_idx = {cls: i for i, cls in enumerate(classes)}

print("Classes:", classes)

# ================================
# CUSTOM DATASET
# ================================
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for split in ["train", "test"]:
            split_path = os.path.join(root_dir, split)

            for defect_type in os.listdir(split_path):
                defect_path = os.path.join(split_path, defect_type)

                if not os.path.isdir(defect_path):
                    continue

                label = "no_defect" if defect_type == "good" else defect_type

                for img_file in os.listdir(defect_path):
                    if img_file.endswith(".png"):
                        img_path = os.path.join(defect_path, img_file)
                        self.samples.append((img_path, class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ================================
# TRANSFORMS
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# DATASET + LOADER
# ================================
dataset = MVTecDataset(DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Total samples:", len(dataset))

# ================================
# MODEL
# ================================
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(classes))

# ================================
# LOSS + OPTIMIZER
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================================
# TRAINING LOOP
# ================================
for epoch in range(EPOCHS):
    print(f"\nStarting Epoch {epoch+1}")

    running_loss = 0.0

    for i, (images, labels) in enumerate(loader):

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed, Avg Loss: {running_loss/len(loader):.4f}")

# ================================
# SAVE MODEL
# ================================
torch.save(model.state_dict(), "resnet_mvtec.pth")

print("\nTraining complete. Model saved as resnet_mvtec.pth")