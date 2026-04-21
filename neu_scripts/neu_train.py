import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

print("Script started")

# ================================
# DATASET CLASS (FIXED)
# ================================
class NEUDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        images_root = os.path.join(root_dir, "images")

        for class_name in os.listdir(images_root):
            class_path = os.path.join(images_root, class_name)

            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    if img.endswith(".jpg"):
                        self.samples.append(
                            (os.path.join(class_path, img), class_name)
                        )

        self.classes = sorted(list(set([s[1] for s in self.samples])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print("Classes:", self.classes)
        print("Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[label]


# ================================
# TRANSFORMS
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# LOAD DATASET
# ================================
dataset_path = "NEU-DET/train"

dataset = NEUDataset(dataset_path, transform)

if len(dataset) == 0:
    print("ERROR: Dataset is empty. Check path.")
    exit()

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ================================
# MODEL
# ================================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ================================
# TRAINING SETUP
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================================
# TRAINING LOOP
# ================================
epochs = 5

for epoch in range(epochs):
    print(f"\nStarting Epoch {epoch+1}")

    model.train()
    total_loss = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed, Avg Loss: {total_loss/len(loader):.4f}")

# ================================
# SAVE MODEL
# ================================
torch.save(model.state_dict(), "resnet_neu.pth")

print("\nTraining complete. Model saved as resnet_neu.pth")


# ================================
# PREDICTION FUNCTION
# ================================
def predict(image, model, classes):
    model.eval()

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

    return {
        "class": classes[pred.item()],
        "confidence": confidence.item()
    }