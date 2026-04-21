import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from SemanticEncoder import secure_encode

# ================================
# LOAD MODEL
# ================================
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(torch.load("resnet_neu.pth", map_location="cpu"))
model.eval()

# ================================
# TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# PREDICT FUNCTION
# ================================
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

    return {
        "class": classes[pred.item()],
        "confidence": float(confidence.item())
    }

# ================================
# TEST IMAGE
# ================================
img_path = "NEU-DET/train/images/crazing/crazing_1.jpg"

prediction = predict(img_path)

semantic_data = {
    "image_id": "sample_1",
    "num_defects": 1,
    "defects": [prediction]
}

print("\nPrediction Output:")
print(semantic_data)
secure_encode(semantic_data)

# print("\nPrediction Output:")
# print(result)