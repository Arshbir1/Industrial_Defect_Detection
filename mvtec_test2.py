import os
import json
import zlib
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from cryptography.fernet import Fernet
from jsonschema import validate

# ================================
# LOAD SCHEMA
# ================================
with open("semantic_schema.json") as f:
    SCHEMA = json.load(f)

# ================================
# LOAD KEY
# ================================
KEY = b'V5PmW8yRshoGqM3M8Ojtsky4lgLmdRkF8hSlov-qsJ8='
cipher = Fernet(KEY)

# ================================
# LOAD MODEL
# ================================
classes = ['crack', 'faulty_imprint', 'no_defect', 'poke', 'scratch', 'squeeze']

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("resnet_mvtec.pth", map_location="cpu", weights_only=True))
model.eval()

# ================================
# TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# SEMANTIC ENCODER
# ================================
def semantic_encoder(image_id, prediction):
    if prediction["class"] == "no_defect":
        defects = []
    else:
        defects = [prediction]

    data = {
        "image_id": image_id,
        "num_defects": len(defects),
        "defects": defects
    }

    validate(instance=data, schema=SCHEMA)
    return data

# ================================
# ENCODE
# ================================
def secure_encode(data):
    json_data = json.dumps(data).encode()
    compressed = zlib.compress(json_data)
    encrypted = cipher.encrypt(compressed)
    return encrypted

# ================================
# RUN PIPELINE
# ================================
dataset_path = "MVTEC-AD/bottle/test"
output_dir = "mvtec_output"

os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".png"):

            img_path = os.path.join(root, file)
            print("Processing:", img_path)
            image_id = file.split(".")[0]

            # LOAD IMAGE
            image = Image.open(img_path).convert("RGB")
            image = transform(image)

            # MODEL PREDICTION
            with torch.no_grad():
                output = model(image.unsqueeze(0))
                probs = F.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            pred_class = classes[pred.item()]

            prediction = {
                "class": pred_class,
                "confidence": float(confidence.item()),
                "bbox": [0, 0, 0, 0]
            }

            # SEMANTIC PIPELINE
            semantic_data = semantic_encoder(image_id, prediction)
            encrypted = secure_encode(semantic_data)

            with open(os.path.join(output_dir, image_id + ".bin"), "wb") as f:
                f.write(encrypted)

print("MVTec model-based semantic encoding complete!")