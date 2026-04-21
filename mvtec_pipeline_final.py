import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from SemanticEncoder import semantic_encoder, secure_encode, save_encoded_file
from SemanticDecoder import secure_decode, interpret
import sys

# ================================
# CONFIG
# ================================
CATEGORY = sys.argv[1]
DATA_PATH = f"MVTEC-AD/{CATEGORY}/test"
MODEL_PATH = os.path.join("mvtec_models", f"resnet_mvtec_{CATEGORY}_final.pth")

# ================================
# LOAD MODEL
# ================================
classes = sorted(os.listdir(DATA_PATH))

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# ================================
# TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

base_output_dir = "mvtec_secure_output"
os.makedirs(base_output_dir, exist_ok=True)

output_dir = os.path.join(base_output_dir, CATEGORY)
os.makedirs(output_dir, exist_ok=True)

# ================================
# PIPELINE LOOP
# ================================
for true_label in classes:

    folder = os.path.join(DATA_PATH, true_label)

    for img_file in os.listdir(folder):

        img_path = os.path.join(folder, img_file)

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        # -------------------------
        # MODEL INFERENCE
        # -------------------------
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        pred_class = classes[pred.item()]

        prediction = {
            "class": pred_class,
            "confidence": float(conf.item())
        }

        # -------------------------
        # SEMANTIC ENCODING
        # -------------------------
        semantic_data = semantic_encoder(img_file, prediction)
        encoded = secure_encode(semantic_data)

        image_id = os.path.splitext(img_file)[0]
        out_path = os.path.join(output_dir, image_id + ".bin")

        save_encoded_file(encoded, out_path)

        # -------------------------
        # DECODING (verification)
        # -------------------------
        decoded = secure_decode(out_path)

        print("\n--- RESULT ---")
        interpret(decoded)

print("\nFull pipeline complete!")