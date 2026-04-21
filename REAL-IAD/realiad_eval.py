import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np

import zlib
from cryptography.fernet import Fernet

from SemanticEncoder import semantic_encoder, secure_encode
from AddNoise import add_semantic_noise, add_channel_noise

# ================================
# CONFIG
# ================================
DATA_PATH = "Real-IAD/audiojack"
MODEL_PATH = "resnet_realiad_audiojack.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "OK": "no_defect",
    "AK": "pit",
    "BX": "deformation",
    "CH": "abrasion",
    "HS": "scratch",
    "PS": "damage",
    "QS": "missing_parts",
    "YW": "foreign_object",
    "ZW": "contamination"
}

# ================================
# LOAD DATA
# ================================
samples = []

for condition in ["OK", "NG"]:
    condition_path = os.path.join(DATA_PATH, condition)

    if condition == "OK":
        for sample in os.listdir(condition_path):
            sample_path = os.path.join(condition_path, sample)

            for file in os.listdir(sample_path):
                if file.endswith(".jpg"):
                    samples.append((os.path.join(sample_path, file), "OK"))

    else:
        for defect_code in os.listdir(condition_path):
            defect_path = os.path.join(condition_path, defect_code)

            for sample in os.listdir(defect_path):
                sample_path = os.path.join(defect_path, sample)

                for file in os.listdir(sample_path):
                    if file.endswith(".jpg"):
                        samples.append((os.path.join(sample_path, file), defect_code))

print(f"\nTotal images: {len(samples)}")

# ================================
# CLASSES
# ================================
classes = sorted(list(set([LABEL_MAP[label] for _, label in samples])))
print("Classes:", classes)

# ================================
# MODEL
# ================================
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ================================
# TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# ================================
# DECODE HELPER
# ================================
KEY = b'w067i_naKDKCAh3hK9XaVQLi-IKP-UDfobPW8TwZbFM='
cipher = Fernet(KEY)

def decode_bytes(encoded_bytes):
    decrypted = cipher.decrypt(encoded_bytes)
    decompressed = zlib.decompress(decrypted)
    return json.loads(decompressed.decode())

# ================================
# MASK → LOCATION
# ================================
def extract_location(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0.0, [0,0,0,0], [0,0]

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    area = len(xs) / mask.size
    cx = float(xs.mean())
    cy = float(ys.mean())

    return area, [x_min, y_min, x_max, y_max], [cx, cy]

# ================================
# METRICS
# ================================
correct = 0
total = 0

decode_no_noise = 0
decode_semantic_noise = 0
decode_channel_noise = 0

image_sizes = []
encoded_sizes = []

# ================================
# MAIN LOOP
# ================================
for idx, (img_path, label_code) in enumerate(samples):

    if idx % 200 == 0:
        print(f"Processing {idx}/{len(samples)}")

    true_label = LABEL_MAP[label_code]

    image_sizes.append(os.path.getsize(img_path))

    image = Image.open(img_path).convert("RGB")
    image = transform(image).to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    pred_class = classes[pred.item()]

    if pred_class == true_label:
        correct += 1
    total += 1

    # =====================
    # LOCATION
    # =====================
    mask_path = img_path.replace(".jpg", ".png")

    if os.path.exists(mask_path):
        area, bbox, center = extract_location(mask_path)
    else:
        area, bbox, center = 0.0, [0,0,0,0], [0,0]

    prediction = {
        "class": pred_class,
        "confidence": float(conf.item()),
        "area": round(area, 4),
        "bbox": bbox,
        "center": [round(center[0],2), round(center[1],2)]
    }

    # =====================
    # ENCODE
    # =====================
    semantic_data = semantic_encoder(os.path.basename(img_path), prediction)
    encoded = secure_encode(semantic_data)

    encoded_sizes.append(len(encoded))

    # =====================
    # NO NOISE
    # =====================
    decoded = decode_bytes(encoded)

    if pred_class == "no_defect":
        if decoded["num_defects"] == 0:
            decode_no_noise += 1
    else:
        if len(decoded["defects"]) > 0:
            if decoded["defects"][0]["class"] == pred_class:
                decode_no_noise += 1

    # =====================
    # SEMANTIC NOISE
    # =====================
    noisy_semantic = add_semantic_noise(
        semantic_data,
        classes=classes,
        drop_prob=0.2,
        flip_prob=0.1
    )

    encoded_noisy = secure_encode(noisy_semantic)
    decoded = decode_bytes(encoded_noisy)

    if pred_class == "no_defect":
        if decoded["num_defects"] == 0:
            decode_semantic_noise += 1
    else:
        if len(decoded["defects"]) > 0:
            if decoded["defects"][0]["class"] == pred_class:
                decode_semantic_noise += 1

    # =====================
    # CHANNEL NOISE
    # =====================
    noisy_channel = add_channel_noise(encoded)

    try:
        decoded = decode_bytes(noisy_channel)

        if pred_class == "no_defect":
            if decoded["num_defects"] == 0:
                decode_channel_noise += 1
        else:
            if len(decoded["defects"]) > 0:
                if decoded["defects"][0]["class"] == pred_class:
                    decode_channel_noise += 1
    except:
        pass

# ================================
# FINAL METRICS
# ================================
accuracy = correct / total
no_noise = decode_no_noise / total
semantic_noise = decode_semantic_noise / total
channel_noise = decode_channel_noise / total

avg_image = sum(image_sizes) / len(image_sizes)
avg_encoded = sum(encoded_sizes) / len(encoded_sizes)

results = {
    "accuracy": accuracy,
    "no_noise": no_noise,
    "semantic_noise": semantic_noise,
    "channel_noise": channel_noise,
    "avg_image_size": avg_image,
    "avg_encoded_size": avg_encoded,
    "compression_ratio": avg_image / avg_encoded
}

# SAVE RESULTS
with open("realiad_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n===== RESULTS =====")
print(json.dumps(results, indent=4))

# ================================
# PLOTS
# ================================

# SIZE COMPARISON
plt.figure()
plt.plot(image_sizes, label="Image Size")
plt.plot(encoded_sizes, label="Semantic Size")
plt.legend()
plt.title("Image vs Semantic Size")
plt.xlabel("Samples")
plt.ylabel("Bytes")
plt.show()

# ACCURACY
plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Accuracy")
plt.show()

# NOISE
plt.figure()
plt.bar(
    ["No Noise", "Semantic Noise", "Channel Noise"],
    [no_noise, semantic_noise, channel_noise]
)
plt.title("Noise Robustness")
plt.show()