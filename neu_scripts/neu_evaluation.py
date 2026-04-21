import os
import json
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

from SemanticEncoder import semantic_encoder, secure_encode
from SemanticDecoder import secure_decode
from AddNoise import add_semantic_noise, add_channel_noise

# ================================
# CONFIG
# ================================
classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
dataset_path = "NEU-DET/train/images"

# ================================
# LOAD MODEL
# ================================
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("resnet_neu.pth", map_location="cpu", weights_only=True))
model.eval()

# ================================
# TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# SEMANTIC NOISE (IMPROVED)
# ================================
# def add_semantic_noise(semantic_data, drop_prob=0.2, flip_prob=0.1):
#     noisy = semantic_data.copy()
#     noisy_defects = []

#     for defect in semantic_data["defects"]:

#         # Drop defect
#         if random.random() < drop_prob:
#             continue

#         new_defect = defect.copy()

#         # Flip class
#         if random.random() < flip_prob:
#             new_defect["class"] = random.choice(classes)

#         noisy_defects.append(new_defect)

#     noisy["defects"] = noisy_defects
#     noisy["num_defects"] = len(noisy_defects)

#     return noisy


# ================================
# CHANNEL NOISE (BIT FLIP)
# ================================
# def add_channel_noise(data, noise_level=0.02):
#     noisy = bytearray(data)
#     for i in range(len(noisy)):
#         if random.random() < noise_level:
#             noisy[i] ^= 0xFF
#     return bytes(noisy)


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
for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    for img_file in os.listdir(class_path):

        img_path = os.path.join(class_path, img_file)

        # IMAGE SIZE
        image_sizes.append(os.path.getsize(img_path))

        # LOAD IMAGE
        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        # MODEL PREDICTION
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        pred_class = classes[pred.item()]

        # ACCURACY
        if pred_class == class_name:
            correct += 1
        total += 1

        prediction = {
            "class": pred_class,
            "confidence": float(confidence.item())
        }

        # =====================
        # SEMANTIC ENCODING
        # =====================
        semantic_data = semantic_encoder(img_file, prediction)
        encoded = secure_encode(semantic_data)

        encoded_sizes.append(len(encoded))

        # Save temp
        with open("temp.bin", "wb") as f:
            f.write(encoded)

        # =====================
        # 1. NO NOISE
        # =====================
        try:
            decoded = secure_decode("temp.bin")
            decoded_class = decoded["defects"][0]["class"]

            if decoded_class == pred_class:
                decode_no_noise += 1
        except:
            pass

        # =====================
        # 2. SEMANTIC NOISE
        # =====================
        noisy_semantic = add_semantic_noise(
            semantic_data,
            classes=classes,
            drop_prob=0.2,
            flip_prob=0.1
        )

        encoded_noisy = secure_encode(noisy_semantic)

        with open("temp_semantic.bin", "wb") as f:
            f.write(encoded_noisy)

        try:
            decoded = secure_decode("temp_semantic.bin")
            decoded_class = decoded["defects"][0]["class"]

            if decoded_class == pred_class:
                decode_semantic_noise += 1
        except:
            pass

        # =====================
        # 3. CHANNEL NOISE
        # =====================
        noisy_channel = add_channel_noise(encoded)

        with open("temp_channel.bin", "wb") as f:
            f.write(noisy_channel)

        try:
            decoded = secure_decode("temp_channel.bin")
            decoded_class = decoded["defects"][0]["class"]

            if decoded_class == pred_class:
                decode_channel_noise += 1
        except:
            pass


# ================================
# FINAL RESULTS
# ================================
accuracy = correct / total
no_noise_rate = decode_no_noise / total
semantic_noise_rate = decode_semantic_noise / total
channel_noise_rate = decode_channel_noise / total

avg_image = sum(image_sizes) / len(image_sizes)
avg_encoded = sum(encoded_sizes) / len(encoded_sizes)

print("\n===== FINAL RESULTS =====")
print("Model Accuracy:", accuracy)
print("Decode Success (No Noise):", no_noise_rate)
print("Decode Success (Semantic Noise):", semantic_noise_rate)
print("Decode Success (Channel Noise):", channel_noise_rate)

print("\nAvg Image Size:", avg_image)
print("Avg Semantic Size:", avg_encoded)
print("Compression Gain (~x):", avg_image / avg_encoded)


# ================================
# PLOTS
# ================================

# SIZE COMPARISON
plt.figure()
plt.plot(image_sizes, label="Image Size")
plt.plot(encoded_sizes, label="Semantic Size")
plt.legend()
plt.title("Image vs Semantic Transmission Size")
plt.xlabel("Samples")
plt.ylabel("Bytes")
plt.show()

# ACCURACY
plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Accuracy")
plt.show()

# NOISE ANALYSIS
plt.figure()
plt.bar(
    ["No Noise", "Semantic Noise", "Channel Noise"],
    [no_noise_rate, semantic_noise_rate, channel_noise_rate]
)
plt.title("Transmission Robustness")
plt.show()