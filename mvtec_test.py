import os
import json
import zlib
from cryptography.fernet import Fernet
from jsonschema import validate

# Load schema
with open("semantic_schema.json") as f:
    SCHEMA = json.load(f)

# Encryption key (same as before)
KEY = b'PASTE_YOUR_KEY'
cipher = Fernet(KEY)

# ----------------------------
# NORMALIZATION
# ----------------------------
def normalize_mvtec(image_path):
    parts = image_path.split(os.sep)
    defect_type = parts[-2]

    if defect_type == "good":
        return None

    return {
        "class": defect_type,
        "bbox": [0, 0, 0, 0],
        "confidence": 1.0
    }

# ----------------------------
# SEMANTIC ENCODER
# ----------------------------
def semantic_encoder(image_id, detections):
    data = {
        "image_id": image_id,
        "num_defects": len(detections),
        "defects": detections
    }

    validate(instance=data, schema=SCHEMA)
    return data

# ----------------------------
# SECURE ENCODE
# ----------------------------
def secure_encode(data):
    json_data = json.dumps(data).encode()
    compressed = zlib.compress(json_data)
    encrypted = cipher.encrypt(compressed)
    return encrypted

# ----------------------------
# RUN PIPELINE
# ----------------------------
dataset_path = "mvtec/bottle/test"
output_dir = "mvtec_output"

os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".png"):
            path = os.path.join(root, file)
            image_id = file.split(".")[0]

            det = normalize_mvtec(path)

            if det is None:
                detections = []
            else:
                detections = [det]

            semantic_data = semantic_encoder(image_id, detections)
            encrypted = secure_encode(semantic_data)

            with open(os.path.join(output_dir, image_id + ".bin"), "wb") as f:
                f.write(encrypted)

print("MVTec semantic encoding complete!")