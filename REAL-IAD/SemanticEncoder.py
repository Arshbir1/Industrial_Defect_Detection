import json
import zlib
from cryptography.fernet import Fernet
from jsonschema import validate

# ================================
# SHARED KEY
# ================================
KEY = b'w067i_naKDKCAh3hK9XaVQLi-IKP-UDfobPW8TwZbFM='
cipher = Fernet(KEY)

# ================================
# SCHEMA (UPDATED)
# ================================
SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string"},
        "num_defects": {"type": "number"},
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string"},
                    "confidence": {"type": "number"},
                    "area": {"type": "number"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "center": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "required": ["class", "confidence"]
            }
        }
    },
    "required": ["image_id", "num_defects", "defects"]
}

# ================================
# ENCODER
# ================================
def semantic_encoder(image_id, prediction):

    defect_class = prediction["class"]

    # Clean + enforce structure
    clean_prediction = {
        "class": defect_class,
        "confidence": float(prediction["confidence"]),
        "area": float(prediction.get("area", 0.0)),
        "bbox": prediction.get("bbox", [0, 0, 0, 0]),
        "center": prediction.get("center", [0, 0])
    }

    if defect_class == "no_defect":
        semantic_data = {
            "image_id": image_id,
            "num_defects": 0,
            "defects": []
        }
    else:
        semantic_data = {
            "image_id": image_id,
            "num_defects": 1,
            "defects": [clean_prediction]
        }

    validate(instance=semantic_data, schema=SCHEMA)
    return semantic_data


# ================================
# ENCODE
# ================================
def secure_encode(semantic_data):
    json_data = json.dumps(semantic_data).encode()
    compressed = zlib.compress(json_data)
    encrypted = cipher.encrypt(compressed)
    return encrypted


def save_encoded_file(data, path):
    with open(path, "wb") as f:
        f.write(data)