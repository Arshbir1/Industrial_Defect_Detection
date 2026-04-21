import json
import zlib
from cryptography.fernet import Fernet
from jsonschema import validate

# ================================
# SAME KEY
# ================================
KEY = b'w067i_naKDKCAh3hK9XaVQLi-IKP-UDfobPW8TwZbFM='
cipher = Fernet(KEY)

# ================================
# KNOWLEDGE BASE (UPDATED)
# ================================
KNOWLEDGE_BASE = {
    "contamination": "foreign contamination",
    "deformation": "shape deformation",
    "missing_parts": "missing component",
    "scratch": "surface scratch",
    "no_defect": "normal surface"
}

# ================================
# SCHEMA
# ================================
SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string"},
        "num_defects": {"type": "number"},
        "defects": {"type": "array"}
    },
    "required": ["image_id", "num_defects", "defects"]
}

# ================================
# DECODE
# ================================
def secure_decode(file_path):
    with open(file_path, "rb") as f:
        encrypted = f.read()

    decrypted = cipher.decrypt(encrypted)
    decompressed = zlib.decompress(decrypted)
    data = json.loads(decompressed.decode())

    validate(instance=data, schema=SCHEMA)
    return data


# ================================
# INTERPRET
# ================================
def interpret(data):
    print("\n--- Decoded Message ---")
    print("Image:", data["image_id"])
    print("Defects:", data["num_defects"])

    if data["num_defects"] == 0:
        print("No defects")
        return

    for d in data["defects"]:
        cls = d.get("class", "unknown")
        conf = d.get("confidence", 0)
        area = d.get("area", 0)

        meaning = KNOWLEDGE_BASE.get(cls, "unknown")

        print(f"{cls} -> {meaning} | Conf: {conf:.3f} | Area: {area:.4f}")