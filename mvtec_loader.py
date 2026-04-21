import os
from SemanticEncoder import semantic_encoder, secure_encode, save_encoded_file

# ================================
# NORMALIZATION FUNCTION
# ================================
def normalize_mvtec(image_path):
    # Get parent folder name directly
    label = os.path.basename(os.path.dirname(image_path))

    if label == "good":
        return {
            "class": "no_defect",
            "confidence": 1.0
        }

    return {
        "class": label,
        "confidence": 1.0
    }


# ================================
# LOAD + ENCODE DATASET
# ================================
def process_mvtec_dataset(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    samples = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)

                prediction = normalize_mvtec(full_path)
                # Debug Statement
                print("Label:", prediction["class"])

                # if prediction is None:
                #     continue

                image_id = os.path.splitext(file)[0]

                # Step 1: Create semantic structure
                semantic_data = semantic_encoder(image_id, prediction)

                # Step 2: Encode (compress + encrypt)
                encoded_data = secure_encode(semantic_data)

                # Step 3: Save file
                output_path = os.path.join(output_dir, image_id + ".bin")
                save_encoded_file(encoded_data, output_path)

                samples.append(semantic_data)

    return samples


# ================================
# TEST
# ================================
if __name__ == "__main__":
    dataset_path = "MVTEC-AD/capsule/test"
    output_dir = "secure_output_mvtec"

    data = process_mvtec_dataset(dataset_path, output_dir)

    print("\nSample semantic outputs:\n")
    for i in range(min(5, len(data))):
        print(data[i])

    print(f"\nTotal processed samples: {len(data)}")
    print(f"Encoded files saved in: {output_dir}")