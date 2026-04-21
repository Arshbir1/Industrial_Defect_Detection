import os

image_dir = "NEU-DET/train/images"
semantic_dir = "semantic_output"

def get_image_size(image_name):
    for root, _, files in os.walk(image_dir):
        if image_name + ".jpg" in files:
            return os.path.getsize(os.path.join(root, image_name + ".jpg"))
    return 0

total_image_size = 0
total_semantic_size = 0

for file in os.listdir(semantic_dir):
    if file.endswith(".json"):
        image_name = os.path.splitext(file)[0]

        img_size = get_image_size(image_name)
        json_size = os.path.getsize(os.path.join(semantic_dir, file))

        total_image_size += img_size
        total_semantic_size += json_size

print("Total image size:", total_image_size / (1024*1024), "MB")
print("Total semantic size:", total_semantic_size / (1024*1024), "MB")
print("Compression ratio:", total_semantic_size / total_image_size)