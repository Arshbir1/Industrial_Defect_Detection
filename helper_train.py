import os
import subprocess

ROOT = "MVTEC-AD"

# Get all categories
categories = [c for c in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, c))]

print("Categories found:", categories)

for category in categories:

    print(f"\n===== TRAINING: {category} =====")

    # Call training script with category
    subprocess.run([
        "python",
        "mvtec_train_full.py",
        category
    ])