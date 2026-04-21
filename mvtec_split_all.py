import os
import shutil
import random

SRC_ROOT = "MVTEC-AD"
DEST_ROOT = "MVTEC_SPLIT"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for category in os.listdir(SRC_ROOT):

    src_test = os.path.join(SRC_ROOT, category, "test")

    if not os.path.isdir(src_test):
        continue

    print(f"Processing {category}...")

    for split in ["train", "val", "test"]:
        for cls in os.listdir(src_test):
            os.makedirs(os.path.join(DEST_ROOT, category, split, cls), exist_ok=True)

    for cls in os.listdir(src_test):

        cls_path = os.path.join(src_test, cls)
        images = os.listdir(cls_path)

        random.shuffle(images)

        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(DEST_ROOT, category, "train", cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(DEST_ROOT, category, "val", cls, img))

        for img in test_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(DEST_ROOT, category, "test", cls, img))

print("All categories split complete!")