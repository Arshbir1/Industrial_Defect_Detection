import os
import random
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

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


class RealIADDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", split_ratio=(0.7, 0.15, 0.15)):
        self.samples = []
        self.transform = transform

        # ----------------------------
        # COLLECT SAMPLES PER CLASS
        # ----------------------------
        class_samples = defaultdict(list)

        for condition in ["OK", "NG"]:
            condition_path = os.path.join(root_dir, condition)

            if not os.path.isdir(condition_path):
                continue

            if condition == "OK":
                for sample in os.listdir(condition_path):
                    sample_path = os.path.join(condition_path, sample)

                    if os.path.isdir(sample_path):
                        class_samples["OK"].append(sample_path)

            else:
                for defect_code in os.listdir(condition_path):
                    defect_path = os.path.join(condition_path, defect_code)

                    for sample in os.listdir(defect_path):
                        sample_path = os.path.join(defect_path, sample)

                        if os.path.isdir(sample_path):
                            class_samples[defect_code].append(sample_path)

        # ----------------------------
        # STRATIFIED SPLIT (FIX)
        # ----------------------------
        selected = []

        for defect_code, sample_list in class_samples.items():
            random.shuffle(sample_list)
            n = len(sample_list)

            train_end = int(n * split_ratio[0])
            val_end = int(n * (split_ratio[0] + split_ratio[1]))

            if split == "train":
                chosen = sample_list[:train_end]
            elif split == "val":
                chosen = sample_list[train_end:val_end]
            else:
                chosen = sample_list[val_end:]

            for s in chosen:
                selected.append((s, defect_code))

        # ----------------------------
        # EXPAND TO IMAGE LEVEL
        # ----------------------------
        for sample_path, defect_code in selected:
            label = LABEL_MAP[defect_code]

            for file in os.listdir(sample_path):
                if file.endswith(".jpg"):
                    img_path = os.path.join(sample_path, file)
                    self.samples.append((img_path, label))

        # ----------------------------
        # CLASS INDEXING
        # ----------------------------
        self.classes = sorted(list(set([s[1] for s in self.samples])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print(f"[{split.upper()}] Classes:", self.classes)
        print(f"[{split.upper()}] Total samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[label]