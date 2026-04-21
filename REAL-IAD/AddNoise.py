import random

# ================================
# SEMANTIC NOISE
# ================================
def add_semantic_noise(semantic_data, classes=None, drop_prob=0.2, flip_prob=0.1):
    noisy = semantic_data.copy()
    noisy_defects = []

    for defect in semantic_data["defects"]:

        # Drop defect (simulate missed detection)
        if classes and random.random() < drop_prob:
            continue

        new_defect = defect.copy()

        # Flip class (simulate misclassification)
        if random.random() < flip_prob:
            new_defect["class"] = random.choice(classes)

        noisy_defects.append(new_defect)

    noisy["defects"] = noisy_defects
    noisy["num_defects"] = len(noisy_defects)

    return noisy


# ================================
# CHANNEL NOISE
# ================================
def add_channel_noise(data, noise_level=0.02):
    noisy = bytearray(data)

    for i in range(len(noisy)):
        if random.random() < noise_level:
            noisy[i] ^= 0xFF  # flip bits

    return bytes(noisy)