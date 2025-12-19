import os
import random
import shutil
import numpy as np
from PIL import Image
import albumentations as A

# ---------------- PATHS ----------------
SOURCE = r"D:\kaggle\tuberculosis-tb-chest-xray-dataset\TB_Chest_Radiography_Database"
DESTINATION = r"D:\DestinationFolder"

classes = ["Normal", "Tuberculosis"]
split_ratio = 0.8

# ---------------- CREATE TRAIN / VAL FOLDERS ----------------
os.makedirs(os.path.join(DESTINATION, "train"), exist_ok=True)
os.makedirs(os.path.join(DESTINATION, "val"), exist_ok=True)

for cls in classes:
    os.makedirs(os.path.join(DESTINATION, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(DESTINATION, "val", cls), exist_ok=True)

# ---------------- SPLIT DATASET ----------------
for cls in classes:
    src_folder = os.path.join(SOURCE, cls)
    images = os.listdir(src_folder)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy TRAIN images
    for img in train_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(DESTINATION, "train", cls)
        )

    # Copy VAL images
    for img in val_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(DESTINATION, "val", cls)
        )

# ---------------- AUGMENT TRAIN ONLY (BALANCE CLASSES) ----------------
augment_transformer = A.Compose([
    A.OneOf([
        A.Blur(p=1),
        A.HorizontalFlip(p=1),
        A.Rotate(limit=(-30, 30), p=1),
        A.VerticalFlip(p=1)
    ], p=1)
])

# Count training images per class
train_counts = {
    cls: len(os.listdir(os.path.join(DESTINATION, "train", cls)))
    for cls in classes
}

max_count = max(train_counts.values())

# Balance training set
for cls in classes:
    train_dir = os.path.join(DESTINATION, "train", cls)
    images = os.listdir(train_dir)

    needed = max_count - len(images)
    if needed <= 0:
        continue  # majority class

    print(f"Augmenting {cls}: adding {needed} images")

    for i in range(needed):
        img_name = random.choice(images)
        img_path = os.path.join(train_dir, img_name)

        img = np.array(Image.open(img_path))
        augmented = augment_transformer(image=img)
        aug_img = augmented["image"]

        aug_pil = Image.fromarray(aug_img)

        name, ext = os.path.splitext(img_name)
        new_name = f"{name}_bal_{i}{ext}"

        # Save augmented image back into SAME train folder
        aug_pil.save(os.path.join(train_dir, new_name))

# ---------------- FINAL COUNTS (SANITY CHECK) ----------------
print("\nFinal training image counts:")
for cls in classes:
    count = len(os.listdir(os.path.join(DESTINATION, "train", cls)))
    print(f"{cls}: {count}")

print("\nValidation image counts (unchanged):")
for cls in classes:
    count = len(os.listdir(os.path.join(DESTINATION, "val", cls)))
    print(f"{cls}: {count}")

