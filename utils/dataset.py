import os
import random
import shutil
import numpy as np
from PIL import Image
import albumentations as A


ORIGINAL_SOURCE = r"D:\kaggle\tuberculosis-tb-chest-xray-dataset\TB_Chest_Radiography_Database"
REFORMED_SOURCE = r"D:\reformedsource"
DESTINATION = r"D:\DestinationFolder"

classes = ["Normal", "Tuberculosis"]
TARGET_INITIAL = 700
TARGET_AUGMENTED = 1500
SPLIT_RATIO = 0.8

for cls in classes:
    os.makedirs(os.path.join(REFORMED_SOURCE, cls), exist_ok=True)


for cls in classes:
    src_cls = os.path.join(ORIGINAL_SOURCE, cls)
    dst_cls = os.path.join(REFORMED_SOURCE, cls)

    images = os.listdir(src_cls)
    random.shuffle(images)

    if cls == "Normal":
        images = images[:TARGET_INITIAL]  

    for img in images:
        shutil.copy(os.path.join(src_cls, img), dst_cls)

augmenter = A.Compose([
    A.OneOf([
        A.Blur(p=1),
        A.HorizontalFlip(p=1),
        A.Rotate(limit=(-30, 30), p=1),
        A.VerticalFlip(p=1)
    ], p=1)
])

for cls in classes:
    cls_dir = os.path.join(REFORMED_SOURCE, cls)
    images = os.listdir(cls_dir)

    needed = TARGET_AUGMENTED - len(images)

    for i in range(needed):
        img_name = random.choice(images)
        img_path = os.path.join(cls_dir, img_name)

        img = np.array(Image.open(img_path))
        aug = augmenter(image=img)["image"]
        aug_pil = Image.fromarray(aug)

        name, ext = os.path.splitext(img_name)
        new_name = f"{name}_aug_{i}{ext}"

        aug_pil.save(os.path.join(cls_dir, new_name))

for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(DESTINATION, split, cls), exist_ok=True)

def cap_images(folder, max_count):
    images = os.listdir(folder)
    random.shuffle(images)
    for img in images[max_count:]:
        os.remove(os.path.join(folder, img))
cap_images(r"D:\reformedsource\Normal", 1500)
cap_images(r"D:\reformedsource\Tuberculosis", 1500)

for cls in classes:
    cls_src = os.path.join(REFORMED_SOURCE, cls)
    images = os.listdir(cls_src)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)

    train_imgs = images[:split_idx]      # 1200
    val_imgs = images[split_idx:]         # 300

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_src, img),
            os.path.join(DESTINATION, "train", cls)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_src, img),
            os.path.join(DESTINATION, "val", cls)
        )


print("\nFINAL DATASET COUNTS:\n")

print("Normal:", len(os.listdir(r"D:\reformedsource\Normal")))
print("TB:", len(os.listdir(r"D:\reformedsource\Tuberculosis")))

for split in ["train", "val"]:
    print(split.upper())
    for cls in classes:
        count = len(os.listdir(os.path.join(DESTINATION, split, cls)))
        print(f"  {cls}: {count}")
