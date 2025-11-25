import os
import shutil
import random
SOURCE=r"D:\kaggle\tuberculosis-tb-chest-xray-dataset\TB_Chest_Radiography_Database"
DESTINATION=r"D:\github\utils"
classes=['Normal',"Tuberculosis"]
split_ratio=0.8
os.makedirs(f"{DESTINATION}/train",exist_ok=True)
os.makedirs(f"{DESTINATION}/val",exist_ok=True)
for x in classes:
    os.makedirs(f"{DESTINATION}/train/{x}",exist_ok=True)
    os.makedirs(f"{DESTINATION}/val/{x}",exist_ok=True)
    src_folder=os.path.join(SOURCE,x)
    images=os.listdir(src_folder)
    random.shuffle(images)
    split_index=int(len(images)*split_ratio)
    train_images=images[:split_index]
    val_images=images[split_index:]
for img in train_images:
        shutil.copy(os.path.join(src_folder, img), f"{DESTINATION}/train/{x}")
for img in val_images:
        shutil.copy(os.path.join(src_folder, img), f"{DESTINATION}/val/{x}")

#D:\kaggle\tuberculosis-tb-chest-xray-dataset\TB_Chest_Radiography_Database\Normal
