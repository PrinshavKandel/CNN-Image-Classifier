import sys
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import os

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ResNet50.keras")
MODEL_PATH2 = os.path.join(os.path.dirname(__file__), "efficientnetb0.keras")
MODEL_PATH3 = os.path.join(os.path.dirname(__file__), "mobilenetv2.keras")

print("Loading model from:", MODEL_PATH)
print("Loading model from:", MODEL_PATH2)
print("Loading model from:", MODEL_PATH3)

IMG_SIZE = 224
THRESHOLD = 0.5  # probability threshold for TB
img_pth = "TB.1.png"

# Preprocess image
def load_and_prepare_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Run prediction for a single model
def predict_tb(model, model_name, img):
    prob_tb = model.predict(img, verbose=0)[0][0]

    if prob_tb >= THRESHOLD:
        label = "TUBERCULOSIS"
        confidence = prob_tb
    else:
        label = "NORMAL (No TB)"
        confidence = 1.0 - prob_tb

    print(f"\n=== Prediction using {model_name} ===")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.4f}")
    print(f"TB Score   : {prob_tb:.4f}")

def main(img_path):
    # Load all models
    print("Loading TB detection models...")
    model1 = tf.keras.models.load_model(MODEL_PATH)
    model2 = tf.keras.models.load_model(MODEL_PATH2)
    model3 = tf.keras.models.load_model(MODEL_PATH3)

    # Preprocess the image
    print(f"Loading image: {img_path}")
    img = load_and_prepare_image(img_path)

    # Run predictions
    predict_tb(model1, "ResNet50", img)
    predict_tb(model2, "EfficientNetB0", img)
    predict_tb(model3, "MobileNetV2", img)

if __name__ == "__main__":
    # Use command-line argument if provided, else fallback to default
    img_path = sys.argv[1] if len(sys.argv) == 2 else img_pth
    main(img_path)
