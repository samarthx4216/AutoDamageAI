"""
predict.py — Run inference on a single image from the command line
==================================================================
Usage:
    python predict.py --image path/to/car.jpg --model model/car_damage_model.h5
"""

import argparse
import numpy as np
import cv2
import tensorflow as tf

LABEL_MAP = {0: "DAMAGED ⚠️", 1: "UNDAMAGED ✅"}


def predict_image(model_path: str, image_path: str):
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load & preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_resized = cv2.resize(img, (128, 128))
    img_scaled  = img_resized / 255.0
    img_input   = np.reshape(img_scaled, [1, 128, 128, 3])

    # Predict
    probs = model.predict(img_input, verbose=0)[0]
    label = int(np.argmax(probs))

    print("\n" + "=" * 40)
    print(f"  Image     : {image_path}")
    print(f"  Prediction: {LABEL_MAP[label]}")
    print(f"  Damaged   : {probs[0] * 100:.2f}%")
    print(f"  Undamaged : {probs[1] * 100:.2f}%")
    print("=" * 40 + "\n")

    return label, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Damage Prediction CLI")
    parser.add_argument("--image", required=True,                        help="Path to car image")
    parser.add_argument("--model", default="model/car_damage_model.h5", help="Path to .h5 model")
    args = parser.parse_args()
    predict_image(args.model, args.image)
