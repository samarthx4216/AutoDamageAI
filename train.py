"""
train.py — Train the Car Damage Detection CNN
=============================================
Run:
    python train.py --data_dir ./data1a --epochs 30 --output model/car_damage_model.h5

Dataset expected structure:
    data1a/
      training/
        00-damage/   ← damaged car images
        01-whole/    ← undamaged car images
      validation/
        00-damage/
        01-whole/
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {"00-damage": 0, "01-whole": 1}
IMG_SIZE  = (128, 128)


def load_images(data_path: str):
    """Load and resize all images from a directory of class sub-folders."""
    data, labels = [], []
    for category in sorted(os.listdir(data_path)):
        cat_path = os.path.join(data_path, category)
        if not os.path.isdir(cat_path) or category not in LABEL_MAP:
            continue
        label = LABEL_MAP[category]
        imgs  = os.listdir(cat_path)
        print(f"  [{category}] {len(imgs)} images  →  label {label}")
        for fname in imgs:
            fpath = os.path.join(cat_path, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)


def build_model(num_classes: int = 2) -> keras.Model:
    """Build the CNN architecture."""
    model = keras.Sequential([
        keras.layers.Conv2D(32,  (3, 3), activation="relu", input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64,  (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),

        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64,  activation="relu"),
        keras.layers.Dense(32,  activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ], name="CarDamageCNN")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, save_path="model/training_history.png"):
    """Save loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"],     label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["accuracy"],     label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Training curves saved → {save_path}")


def main(args):
    print("\n=== Car Damage Detection — Training ===\n")

    train_path = os.path.join(args.data_dir, "training")
    print(f"Loading training images from: {train_path}")
    X, Y = load_images(train_path)
    print(f"\nTotal images loaded: {len(X)}")

    # Train / test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # Normalise
    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    print(f"\nSplit  →  train: {len(X_train)}  |  test: {len(X_test)}")

    # Build & summarise
    model = build_model()
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    print(f"\nTraining for up to {args.epochs} epochs…\n")
    history = model.fit(
        X_train, Y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nTest Accuracy : {acc * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"\nModel saved → {args.output}")

    # Plot
    plot_history(history)
    print("\nDone! ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Car Damage CNN")
    parser.add_argument("--data_dir",   default="./data1a",                  help="Root data directory")
    parser.add_argument("--epochs",     default=30, type=int,                help="Max training epochs")
    parser.add_argument("--batch_size", default=32, type=int,                help="Batch size")
    parser.add_argument("--output",     default="model/car_damage_model.h5", help="Output model path")
    main(parser.parse_args())
