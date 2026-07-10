"""
MangoMatters - Evaluation Script
--------------------------------
Usage:
    python src/evaluate.py --test_dir data/test --model_path models/best_model.keras
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import IMG_SIZE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="data/test")
    p.add_argument("--model_path", default="models/best_model.keras")
    p.add_argument("--class_names_path", default="models/class_names.json")
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.class_names_path) as f:
        class_names = json.load(f)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.test_dir,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode="categorical",
        shuffle=False,
    )
    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    test_ds_norm = test_ds.map(lambda x, y: (normalization(x), y))

    model = tf.keras.models.load_model(args.model_path)

    y_true, y_pred = [], []
    for images, labels in test_ds_norm:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Mango Leaf Disease Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    print(f"Saved confusion matrix -> {args.output_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
