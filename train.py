"""
MangoMatters - Training Script
------------------------------
Usage:
    python src/train.py --train_dir data/train --val_dir data/val --epochs 25
"""

import argparse
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from data_preprocessing import build_datasets, IMG_SIZE, BATCH_SIZE
from model import build_cnn, build_transfer_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="data/train")
    p.add_argument("--val_dir", default="data/val")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--model_type", choices=["cnn", "transfer"], default="cnn")
    p.add_argument("--output_dir", default="models")
    return p.parse_args()


def plot_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    print(f"Saved training curves -> {output_dir}/training_curves.png")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(
        args.train_dir, args.val_dir, img_size=IMG_SIZE, batch_size=args.batch_size
    )
    print(f"Classes found: {class_names}")

    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    if args.model_type == "cnn":
        model = build_cnn(input_shape=(*IMG_SIZE, 3), num_classes=len(class_names))
    else:
        model = build_transfer_model(input_shape=(*IMG_SIZE, 3), num_classes=len(class_names))

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, args.output_dir)

    final_path = os.path.join(args.output_dir, "mango_disease_model.keras")
    model.save(final_path)
    print(f"Saved final model -> {final_path}")

    best_val_acc = max(history.history["val_accuracy"])
    print(f"Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
