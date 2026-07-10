"""
MangoMatters - Data Preprocessing
---------------------------------
Loads mango leaf images, resizes them with OpenCV, and builds the
training and validation datasets.

Expected folder layout:

    data/
        train/
            Anthracnose/
            Bacterial Canker/
            Cutting Weevil/
            Die Back/
            Gall Midge/
            Healthy/
            Powdery Mildew/
            Sooty Mould/
        val/
            <same class subfolders>
        test/
            <same class subfolders>
"""

import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def opencv_preprocess(image_path, target_size=IMG_SIZE):
    """Read an image with OpenCV, denoise it, and resize it."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 15)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_dataset_dir(src_dir, dst_dir, target_size=IMG_SIZE):
    """Preprocess every image in a folder and keep the class structure."""
    os.makedirs(dst_dir, exist_ok=True)
    classes = [c for c in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, c))]

    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        dst_cls_dir = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        for fname in os.listdir(src_cls_dir):
            src_path = os.path.join(src_cls_dir, fname)
            dst_path = os.path.join(dst_cls_dir, fname)
            try:
                img = opencv_preprocess(src_path, target_size)
                cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            except FileNotFoundError:
                print(f"Skipping unreadable file: {src_path}")

    print(f"Preprocessed {len(classes)} classes -> {dst_dir}")


def build_datasets(train_dir, val_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Build train and validation datasets with light augmentation."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="augmentation",
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    def prep_train(x, y):
        x = augmentation(x, training=True)
        x = normalization(x)
        return x, y

    def prep_eval(x, y):
        x = normalization(x)
        return x, y

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_eval, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


if __name__ == "__main__":
    pass
