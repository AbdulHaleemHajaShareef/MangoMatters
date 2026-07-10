"""
MangoMatters - Single Image Prediction
--------------------------------------
Usage:
    python src/predict.py --image path/to/leaf.jpg --model_path models/best_model.keras
"""

import argparse
import json

import numpy as np
import tensorflow as tf

from data_preprocessing import opencv_preprocess, IMG_SIZE

# Basic guidance for demo use only.
RECOMMENDATIONS = {
    "Healthy": "No action needed. Continue routine monitoring.",
    "Anthracnose": "Remove and destroy infected leaves/twigs; apply a copper-based fungicide.",
    "Bacterial Canker": "Prune infected branches with sterilized tools; avoid overhead irrigation.",
    "Cutting Weevil": "Remove affected shoots; use recommended insecticide during early infestation.",
    "Die Back": "Prune dead wood well below the infected area; disinfect pruning tools between cuts.",
    "Gall Midge": "Remove infested leaves/buds; apply insecticide at early flush stage.",
    "Powdery Mildew": "Apply sulfur-based fungicide; improve air circulation via pruning.",
    "Sooty Mould": "Control sap-sucking insects (aphids/scale) that cause the mould; wash foliage.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model_path", default="models/best_model.keras")
    p.add_argument("--class_names_path", default="models/class_names.json")
    return p.parse_args()


def predict(image_path, model_path, class_names_path):
    with open(class_names_path) as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path)

    img = opencv_preprocess(image_path, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    label = class_names[top_idx]
    confidence = float(preds[top_idx])

    return label, confidence, preds, class_names


if __name__ == "__main__":
    args = parse_args()
    label, confidence, preds, class_names = predict(
        args.image, args.model_path, args.class_names_path
    )

    print(f"\nPrediction: {label}  (confidence: {confidence:.1%})")
    print(f"Recommendation: {RECOMMENDATIONS.get(label, 'Consult a local agronomist.')}\n")

    print("Full class probabilities:")
    for name, prob in sorted(zip(class_names, preds), key=lambda x: -x[1]):
        print(f"  {name:<20s} {prob:.1%}")
