# MangoMatters

MangoMatters is a simple mango leaf disease classifier built as a student
project. It uses TensorFlow for the model and OpenCV for image preprocessing.

## What it does

- Prepares mango leaf images for training
- Trains a CNN model on the dataset
- Evaluates the trained model on a test split
- Predicts the disease category for a single image

## Dataset

This project uses the public **MangoLeafBD** dataset on Kaggle, which has
about 4,000 images and 8 classes:

https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset

```bash
# Requires a Kaggle account + API token (~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d aryashah2k/mango-leaf-disease-dataset
unzip mango-leaf-disease-dataset.zip -d data/raw
```

After that, split the images into `data/train`, `data/val`, and `data/test`.
An 80/10/10 split is a good starting point, and the class folder names should
stay the same in each split.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --train_dir data/train --val_dir data/val --epochs 25
```

This saves the trained model in `models/` and also stores the class names
used during training.

## Evaluate

```bash
python src/evaluate.py --test_dir data/test --model_path models/best_model.keras
```

This prints a classification report and saves a confusion matrix image.

## Predict

```bash
python src/predict.py --image data/test/Anthracnose/some_leaf.jpg
```

## Project structure

```
MangoMatters/
├── data/                  # not tracked in git
│   ├── train/
│   ├── val/
│   └── test/
├── models/                # trained weights and class names
├── outputs/               # evaluation outputs
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Results

| Metric              | Value |
|---------------------|-------|
| Validation accuracy | —     |
| Test accuracy       | —     |
| # classes           | 8     |
| # training images   | —     |

Update this table after you train the model so the repository shows real
results.
