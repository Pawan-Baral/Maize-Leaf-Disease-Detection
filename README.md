# Maize Leaf Disease Detection Prototype

A deep learning project that detects maize (corn) leaf diseases from images using Convolutional Neural Networks (CNN) with transfer learning.

Developed as a final year project at Pokhara University, School of Engineering.

---

## Diseases Detected

| Class | Description |
|-------|-------------|
| Blight (Northern Leaf Blight) | Fungal disease causing large, cigar-shaped lesions |
| Common Rust | Fungal disease with orange-brown pustules on leaves |
| Grey Leaf Spot | Fungal disease causing rectangular grey lesions |
| Healthy | No disease detected |

---

## Models Compared

Four CNN architectures were trained and compared:

| Model | Year | Size | Notes |
|-------|------|------|-------|
| MobileNetV2 | 2018 | ~14 MB | Lightweight, best for mobile deployment |
| EfficientNetB0 | 2019 | ~20 MB | Accurate, good balance of size and performance |
| ResNet50 | 2015 | ~100 MB | Classic deep model, strong baseline |
| ConvNeXtTiny | 2022 | ~110 MB | Modern CNN with Transformer-inspired design |

All models use two-phase transfer learning:
- **Phase 1** — Frozen base model, train top layers only (LR = 0.001)
- **Phase 2** — Fine-tune top layers of base model (LR = 0.00005)

---

## Dataset

- **Source:** Consolidated Corn Dataset (Kaggle — yasirahmad0810)
- **Total images:** ~6,516 cleaned images
- **Split:** 70% train / 15% validation / 15% test
- **Preprocessing:** Duplicate removal, corruption check, size filtering
- **Augmentation:** Random rotation, flip, zoom, shift

---

## Project Structure

```
maize_leaf_disease_detection_prototype/
│
├── train.py              # Main training script
├── predict.py            # Predict disease from a single image
├── compare_models.py     # Compare all trained models
├── requirements.txt      # Python dependencies
├── .gitignore
├── README.md
│
└── data/                 # Put your dataset here
    ├── Healthy/
    ├── Northern Leaf Blight/
    ├── Common Rust/
    └── Gray Leaf Spot/
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/maize_leaf_disease_detection_prototype.git
cd maize_leaf_disease_detection_prototype
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Download the dataset from Kaggle:
```
https://www.kaggle.com/datasets/yasirahmad0810/consolidated-corn-dataset
```

Place the class folders inside a `data/` directory:
```
data/
    Healthy/
    Northern Leaf Blight/
    Common Rust/
    Gray Leaf Spot/
```

---

## Training

Train a model by running:

```bash
python train.py
```

To change the model, open `train.py` and change the `MODEL_CHOICE` variable:

```python
MODEL_CHOICE = "MobileNetV2"    # or EfficientNetB0 | ResNet50 | ConvNeXtTiny
```

Training will:
1. Clean and split the dataset automatically (70/15/15)
2. Train Phase 1 (frozen base) for 10 epochs
3. Train Phase 2 (fine-tuning) for up to 20 epochs
4. Save the model, metrics, and plots to `outputs/`

---

## Prediction

Predict the disease of a single leaf image:

```bash
python predict.py --image path/to/leaf.jpg
python predict.py --image path/to/leaf.jpg --model EfficientNetB0
```

---

## Compare Models

After training all 4 models, generate a comparison chart:

```bash
python compare_models.py
```

This produces a bar chart comparing Accuracy, Precision, Recall and F1 Score across all models.

---

## Running on Google Colab / Kaggle

This project was developed and trained on:
- **Google Colab** (T4 GPU) — MobileNetV2
- **Kaggle Notebooks** (T4 GPU x2) — EfficientNetB0, ResNet50, ConvNeXtTiny

To run on Colab or Kaggle, upload this repository and change the paths:

```python
DATASET_PATH = "/content/drive/MyDrive/maize_leaf_disease_detection_prototype/data"
OUTPUT       = "/content/drive/MyDrive/maize_leaf_disease_detection_prototype/outputs"
```

---

## Outputs

After training each model, the following files are saved to `outputs/`:

| File | Description |
|------|-------------|
| `{Model}_final.keras` | Trained model file |
| `{Model}_confusion_matrix.png` | Confusion matrix on test set |
| `{Model}_per_class_accuracy.png` | Per-class accuracy bar chart |
| `{Model}_roc_curves.png` | ROC curves with AUC scores |
| `{Model}_training_history.png` | Accuracy and loss curves |
| `{Model}_classification_report.txt` | Precision, recall, F1 per class |
| `{Model}_results.json` | Summary metrics for comparison |
| `model_comparison.png` | All models compared in one chart |
| `class_names.json` | Class label mapping |

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| MobileNetV2 | TBD | TBD | TBD | TBD |
| EfficientNetB0 | TBD | TBD | TBD | TBD |
| ResNet50 | TBD | TBD | TBD | TBD |
| ConvNeXtTiny | TBD | TBD | TBD | TBD |

*Update this table with your actual results after training.*

---

## Future Work

- GAN-based data augmentation for minority classes
- Collection and integration of local Nepali maize leaf images
- Fall Armyworm disease detection as a separate model
- Mobile application deployment using Flutter + Flask REST API
- Expand to 7 disease classes

---

## Authors

- **Pawan** — Pokhara University, School of Engineering
- Supervised by: Udaya Raj Dhungana

---

## License

This project is for academic purposes only.
