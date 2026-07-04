# ============================================================
#  MAIZE LEAF DISEASE DETECTION PROTOTYPE
#  train.py — Complete Training Pipeline
#  Trains one of 4 CNN models on the consolidated corn dataset
#  Models: MobileNetV2 | EfficientNetB0 | ResNet50 | ConvNeXtTiny
# ============================================================

import os, shutil, hashlib, json, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    MobileNetV2, EfficientNetB0, ResNet50, ConvNeXtTiny
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from PIL import Image

# ============================================================
#  CONFIGURATION — change these as needed
# ============================================================

MODEL_CHOICE = "MobileNetV2"   # MobileNetV2 | EfficientNetB0 | ResNet50 | ConvNeXtTiny
DATASET_PATH = "data"          # Root folder containing class subfolders
OUTPUT       = "outputs"       # Where models and plots are saved
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_P1    = 10              # Phase 1 epochs (frozen base)
EPOCHS_P2    = 20              # Phase 2 epochs (fine-tuning)
SEED         = 42

# Train / Validation / Test split ratios
TRAIN_RATIO  = 0.70
VALID_RATIO  = 0.15
TEST_RATIO   = 0.15

# Standard class names — map your folder names here
NAME_MAP = {
    "Healthy"              : "Healthy",
    "Gray Leaf Spot"       : "Grey_Leaf_Spot",
    "Northern Leaf Blight" : "Blight",
    "Common Rust"          : "Common_Rust",
    # Add more mappings if your folder names differ
    "Blight"               : "Blight",
    "Common_Rust"          : "Common_Rust",
    "Grey_Leaf_Spot"       : "Grey_Leaf_Spot",
    "Gray_Leaf_Spot"       : "Grey_Leaf_Spot",
}

CLASSES = list(set(NAME_MAP.values()))
os.makedirs(OUTPUT, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TF version  : {tf.__version__}")
print(f"GPU         : {tf.config.list_physical_devices('GPU')}")
print(f"Model       : {MODEL_CHOICE}")
print(f"Dataset     : {DATASET_PATH}")


# ============================================================
#  STEP 1 — CLEAN AND SPLIT DATASET
# ============================================================

def clean_and_split(dataset_path, name_map, classes,
                    split_path, ratios, seed):
    """
    Reads images from dataset_path, removes corrupted/duplicate/small
    images, then splits into train/valid/test folders.
    """
    splits   = ['train', 'valid', 'test']
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(split_path, split, cls),
                        exist_ok=True)

    seen, removed = set(), 0
    totals = {s: 0 for s in splits}

    print("\n📂 Cleaning and splitting dataset...\n")

    for folder in os.listdir(dataset_path):
        if folder not in name_map:
            continue
        standard = name_map[folder]
        src      = os.path.join(dataset_path, folder)
        files    = [f for f in os.listdir(src)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]

        # Clean
        clean = []
        for f in files:
            path = os.path.join(src, f)
            try:
                img = Image.open(path); img.verify()
                img = Image.open(path).convert("RGB")
            except:
                removed += 1; continue
            if img.size[0] < 64 or img.size[1] < 64:
                removed += 1; continue
            with open(path,'rb') as fh:
                h = hashlib.md5(fh.read()).hexdigest()
            if h in seen:
                removed += 1; continue
            seen.add(h)
            clean.append(f)

        # Shuffle and split
        random.shuffle(clean)
        n       = len(clean)
        n_train = int(n * ratios[0])
        n_valid = int(n * ratios[1])
        split_files = {
            'train': clean[:n_train],
            'valid': clean[n_train:n_train + n_valid],
            'test' : clean[n_train + n_valid:]
        }

        for split, sfiles in split_files.items():
            dst = os.path.join(split_path, split, standard)
            for i, f in enumerate(sfiles):
                shutil.copy2(os.path.join(src, f),
                             os.path.join(dst, f"{standard[:4]}_{i}_{f}"))
            totals[split] += len(sfiles)

        print(f"   {standard:<20} "
              f"train={len(split_files['train'])}  "
              f"valid={len(split_files['valid'])}  "
              f"test={len(split_files['test'])}")

    print(f"\n   Removed (corrupt/duplicate): {removed}")
    print(f"\n📊 Final split:")
    print(f"   Train : {totals['train']} images  ({int(ratios[0]*100)}%)")
    print(f"   Valid : {totals['valid']} images  ({int(ratios[1]*100)}%)")
    print(f"   Test  : {totals['test']} images   ({int(ratios[2]*100)}%)")
    print("✅ Split complete\n")


SPLIT_PATH = os.path.join(OUTPUT, "split_data")

if not os.path.exists(SPLIT_PATH):
    clean_and_split(
        DATASET_PATH, NAME_MAP, CLASSES,
        SPLIT_PATH, [TRAIN_RATIO, VALID_RATIO, TEST_RATIO], SEED
    )
else:
    print("✅ Split data already exists — skipping\n")


# ============================================================
#  STEP 2 — LOAD IMAGES
# ============================================================

train_datagen = ImageDataGenerator(
    rotation_range     = 20,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    horizontal_flip    = True,
    zoom_range         = 0.2
)
plain_datagen = ImageDataGenerator()

train_ds = train_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, 'train'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='sparse', shuffle=True
)
val_ds = plain_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, 'valid'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='sparse', shuffle=False
)
test_ds = plain_datagen.flow_from_directory(
    os.path.join(SPLIT_PATH, 'test'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='sparse', shuffle=False
)

class_names = list(train_ds.class_indices.keys())
num_classes = len(class_names)

print(f"✅ Classes : {class_names}")
print(f"   Train   : {train_ds.samples}")
print(f"   Valid   : {val_ds.samples}")
print(f"   Test    : {test_ds.samples}\n")

with open(os.path.join(OUTPUT, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)


# ============================================================
#  STEP 3 — BUILD MODEL
# ============================================================

def build_model(choice, num_classes, img_size):
    """
    Builds a transfer learning model with the chosen base architecture.
    Each model uses its own preprocessing layer.
    """
    inputs = tf.keras.Input(shape=(*img_size, 3))

    if choice == "MobileNetV2":
        x          = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        base_model = MobileNetV2(weights='imagenet',
                        include_top=False, input_shape=(*img_size, 3))
    elif choice == "EfficientNetB0":
        x          = tf.keras.applications.efficientnet.preprocess_input(inputs)
        base_model = EfficientNetB0(weights='imagenet',
                        include_top=False, input_shape=(*img_size, 3))
    elif choice == "ResNet50":
        x          = tf.keras.applications.resnet50.preprocess_input(inputs)
        base_model = ResNet50(weights='imagenet',
                        include_top=False, input_shape=(*img_size, 3))
    elif choice == "ConvNeXtTiny":
        x          = tf.keras.applications.convnext.preprocess_input(inputs)
        base_model = ConvNeXtTiny(weights='imagenet',
                        include_top=False, input_shape=(*img_size, 3))
    else:
        raise ValueError(f"Unknown model: {choice}")

    base_model.trainable = False

    x       = base_model(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs), base_model


model, base_model = build_model(MODEL_CHOICE, num_classes, IMG_SIZE)
model.compile(
    optimizer = tf.keras.optimizers.Adam(0.001),
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)
print(f"✅ {MODEL_CHOICE} built — {model.count_params():,} params\n")


# ============================================================
#  STEP 4 — CLASS WEIGHTS
# ============================================================

cw_array = compute_class_weight('balanced',
    classes=np.unique(train_ds.classes),
    y=train_ds.classes)
cw = dict(enumerate(cw_array))

print("⚖️  Class weights:")
for i, name in enumerate(class_names):
    print(f"   {name:<20} {cw[i]:.3f}")
print()


# ============================================================
#  STEP 5 — CALLBACKS
# ============================================================

CKPT_DIR = os.path.join(OUTPUT, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

def get_callbacks(prefix):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = os.path.join(CKPT_DIR, f"{prefix}_best.keras"),
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-7,
            verbose  = 1
        )
    ]


# ============================================================
#  STEP 6 — PHASE 1: FROZEN BASE
# ============================================================

print(f"🌽 Phase 1 — {MODEL_CHOICE} frozen base...\n")

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = EPOCHS_P1,
    callbacks       = get_callbacks(f"{MODEL_CHOICE}_p1"),
    class_weight    = cw
)

with open(os.path.join(OUTPUT, f'{MODEL_CHOICE}_history_p1.json'),'w') as f:
    json.dump(history.history, f)

best_p1 = max(history.history['val_accuracy'])
print(f"\n✅ Phase 1 done — best val acc: {best_p1*100:.2f}%\n")


# ============================================================
#  STEP 7 — PHASE 2: FINE-TUNING
# ============================================================

FREEZE_AT = {
    "MobileNetV2"    : 100,
    "EfficientNetB0" : 150,
    "ResNet50"       : 140,
    "ConvNeXtTiny"   : 100,
}

base_model.trainable = True
for layer in base_model.layers[:FREEZE_AT[MODEL_CHOICE]]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.00005),
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)

total_epochs = EPOCHS_P1 + EPOCHS_P2
print(f"🌽 Phase 2 — {MODEL_CHOICE} fine-tuning...\n")

history_fine = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = total_epochs,
    initial_epoch   = history.epoch[-1],
    callbacks       = get_callbacks(f"{MODEL_CHOICE}_p2"),
    class_weight    = cw
)

with open(os.path.join(OUTPUT, f'{MODEL_CHOICE}_history_p2.json'),'w') as f:
    json.dump(history_fine.history, f)

model.save(os.path.join(OUTPUT, f'{MODEL_CHOICE}_final.keras'))

best_p2 = max(history_fine.history['val_accuracy'])
print(f"\n✅ {MODEL_CHOICE} training complete!")
print(f"   Phase 1 best val acc : {best_p1*100:.2f}%")
print(f"   Phase 2 best val acc : {best_p2*100:.2f}%\n")


# ============================================================
#  STEP 8 — EVALUATE ON TEST SET
# ============================================================

colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0']

print(f"🔍 Evaluating on test set...")
test_ds.reset()
preds        = model.predict(test_ds, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_ds.classes

# 1. Classification report
print(f"\n{'='*60}")
print(f"{MODEL_CHOICE} — TEST SET CLASSIFICATION REPORT")
print(f"{'='*60}")
report = classification_report(true_classes, pred_classes,
                                target_names=class_names)
print(report)
with open(os.path.join(OUTPUT, f'{MODEL_CHOICE}_classification_report.txt'), 'w') as f:
    f.write(f"{MODEL_CHOICE} — TEST SET CLASSIFICATION REPORT\n\n")
    f.write(report)

# 2. Confusion matrix
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names, linewidths=0.5)
plt.title(f'{MODEL_CHOICE} — Confusion Matrix (Test Set)',
          fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, f'{MODEL_CHOICE}_confusion_matrix.png'), dpi=150)
plt.show()

# 3. Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(9, 5))
bars = plt.bar(class_names, per_class_acc*100, color=colors)
plt.title(f'{MODEL_CHOICE} — Per-Class Accuracy (Test Set)',
          fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 115])
plt.xticks(rotation=20, ha='right')
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1.5,
             f'{acc*100:.1f}%', ha='center',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, f'{MODEL_CHOICE}_per_class_accuracy.png'), dpi=150)
plt.show()

# 4. ROC curves
true_bin = label_binarize(true_classes, classes=list(range(num_classes)))
plt.figure(figsize=(9, 6))
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(true_bin[:,i], preds[:,i])
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{name}  AUC={auc(fpr,tpr):.2f}')
plt.plot([0,1],[0,1],'k--')
plt.title(f'{MODEL_CHOICE} — ROC Curves (Test Set)',
          fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, f'{MODEL_CHOICE}_roc_curves.png'), dpi=150)
plt.show()

# 5. Training history
with open(os.path.join(OUTPUT, f'{MODEL_CHOICE}_history_p1.json')) as f:
    h1 = json.load(f)
with open(os.path.join(OUTPUT, f'{MODEL_CHOICE}_history_p2.json')) as f:
    h2 = json.load(f)

acc      = h1['accuracy']     + h2['accuracy']
val_acc  = h1['val_accuracy'] + h2['val_accuracy']
loss     = h1['loss']         + h2['loss']
val_loss = h1['val_loss']     + h2['val_loss']
p1_end   = len(h1['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'{MODEL_CHOICE} — Training History (Phase 1 + Phase 2)',
             fontsize=14, fontweight='bold')
for ax, train, val, title in zip(
    axes, [acc, loss], [val_acc, val_loss], ['Accuracy', 'Loss']
):
    ax.plot(train, label='Train',      marker='o', markersize=3)
    ax.plot(val,   label='Validation', marker='o', markersize=3)
    ax.axvline(x=p1_end, color='gray', linestyle='--',
               linewidth=1, label='Phase 2 starts')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, f'{MODEL_CHOICE}_training_history.png'), dpi=150)
plt.show()

# 6. Final metrics summary
p = precision_score(true_classes, pred_classes, average=None)
r = recall_score(true_classes,    pred_classes, average=None)
f = f1_score(true_classes,        pred_classes, average=None)
overall = np.mean(pred_classes == true_classes)

print(f"\n{'='*65}")
print(f"{MODEL_CHOICE} — FINAL TEST METRICS")
print(f"{'='*65}")
print(f"{'Class':<22}{'Precision':>10}{'Recall':>8}"
      f"{'F1':>8}{'Accuracy':>10}")
print("-"*65)
for i, name in enumerate(class_names):
    print(f"{name:<22}{p[i]:>10.3f}{r[i]:>8.3f}"
          f"{f[i]:>8.3f}{per_class_acc[i]:>9.1%}")
print("-"*65)
print(f"{'Overall (macro)':<22}"
      f"{precision_score(true_classes,pred_classes,average='macro'):>10.3f}"
      f"{recall_score(true_classes,pred_classes,average='macro'):>8.3f}"
      f"{f1_score(true_classes,pred_classes,average='macro'):>8.3f}"
      f"{overall:>9.1%}")
print("="*65)

# Save results JSON for comparison
results = {
    'model'    : MODEL_CHOICE,
    'accuracy' : float(overall),
    'precision': float(precision_score(true_classes,pred_classes,average='macro')),
    'recall'   : float(recall_score(true_classes,pred_classes,average='macro')),
    'f1'       : float(f1_score(true_classes,pred_classes,average='macro')),
}
with open(os.path.join(OUTPUT, f'{MODEL_CHOICE}_results.json'),'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All outputs saved to: {OUTPUT}/")
print(f"   Run again with different MODEL_CHOICE to train next model.")
