# ================================================================
#  MAIZE LEAF DISEASE DETECTION PROTOTYPE
#  Google Colab Version
#
#  HOW TO USE THIS:
#  Copy each CELL block into a separate cell in Colab.
#  Run them one by one, top to bottom.
# ================================================================


# ================================================================
# CELL 1 — Mount Google Drive
# ================================================================
# Your dataset must be in your Google Drive.
# Put it like this in your Drive:
#
#   My Drive/
#       maize_leaf_disease_detection_prototype/
#           data/
#               Blight/
#               Common_Rust/
#               Grey_Leaf_Spot/
#               Healthy/
#
# Then run this cell — it will ask you to sign in and allow access.

from google.colab import drive
drive.mount('/content/drive')

# ================================================================
# CELL 2 — Check your dataset is found
# ================================================================
import os

DATASET_PATH = "/content/drive/MyDrive/maize_leaf_disease_detection_prototype/data"

# List what's inside your data folder
print("📂 Folders found in your dataset:")
for folder in sorted(os.listdir(DATASET_PATH)):
    full_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(full_path):
        count = len(os.listdir(full_path))
        print(f"   {folder:25s} → {count} images")

# ================================================================
# CELL 3 — Imports and Configuration
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Settings — you can change these
IMG_SIZE   = (224, 224)   # All images resized to this
BATCH_SIZE = 32           # Images processed at once (lower to 16 if it crashes)
EPOCHS     = 20           # Max training rounds

# ================================================================
# CELL 4 — Load and Prepare Images
# ================================================================
# ImageDataGenerator:
#   - Loads images from disk in batches (memory efficient)
#   - Normalizes pixel values from 0-255 to 0-1
#   - Augments images (flips, rotates) so model learns better

datagen = ImageDataGenerator(
    rescale            = 1.0 / 255,   # Normalize pixels 0-1
    rotation_range     = 20,
    width_shift_range  = 0.2,
    height_shift_range = 0.2,
    horizontal_flip    = True,
    zoom_range         = 0.2,
    validation_split   = 0.2          # 20% of images used for validation
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'training',
    shuffle     = True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'validation',
    shuffle     = False
)

class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)

print(f"\n✅ Classes detected ({num_classes} total): {class_names}")
print(f"   Training images  : {train_data.samples}")
print(f"   Validation images: {val_data.samples}")

# ================================================================
# CELL 5 — Build the Model (Transfer Learning)
# ================================================================
# We use MobileNetV2 — already trained on 1 million images.
# We reuse its ability to detect shapes, textures, edges.
# Then we add our own layer to classify maize diseases.

base_model = MobileNetV2(
    weights     = 'imagenet',    # Pre-trained weights
    include_top = False,         # Remove original classifier
    input_shape = (*IMG_SIZE, 3) # 3 = RGB
)
base_model.trainable = False     # Freeze — don't change its weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),          # Flatten to 1D
    layers.Dense(128, activation='relu'),     # Our hidden layer (128 neurons)
    layers.Dropout(0.3),                      # Prevent memorization
    layers.Dense(num_classes, activation='softmax')  # Output: probability per class
])

model.compile(
    optimizer = 'adam',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

# ================================================================
# CELL 6 — Train the Model
# ================================================================
# EarlyStopping  : stops if val_accuracy doesn't improve (saves time)
# ModelCheckpoint: saves the best model automatically

callbacks = [
    EarlyStopping(
        monitor              = 'val_accuracy',
        patience             = 5,
        restore_best_weights = True,
        verbose              = 1
    ),
    ModelCheckpoint(
        filepath        = '/content/drive/MyDrive/maize_leaf_disease_detection_prototype/best_model.keras',
        monitor         = 'val_accuracy',
        save_best_only  = True,
        verbose         = 1
    )
]

print("🌽 Training started...\n")
history = model.fit(
    train_data,
    validation_data = val_data,
    epochs          = EPOCHS,
    callbacks       = callbacks
)

# Save history so you never lose it after a restart
import json
history_dict = {
    'accuracy'     : history.history['accuracy'],
    'val_accuracy' : history.history['val_accuracy'],
    'loss'         : history.history['loss'],
    'val_loss'     : history.history['val_loss']
}
with open('/content/drive/MyDrive/maize_leaf_disease_detection_prototype/history.json', 'w') as f:
    json.dump(history_dict, f)
print("History saved to Drive.")

# Save class names to Drive
with open('/content/drive/MyDrive/maize_leaf_disease_detection_prototype/class_names.json', 'w') as f:
    json.dump(class_names, f)

print("\n✅ Training done! Model saved to your Google Drive.")

# ================================================================
# CELL 7 — Plot Results
# ================================================================
# Accuracy should go UP. Loss should go DOWN.
# If training accuracy >> val_accuracy: model is overfitting.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Maize Leaf Disease Detection Prototype — Training Results", fontsize=14)

axes[0].plot(history.history['accuracy'],     label='Train', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation', marker='o')
axes[0].set_title('Accuracy (higher = better)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

axes[1].plot(history.history['loss'],     label='Train', marker='o')
axes[1].plot(history.history['val_loss'], label='Validation', marker='o')
axes[1].set_title('Loss (lower = better)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/maize_leaf_disease_detection_prototype/training_results.png', dpi=150)
plt.show()
print("📊 Plot saved to your Google Drive.")

# ================================================================
# CELL 8 — Evaluate on Validation Set
# ================================================================
loss, acc = model.evaluate(val_data, verbose=0)
print(f"\n📊 Final Results:")
print(f"   Validation Accuracy : {acc * 100:.2f}%")
print(f"   Validation Loss     : {loss:.4f}")

# ================================================================
# CELL 9 — Predict on a New Image
# ================================================================
# Change image_path to any maize leaf image you want to test.
# You can upload an image to Colab using the folder icon on the left.

from tensorflow.keras.preprocessing import image as keras_image

def predict(image_path):
    img       = keras_image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds     = model.predict(img_array, verbose=0)
    idx       = np.argmax(preds[0])
    label     = class_names[idx]
    confidence= preds[0][idx] * 100

    print(f"\n🌽 Prediction: {label}  ({confidence:.1f}% confident)\n")
    print("All class probabilities:")
    for name, prob in zip(class_names, preds[0]):
        bar = '█' * int(prob * 40)
        print(f"   {name:20s} {prob*100:5.1f}%  {bar}")

    plt.figure(figsize=(5, 5))
    plt.imshow(keras_image.load_img(image_path))
    plt.title(f"{label}\n{confidence:.1f}% confidence", fontsize=13)
    plt.axis('off')
    plt.show()

# ---- Change this path to your image ----
image_path = "/content/your_image.jpg"   # upload an image to Colab first
predict(image_path)
