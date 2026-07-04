# ============================================================
#  MAIZE LEAF DISEASE DETECTION PROTOTYPE
#  predict.py — Predict disease from a single image
#
#  Usage:
#    python predict.py --image path/to/leaf.jpg
#    python predict.py --image path/to/leaf.jpg --model MobileNetV2
# ============================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
from tensorflow.keras.preprocessing import image as keras_image

# ── Argument parser ──────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Predict maize leaf disease from an image"
)
parser.add_argument('--image',  required=True,
                    help='Path to the leaf image')
parser.add_argument('--model',  default='MobileNetV2',
                    choices=['MobileNetV2','EfficientNetB0',
                             'ResNet50','ConvNeXtTiny'],
                    help='Which trained model to use')
parser.add_argument('--output', default='outputs',
                    help='Folder where trained models are saved')
args = parser.parse_args()

IMG_SIZE = (224, 224)

# ── Load model and class names ────────────────────────────────
model_path = os.path.join(args.output, f'{args.model}_final.keras')
names_path = os.path.join(args.output, 'class_names.json')

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print(f"   Train the model first using train.py")
    exit(1)

print(f"Loading {args.model}...")
model = tf.keras.models.load_model(model_path)

with open(names_path) as f:
    class_names = json.load(f)

print(f"✅ Model loaded — classes: {class_names}\n")

# ── Load and preprocess image ─────────────────────────────────
img       = keras_image.load_img(args.image, target_size=IMG_SIZE)
img_array = keras_image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

# ── Predict ───────────────────────────────────────────────────
preds      = model.predict(img_array, verbose=0)
idx        = np.argmax(preds[0])
label      = class_names[idx]
confidence = preds[0][idx] * 100

print(f"🌽 Prediction  : {label}")
print(f"   Confidence  : {confidence:.1f}%\n")
print("All class probabilities:")
for name, prob in zip(class_names, preds[0]):
    bar = '█' * int(prob * 40)
    print(f"   {name:<20} {prob*100:5.1f}%  {bar}")

# ── Show image ────────────────────────────────────────────────
plt.figure(figsize=(5, 5))
plt.imshow(keras_image.load_img(args.image))
plt.title(f"{label}\n{confidence:.1f}% confidence", fontsize=13)
plt.axis('off')
plt.tight_layout()
plt.savefig('prediction_result.png', dpi=150)
plt.show()
print("\n✅ Result image saved as prediction_result.png")
