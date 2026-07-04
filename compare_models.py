# ============================================================
#  MAIZE LEAF DISEASE DETECTION PROTOTYPE
#  compare_models.py — Compare all trained models
#  Run this after training all 4 models
# ============================================================

import os, json
import numpy as np
import matplotlib.pyplot as plt

OUTPUT = "outputs"
MODELS = ["MobileNetV2", "EfficientNetB0", "ResNet50", "ConvNeXtTiny"]
colors = ['#2196F3','#FF5722','#4CAF50','#9C27B0']

results = []
for model_name in MODELS:
    path = os.path.join(OUTPUT, f"{model_name}_results.json")
    if os.path.exists(path):
        with open(path) as f:
            results.append(json.load(f))
        print(f"✅ Loaded {model_name}")
    else:
        print(f"⚠️  {model_name} not found — train it first")

if not results:
    print("No results found. Train models first using train.py")
    exit(1)

# ── Print comparison table ────────────────────────────────────
print(f"\n{'='*70}")
print("MAIZE LEAF DISEASE DETECTION — MODEL COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}"
      f"{'Recall':>8}{'F1 Score':>10}")
print("-"*70)
for r in results:
    print(f"{r['model']:<20}{r['accuracy']:>10.1%}"
          f"{r['precision']:>11.3f}"
          f"{r['recall']:>8.3f}"
          f"{r['f1']:>10.3f}")
print("="*70)

best = max(results, key=lambda x: x['f1'])
print(f"\n🏆 Best model : {best['model']}")
print(f"   Accuracy   : {best['accuracy']:.1%}")
print(f"   F1 Score   : {best['f1']:.3f}")

# ── Bar chart comparison ──────────────────────────────────────
metrics = ['accuracy','precision','recall','f1']
labels  = ['Accuracy','Precision','Recall','F1 Score']
x       = np.arange(len(metrics))
width   = 0.2

fig, ax = plt.subplots(figsize=(13, 6))
for i, (r, color) in enumerate(zip(results, colors)):
    vals = [r[m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width,
                  label=r['model'], color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.2f}', ha='center',
                fontsize=7, fontweight='bold')

ax.set_title('Maize Leaf Disease Detection — Model Comparison',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim([0, 1.15])
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'model_comparison.png'), dpi=150)
plt.show()
print(f"\n✅ Comparison chart saved to {OUTPUT}/model_comparison.png")
