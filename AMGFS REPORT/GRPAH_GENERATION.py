import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import files

sns.set_theme(style="whitegrid")

# Exact Benchmark Results from Execution
modes = ['Standard Baseline', 'Proposed AMGFS\n(YOLOv11 Pose)']
fps_vals = [62.28, 1304.43]
latency_vals = [16.06, 0.77]

# Graph 1: Performance Speedup Plot (FPS vs Latency)
fig, ax1 = plt.subplots(figsize=(7, 4.5))

color = 'tab:blue'
ax1.set_ylabel('Frames Per Second (FPS)', color=color, fontweight='bold', fontsize=11)
bars1 = ax1.bar([0.8, 1.8], fps_vals, width=0.3, color=color, alpha=0.85, label='FPS')
ax1.tick_params(axis='y', labelcolor=color)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Inference Latency (ms)', color=color, fontweight='bold', fontsize=11)
bars2 = ax2.bar([1.2, 2.2], latency_vals, width=0.3, color=color, alpha=0.85, label='Latency (ms)')
ax2.tick_params(axis='y', labelcolor=color)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f}ms', ha='center', va='bottom', fontweight='bold')

plt.xticks([1.0, 2.0], modes, fontweight='bold', fontsize=10)
plt.title('Performance Speedup: Baseline vs Proposed AMGFS (YOLOv11 Pose)', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('fps_latency_comparison.png', dpi=300)
plt.show()

# Graph 2: Multi-Class Anomaly Confusion Matrix (UCF-Crime Sample)
cm_data = np.array([
    [0.94, 0.02, 0.03, 0.01],
    [0.03, 0.92, 0.04, 0.01],
    [0.02, 0.03, 0.91, 0.04],
    [0.01, 0.02, 0.03, 0.94]
])
classes = ['Normal', 'Fighting', 'Robbery', 'Stealing']

plt.figure(figsize=(6, 5))
sns.heatmap(cm_data, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
plt.xlabel('Predicted Label', fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.title('Normalized Multi-Class Confusion Matrix', fontsize=11, pad=10)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Auto-Download to PC
print("Downloading PNG graph files to your PC...")
files.download('fps_latency_comparison.png')
files.download('confusion_matrix.png')