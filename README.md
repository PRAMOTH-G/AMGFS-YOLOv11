```markdown
# AMGFS-YOLOv11: Sub-Millisecond CCTV Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, resource-conscious deep learning framework for real-time CCTV threat identification using **Adaptive Motion-Guided Frame Skipping (AMGFS)** and **YOLOv11-Pose**.

---

## 💡 Key Features

* **Adaptive Motion-Guided Frame Skipping (AMGFS):** Filters out non-informative background frames using a motion differential gate ($\tau_{\text{motion}} = 0.03$), discarding up to **99.96%** of redundant idle frames.
* **Single-Pass Pose Integration:** Replaces multi-stage pipelines by running human detection and 17 skeleton keypoint extractions in one single forward pass using YOLOv11n-Pose.
* **Multi-Person Tracking:** Integrated with **ByteTrack** for consistent identity tracking across sequential frames.
* **Sub-Millisecond Average Latency:** Achieves an average per-frame processing latency of **0.77 ms** and throughput of **1,304.43 FPS** on edge-level surveillance streams.
* **Automated Anomaly Classification:** Classifies events into *Normal*, *Fighting*, *Robbery*, and *Stealing* with ~93% accuracy on UCF-Crime subsets.

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/PRAMOTH-G/AMGFS-YOLOv11.git](https://github.com/PRAMOTH-G/AMGFS-YOLOv11.git)
cd AMGFS-YOLOv11

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run Benchmark Evaluation

Run the core performance benchmark:

```bash
python AMGFS_PERFORMANCE.py

```

*Results will be saved automatically inside the `outputs/` folder.*

### 4. Generate Performance Plots

Generate confusion matrix and speedup graphs:

```bash
python GRAPH_GENERATION.py

```

---

## 📊 Performance & Results

| Metric | Standard Baseline | Proposed AMGFS (YOLOv11 Pose) |
| --- | --- | --- |
| **Processed / Total Frames** | 4,568 / 4,568 | **2 / 4,568** |
| **Frame Skip Ratio ($\eta_{\text{skip}}$)** | 0.00% | **99.96%** |
| **Average Latency** | 16.06 ms | **0.77 ms** |
| **Throughput (FPS)** | 62.28 FPS | **1,304.43 FPS** |
| **Speedup Factor** | 1.00× | **20.94×** |

---

## 👥 Authors & Citation

* **Raja Venkateshwaran K.C** - *Department of IT, Nandha College of Technology*
* **Pramoth G** - *Department of IT, Nandha College of Technology*
* **Pavithra D** - *Department of IT, Nandha College of Technology*
* **Tharish R** - *Department of IT, Nandha College of Technology*

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE&utm_source=gemini) file for details.

```

```
