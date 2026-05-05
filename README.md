# Smartphone-Based Fall Detection and Activity Recognition
### CS 286 — Wearable AI & mHealth | San Jose State University | Spring 2026

> **Authors:** Mrunal Kotkar · Sushant Lokhande  
> **Dataset:** [MobiFall Dataset v2.0](https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20) (24 subjects, 4 fall types, 9 ADLs)

---

## Overview

This project investigates smartphone-based fall detection and human activity recognition using raw inertial sensor data (accelerometer + gyroscope). Three classification tasks are studied:

- **Binary** — Fall vs. non-fall (ADL)
- **Fall4** — 4-class fall-type classification (FOL, FKL, BSC, SDL)
- **Multi13** — 13-class activity recognition (4 falls + 9 ADLs)

Three modeling families are compared: classical machine learning on handcrafted features, deep learning directly on raw sensor windows (ANN / CNN / LSTM), and transfer learning via UCI HAR pretraining and ImageNet spectrogram fine-tuning. Generalization is evaluated using Leave-One-Subject-Out (LOSO) and Leave-One-Event-Out (LOEO) protocols in addition to a standard stratified split.

---

## Results Summary

### Binary Fall Detection (3-second window)

| Model | Accuracy | F1-Score |
|---|---|---|
| XGBoost (Classical ML) | 0.9509 | 0.9562 |
| ANN | 0.9699 | 0.9677 |
| LSTM | 0.9516 | 0.9494 |
| CNN (**best**) | **0.9935** | **0.9931** |
| VGG16 (Spectrogram TL) | 0.9895 | 0.9889 |
| ResNet50 (Spectrogram TL) | 0.9791 | 0.9778 |
| DeepCNN (UCI HAR → MobiFall) | 0.9909 | 0.9903 |

### CNN Generalization (Binary, 3s window)

| Protocol | Accuracy | F1-Score | Folds |
|---|---|---|---|
| Standard split | 0.9935 | 0.9931 | 1 |
| LOSO | 0.9814 | 0.9684 | 24 |
| LOEO | 0.9917 | 0.9951 | 288 |

---

## Repository Structure

```
286_final/
├── README.md
├── notebooks/
│   ├── 01_classical_ml_baseline.ipynb
│   ├── 02_deep_learning_ann_cnn_lstm.ipynb
│   ├── 03_loso_loeo_evaluation.ipynb
│   ├── 04_transfer_uci_har_pretrain.ipynb
│   └── 05_transfer_spectrogram_resnet_vgg.ipynb
└── report/
    └── CS286_Demo2_Report.pdf
```

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | `01_classical_ml_baseline.ipynb` | Downloads MobiFall v2 from Kaggle, extracts handcrafted statistical features (mean, std, max, skewness, kurtosis) per window per channel, and trains six classical classifiers: Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, XGBoost. |
| 02 | `02_deep_learning_ann_cnn_lstm.ipynb` | Full preprocessing pipeline (sync, resample to 100 Hz, 1s/2s/3s windowing, class balancing), followed by ANN, CNN, and LSTM training across all three tasks. Generates confusion matrices and training/validation loss curves. |
| 03 | `03_loso_loeo_evaluation.ipynb` | Implements LOSO (24 folds, one subject held out) and LOEO (288 folds, one fall event held out) evaluation of the CNN on binary fall detection. Reports per-fold accuracy, F1, precision, and recall. |
| 04 | `04_transfer_uci_har_pretrain.ipynb` | Downloads UCI HAR, builds the DeepCNN HAR backbone (~1M params, residual connections, separable convolutions, GAP), trains to 94.1% on UCI HAR, and saves weights for downstream transfer to MobiFall. |
| 05 | `05_transfer_spectrogram_resnet_vgg.ipynb` | Converts MobiFall windows to STFT spectrogram images (224×224×3, acc top / gyro bottom), then fine-tunes ResNet50 and VGG16 with a two-stage protocol (frozen backbone → partial unfreeze) on binary, fall4, and multi13 tasks. |

---

## Pipeline

```
MobiFall v2.0 (raw .txt files)
        │
        ▼
  Preprocessing
  ┌─────────────────────────────────────────────┐
  │  Sync acc + gyro timestamps                 │
  │  Resample to 100 Hz                         │
  │  Window: 1s / 2s / 3s, 50% overlap         │
  │  Balance: cap long ADL recordings           │
  └─────────────────────────────────────────────┘
        │
        ├──► Statistical features ──► Classical ML (01)
        │
        ├──► Raw 6-channel windows ──► ANN / CNN / LSTM (02)
        │                         └──► LOSO / LOEO eval (03)
        │
        ├──► UCI HAR pretraining ──► DeepCNN fine-tune on MobiFall (04)
        │
        └──► STFT Spectrograms ──► ResNet50 / VGG16 fine-tune (05)
```

---

## Setup

All notebooks are designed to run on **Google Colab** (GPU recommended). Each notebook installs its own dependencies in the first cell.

**Common dependencies:**
```
tensorflow >= 2.12
scikit-learn
xgboost
scipy
kagglehub
numpy
pandas
matplotlib
```

**Dataset access:**  
Notebooks 01–03 download MobiFall v2.0 automatically via `kagglehub`. A Kaggle account and API token are required. Notebook 04 downloads UCI HAR directly from the UCI ML Repository. No manual data download is needed.

---

## Key Findings

1. **CNN on raw 3-second windows is the strongest model** across all three tasks, outperforming classical ML by ~4% and LSTM by ~4% on binary detection, and by 14–17% on the harder fall-type and 13-class tasks.

2. **Longer windows consistently help.** Performance improves monotonically from 1s → 2s → 3s for all model families, confirming that the full fall motion pattern requires at least 3 seconds of context.

3. **UCI HAR transfer hurts on fall-type tasks** (negative transfer). Because UCI HAR contains no fall events, the pretrained features suppress the brief impact peaks that distinguish fall subtypes, reducing fall4 accuracy to 60.6% vs. 80.6% for training from scratch.

4. **Spectrogram transfer (VGG16) outperforms UCI HAR transfer on harder tasks**, reaching 70.6% on fall4 and 82.6% on multi13 — even though ImageNet features were learned from natural photographs rather than sensor data.

5. **FOL/FKL confusion is the dominant error mode** across all architectures. Both are forward falls that differ only in which body part contacts the ground first — a sub-second kinematic difference that 3-second pocket-mounted sensing cannot reliably resolve.

6. **The CNN generalizes well** under strict protocols: LOSO accuracy drops only 1.2% from the standard split (0.9814 vs. 0.9935), and LOEO achieves perfect precision (1.0000) with 99.2% recall.

---

## Citation

If you use this codebase or refer to this work, please cite:

```
Kotkar, M., & Lokhande, S. (2026). Smartphone-Based Fall Detection and Fall-Type
Classification. Proceedings of CS 2026, San Jose State University.
```

---

## License

This project is for academic purposes. The MobiFall Dataset v2.0 is subject to its own license terms; please refer to the [original dataset page](https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20) before reuse.
