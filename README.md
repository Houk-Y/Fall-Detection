# 🚨 Fall Detection Using Deep Learning

> A comparative study of CNN, MobileNetV2, EfficientNetB0, and ResNet50 for real-time fall detection — with a Streamlit web app and Telegram Bot alert system.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Models Compared](#models-compared)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Falls are the second leading cause of accidental injury deaths globally, responsible for approximately **684,000 fatal deaths per year** (WHO). This project builds a real-time fall detection system using deep learning on image data, comparing four architectures to find the best model for deployment.

The best-performing model — **MobileNetV2** — is deployed in a live web application built with **Streamlit**, supporting both webcam streaming and video file upload, with automated **Telegram Bot alerts** when a fall is detected.

---

## Problem Statement

- 1 in 4 older adults fall each year
- Falls cause ~3 million emergency department visits annually (USA)
- Conventional wearable detectors require user compliance and physical attachment
- This project offers a **non-intrusive, camera-based alternative** using deep learning

---

## Models Compared

| Model          | Type               | Base Frozen   |
| -------------- | ------------------ | ------------- |
| Custom CNN     | Built from scratch | —             |
| MobileNetV2 ⭐ | Transfer learning  | ✅ Yes        |
| EfficientNetB0 | Transfer learning  | ❌ Fine-tuned |
| ResNet50       | Transfer learning  | ✅ Yes        |

> ⭐ MobileNetV2 selected as best model for deployment

All models share the same classification head:

```
GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```

---

## Dataset

- **Format:** YOLO-style image + label pairs
- **Input size:** 256 × 256 px (RGB)
- **Labels:** `0` = Fall · `1` = Non-fall
- **Split:** 80% training / 20% test (stratified)
- **Normalization:** pixel values ÷ 255 → [0.0, 1.0]

### Data Augmentation (training only)

| Technique            | Value   |
| -------------------- | ------- |
| Rotation             | ±20°    |
| Width / Height shift | ±10%    |
| Zoom                 | ±20%    |
| Shear                | ±20%    |
| Horizontal flip      | Enabled |
| Fill mode            | Nearest |

---

## Project Structure

```
fall-detection/
│
├── fall_detection.ipynb       # Training notebook (Google Colab)
├── app.py                     # Streamlit deployment app
├── mobilenet_model.h5         # Saved best model weights
│
├── dataset/
│   ├── images/
│   │   └── train/             # Training images
│   └── labels/
│       └── train/             # YOLO annotation .txt files
│
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fall-detection.git
cd fall-detection
```

### 2. Install dependencies

```bash
pip install streamlit tensorflow opencv-python streamlit-webrtc requests numpy
```

### 3. Configure Telegram Bot

Open `app.py` and set your credentials:

```python
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID   = "YOUR_TELEGRAM_CHAT_ID"
```

> To create a bot: open Telegram → search **@BotFather** → `/newbot` → copy the token.
> To get your chat ID: send a message to your bot, then visit:
> `https://api.telegram.org/bot<TOKEN>/getUpdates`

---

## Usage

### Training (Google Colab / Jupyter)

Open `fall_detection.ipynb` and run all cells. The notebook will:

1. Load and preprocess the dataset
2. Apply data augmentation
3. Train all four models for 30 epochs
4. Evaluate and compare results
5. Save the best model as `mobilenet_model.h5`

### Running the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

**Input modes:**

- **Live Webcam** — real-time detection via WebRTC
- **Upload Video** — process an MP4, AVI, or MOV file

---

## Results

| Model          | Accuracy | Precision | Recall (Fall) | F1-Score |
| -------------- | -------- | --------- | ------------- | -------- |
| Custom CNN     | —        | —         | —             | —        |
| MobileNetV2 ⭐ | —        | —         | —             | —        |
| EfficientNetB0 | —        | —         | —             | —        |
| ResNet50       | —        | —         | —             | —        |

> Fill in values after running the training notebook.

**Training config:**

```
Optimizer : Adam (lr = 1e-4)
Loss      : Binary Cross-Entropy
Epochs    : 30
Batch     : 32
Platform  : Google Colab (GPU)
```

---

## Deployment

The app uses **MobileNetV2** for inference with the following logic:

```python
probability_fall = 1 - model.predict(frame)

if probability_fall >= 0.5:
    label = "Fall"      # class 0 — red overlay
else:
    label = "Non-Fall"  # class 1 — green overlay
```

### Telegram Alert System

| Setting         | Value                             |
| --------------- | --------------------------------- |
| Alert threshold | 0.8 probability                   |
| Cooldown period | 20 seconds                        |
| Delivery        | Async background thread           |
| Payload         | JPEG snapshot + probability score |

When a fall probability exceeds **0.8**, a photo of the frame is sent instantly to your Telegram chat — without blocking the video stream.

---

## Future Work

- Video-based temporal modeling (LSTM, 3D-CNN)
- Multi-modal fusion (camera + accelerometer)
- Edge deployment on Raspberry Pi or NVIDIA Jetson
- Larger and more diverse datasets across populations and environments

---

## References

1. CDC, "Facts About Falls," Jan. 2026
2. WHO, "Falls Fact Sheet," Apr. 2021
3. Sandler et al., "MobileNetV2," CVPR 2018
4. Tan & Le, "EfficientNet," ICML 2019
5. He et al., "ResNet," CVPR 2016
6. Núñez-Marcos et al., "Vision-Based Fall Detection with CNNs," 2017

---

## License

This project is for academic research purposes.
