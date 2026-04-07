# Fall Detection - Video Classification

This project implements and compares four different Deep Learning architectures to classify videos as `fall` or `non-fall`. It evaluates both spatial (frame-based) and temporal (sequence-based) approaches to identify the most effective method for fall detection.

## 🚀 Project Overview

The system compares the following models to determine the best performer:

- **Simple CNN**: A custom 3-block baseline convolutional network.
- **ResNet18**: Transfer learning using a pre-trained ImageNet-1K backbone.
- **MobileNetV2**: An efficient, lightweight model designed for edge deployment.
- **CNN + LSTM**: A temporal model using MobileNetV2 for feature extraction and a Bi-Directional LSTM to analyze motion dynamics over a sequence of 16 frames.

## 📂 Dataset Structure

To run the notebook, organize your data in the following directory format:

```text
dataset/
├── fall/
│   ├── video_001.mp4
│   └── video_002.avi
└── non-fall/
    ├── video_101.mp4
    └── video_102.mkv
```

_Supported formats include: .mp4, .avi, .mov, .mkv, and .wmv._

## 🛠️ Setup Instructions

1.  **Clone the repository** and ensure `fall_detection.ipynb` is in your working directory.
2.  **Install Dependencies**:
    ```bash
    pip install numpy pandas torch torchvision opencv-python Pillow scikit-learn matplotlib seaborn
    ```
3.  **Hardware**: The code automatically detects and uses **CUDA** (GPU) if available; otherwise, it defaults to CPU.

## 📊 Training Pipeline

- **Data Split**: 70% Train | 15% Validation | 15% Test.
- **Preprocessing**:
  - Frames are resized to **224x224**.
  - **Augmentation**: Includes random horizontal flips, rotations, and color jittering.
  - **Normalization**: Uses ImageNet statistics ($mean=[0.485, 0.456, 0.406]$, $std=[0.229, 0.224, 0.225]$).
- **Optimization**: Adam optimizer with a learning rate of $1e-4$ and a `ReduceLROnPlateau` scheduler.
- **Early Stopping**: Implemented with a patience of 7 epochs to prevent overfitting.

## 🏆 Results & Deployment

Upon completion, the notebook performs the following:

1.  Displays a comparative table of **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
2.  Saves the best-performing model weights as `best_model.pth`.
3.  Generates `model_meta.json` containing configuration details (image size, class names, and sequence flags) for deployment in applications like Streamlit.
