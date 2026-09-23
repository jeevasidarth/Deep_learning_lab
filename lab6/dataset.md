# Dataset Documentation: Deep Learning Laboratory (Experiment 6)

This document provides detailed descriptions of the three datasets utilized in **Experiment 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding**.

---

## 1. UCI Human Activity Recognition (HAR) Using Smartphones Dataset

### 📌 Overview
The **UCI HAR Dataset** is a benchmark dataset for human activity recognition based on smartphone inertial sensors. It records tri-axial angular velocity and linear acceleration collected from a waist-mounted smartphone carried by 30 human subjects performing daily activities.

### 📐 Specifications & Structure
- **Data Source**: UCI Machine Learning Repository
- **Sampling Frequency**: 50 Hz (50 readings per second)
- **Windowing**: Fixed-width sliding windows of **2.56 seconds** with **50% overlap** ($128$ readings/time steps per window).
- **Input Channels (9 Inertial Signals)**:
  1. `total_acc_x`: Total acceleration along the X-axis ($g$).
  2. `total_acc_y`: Total acceleration along the Y-axis ($g$).
  3. `total_acc_z`: Total acceleration along the Z-axis ($g$).
  4. `body_acc_x`: Estimated body acceleration along the X-axis ($g$).
  5. `body_acc_y`: Estimated body acceleration along the Y-axis ($g$).
  6. `body_acc_z`: Estimated body acceleration along the Z-axis ($g$).
  7. `body_gyro_x`: Angular velocity around the X-axis ($\text{rad/s}$).
  8. `body_gyro_y`: Angular velocity around the Y-axis ($\text{rad/s}$).
  9. `body_gyro_z`: Angular velocity around the Z-axis ($\text{rad/s}$).

### 🏷️ Activity Classes (6 Classes)
1. `WALKING` (Dynamic activity)
2. `WALKING_UPSTAIRS` (Dynamic activity)
3. `WALKING_DOWNSTAIRS` (Dynamic activity)
4. `SITTING` (Static posture)
5. `STANDING` (Static posture)
6. `LAYING` (Static posture)

### ⚙️ Preprocessing & Normalization
- **Class-Balanced Subset**: A stratified subset of 500 windows per class (3,000 total windows) or full partition.
- **Data Split**: Stratified **70% Training / 15% Validation / 15% Testing**.
- **Per-Channel Z-Score Normalization**:
  $$\hat{X}_{t, c} = \frac{X_{t, c} - \mu_c}{\sigma_c}$$
  where mean $\mu_c$ and standard deviation $\sigma_c$ are calculated **strictly from the training split** to eliminate data leakage.
- **Prepared Output Tensor Shape**:
  - Training set: $(2100, 128, 9)$
  - Validation set: $(450, 128, 9)$
  - Testing set: $(450, 128, 9)$

---

## 2. Subsampled UCF101 Action Recognition Video Dataset

### 📌 Overview
**UCF101** is an action recognition dataset of realistic action videos collected from YouTube, containing 101 action categories. For Part 6 of Experiment 6, a subset of 4 distinct human action categories is used to demonstrate CNN feature extraction combined with temporal sequence modeling.

### 🎥 Selected Classes (4 Categories)
1. `ApplyEyeMakeup`
2. `ApplyLipstick`
3. `Archery`
4. `BabyCrawling`

### ⚙️ Frame Processing & Feature Extraction Pipeline
- **Frame Sampling**: 10 frames sampled uniformly across the duration of each video clip.
- **Spatial Resizing**: All sampled frames are resized to $224 \times 224 \times 3$ RGB pixels.
- **Group-Based Splitting**: Split into Train/Validation/Test based on video group numbers (`g01` to `g25`) to prevent data leakage from clips originating from the same source video.
- **CNN Feature Backbone**: Pre-trained **MobileNetV2** (ImageNet weights, classification head removed).
  - Global average pooling outputs a **1280-dimensional feature vector** for each frame.
- **Processed Video Input Tensor Shape**: $(N_{\text{videos}}, 10, 1280)$

### 🧪 Temporal Order Shuffle Test
To evaluate whether the recurrent model actually learns sequential temporal dynamics or relies merely on static visual features, frame features are randomly shuffled along the time axis during testing:
$$\mathbf{F}_{\text{shuffled}} = \text{Shuffle}(\mathbf{F}_{1:10}, \text{dim}=\text{time})$$
A noticeable drop in test accuracy confirms reliance on temporal context.

---

## 3. Synthetic Sequence Reversal Dataset

### 📌 Overview
A synthetic task generated to evaluate Sequence-to-Sequence (Seq2Seq) Encoder-Decoder LSTM networks. The task requires the network to read an arbitrary sequence of discrete tokens and output the sequence in reverse order.

### 📐 Task Definition
- **Input Sequence**: $[x_1, x_2, x_3, \dots, x_L]$
- **Target Sequence**: $[x_L, \dots, x_3, x_2, x_1]$
- **Vocabulary Size**: 10 tokens:
  - Token `0`: Special `START` / `PADDING` token.
  - Tokens `1` through `9`: Input payload tokens.
- **Sequence Length ($L$)**: Fixed sequence length (e.g., $L=5$).

### 🔄 Training & Inference Mechanism
- **Training**: Uses **Teacher Forcing** where the ground-truth target token $y_{t-1}$ is fed into the Decoder LSTM at step $t$:
  $$\mathbf{x}_{\text{dec}, t} = y_{t-1}$$
- **Inference**: Autoregressive greedy decoding where the Decoder uses its own previous output prediction $\hat{y}_{t-1}$ as input for step $t$:
  $$\mathbf{x}_{\text{dec}, t} = \hat{y}_{t-1}$$
