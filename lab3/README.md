# Week 3 - Experiment 3: Implementation of Convolutional Neural Networks (CNNs) for Image Classification

## Experiment Objectives

This experiment focuses on understanding, implementing, and evaluating a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 Dataset** with **TensorFlow / Keras**.

The objectives of this experiment are to:
- Explain the convolution operation and parameter sharing.
- Compute output feature map dimensions based on stride, padding, and kernel size.
- Visualize early-layer feature maps to understand what patterns filters learn.
- Implement and compare **Max Pooling** and **Average Pooling** layers.
- Calculate total CNN parameters layer-by-layer.
- Build, train, and evaluate a CNN model for multi-class image classification.
- Analyze model performance and overfitting patterns on the test set.

---

# Folder Structure

```
lab3/
│
├── CNN.ipynb               # Jupyter notebook with complete implementation
├── requirements.txt         # List of Python dependencies
├── README.md                # Project documentation and guide
└── Experiment_3.tex        # LaTeX source code for the lab report
```

---

# Dataset Information

**Dataset:** CIFAR-10

The dataset contains $32 \times 32$ RGB (color) images across 10 balanced object classes.

### Dataset Summary

- **Training Samples:** 50,000 (perfectly balanced with 5,000 images per class)
- **Testing Samples:** 10,000
- **Image Size:** $32 \times 32 \times 3$ (RGB)
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

# Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The experiment relies on:
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Using the Jupyter Notebook

Open the notebook in Jupyter Notebook, JupyterLab, or VS Code:

```bash
jupyter notebook CNN.ipynb
```

Run the cells sequentially from top to bottom.

---

# Implementation Workflow & Tasks

The notebook implements the following experimental procedure:

### Task 1 — Dataset Exploration & Visualization
- Load the CIFAR-10 dataset using Keras.
- Display a grid of 10 sample images with class labels.
- Verify dimensions: Training images `(50000, 32, 32, 3)`, Testing images `(10000, 32, 32, 3)`.
- Plot class distribution across the training dataset (confirming balance).
- Normalize pixel values from $[0, 255]$ to $[0.0, 1.0]$.

### Task 2 — Convolution with Different Kernel Sizes
- Apply an 8-filter convolution layer to a sample image with `valid` padding and stride 1.
- Compare output dimensions for kernel sizes $3\times3$, $5\times5$, and $7\times7$ (verifying the output size formula).

### Task 3 — Effect of Stride and Padding
- Analyze output spatial dimensions for $3\times3$ kernels with different combinations of stride (1 vs 2) and padding (`valid` vs `same`).

### Task 4 — Feature Map Visualization
- Extract activations of the first convolution layer for a test image.
- Plot 8 of the 32 learned filters to observe edge detection, color-contrast regions, and texture blobs.

### Task 5 — Max Pooling vs. Average Pooling
- Build and train an identical CNN architecture replacing Keras `MaxPooling2D` with `AveragePooling2D`.
- Compare performance (accuracy, test loss, and generalization calibration) over 20 epochs.

### Task 6 & 7 — CNN Construction, Training, and Evaluation
- **Architecture:** 
  $$\text{Input } (32\times32\times3) \rightarrow \text{Conv2D(32, 3}\times\text{3, ReLU)} \rightarrow \text{MaxPooling2D(2}\times\text{2)} \rightarrow \text{Conv2D(64, 3}\times\text{3, ReLU)} \rightarrow \text{MaxPooling2D(2}\times\text{2)} \rightarrow \text{Flatten} \rightarrow \text{Dense(128, ReLU)} \rightarrow \text{Dense(10, Softmax)}$$
- **Total Trainable Parameters:** 545,098
- **Training Parameters:** Optimizer: Adam, Loss: Sparse Categorical Crossentropy, Epochs: 20, Batch Size: 32, Val Split: 20%.
- **Evaluation:** Precision, Recall, F1-Score, and Confusion Matrix on the test set.

---

# Experimental Results Summary

- **Overfitting Analysis:** Training accuracy reached **96.47%** by epoch 20, while validation accuracy peaked at **70.6%** (epoch 8) and declined to **67.85%**. Validation loss rose continuously after epoch 5, confirming overfitting to training data.
- **Test Performance (Max Pooling):**
  - Test Accuracy: **66.94%**
  - Weighted F1-score: **0.6671**
- **Pooling Comparison:**
  - **Max Pooling:** Test Accuracy = 66.94%, Test Loss = 2.2486
  - **Average Pooling:** Test Accuracy = **67.44%**, Test Loss = **1.7950**
  - Average pooling yielded marginally higher accuracy and better calibration (lower loss) for this configuration.
