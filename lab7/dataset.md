# Dataset Documentation: MNIST Handwritten Digits

## Overview
This document provides details on the **MNIST (Modified National Institute of Standards and Technology)** handwritten digit dataset used in **Experiment 7** for training and evaluating Fully Connected Autoencoders, Convolutional Autoencoders (CAE), Denoising Autoencoders (DAE), and Variational Autoencoders (VAE).

---

## Dataset Specifications

| Property | Details |
| :--- | :--- |
| **Dataset Name** | MNIST Handwritten Digits |
| **Source** | Yann LeCun, Corinna Cortes, Christopher J.C. Burges |
| **API Provider** | `tf.keras.datasets.mnist` |
| **Image Resolution** | $28 \times 28$ pixels |
| **Channels** | 1 (Grayscale) |
| **Pixel Value Range** | Raw: $[0, 255]$ (uint8) → Normalized: $[0.0, 1.0]$ (float32) |
| **Total Samples** | 70,000 (60,000 Training + 10,000 Test) |
| **Classes** | 10 classes (Digits 0 through 9) |

---

## Experimental Setup & Data Subset

To balance computational efficiency and statistical stability for laboratory execution:
- **Training Set Size**: 10,000 images sampled from the full training set.
- **Test Set Size**: 2,000 images sampled from the test set.

### Unsupervised / Self-Supervised Learning Setup
In autoencoder tasks, ground-truth digit labels $y \in \{0, \dots, 9\}$ are **not** used during training. The input image $x$ serves as both the network input and the reconstruction target ($x \to \text{Encoder} \to z \to \text{Decoder} \to \hat{x}$).

---

## Data Preprocessing Pipeline

1. **Pixel Normalization**:
   Pixel intensities are scaled from integer range $[0, 255]$ to floating-point range $[0.0, 1.0]$:
   $$x_{\text{normalized}} = \frac{x}{255.0}$$

2. **Reshaping & Spatial Formatting**:
   - **Fully Connected Autoencoder**: Flattened from 2D matrices into 1D vectors of shape $(784,)$:
     $$x_{\text{flat}} \in \mathbb{R}^{784}$$
   - **Convolutional / DAE / VAE**: Preserved 2D spatial dimensions with channel dimension explicit shape $(28, 28, 1)$:
     $$x_{\text{spatial}} \in \mathbb{R}^{28 \times 28 \times 1}$$

3. **Controlled Image Corruption (For Denoising Autoencoder)**:
   - **Additive Gaussian Noise**:
     $$x_{\text{noisy}} = \text{clip}\left(x + \mathcal{N}(0, \sigma^2), \, 0.0, \, 1.0\right) \quad \text{with } \sigma = 0.3$$
   - **Salt-and-Pepper Noise**:
     Randomly setting $10\%$ of pixels ($p=0.1$) to either $0.0$ (pepper) or $1.0$ (salt).

---

## Data Visualization & Distribution

- **Class Balance**: The dataset contains a balanced distribution across all 10 digit classes ($\approx 10\%$ per digit).
- **Latent Manifold Visualization**: During evaluation, latent embeddings $z$ are color-coded using original digit labels to verify class cluster separation in 2D/16D latent spaces.

---

## How Data is Loaded in `deep7.ipynb`

```python
import tensorflow as tf

# Load raw dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select experimental subset
x_train_sub = x_train[:10000]
x_test_sub = x_test[:2000]

# Normalize pixel values
x_train_sub = x_train_sub.astype('float32') / 255.0
x_test_sub = x_test_sub.astype('float32') / 255.0

# Add channel dimension for CNN models
x_train_conv = x_train_sub[..., tf.newaxis]
x_test_conv = x_test_sub[..., tf.newaxis]
```
