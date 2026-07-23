# Dataset

## Fashion-MNIST Dataset

### Overview

The Fashion-MNIST dataset is a multi-class image classification dataset designed as a direct drop-in replacement for the classic MNIST dataset. It consists of Zalando's article images—60,000 training examples and 10,000 test examples.

Each example is a 28x28 grayscale image associated with a label from 10 clothing categories. This dataset is widely used for benchmarking machine learning and deep learning classification algorithms.

---

## Dataset Information

- **Dataset Name:** Fashion-MNIST
- **Task:** Multi-Class Image Classification
- **Number of Instances:** 70,000 (60,000 Training + 10,000 Testing)
- **Number of Features:** 784 (28 × 28 flattened pixel values)
- **Number of Classes:** 10
- **Value Range:** Grayscale pixel intensities from 0 to 255

---

## Target Classes

| Class | Class Name | Description |
|-------|------------|-------------|
| 0 | T-shirt/top | Top wear / T-shirt |
| 1 | Trouser | Pants / Trousers |
| 2 | Pullover | Pullover / Sweater |
| 3 | Dress | Dress |
| 4 | Coat | Outerwear / Coat |
| 5 | Sandal | Open-toed sandal |
| 6 | Shirt | Buttoned / Formal shirt |
| 7 | Sneaker | Athletic sneaker |
| 8 | Bag | Handbag / Tote |
| 9 | Ankle boot | High boot |

---

## Data Format & Preprocessing

- **Input Dimension:** Each image is originally a $28 \times 28$ grayscale matrix.
- **Flattening:** Reshaped into a 1D vector of size 784 ($28 \times 28 = 784$).
- **Normalization:** Pixel intensities in $[0, 255]$ are scaled to $[0.0, 1.0]$ by dividing by 255.0.
- **Label Encoding:** Integer class labels (0 to 9) are one-hot encoded into binary target vectors of length 10 for training with categorical cross-entropy loss.

---

## Source

The dataset was created by:

**Han Xiao, Kashif Rasul, and Roland Vollgraf**

Zalando Research, Berlin, Germany

Publicly available at: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

---

## License

The Fashion-MNIST dataset is released under the **MIT License**.

---

## Usage in this Experiment

This dataset is used to train and evaluate a **Multi-Layer Perceptron (MLP)** classifier.

The workflow includes:
- Data loading via `keras.datasets.fashion_mnist`
- Data preprocessing (flattening, normalization, one-hot encoding)
- Feature visualization (sample grids and class distribution plots)
- Train-validation-test split
- Baseline MLP model construction & training
- Hyperparameter tuning via `RandomizedSearchCV` with 5-fold cross-validation
- Model evaluation and performance comparison (Baseline vs. Optimized)
