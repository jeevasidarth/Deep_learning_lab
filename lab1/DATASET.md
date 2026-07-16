# Dataset

## Banknote Authentication Dataset

### Overview

The Banknote Authentication dataset is a binary classification dataset used to distinguish between genuine and forged banknotes based on statistical features extracted from digital images.

The features were obtained by applying Wavelet Transform on images of banknotes. This dataset is commonly used for machine learning and deep learning classification tasks.

---

## Dataset Information

- **Dataset Name:** Banknote Authentication
- **Task:** Binary Classification
- **Number of Instances:** 1372
- **Number of Features:** 4
- **Number of Classes:** 2

---

## Features

| Feature | Description |
|---------|-------------|
| Variance | Variance of the Wavelet Transformed image |
| Skewness | Skewness of the Wavelet Transformed image |
| Curtosis | Kurtosis (Curtosis) of the Wavelet Transformed image |
| Entropy | Entropy of the image |

---

## Target Variable

| Class | Meaning |
|-------|---------|
| 0 | Genuine Banknote |
| 1 | Forged Banknote |

---

## Data Format

Each row in the dataset contains five comma-separated values:

```text
variance,skewness,curtosis,entropy,class
```

Example:

```text
3.6216,8.6661,-2.8073,-0.44699,0
4.5459,8.1674,-2.4586,-1.4621,0
3.866,-2.6383,1.9242,0.10645,0
```

---

## Source

The dataset was originally created by:

**Volker Lohweg**

University of Applied Sciences, Ostwestfalen-Lippe, Germany

It is publicly available through the UCI Machine Learning Repository and is widely used for educational and research purposes.

---

## License

This dataset is intended for educational and research use.

---

## Usage in this Experiment

This dataset is used to train and evaluate a **Perceptron** classifier for distinguishing genuine and forged banknotes.

The workflow includes:

- Data loading
- Data preprocessing
- Feature visualization
- Train-test split
- Perceptron model training
- Performance evaluation
- Decision boundary visualization
