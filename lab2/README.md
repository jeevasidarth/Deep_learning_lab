# Week 2 - Experiment 2: Multi-Layer Perceptron (MLP) Classifier

## Experiment Objectives

This experiment focuses on understanding, implementing, and optimizing a **Multi-Layer Perceptron (MLP)** algorithm for multi-class image classification using the **Fashion-MNIST Dataset** with **TensorFlow / Keras**.

The objectives of this experiment are to:

- Understand the working principle and architecture of a Multi-Layer Perceptron (MLP).
- Perform Exploratory Data Analysis (EDA) on an image dataset.
- Preprocess image data by flattening $28 \times 28$ matrices into 784-dimensional vectors and normalizing pixel values to $[0, 1]$.
- Convert integer class labels into one-hot encoded vectors for categorical cross-entropy loss.
- Construct and train a baseline MLP model with Keras.
- Perform automated hyperparameter optimization using **RandomizedSearchCV** and **SciKeras**.
- Evaluate model performance using standard classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Visualize training progress, search results, and model performance comparisons through various plots.

---

# Folder Structure

```
lab2/
│
├── MLP.ipynb
├── requirements.txt
├── DATASET.md
├── README.md
├── Experiment_2.tex
├── Experiment_2_Lab_Manual.pdf
│
└── figures/
    ├── Sample_Images.eps
    ├── Class_Distribution.eps
    ├── Training_Accuracy.eps
    ├── Validation_Accuracy.eps
    ├── Training_Loss.eps
    ├── Validation_Loss.eps
    ├── Confusion_Matrix_Baseline.eps
    ├── Confusion_Matrix_Optimized.eps
    ├── Hyperparameter_Search_Results.eps
    └── Model_Comparison.eps
```

---

# Dataset Information

**Dataset:** Fashion-MNIST Dataset

The dataset contains $28 \times 28$ grayscale images of Zalando's article products across 10 distinct categories.

### Dataset Summary

- **Training Samples:** 60,000
- **Testing Samples:** 10,000
- **Number of Features:** 784 ($28 \times 28$ pixels)
- **Target Classes:** 10

| Class | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

For more details, refer to **DATASET.md**.

---

# Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The experiment uses:

- TensorFlow / Keras
- SciKeras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

# Running the Experiment

## Using the Jupyter Notebook

Open the notebook in Jupyter Notebook, JupyterLab, or VS Code:

```bash
jupyter notebook MLP.ipynb
```

Run the notebook sequentially from top to bottom.

---

# Workflow

The notebook is organized into the following sections:

### Task 1 — Exploratory Data Analysis & Visualization

- Load Fashion-MNIST dataset
- Print dataset dimensions and class count
- Display ten sample images from the training dataset
- Plot class distribution across training samples

---

### Task 2 — Data Preprocessing

- Reshape $28 \times 28$ images into 784-dimensional flat feature vectors
- Normalize pixel values to $[0, 1]$
- One-hot encode integer labels into 10-class target vectors

---

### Task 3 — Model Construction

Build baseline MLP model architecture:
- Input layer ($784$)
- Dense layer (128 units, ReLU activation)
- Dense layer (64 units, ReLU activation)
- Output layer (10 units, Softmax activation)
- Display `model.summary()`

---

### Task 4 — Model Training

Compile and train the baseline model:
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Metric: Accuracy
- Train for 20 epochs with batch size 32
- Plot training/validation accuracy and loss curves

---

### Task 5 — Model Evaluation

Evaluate baseline model on test dataset:
- Compute Accuracy, Precision, Recall, and F1-score
- Generate Confusion Matrix and Classification Report

---

### Task 6 & 7 — Hyperparameter Optimization & Model Analysis

- Define tunable MLP builder function and search space (hidden layers, hidden neurons, learning rate, optimizer, activation function, dropout rate, batch size, epochs)
- Perform 5-fold cross-validation with `RandomizedSearchCV` via `SciKeras`
- Record best hyperparameters and plot search trial results
- Retrain final model on full training set using optimal parameters
- Evaluate optimized model on testing dataset
- Plot confusion matrix for optimized model and performance comparison bar chart (Baseline vs Optimized)

---

# Generated Outputs

Running the notebook generates the following visualizations inside the **figures/** directory:

| Figure | Description |
|--------|-------------|
| Sample Images | Display of ten sample Fashion-MNIST images |
| Class Distribution | Bar chart showing class balance in training set |
| Training Accuracy | Training accuracy evolution over 20 epochs |
| Validation Accuracy | Validation accuracy evolution over 20 epochs |
| Training Loss | Training loss reduction over 20 epochs |
| Validation Loss | Validation loss convergence over 20 epochs |
| Confusion Matrix Baseline | Confusion matrix for baseline MLP classifier |
| Confusion Matrix Optimized | Confusion matrix for hyperparameter-tuned MLP classifier |
| Hyperparameter Search Results | Trial-wise mean CV accuracy curve from RandomizedSearchCV |
| Model Comparison | Bar chart comparing Baseline vs Optimized model metrics |

All generated plots are stored inside the **figures/** directory.

---

# Learning Outcomes

After completing this experiment, you will be able to:

- Understand the Multi-Layer Perceptron (MLP) architecture and activation functions.
- Preprocess image datasets through flattening, normalization, and one-hot encoding.
- Build, train, and evaluate deep neural networks using TensorFlow/Keras.
- Implement automated hyperparameter tuning using `RandomizedSearchCV` and SciKeras wrappers.
- Evaluate multi-class classification models using standard metrics and confusion matrices.
- Compare baseline and hyperparameter-tuned neural network performance.
