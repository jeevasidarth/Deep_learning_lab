# Week 1 - Experiment 1: Perceptron Classifier

## Experiment Objectives

This experiment focuses on understanding and implementing the **Single-Layer Perceptron** algorithm for binary classification using the **Banknote Authentication Dataset**.

The objectives of this experiment are to:

- Understand the working principle of a Single-Layer Perceptron.
- Implement the Perceptron learning algorithm from scratch using **NumPy**.
- Perform Exploratory Data Analysis (EDA) on the dataset.
- Preprocess the data using **Min-Max Scaling**.
- Train the Perceptron model using gradient-based weight updates.
- Evaluate the model using standard classification metrics.
- Visualize the learning process through various plots and decision boundaries.
- Analyze the effect of different learning rates on convergence.

---

# Folder Structure

```
Lab-01-Perceptron/
│
├── perceptron.ipynb
├── perceptron.py
├── data_banknote_authentication.txt
├── DATASET.md
├── requirements.txt
├── report.pdf
├── README.md
│
└── figures/
    ├── Bias_vs_Epoch.eps
    ├── Boxplots.eps
    ├── Confusion_Matrix.eps
    ├── Decision_boundary.eps
    ├── Heatmap.eps
    ├── Histogram.eps
    ├── LearningRate_vs_Misclassified.eps
    ├── Misclassified_vs_Epoch.eps
    ├── Scatter.eps
    └── Weights_and_Bias_vs_Epoch.eps
```

---

# Dataset Information

**Dataset:** Banknote Authentication Dataset

The dataset contains statistical features extracted from Wavelet Transformed images of genuine and forged banknotes.

### Dataset Summary

- **Number of Samples:** 1372
- **Number of Features:** 4
- **Target Classes:** 2

| Class | Description |
|------|-------------|
| 0 | Genuine Banknote |
| 1 | Forged Banknote |

### Input Features

- Variance
- Skewness
- Curtosis
- Entropy

For more details, refer to **DATASET.md**.

---

# Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The experiment uses:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# Running the Experiment

There are two ways to execute this experiment.

## Option 1: Using the Python Script

Navigate to the experiment directory.

```bash
cd Lab-01-Perceptron
```

Run

```bash
python perceptron.py
```

The script performs the following:

- Loads the Banknote Authentication dataset
- Performs data preprocessing
- Normalizes the feature values
- Splits the dataset into training and testing sets
- Trains a Perceptron classifier
- Evaluates model performance
- Saves all generated figures inside the `figures/` directory

---

## Option 2: Using the Jupyter Notebook

Open the notebook.

```bash
jupyter notebook perceptron.ipynb
```

or open it directly using VS Code or JupyterLab.

Run the notebook sequentially from top to bottom.

---

# Workflow

The notebook is organized into the following sections.

### Task 1 — Exploratory Data Analysis

- Load the dataset
- Display dataset information
- Generate descriptive statistics
- Check for missing values

---

### Task 2 — Data Visualization

Generate visualizations including:

- Histograms
- Scatter plots
- Box plots
- Correlation heatmap

---

### Task 3 — Data Preprocessing

- Feature scaling using Min-Max Scaler
- Train-Test Split

---

### Task 4 — Perceptron Implementation

Implement the Perceptron algorithm from scratch:

- Weight initialization
- Bias initialization
- Forward propagation
- Step activation function
- Error computation
- Weight updates
- Bias updates

---

### Task 5 — Model Training

Train the Perceptron over multiple epochs while monitoring:

- Number of misclassified samples
- Weight updates
- Bias updates

---

### Task 6 — Model Evaluation

Evaluate the trained model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

### Task 7 — Model Analysis

Visualize the learning behavior of the Perceptron using:

- Bias vs Epoch
- Misclassified Samples vs Epoch
- Learning Rate vs Misclassified Samples
- Weight and Bias Evolution
- Decision Boundary
- Confusion Matrix

---

# Generated Outputs

Running the notebook/script generates the following visualizations.

| Figure | Description |
|---------|-------------|
| Histogram | Distribution of each feature |
| Scatter Plot | Relationship between selected features |
| Box Plot | Detection of feature spread and outliers |
| Heatmap | Feature correlation matrix |
| Decision Boundary | Learned decision boundary of the Perceptron |
| Confusion Matrix | Classification performance |
| Bias vs Epoch | Evolution of bias during training |
| Misclassified vs Epoch | Training convergence |
| Learning Rate vs Misclassified | Effect of learning rate |
| Weights and Bias vs Epoch | Parameter evolution during training |

All generated plots are stored inside the **figures/** directory.

---

# Learning Outcomes

After completing this experiment, you will be able to:

- Understand the Perceptron learning algorithm.
- Implement a binary classifier from scratch.
- Perform preprocessing and feature scaling.
- Analyze classification performance using evaluation metrics.
- Interpret training convergence through visualizations.
- Understand how learning rate influences model convergence.
