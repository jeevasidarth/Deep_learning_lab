# Week 1 - Experiment 1: Perceptron Classifier & Logic Gates

## Experiment Objectives

This experiment focuses on understanding and implementing the **Single-Layer Perceptron** algorithm for binary classification across two core tasks:

1. **Logic Gates Classification ([logicgates.ipynb](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/logicgates.ipynb))**: Implementing Perceptron models to simulate fundamental logic operations (**AND**, **OR**, and **NOT** gates) and visualizing their decision boundaries across training epochs to demonstrate linear separability.
2. **Banknote Authentication Classification ([perceptronlab1.ipynb](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/perceptronlab1.ipynb))**: Building a binary classifier from scratch to distinguish genuine banknotes from forged ones using statistical features extracted from Wavelet Transformed images.

The primary learning objectives are:
- Understand the mathematical foundation of a Single-Layer Perceptron.
- Implement the step activation function and Perceptron weight update rule from scratch using **NumPy**.
- Demonstrate linear separability by training Perceptrons on logic gates (**AND**, **OR**, **NOT**).
- Perform Exploratory Data Analysis (EDA) and feature scaling (**Min-Max Scaling**) on real-world datasets.
- Train the Perceptron model iteratively, tracking misclassification errors and weight/bias updates over epochs.
- Evaluate classification performance using Accuracy, Precision, Recall, F1 Score, and Confusion Matrices.
- Visualize learned decision boundaries and parameter convergence.

---

# Folder Structure

```
lab1/
│
├── logicgates.ipynb                      # Perceptron implementation for AND, OR, and NOT logic gates
├── perceptronlab1.ipynb                  # Perceptron classifier for Banknote Authentication dataset
├── data_banknote_authentication.txt.txt  # Banknote dataset source file
├── DATASET.md                            # Detailed documentation of the Banknote dataset
├── requirements.txt                      # Project dependencies
├── lab_1_perceptron (1).pdf              # Lab experiment reference manual
├── README.md                             # Experiment documentation
│
└── figures/                              # Generated visualizations and EPS plots
    ├── AND_Epoch_*.eps
    ├── OR_Epoch_*.eps
    ├── NOT_Epoch_*.eps
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

# Dataset & Task Summary

### 1. Logic Gates Task ([logicgates.ipynb](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/logicgates.ipynb))
Synthetically generated truth table datasets for basic logic gates:
- **AND Gate:** 2 inputs ($x_1, x_2$), Output $y=1$ only when $x_1=1$ and $x_2=1$.
- **OR Gate:** 2 inputs ($x_1, x_2$), Output $y=1$ if at least one input is $1$.
- **NOT Gate:** 1 input ($x_1$), Output $y=1$ when $x_1=0$.

### 2. Banknote Authentication Task ([perceptronlab1.ipynb](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/perceptronlab1.ipynb))
- **Number of Samples:** 1372
- **Number of Features:** 4 (Variance, Skewness, Curtosis, Entropy)
- **Target Classes:** 2 (0: Genuine Banknote, 1: Forged Banknote)

For full dataset specs, refer to [DATASET.md](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/DATASET.md).

---

# Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Core libraries used:
- **NumPy** — Matrix operations and weight updates
- **Pandas** — Data loading and manipulation
- **Matplotlib & Seaborn** — Decision boundary plots and EDA visualizations
- **Scikit-learn** — Data splitting and evaluation metrics

---

# Running the Experiment

There are two primary Jupyter Notebooks for this experiment:

## 1. Logic Gates Implementation ([logicgates.ipynb](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/logicgates.ipynb))

Launch the notebook via VS Code, JupyterLab, or Jupyter Notebook:

```bash
jupyter notebook logicgates.ipynb
```

**Notebook Workflow:**
1. Defines truth tables for **AND**, **OR**, and **NOT** logic gates.
2. Implements a Step Activation Function ($f(z) = 1 \text{ if } z \ge 0 \text{ else } 0$).
3. Implements Perceptron forward pass and training loop with learning rate $\alpha = 0.5$.
4. Computes weight updates: $w \leftarrow w + \alpha \cdot (y - \hat{y}) \cdot x$.
5. Plots and exports 2D decision boundary figures (`.eps`) for each training epoch until full convergence.

---

## 2. Banknote Authentication Perceptron ([perceptronlab1.ipynb](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/perceptronlab1.ipynb))

Launch the notebook:

```bash
jupyter notebook perceptronlab1.ipynb
```

**Notebook Workflow:**
1. **EDA & Visualization:** Loads banknote dataset, inspects feature statistics, plots histograms, scatter plots, boxplots, and correlation heatmaps.
2. **Preprocessing:** Applies Min-Max Feature Scaling and splits data into training/testing sets.
3. **Perceptron Training:** Trains a binary Perceptron classifier across multiple epochs.
4. **Evaluation:** Measures Accuracy, Precision, Recall, F1 Score, and plots Confusion Matrix.
5. **Model Analysis:** Visualizes decision boundaries, parameter evolution (Weights and Bias vs. Epoch), misclassification convergence, and learning rate sensitivity.

---

# Key Output Visualizations

| Category | Visualizations | Description |
|----------|----------------|-------------|
| **Logic Gates** | `AND_Epoch_*.eps` | Epoch-by-epoch decision boundary convergence for 2-input AND gate |
| | `OR_Epoch_*.eps` | Epoch-by-epoch decision boundary convergence for 2-input OR gate |
| | `NOT_Epoch_*.eps` | Decision boundary convergence for 1-input NOT gate |
| **Banknote EDA** | `Histogram.eps` | Distribution of banknote wavelet features |
| | `Scatter.eps` & `Boxplots.eps` | Feature relationships, spread, and outlier detection |
| | `Heatmap.eps` | Feature correlation matrix |
| **Banknote Performance** | `Decision_boundary.eps` | Learned linear boundary separating genuine and forged banknotes |
| | `Confusion_Matrix.eps` | Classification performance matrix |
| | `Misclassified_vs_Epoch.eps` | Perceptron convergence plot over epochs |
| | `LearningRate_vs_Misclassified.eps` | Impact of learning rate variations on model convergence |
| | `Weights_and_Bias_vs_Epoch.eps` | Trajectory of weights and bias parameters during training |

All generated figures are stored in the [figures](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1/figures) directory.

---

# Learning Outcomes

By completing this experiment, you will learn to:
- Formulate and code a Single-Layer Perceptron algorithm from scratch.
- Understand linear separability by demonstrating how Perceptrons solve AND, OR, and NOT gates.
- Implement step activation functions and error-driven weight update rules.
- Preprocess real-world dataset features using Min-Max normalization.
- Analyze model convergence through misclassification plots, decision boundary visualizer, and parameter tracking.
