# Experiment 5: Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

This repository folder contains the complete implementation and documentation for **Experiment 5** of the **CS3807 Deep Learning Laboratory** course at Shiv Nadar University Chennai.

The objective of this experiment is to systematically analyze the effects of various architectural and training decisions (weight initialization, regularization, optimization algorithms, hyperparameters, transfer learning, fine-tuning, and cross-validation) on image classification performance. The model architecture used is **MobileNetV2** and the dataset is the **Oxford-IIIT Pet Dataset**.

---

## Table of Contents
1. [What Has Been Done](#what-has-been-done)
2. [Folder Structure](#folder-structure)
3. [How to Run the Code](#how-to-run-the-code)
4. [Key Insights and Discussion](#key-insights-and-discussion)

---

## What Has Been Done

The experiment is divided into 10 structured tasks, implemented across three modular Jupyter Notebooks:

### 1. Dataset Setup & Preprocessing (Task 1)
- Loaded the **Oxford-IIIT Pet Dataset** using TensorFlow Datasets (`tfds`).
- Resized all images to a uniform $224 \times 224 \times 3$ size.
- Preprocessed pixels to the range `[-1, 1]` for compatibility with ImageNet-pretrained **MobileNetV2**.
- Structured splits: Training (3,680 images) and Test (3,669 images).

### 2. Weight Initialization Study (Task 2)
- Evaluated four initialization techniques on a baseline CNN:
  - **Zero Initialization** (fails to learn due to symmetry breaking issues).
  - **Random Normal Initialization** (sensitive to variance; can lead to exploding/vanishing activations).
  - **Xavier/Glorot Initialization** (scales weights dynamically for tanh-like activations).
  - **He Initialization** (optimal scaling for ReLU activations, preventing vanishing/exploding gradients).
- Saved comparison plots for **Training Loss vs. Epoch** (Plot 1) and **Validation Accuracy vs. Epoch** (Plot 2).

### 3. Regularization & Overfitting Analysis (Task 3)
- Evaluated overfitting mitigations on the CNN:
  - **No Regularization** (leads to a large generalization gap).
  - **L2 Regularization (Weight Decay)** (constrains weight magnitude).
  - **Dropout** (randomly disables activations during training to force redundant representations).
  - **Batch Normalization** (stabilizes hidden activation distributions).
- Plotted **Training vs. Validation Accuracy** (Plot 3) and **Training vs. Validation Loss** (Plot 4).

### 4. Batch Normalization (Task 4)
- **Numerical Verification:** Validated the batch mean, variance, normalization, and scaling equations mathematically and programmatically.
- **Empirical Evaluation:** Compared CNN training curves with and without Batch Normalization (Plot 5), demonstrating faster and more stable convergence when BN is enabled.

### 5. Optimizer Comparison (Task 5)
- Evaluated four optimization algorithms:
  - **SGD** (slow convergence, high oscillation).
  - **Momentum** (smooths updates by incorporating running averages of past gradients).
  - **RMSProp** (adapts learning rates based on moving averages of squared gradients).
  - **Adam** (combines Momentum and RMSProp, yielding the fastest, most stable training convergence).
- Documented performance in Plot 6 (Loss) and Plot 7 (Validation Accuracy).

### 6. Hyperparameter Tuning (Task 6)
- Conducted systematic sweeps on key hyperparameters to study their effects on validation accuracy:
  - **Learning Rate Sweep** (Plot 8): Showed that large learning rates overshoot minima, while small learning rates converge too slowly.
  - **Batch Size Sweep** (Plot 9): Assessed training stability and speed vs. hardware limits.
  - **Dropout Rate Sweep** (Plot 10): Found the optimal drop rate that balances underfitting and overfitting.

### 7. Transfer Learning & Fine-Tuning (Task 7)
- Leveraged pre-trained **MobileNetV2** weights on ImageNet:
  - **Case A: Feature Extraction:** Froze the entire MobileNetV2 base and trained only a new classifier head on top (10 epochs).
  - **Case B: Fine-Tuning:** Unfroze the top ~30 layers of the base model and trained them with a highly reduced learning rate ($1 \times 10^{-5}$) to adapt pre-trained features without catastrophic forgetting (10 epochs).
- Plotted comparisons of **Feature Extraction vs. Fine-Tuning** (Plot 11 & Plot 12).

### 8. 5-Fold Cross-Validation (Task 8)
- Performed **5-Fold Cross-Validation** on the full training set (3,680 images) across 4 configurations to select the best hyperparameter settings.
- Plotted mean accuracy with standard deviation error bars (Plot 13) to evaluate model consistency.

### 9. Final Model Evaluation (Task 9)
- Re-trained the selected best configuration from Task 8 on the **full training dataset** (including fine-tuning).
- Evaluated the model **once** on the untouched test set (3,669 images).
- Visualized results with a **Confusion Matrix** (Plot 14) and verified misclassified images (Plot 15) to understand model weaknesses.

### 10. Discussion & LaTeX Report (Task 10)
- Compiled complete mathematical calculations, plots, and analysis into the LaTeX document `experiment_5lat.tex`.

---

## Folder Structure

```
lab5/
├── .gitignore                          # Ignores the local fig/ directory containing plots
├── README.md                           # This file (explaining tasks and execution instructions)
├── dataset.md                          # Detailed breakdown of the Oxford-IIIT Pet Dataset
├── deep_5.ipynb                        # First notebook: Tasks 1-6 (Setup, Init, Reg, BN, Opt, LR, Batch Size)
├── CS3807_Exp5_Remaining_Tasks.ipynb   # Second notebook: Task 6 (Dropout) & Task 7 (Transfer Learning)
├── CS3807_Exp5_Tasks_8_9_10.ipynb      # Third notebook: Tasks 8-10 (5-Fold CV, Final Eval, Discussion Qs)
├── experiment_5lat.tex                 # LaTeX code for generating the final lab report
└── fig/                                # (Ignored locally) Subdirectory containing generated PNG plots
```

---

## How to Run the Code

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries. If running in **Google Colab** (highly recommended for access to GPUs), the packages can be installed directly inside the notebooks.

Required packages:
```bash
pip install -U protobuf tensorflow tensorflow-datasets tensorflow-metadata importlib_resources scikit-learn matplotlib seaborn
```

### 2. Notebook Execution Sequence
The notebooks are structured to run sequentially. Some notebooks save training outputs/checkpoints to **Google Drive** using the `save_result` helper function so they can be loaded in subsequent steps.

1. **`deep_5.ipynb`**
   - **What to do:** Mount Google Drive, install requirements, and run all cells.
   - **Scope:** Runs the data pipeline, weight initialization study, regularization comparisons, batch normalization checks, optimizer comparisons, and the batch size tuning sweep.

2. **`CS3807_Exp5_Remaining_Tasks.ipynb`**
   - **What to do:** Mount Google Drive and run all cells sequentially.
   - **Scope:** Picks up right after the batch size sweep. Evaluates dropout rates (Plot 10) and performs the transfer learning/fine-tuning sweep (Plots 11-12). Saves the resulting model to Drive.

3. **`CS3807_Exp5_Tasks_8_9_10.ipynb`**
   - **What to do:** Mount Google Drive and run all cells sequentially.
   - **Scope:** Performs the 5-fold cross-validation on the candidate configurations (Task 8), evaluates the best model on the test set, generates the confusion matrix (Task 9), and summarizes the overall results table (Task 10).

---

## Key Insights and Discussion

- **Why Weight Initialization Matters:** Poor initialization (e.g. zero initialization) prevents symmetry breaking, collapsing the capacity of a layer into a single neuron. He initialization works best with ReLU activations because it accounts for the zeroing-out of negative values.
- **Regularization Strategy:** Dropouts and L2 regularization successfully narrow the generalization gap (difference between train and validation accuracy), preventing the network from overfitting.
- **Optimizer Effectiveness:** **Adam** combines the advantages of adaptive learning rates (RMSProp) and momentum, offering the fastest and most stable descent path compared to plain SGD.
- **Fine-Tuning LR Constraint:** Fine-tuning the top layers of pre-trained MobileNetV2 must use a **much smaller learning rate** ($1 \times 10^{-5}$ vs $1 \times 10^{-3}$) to prevent "catastrophic forgetting" of the valuable pre-trained ImageNet features.
- **Cross-Validation Value:** Evaluating configurations on multiple folds prevents selecting a configuration that performs well on a single validation split purely by chance. Looking at both mean accuracy and standard deviation (SD) ensures you choose a model that is both highly accurate and consistent.
