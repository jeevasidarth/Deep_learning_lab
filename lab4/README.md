# Experiment 4: Comparative Study of Deep CNN Architectures Using Transfer Learning

This directory contains the implementation and analysis for Experiment 4: **Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning**.

## Overview of the Experiment
The goal of this experiment is to classify images from the **CIFAR-10** dataset using **VGG16** (pre-trained on ImageNet) through **Transfer Learning**.

The implementation is divided into two primary phases:
1. **Feature Extraction (Baseline Training):** 
   - We freeze all the weights of the pre-trained `VGG16` base model.
   - We append custom classification layers: a `GlobalAveragePooling2D` layer, a `Dense` layer with 256 units (`ReLU`), and a final `Dense` output layer with 10 units (`Softmax`).
   - We train only the custom classification head for **15 epochs** with a learning rate of `0.001` using `Adam` optimizer and `categorical_crossentropy` loss.
   
2. **Fine-Tuning:**
   - We unfreeze the last convolutional block of VGG16 (`block5`).
   - We re-compile the model with a much lower learning rate (`0.0001`) to prevent destroying the pre-trained features.
   - We continue training for **8 additional epochs**.

## Performance Results
- **Feature Extraction (15 epochs):** Achieved a best validation accuracy of **62.19%**.
- **Fine-Tuning (8 epochs):** Achieved a final test accuracy of **74.05%** (an improvement of **11.86%**).
- **Classification Metrics:**
  - Test Accuracy: **74.05%**
  - Precision: **74.02%**
  - Recall: **74.05%**
  - F1-Score: **73.91%**

## Project Structure
- `Deep_4.ipynb`: Jupyter notebook containing the full TensorFlow/Keras pipeline for loading, preprocessing, model definition, training, evaluation, and plotting.
- `Experiment_4.tex`: LaTeX source file for the experiment report.
- `Experiment_4_Lab_Manual.pdf`: Lab instruction manual.
- `img/`: Folder containing generated plots (ignored in Git).
  - `training_samples.png`: Grid of sample training images from CIFAR-10.
  - `confusion_matrix .eps`: Seaborn heatmap representing the confusion matrix.

## Prerequisites
Ensure you have the following installed:
- Python 3.10+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install the dependencies:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## How to Run
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Deep_4.ipynb
   ```
2. Run all cells sequentially to download CIFAR-10, train the model, fine-tune the model, and view the final classification reports and confusion matrices.
