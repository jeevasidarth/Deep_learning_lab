# Deep Learning Laboratory

A collection of Deep Learning lab experiments, assignments, and implementations covering neural networks, MLP classifiers, CNNs, optimization techniques, and practical deep learning applications using TensorFlow, Keras, NumPy, Pandas, Scikit-learn, and Matplotlib.

## Repository Structure

The experiments in this repository are organized as follows:

### 📂 [lab1](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab1) - Experiment 1: Single-Layer Perceptron
- **Topic:** Implementation and training of a Single-Layer Perceptron.
- **Tasks:**
  - Logic gates representation (AND, OR, NAND, NOR) and analysis of linear separability limits (e.g. XOR gate).
  - Classification of the banknote authentication dataset (`data_banknote_authentication.txt`).
  - Parameter visualization: weights and bias convergence over epochs.
- **Technologies:** NumPy, Pandas, Matplotlib.

### 📂 [lab2](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab2) - Experiment 2: Multi-Layer Perceptron (MLP) Classifier
- **Topic:** Multi-class classification using MLPs.
- **Dataset:** Fashion-MNIST.
- **Tasks:**
  - Build baseline MLP architecture for multi-class clothing item classification.
  - Hyperparameter optimization using `RandomizedSearchCV` via `SciKeras`.
  - Performance evaluation using accuracy, precision, recall, F1-scores, and confusion matrices.
- **Technologies:** TensorFlow, Keras, SciKeras, Scikit-learn.

### 📂 [lab3](file:///c:/Users/sjeev/Documents/Sem5/Deeplearn_lab/lab3) - Experiment 3: Convolutional Neural Networks (CNNs) for Image Classification
- **Topic:** Image classification using CNNs, parameter calculations, and pooling methods.
- **Dataset:** CIFAR-10.
- **Tasks:**
  - Investigate the effect of kernel sizes, strides, and padding options.
  - Extract and visualize intermediate feature maps (activations).
  - Train and compare CNN architectures using **Max Pooling** vs. **Average Pooling**.
  - Detailed classification report and diagonal confusion matrix analysis.
- **Technologies:** TensorFlow, Keras, Matplotlib, Seaborn.

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jeevasidarth/Deep_learning_lab.git
   cd Deep_learning_lab
   ```

2. **Run an experiment:**
   Navigate to the respective lab folder (e.g., `/lab3`) and check the local `README.md` for specific execution instructions.
