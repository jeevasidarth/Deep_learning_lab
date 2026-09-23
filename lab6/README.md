# Lab 6: End-to-End Study of RNN, LSTM, and GRU for Sequence Learning and Video Understanding

## CS3807 -- Deep Learning Laboratory (Experiment 6)
**Department of Artificial Intelligence & Data Science**  
**Shiv Nadar University, Chennai**

---

## 📌 Overview

This repository contains the complete codebase, data pipelines, model architectures, and evaluation suites for **Experiment 6: End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding**.

The experiment covers:
1. **Sensor Sequence Classification**: Comparing Vanilla Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) on tri-axial inertial sensor data from the **UCI Human Activity Recognition (HAR)** dataset.
2. **Backpropagation Through Time (BPTT) & Sequence Length Sensitivity**: Investigating gradient behavior, convergence stability, and performance across varying temporal window lengths ($T \in \{32, 64, 128\}$).
3. **Video Action Recognition**: Building a hybrid **CNN + Recurrent Architecture** (MobileNetV2 feature extractor combined with LSTM/GRU) on UCF101 video frame sequences, complemented by a **Temporal Order Shuffle Test**.
4. **Sequence-to-Sequence (Seq2Seq) Modeling**: Implementing an Encoder-Decoder LSTM network with **Teacher Forcing** for sequence reversal and evaluating token vs. sequence accuracy.

---

## 📁 Repository Structure

```
lab6/
├── deep6.ipynb                # Primary Jupyter Notebook containing all 7 parts
├── Experiment_6(1).tex        # Complete LaTeX source for the experiment report
├── Experiment_6.pdf           # Compiled PDF report
├── README.md                  # Project overview and execution instructions
├── dataset.md                 # Detailed documentation of datasets used
└── fig/                       # Generated plots and visualizations
    ├── png/                   # PNG formatted figures
    └── eps/                   # Vector EPS figures (embedded in LaTeX report)
```

---

## ⚙️ Requirements & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab (GPU recommended for Part 6 Video Understanding)

### Required Libraries
Install the required dependencies using `pip`:

```bash
pip install numpy matplotlib pandas scikit-learn tensorflow
```

---

## 🚀 Execution Guide

### Running via Jupyter Notebook / Google Colab

1. Launch Jupyter Notebook or upload `deep6.ipynb` to Google Colab:
   ```bash
   jupyter notebook deep6.ipynb
   ```

2. Executing the Notebook:
   - **Part 1 (Data Setup & Preprocessing)**: Downloads and normalizes the UCI HAR dataset. Saves processed arrays to `data/har_prepared.npz`.
   - **Part 2 (Numerical Forward Pass Verification)**: Verifies step-by-step hidden state computations ($h_1, h_2, h_3$) comparing NumPy against Keras.
   - **Part 3 (Model Building & Training)**: Trains Vanilla RNN, LSTM, and GRU models with identical hyperparameters (32 units, Dropout 0.2, Adam optimizer, 30 epochs, batch size 32).
   - **Part 4 (Test Evaluation & Comparison)**: Evaluates trained models on the test set, generates confusion matrices, and prints performance comparison tables.
   - **Part 5 (Sequence Length Study)**: Trains models on truncated sequence lengths ($T = 32, 64, 128$) to evaluate BPTT behavior and classification accuracy.
   - **Part 6 (Video Action Recognition)**: Downloads selected UCF101 classes, extracts MobileNetV2 frame features, trains CNN-LSTM / CNN-GRU classifiers, and performs the temporal shuffle sanity check.
   - **Part 7 (Seq2Seq Sequence Reversal)**: Generates synthetic sequence reversal data, trains an Encoder-Decoder LSTM with Teacher Forcing, and evaluates inference performance with greedy decoding.

---

## 🔬 Model Architectures & Summary

| Task / Part | Input Shape | Model Architecture | Key Parameters |
| :--- | :--- | :--- | :--- |
| **Part 3: HAR Classification** | $(T, 9)$ where $T=128$ | Input $\rightarrow$ SimpleRNN / LSTM / GRU (32 units) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(16, ReLU) $\rightarrow$ Dense(6, Softmax) | 32 hidden units, Adam ($10^{-3}$), Categorical Cross-Entropy |
| **Part 5: Seq Length Study** | $(T, 9), T \in \{32, 64, 128\}$ | Same as Part 3 across truncated sequence steps | Sequence truncation comparison |
| **Part 6: Video Understanding** | $(10, 1280)$ | MobileNetV2 (Frozen Backbone) $\rightarrow$ LSTM / GRU (32 units) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(4, Softmax) | 10 frames/video, MobileNetV2 1280-dim feature vector |
| **Part 7: Seq2Seq Reversal** | $(L, 1)$ discrete tokens | Encoder LSTM (32 units) $\rightarrow$ Context State $(h, c)$ $\rightarrow$ Decoder LSTM (32 units) | Teacher Forcing during training; Greedy Decoding during inference |

---

## 📊 Key Results Summary

- **HAR Activity Recognition**: GRU and LSTM demonstrate superior convergence stability and higher test F1-scores compared to Vanilla RNN, which suffers from vanishing gradients over $T=128$ steps.
- **Sequence Length Sensitivity**: As sequence length $T$ increases from 32 to 128, LSTM/GRU effectively exploit longer context, whereas Vanilla RNN accuracy plateaus or degrades due to BPTT limitations.
- **Video Understanding**: CNN-LSTM achieves high classification accuracy on UCF101 video clips. Shuffling frame order temporally degrades accuracy, confirming the model relies on sequential temporal dynamics.
- **Seq2Seq Reversal**: Encoder-Decoder LSTM achieves high token and sequence-level accuracy when reversing input token sequences.

---

## 📜 References & Citation

- **UCI HAR Dataset**: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013.
- **UCF101 Dataset**: Khurram Soomro, Amir Roshan Zamir and Mubarak Shah. *UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild*. CRCV-TR-12-01, 2012.
