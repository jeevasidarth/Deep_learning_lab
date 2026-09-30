# Experiment 7: Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders

## Overview
This repository folder contains the complete code, experimental report, visual results, and performance evaluation for **Experiment 7** of the Deep Learning Laboratory (CS3807) at Shiv Nadar University Chennai.

The experiment explores unsupervised representation learning, image reconstruction, noise reduction, and generative modeling using autoencoders on the **MNIST Handwritten Digit Dataset**.

---

## Repository Structure

```
lab7/
├── deep7.ipynb             # Jupyter notebook with complete implementation, training loops, and visualizations
├── Experiment_7.pdf        # Compiled LaTeX report detailing theoretical background, analysis, and findings
├── Experiment_7.tex        # Complete LaTeX source code for the lab report
├── README.md               # Main instructions and project documentation
├── dataset.md              # Detailed documentation of the MNIST dataset and preprocessing steps
└── fig/                    # High-resolution figures and evaluation output
    ├── png/                # Exported PNG plots (reconstructions, loss curves, latent spaces, samples)
    └── eps/                # EPS vector figures, consolidated_results.csv, and LaTeX tables
```

---

## Models Implemented

1. **Fully Connected Autoencoder (FC Autoencoder)**
   - **Encoder**: Dense 784 → 128 → 32 → 16 (Latent bottleneck)
   - **Decoder**: Dense 16 → 32 → 128 → 784
   - **Loss**: Binary Cross-Entropy / Mean Squared Error

2. **Convolutional Autoencoder (CAE)**
   - **Encoder**: Conv2D (32, 3x3) → MaxPool2D → Conv2D (16, 3x3) → MaxPool2D → Conv2D (8, 3x3) → Latent (8x8x8)
   - **Decoder**: Conv2D (8, 3x3) → UpSampling2D → Conv2D (16, 3x3) → UpSampling2D → Conv2D (32, 3x3) → Conv2D (1, 3x3, Sigmoid)
   - Preserves spatial structure and features superior reconstruction capabilities compared to dense architectures.

3. **Denoising Convolutional Autoencoder (DAE)**
   - Trained on corrupted inputs (Additive Gaussian noise $\sigma=0.3$ and Salt-and-Pepper noise $p=0.1$) with clean original images as ground truth targets.
   - Learns robust feature representations by projecting noisy inputs back onto the clean data manifold.

4. **Variational Autoencoder (VAE)**
   - **Probabilistic Bottleneck**: Encodes input into mean $\mu$ and log-variance $\log\sigma^2$ parameters of a Gaussian distribution.
   - **Reparameterization Trick**: $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
   - **Loss Function**: Combined Reconstruction Loss (BCE/MSE) + Kullback-Leibler (KL) Divergence penalty $\mathcal{D}_{KL}(q(z|x) \parallel p(z))$.
   - Enables smooth latent manifold interpolation and random generation of new handwritten digit images.

---

## Quantitative Results Comparison

Evaluating on the test set using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Structural Similarity Index Measure (SSIM):

| Model | MSE | MAE | SSIM | Parameters | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FC Autoencoder** | 0.01992 | 0.05442 | 0.7749 | 211,040 | 12.0 |
| **Convolutional Autoencoder (CAE)** | 0.00264 | 0.01554 | **0.9725** | 74,497 | 20.7 |
| **Denoising CAE** | 0.00444 | 0.02111 | 0.9480 | 74,497 | 17.8 |
| **Variational Autoencoder (VAE)** | 0.05474 | 0.12666 | 0.3661 | 134,165 | 26.9 |

---

## Prerequisites & Dependencies

To execute the notebook and run the code locally, ensure Python 3.8+ is installed along with the following packages:

```bash
pip install tensorflow numpy matplotlib pandas scikit-image scipy
```

---

## How to Run the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jeevasidarth/Deep_learning_lab.git
   cd Deep_learning_lab/lab7
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook deep7.ipynb
   ```
   *Alternatively, open `deep7.ipynb` in VS Code, JupyterLab, or upload to Google Colab.*

3. **Execute Cells Sequentially**:
   - Cell groups are structured logically:
     - **Data Preparation**: Loading MNIST dataset, normalizing pixel values $[0, 1]$, and applying corruption for denoising.
     - **Model Definitions**: Keras functional API implementations of FC Autoencoder, CAE, DAE, and VAE.
     - **Training Loops**: Model compilation with Adam optimizer and fitting over specified epochs.
     - **Evaluation & Visualizations**: Computing MSE, MAE, SSIM metrics and generating comparison grids saved to `fig/png/` and `fig/eps/`.

---

## Author & Acknowledgments
- **Author**: Jeeva Sidarth
- **Course**: CS3807 - Deep Learning Laboratory
- **Department**: Artificial Intelligence & Data Science, Shiv Nadar University Chennai
