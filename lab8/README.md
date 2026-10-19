# Experiments 8 & 9: Effect of Attention Type and Attention Position in a Pretrained ResNet50

## Overview
This repository contains the complete implementation, experimental findings, quantitative ablation analysis, and qualitative interpretability visualizations for **Experiments 8 and 9** of the Deep Learning Laboratory (CS3807) at Shiv Nadar University Chennai.

The primary objective is to investigate how various **attention mechanisms** (Channel, Spatial, Sequential Channel-Spatial, Lightweight Self-Attention, ECA, CBAM, and Coordinate Attention) alter the internal representations of an ImageNet-pretrained **ResNet50** architecture, and to evaluate how the **insertion position** across the network's feature hierarchy governs 5-class flower classification performance on the **TensorFlow Flowers (`tf_flowers`)** dataset.

---

## Repository Structure

```
lab8/
├── lab8.ipynb                  # Complete Jupyter notebook with data pipelines, attention layer definitions, training loops, and visualizations
├── Experiment_8_9_Report.pdf   # Comprehensive 37-page laboratory experimental report and theoretical analysis
├── Academics_SNUC.pdf          # Official university lab assignment handout and question manual
├── README.md                   # Main instructions, model details, results, and execution guide
├── dataset.md                  # Detailed documentation of the tf_flowers dataset, stratified splits, and preprocessing
└── fig/                        # High-resolution vector figures (35 EPS plots covering all ablations, Grad-CAM, and maps)
    ├── plot1_sample_images.eps
    ├── plot2_A0_accuracy.eps
    ├── plot3_A0_loss.eps
    ├── plot4_A0_confusion.eps
    ├── plot5_type_accuracy.eps
    ├── plot6_type_f1.eps
    ├── plot7_type_acc_vs_params.eps
    ├── plot8_type_f1_vs_time.eps
    ├── plot9_type_acc_curves.eps
    ├── plot10_type_loss_curves.eps
    ├── plot11_position_accuracy.eps
    ├── plot12_position_f1.eps
    ├── plot13_multi_position.eps
    ├── plot14_order_accuracy.eps
    ├── plot15_kernel_size.eps
    ├── plot16_mech_accuracy.eps
    ├── plot17_mech_f1.eps
    ├── plot18_all_acc_vs_params.eps
    ├── plot19_all_f1_vs_time.eps
    ├── plot20_position_acc_curves.eps
    ├── plot21_mech_loss_curves.eps
    ├── plot22_feature_and_spatial_maps.eps
    ├── plot23_top_channels_before_after.eps
    ├── plot24_channel_weights.eps
    ├── plot25_self_attention.eps
    ├── plot26_gradcam_main.eps
    ├── plot27_gradcam_established.eps
    ├── plot28_position_maps.eps
    ├── plot29_internal_vs_gradcam.eps
    ├── plot30_confusion_comparison.eps
    ├── plot31_qualitative_comparison.eps
    ├── plot32_kernel_order_maps.eps
    ├── plot33_gradcam_stats.eps
    ├── plot34_channel_weight_distribution.eps
    └── plot35_perclass_f1_change.eps
```

---

## Attention Mechanisms Implemented

Every attention mechanism is constructed as a custom Keras Layer maintaining identical input-output tensor shapes:

1. **Channel Attention (Squeeze-and-Excitation / SE)**:
   - Squeezes spatial context via Global Average Pooling: $z_c = \frac{1}{HW}\sum_{i,j} F(i,j,c)$.
   - Excitation via two FC layers with reduction ratio $r=16$: $M_c(F) = \sigma(W_2 \, \text{ReLU}(W_1 z))$.
   - Re-scales feature maps: $F_c = F \odot M_c(F)$.

2. **Spatial Attention**:
   - Exploits inter-spatial relationships by computing channel-wise mean and max projections:
     $$M_s(F) = \sigma\left(\text{Conv}_{k \times k}([F_{\text{avg}} \,\|\, F_{\text{max}}])\right) \quad (\text{default } k=7)$$
   - Re-scales feature maps: $F_s = F \odot M_s(F)$.

3. **Sequential Channel-Spatial Attention (CBAM-style)**:
   - Evaluates sequential application: $F_{cs} = M_s(F_c) \odot F_c$.
   - The inverted sequence ($F_{sc} = M_c(F_s) \odot F_s$) was also benchmarked in the order ablation study.

4. **Lightweight Self-Attention**:
   - Project feature tensor $F \in \mathbb{R}^{H \times W \times C}$ into Query ($Q$) and Key ($K$) via $1 \times 1$ convs to $C/8$ channels, and Value ($V$) via $1 \times 1$ conv to $C$ channels.
   - Computes scaled dot-product attention map:
     $$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$
   - Residual output with learnable scalar $\gamma$ initialized to $0$: $F_{SA} = F + \gamma \, (AV)$.

5. **Established Attention Modules**:
   - **ECA (Efficient Channel Attention)**: Captures local cross-channel interaction using 1D convolution with adaptive kernel size $k=5$, without dimensionality reduction.
   - **CBAM (Convolutional Block Attention Module)**: Dual channel pooling (GAP + GMP) through shared MLP + $7 \times 7$ spatial conv.
   - **Coordinate Attention**: Disentangled direction-aware 1D pooling along height and width to preserve precise spatial positions.

6. **Combined Attention**:
   - Sequential chain of Channel $\to$ Spatial $\to$ Self-Attention.

---

## Candidate Insertion Positions in ResNet50

ResNet50 contains four residual stages whose spatial resolution decreases while channel depth increases:

| Position Tag | Stage Location | Output Resolution | Channels | Semantic Representation |
| :---: | :--- | :---: | :---: | :--- |
| **P1** | After `conv2_block3_out` | $56 \times 56$ | 256 | Low-level textures, edges, colors |
| **P2** | After `conv3_block4_out` | $28 \times 28$ | 512 | Intermediate local patterns, shapes |
| **P3** | After `conv4_block6_out` | $14 \times 14$ | 1024 | Mid-high semantic parts, petal structures |
| **P4** | After `conv5_block3_out` | $7 \times 7$ | 2048 | High-level class-discriminative semantics |

---

## Transfer Learning Protocol

All models follow a rigorous, strictly controlled two-stage training protocol:
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$)
- **Loss**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Stage 1 (Head + Attention Training)**: ResNet50 backbone frozen; only the newly added attention module and dense classification head (Dropout 0.3 + Dense 5) are trained for 10 epochs at $\text{LR} = 1 \times 10^{-3}$.
- **Stage 2 (Fine-Tuning)**: Unfreeze `conv5_x` stage (top residual block) and fine-tune end-to-end for 10 epochs at $\text{LR} = 1 \times 10^{-4}$.
- **Checkpointing**: The model weights corresponding to the minimum validation loss during fine-tuning are restored prior to test evaluation.

---

## Quantitative Results & Ablation Studies

All evaluations are conducted on the unaugmented test set of 551 images.

### 1. Main Performance Results

| Model | Position | Accuracy | Precision | Recall | Macro F1 | Total Parameters | Added Params | Inference (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (ResNet50)** | – | 0.9347 | 0.9385 | 0.9310 | 0.9340 | 23,597,957 | 0 | 8.154 |
| **Channel Attention (SE)** | P3 | 0.9256 | 0.9283 | 0.9243 | 0.9260 | 23,730,117 | +132,160 | 7.708 |
| **Spatial Attention** | P3 | 0.9292 | 0.9335 | 0.9278 | 0.9298 | 23,598,056 | +99 | 8.597 |
| **Channel+Spatial** | P2 | 0.9401 | 0.9462 | 0.9378 | 0.9412 | 23,631,368 | +33,411 | 8.478 |
| **Channel+Spatial (Best)**| **P3** | **0.9510** | 0.9525 | **0.9513** | **0.9513** | 23,730,216 | +132,259 | 8.328 |
| **Channel+Spatial** | P4 | 0.9347 | 0.9414 | 0.9309 | 0.9348 | 24,124,520 | +526,563 | 8.172 |
| **Self-Attention** | **P3** | **0.9510** | **0.9534** | 0.9509 | **0.9518** | 24,908,678 | +1,310,721 | 7.551 |
| **ECA** | P3 | 0.9292 | 0.9293 | 0.9321 | 0.9288 | 23,597,962 | +5 | 7.911 |
| **CBAM** | P3 | 0.9274 | 0.9397 | 0.9215 | 0.9277 | 23,730,216 | +132,259 | 8.262 |
| **Coordinate Attention** | P3 | 0.9147 | 0.9215 | 0.9123 | 0.9150 | 23,729,960 | +132,003 | 8.450 |
| **Combined (Ch+Sp+SA)** | P3 | 0.9201 | 0.9210 | 0.9207 | 0.9207 | 25,040,937 | +1,442,980 | 8.581 |

### 2. Attention Position Ablation (Channel $\to$ Spatial)

| Position | Resolution | Accuracy | Macro F1 | Added Params | Inference (ms/img) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | $56 \times 56 \times 256$ | 0.9328 | 0.9323 | +8,563 | 7.986 |
| **P2** | $28 \times 28 \times 512$ | 0.9401 | 0.9412 | +33,411 | 8.478 |
| **P3** | $14 \times 14 \times 1024$ | **0.9510** | **0.9513** | +132,259 | 8.328 |
| **P4** | $7 \times 7 \times 2048$ | 0.9347 | 0.9348 | +526,563 | 8.172 |

### 3. Single vs. Multi-Stage Attention Insertion

| Stages | Accuracy | Macro F1 | Added Params | Total Params |
| :---: | :---: | :---: | :---: | :---: |
| **P3 only** | **0.9510** | **0.9513** | +132,259 | 23,730,216 |
| **P4 only** | 0.9347 | 0.9348 | +526,563 | 24,124,520 |
| **P3 + P4** | 0.9292 | 0.9287 | +658,822 | 24,256,779 |
| **P2 + P3 + P4** | 0.9310 | 0.9308 | +692,233 | 24,290,190 |

### 4. Attention Ordering & Spatial Kernel Size Ablations
- **Order Study at P3**:
  - `Channel → Spatial`: **95.10%** accuracy (F1: 0.9513)
  - `Spatial → Channel`: **93.28%** accuracy (F1: 0.9320)
- **Spatial Kernel Size $k$ at P3**:
  - $k = 3 \times 3$: 92.92% accuracy
  - $k = 5 \times 5$: 90.56% accuracy
  - $k = 7 \times 7$: **95.10%** accuracy

---

## Key Findings & Conclusions

1. **Position Matters More Than Mechanism**: Attention inserted at intermediate stage **P3** achieves superior performance across mechanisms because feature representations possess sufficient semantic abstraction while retaining adequate spatial structure ($14 \times 14$). Early stages (P1) lack semantic relevance, while late stages (P4, $7 \times 7$) have degraded spatial resolution.
2. **More Attention Does Not Equal Better Performance**: Multi-stage insertions ($P_3+P_4$, $P_2+P_3+P_4$) and complex cascades (Combined) introduce excessive unregularized parameters, causing validation loss degradation on small datasets ($N=2,569$).
3. **Internal Attention Maps vs. Grad-CAM**: Internal spatial attention maps show where the layer selectively gates activations, whereas Grad-CAM reflects class-discriminative gradients backpropagated from the output.

---

## Prerequisites & Installation

To run the notebook locally or on Google Colab, ensure Python 3.9+ and the required dependencies are installed:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn scipy pypdfium2
```

---

## How to Run the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jeevasidarth/Deep_learning_lab.git
   cd Deep_learning_lab/lab8
   ```

2. **Open the Notebook**:
   - **Google Colab**: Upload `lab8.ipynb` to Google Colab and select a GPU runtime (Runtime $\to$ Change runtime type $\to$ T4 GPU).
   - **Local Environment / JupyterLab**:
     ```bash
     jupyter notebook lab8.ipynb
     ```

3. **Step-by-Step Execution**:
   - **Cell 1–3**: Environment configuration, figure styling, dataset download (`tf_flowers`), and stratified train/val/test splitting.
   - **Cell 4–5**: `tf.data` input pipeline creation with data augmentation, sample visualization (Plot 1).
   - **Cell 6–8**: ResNet50 stage-submodel partitioning and baseline hyperparameter tuning.
   - **Cell 9–13**: Training and evaluation of baseline model (A0), loss/accuracy curves (Plots 2–3), and confusion matrix (Plot 4).
   - **Cell 14–20**: Custom attention layers definition, unit testing, and attention-type ablation (A1–A5).
   - **Cell 21–31**: Insertion position ablations (P1–P4), multi-stage, order, kernel, and established mechanisms (ECA, CBAM, Coordinate), followed by Grad-CAM and internal map exports to `fig/`.

---

## Author & Acknowledgments
- **Author**: Jeeva Sidarth
- **Degree**: B.Tech Artificial Intelligence & Data Science
- **Course**: CS3807 – Deep Learning Laboratory
- **Institution**: Shiv Nadar University Chennai
