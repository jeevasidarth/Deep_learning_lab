# Dataset Documentation: TensorFlow Flowers (`tf_flowers`)

## Overview
This document provides comprehensive documentation for the **TensorFlow Flowers** (`tf_flowers`) dataset used in **Experiments 8 and 9** of the Deep Learning Laboratory (CS3807) at Shiv Nadar University Chennai. The dataset is employed to evaluate the impact of various attention mechanisms (Channel, Spatial, Sequential Channel-Spatial, Self-Attention, ECA, CBAM, Coordinate Attention) and their insertion positions across an ImageNet-pretrained **ResNet50** architecture.

---

## Dataset Specifications

| Property | Details |
| :--- | :--- |
| **Dataset Name** | TensorFlow Flowers (`tf_flowers`) |
| **Origin / Source** | Google Creative Lab / TensorFlow Datasets |
| **Download Source** | `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz` |
| **Task** | 5-Class Natural Image Classification |
| **Classes** | Daisy, Dandelion, Roses, Sunflowers, Tulips |
| **Total Images** | 3,670 RGB images |
| **Original Resolution** | Variable (ranging from $320 \times 240$ to $1024 \times 768$) |
| **Target Resolution** | $224 \times 224 \times 3$ pixels |
| **Color Channels** | 3 (RGB) |
| **Format** | JPEG |

---

## Class Distribution & Stratified Splits

To ensure scientifically valid and reproducible ablation studies, a single stratified split ($70\% : 15\% : 15\%$) was generated once using a fixed seed (`seed = 42`) and saved to persistent storage. All experiments and ablation variants evaluate on the exact same train, validation, and test subsets.

| Class Index | Class Name | Train Samples ($70\%$) | Validation Samples ($15\%$) | Test Samples ($15\%$) | Total Samples |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | **daisy** | 443 | 95 | 95 | 633 |
| 1 | **dandelion** | 629 | 134 | 135 | 898 |
| 2 | **roses** | 449 | 96 | 96 | 641 |
| 3 | **sunflowers** | 489 | 105 | 105 | 699 |
| 4 | **tulips** | 559 | 120 | 120 | 799 |
| **Total** | **5 Classes** | **2,569** | **550** | **551** | **3,670** |

---

## Dataset Challenges & Characteristics

The `tf_flowers` dataset represents an ideal benchmark for analyzing attention mechanisms in computer vision due to several intrinsic visual complexities:

1. **Intra-class Variance**: Significant differences in flower maturity, blooming stages, color mutations (e.g., yellow vs. red tulips), and multi-flower clusters vs. isolated single blooms.
2. **Inter-class Similarity**: High visual and morphological resemblance between classes, specifically young daisies and dandelions, or closed rosebuds and tulips.
3. **Background Clutter & Natural Noise**: Natural outdoor backgrounds containing soil, grass, leaves, fences, sunlight reflections, and insect pollinators that require the network to suppress irrelevant context.
4. **Scale & Viewpoint Variations**: Close-up macro shots highlighting fine petal/pistil textures alongside wide-angle field perspectives with complex depth of field and lens blur.

---

## Data Preprocessing Pipeline

The data ingestion pipeline in `lab8.ipynb` follows standard deep learning practices for transfer learning on ResNet50:

### 1. Image Resizing & Formatting
- All images are decoded from JPEG format, resized to a standardized spatial resolution of $224 \times 224$ pixels using bilinear interpolation, and stored as `uint8` tensors.

### 2. ResNet50 Preprocessing
- Pixel values are converted using `tf.keras.applications.resnet50.preprocess_input`:
  - Zero-centers each color channel with respect to the ImageNet dataset mean:
    $$\mu = [103.939, 116.779, 123.68] \quad (\text{BGR order})$$
  - Converts RGB input to BGR format without scale normalization to $[0, 1]$.

### 3. Data Augmentation (Training Split Only)
Data augmentation is applied exclusively to the training set dynamically via `tf.data.Dataset` mapping to mitigate overfitting:
- **Random Horizontal Flip**: `p = 0.5`
- **Random Rotation**: Uniform in range $[-18^\circ, +18^\circ]$ ($\pm 5\%$ of $360^\circ$)
- **Random Zoom**: Up to $\pm 10\%$ scale shift
- **Random Translation**: Horizontal and vertical shifts up to $\pm 10\%$
- **Contrast Adjustment**: Factor range $[0.9, 1.1]$

The validation and test pipelines receive unaugmented images with deterministic ImageNet preprocessing.

---

## Dataset Loading Code Example

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Stratified split partitions: train_ds, val_ds, test_ds
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def prepare_pipeline(x_train, y_train, x_val, y_val, x_test, y_test):
    # Train pipeline with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = (train_ds
                .shuffle(buffer_size=1024, seed=42)
                .map(augment, num_parallel_calls=AUTOTUNE)
                .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

    # Test/Val pipelines (deterministic)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = (test_ds
               .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(AUTOTUNE))
    
    return train_ds, test_ds
```
