# Oxford-IIIT Pet Dataset

This document provides a detailed overview of the dataset used in **Experiment 5** for training, evaluating, and fine-tuning the CNN models (specifically using the MobileNetV2 architecture).

---

## 1. Dataset Overview
The **Oxford-IIIT Pet Dataset** is a fine-grained image classification dataset consisting of 37 categories of pets (breeds of cats and dogs). The dataset was created by the Visual Geometry Group (VGG) at the University of Oxford in 2012.

- **Total Images:** 7,349 images
- **Number of Classes:** 37
  - **Cat Breeds:** 12 classes (approx. 2,400 images)
  - **Dog Breeds:** 25 classes (approx. 4,950 images)
- **Distribution:** Approximately 200 images per class.
- **Variations:** The dataset is highly challenging due to substantial variations in scale, pose, lighting, and background.

---

## 2. Breed Classification (Classes)

Below are the 37 breeds present in the dataset:

| Category | Breeds |
| :--- | :--- |
| **Cats (12 Breeds)** | Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx. |
| **Dogs (25 Breeds)** | American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle, Boxer, Chihuahua, English Cocker Spaniel, English Setter, German Shorthaired Pointer, Great Pyrenees, Havanese, Japanese Chin, Keeshond, Leonberger, Miniature Pinscher, Newfoundland, Pomeranian, Pug, Saint Bernard, Samoyed, Scottish Terrier, Shiba Inu, Staffordshire Bull Terrier, Wheaten Terrier, Yorkshire Terrier. |

---

## 3. Data Split
For this experiment, the dataset is structured and split into:
* **Training Set:** 3,680 images (used for model fitting and hyperparameter sweeps).
* **Validation Set:** Subsplit from the training data (typically 10-20%) used to monitor overfitting and guide hyperparameter tuning.
* **Test Set:** 3,669 images (kept completely untouched during training and tuning, used only for the single final evaluation).

During the **5-Fold Cross-Validation** stage (Task 8), the entire full training split (3,680 images) is split dynamically into 5 folds (using `KFold` indexing) to evaluate model configuration consistency.

---

## 4. Preprocessing and Augmentation
To make the dataset compatible with the **MobileNetV2** architecture:
1. **Resizing:** All images are resized to a uniform spatial dimension of **$224 \times 224 \times 3$** (RGB).
2. **Normalization:** Input images are normalized so that pixel values lie in the range **`[-1, 1]`**, matching the expected inputs of the MobileNetV2 pretrained model (`tf.keras.applications.mobilenet_v2.preprocess_input`).
3. **Data Augmentation:** To prevent overfitting (Task 3 & 7), the following random transformations are applied to the training dataset:
   - Random horizontal flips
   - Random rotations (up to 15 degrees/0.04 fraction)
   - Random zoom (up to 10%)
   - Random contrast adjustments

---

## 5. Significance for Transfer Learning
The Oxford-IIIT Pet dataset is a classic benchmark for **fine-grained classification** (where differences between classes—e.g., *English Setter* vs *German Shorthaired Pointer*—are highly subtle). Starting from scratch would require a massive amount of labeled data to learn these fine differences. 

By leveraging **Transfer Learning** with a lightweight model pre-trained on ImageNet (such as **MobileNetV2**), the network reuses general features (edges, textures, shapes) and only needs to adapt the top-level representations to distinguish specific pet features.
