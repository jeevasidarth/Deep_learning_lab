# CIFAR-10 Dataset Description

The project uses the **CIFAR-10** (Canadian Institute for Advanced Research) dataset, which is a standard benchmark dataset for image classification.

## Key Statistics
- **Total Images:** 60,000
- **Image Size:** 32x32 pixels
- **Color Channels:** 3 (RGB)
- **Number of Classes:** 10
- **Training Set:** 50,000 images
- **Testing Set:** 10,000 images

## Dataset Classes
The 10 classes of the dataset are mutually exclusive and represent:
1. **Airplane**
2. **Automobile**
3. **Bird**
4. **Cat**
5. **Deer**
6. **Dog**
7. **Frog**
8. **Horse**
9. **Ship**
10. **Truck**

## Data Preprocessing
Before feeding the data to the VGG16 model, the following preprocessing steps are performed:
1. **Normalization:** Pixel values, which are originally in the range `[0, 255]`, are converted to float32 and normalized to the range `[0, 1]` by dividing by `255.0`.
2. **Label Encoding:** Integer class labels are converted to one-hot encoded vectors (using `to_categorical`) to match the categorical crossentropy loss requirements.
