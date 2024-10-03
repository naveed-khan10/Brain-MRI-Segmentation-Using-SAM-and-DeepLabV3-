# DeepLabV3 Brain MRI Segmentation

## Overview
This project implements MRI brain tumor segmentation using the DeepLabV3 model with a ResNet50 backbone. The approach uses deep learning to segment anomalies from MRI images.

## Steps Involved

### 1. Library Imports
Essential libraries such as **NumPy**, **Pandas**, **OpenCV**, **PyTorch**, and **torchvision models** are imported. These libraries handle image processing, data manipulation, and deep learning model implementation.

### 2. Data Loading and Preprocessing
MRI images and their corresponding masks are loaded and preprocessed using **OpenCV**. The images are resized to 128x128 and normalized to fit the input requirements of the deep learning model. The masks are binarized to represent brain anomalies.

### 3. Dataset Splitting
The dataset is split into three sets:
- **Training set**: 75%
- **Validation set**: 12.5%
- **Test set**: 12.5%

### 4. Model Definition
The **DeepLabV3** model is defined with a **ResNet50** backbone. The output layer is modified for binary segmentation (anomalies vs. no anomalies).

### 5. Custom Dataset Class
A custom **PyTorch Dataset** class is used to load the MRI images and masks, with transformations applied for normalization. This ensures the data is in the correct format for model input.

### 6. Data Loaders
DataLoader instances for the training, validation, and test datasets are set up to feed the data in batches to the model during training and evaluation.

### 7. Model Training
The model is trained using **Binary Cross-Entropy Loss (BCEWithLogitsLoss)** and the **Adam** optimizer. Accuracy is computed by thresholding the model output at 0.5 for binary classification. The training loop reports both training and validation loss and accuracy after each epoch.

### 8. Results
After training, the model achieves high segmentation accuracy:
- **Training accuracy**: Exceeds 99%
- **Validation accuracy**: Above 99%

The model successfully segments brain anomalies in MRI scans.

## Requirements
- Python 3.x
- NumPy
- Pandas
- OpenCV
- PyTorch
- Torchvision
