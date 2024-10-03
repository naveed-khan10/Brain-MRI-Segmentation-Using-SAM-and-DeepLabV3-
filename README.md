# 1: Brain MRI Segmentation using Segment Anything Model (SAM)

## Project Overview
This project focuses on building a  MRI brain tumor segmentation pipeline using the **LGG MRI dataset** and the **Segment Anything Model (SAM)**. The main steps involve data processing, model training using the MONAI framework, and evaluating the model's performance.

## Links
- **Dataset**: [(LGG) MRI dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Jupyter Notebook**: [Brain MRI Segmentation Using SAM](https://github.com/naveed-khan10/Brain-MRI-Segmentation-Using-SAM-and-DeepLabV3-/blob/main/Brain%20MRI%20Segmentation%20Using%20SAM.ipynb)
- 
## 1. Data Collection
- **Dataset**: [Low-Grade Glioma (LGG) MRI dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) containing MRI images with tumor masks.
- **Task**: Load the MRI image and mask file paths using Python.

## 2. Data Processing
- **Patient IDs**: Functions to extract patient IDs from file paths.
- **Image Labeling**: Label MRI images based on the presence of a tumor.
- **DataFrame Creation**: Combine images, masks, and labels into a unified DataFrame.

## 3. Dataset Splitting
- **Training & Testing Sets**: Split the dataset into 90% for training and 10% for testing.
- **DataFrames**: Create separate DataFrames for the train and test sets.

## 4. Data Visualization
- **Train-Test Split Visualization**: Show a pie chart of the dataset split.
- **Sample Images**: Display a few MRI images with their corresponding masks for reference.

## 5. Model Setup
- **Framework**: Use the MONAI framework to build the model for MRI image segmentation.
- **Parameters**:
  - Batch size
  - Learning rate
  - Number of epochs

## 6. Model Training
- **Optimizer**: Adam optimizer
- **Loss Function**: Focal Loss
- **Training Process**: Train the model while monitoring the loss and Intersection over Union (IoU) score.

## 7. Training Results
- **Loss Curve**: Visualize the training loss over epochs.
- **IoU Curve**: Display the IoU score across the training epochs to evaluate segmentation performance.

## Conclusion
This project demonstrates an effective approach for segmenting brain tumors from MRI scans using deep learning models. By leveraging the Segment Anything Model (SAM) and the MONAI framework, we aim to achieve high-performance tumor segmentation.



# __________________________________________________________

# 2: DeepLabV3 Brain MRI Segmentation

## Overview
This project implements MRI brain tumor segmentation using the **DeepLabV3** model with a **ResNet50 backbone**. The approach utilizes deep learning to segment anomalies from MRI images.

## Links
- **Dataset**: [(LGG) MRI dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Jupyter Notebook**: [Brain MRI Segmentation Using DeepLabV3](https://github.com/naveed-khan10/Brain-MRI-Segmentation-Using-SAM-and-DeepLabV3-/blob/main/Brain%20MRI%20Segmentation%20Using%20DeepLabV3%20.ipynb)

## Steps Involved

### 1. Library Imports
Essential libraries such as NumPy, Pandas, OpenCV, PyTorch, and torchvision models are imported. These libraries handle image processing, data manipulation, and deep learning model implementation.

### 2. Data Loading and Preprocessing
MRI images and their corresponding masks are loaded and preprocessed using OpenCV. The images are resized to 128x128 and normalized to fit the input requirements of the deep learning model. The masks are binarized to represent brain anomalies.

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
**DataLoader** instances for the training, validation, and test datasets are set up to feed the data in batches to the model during training and evaluation.

### 7. Model Training
The model is trained using **Binary Cross-Entropy Loss (BCEWithLogitsLoss)** and the **Adam optimizer**. Accuracy is computed by thresholding the model output at 0.5 for binary classification. The training loop reports both training and validation loss and accuracy after each epoch.

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

