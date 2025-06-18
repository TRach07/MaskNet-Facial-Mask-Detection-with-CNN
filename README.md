# MaskNet: Facial Mask Detection with CNN

## Overview
MaskNet is a deep learning project focused on detecting facial masks in images using Convolutional Neural Networks (CNNs). Developed in a post-pandemic context, this project aims to automate the identification of individuals wearing masks (`WithMask`) or not (`WithoutMask`) in public spaces, supporting health and safety monitoring systems.

The project leverages the **Face Mask 12K Images Dataset** from Kaggle, containing ~12,000 images split into training (~10,000), validation (~800), and test (~992) sets. Three CNN models are implemented and compared:
- **Simple CNN**: A lightweight baseline model.
- **Deep CNN**: A more complex architecture with regularization techniques.
- **VGG16**: A pre-trained model using transfer learning.

The models achieve high accuracy, with test accuracies of 98.79% (Simple CNN), 99.40% (Deep CNN), and 99.60% (VGG16), demonstrating robust performance for binary classification.

## Features
- **Data Preprocessing**: Images resized to 128x128 pixels, normalized, and augmented using `ImageDataGenerator` (rotation, shift, zoom, flip).
- **Model Training**: Includes early stopping, learning rate reduction, and comprehensive evaluation metrics (accuracy, loss, confusion matrix, classification report).
- **Visualizations**: Learning curves, confusion matrices, and sample predictions for model analysis.
- **Pipeline**: End-to-end workflow from data loading to model evaluation, documented in a Jupyter Notebook.

## Dataset
The **Face Mask 12K Images Dataset** is organized as follows:
- **Train**: 10,000 images (5,000 `WithMask`, 5,000 `WithoutMask`)
- **Validation**: 800 images (400 `WithMask`, 400 `WithoutMask`)
- **Test**: 992 images (483 `WithMask`, 509 `WithoutMask`)

The dataset is stored in Google Drive and accessed via the notebook. To use it, download the dataset from [Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) and update the path in the notebook.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, Seaborn, OpenCV
- Scikit-learn
- Google Colab (with GPU support recommended)

Install dependencies using:
```bash
pip install tensorflow numpy matplotlib seaborn opencv-python scikit-learn
