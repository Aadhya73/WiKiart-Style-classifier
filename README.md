# WikiArt Style Classifier

An image classification project that uses deep learning to predict the
artistic style of a painting. The project uses the WikiArt-10K dataset
and a ResNet50-based model with spatial attention, channel attention,
residual feature refinement, and transfer learning.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reported Results](#reported-results)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Reference](#reference)

## Overview

The objective is to classify artwork images into artistic style
categories using a convolutional neural network.

The model learns visual representations from labelled artwork images
and predicts a style label for an input painting. The project also
compares two approaches to handling class imbalance:

1. Class-weighted training.
2. Balanced class sampling.

The implementation is inspired by research on attention mechanisms,
residual connections, and transfer learning for digital art
classification. It is an adaptation of these ideas, not an exact
reproduction of the referenced research paper.

## Features

- WikiArt-10K image dataset preparation.
- Image validation and preprocessing.
- Mapping of original style IDs to model labels.
- Stratified training, validation, and test splits.
- ImageNet-pretrained ResNet50 backbone.
- Spatial and channel attention modules.
- Residual feature refinement.
- Two-stage training and fine-tuning.
- Class-weighted and balanced-sampling experiments.
- Evaluation using accuracy, macro F1-score, and weighted F1-score.
- Classification report and confusion matrix.
- Streamlit interface for artwork style prediction.

## Model Architecture

The model uses the following pipeline:

1. Input image resized to 224 × 224 pixels.
2. Training-time data augmentation.
3. ResNet50 preprocessing.
4. ImageNet-pretrained ResNet50 feature extractor.
5. Spatial attention and channel attention.
6. Residual feature refinement block.
7. Global average pooling.
8. Dense classification layer with 256 units.
9. Batch normalization, LeakyReLU, and dropout.
10. Softmax output for the observed style classes.

The custom classification head uses dropout with a rate of 0.4.
The model is trained in two stages: first with the backbone frozen,
then by fine-tuning selected ResNet50 layers.

## Dataset

This project uses the WikiArt-10K dataset.

- Dataset: [WikiArt-10K on Hugging Face](https://huggingface.co/datasets/tellow-it-1/wikiart-10k)
- Task: Artistic style classification.
- Target label: `style`.
- Input format: RGB images.
- Model input size: 224 × 224 pixels.
- Number of observed style classes in the processed dataset: 16.

The dataset also contains other information, such as artist and genre
labels. The current classifier uses artistic style as its target.

### Dataset availability

The raw dataset files and processed images are not included in this
repository. Download the dataset separately and place the required
Parquet files inside a folder named `dataset/` in the project root.

The preparation scripts expect the files to be readable Parquet files
with the image and style fields used by the code.

## Data Preparation

The preparation pipeline consists of the following steps.

### 1. Validate images

```bash
python validate_images.py