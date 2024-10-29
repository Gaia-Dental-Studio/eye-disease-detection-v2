# Ocular-Disease-Detectionv1.2

## Project Overview

### Introduction

This is the documentation for ocular disease classification 

This repository contains the following contents.
- Sample program
- Ocular disease classification (partial binary classification model)
- Ocular disease classification (multiclass classification model)
- Notebook for partial-binary and multiclass classification for ocular disease classification

### TODO

- Add more dataset samples
- Try another preprocessing methods

## Problem Statement

- This Model aims to classify 6 major disease from Ocular dataset and 1 clas "Other Disease" which is contain several disease with small samples, and "Normal" for normal fundus eye.

Here are the labels:
Normal (N),
Diabetes (D),
Glaucoma (G),
Cataract (C),
Age related Macular Degeneration (A),
Hypertension (H),
Pathological Myopia (M),
Other diseases/abnormalities (O)

## Data

- Dataset we have used for this model is from Ocular-disease dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

## Model Architecture

- For the partial-binary model we are using EfficientNetB0 weight for transfer learning using ImageNet dataset from [Keras](https://keras.io/api/applications/efficientnet/#efficientnetb0-function). We choosing EfficientNetB0 because the models size only 29 MB but can reach 93.3% Top-5 Accuracy, which is light weight but good enough for classification.

- For the single-multiclass model we are using ResNet-50 weight for transfer learning using ImageNet dataset from [Keras](https://keras.io/api/applications/resnet/#resnet50-function).

## Training Process

- For partial-binary models
- For single-multiclass models

## Evaluation

- For evaluation we using binary-partial models and single-multiclass model classification and compare the results

- performance evaluation metrics are:
    - categorical_crossentrophy
    - training accuracy
    - validation accuracy
    - F1-score

## Deployment

### Requirements

- Python 3.9
- Tensorflow 
- Pandas
- Numpy
- Seaborn 
- Matplotlib

### Temporary Local Deployment

Check out the [Github link](https://github.com/gabriel-tama/ocular-disease-classification.git) for temporary demonstration.

### Installation 


## Codebase
