# Ocular Disease Classification App

This repository contains code for an Ocular Disease Classification App built using Streamlit and TensorFlow. The app is designed to classify ocular diseases from retinal fundus images.

## Features

- **Image Classification**: The app allows users to upload retinal fundus images for classification.
- **Multi-class Classification**: The model can classify images into multiple categories of ocular diseases.
- **Real-time Prediction**: Users can see real-time predictions along with confidence scores.
- **Easy to Use Interface**: Streamlit provides a user-friendly interface for seamless interaction.

## Requirements

- Python 3.x
- TensorFlow
- Streamlit

## Windows Installation

1. Clone this repository:

   ```
   git clone https://github.com/Gaia-Dental-Studio/eye-disease-detection-v2.git
   ```

2. Navigate to the project directory:

   ```
   cd eye-disease-detection-v2
   ```

3. Set up the conda environment:

   ```
   conda create -n ocular
   ```

   ```
   conda activate ocular
   ```

4. Install the required dependencies:

   ```
   pip install -r ./requirements.txt
   ```

5. Create a folder named models inside the repo, Copy all the models file into ./models the [GoogleDrive](https://drive.google.com/drive/folders/1Qp8I4YN3N47IjmwXA18RxwB1z-T4feXS) Exclude the model.h5 file

## macOS Installation 

1. Clone this repository:

   ```
   git clone https://github.com/Gaia-Dental-Studio/eye-disease-detection-v2.git
   ```

2. Navigate to the project directory:

   ```
   cd ocular-disease-classification
   ```

3. Make a virtual environtment
   ```
   conda create -n ocular python==3.9
   ```

4. Installing Dependencies
   ```
   pip install -r ./requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```
   streamlit run ocular.py
   ```

2. Open your web browser and navigate to the provided URL.

3. Select the Model Type, Full model is the softmax model, Partial Model is the Sigmoid Model

4. Upload a retinal fundus image using the interface.

5. View the predicted class and confidence score.

## Technical Information Update
Documentation Version: 10.02.2025

- The newest trained model was stored in gdrive for [Partial Model](https://drive.google.com/drive/folders/1Ny-qD4Uj94Y1UWV6DQnWAQnHHHWfpROw?usp=drive_link) and [Full model](https://drive.google.com/file/d/1haaczI0ExuMDDrqeXi69rkZABhrM31cz/view?usp=drive_link).
- The training code to produce those model was stored at this [Link](https://drive.google.com/drive/folders/18tQsgdb3OBW6Q8fnYeisnbFD4T1siXq0?usp=drive_link).
- The training was still done using the same datasets which is [ODIR5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) from kaggle.
- The partial model is a binary classification case where the dataset of one disease is compared to others. Considering the datasets (images) available for each class, the classification was only done on five classes. The partial model was trained using EfficientNetB0 architecture. The following are the classification class and its accuracy obtained during the training:
   [-]Cataract: 98%
   [-]Glaucoma: 88%
   [-]Hypertansive-retinopathy: 88%
   [-]Diabetic-retinopathy: 89%
   [-]Other:87%
- The full model is a multiclass classification case where all class datasets are used at once in the training process, resulting in one h5 model. This model is trained by fine-tuning the ResNet50 architecture. The accuracy during the training process only reach 62% where the f1-score of class variation in range 30%-70%.
- The API endpoint code for this model version are api.py (Flask-based backend) and app.py (Streamlit frontend). To run the demo app, run ```python api.py``` and ```streamlit run app.py``` simultaneously in terminal or cmd.

**NOTE:**
- The available datasets have an imbalanced data distribution among classes. In the training process, no imbalance in data handling was implemented. Therefore, implementing the imbalanced data handling method might improve the model's accuracy.
- This is a handover of the Clear-see project where the training code for multiclass classification is missing. Therefore, a new training code was written from the available information to develop a new model.