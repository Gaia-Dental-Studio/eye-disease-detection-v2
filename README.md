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
   git clone https://github.com/gabriel-tama/ocular-disease-classification.git
   ```

2. Navigate to the project directory:

   ```
   cd ocular-disease-classification
   ```

3. Navigate to the project directory:

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
   git clone https://github.com/gabriel-tama/ocular-disease-classification.git
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
