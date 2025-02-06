# frontend.py
import streamlit as st
import requests
from PIL import Image
import numpy as np

# Backend URL
API_URL = "http://127.0.0.1:5000/predict"

st.title('Ocular Disease Classifier')

# Sidebar model selection
st.sidebar.header("Model Selection")
classification_selection = st.sidebar.radio("Select model", options=["Full Model", "Partial Model"])

# File uploader
file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if file:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", width=400)

    # Send image and model type to backend
    if st.button("Classify"):
        st.write("Running inference...")
        with st.spinner("Classifying..."):
            # Prepare request payload
            files = {"file": file.getvalue()}
            data = {"model_type": classification_selection}

            try:
                # Request prediction from Flask API
                response = requests.post(API_URL, files=files, data=data)
                response_data = response.json()

                if classification_selection == "Full Model":
                    best_prediction = response_data["best_prediction"]
                    best_confidence = response_data["best_confidence"]
                    top_3_confidences = response_data["top_3_confidences"]
                    all_confidences = response_data.get("all_confidences", {}) # Retrieve all confidences for detailed view

                    st.success(f"Predicted class: {best_prediction}")
                    st.info(f"Confidence score: {best_confidence}%")

                    st.write("Top 3 Confidences:")
                    for class_name, confidence in top_3_confidences.items():
                        st.write(f"{class_name}: {confidence}%")

                    # Optional: Display all confidences for detailed view
                    # if st.checkbox("Show all class confidences"):
                    #     st.write("All Class Confidences:")
                    #     for class_name, confidence in all_confidences.items():
                    #         st.write(f"{class_name}: {confidence}%")

                else:  # Partial Model
                    top_3_results = response_data["top_3_results"]
                    best_disease = response_data["best_disease"]
                    all_results = response_data.get("all_results", {}) # Retrieve all results

                    st.success(f"Best Result: {best_disease.capitalize()}")
                    st.write("Top 3 Results:")
                    for disease, confidence in top_3_results.items():
                        st.write(f"Model {disease.capitalize()}: {confidence}%")


            except requests.exceptions.RequestException as e:
                st.error("Error in server request. Please try again.")
