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
                    predictions = response_data.get("predictions", [])
                    MULTILABEL_CLASS = ['Age related Macular Degeneration', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Pathological Myopia', 'Normal', 'Other diseases/abnormalities']
                    
                    # Display predictions
                    for idx, confidence in enumerate(predictions[0]):
                        st.write(f"Confidence of {MULTILABEL_CLASS[idx]}: {confidence:.4f}")
                    best_index = np.argmax(predictions[0])
                    st.write(f"Best Result: {MULTILABEL_CLASS[best_index]}")
                    st.write(f"Confidence: {predictions[0][best_index]:.4f}")
                
                else:  # Partial Model predictions
                    for disease, confidence in response_data.items():
                        st.write(f"Model {disease.capitalize()}: {confidence:.4f}")
                    best_disease = max(response_data, key=response_data.get)
                    st.write(f"Best Result: {best_disease.capitalize()}")
            except requests.exceptions.RequestException as e:
                st.error("Error in server request. Please try again.")
